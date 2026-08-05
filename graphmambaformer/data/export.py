"""Export a :class:`SyntheticDataset` to standard genomics file formats.

An aligner's natural output is a BAM (reads mapped to a reference). This module
writes the dataset's *ground-truth* alignments as that gold standard, plus the
companion files needed for the graph and the extra supervision that BAM cannot
represent:

    reference.fasta   reference sequences
    reads.fastq       reads + Phred qualities (as sequenced)
    truth.sam         ground-truth alignments (FLAG / POS / MAPQ / CIGAR)
    graph.gfa         pangenome graph (segments + typed links)
    labels.json       seeds (true/false + 12-dim features) + head labels

SAM is written as plain text (no dependencies). Convert to a sorted, indexed
BAM with samtools if it is installed (see :func:`sam_to_bam`), or set
``to_bam=True`` in :func:`export_dataset`.

Reverse-strand reads are emitted per the SAM spec: SEQ/QUAL are given on the
forward strand (reverse-complemented / reversed) and the CIGAR is ordered along
the forward reference, with FLAG bit 0x10 set.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

from .synthetic import ReadRecord, Reference, SyntheticDataset, reverse_complement

# SAM FLAG bits we use
FLAG_REVERSE = 0x10
FLAG_UNMAPPED = 0x4
FLAG_SECONDARY = 0x100
FLAG_SUPPLEMENTARY = 0x800

_MAX_PHRED = 93


def _refname(ref_id: int) -> str:
    return f"ref{ref_id}"


def _qual_to_ascii(quals: list[int]) -> str:
    return "".join(chr(min(_MAX_PHRED, max(0, q)) + 33) for q in quals)


def _cigar_ref_len(cigar: list[tuple[str, int]]) -> int:
    return sum(n for op, n in cigar if op in ("=", "X", "D"))


def _cigar_query_len(cigar: list[tuple[str, int]]) -> int:
    return sum(n for op, n in cigar if op in ("=", "X", "I", "S"))


def _cigar_str(cigar: list[tuple[str, int]]) -> str:
    return "".join(f"{n}{op}" for op, n in cigar) if cigar else "*"


def _is_mapped(rec: ReadRecord) -> bool:
    """A read is mapped iff its CIGAR consumes some reference and it has a span."""
    return bool(rec.seq) and rec.ref_end > rec.ref_start and _cigar_ref_len(rec.cigar) > 0


# --------------------------------------------------------------------------- #
# FASTA / FASTQ
# --------------------------------------------------------------------------- #
def write_fasta(references: dict[int, Reference], path: str, width: int = 70) -> None:
    with open(path, "w") as fh:
        for rid, ref in references.items():
            fh.write(f">{_refname(rid)} length={len(ref.seq)} "
                     f"gc={ref.gc_content:.3f} repeat={ref.repeat_content:.3f}\n")
            for i in range(0, len(ref.seq), width):
                fh.write(ref.seq[i : i + width] + "\n")


def write_fastq(records: list[ReadRecord], path: str) -> None:
    with open(path, "w") as fh:
        for rec in records:
            seq = rec.seq if rec.seq else ""
            qual = _qual_to_ascii(rec.quals) if rec.quals else ""
            fh.write(f"@{rec.read_id} mod={rec.modality}"
                     f"{' edge=' + rec.edge_case if rec.edge_case else ''}\n")
            fh.write(seq + "\n+\n" + qual + "\n")


# --------------------------------------------------------------------------- #
# SAM (ground-truth alignments)
# --------------------------------------------------------------------------- #
def _sam_fields_for_alignment(
    rec: ReadRecord,
    ref_id: int,
    ref_start: int,
    strand: int,
    cigar: list[tuple[str, int]],
    seq: str,
    quals: list[int],
    flag: int,
) -> list[str]:
    """Build one SAM line, reorienting to the forward strand for reverse reads."""
    if strand == -1:
        flag |= FLAG_REVERSE
        cigar = list(reversed(cigar))
        seq = reverse_complement(seq)
        quals = list(reversed(quals))
    # SAM invariant: query-consuming CIGAR length must equal SEQ length. If it
    # does not (e.g. a hard-clipped supplementary), drop SEQ/QUAL to '*'.
    if seq and _cigar_query_len(cigar) != len(seq):
        seq = ""
        quals = []
    qual_str = _qual_to_ascii(quals) if quals else "*"
    return [
        rec.read_id,
        str(flag),
        _refname(ref_id),
        str(ref_start + 1),  # SAM POS is 1-based
        str(rec.mapq),
        _cigar_str(cigar),
        "*",  # RNEXT
        "0",  # PNEXT
        "0",  # TLEN
        seq if seq else "*",
        qual_str,
    ]


def write_sam(dataset: SyntheticDataset, path: str, split: Optional[str] = None) -> None:
    refs = dataset.references
    if split is not None:
        records = list(dataset.splits.get(split, []))
    else:
        records = dataset.all_records()

    with open(path, "w") as fh:
        fh.write("@HD\tVN:1.6\tSO:unsorted\n")
        for rid, ref in refs.items():
            fh.write(f"@SQ\tSN:{_refname(rid)}\tLN:{len(ref.seq)}\n")
        fh.write("@PG\tID:graphmambaformer-synth\tPN:graphmambaformer\t"
                 "DS:synthetic ground-truth alignments\n")

        for rec in records:
            if not _is_mapped(rec):
                # unmapped record
                qual = _qual_to_ascii(rec.quals) if rec.quals else "*"
                fh.write("\t".join([
                    rec.read_id, str(FLAG_UNMAPPED), "*", "0", "0", "*",
                    "*", "0", "0", rec.seq or "*", qual,
                ]) + f"\tXC:Z:{rec.edge_case or 'unmapped'}\n")
                continue

            fields = _sam_fields_for_alignment(
                rec, rec.ref_id, rec.ref_start, rec.strand, rec.cigar,
                rec.seq, rec.quals, flag=0,
            )
            tags = [f"NM:i:{_edit_distance(rec)}", f"XM:Z:{rec.modality}"]
            if rec.is_chimeric and rec.supplementary:
                sup = rec.supplementary
                tags.append(
                    f"SA:Z:{_refname(rec.ref_id)},{sup['ref_start'] + 1},"
                    f"{'-' if sup['strand'] == -1 else '+'},"
                    f"{_cigar_str(sup['cigar'])},{rec.mapq},0;"
                )
            fh.write("\t".join(fields + tags) + "\n")

            # supplementary alignment for chimeric reads
            if rec.is_chimeric and rec.supplementary:
                sup = rec.supplementary
                sup_fields = _sam_fields_for_alignment(
                    rec, rec.ref_id, sup["ref_start"], sup["strand"], sup["cigar"],
                    rec.seq, rec.quals, flag=FLAG_SUPPLEMENTARY,
                )
                fh.write("\t".join(sup_fields + [f"XM:Z:{rec.modality}"]) + "\n")


def _edit_distance(rec: ReadRecord) -> int:
    """NM tag: mismatches + inserted + deleted bases."""
    return sum(n for op, n in rec.cigar if op in ("X", "I", "D"))


# --------------------------------------------------------------------------- #
# GFA (pangenome graph)
# --------------------------------------------------------------------------- #
def write_gfa(dataset: SyntheticDataset, path: str) -> None:
    from .synthetic import EDGE_TYPES

    id2name = {v: k for k, v in EDGE_TYPES.items()}
    with open(path, "w") as fh:
        fh.write("H\tVN:Z:1.0\n")
        for rid, ref in dataset.references.items():
            g = ref.graph
            for nidx, nseq in enumerate(g.node_seqs):
                seg = f"{_refname(rid)}_n{nidx}"
                start = g.node_ref_start[nidx]
                fh.write(f"S\t{seg}\t{nseq}\tLN:i:{len(nseq)}\tRS:i:{start}\n")
            for (src, dst), et in zip(g.edge_index, g.edge_type):
                a = f"{_refname(rid)}_n{src}"
                b = f"{_refname(rid)}_n{dst}"
                # zt tag records our discrete edge type (GFA has no native type)
                fh.write(f"L\t{a}\t+\t{b}\t+\t0M\tzt:Z:{id2name.get(et, str(et))}\n")


def read_gfa(path: str) -> "PangenomeGraph":
    """Parse a GFA (``S``/``L`` lines) into a :class:`PangenomeGraph`.

    The inverse of :func:`write_gfa`. Node order follows the file's ``S`` lines;
    edges come from ``L`` lines, mapping our ``zt:Z:<name>`` edge-type tag back
    to its :data:`EDGE_TYPES` id (defaulting to ``ref_link`` when absent).
    ``RS:i:`` segment tags populate ``node_ref_start`` (``-1`` if missing).

    Works for the synthetic graph, the real HPRC reference-backbone window, and
    any ``vg convert -f`` output.
    """
    from .synthetic import EDGE_TYPES, PangenomeGraph

    node_index: dict[str, int] = {}
    node_seqs: list[str] = []
    node_ref_start: list[int] = []
    edge_index: list[tuple[int, int]] = []
    edge_type: list[int] = []

    def _tags(fields: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in fields:
            parts = f.split(":", 2)
            if len(parts) == 3:
                out[parts[0]] = parts[2]
        return out

    with open(path) as fh:
        for line in fh:
            if line.startswith("S\t"):
                f = line.rstrip("\n").split("\t")
                name, seq = f[1], f[2]
                tags = _tags(f[3:])
                node_index[name] = len(node_seqs)
                node_seqs.append(seq)
                node_ref_start.append(int(tags.get("RS", -1)))
            elif line.startswith("L\t"):
                f = line.rstrip("\n").split("\t")
                a, b = f[1], f[3]
                if a not in node_index or b not in node_index:
                    continue
                tags = _tags(f[6:])
                et_name = tags.get("zt", "ref_link")
                edge_index.append((node_index[a], node_index[b]))
                edge_type.append(EDGE_TYPES.get(et_name, EDGE_TYPES["ref_link"]))

    backbone = [i for i, s in enumerate(node_ref_start) if s >= 0]
    return PangenomeGraph(
        node_seqs=node_seqs,
        edge_index=edge_index,
        edge_type=edge_type,
        backbone_path=backbone,
        node_ref_start=node_ref_start,
    )


# --------------------------------------------------------------------------- #
# JSON (seeds + head labels BAM/GFA can't hold)
# --------------------------------------------------------------------------- #
def write_labels_json(dataset: SyntheticDataset, path: str) -> None:
    out: dict = {"config": _config_to_dict(dataset.config), "splits": {}}
    for split, records in dataset.splits.items():
        recs = []
        for rec in records:
            recs.append({
                "read_id": rec.read_id,
                "ref_id": rec.ref_id,
                "modality": rec.modality,
                "strand": rec.strand,
                "ref_start": rec.ref_start,
                "ref_end": rec.ref_end,
                "cigar": rec.cigar_string,
                "mapq": rec.mapq,
                "read_len": len(rec.seq),
                "seeds": [
                    {
                        "read_pos": s.read_pos, "ref_pos": s.ref_pos,
                        "length": s.length, "strand": s.strand,
                        "is_true": s.is_true,
                        "features": [round(f, 6) for f in s.features],
                    }
                    for s in rec.seeds
                ],
                "num_true_seeds": rec.num_true_seeds,
                "num_false_seeds": rec.num_false_seeds,
                "methylation": rec.methylation,
                "splice_junctions": rec.splice_junctions,
                "barcode": rec.barcode,
                "umi": rec.umi,
                "is_chimeric": rec.is_chimeric,
                "supplementary": rec.supplementary,
                "edge_case": rec.edge_case,
            })
        out["splits"][split] = recs
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


def _config_to_dict(cfg) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}


# --------------------------------------------------------------------------- #
# BAM conversion (optional, requires samtools)
# --------------------------------------------------------------------------- #
def sam_to_bam(sam_path: str, bam_path: Optional[str] = None) -> Optional[str]:
    """Convert a SAM to a sorted, indexed BAM.

    Uses the ``samtools`` binary if it is on ``PATH``; otherwise falls back to
    ``pysam`` (which bundles htslib, so no external install is needed). Returns
    the BAM path on success, or ``None`` if neither samtools nor pysam is
    available.
    """
    bam_path = bam_path or (os.path.splitext(sam_path)[0] + ".bam")

    if shutil.which("samtools") is not None:
        subprocess.run(["samtools", "sort", "-o", bam_path, sam_path], check=True)
        subprocess.run(["samtools", "index", bam_path], check=True)
        return bam_path

    try:
        import pysam  # bundles htslib; no external samtools needed
    except ImportError:
        return None

    pysam.sort("-o", bam_path, sam_path)
    pysam.index(bam_path)
    return bam_path


# --------------------------------------------------------------------------- #
# BAM reading (real or synthetic BAMs -> ReadRecord objects)
# --------------------------------------------------------------------------- #
# pysam CIGAR op codes -> our string ops. BAM uses M/I/D/N/S/H/P/=/X; the
# synthetic generator only emits =/X/I/D/S, but real BAMs (e.g. HG002) use M.
_CIGAR_CODE_TO_OP: dict[int, str] = {
    0: "M",  # BAM_CMATCH (match or mismatch)
    1: "I",  # BAM_CINS
    2: "D",  # BAM_CDEL
    3: "N",  # BAM_CREF_SKIP (intron)
    4: "S",  # BAM_CSOFT_CLIP
    5: "H",  # BAM_CHARD_CLIP
    6: "P",  # BAM_CPAD
    7: "=",  # BAM_CEQUAL
    8: "X",  # BAM_CDIFF
}


def _pysam_read_mode(path: str) -> str:
    """pysam open mode for a reads file. ``.ubam`` is binary BAM, not text SAM."""
    low = path.lower()
    if low.endswith((".bam", ".ubam")):
        return "rb"
    if low.endswith(".cram"):
        return "rc"
    return "r"


def read_bam(
    bam_path: str,
    region: Optional[str] = None,
    limit: Optional[int] = None,
    modality: str = "pacbio_hifi",
    include_unmapped: bool = False,
) -> list[ReadRecord]:
    """Read alignments from a BAM (or SAM/CRAM) into :class:`ReadRecord` objects.

    Designed both for the synthetic BAMs this module writes and for real
    single-sample long-read BAMs (e.g. the HG002 PacBio-HiFi file). Reads a
    small ``region`` and/or capped ``limit`` so a slice can be pulled onto CPU
    without touching the whole (~100 GB) file.

    Args:
        bam_path: path to a ``.bam`` / ``.sam`` / ``.cram`` file.
        region: optional ``samtools``-style region (e.g. ``"chr20:1000000-1100000"``).
            Requires a coordinate-sorted, indexed BAM. ``None`` streams from the start.
        limit: stop after this many returned records (``None`` = all).
        modality: modality tag stamped on every record (default ``"pacbio_hifi"``).
        include_unmapped: if ``True``, also yield unmapped reads (seq only, no
            alignment); by default they are skipped.

    Returns:
        A list of :class:`ReadRecord`. Seed/head supervision fields are left
        empty (real BAMs carry no seed labels); ``ref_id`` is the index of the
        reference name in the BAM header, and ``ref_positions`` gives a
        forward-reference coordinate per query base (``-1`` for insertions/clips).
    """
    import pysam  # local import so pysam stays an optional dependency

    from .formats import validate_modality

    modality = validate_modality(modality)
    records: list[ReadRecord] = []
    open_mode = _pysam_read_mode(bam_path)
    # check_sq=False so unaligned BAM/CRAM (no @SQ lines) can still be read.
    with pysam.AlignmentFile(bam_path, open_mode, check_sq=False) as af:
        ref_id_of = {name: i for i, name in enumerate(af.references)}
        itr = af.fetch(region=region) if region is not None else af.fetch(until_eof=True)
        for aln in itr:
            if aln.is_secondary or aln.is_supplementary:
                continue
            if aln.is_unmapped and not include_unmapped:
                continue

            rec = _alignment_to_record(aln, ref_id_of, modality)
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _alignment_to_record(aln, ref_id_of: dict[str, int], modality: str) -> ReadRecord:
    seq = aln.query_sequence or ""
    quals = list(aln.query_qualities) if aln.query_qualities is not None else [0] * len(seq)

    if aln.is_unmapped:
        return ReadRecord(
            read_id=aln.query_name, ref_id=-1, modality=modality,
            seq=seq, quals=quals, ref_start=0, ref_end=0, strand=1,
            cigar=[], ref_positions=[-1] * len(seq), mapq=int(aln.mapping_quality),
            edge_case="unmapped",
        )

    cigar = [
        (_CIGAR_CODE_TO_OP.get(op, "M"), length)
        for op, length in (aln.cigartuples or [])
    ]
    # per-query-base forward-reference coordinate (None -> -1 for ins/clip)
    ref_positions = [
        -1 if p is None else p
        for p in aln.get_reference_positions(full_length=True)
    ]
    return ReadRecord(
        read_id=aln.query_name,
        ref_id=ref_id_of.get(aln.reference_name, -1),
        modality=modality,
        seq=seq,
        quals=quals,
        ref_start=int(aln.reference_start),
        ref_end=int(aln.reference_end) if aln.reference_end is not None else int(aln.reference_start),
        strand=-1 if aln.is_reverse else 1,
        cigar=cigar,
        ref_positions=ref_positions,
        mapq=int(aln.mapping_quality),
    )


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def export_dataset(
    dataset: SyntheticDataset,
    out_dir: str,
    to_bam: bool = False,
) -> dict[str, str]:
    """Write the full standard-format bundle into ``out_dir``.

    Returns a dict of ``{name: path}`` for every file written.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "fasta": os.path.join(out_dir, "reference.fasta"),
        "fastq": os.path.join(out_dir, "reads.fastq"),
        "sam": os.path.join(out_dir, "truth.sam"),
        "gfa": os.path.join(out_dir, "graph.gfa"),
        "labels": os.path.join(out_dir, "labels.json"),
    }
    write_fasta(dataset.references, paths["fasta"])
    write_fastq(dataset.all_records(), paths["fastq"])
    write_sam(dataset, paths["sam"])
    write_gfa(dataset, paths["gfa"])
    write_labels_json(dataset, paths["labels"])

    if to_bam:
        bam = sam_to_bam(paths["sam"])
        if bam is not None:
            paths["bam"] = bam
    return paths
