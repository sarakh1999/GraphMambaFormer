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

from ..progress import progress
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
        for rec in progress(records, desc="write FASTQ", unit="read", leave=False):
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

        for rec in progress(records, desc="write SAM", unit="read", leave=False):
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

    Accepts plain ``.gfa`` or gzipped ``.gfa.gz`` (magic-byte detection).
    Works for the synthetic graph, the real HPRC reference-backbone window, and
    any ``vg convert -f`` output.
    """
    from .synthetic import EDGE_TYPES, PangenomeGraph

    node_index: dict[str, int] = {}
    node_seqs: list[str] = []
    node_ref_start: list[int] = []
    edge_index: list[tuple[int, int]] = []
    edge_type: list[int] = []
    # Raw path/walk lines, resolved to node ids after every S line is known
    # (a P/W line may reference a segment defined later in the file).
    raw_paths: list[list[str]] = []

    def _walk_segments(walk: str) -> list[str]:
        """Segment names from a GFA1.1 W-line walk like ``>s1>s2<s3``."""
        names: list[str] = []
        cur: list[str] = []
        for ch in walk:
            if ch in "><":
                if cur:
                    names.append("".join(cur))
                    cur = []
            else:
                cur.append(ch)
        if cur:
            names.append("".join(cur))
        return names

    def _tags(fields: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in fields:
            parts = f.split(":", 2)
            if len(parts) == 3:
                out[parts[0]] = parts[2]
        return out

    def _open(path: str):
        with open(path, "rb") as probe:
            gzipped = probe.read(2) == b"\x1f\x8b"
        if gzipped:
            import gzip
            return gzip.open(path, "rt")
        return open(path)

    with _open(path) as fh:
        for line in progress(fh, desc=f"read GFA {os.path.basename(path)}", unit="line", leave=False):
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
            elif line.startswith("P\t"):
                # P<tab>name<tab>seg1+,seg2-,...<tab>overlaps
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3 and f[2] and f[2] != "*":
                    raw_paths.append([s[:-1] if s[-1:] in "+-" else s
                                      for s in f[2].split(",")])
            elif line.startswith("W\t"):
                # W<tab>sample<tab>hap<tab>seq<tab>start<tab>end<tab>walk
                f = line.rstrip("\n").split("\t")
                if len(f) >= 7 and f[6] and f[6] != "*":
                    raw_paths.append(_walk_segments(f[6]))

    # Resolve segment names to node ids now that node_index is complete.
    haplotype_paths: list[list[int]] = []
    for segs in raw_paths:
        nodes = [node_index[s] for s in segs if s in node_index]
        if nodes:
            haplotype_paths.append(nodes)

    backbone = [i for i, s in enumerate(node_ref_start) if s >= 0]
    return PangenomeGraph(
        node_seqs=node_seqs,
        edge_index=edge_index,
        edge_type=edge_type,
        backbone_path=backbone,
        node_ref_start=node_ref_start,
        haplotype_paths=haplotype_paths,
    )


# --------------------------------------------------------------------------- #
# JSON (seeds + head labels BAM/GFA can't hold)
# --------------------------------------------------------------------------- #
def write_labels_json(dataset: SyntheticDataset, path: str) -> None:
    out: dict = {"config": _config_to_dict(dataset.config), "splits": {}}
    for split, records in dataset.splits.items():
        recs = []
        for rec in progress(records, desc=f"labels JSON {split}", unit="read", leave=False):
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
    reference_fasta: Optional[str] = None,
    as_sequences: bool = False,
    subsample_to: Optional[int] = None,
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
        limit: stop after this many returned records (``None`` = all). This is
            a leftmost head-truncation; prefer ``subsample_to`` for downsampling.
        subsample_to: if set and a ``region`` is given, keep a coverage-uniform
            random subset of ~this many primary reads, chosen by hashing the
            read name, rather than the leftmost ``subsample_to`` reads. This
            spreads the kept reads evenly across the window (e.g. a true ~12x
            downsample of a deep BAM) instead of piling full depth on the
            window's left edge, and it skips decoding the reads it drops so a
            deep window is read several times faster.
        modality: modality tag stamped on every record (default ``"pacbio_hifi"``).
        include_unmapped: if ``True``, also yield unmapped reads (seq only, no
            alignment); by default they are skipped.
        reference_fasta: reference FASTA required to decode many CRAM files.
        as_sequences: if ``True``, drop prior alignment fields and return
            remappable sequences (needed when an aligned Illumina CRAM or ONT
            BAM is used as *input* to our aligner rather than as truth).

    Returns:
        A list of :class:`ReadRecord`. Seed/head supervision fields are left
        empty (real BAMs carry no seed labels); ``ref_id`` is the index of the
        reference name in the BAM header, and ``ref_positions`` gives a
        forward-reference coordinate per query base (``-1`` for insertions/clips).
        When ``as_sequences`` is set, every record is unmapped.
    """
    import zlib

    import pysam  # local import so pysam stays an optional dependency

    from .formats import validate_modality
    from .synthetic import reverse_complement

    modality = validate_modality(modality)
    records: list[ReadRecord] = []
    open_mode = _pysam_read_mode(bam_path)
    open_kwargs: dict = {"check_sq": False}
    if reference_fasta:
        open_kwargs["reference_filename"] = reference_fasta
    # check_sq=False so unaligned BAM/CRAM (no @SQ lines) can still be read.
    with pysam.AlignmentFile(bam_path, open_mode, **open_kwargs) as af:
        ref_id_of = {name: i for i, name in enumerate(af.references)}
        # Coverage-uniform downsample. When a target read count is given (e.g. a
        # 12x illumina cap on a much deeper BAM), keep a random subset chosen by
        # hashing the read name so the survivors are spread evenly across the
        # window, rather than head-truncating to the leftmost reads (which piles
        # full depth on the window's left edge and leaves the rest empty). A
        # cheap C-level primary-read count sets the keep fraction; the hash test
        # runs before the expensive record decode, so dropped reads are ~free.
        # Needs a region for the count to be cheap; without one we leave the
        # stream intact and rely on ``limit``.
        keep_thresh: Optional[int] = None
        salt = b""
        if subsample_to is not None and region is not None:
            try:
                n_primary = af.count(region=region, read_callback="all")
            except Exception:  # pragma: no cover - count is best-effort
                n_primary = 0
            if n_primary > subsample_to:
                keep_thresh = int((subsample_to / n_primary) * (1 << 32))
                salt = (region or "").encode() + b"|"
        itr = af.fetch(region=region) if region is not None else af.fetch(until_eof=True)
        label = os.path.basename(bam_path)
        for aln in progress(itr, desc=f"read BAM {label}", unit="aln", leave=False):
            if aln.is_secondary or aln.is_supplementary:
                continue
            if aln.is_unmapped and not include_unmapped and not as_sequences:
                continue
            if keep_thresh is not None:
                h = zlib.crc32(salt + (aln.query_name or "").encode()) & 0xFFFFFFFF
                if h >= keep_thresh:
                    continue

            if as_sequences:
                seq = aln.query_sequence or ""
                if not seq:
                    continue
                quals = (
                    list(aln.query_qualities)
                    if aln.query_qualities is not None
                    else [0] * len(seq)
                )
                # Restore original sequenced orientation for remapping.
                if aln.is_reverse:
                    seq = reverse_complement(seq)
                    quals = list(reversed(quals))
                # Positions are now in the forward/original-strand frame.
                methylation = _extract_methylation(aln, forward=True)
                mate_index = 1 if aln.is_read1 else (2 if aln.is_read2 else 0)
                pair_id = aln.query_name if aln.is_paired and mate_index else None
                read_id = f"{aln.query_name}/{mate_index}" if pair_id else aln.query_name
                records.append(
                    ReadRecord(
                        read_id=read_id,
                        ref_id=-1,
                        modality=modality,
                        seq=seq,
                        quals=quals,
                        ref_start=0,
                        ref_end=0,
                        strand=1,
                        cigar=[],
                        ref_positions=[-1] * len(seq),
                        mapq=0,
                        edge_case="from_aligned_input",
                        methylation=methylation,
                        pair_id=pair_id,
                        mate_index=mate_index,
                    )
                )
            else:
                records.append(_alignment_to_record(aln, ref_id_of, modality))
            if limit is not None and len(records) >= limit:
                break
    return records


def _extract_methylation(aln, *, forward: bool) -> list[tuple[int, int]]:
    """Read MM/ML modified-base tags into ``(read_pos, called)`` pairs.

    ONT and PacBio deliver base modifications (e.g. 5mC) as SAM ``MM``/``ML``
    tags — the reason a uBAM is preferred over FASTQ. pysam surfaces them as
    ``modified_bases`` (positions relative to the stored ``SEQ``) or
    ``modified_bases_forward`` (positions relative to the original sequenced
    strand). ``forward`` selects whichever matches the orientation of the
    ``seq`` on the resulting :class:`ReadRecord`, so a downstream methylation
    head indexes calls in the same frame as the bases it stores.

    A position is a 1 when its probability qual (0-255, i.e. ``256*p``) clears
    0.5 (``>= 128``) or when the caller left it unknown (``-1``); otherwise 0.
    Multiple modification codes on one base collapse to a positive call.
    """
    try:
        mods = aln.modified_bases_forward if forward else aln.modified_bases
    except Exception:  # pragma: no cover - htslib parse guard
        return []
    if not mods:
        return []
    calls: dict[int, int] = {}
    for positions in mods.values():
        for pos, qual in positions:
            called = 1 if (qual is None or qual < 0 or qual >= 128) else 0
            calls[pos] = max(calls.get(pos, 0), called)
    return sorted(calls.items())


def _alignment_to_record(aln, ref_id_of: dict[str, int], modality: str) -> ReadRecord:
    seq = aln.query_sequence or ""
    quals = list(aln.query_qualities) if aln.query_qualities is not None else [0] * len(seq)
    # MM/ML positions are relative to the stored SEQ, which is what we keep here.
    methylation = _extract_methylation(aln, forward=False)
    mate_index = 1 if aln.is_read1 else (2 if aln.is_read2 else 0)
    pair_id = aln.query_name if aln.is_paired and mate_index else None
    read_id = f"{aln.query_name}/{mate_index}" if pair_id else aln.query_name
    pair_fields = {
        "pair_id": pair_id,
        "mate_index": mate_index,
        "mate_ref_id": int(aln.next_reference_id)
        if aln.next_reference_id is not None else -1,
        "mate_ref_start": int(aln.next_reference_start)
        if aln.next_reference_start is not None else -1,
        "mate_strand": -1 if aln.mate_is_reverse else 1,
        "template_length": int(aln.template_length),
        "proper_pair": bool(aln.is_proper_pair),
    }

    if aln.is_unmapped:
        return ReadRecord(
            read_id=read_id, ref_id=-1, modality=modality,
            seq=seq, quals=quals, ref_start=0, ref_end=0, strand=1,
            cigar=[], ref_positions=[-1] * len(seq), mapq=int(aln.mapping_quality),
            edge_case="unmapped", methylation=methylation,
            **pair_fields,
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
        read_id=read_id,
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
        methylation=methylation,
        **pair_fields,
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
