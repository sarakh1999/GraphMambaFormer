"""Format I/O layer for the alignment pipeline.

Defines the pipeline's file-format contract:

    INPUT   FASTQ (plain or .gz) | BAM / uBAM / SAM / CRAM | GFA
    OUTPUT  BAM | CRAM | GFA | GBZ

Every modality in :data:`~graphmambaformer.config.MODALITIES` can be loaded
from any of the input formats; :func:`validate_modality` resolves the common
aliases (``nanopore`` -> ``ont``, ``hifi`` -> ``pacbio_hifi``, ...) and rejects
anything it does not recognize rather than letting a typo reach the encoder.

Readers turn a file into in-memory objects the pipeline consumes:
  * reads  -> ``list[ReadRecord]``   (FASTQ / BAM / SAM / CRAM)
  * graph  -> ``PangenomeGraph``     (GFA)

Writers serialize them back out:
  * reads  -> BAM / CRAM             (via pysam's bundled htslib)
  * graph  -> GFA                    (plain text)  /  GBZ  (via the ``vg`` binary)

BAM/CRAM use pysam, so no external ``samtools`` is required. GBZ is vg's binary
graph index and genuinely requires the ``vg`` executable; :func:`write_gbz`
shells out to it and raises a clear error (with the exact command) if vg is not
installed.

NOTE ON ALIGNMENTS: the model's alignment decoder / output heads are not built
yet, so reads carry alignment fields only when they come from an already-aligned
source (a truth BAM or the synthetic dataset). Reads read from FASTQ are written
as an *unaligned* BAM/CRAM (uBAM) — a valid reads container. Once the decoder
lands, the same writers emit the model's predicted alignments unchanged.

NOTE ON uBAM: a uBAM has no ``@SQ`` lines and every record is unmapped, which is
how ONT and PacBio deliver reads (it preserves per-base tags such as MM/ML that
FASTQ cannot carry). :func:`read_reads` therefore includes unmapped records when
the file is unaligned; asking for mapped-only there would silently return an
empty list.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

from .export import read_bam, read_gfa  # re-exported: BAM & GFA readers
from .synthetic import PangenomeGraph, ReadRecord, Reference, reverse_complement

# our CIGAR op chars -> BAM/pysam integer op codes
_OP2CODE = {"M": 0, "I": 1, "D": 2, "N": 3, "S": 4, "H": 5, "P": 6, "=": 7, "X": 8}
_QUERY_OPS = {"M", "I", "S", "=", "X"}
_REF_OPS = {"M", "D", "N", "=", "X"}
_MAX_PHRED = 93
FLAG_REVERSE = 0x10
FLAG_UNMAPPED = 0x4


# --------------------------------------------------------------------------- #
# Modalities
# --------------------------------------------------------------------------- #
# Spellings people actually type, mapped onto the canonical MODALITIES keys.
_MODALITY_ALIASES = {
    "nanopore": "ont", "ont_r9": "ont", "ont_r10": "ont", "oxford_nanopore": "ont",
    "hifi": "pacbio_hifi", "pacbio": "pacbio_hifi", "pb": "pacbio_hifi",
    "ccs": "pacbio_hifi", "revio": "pacbio_hifi",
    "ngs": "illumina", "short_read": "illumina", "dnbseq": "illumina",
    "mgi": "illumina", "ultima": "illumina", "ion_torrent": "illumina",
    "rna": "rna_seq", "rnaseq": "rna_seq",
    "methylation": "bisulfite", "wgbs": "bisulfite",
    "sc": "single_cell", "scrna": "single_cell",
    "10x": "linked_reads", "chromium": "linked_reads",
}


def validate_modality(modality: str) -> str:
    """Resolve a modality name to a canonical :data:`MODALITIES` key.

    A typo would otherwise ride along on every record and only surface much
    later as a ``KeyError`` in the encoder's modality embedding, so it is worth
    rejecting at the point the data is read.
    """
    from ..config import MODALITIES

    key = str(modality).strip().lower().replace("-", "_")
    if key in MODALITIES:
        return key
    if key in _MODALITY_ALIASES:
        return _MODALITY_ALIASES[key]
    raise ValueError(
        f"unknown modality {modality!r}. Valid: {sorted(MODALITIES)}. "
        f"Aliases: {sorted(_MODALITY_ALIASES)}"
    )


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _open_text(path: str):
    """Open a text file, transparently decompressing gzip.

    Detection is by magic bytes rather than extension: real FASTQ arrives
    gzipped far more often than not, and not always with a ``.gz`` suffix.
    """
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    if gzipped:
        import gzip

        return gzip.open(path, "rt")
    return open(path)


def read_fastq(path: str, modality: str = "pacbio_hifi") -> list[ReadRecord]:
    """Parse a FASTQ file into unaligned :class:`ReadRecord` objects.

    Plain or gzipped, detected by content. If a read header carries a
    ``mod=<modality>`` field (as written by :func:`write_fastq`), that modality
    is used; otherwise ``modality`` applies.
    """
    modality = validate_modality(modality)
    records: list[ReadRecord] = []

    def _flush(name: str, seq: str, qual: str) -> None:
        mod = modality
        for tok in name.split():
            if tok.startswith("mod="):
                mod = validate_modality(tok[4:])
        quals = [ord(c) - 33 for c in qual] if qual else [0] * len(seq)
        records.append(ReadRecord(
            read_id=name.split()[0] if name else f"read{len(records)}",
            ref_id=-1, modality=mod, seq=seq, quals=quals,
            ref_start=0, ref_end=0, strand=1, cigar=[],
            ref_positions=[-1] * len(seq), mapq=0,
        ))

    with _open_text(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            if not header.startswith("@"):
                continue
            seq = fh.readline().rstrip("\n")
            fh.readline()  # '+'
            qual = fh.readline().rstrip("\n")
            _flush(header[1:].rstrip("\n"), seq, qual)
    return records


def is_unaligned_bam(path: str) -> bool:
    """True when a BAM/SAM/CRAM carries no ``@SQ`` lines, i.e. it is a uBAM.

    ONT and PacBio ship unaligned BAM as their native delivery format (it keeps
    per-base tags that FASTQ cannot, such as MM/ML methylation), so this is a
    normal input, not a degenerate one.
    """
    import pysam

    from .export import _pysam_read_mode

    with pysam.AlignmentFile(path, _pysam_read_mode(path), check_sq=False) as af:
        return len(af.references) == 0


def read_reads(path: str, modality: str = "pacbio_hifi", **kwargs) -> list[ReadRecord]:
    """Dispatch a reads file to the right reader by extension.

    Handles FASTQ (plain or gzipped) and BAM/SAM/CRAM, aligned or not. For a
    uBAM every record is unmapped, so unmapped reads are included by default
    there — otherwise the read would silently yield nothing.
    """
    modality = validate_modality(modality)
    low = path.lower()
    if low.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
        return read_fastq(path, modality=modality)
    if low.endswith((".bam", ".sam", ".cram", ".ubam")):
        kwargs.setdefault("include_unmapped", is_unaligned_bam(path))
        return read_bam(path, modality=modality, **kwargs)
    raise ValueError(
        f"unrecognized reads format: {path}. "
        "Expected FASTQ (.fastq/.fq, optionally .gz) or BAM/SAM/CRAM."
    )


# --------------------------------------------------------------------------- #
# Read writers — BAM / CRAM
# --------------------------------------------------------------------------- #
def _cigar_query_len(cigar: list[tuple[str, int]]) -> int:
    return sum(n for op, n in cigar if op in _QUERY_OPS)


def _sq_from_references(
    references: Optional[dict[int, Reference]],
    contig_names: Optional[dict[int, str]] = None,
):
    """Build the SAM ``@SQ`` header and a ``ref_id -> row`` index.

    ``contig_names`` overrides the default synthetic ``ref{rid}`` names so a run
    against a real reference (e.g. GRCh38 ``chr21``) emits contig names that
    match the FASTA a downstream caller like DeepVariant is given.
    """
    if not references:
        return [], {}
    names = contig_names or {}
    order = sorted(references)
    sq = [
        {"SN": names.get(rid, f"ref{rid}"), "LN": len(references[rid].seq)}
        for rid in order
    ]
    ref_index = {rid: i for i, rid in enumerate(order)}
    return sq, ref_index


def _is_mapped(rec: ReadRecord) -> bool:
    return (rec.ref_id is not None and rec.ref_id >= 0 and bool(rec.seq)
            and rec.ref_end > rec.ref_start and bool(rec.cigar))


def _make_segment(pysam, rec: ReadRecord, ref_index: dict[int, int]):
    a = pysam.AlignedSegment()
    a.query_name = rec.read_id
    seq, quals, cigar, flag = rec.seq, list(rec.quals), list(rec.cigar), 0
    mapped = _is_mapped(rec) and rec.ref_id in ref_index

    if mapped and rec.strand == -1:  # reorient to forward strand (SAM spec)
        flag |= FLAG_REVERSE
        cigar = list(reversed(cigar))
        seq = reverse_complement(seq)
        quals = list(reversed(quals))

    # SAM invariant: query-consuming CIGAR length must equal SEQ length.
    if mapped and _cigar_query_len(cigar) != len(seq):
        mapped = False

    if mapped:
        a.flag = flag
        a.reference_id = ref_index[rec.ref_id]
        a.reference_start = rec.ref_start
        a.mapping_quality = int(rec.mapq)
        a.cigartuples = [(_OP2CODE.get(op, 0), n) for op, n in cigar]
    else:
        a.flag = FLAG_UNMAPPED
        a.reference_id = -1
        a.reference_start = -1
        a.mapping_quality = 0
        a.cigartuples = None

    a.query_sequence = seq or "*"
    if seq:
        a.query_qualities = [min(_MAX_PHRED, max(0, q)) for q in quals[: len(seq)]] \
            or None
    return a


def write_bam(
    records: Iterable[ReadRecord],
    path: str,
    references: Optional[dict[int, Reference]] = None,
    sort: bool = True,
    index: bool = True,
    contig_names: Optional[dict[int, str]] = None,
) -> str:
    """Write reads to a (sorted, indexed) BAM via pysam.

    Mapped records (from a truth BAM / the synthetic dataset) are written with
    their POS/CIGAR/MAPQ; records with no alignment become unmapped BAM records.

    ``contig_names`` maps ``ref_id`` to the ``@SQ`` name to emit, so a run
    against a real reference writes ``chr21`` (matching the caller's FASTA)
    rather than the synthetic default ``ref0``.
    """
    import pysam

    records = list(records)
    sq, ref_index = _sq_from_references(references, contig_names)
    header = {"HD": {"VN": "1.6", "SO": "coordinate" if sort else "unsorted"}}
    if sq:
        header["SQ"] = sq
    header["PG"] = [{"ID": "graphmambaformer", "PN": "graphmambaformer",
                     "DS": "reads/alignments emitted by the format I/O layer"}]

    tmp = path + ".unsorted.bam"
    with pysam.AlignmentFile(tmp, "wb", header=header) as out:
        for rec in records:
            out.write(_make_segment(pysam, rec, ref_index))

    if sort:
        pysam.sort("-o", path, tmp)
        os.remove(tmp)
        if index and sq:  # index needs coordinate-sorted refs
            pysam.index(path)
    else:
        os.replace(tmp, path)
    return path


def write_cram(
    records: Iterable[ReadRecord],
    path: str,
    reference_fasta: str,
    references: Optional[dict[int, Reference]] = None,
    sort: bool = True,
    contig_names: Optional[dict[int, str]] = None,
) -> str:
    """Write reads to CRAM (reference-compressed) via pysam.

    CRAM stores only differences from the reference, so a ``reference_fasta``
    (matching the SQ names, e.g. ``ref0``) is required. We build a sorted BAM
    first, then transcode to CRAM.
    """
    import pysam

    if not os.path.exists(reference_fasta + ".fai"):
        pysam.faidx(reference_fasta)

    tmp_bam = path + ".tmp.bam"
    write_bam(
        records, tmp_bam, references=references, sort=sort, index=False,
        contig_names=contig_names,
    )
    with pysam.AlignmentFile(tmp_bam, "rb") as bam, \
            pysam.AlignmentFile(path, "wc", template=bam,
                                reference_filename=reference_fasta) as cram:
        for aln in bam.fetch(until_eof=True):
            cram.write(aln)
    os.remove(tmp_bam)
    return path


# --------------------------------------------------------------------------- #
# Graph writers — GFA / GBZ
# --------------------------------------------------------------------------- #
def write_gfa_graph(
    graph: PangenomeGraph, path: str, contig: str = "ref", version: str = "1.0"
) -> str:
    """Write a :class:`PangenomeGraph` to GFA (inverse of :func:`read_gfa`)."""
    from .synthetic import EDGE_TYPES

    id2name = {v: k for k, v in EDGE_TYPES.items()}
    with open(path, "w") as fh:
        fh.write(f"H\tVN:Z:{version}\n")
        for nidx, nseq in enumerate(graph.node_seqs):
            rs = graph.node_ref_start[nidx] if nidx < len(graph.node_ref_start) else -1
            tags = f"\tLN:i:{len(nseq)}" + (f"\tRS:i:{rs}" if rs >= 0 else "")
            fh.write(f"S\t{contig}_n{nidx}\t{nseq}{tags}\n")
        for (src, dst), et in zip(graph.edge_index, graph.edge_type):
            a, b = f"{contig}_n{src}", f"{contig}_n{dst}"
            fh.write(f"L\t{a}\t+\t{b}\t+\t0M\tzt:Z:{id2name.get(et, str(et))}\n")
    return path


def write_gbz(gfa_path: str, gbz_path: str) -> str:
    """Convert a GFA to GBZ using the ``vg`` binary.

    GBZ is vg's compressed graph+haplotype index; there is no pure-Python
    writer, so this requires ``vg`` on PATH. Raises ``RuntimeError`` with the
    exact command to run if vg is missing.
    """
    if shutil.which("vg") is None:
        raise RuntimeError(
            "GBZ output requires the `vg` binary, which is not installed.\n"
            "Install vg (https://github.com/vgteam/vg/releases), then run:\n"
            f"  vg gbwt -G {gfa_path} --gbz-format -g {gbz_path}\n"
            "or:\n"
            f"  vg convert --gfa-in {gfa_path} --gbz-out > {gbz_path}"
        )
    # Preferred modern command; falls back to `vg convert` on older builds.
    try:
        subprocess.run(["vg", "gbwt", "-G", gfa_path, "--gbz-format",
                        "-g", gbz_path], check=True)
    except subprocess.CalledProcessError:
        with open(gbz_path, "wb") as out:
            subprocess.run(["vg", "convert", "--gfa-in", gfa_path, "--gbz-out"],
                           check=True, stdout=out)
    return gbz_path


# --------------------------------------------------------------------------- #
# Top-level convenience dispatch
# --------------------------------------------------------------------------- #
def convert_graph(in_gfa: str, out_path: str) -> str:
    """Convert a GFA to GFA (normalized round-trip) or GBZ, by out extension."""
    graph = read_gfa(in_gfa)
    if out_path.lower().endswith(".gbz"):
        return write_gbz(in_gfa, out_path)
    return write_gfa_graph(graph, out_path)


def convert_reads(
    in_path: str,
    out_path: str,
    references: Optional[dict[int, Reference]] = None,
    reference_fasta: Optional[str] = None,
    modality: str = "pacbio_hifi",
) -> str:
    """Convert a reads file (FASTQ/BAM/SAM/CRAM) to BAM or CRAM by out extension."""
    records = read_reads(in_path, modality=modality)
    low = out_path.lower()
    if low.endswith(".cram"):
        if not reference_fasta:
            raise ValueError("CRAM output requires reference_fasta=")
        return write_cram(records, out_path, reference_fasta, references=references)
    if low.endswith(".bam"):
        return write_bam(records, out_path, references=references)
    raise ValueError(f"unsupported reads output format: {out_path}")
