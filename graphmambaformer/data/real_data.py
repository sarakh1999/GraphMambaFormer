"""Build training / evaluation inputs from **real** genomics files.

Where :mod:`reference_build` turns a synthetic :class:`Reference` (which already
carries its own graph and per-read truth) into a pipeline reference, this module
does the same job starting from the files a real project actually has on disk:

* a reference **FASTA** (linear genome, one contig or a windowed slice),
* an optional pangenome **GFA** graph (real HPRC window, or ``vg convert -f``
  output) so the graph towers see real nodes and edges,
* reads as **FASTQ** (inference only) or an aligned **BAM/SAM/CRAM** truth set
  (``ref_start`` / ``ref_end`` / ``cigar`` / ``mapq`` per read → supervision).

The same ``--ref-mode {linear,pangenome,both}`` switch used for synthetic data
maps onto real data here: *linear* passes only the FASTA sequence, *pangenome*
also attaches the parsed GFA graph.

Everything is coordinate-aware: when the reference is a ``region`` window
(``chr21:5,000,000-6,000,000``) the reads' reference coordinates are shifted by
the window start and any read falling outside the window is dropped, so the
truth the trainer sees is expressed in the same frame as ``reference.ref_seq``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .export import read_bam, read_gfa
from .formats import read_reads, validate_modality
from .reference_build import pangenome_graph_batch
from .synthetic import ReadRecord

__all__ = [
    "RealReference",
    "parse_region",
    "read_fasta_contig",
    "build_reference_from_files",
    "load_real_reads",
    "build_batches",
]


@dataclass
class RealReference:
    """A pipeline reference index built from real files, plus its provenance."""

    reference: object          #: the pipeline ``ReferenceIndex``
    ref_seq: str               #: the (possibly windowed) linear sequence
    contig: str                #: contig / chromosome name for the BAM ``@SQ``
    ref_id: int                #: integer id used inside the pipeline
    offset: int = 0            #: window start in the original contig (0 = full)
    with_graph: bool = False   #: whether a pangenome graph was attached
    label: str = "linear"      #: "linear" or "pangenome"
    gfa_path: Optional[str] = None
    fasta_path: Optional[str] = None
    n_nodes: int = 0
    n_edges: int = 0

    @property
    def length(self) -> int:
        return len(self.ref_seq)


def parse_region(region: Optional[str]) -> tuple[Optional[str], int, Optional[int]]:
    """Parse a ``samtools``-style region into ``(contig, start0, end)``.

    ``"chr21"`` → ``("chr21", 0, None)``; ``"chr21:1,000-2,000"`` →
    ``("chr21", 999, 2000)`` (0-based, half-open like pysam's ``fetch``).
    Returns ``(None, 0, None)`` for ``None``/empty.
    """
    if not region:
        return None, 0, None
    if ":" not in region:
        return region, 0, None
    contig, span = region.rsplit(":", 1)
    span = span.replace(",", "").strip()
    if "-" in span:
        a, b = span.split("-", 1)
        start = int(a) - 1 if a else 0            # 1-based inclusive -> 0-based
        end = int(b) if b else None
    else:
        start, end = int(span) - 1, None
    return contig, max(0, start), end


def read_fasta_contig(
    path: str,
    contig: Optional[str] = None,
    region: Optional[str] = None,
) -> tuple[str, str, int]:
    """Return ``(contig_name, sequence, offset)`` from a FASTA.

    * ``region`` (``chr21:1-1000``) wins if given and selects both the contig
      and the window; ``offset`` is the window's 0-based start.
    * else ``contig`` selects a named contig (offset 0).
    * else the first contig is used (offset 0).

    Uses pysam's faidx (bundled htslib), building the ``.fai`` on demand.
    """
    import pysam

    if not os.path.exists(path):
        raise FileNotFoundError(f"reference FASTA not found: {path}")
    if not os.path.exists(path + ".fai"):
        pysam.faidx(path)

    reg_contig, start, end = parse_region(region)
    fa = pysam.FastaFile(path)
    try:
        names = list(fa.references)
        if not names:
            raise ValueError(f"no contigs in reference {path}")
        name = reg_contig or contig or names[0]
        if name not in names:
            raise ValueError(
                f"contig {name!r} not in {path} (have: {', '.join(names[:5])}"
                f"{'...' if len(names) > 5 else ''})"
            )
        seq = fa.fetch(name, start or None, end).upper()
    finally:
        fa.close()
    return name, seq, int(start or 0)


def build_reference_from_files(
    pipeline,
    fasta: str,
    *,
    gfa: Optional[str] = None,
    contig: Optional[str] = None,
    region: Optional[str] = None,
    ref_id: int = 0,
    with_graph: Optional[bool] = None,
    kmer_size: int = 3,
    device=None,
) -> RealReference:
    """Build a pipeline reference index from a real FASTA (+ optional GFA graph).

    * ``with_graph`` defaults to ``True`` when a ``gfa`` is given, else ``False``.
      Force it either way to build a *linear* index even when a GFA exists.
    * ``region`` windows the FASTA; the returned :class:`RealReference` records
      the ``offset`` so reads can be shifted into the window's frame.
    """
    contig_name, ref_seq, offset = read_fasta_contig(fasta, contig=contig, region=region)
    want_graph = (gfa is not None) if with_graph is None else with_graph

    if not want_graph or gfa is None:
        reference = pipeline.build_reference(ref_seq, ref_id=ref_id)
        return RealReference(
            reference=reference, ref_seq=ref_seq, contig=contig_name,
            ref_id=ref_id, offset=offset, with_graph=False, label="linear",
            fasta_path=fasta,
        )

    if not os.path.exists(gfa):
        raise FileNotFoundError(f"pangenome GFA not found: {gfa}")
    graph = read_gfa(gfa)
    edge_index = (
        np.asarray(graph.edge_index, dtype=np.int64).T
        if graph.edge_index
        else np.zeros((2, 0), dtype=np.int64)
    )
    batch = pangenome_graph_batch(graph, kmer_size=kmer_size, device=device)
    reference = pipeline.build_reference(
        ref_seq,
        ref_id=ref_id,
        node_seqs=graph.node_seqs,
        node_ref_start=graph.node_ref_start,
        backbone_path=graph.backbone_path,
        edge_index=edge_index,
        graph=batch,
    )
    return RealReference(
        reference=reference, ref_seq=ref_seq, contig=contig_name,
        ref_id=ref_id, offset=offset, with_graph=True, label="pangenome",
        gfa_path=gfa, fasta_path=fasta,
        n_nodes=len(graph.node_seqs), n_edges=len(graph.edge_index),
    )


def _shift_read_into_window(
    rec: ReadRecord, offset: int, window_len: int, ref_id: int
) -> Optional[ReadRecord]:
    """Move a truth read into the ``[0, window_len)`` frame; drop it if outside.

    Real truth BAM coordinates are absolute contig positions. When the reference
    is a window, subtract the window start and keep only reads whose whole span
    lands inside the window (so the placement target is unambiguous).
    """
    if rec.ref_id == -1 or not rec.cigar:
        return None                                  # unmapped -> no truth
    start = rec.ref_start - offset
    end = rec.ref_end - offset
    if start < 0 or end > window_len:
        return None
    rec.ref_id = ref_id
    rec.ref_start = start
    rec.ref_end = end
    rec.ref_positions = [
        (p - offset) if (p is not None and p >= 0) else -1 for p in rec.ref_positions
    ]
    return rec


def load_real_reads(
    *,
    reads: Optional[Sequence[str]] = None,
    truth_bam: Optional[str] = None,
    modality: str = "illumina",
    region: Optional[str] = None,
    max_reads: Optional[int] = None,
    reference: Optional[RealReference] = None,
    require_truth: bool = False,
) -> tuple[list[ReadRecord], bool]:
    """Load real reads, returning ``(records, has_truth)``.

    * ``truth_bam`` — an aligned BAM/SAM/CRAM. Reads come back with real
      ``ref_start``/``ref_end``/``cigar``/``mapq`` (supervision), shifted into
      the reference window when ``reference.offset`` is set. **Required for
      training.**
    * ``reads`` — FASTQ(.gz)/BAM files for inference. No locus truth, so
      ``has_truth`` is ``False`` and only end-to-end alignment (not per-head
      loss) is meaningful.

    Read ids are made unique (mates aligned single-end) by suffixing an index.
    """
    modality = validate_modality(modality)
    offset = reference.offset if reference else 0
    window_len = reference.length if reference else None
    ref_id = reference.ref_id if reference else 0

    records: list[ReadRecord] = []
    has_truth = False

    if truth_bam:
        if not os.path.exists(truth_bam):
            raise FileNotFoundError(f"truth BAM not found: {truth_bam}")
        raw = read_bam(truth_bam, region=region, limit=None, modality=modality)
        for rec in raw:
            if window_len is not None:
                shifted = _shift_read_into_window(rec, offset, window_len, ref_id)
                if shifted is None:
                    continue
                records.append(shifted)
            else:
                rec.ref_id = ref_id
                records.append(rec)
            if max_reads and len(records) >= max_reads:
                break
        has_truth = True
    elif reads:
        for p in reads:
            if not os.path.exists(p):
                raise FileNotFoundError(f"reads file not found: {p}")
            batch = read_reads(p, modality=modality)
            records.extend(batch)
            if max_reads and len(records) >= max_reads:
                records = records[:max_reads]
                break
    else:
        raise ValueError("provide either truth_bam= (training/eval) or reads= (inference)")

    if require_truth and not has_truth:
        raise ValueError(
            "training on real data needs a truth alignment: pass --truth-bam "
            "(an aligned BAM/SAM/CRAM). FASTQ-only reads carry no locus labels."
        )

    for i, rec in enumerate(records):
        rec.read_id = f"{rec.read_id}#{i}"
    return records, has_truth


def build_batches(
    reads: Sequence[ReadRecord], reference: RealReference, batch_size: int
) -> list[tuple]:
    """Chunk reads into ``(reads, reference_index)`` batches for train/eval."""
    idx = reference.reference
    return [
        (list(reads[i : i + batch_size]), idx)
        for i in range(0, len(reads), batch_size)
    ]
