"""Build training / evaluation inputs from **real** genomics files.

Where :mod:`reference_build` turns a synthetic :class:`Reference` (which already
carries its own graph and per-read truth) into a pipeline reference, this module
does the same job starting from the files a real project actually has on disk:

* a reference **FASTA** (linear genome, one contig or a windowed slice),
* an optional pangenome **GFA** graph (real HPRC window, or ``vg convert -f``
  output) so the graph towers see real nodes and edges,
* reads as **FASTQ** (inference, or training via classical pseudo-labels) or an
  aligned **BAM/SAM/CRAM** truth set
  (``ref_start`` / ``ref_end`` / ``cigar`` / ``mapq`` per read → supervision).
  When training without a truth BAM, :func:`~graphmambaformer.training.pseudo_label_reads`
  maps the FASTQ with the classical pipeline and attaches those labels.

The same ``--ref-mode {linear,pangenome,both}`` switch used for synthetic data
maps onto real data here: *linear* passes only the FASTA sequence, *pangenome*
also attaches the parsed GFA graph.

Everything is coordinate-aware: when the reference is a ``region`` window
(``chr21:5,000,000-6,000,000``) the reads' reference coordinates are shifted by
the window start and any read falling outside the window is dropped, so the
truth the trainer sees is expressed in the same frame as ``reference.ref_seq``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .export import read_bam, read_gfa
from .formats import read_paired_fastq, read_reads, validate_modality
from .reference_build import pangenome_graph_batch
from .synthetic import ReadRecord

# On-disk HPRC layout under ``data/hprc/reads/<SAMPLE>/{hifi,illumina,ont}/``.
# Values are (subdirectory name, canonical modality, default file globs).
HPRC_MODALITY_DIRS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "hifi": ("hifi", "pacbio_hifi", ("*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq", "*.bam")),
    "pacbio_hifi": (
        "hifi",
        "pacbio_hifi",
        ("*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq", "*.bam"),
    ),
    "illumina": (
        "illumina",
        "illumina",
        ("*.cram", "*.bam", "*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq"),
    ),
    "ont": (
        "ont",
        "ont",
        ("*.bam", "*.cram", "*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq"),
    ),
}

__all__ = [
    "RealReference",
    "parse_region",
    "read_fasta_contig",
    "build_reference_from_files",
    "build_dual_reference_from_files",
    "load_real_reads",
    "build_batches",
    "HPRC_MODALITY_DIRS",
    "discover_hprc_reads",
    "expand_read_inputs",
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
        return _build_linear(
            pipeline, ref_seq, contig_name, offset, ref_id=ref_id, fasta=fasta
        )
    return _build_pangenome(
        pipeline, ref_seq, contig_name, offset, gfa,
        ref_id=ref_id, kmer_size=kmer_size, device=device, fasta=fasta,
    )


def _build_linear(
    pipeline, ref_seq, contig_name, offset, *, ref_id, fasta
) -> RealReference:
    """Wrap a linear-only pipeline index (no graph towers) as a RealReference."""
    reference = pipeline.build_reference(ref_seq, ref_id=ref_id)
    return RealReference(
        reference=reference, ref_seq=ref_seq, contig=contig_name,
        ref_id=ref_id, offset=offset, with_graph=False, label="linear",
        fasta_path=fasta,
    )


def _build_pangenome(
    pipeline, ref_seq, contig_name, offset, gfa, *, ref_id, kmer_size, device, fasta
) -> RealReference:
    """Attach a parsed GFA graph on top of ``ref_seq`` as a RealReference."""
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


def build_dual_reference_from_files(
    pipeline,
    fasta: str,
    *,
    gfa: str,
    contig: Optional[str] = None,
    region: Optional[str] = None,
    ref_id: int = 0,
    kmer_size: int = 3,
    device=None,
) -> dict[str, RealReference]:
    """Build the linear **and** pangenome references from one FASTA read.

    Both arms share the same (windowed) ``ref_seq`` and coordinate frame — the
    pangenome index just adds graph context on top — so the FASTA is parsed once
    here instead of once per :func:`build_reference_from_files` call. The result
    (``{"linear": ..., "pangenome": ...}``) is exactly what
    :class:`~graphmambaformer.alignment.DualReferenceAligner` consumes to align
    against both in a single pass.
    """
    if not gfa:
        raise ValueError(
            "build_dual_reference_from_files needs a GFA for the pangenome arm"
        )
    contig_name, ref_seq, offset = read_fasta_contig(fasta, contig=contig, region=region)
    linear = _build_linear(
        pipeline, ref_seq, contig_name, offset, ref_id=ref_id, fasta=fasta
    )
    pangenome = _build_pangenome(
        pipeline, ref_seq, contig_name, offset, gfa,
        ref_id=ref_id, kmer_size=kmer_size, device=device, fasta=fasta,
    )
    return {"linear": linear, "pangenome": pangenome}


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


def expand_read_inputs(inputs: Sequence[str]) -> list[str]:
    """Expand paths / directories / globs into a sorted list of concrete files."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in inputs:
        path = os.path.expanduser(raw)
        candidates: list[str] = []
        if any(ch in path for ch in "*?[]"):
            candidates = sorted(glob.glob(path))
        elif os.path.isdir(path):
            for pattern in (
                "*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq",
                "*.bam", "*.ubam", "*.cram", "*.sam",
            ):
                candidates.extend(sorted(glob.glob(os.path.join(path, pattern))))
        else:
            candidates = [path]
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def discover_hprc_reads(
    sample: str,
    modality: str,
    *,
    reads_root: str = "data/hprc/reads",
) -> tuple[list[str], str]:
    """Resolve HPRC on-disk files for one sample + modality.

    Expects::

        {reads_root}/{sample}/hifi/*.fastq.gz
        {reads_root}/{sample}/illumina/*.cram
        {reads_root}/{sample}/ont/*.bam

    Returns ``(paths, canonical_modality)``.
    """
    key = str(modality).strip().lower().replace("-", "_")
    if key not in HPRC_MODALITY_DIRS:
        # Allow canonical names via validate_modality aliases.
        canon = validate_modality(modality)
        for alias, (subdir, canon_name, patterns) in HPRC_MODALITY_DIRS.items():
            if canon_name == canon:
                key = alias
                break
        else:
            raise ValueError(
                f"unsupported HPRC modality {modality!r}. "
                f"Use one of: {sorted(set(HPRC_MODALITY_DIRS))}"
            )
    subdir, canon, patterns = HPRC_MODALITY_DIRS[key]
    folder = os.path.join(reads_root, sample, subdir)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"HPRC reads folder not found: {folder} "
            f"(expected layout reads_root/sample/{{hifi,illumina,ont}}/)"
        )
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(os.path.join(folder, pattern))))
    # De-dupe while preserving order.
    seen: set[str] = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if not uniq:
        raise FileNotFoundError(
            f"no {canon} read files under {folder} (looked for {patterns})"
        )
    return uniq, canon


def load_real_reads(
    *,
    reads: Optional[Sequence[str]] = None,
    truth_bam: Optional[str] = None,
    modality: str = "illumina",
    region: Optional[str] = None,
    max_reads: Optional[int] = None,
    reference: Optional[RealReference] = None,
    require_truth: bool = False,
    layout: str = "auto",
    reference_fasta: Optional[str] = None,
    as_sequences: Optional[bool] = None,
) -> tuple[list[ReadRecord], bool]:
    """Load real reads, returning ``(records, has_truth)``.

    * ``truth_bam`` — an aligned BAM/SAM/CRAM. Reads come back with real
      ``ref_start``/``ref_end``/``cigar``/``mapq`` (supervision), shifted into
      the reference window when ``reference.offset`` is set. **Required for
      training.**
    * ``reads`` — FASTQ(.gz)/BAM/CRAM files for inference. Paths may be files,
      directories, or globs. Two Illumina FASTQs are auto-detected as paired
      R1/R2; single FASTQ/BAM/CRAM remains single-end. ``layout`` can force
      ``single`` or ``paired``.
    * ``as_sequences`` — when loading aligned BAM/CRAM as *input* (e.g. HPRC
      Illumina ``.final.cram`` or ONT Dorado BAMs), strip prior coordinates so
      the aligner remaps them. Defaults to ``True`` for BAM/CRAM under
      ``reads=`` and ``False`` for ``truth_bam=``.
    * ``reference_fasta`` — required to decode many CRAM inputs.

    Paired mates are independently aligned model rows but retain fragment
    metadata for paired BAM/SAM output. Long reads remain single-end.
    """
    modality = validate_modality(modality)
    if layout not in {"auto", "single", "paired"}:
        raise ValueError("layout must be auto, single, or paired")
    offset = reference.offset if reference else 0
    window_len = reference.length if reference else None
    ref_id = reference.ref_id if reference else 0
    fasta = reference_fasta or (reference.fasta_path if reference else None)

    records: list[ReadRecord] = []
    has_truth = False

    if truth_bam:
        if not os.path.exists(truth_bam):
            raise FileNotFoundError(f"truth BAM not found: {truth_bam}")
        raw = read_bam(
            truth_bam,
            region=region,
            limit=None,
            modality=modality,
            reference_fasta=fasta,
            as_sequences=False,
        )
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
        paths = expand_read_inputs(list(reads))
        if not paths:
            raise FileNotFoundError(f"no read files matched: {list(reads)}")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"reads file not found: {p}")
        fastq = lambda p: p.lower().endswith(  # noqa: E731
            (".fastq", ".fq", ".fastq.gz", ".fq.gz")
        )
        aligned = lambda p: p.lower().endswith(  # noqa: E731
            (".bam", ".sam", ".cram", ".ubam")
        )
        paired = layout == "paired" or (
            layout == "auto" and modality == "illumina"
            and len(paths) == 2 and all(fastq(p) for p in paths)
        )
        # Only strip when the caller opts in (align_ours does for HPRC CRAM/BAM).
        strip = bool(as_sequences)
        # FASTQ never carries coords; only BAM/CRAM need as_sequences.
        if paired:
            if len(paths) != 2 or not all(fastq(p) for p in paths):
                raise ValueError(
                    "paired layout needs exactly two FASTQ files: R1 then R2"
                )
            records = read_paired_fastq(paths[0], paths[1], modality=modality)
            if max_reads:
                records = records[:max_reads]
        else:
            for p in paths:
                if aligned(p):
                    batch = read_reads(
                        p,
                        modality=modality,
                        region=region,
                        reference_fasta=fasta,
                        as_sequences=strip,
                        include_unmapped=True,
                    )
                else:
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

    # Preserve already-unique IDs (including pair/1 and pair/2). Only suffix
    # collisions, common when multiple single-end files reuse read names.
    seen: dict[str, int] = {}
    for rec in records:
        count = seen.get(rec.read_id, 0)
        seen[rec.read_id] = count + 1
        if count:
            rec.read_id = f"{rec.read_id}#{count}"
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
