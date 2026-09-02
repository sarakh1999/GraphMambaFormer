"""Classical pseudo-labels for real-data training without an external truth BAM.

When only FASTQ (+ reference) are available, the built-in ``fast`` aligner
produces locus / CIGAR / MAPQ labels that :class:`TargetBuilder` can supervise
against. This is distillation from the classical pipeline, not GIAB gold truth.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from ..alignment.pipeline import build_pipeline
from ..config import PipelineConfig
from ..data.alignment_io import alignments_to_records, write_alignments
from ..data.synthetic import ReadRecord

__all__ = ["pseudo_label_reads", "mapped_only"]


def mapped_only(records: Sequence[ReadRecord]) -> list[ReadRecord]:
    """Drop unmapped / empty-CIGAR rows — they cannot supervise locus heads."""
    return [
        r
        for r in records
        if r.ref_id >= 0 and r.cigar and r.ref_end > r.ref_start
    ]


def pseudo_label_reads(
    reads: Sequence,
    reference,
    *,
    modality: str = "illumina",
    batch_size: int = 64,
    device=None,
    write_bam: Optional[str] = None,
    references: Optional[dict] = None,
    contig_names: Optional[dict] = None,
    reference_fasta: Optional[str] = None,
    keep_unmapped: bool = False,
) -> list[ReadRecord]:
    """Align ``reads`` with the classical pipeline and return labeled records.

    ``reference`` must be a pipeline :class:`ReferenceIndex` (e.g.
    ``RealReference.reference``). When ``write_bam`` is set, the full
    (including unmapped) alignment set is written for inspection/reuse.
    """
    if not reads:
        return []

    classical = build_pipeline(
        PipelineConfig(mode="fast", batch_size=max(1, int(batch_size))),
        device=device,
    )
    results, stats = classical.align(list(reads), reference)
    labeled = alignments_to_records(
        results, reads, modality=modality, include_secondary=False
    )

    if write_bam:
        os.makedirs(os.path.dirname(os.path.abspath(write_bam)) or ".", exist_ok=True)
        write_alignments(
            results,
            reads,
            write_bam,
            references=references,
            reference_fasta=reference_fasta,
            modality=modality,
            contig_names=contig_names,
        )

    if keep_unmapped:
        return labeled
    kept = mapped_only(labeled)
    n_drop = len(labeled) - len(kept)
    # Surface rates so a silent all-unmapped run is obvious in the log.
    summary = stats.summary() if hasattr(stats, "summary") else stats
    print(
        f"pseudo-labels from classical aligner: "
        f"{len(kept):,} mapped / {len(labeled):,} total"
        + (f" (dropped {n_drop:,} unmapped)" if n_drop else "")
        + f"  stats={summary}"
    )
    return kept
