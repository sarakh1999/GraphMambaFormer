"""Align one set of reads against several references in a single pass.

Running ``--ref-mode both`` used to mean aligning every read against the linear
reference and then, separately, against the pangenome reference — two full
invocations that repeat all the read-side work. :class:`DualReferenceAligner`
does it once:

* **One pipeline** — the model weights, the seeding / chaining / extension
  engines and the accelerator context are built a single time and shared across
  references, instead of standing up a second pipeline.
* **One read encoding** — the neural core's input tensor is a pure function of
  the reads, so for the neural (``hybrid``) mode it is built once per batch and
  reused for every reference's forward pass. Only the graph-dependent forward
  pass itself, and the reference-specific classical stages (seeding against the
  reference's own index, graph-aware chaining), rerun per reference — because
  those genuinely differ between a linear genome and a pangenome graph.
* **One integration step** — the per-reference primaries are folded into a
  single concordance decision per read via :class:`MultiReferenceIntegrator`,
  so the caller can emit one integrated BAM *and/or* the per-reference BAMs
  from the same run.

The references passed in are expected to describe the same coordinate frame
(same ``ref_seq`` / contig window); the pangenome index simply adds graph
context on top. That is exactly what :func:`build_dual_reference_from_files`
produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .pipeline import (
    HybridAlignmentPipeline,
    PipelineStats,
    ReferenceIndex,
    _encode,
    as_read_batch,
)
from ..progress import progress
from .postprocessing import (
    ConcordanceResult,
    MultiReferenceIntegrator,
    ReferenceCandidate,
)
from .types import ReadAlignments

__all__ = ["DualReferenceAligner", "DualAlignmentResult"]


@dataclass
class DualAlignmentResult:
    """Outputs of one :meth:`DualReferenceAligner.align` call.

    ``per_reference`` keeps every reference's alignments (one list per reference,
    aligned to read order) so per-reference BAMs stay available. ``integrated``,
    when requested, holds one :class:`ConcordanceResult` per read.
    """

    read_ids: list[str]
    per_reference: dict[str, list[ReadAlignments]]
    stats: dict[str, PipelineStats]
    integrated: Optional[list[ConcordanceResult]] = None

    @property
    def names(self) -> list[str]:
        return list(self.per_reference)

    def integrated_alignments(self) -> list[ReadAlignments]:
        """The winning reference's alignments per read, in read order.

        Picks, for each read, the full :class:`ReadAlignments` from whichever
        reference the integrator chose as primary (falling back to the first
        reference when no candidate mapped). Because the references share a
        coordinate frame, these records can be written to a single BAM.
        """
        if self.integrated is None:
            raise ValueError("align(..., integrate=True) is required for integrated output")
        names = self.names
        default = names[0]
        out: list[ReadAlignments] = []
        for row, concordance in enumerate(self.integrated):
            winner = (
                concordance.primary.reference
                if concordance.primary is not None
                else default
            )
            out.append(self.per_reference[winner][row])
        return out


class DualReferenceAligner:
    """Align reads against several references in one pass over a shared pipeline."""

    def __init__(
        self,
        pipeline,
        integrator: MultiReferenceIntegrator | None = None,
        priors: Mapping | None = None,
    ):
        self.pipeline = pipeline
        self.integrator = integrator or MultiReferenceIntegrator()
        # Optional population priors, keyed by either ``(read_id, name)`` or a
        # bare reference ``name``; both default to 1.0 (uninformative).
        self.priors = dict(priors or {})

    def align(
        self,
        reads: Sequence[str] | Sequence[object],
        references: Mapping[str, ReferenceIndex],
        read_ids: Optional[Sequence[str]] = None,
        integrate: bool = True,
    ) -> DualAlignmentResult:
        if not references:
            raise ValueError("at least one reference is required")

        batch = as_read_batch(reads, read_ids)
        names = list(references)
        per_reference: dict[str, list[ReadAlignments]] = {name: [] for name in names}
        stats: dict[str, PipelineStats] = {name: PipelineStats() for name in names}

        share_encoding = self._can_share_encoding()
        batch_size = max(1, int(self.pipeline.cfg.batch_size))

        steps = range(0, len(batch), batch_size)
        for start in progress(steps, desc="dual-align", unit="batch", leave=False):
            chunk = batch.slice(start, start + batch_size)
            if not len(chunk):
                continue
            encoded = None
            if share_encoding:
                encoded = _encode(
                    chunk.seqs,
                    self.pipeline.device,
                    self.pipeline.cfg.max_read_len,
                    quals=chunk.quals,
                )
            for name in names:
                results, batch_stats = self.pipeline.align_batch(
                    chunk, references[name], chunk.ids, encoded=encoded
                )
                per_reference[name].extend(results)
                stats[name].merge(batch_stats)

        integrated = None
        if integrate:
            integrated = self._integrate(list(batch.ids), per_reference, names)
        return DualAlignmentResult(list(batch.ids), per_reference, stats, integrated)

    def _can_share_encoding(self) -> bool:
        """Only the pure neural path both encodes and encodes the whole batch."""
        return (
            isinstance(self.pipeline, HybridAlignmentPipeline)
            and self.pipeline.uses_neural_scoring
        )

    def _integrate(
        self,
        read_ids: Sequence[str],
        per_reference: Mapping[str, list[ReadAlignments]],
        names: Sequence[str],
    ) -> list[ConcordanceResult]:
        out: list[ConcordanceResult] = []
        for row, read_id in enumerate(read_ids):
            candidates: list[ReferenceCandidate] = []
            for name in names:
                record = per_reference[name][row].primary
                if record is None:
                    continue
                prior = self.priors.get(
                    (read_id, name), self.priors.get(name, 1.0)
                )
                candidates.append(ReferenceCandidate(name, record, float(prior)))
            out.append(self.integrator.integrate(candidates))
        return out
