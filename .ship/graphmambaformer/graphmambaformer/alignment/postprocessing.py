"""Stage 5 — evidence-preserving alignment post-processing.

The classes in this module deliberately separate *evidence* from *policy*.
Correction never overwrites a read in place, population priors never create a
mapping, liftover fails closed across uncovered blocks, and multi-reference
integration keeps discordant candidates visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping, Sequence

import numpy as np

from .types import AlignmentRecord


@dataclass(frozen=True)
class CorrectionEdit:
    """One proposed read correction in oriented-read coordinates."""

    read_pos: int
    ref_pos: int
    old: str
    new: str
    operation: str


@dataclass(frozen=True)
class CorrectionResult:
    """A corrected sequence plus the edits supporting it."""

    sequence: str
    edits: tuple[CorrectionEdit, ...] = ()
    applied: bool = False
    reason: str = ""


class ReadCorrector:
    """Propose conservative reference-guided corrections from an extended CIGAR.

    Only low-quality mismatches are changed. Insertions and deletions are
    reported as edits but are not applied because changing read length would
    invalidate qualities and downstream read coordinates.
    """

    def __init__(self, max_base_quality: int = 15, min_mapq: int = 20):
        self.max_base_quality = int(max_base_quality)
        self.min_mapq = int(min_mapq)

    def correct(
        self,
        read: str,
        record: AlignmentRecord,
        reference: str,
        qualities: Sequence[int] | None = None,
    ) -> CorrectionResult:
        if not record.is_mapped or not record.cigar:
            return CorrectionResult(read, reason="unmapped_or_unextended")
        if record.mapq < self.min_mapq:
            return CorrectionResult(read, reason="mapq_below_threshold")
        if record.ref_start < 0 or record.ref_end > len(reference):
            return CorrectionResult(read, reason="reference_span_out_of_bounds")

        oriented = read if record.strand > 0 else _reverse_complement(read)
        oriented_qual = (
            list(qualities)
            if qualities is not None and record.strand > 0
            else list(reversed(qualities))
            if qualities is not None
            else None
        )
        corrected = list(oriented)
        edits: list[CorrectionEdit] = []
        q = 0
        r = record.ref_start

        for op, length in record.cigar:
            if op in ("S", "H"):
                q += length if op == "S" else 0
                continue
            if op in ("=", "M"):
                q += length
                r += length
                continue
            if op == "X":
                for offset in range(length):
                    if q + offset >= len(oriented) or r + offset >= len(reference):
                        return CorrectionResult(read, reason="cigar_out_of_bounds")
                    old, new = oriented[q + offset], reference[r + offset]
                    edits.append(CorrectionEdit(q + offset, r + offset, old, new, "X"))
                    quality = oriented_qual[q + offset] if oriented_qual is not None else None
                    if quality is not None and quality <= self.max_base_quality:
                        corrected[q + offset] = new
                q += length
                r += length
                continue
            if op == "I":
                inserted = oriented[q : q + length]
                edits.append(CorrectionEdit(q, r, inserted, "", "I"))
                q += length
                continue
            if op in ("D", "N"):
                deleted = reference[r : r + length]
                edits.append(CorrectionEdit(q, r, "", deleted, op))
                r += length
                continue
            raise ValueError(f"unsupported CIGAR operation {op!r}")

        sequence = "".join(corrected)
        if record.strand < 0:
            sequence = _reverse_complement(sequence)
        applied = sequence != read
        reason = "corrected_low_quality_mismatches" if applied else "no_safe_edits"
        return CorrectionResult(sequence, tuple(edits), applied, reason)


class PopulationAwareMAPQ:
    """Bayesian MAPQ adjustment using a bounded population prior ratio.

    ``prior`` is the candidate-locus prior and ``baseline_prior`` is the prior
    assumed by the original calibrator. The adjustment is made in error-odds
    space and capped so population frequency cannot overwhelm read evidence.
    """

    def __init__(self, max_mapq: int = 60, max_prior_shift: float = 10.0):
        self.max_mapq = int(max_mapq)
        self.max_prior_shift = float(max_prior_shift)

    def adjust(
        self,
        mapq: int,
        prior: float,
        baseline_prior: float = 0.5,
    ) -> int:
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be strictly between 0 and 1")
        if not 0.0 < baseline_prior < 1.0:
            raise ValueError("baseline_prior must be strictly between 0 and 1")

        error = float(np.clip(10.0 ** (-max(mapq, 0) / 10.0), 1e-12, 1 - 1e-12))
        odds_correct = (1.0 - error) / error
        prior_odds = prior / (1.0 - prior)
        baseline_odds = baseline_prior / (1.0 - baseline_prior)
        ratio = float(
            np.clip(prior_odds / baseline_odds, 1.0 / self.max_prior_shift, self.max_prior_shift)
        )
        posterior_error = 1.0 / (1.0 + odds_correct * ratio)
        adjusted = -10.0 * math.log10(max(posterior_error, 1e-12))
        return int(np.clip(round(adjusted), 0, self.max_mapq))


@dataclass(frozen=True)
class LiftoverBlock:
    """One ungapped affine block from an ALT contig to the primary assembly."""

    alt_contig: str
    alt_start: int
    alt_end: int
    primary_contig: str
    primary_start: int
    strand: int = 1

    def __post_init__(self) -> None:
        if self.alt_start < 0 or self.alt_end <= self.alt_start:
            raise ValueError("invalid ALT interval")
        if self.primary_start < 0:
            raise ValueError("primary_start must be non-negative")
        if self.strand not in (-1, 1):
            raise ValueError("strand must be +1 or -1")

    @property
    def length(self) -> int:
        return self.alt_end - self.alt_start


@dataclass(frozen=True)
class LiftedInterval:
    contig: str
    start: int
    end: int
    strand: int


class CoordinateLiftover:
    """Exact block liftover for ALT-contig intervals.

    An interval must be wholly contained in one block. Crossing a chain gap is
    rejected instead of silently returning an approximate coordinate.
    """

    def __init__(self, blocks: Sequence[LiftoverBlock]):
        self._by_contig: dict[str, list[LiftoverBlock]] = {}
        for block in blocks:
            self._by_contig.setdefault(block.alt_contig, []).append(block)
        for contig, values in self._by_contig.items():
            values.sort(key=lambda block: block.alt_start)
            for left, right in zip(values, values[1:]):
                if right.alt_start < left.alt_end:
                    raise ValueError(f"overlapping liftover blocks on {contig}")

    def lift(self, contig: str, start: int, end: int, strand: int = 1) -> LiftedInterval | None:
        if start < 0 or end <= start or strand not in (-1, 1):
            raise ValueError("invalid source interval")
        for block in self._by_contig.get(contig, ()):
            if block.alt_start <= start and end <= block.alt_end:
                if block.strand > 0:
                    lifted_start = block.primary_start + (start - block.alt_start)
                    lifted_end = block.primary_start + (end - block.alt_start)
                else:
                    block_end = block.primary_start + block.length
                    lifted_start = block_end - (end - block.alt_start)
                    lifted_end = block_end - (start - block.alt_start)
                return LiftedInterval(
                    block.primary_contig,
                    lifted_start,
                    lifted_end,
                    strand * block.strand,
                )
        return None


@dataclass(frozen=True)
class ReferenceCandidate:
    """A candidate alignment labelled by the reference that produced it."""

    reference: str
    record: AlignmentRecord
    prior: float = 1.0
    lifted: LiftedInterval | None = None


@dataclass(frozen=True)
class ConcordanceResult:
    primary: ReferenceCandidate | None
    candidates: tuple[ReferenceCandidate, ...]
    concordant: bool
    confidence: float
    reason: str


class MultiReferenceIntegrator:
    """Integrate candidates without discarding cross-reference disagreement."""

    def __init__(self, concordance_slop: int = 50):
        self.concordance_slop = int(concordance_slop)

    @staticmethod
    def _score(candidate: ReferenceCandidate) -> float:
        record = candidate.record
        if not record.is_mapped:
            return float("-inf")
        evidence = record.alignment_score if record.alignment_score else record.chain_score
        return float(evidence) + record.mapq / 10.0 + math.log(max(candidate.prior, 1e-12))

    def integrate(self, candidates: Sequence[ReferenceCandidate]) -> ConcordanceResult:
        mapped = [candidate for candidate in candidates if candidate.record.is_mapped]
        if not mapped:
            return ConcordanceResult(None, tuple(candidates), False, 0.0, "no_mapped_candidates")

        ordered = sorted(mapped, key=self._score, reverse=True)
        best = ordered[0]
        best_interval = best.lifted or LiftedInterval(
            best.reference,
            best.record.ref_start,
            best.record.ref_end,
            best.record.strand,
        )
        agreements = 0
        for candidate in ordered:
            interval = candidate.lifted or LiftedInterval(
                candidate.reference,
                candidate.record.ref_start,
                candidate.record.ref_end,
                candidate.record.strand,
            )
            if (
                interval.contig == best_interval.contig
                and interval.strand == best_interval.strand
                and abs(interval.start - best_interval.start) <= self.concordance_slop
            ):
                agreements += 1

        concordant = agreements == len(ordered)
        if len(ordered) == 1:
            confidence = 1.0
        else:
            first, second = self._score(ordered[0]), self._score(ordered[1])
            confidence = float(1.0 / (1.0 + math.exp(-np.clip(first - second, -50, 50))))
        return ConcordanceResult(
            best,
            tuple(ordered),
            concordant,
            confidence,
            "all_references_agree" if concordant else "reference_discordance",
        )


@dataclass
class PostProcessingResult:
    """Stage-5 outputs for one read."""

    record: AlignmentRecord
    correction: CorrectionResult | None = None
    population_mapq: int | None = None
    lifted: LiftedInterval | None = None
    concordance: ConcordanceResult | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def with_mapq(record: AlignmentRecord, mapq: int) -> AlignmentRecord:
    """Return a copied record with a new MAPQ."""

    return replace(record, mapq=int(mapq))


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]
