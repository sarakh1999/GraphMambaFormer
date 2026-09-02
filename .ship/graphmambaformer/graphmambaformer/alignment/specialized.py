"""Stage 6 — repeat, paralog, and HLA/MHC resolution.

These resolvers consume explicit candidate references and report uncertainty.
They do not infer a biological call from a class index without an allele
catalogue, which keeps model logits and named clinical alleles separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Mapping, Sequence

import numpy as np

from ..accel.parallel import parallel_map


@dataclass(frozen=True)
class RepeatLocus:
    family: str
    contig: str
    start: int
    end: int
    consensus: str


@dataclass(frozen=True)
class RepeatResolution:
    family: str | None
    locus: RepeatLocus | None
    confidence: float
    scores: tuple[tuple[str, float], ...]


class RepeatFamilyResolver:
    """Resolve a read against repeat-family consensuses with edit likelihoods."""

    def __init__(
        self,
        loci: Sequence[RepeatLocus],
        max_distance: int = 4096,
        workers: int | None = None,
    ):
        self.loci = tuple(loci)
        self.max_distance = int(max_distance)
        self._workers = workers
        # Pre-encode consensuses once so resolve() is not dominated by uppercasing.
        self._consensus_codes = tuple(_as_codes(locus.consensus) for locus in self.loci)

    def resolve(self, read: str, candidate_families: Sequence[str] | None = None) -> RepeatResolution:
        allowed = set(candidate_families) if candidate_families is not None else None
        read_fwd = _as_codes(read)
        read_rc = _as_codes(_reverse_complement(read))
        candidates = [
            (i, locus)
            for i, locus in enumerate(self.loci)
            if allowed is None or locus.family in allowed
        ]

        def score_one(item: tuple[int, RepeatLocus]) -> tuple[RepeatLocus, float] | None:
            i, locus = item
            consensus = self._consensus_codes[i]
            distance = min(
                _semiglobal_edit_codes(read_fwd, consensus, self.max_distance),
                _semiglobal_edit_codes(read_rc, consensus, self.max_distance),
            )
            if distance > self.max_distance:
                return None
            length = max(len(read), 1)
            return locus, 1.0 - distance / length

        scored = [
            row
            for row in parallel_map(
                score_one, candidates, workers=self._workers, min_items=4
            )
            if row is not None
        ]
        if not scored:
            return RepeatResolution(None, None, 0.0, ())

        scored.sort(key=lambda item: item[1], reverse=True)
        best_locus, best = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        confidence = float(np.clip(best - second, 0.0, 1.0))
        return RepeatResolution(
            best_locus.family,
            best_locus,
            confidence,
            tuple((locus.family, score) for locus, score in scored),
        )


@dataclass(frozen=True)
class DiagnosticSite:
    """One paralog-informative site in read-oriented coordinates."""

    read_pos: int
    alleles: Mapping[str, str]
    base_quality: int = 30


@dataclass(frozen=True)
class ParalogResolution:
    paralog: str | None
    confidence: float
    log_likelihoods: tuple[tuple[str, float], ...]
    informative_sites: int


class ParalogDisambiguator:
    """Choose a paralog by base-quality-aware diagnostic-site likelihood."""

    def resolve(
        self,
        read: str,
        sites: Sequence[DiagnosticSite],
        priors: Mapping[str, float] | None = None,
    ) -> ParalogResolution:
        names = sorted({name for site in sites for name in site.alleles})
        if not names:
            return ParalogResolution(None, 0.0, (), 0)
        priors = priors or {}
        logp = {
            name: math.log(max(float(priors.get(name, 1.0 / len(names))), 1e-12))
            for name in names
        }
        informative = 0
        for site in sites:
            if not 0 <= site.read_pos < len(read):
                continue
            observed = read[site.read_pos].upper()
            if observed not in "ACGT":
                continue
            informative += 1
            error = float(np.clip(10.0 ** (-site.base_quality / 10.0), 1e-6, 0.75))
            for name in names:
                expected = site.alleles.get(name, "N").upper()
                likelihood = 1.0 - error if observed == expected else error / 3.0
                logp[name] += math.log(max(likelihood, 1e-12))

        ordered = sorted(logp.items(), key=lambda item: item[1], reverse=True)
        if informative == 0:
            return ParalogResolution(None, 0.0, tuple(ordered), 0)
        values = np.array([value for _, value in ordered], dtype=np.float64)
        posterior = np.exp(values - values.max())
        posterior /= posterior.sum()
        return ParalogResolution(
            ordered[0][0],
            float(posterior[0]),
            tuple(ordered),
            informative,
        )


@dataclass(frozen=True)
class HLAAllele:
    name: str
    sequence: str
    gene: str


@dataclass(frozen=True)
class HLAAlignment:
    read_id: str
    allele: str
    gene: str
    edit_distance: int
    identity: float
    log_likelihood: float


class HLAAlleleAligner:
    """Allele-specific alignment against a named IMGT/HLA-style catalogue."""

    def __init__(
        self,
        alleles: Sequence[HLAAllele],
        max_distance: int = 4096,
        workers: int | None = None,
    ):
        if len({allele.name for allele in alleles}) != len(alleles):
            raise ValueError("HLA allele names must be unique")
        self.alleles = tuple(alleles)
        self.max_distance = int(max_distance)
        self._workers = workers
        self._allele_codes = tuple(_as_codes(allele.sequence) for allele in self.alleles)

    def align(
        self,
        read_id: str,
        read: str,
        gene: str | None = None,
        top_k: int = 8,
        error_rate: float = 0.01,
    ) -> list[HLAAlignment]:
        if not 0.0 < error_rate < 0.75:
            raise ValueError("error_rate must be in (0, 0.75)")
        read_fwd = _as_codes(read)
        read_rc = _as_codes(_reverse_complement(read))
        candidates = [
            (i, allele)
            for i, allele in enumerate(self.alleles)
            if gene is None or allele.gene == gene
        ]
        log_match = math.log(1.0 - error_rate)
        log_mismatch = math.log(error_rate / 3.0)
        span = max(len(read), 1)

        def score_one(item: tuple[int, HLAAllele]) -> HLAAlignment | None:
            i, allele = item
            # HLA reads normally cover only a fragment of a full allele. Use a
            # semi-global distance (free allele prefix/suffix) in both read
            # orientations rather than penalising all unsequenced allele bases.
            distance = min(
                _semiglobal_edit_codes(read_fwd, self._allele_codes[i], self.max_distance),
                _semiglobal_edit_codes(read_rc, self._allele_codes[i], self.max_distance),
            )
            if distance > self.max_distance:
                return None
            matches = max(span - distance, 0)
            return HLAAlignment(
                read_id,
                allele.name,
                allele.gene,
                distance,
                1.0 - distance / span,
                matches * log_match + distance * log_mismatch,
            )

        out = [
            row
            for row in parallel_map(
                score_one, candidates, workers=self._workers, min_items=4
            )
            if row is not None
        ]
        out.sort(key=lambda item: (-item.log_likelihood, item.allele))
        return out[: max(int(top_k), 0)]


@dataclass(frozen=True)
class DiploidHLAType:
    gene: str
    allele1: str | None
    allele2: str | None
    quality: float
    log_likelihood: float
    runner_up_log_likelihood: float
    reads_used: int


class DiploidMHCTyper:
    """Maximum-likelihood diploid genotype from per-read allele likelihoods.

    For genotype ``(a,b)`` each read is generated by either chromosome with
    equal probability, so its likelihood is ``0.5 P(read|a)+0.5 P(read|b)``.
    """

    def type_gene(
        self,
        gene: str,
        alignments: Sequence[HLAAlignment],
        allele_prior: Mapping[str, float] | None = None,
    ) -> DiploidHLAType:
        rows = [row for row in alignments if row.gene == gene]
        by_read: dict[str, dict[str, float]] = {}
        for row in rows:
            by_read.setdefault(row.read_id, {})[row.allele] = row.log_likelihood
        alleles = sorted({row.allele for row in rows})
        if not alleles or not by_read:
            return DiploidHLAType(gene, None, None, 0.0, float("-inf"), float("-inf"), 0)

        prior = allele_prior or {}
        pairs: list[tuple[tuple[str, str], float]] = []
        for a, b in itertools.combinations_with_replacement(alleles, 2):
            score = math.log(max(prior.get(a, 1.0 / len(alleles)), 1e-12))
            score += math.log(max(prior.get(b, 1.0 / len(alleles)), 1e-12))
            if a != b:
                score += math.log(2.0)  # unordered Hardy-Weinberg genotype
            for likelihoods in by_read.values():
                la = likelihoods.get(a, -1e6)
                lb = likelihoods.get(b, -1e6)
                high = max(la, lb)
                score += high + math.log(0.5 * math.exp(la - high) + 0.5 * math.exp(lb - high))
            pairs.append(((a, b), score))

        pairs.sort(key=lambda item: item[1], reverse=True)
        (a, b), best = pairs[0]
        second = pairs[1][1] if len(pairs) > 1 else float("-inf")
        delta = best - second if np.isfinite(second) else 100.0
        quality = float(np.clip(10.0 * delta / math.log(10.0), 0.0, 99.0))
        return DiploidHLAType(gene, a, b, quality, best, second, len(by_read))


def _as_codes(sequence: str) -> list[int]:
    """ASCII upper-case base codes as a plain Python list (fast DP inner loop)."""
    return [ord(ch) & ~32 if 97 <= ord(ch) <= 122 else ord(ch) for ch in sequence]


def _semiglobal_edit_distance(
    query: str, target: str, max_distance: int | None = None
) -> int:
    """Edit distance from all of ``query`` to the best target substring."""
    return _semiglobal_edit_codes(
        _as_codes(query), _as_codes(target), max_distance=max_distance
    )


def _semiglobal_edit_codes(
    query: Sequence[int],
    target: Sequence[int],
    max_distance: int | None = None,
) -> int:
    """Semi-global edit distance over pre-encoded base codes.

    Free end gaps on the target (allele / consensus), so a short read is not
    charged for the unsequenced flanks. ``max_distance`` enables early abort:
    once every cell of a row exceeds the cap, further work cannot improve the
    answer, which matters when scoring large HLA catalogues.
    """
    n = len(query)
    m = len(target)
    if n == 0:
        return 0
    if m == 0:
        return n
    limit = None if max_distance is None else int(max_distance)
    previous = [0] * (m + 1)  # free target prefix
    for i, qb in enumerate(query, 1):
        current = [0] * (m + 1)
        current[0] = i
        row_min = i
        for j, tb in enumerate(target, 1):
            cost = 0 if qb == tb else 1
            val = previous[j] + 1
            ins = current[j - 1] + 1
            if ins < val:
                val = ins
            diag = previous[j - 1] + cost
            if diag < val:
                val = diag
            current[j] = val
            if val < row_min:
                row_min = val
        if limit is not None and row_min > limit:
            return limit + 1
        previous = current
    return min(previous)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]
