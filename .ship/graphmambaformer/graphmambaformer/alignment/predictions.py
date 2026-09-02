"""Stage 7 — calibrated predictive-genomics aggregation.

The neural heads emit anonymous logits. This module turns evidence into named,
auditable calls only when the required panel/catalogue is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VariantSite:
    contig: str
    position: int
    ref: str
    alt: str
    identifier: str = ""

    @property
    def key(self) -> str:
        return self.identifier or f"{self.contig}:{self.position}:{self.ref}>{self.alt}"


@dataclass(frozen=True)
class GenotypeCall:
    site: VariantSite
    genotype: tuple[int, int]
    quality: float
    posterior: tuple[float, float, float]
    imputed: bool = False


class VariantGenotyper:
    """Combine diploid genotype likelihoods with population allele priors."""

    GENOTYPES = ((0, 0), (0, 1), (1, 1))

    def call(
        self,
        site: VariantSite,
        log_likelihoods: Sequence[float] | None,
        alt_frequency: float = 0.5,
    ) -> GenotypeCall:
        if not 0.0 <= alt_frequency <= 1.0:
            raise ValueError("alt_frequency must be in [0, 1]")
        p = float(np.clip(alt_frequency, 1e-9, 1.0 - 1e-9))
        prior = np.array([(1 - p) ** 2, 2 * p * (1 - p), p**2], dtype=np.float64)
        imputed = log_likelihoods is None
        if log_likelihoods is None:
            log_post = np.log(prior)
        else:
            likelihood = np.asarray(log_likelihoods, dtype=np.float64)
            if likelihood.shape != (3,) or not np.isfinite(likelihood).all():
                raise ValueError("log_likelihoods must contain three finite values")
            log_post = likelihood + np.log(prior)
        posterior = np.exp(log_post - log_post.max())
        posterior /= posterior.sum()
        order = np.argsort(-posterior)
        best, second = int(order[0]), int(order[1])
        quality = -10.0 * math.log10(max(1.0 - float(posterior[best]), 1e-12))
        # Quality is based on total non-best posterior, not merely best-second.
        return GenotypeCall(
            site,
            self.GENOTYPES[best],
            min(quality, 99.0),
            tuple(float(x) for x in posterior),
            imputed,
        )

    def call_panel(
        self,
        sites: Sequence[VariantSite],
        likelihoods: Mapping[str, Sequence[float]],
        frequencies: Mapping[str, float],
    ) -> list[GenotypeCall]:
        return [
            self.call(
                site,
                likelihoods.get(site.key),
                frequencies.get(site.key, 0.5),
            )
            for site in sites
        ]


@dataclass(frozen=True)
class PhaseObservation:
    left: str
    right: str
    same_haplotype: bool
    weight: float = 1.0


@dataclass(frozen=True)
class PhasedVariant:
    site: str
    phase: int
    block: int
    confidence: float


class HaplotypePhaser:
    """Read-backed weighted parity phasing with conflict detection.

    Maximum-weight observations are accepted first. A lower-weight edge that
    contradicts an established phase block is retained as conflict evidence but
    cannot flip the already better-supported block.
    """

    def phase(
        self,
        heterozygous_sites: Sequence[str],
        observations: Sequence[PhaseObservation],
    ) -> list[PhasedVariant]:
        parent = {site: site for site in heterozygous_sites}
        parity = {site: 0 for site in heterozygous_sites}
        support = {site: 0.0 for site in heterozygous_sites}
        conflict = {site: 0.0 for site in heterozygous_sites}

        def find(site: str) -> tuple[str, int]:
            if parent[site] == site:
                return site, parity[site]
            root, up = find(parent[site])
            parity[site] ^= up
            parent[site] = root
            return root, parity[site]

        for obs in sorted(observations, key=lambda item: item.weight, reverse=True):
            if obs.left not in parent or obs.right not in parent or obs.weight <= 0:
                continue
            left_root, left_parity = find(obs.left)
            right_root, right_parity = find(obs.right)
            desired = 0 if obs.same_haplotype else 1
            if left_root == right_root:
                target = support if (left_parity ^ right_parity) == desired else conflict
                target[left_root] += obs.weight
                continue
            # parity[right_root] relative to left_root.
            parent[right_root] = left_root
            parity[right_root] = left_parity ^ right_parity ^ desired
            support[left_root] += support[right_root] + obs.weight
            conflict[left_root] += conflict[right_root]

        roots = sorted({find(site)[0] for site in heterozygous_sites})
        block_id = {root: index for index, root in enumerate(roots)}
        out = []
        for site in heterozygous_sites:
            root, value = find(site)
            total = support[root] + conflict[root]
            confidence = support[root] / total if total else 0.0
            out.append(PhasedVariant(site, value, block_id[root], confidence))
        return out


@dataclass(frozen=True)
class AncestryMarker:
    contig: str
    position: int
    log_likelihoods: tuple[float, ...]


@dataclass(frozen=True)
class AncestrySegment:
    contig: str
    start: int
    end: int
    population: str
    confidence: float


class AncestryPainter:
    """Viterbi local-ancestry painting with distance-aware switch penalties."""

    def __init__(self, populations: Sequence[str], switch_rate_per_base: float = 1e-8):
        if not populations:
            raise ValueError("at least one population is required")
        if switch_rate_per_base <= 0:
            raise ValueError("switch_rate_per_base must be positive")
        self.populations = tuple(populations)
        self.switch_rate = float(switch_rate_per_base)

    def paint(self, markers: Sequence[AncestryMarker]) -> list[AncestrySegment]:
        if not markers:
            return []
        ordered = sorted(markers, key=lambda marker: (marker.contig, marker.position))
        result: list[AncestrySegment] = []
        start = 0
        while start < len(ordered):
            stop = start + 1
            while stop < len(ordered) and ordered[stop].contig == ordered[start].contig:
                stop += 1
            result.extend(self._paint_contig(ordered[start:stop]))
            start = stop
        return result

    def _paint_contig(self, markers: Sequence[AncestryMarker]) -> list[AncestrySegment]:
        n, k = len(markers), len(self.populations)
        emissions = np.asarray([marker.log_likelihoods for marker in markers], dtype=np.float64)
        if emissions.shape != (n, k) or not np.isfinite(emissions).all():
            raise ValueError("every ancestry marker needs one finite likelihood per population")
        dp = np.full((n, k), -np.inf)
        parent = np.zeros((n, k), dtype=np.int64)
        dp[0] = emissions[0] - math.log(k)
        for row in range(1, n):
            distance = max(markers[row].position - markers[row - 1].position, 1)
            switch = float(np.clip(1.0 - math.exp(-self.switch_rate * distance), 1e-12, 1 - 1e-12))
            stay_log = math.log(1.0 - switch)
            switch_log = math.log(switch / max(k - 1, 1)) if k > 1 else -np.inf
            transitions = np.full((k, k), switch_log)
            np.fill_diagonal(transitions, stay_log)
            scores = dp[row - 1][:, None] + transitions
            parent[row] = scores.argmax(axis=0)
            dp[row] = emissions[row] + scores.max(axis=0)

        path = np.zeros(n, dtype=np.int64)
        path[-1] = int(dp[-1].argmax())
        for row in range(n - 1, 0, -1):
            path[row - 1] = parent[row, path[row]]

        segments: list[AncestrySegment] = []
        begin = 0
        for row in range(1, n + 1):
            if row < n and path[row] == path[begin]:
                continue
            state = int(path[begin])
            local = emissions[begin:row]
            margin = local[:, state] - np.max(
                np.where(np.arange(k) == state, -np.inf, local), axis=1
            ) if k > 1 else np.full(row - begin, np.inf)
            confidence = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))))
            segments.append(
                AncestrySegment(
                    markers[begin].contig,
                    markers[begin].position,
                    markers[row - 1].position + 1,
                    self.populations[state],
                    confidence,
                )
            )
            begin = row
        return segments


@dataclass(frozen=True)
class ClinicalRegion:
    name: str
    contig: str
    start: int
    end: int
    category: str = ""
    action: str = ""


class ClinicalRegionFlagger:
    """Indexed half-open interval overlap against a versioned region catalogue."""

    def __init__(self, regions: Sequence[ClinicalRegion]):
        self._regions: dict[str, list[ClinicalRegion]] = {}
        self._starts: dict[str, list[int]] = {}
        for region in regions:
            if region.start < 0 or region.end <= region.start:
                raise ValueError(f"invalid clinical interval {region.name!r}")
            self._regions.setdefault(region.contig, []).append(region)
        for values in self._regions.values():
            values.sort(key=lambda region: region.start)
        self._starts = {
            contig: [region.start for region in values]
            for contig, values in self._regions.items()
        }

    def flag(self, contig: str, start: int, end: int) -> list[ClinicalRegion]:
        if start < 0 or end <= start:
            raise ValueError("invalid query interval")
        values = self._regions.get(contig, ())
        stop = bisect.bisect_left(self._starts.get(contig, ()), end)
        return [
            region
            for region in values[:stop]
            if region.start < end and start < region.end
        ]


@dataclass(frozen=True)
class StarAlleleDefinition:
    gene: str
    name: str
    required: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()
    activity_score: float | None = None


@dataclass(frozen=True)
class PGxCall:
    gene: str
    allele1: str | None
    allele2: str | None
    phenotype: str
    confidence: float
    candidates: tuple[str, ...] = ()


class PGxStarAlleleCaller:
    """Call star alleles by explicit definition matching.

    Copy-number and structural alleles require those events to be represented in
    ``observed_variants``; absence of evidence never satisfies a required event.
    """

    def __init__(
        self,
        definitions: Sequence[StarAlleleDefinition],
        phenotype_rules: Mapping[tuple[float, float], str] | None = None,
    ):
        self.definitions = tuple(definitions)
        self.phenotype_rules = dict(phenotype_rules or {})

    def call(
        self,
        gene: str,
        haplotype1_variants: Sequence[str],
        haplotype2_variants: Sequence[str],
    ) -> PGxCall:
        first = self._matches(gene, set(haplotype1_variants))
        second = self._matches(gene, set(haplotype2_variants))
        if not first or not second:
            return PGxCall(gene, None, None, "no_call", 0.0, tuple(first + second))

        a, b = first[0], second[0]
        ambiguous = len(first) > 1 or len(second) > 1
        phenotype = "unknown"
        if a.activity_score is not None and b.activity_score is not None:
            pair = tuple(sorted((a.activity_score, b.activity_score)))
            phenotype = self.phenotype_rules.get(pair, f"activity_score_{sum(pair):g}")
        candidates = tuple(definition.name for definition in first + second)
        return PGxCall(
            gene,
            a.name,
            b.name,
            phenotype,
            0.5 if ambiguous else 1.0,
            candidates,
        )

    def _matches(self, gene: str, observed: set[str]) -> list[StarAlleleDefinition]:
        matches = [
            definition
            for definition in self.definitions
            if definition.gene == gene
            and definition.required <= observed
            and not (definition.forbidden & observed)
        ]
        # More specific definitions win; ties remain visible as ambiguity.
        matches.sort(key=lambda definition: (-len(definition.required), definition.name))
        if not matches:
            return []
        specificity = len(matches[0].required)
        return [definition for definition in matches if len(definition.required) == specificity]


@dataclass
class PredictionReport:
    genotypes: list[GenotypeCall] = field(default_factory=list)
    phase: list[PhasedVariant] = field(default_factory=list)
    ancestry: list[AncestrySegment] = field(default_factory=list)
    clinical_regions: list[ClinicalRegion] = field(default_factory=list)
    pgx: list[PGxCall] = field(default_factory=list)
