"""Seven-stage orchestration from reads through clinical prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..accel.parallel import default_worker_count, parallel_map
from ..progress import progress
from .pipeline import AlignmentPipeline, ReferenceIndex
from .postprocessing import (
    ConcordanceResult,
    ConsensusPolisher,
    CoordinateLiftover,
    CorrectionResult,
    MultiReferenceIntegrator,
    PopulationAwareMAPQ,
    ReadCorrector,
    ReferenceCandidate,
)
from .predictions import (
    AncestryMarker,
    AncestryPainter,
    ClinicalRegionFlagger,
    HaplotypePhaser,
    PGxStarAlleleCaller,
    PhaseObservation,
    PredictionReport,
    VariantGenotyper,
    VariantSite,
)
from .specialized import (
    DiagnosticSite,
    DiploidHLAType,
    DiploidMHCTyper,
    HLAAlignment,
    HLAAlleleAligner,
    ParalogDisambiguator,
    ParalogResolution,
    RepeatFamilyResolver,
    RepeatResolution,
)
from .types import AlignmentRecord, ReadAlignments


@dataclass
class PredictionEvidence:
    """Cohort/sample evidence required by Stage 7."""

    sites: Sequence[VariantSite] = ()
    genotype_log_likelihoods: Mapping[str, Sequence[float]] = field(default_factory=dict)
    allele_frequencies: Mapping[str, float] = field(default_factory=dict)
    heterozygous_sites: Sequence[str] = ()
    phase_observations: Sequence[PhaseObservation] = ()
    ancestry_markers: Sequence[AncestryMarker] = ()
    pgx_haplotypes: Mapping[str, tuple[Sequence[str], Sequence[str]]] = field(
        default_factory=dict
    )


@dataclass
class SpecializedEvidence:
    """Per-read evidence used by Stage 6."""

    repeat_families: Mapping[str, Sequence[str]] = field(default_factory=dict)
    paralog_sites: Mapping[str, Sequence[DiagnosticSite]] = field(default_factory=dict)
    paralog_priors: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    hla_gene: Mapping[str, str] = field(default_factory=dict)


@dataclass
class SevenStageResources:
    """Versioned resources used after the core aligner."""

    reference_sequences: Mapping[str, str]
    versions: Mapping[str, str] = field(default_factory=dict)
    population_priors: Mapping[tuple[str, str], float] = field(default_factory=dict)
    baseline_prior: float = 0.5
    liftover: Mapping[str, CoordinateLiftover] = field(default_factory=dict)
    repeat_resolver: RepeatFamilyResolver | None = None
    paralog_disambiguator: ParalogDisambiguator | None = None
    hla_aligner: HLAAlleleAligner | None = None
    hla_typer: DiploidMHCTyper | None = None
    ancestry_painter: AncestryPainter | None = None
    clinical_flagger: ClinicalRegionFlagger | None = None
    pgx_caller: PGxStarAlleleCaller | None = None
    #: Optional GenomeWorks ``cudapoa`` consensus polisher (Stage 5). When set,
    #: reads whose primary alignments cluster at the same locus are collapsed
    #: into a POA consensus. ``None`` (default) skips POA entirely, so behaviour
    #: is unchanged unless a polisher is supplied.
    consensus_polisher: ConsensusPolisher | None = None
    #: Locus bucket (bp) for grouping reads before POA polishing.
    consensus_locus_bucket: int = 50


@dataclass
class ReadStageResult:
    read_id: str
    alignments: Mapping[str, ReadAlignments]
    concordance: ConcordanceResult
    corrected: CorrectionResult | None = None
    repeat: RepeatResolution | None = None
    paralog: ParalogResolution | None = None
    hla: tuple[HLAAlignment, ...] = ()
    completed_stages: tuple[int, ...] = ()

    @property
    def primary(self) -> AlignmentRecord | None:
        return self.concordance.primary.record if self.concordance.primary else None


@dataclass
class LocusConsensus:
    """A GenomeWorks ``cudapoa`` consensus over the reads clustered at a locus."""

    reference: str
    ref_start: int
    depth: int
    consensus: str
    backend: str
    read_ids: tuple[str, ...] = ()


@dataclass
class SevenStageResult:
    reads: list[ReadStageResult]
    hla_types: list[DiploidHLAType]
    predictions: PredictionReport
    stage_counts: dict[int, int]
    resource_versions: dict[str, str]
    #: Per-locus POA consensus from the optional ``cudapoa`` polisher (empty
    #: unless ``SevenStageResources.consensus_polisher`` was supplied).
    locus_consensus: list["LocusConsensus"] = field(default_factory=list)


class SevenStagePipeline:
    """Run all seven diagram stages with explicit downstream resources.

    Stages 1–4 are executed by ``AlignmentPipeline`` once per reference. Stage 5
    integrates and post-processes those alignments, Stage 6 resolves specialized
    loci, and Stage 7 aggregates sample-level calls.
    """

    def __init__(
        self,
        aligner: AlignmentPipeline,
        resources: SevenStageResources,
        corrector: ReadCorrector | None = None,
        mapq_adjuster: PopulationAwareMAPQ | None = None,
        integrator: MultiReferenceIntegrator | None = None,
        strict: bool = True,
        workers: int | None = None,
    ):
        self.aligner = aligner
        self.resources = resources
        self.corrector = corrector or ReadCorrector()
        self.mapq_adjuster = mapq_adjuster or PopulationAwareMAPQ()
        self.integrator = integrator or MultiReferenceIntegrator()
        self.genotyper = VariantGenotyper()
        self.phaser = HaplotypePhaser()
        # Prefer an explicit budget, else inherit the aligner's host thread count
        # so Stages 5-6 stay congruent with Stage 1/3 parallelism.
        inherited = getattr(aligner, "_stage_workers", None)
        self._workers = default_worker_count(
            workers if workers is not None else inherited
        )
        if strict:
            if not aligner.uses_neural_scoring:
                raise ValueError(
                    "strict seven-stage execution requires an alignment model "
                    "with Stage-4 scoring heads"
                )
            if not aligner.cfg.run_extension:
                raise ValueError("strict seven-stage execution requires Stage 3")
            required = (
                "repeat_resolver",
                "paralog_disambiguator",
                "hla_aligner",
                "hla_typer",
                "ancestry_painter",
                "clinical_flagger",
                "pgx_caller",
            )
            missing = [name for name in required if getattr(resources, name) is None]
            if missing:
                raise ValueError(
                    "strict seven-stage execution requires resources: "
                    + ", ".join(missing)
                )
            version_keys = ("reference", "population", "repeat", "hla", "clinical", "pgx")
            unversioned = [name for name in version_keys if not resources.versions.get(name)]
            if unversioned:
                raise ValueError(
                    "strict seven-stage execution requires resource versions: "
                    + ", ".join(unversioned)
                )

    def run(
        self,
        reads: Sequence[str],
        references: Mapping[str, ReferenceIndex],
        read_ids: Sequence[str] | None = None,
        qualities: Sequence[Sequence[int]] | None = None,
        specialized_evidence: SpecializedEvidence | None = None,
        evidence: PredictionEvidence | None = None,
    ) -> SevenStageResult:
        if not references:
            raise ValueError("at least one reference is required")
        if set(references) - set(self.resources.reference_sequences):
            missing = sorted(set(references) - set(self.resources.reference_sequences))
            raise ValueError(f"missing reference sequences for {missing}")
        if read_ids is None:
            read_ids = [f"read{i}" for i in range(len(reads))]
        if len(read_ids) != len(reads):
            raise ValueError("read_ids length must match reads")
        if qualities is not None and len(qualities) != len(reads):
            raise ValueError("qualities length must match reads")

        per_reference: dict[str, list[ReadAlignments]] = {}
        ref_items = list(references.items())

        def _align_one(item: tuple[str, ReferenceIndex]):
            name, reference = item
            alignments, _ = self.aligner.align(reads, reference, read_ids)
            return name, alignments

        # Independent references align concurrently; each call already fans its
        # own Stage 1/3 loops across host threads.
        for name, alignments in parallel_map(
            _align_one, ref_items, workers=min(self._workers, len(ref_items)), min_items=2
        ):
            per_reference[name] = alignments

        specialized_evidence = specialized_evidence or SpecializedEvidence()
        completed = [1, 2]
        if self.aligner.cfg.run_extension:
            completed.append(3)
        if self.aligner.uses_neural_scoring:
            completed.append(4)
        completed.append(5)
        if all(
            (
                self.resources.repeat_resolver,
                self.resources.paralog_disambiguator,
                self.resources.hla_aligner,
                self.resources.hla_typer,
            )
        ):
            completed.append(6)

        # Per-read Stage 5/6 work is intentionally serial here: the heavy HLA /
        # repeat edit-distance loops hold the GIL, so a thread pool would only
        # add nesting on top of the allele-level pools inside those resolvers.
        # Those resolvers already fan out across catalogue entries themselves.
        stage_reads: list[ReadStageResult] = []
        hla_rows: list[HLAAlignment] = []
        for row, (read_id, read) in enumerate(
            progress(zip(read_ids, reads), total=len(reads),
                     desc="stage 5-7 per-read", unit="read", leave=False)
        ):
            by_reference = {
                name: alignments[row] for name, alignments in per_reference.items()
            }
            candidates: list[ReferenceCandidate] = []
            for name, alignments in by_reference.items():
                record = alignments.primary
                if record is None:
                    continue
                prior = self.resources.population_priors.get((read_id, name), 0.5)
                if record.is_mapped:
                    record.mapq = self.mapq_adjuster.adjust(
                        record.mapq, prior, self.resources.baseline_prior
                    )
                lifted = None
                if record.is_mapped and name in self.resources.liftover:
                    lifted = self.resources.liftover[name].lift(
                        name, record.ref_start, record.ref_end, record.strand
                    )
                candidates.append(ReferenceCandidate(name, record, prior, lifted))
            concordance = self.integrator.integrate(candidates)

            correction = None
            if concordance.primary is not None:
                winner = concordance.primary
                correction = self.corrector.correct(
                    read,
                    winner.record,
                    self.resources.reference_sequences[winner.reference],
                    None if qualities is None else qualities[row],
                )

            repeat = (
                self.resources.repeat_resolver.resolve(
                    read, specialized_evidence.repeat_families.get(read_id)
                )
                if self.resources.repeat_resolver is not None
                else None
            )
            paralog = (
                self.resources.paralog_disambiguator.resolve(
                    read,
                    specialized_evidence.paralog_sites.get(read_id, ()),
                    specialized_evidence.paralog_priors.get(read_id),
                )
                if self.resources.paralog_disambiguator is not None
                else None
            )
            hla: tuple[HLAAlignment, ...] = ()
            if self.resources.hla_aligner is not None:
                hla = tuple(
                    self.resources.hla_aligner.align(
                        read_id, read, gene=specialized_evidence.hla_gene.get(read_id)
                    )
                )
                hla_rows.extend(hla)
            stage_reads.append(
                ReadStageResult(
                    read_id,
                    by_reference,
                    concordance,
                    correction,
                    repeat,
                    paralog,
                    hla,
                    completed_stages=tuple(completed),
                )
            )

        hla_types: list[DiploidHLAType] = []
        if self.resources.hla_typer is not None and hla_rows:
            for gene in sorted({row.gene for row in hla_rows}):
                hla_types.append(self.resources.hla_typer.type_gene(gene, hla_rows))

        locus_consensus = self._polish_loci(reads, list(read_ids), stage_reads)

        predictions = self._predict(evidence or PredictionEvidence())
        stage7_complete = all(
            (
                self.resources.ancestry_painter,
                self.resources.clinical_flagger,
                self.resources.pgx_caller,
            )
        )
        if stage7_complete:
            for result in stage_reads:
                if 7 not in result.completed_stages:
                    result.completed_stages = result.completed_stages + (7,)
        stage_counts = {
            stage: sum(stage in row.completed_stages for row in stage_reads)
            for stage in range(1, 8)
        }
        return SevenStageResult(
            stage_reads,
            hla_types,
            predictions,
            stage_counts,
            dict(self.resources.versions),
            locus_consensus=locus_consensus,
        )

    def _polish_loci(
        self,
        reads: Sequence[str],
        read_ids: Sequence[str],
        stage_reads: Sequence[ReadStageResult],
    ) -> list["LocusConsensus"]:
        """Stage-5 GenomeWorks ``cudapoa`` polishing of per-locus read clusters.

        Reads whose primary alignment lands in the same reference locus bucket
        are collapsed into one POA consensus — the canonical ``cudapoa`` use for
        correcting a noisy pile-up. No-op unless a polisher resource is supplied
        and the ``AccelConfig.genomeworks`` master switch is on, so the default
        pipeline is unchanged.
        """
        polisher = self.resources.consensus_polisher
        if polisher is None:
            return []
        if not getattr(getattr(self.aligner, "accel", None), "cfg", None) or not getattr(
            self.aligner.accel.cfg, "genomeworks", True
        ):
            return []

        bucket = max(1, int(self.resources.consensus_locus_bucket))
        clusters: dict[tuple[str, int], list[int]] = {}
        for row, result in enumerate(stage_reads):
            primary = result.concordance.primary
            if primary is None or not primary.record.is_mapped:
                continue
            key = (primary.reference, int(primary.record.ref_start) // bucket)
            clusters.setdefault(key, []).append(row)

        # One batched cudapoa dispatch over every qualifying locus cluster: the
        # independent per-locus POAs run together on the GPU (verify-then-trust,
        # host fallback) instead of a Python loop over loci. Ordering matches the
        # old per-cluster loop (sorted clusters).
        ordered = [
            (reference, rows)
            for (reference, _bucket_idx), rows in sorted(clusters.items())
            if len(rows) >= polisher.min_depth
        ]
        if not ordered:
            return []
        polished_all = polisher.consensus_batch(
            [[reads[r] for r in rows] for _reference, rows in ordered]
        )

        out: list[LocusConsensus] = []
        for (reference, rows), polished in zip(ordered, polished_all):
            if not polished.consensus:
                continue
            out.append(
                LocusConsensus(
                    reference=reference,
                    ref_start=min(
                        int(stage_reads[r].concordance.primary.record.ref_start)
                        for r in rows
                    ),
                    depth=polished.depth,
                    consensus=polished.consensus,
                    backend=polished.backend,
                    read_ids=tuple(read_ids[r] for r in rows),
                )
            )
        return out

    def _predict(self, evidence: PredictionEvidence) -> PredictionReport:
        report = PredictionReport()
        report.genotypes = self.genotyper.call_panel(
            evidence.sites,
            evidence.genotype_log_likelihoods,
            evidence.allele_frequencies,
        )
        report.phase = self.phaser.phase(
            evidence.heterozygous_sites, evidence.phase_observations
        )
        if self.resources.ancestry_painter is not None:
            report.ancestry = self.resources.ancestry_painter.paint(
                evidence.ancestry_markers
            )
        if self.resources.clinical_flagger is not None:
            seen = set()
            for call in report.genotypes:
                for region in self.resources.clinical_flagger.flag(
                    call.site.contig, call.site.position, call.site.position + len(call.site.ref)
                ):
                    key = (region.name, region.contig, region.start, region.end)
                    if key not in seen:
                        seen.add(key)
                        report.clinical_regions.append(region)
        if self.resources.pgx_caller is not None:
            for gene, (first, second) in evidence.pgx_haplotypes.items():
                report.pgx.append(self.resources.pgx_caller.call(gene, first, second))
        return report
