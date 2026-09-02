"""Independent correctness checks for Stages 5–7."""

import numpy as np

from graphmambaformer.alignment import (
    AncestryPainter,
    ClinicalRegion,
    ClinicalRegionFlagger,
    CoordinateLiftover,
    DiagnosticSite,
    DiploidMHCTyper,
    HLAAllele,
    HLAAlleleAligner,
    HaplotypePhaser,
    LiftoverBlock,
    MultiReferenceIntegrator,
    PGxStarAlleleCaller,
    ParalogDisambiguator,
    PopulationAwareMAPQ,
    PredictionEvidence,
    ReadCorrector,
    RepeatFamilyResolver,
    RepeatLocus,
    SevenStagePipeline,
    SevenStageResources,
    SpecializedEvidence,
    StarAlleleDefinition,
    VariantGenotyper,
    VariantSite,
    build_pipeline,
)
from graphmambaformer.alignment.postprocessing import ReferenceCandidate
from graphmambaformer.alignment.predictions import (
    AncestryMarker,
    PhaseObservation,
)
from graphmambaformer.alignment.types import AlignmentRecord
from graphmambaformer.config import CoreModelConfig, GraphMambaConfig, PipelineConfig
from graphmambaformer.models import build_core_model


def _record(**kwargs):
    defaults = dict(
        read_id="r1",
        read_len=4,
        ref_id=0,
        ref_start=0,
        ref_end=4,
        strand=1,
        mapq=40,
        chain_score=20.0,
        alignment_score=30.0,
        cigar=[("=", 1), ("X", 1), ("=", 2)],
        is_mapped=True,
        is_primary=True,
    )
    defaults.update(kwargs)
    return AlignmentRecord(**defaults)


def test_error_correction_is_quality_and_mapq_gated():
    result = ReadCorrector(max_base_quality=15, min_mapq=20).correct(
        "ATGT", _record(), "ACGT", [30, 5, 30, 30]
    )
    assert result.sequence == "ACGT"
    assert result.applied and result.edits[0].operation == "X"

    high_quality = ReadCorrector(max_base_quality=15).correct(
        "ATGT", _record(), "ACGT", [30, 40, 30, 30]
    )
    assert high_quality.sequence == "ATGT" and not high_quality.applied

    low_mapq = ReadCorrector(min_mapq=50).correct(
        "ATGT", _record(), "ACGT", [30, 5, 30, 30]
    )
    assert low_mapq.reason == "mapq_below_threshold"


def test_population_mapq_uses_bounded_bayesian_odds():
    adjuster = PopulationAwareMAPQ(max_prior_shift=10)
    assert adjuster.adjust(30, 0.9) > 30
    assert adjuster.adjust(30, 0.1) < 30
    assert adjuster.adjust(30, 0.5) == 30


def test_alt_liftover_handles_forward_reverse_and_gaps():
    lift = CoordinateLiftover(
        [
            LiftoverBlock("ALT1", 0, 100, "chr6", 1000, 1),
            LiftoverBlock("ALT1", 200, 300, "chr6", 2000, -1),
        ]
    )
    assert lift.lift("ALT1", 10, 20).start == 1010
    reverse = lift.lift("ALT1", 210, 220, strand=1)
    assert (reverse.start, reverse.end, reverse.strand) == (2080, 2090, -1)
    assert lift.lift("ALT1", 90, 210) is None


def test_multi_reference_concordance_preserves_disagreement():
    integrator = MultiReferenceIntegrator(concordance_slop=10)
    close = integrator.integrate(
        [
            ReferenceCandidate("chr1", _record(ref_start=100, ref_end=104)),
            ReferenceCandidate("chr1", _record(ref_start=105, ref_end=109, mapq=30)),
        ]
    )
    assert close.concordant and close.primary is not None
    far = integrator.integrate(
        [
            ReferenceCandidate("chr1", _record(ref_start=100, ref_end=104)),
            ReferenceCandidate("chr1", _record(ref_start=500, ref_end=504, mapq=30)),
        ]
    )
    assert not far.concordant and len(far.candidates) == 2


def test_repeat_and_paralog_resolution():
    resolver = RepeatFamilyResolver(
        [
            RepeatLocus("Alu", "chr1", 0, 8, "ACGTACGT"),
            RepeatLocus("LINE1", "chr2", 0, 8, "TTTTGGGG"),
        ]
    )
    call = resolver.resolve("ACGTACGT")
    assert call.family == "Alu" and call.locus.contig == "chr1"

    sites = [
        DiagnosticSite(0, {"P1": "A", "P2": "G"}, 40),
        DiagnosticSite(1, {"P1": "C", "P2": "T"}, 40),
    ]
    paralog = ParalogDisambiguator().resolve("AC", sites)
    assert paralog.paralog == "P1" and paralog.confidence > 0.99


def test_hla_alignment_and_diploid_likelihood():
    aligner = HLAAlleleAligner(
        [
            HLAAllele("A*01:01", "GGGACGTACGTCCC", "HLA-A"),
            HLAAllele("A*02:01", "GGGTTTTGGGGCCC", "HLA-A"),
        ]
    )
    rows = []
    for index, read in enumerate(("ACGTACGT", "TTTTGGGG", "ACGTACGT", "TTTTGGGG")):
        rows.extend(aligner.align(f"r{index}", read, gene="HLA-A", top_k=2))
    typed = DiploidMHCTyper().type_gene("HLA-A", rows)
    assert {typed.allele1, typed.allele2} == {"A*01:01", "A*02:01"}
    assert typed.reads_used == 4 and typed.quality > 0


def test_variant_genotyping_and_imputation_use_priors():
    site = VariantSite("chr1", 10, "A", "G", "v1")
    genotyper = VariantGenotyper()
    observed = genotyper.call(site, [-10.0, 0.0, -10.0], alt_frequency=0.1)
    assert observed.genotype == (0, 1) and not observed.imputed
    imputed = genotyper.call(site, None, alt_frequency=0.95)
    assert imputed.genotype == (1, 1) and imputed.imputed
    assert np.isclose(sum(imputed.posterior), 1.0)


def test_haplotype_phasing_respects_weighted_parity():
    phased = HaplotypePhaser().phase(
        ["v1", "v2", "v3"],
        [
            PhaseObservation("v1", "v2", True, 10),
            PhaseObservation("v2", "v3", False, 8),
            PhaseObservation("v1", "v3", True, 1),  # weaker contradiction
        ],
    )
    values = {call.site: call.phase for call in phased}
    assert values["v1"] == values["v2"]
    assert values["v1"] != values["v3"]
    assert len({call.block for call in phased}) == 1


def test_ancestry_painting_clinical_flags_and_pgx_calls():
    painter = AncestryPainter(["EUR", "AFR"], switch_rate_per_base=1e-3)
    segments = painter.paint(
        [
            AncestryMarker("chr1", 10, (4.0, 0.0)),
            AncestryMarker("chr1", 20, (4.0, 0.0)),
            AncestryMarker("chr1", 10000, (0.0, 4.0)),
        ]
    )
    assert segments[0].population == "EUR"
    assert segments[-1].population == "AFR"

    flagger = ClinicalRegionFlagger(
        [ClinicalRegion("CYP2D6", "chr22", 100, 200, "PGx")]
    )
    assert flagger.flag("chr22", 150, 151)[0].name == "CYP2D6"
    assert flagger.flag("chr22", 200, 201) == []  # half-open boundary

    caller = PGxStarAlleleCaller(
        [
            StarAlleleDefinition("CYP2D6", "*1", frozenset(), frozenset({"v2"}), 1.0),
            StarAlleleDefinition("CYP2D6", "*2", frozenset({"v2"}), frozenset(), 1.0),
        ],
        {(1.0, 1.0): "normal_metabolizer"},
    )
    pgx = caller.call("CYP2D6", [], ["v2"])
    assert (pgx.allele1, pgx.allele2, pgx.phenotype) == (
        "*1",
        "*2",
        "normal_metabolizer",
    )


def test_seven_stage_orchestrator_runs_every_stage():
    reference_sequence = (
        "GATTACAGCGTACCTAGGCTAACCGTTAACGGCATTCGATCGTACGATGCTAGCTAGGATCCGA"
    )
    read = reference_sequence[12:44]
    cfg = PipelineConfig(mode="hybrid")
    cfg.seeding.modes = ("fmindex",)
    cfg.seeding.kmer = 7
    cfg.chaining.min_chain_score = 1.0
    gm = GraphMambaConfig(d_model=32, n_mamba_layers=1, n_gat_layers=1)
    model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=gm)
    ).model
    model.eval()
    aligner = build_pipeline(cfg, model=model, device="cpu")
    reference = aligner.build_reference(reference_sequence)

    clinical = ClinicalRegionFlagger(
        [ClinicalRegion("test_region", "ref", 12, 50, "validation")]
    )
    pgx = PGxStarAlleleCaller(
        [StarAlleleDefinition("GENE", "*1", frozenset(), frozenset(), 1.0)]
    )
    resources = SevenStageResources(
        reference_sequences={"ref": reference_sequence},
        versions={
            "reference": "test-v1",
            "population": "test-v1",
            "repeat": "test-v1",
            "hla": "test-v1",
            "clinical": "test-v1",
            "pgx": "test-v1",
        },
        repeat_resolver=RepeatFamilyResolver(
            [RepeatLocus("test_repeat", "ref", 12, 44, read)]
        ),
        paralog_disambiguator=ParalogDisambiguator(),
        hla_aligner=HLAAlleleAligner([HLAAllele("A*01:01", read, "HLA-A")]),
        hla_typer=DiploidMHCTyper(),
        ancestry_painter=AncestryPainter(["POP"]),
        clinical_flagger=clinical,
        pgx_caller=pgx,
    )
    stage6 = SpecializedEvidence(
        paralog_sites={
            "r1": [
                DiagnosticSite(0, {"P1": read[0], "P2": "N"}),
                DiagnosticSite(1, {"P1": read[1], "P2": "N"}),
            ]
        },
        hla_gene={"r1": "HLA-A"},
    )
    site = VariantSite("ref", 20, reference_sequence[20], "A", "v1")
    stage7 = PredictionEvidence(
        sites=[site],
        genotype_log_likelihoods={"v1": [-5, 0, -5]},
        heterozygous_sites=["v1"],
        ancestry_markers=[AncestryMarker("ref", 20, (0.0,))],
        pgx_haplotypes={"GENE": ([], [])},
    )
    result = SevenStagePipeline(aligner, resources).run(
        [read],
        {"ref": reference},
        ["r1"],
        specialized_evidence=stage6,
        evidence=stage7,
    )
    assert result.stage_counts == {stage: 1 for stage in range(1, 8)}
    assert result.reads[0].completed_stages == tuple(range(1, 8))
    assert result.reads[0].repeat.family == "test_repeat"
    assert result.reads[0].paralog.paralog == "P1"
    assert result.hla_types[0].allele1 == "A*01:01"
    assert result.predictions.genotypes[0].genotype == (0, 1)
    assert result.predictions.clinical_regions[0].name == "test_region"
    assert result.predictions.pgx[0].allele1 == "*1"
