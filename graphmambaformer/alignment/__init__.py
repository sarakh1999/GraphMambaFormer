"""The seven-stage alignment and predictive-genomics pipeline.

Stages 1-3 are classical, array-programmed algorithms (exact-match seeding,
affine-gap anchor chaining, banded affine DP); Stage 4 is the neural scoring
bridge that lets :class:`~graphmambaformer.models.GraphMambaModel` prune anchors,
re-rank chains, and calibrate MAPQ. :func:`build_pipeline` selects the pipeline
mode — ``"hybrid"`` (accuracy, the default), ``"fast"`` (throughput), or
``"two_pass"`` (fast path plus a neural rescue for the hard tail).

Stages 5-7 are resource-driven: post-processing, specialized Repeat/HLA
resolution, and sample-level predictive aggregation. :class:`SevenStagePipeline`
orchestrates all stages while the individual algorithms remain independently
testable.
"""

# ``agnes.py`` holds the opt-in AGNES seed-graph chainer (used only when
# ``chaining.chainer="agnes"``). It is currently *uncommitted* work that the
# recurring sync/"local push" wipe removed from the tree, so guard the import:
# the default/fast/hybrid pipelines and the WFA-GPU path do not need it, and
# this restores itself automatically once ``agnes.py`` is back. We only swallow
# the error when the file is genuinely absent, so real bugs in a restored
# ``agnes.py`` still surface.
try:
    from .agnes import (
        AgnesChainer,
        AgnesConfig,
        AgnesResult,
        AgnesSeedClassifier,
        EdgeConv,
        SeedGraph,
        build_seed_graph,
        chain_dynamic_program,
        confidence_metric,
        node_features_from_anchors,
    )
except ImportError as _agnes_exc:
    import os as _os
    import warnings as _warnings

    if _os.path.exists(_os.path.join(_os.path.dirname(__file__), "agnes.py")):
        raise  # agnes.py exists -> this is a genuine import error, do not mask it

    _warnings.warn(
        f"graphmambaformer.alignment.agnes is unavailable ({_agnes_exc}); the "
        "AGNES chainer is disabled. Restore agnes.py to re-enable it "
        "(default/fast/hybrid pipelines and WFA-GPU do not require it).",
        stacklevel=2,
    )

    class _AgnesUnavailable:
        """Bound to every AGNES symbol while ``agnes.py`` is missing.

        Instantiating or calling any of them raises a clear error instead of a
        cryptic ``NoneType`` failure downstream.
        """

        def __init__(self, *_a, **_k):
            raise RuntimeError(
                "AGNES is unavailable: graphmambaformer/alignment/agnes.py was "
                "wiped (uncommitted work). Restore it to use the AGNES chainer."
            )

    AgnesChainer = AgnesConfig = AgnesResult = AgnesSeedClassifier = _AgnesUnavailable
    EdgeConv = SeedGraph = _AgnesUnavailable
    build_seed_graph = chain_dynamic_program = _AgnesUnavailable
    confidence_metric = node_features_from_anchors = _AgnesUnavailable
from .chaining import AffineChainer, ChainingContext, GraphDistanceOracle
from .dual_reference import DualAlignmentResult, DualReferenceAligner
from .extension import ExtensionEngine, WavefrontAligner, banded_affine_sw_batch
from .end_to_end import (
    LocusConsensus,
    PredictionEvidence,
    ReadStageResult,
    SevenStagePipeline,
    SevenStageResources,
    SevenStageResult,
    SpecializedEvidence,
)
from .pipeline import (
    AlignmentPipeline,
    FastAlignmentPipeline,
    HybridAlignmentPipeline,
    PIPELINE_REGISTRY,
    PipelineStats,
    ReferenceIndex,
    TwoPassAligner,
    build_pipeline,
)
from .scoring import NeuralScorer, ScoredBatch, chain_features, encode_read_batch
from .postprocessing import (
    ConsensusPolisher,
    ConsensusResult,
    CoordinateLiftover,
    LiftoverBlock,
    MultiReferenceIntegrator,
    PopulationAwareMAPQ,
    ReadCorrector,
)
from .predictions import (
    AncestryPainter,
    ClinicalRegion,
    ClinicalRegionFlagger,
    HaplotypePhaser,
    PGxStarAlleleCaller,
    StarAlleleDefinition,
    VariantGenotyper,
    VariantSite,
)
from .seeding import (
    FMIndex,
    MinimizerIndex,
    SeedIndexBundle,
    SeedingEngine,
    encode_bases,
    reverse_complement_codes,
)
from .types import (
    AlignmentRecord,
    AnchorSet,
    Chain,
    ExtensionResult,
    ReadAlignments,
    SEEDING_SOURCES,
    cigar_read_length,
    cigar_ref_length,
    merge_cigar,
    run_length_encode,
    source_id,
)
from .specialized import (
    DiagnosticSite,
    DiploidMHCTyper,
    HLAAllele,
    HLAAlleleAligner,
    ParalogDisambiguator,
    RepeatFamilyResolver,
    RepeatLocus,
)

__all__ = [
    # types
    "AlignmentRecord",
    "AnchorSet",
    "Chain",
    "ExtensionResult",
    "ReadAlignments",
    "SEEDING_SOURCES",
    "source_id",
    "run_length_encode",
    "merge_cigar",
    "cigar_read_length",
    "cigar_ref_length",
    # stage 1
    "SeedingEngine",
    "SeedIndexBundle",
    "MinimizerIndex",
    "FMIndex",
    "encode_bases",
    "reverse_complement_codes",
    # stage 2
    "AffineChainer",
    "ChainingContext",
    "GraphDistanceOracle",
    # stage 2 — standalone AGNES hybrid chainer (Arafat et al., 2025)
    "AgnesChainer",
    "AgnesConfig",
    "AgnesResult",
    "AgnesSeedClassifier",
    "EdgeConv",
    "SeedGraph",
    "build_seed_graph",
    "chain_dynamic_program",
    "confidence_metric",
    "node_features_from_anchors",
    # stage 3
    "ExtensionEngine",
    "WavefrontAligner",
    "banded_affine_sw_batch",
    # stage 4
    "NeuralScorer",
    "ScoredBatch",
    "chain_features",
    "encode_read_batch",
    # stage 5
    "ReadCorrector",
    "ConsensusPolisher",
    "ConsensusResult",
    "PopulationAwareMAPQ",
    "LiftoverBlock",
    "CoordinateLiftover",
    "MultiReferenceIntegrator",
    # stage 6
    "RepeatLocus",
    "RepeatFamilyResolver",
    "DiagnosticSite",
    "ParalogDisambiguator",
    "HLAAllele",
    "HLAAlleleAligner",
    "DiploidMHCTyper",
    # stage 7
    "VariantSite",
    "VariantGenotyper",
    "HaplotypePhaser",
    "AncestryPainter",
    "ClinicalRegion",
    "ClinicalRegionFlagger",
    "StarAlleleDefinition",
    "PGxStarAlleleCaller",
    # seven-stage orchestration
    "PredictionEvidence",
    "SpecializedEvidence",
    "LocusConsensus",
    "SevenStageResources",
    "ReadStageResult",
    "SevenStageResult",
    "SevenStagePipeline",
    # pipeline
    "AlignmentPipeline",
    "HybridAlignmentPipeline",
    "FastAlignmentPipeline",
    "TwoPassAligner",
    "PipelineStats",
    "ReferenceIndex",
    "PIPELINE_REGISTRY",
    "build_pipeline",
    # multi-reference (linear + pangenome) in one pass
    "DualReferenceAligner",
    "DualAlignmentResult",
]
