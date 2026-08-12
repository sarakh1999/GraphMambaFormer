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

from .chaining import AffineChainer, ChainingContext, GraphDistanceOracle
from .extension import ExtensionEngine, WavefrontAligner, banded_affine_sw_batch
from .end_to_end import (
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
]
