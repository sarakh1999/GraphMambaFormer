"""The alignment pipeline: seed -> chain -> extend -> score -> post.

Stages 1-3 are classical, array-programmed algorithms (exact-match seeding,
affine-gap anchor chaining, banded affine DP); Stage 4 is the neural scoring
bridge that lets :class:`~graphmambaformer.models.GraphMambaModel` prune anchors,
re-rank chains, and calibrate MAPQ. :func:`build_pipeline` selects the pipeline
mode — ``"hybrid"`` (accuracy, the default), ``"fast"`` (throughput), or
``"two_pass"`` (fast path plus a neural rescue for the hard tail).

Stages 1-5 of the architecture's seven are implemented here. Stage 6 (repeat /
HLA resolution) and Stage 7 (the predictive-genomics aggregation that turns
per-read multi-task outputs into a sample-level VCF and report) are not built
yet; the per-read heads those stages consume do exist, in
:mod:`graphmambaformer.heads.multitask_heads`.
"""

from .chaining import AffineChainer, ChainingContext, GraphDistanceOracle
from .extension import ExtensionEngine, WavefrontAligner, banded_affine_sw_batch
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
