"""GraphMambaFormer — a bidirectional Graph-Mamba-2 universal alignment engine.

**Core model** (Figure 1A + 1B): :class:`GraphMambaModel` — sequence encoder +
bidirectional Mamba-2 tower + pangenome graph encoder + GATv2 tower, fused by
cross-attention, with a mapping head, seed/chain scoring heads, and an optional
set of ten predictive-genomics heads (:class:`MultiTaskGraphMamba`).
:func:`build_core_model` selects the architecture; ``"graphmamba"`` is the default
and the earlier ``GraphMambaFormerEncoder`` backbones stay available as ablations.

**Alignment pipeline**: seed -> chain -> extend -> score -> post, with
:func:`build_pipeline` selecting the mode — ``"hybrid"`` (accuracy, the default),
``"fast"`` (throughput), or ``"two_pass"``. The classical stages are array
programmed and the GPU stack in :mod:`graphmambaformer.accel` supplies CuPy
RawKernels and Triton fused ops behind portable PyTorch fallbacks.

**Training**: :class:`GraphMambaLoss` covers the alignment stages and the
multi-task heads, balanced by Kendall uncertainty weighting.
"""

from .accel import AccelCapabilities, AccelContext, detect_capabilities
from .alignment import (
    AffineChainer,
    AlignmentPipeline,
    AlignmentRecord,
    AnchorSet,
    Chain,
    DualAlignmentResult,
    DualReferenceAligner,
    ExtensionEngine,
    ExtensionResult,
    FastAlignmentPipeline,
    HybridAlignmentPipeline,
    NeuralScorer,
    PipelineStats,
    PredictionEvidence,
    ReadAlignments,
    ReferenceIndex,
    SevenStagePipeline,
    SevenStageResources,
    SevenStageResult,
    SpecializedEvidence,
    SeedingEngine,
    TwoPassAligner,
    build_pipeline,
)
from .blocks.hybrid_block import GraphMambaFormerBlock, SubLayerFactory
from .blocks.mambaformer import MambaFormer
from .config import (
    CORE_ARCHITECTURES,
    LONG_READ_MODALITIES,
    MODALITIES,
    NUM_MODALITIES,
    PIPELINE_MODES,
    SEEDING_MODES,
    AccelConfig,
    AttentionConfig,
    BlockConfig,
    ChainingConfig,
    CoreModelConfig,
    ExtensionConfig,
    GATConfig,
    GraphEncoderConfig,
    GraphMambaConfig,
    LossConfig,
    Mamba1Config,
    Mamba2Config,
    MambaFormerConfig,
    ModelConfig,
    MultiTaskConfig,
    PipelineConfig,
    ReadEncoderConfig,
    ScoringConfig,
    SeedingConfig,
    modality_id,
)
from .data import (
    AlignmentDataset,
    EDGE_TYPES,
    PangenomeGraph,
    ReadRecord,
    Reference,
    Seed,
    SyntheticConfig,
    SyntheticDataset,
    build_datasets,
    collate_reads,
    generate_dataset,
    graph_to_encoder_inputs,
    load_dataset,
    preset,
    save_dataset,
)
from .encoders.graph_encoder import (
    GraphEncoding,
    ReferenceGraphEncoder,
    laplacian_positional_encoding,
)
from .encoders.read_encoder import ModalityAwareReadEncoder
from .layers.attention import MultiHeadSelfAttention
from .layers.bimamba import BiMamba1, BiMamba2
from .layers.gat import GATv2Layer
from .layers.cross_attention import CrossAttentionFusion, MultiHeadCrossAttention
from .layers.mamba1 import Mamba1Mixer
from .layers.mamba2 import Mamba2Mixer
from .losses import AlignmentLoss, GraphMambaLoss, MultiTaskLoss
from .model import GraphMambaFormerEncoder
from .models import (
    CoreModelSpec,
    GraphBatch,
    GraphMambaModel,
    GraphMambaOutput,
    MultiTaskGraphMamba,
    build_core_model,
)
from .heads import (
    ChainScoringHead,
    ComplexityRouter,
    MappingHead,
    MultiTaskHeads,
    SeedScoringHead,
)
from .encoders.sequence_encoder import SequenceEncoder
from .tokenization import KmerTokenizer
from .device import (
    device_summary,
    get_device,
    resolve_device_ids,
    unwrap_model,
    wrap_data_parallel,
)

__all__ = [
    # configs
    "ModelConfig",
    "ReadEncoderConfig",
    "GraphEncoderConfig",
    "Mamba1Config",
    "Mamba2Config",
    "AttentionConfig",
    "GATConfig",
    "MambaFormerConfig",
    "BlockConfig",
    "GraphMambaConfig",
    "CoreModelConfig",
    "MultiTaskConfig",
    "SeedingConfig",
    "ChainingConfig",
    "ExtensionConfig",
    "ScoringConfig",
    "PipelineConfig",
    "LossConfig",
    "AccelConfig",
    "MODALITIES",
    "NUM_MODALITIES",
    "LONG_READ_MODALITIES",
    "SEEDING_MODES",
    "PIPELINE_MODES",
    "CORE_ARCHITECTURES",
    "modality_id",
    # core model
    "GraphMambaModel",
    "MultiTaskGraphMamba",
    "GraphMambaOutput",
    "GraphBatch",
    "CoreModelSpec",
    "build_core_model",
    "SequenceEncoder",
    "CrossAttentionFusion",
    "MultiHeadCrossAttention",
    # heads
    "MappingHead",
    "ComplexityRouter",
    "SeedScoringHead",
    "ChainScoringHead",
    "MultiTaskHeads",
    # losses
    "GraphMambaLoss",
    "AlignmentLoss",
    "MultiTaskLoss",
    # alignment stages + pipeline
    "SeedingEngine",
    "AffineChainer",
    "ExtensionEngine",
    "NeuralScorer",
    "AnchorSet",
    "Chain",
    "ExtensionResult",
    "AlignmentRecord",
    "ReadAlignments",
    "AlignmentPipeline",
    "HybridAlignmentPipeline",
    "FastAlignmentPipeline",
    "TwoPassAligner",
    "DualReferenceAligner",
    "DualAlignmentResult",
    "ReferenceIndex",
    "SevenStagePipeline",
    "SevenStageResources",
    "SevenStageResult",
    "PipelineStats",
    "PredictionEvidence",
    "SpecializedEvidence",
    "build_pipeline",
    # gpu acceleration
    "AccelContext",
    "AccelCapabilities",
    "detect_capabilities",
    # tokenizer
    "KmerTokenizer",
    # encoders
    "ModalityAwareReadEncoder",
    "ReferenceGraphEncoder",
    "GraphEncoding",
    "laplacian_positional_encoding",
    # layers / blocks
    "Mamba1Mixer",
    "Mamba2Mixer",
    "BiMamba1",
    "BiMamba2",
    "MultiHeadSelfAttention",
    "GATv2Layer",
    "MambaFormer",
    "GraphMambaFormerBlock",
    "SubLayerFactory",
    "GraphMambaFormerEncoder",
    # synthetic data
    "SyntheticConfig",
    "SyntheticDataset",
    "generate_dataset",
    "preset",
    "Reference",
    "ReadRecord",
    "Seed",
    "PangenomeGraph",
    "EDGE_TYPES",
    "AlignmentDataset",
    "collate_reads",
    "graph_to_encoder_inputs",
    "build_datasets",
    "save_dataset",
    "load_dataset",
    # device
    "get_device",
    "device_summary",
    "resolve_device_ids",
    "wrap_data_parallel",
    "unwrap_model",
]
