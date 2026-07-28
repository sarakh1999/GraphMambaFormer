"""GraphMambaFormer — a bidirectional Graph-Mamba-2 universal alignment engine.

Implemented so far (Figure 1A + 1B, Layer 1):
  - ModalityAwareReadEncoder   (read modality-aware input encoding)
  - ReferenceGraphEncoder      (pangenome graph input encoding)
  - BiMamba2 / Mamba2Mixer     (bidirectional Mamba-2 sequence mixer)
  - GraphMambaFormerBlock      (extensible hybrid block: attention/GAT hooks)
  - GraphMambaFormerEncoder    (top-level assembly)
"""

from .blocks.hybrid_block import GraphMambaFormerBlock, SubLayerFactory
from .blocks.mambaformer import MambaFormer
from .config import (
    LONG_READ_MODALITIES,
    MODALITIES,
    NUM_MODALITIES,
    AttentionConfig,
    BlockConfig,
    GATConfig,
    GraphEncoderConfig,
    Mamba1Config,
    Mamba2Config,
    MambaFormerConfig,
    ModelConfig,
    ReadEncoderConfig,
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
from .layers.mamba1 import Mamba1Mixer
from .layers.mamba2 import Mamba2Mixer
from .model import GraphMambaFormerEncoder
from .tokenization import KmerTokenizer
from .device import device_summary, get_device

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
    "MODALITIES",
    "NUM_MODALITIES",
    "LONG_READ_MODALITIES",
    "modality_id",
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
]
