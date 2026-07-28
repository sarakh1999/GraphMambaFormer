"""Configuration dataclasses for the GraphMambaFormer universal alignment engine.

These configs mirror the parameters shown in the proposed Figure 1 architecture
(``figure1_architecture_v2.html``). Current development focuses on the long-read
modalities (PacBio HiFi and ONT), but the modality registry keeps room for the
remaining modalities so encoders/adapters can be extended without breaking ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Modality registry. The integer id doubles as the modality-conditioning token
# index used by the read encoder and (later) the LoRA adapter selector.
# --------------------------------------------------------------------------- #
MODALITIES: dict[str, int] = {
    "illumina": 0,
    "pacbio_hifi": 1,
    "ont": 2,
    "rna_seq": 3,
    "bisulfite": 4,
    "single_cell": 5,
    "linked_reads": 6,
}
NUM_MODALITIES: int = len(MODALITIES)

# Modalities under active development (long reads).
LONG_READ_MODALITIES: tuple[str, ...] = ("pacbio_hifi", "ont")


def modality_id(name: str) -> int:
    """Resolve a modality name to its conditioning-token id."""
    try:
        return MODALITIES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown modality {name!r}. Known: {sorted(MODALITIES)}"
        ) from exc


@dataclass
class ReadEncoderConfig:
    """Modality-Aware Read Encoder (Figure 1A, left).

    k-mer tokenization + base-quality embedding + sinusoidal positional
    encoding + modality-conditioning token -> ``d_model``.
    """

    d_model: int = 512
    kmer_size: int = 3
    kmer_stride: int = 1
    max_quality: int = 94  # Phred quality bins covered by the embedding (0..93)
    num_modalities: int = NUM_MODALITIES
    dropout: float = 0.1
    prepend_modality_token: bool = True
    pad_idx: int = 0
    # MambaFormer's leading Mamba layer can supply positional information, so
    # sinusoidal PE is optional (set False for a "pure" MambaFormer backbone).
    use_positional_encoding: bool = True


@dataclass
class GraphEncoderConfig:
    """Reference Graph Encoder (Figure 1A, right).

    GFA/rGFA pangenome graph -> k-mer node features + Laplacian positional
    encoding + edge-type encoding -> ``d_model``.
    """

    d_model: int = 512
    kmer_size: int = 3
    d_kmer: int = 128  # dimensionality of the per-node k-mer embedding
    num_edge_types: int = 8  # 8 discrete edge types in the figure
    lap_pe_dim: int = 16  # number of Laplacian eigenvectors used as node PE
    d_edge: int = 128  # edge-type embedding dim (consumed by the future GAT layer)
    dropout: float = 0.1
    lap_pe_sign_flip: bool = True  # random eigenvector sign flip during training


@dataclass
class Mamba1Config:
    """Original Mamba-1 selective SSM mixer (reference port).

    Defaults mirror the reference repo's ``mamba_simple.Mamba``
    (krafton-ai/mambaformer-icl): ``d_state=16``, ``d_conv=4``, ``expand=2``,
    ``dt_rank="auto"`` (-> ceil(d_model / 16)).
    """

    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init: str = "random"  # "random" | "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    # Try the CUDA `mamba_ssm` kernels when available; otherwise use the
    # pure-PyTorch reference scan (CPU/MPS friendly).
    use_fast_path: bool = True
    # Share parameters between forward and reverse scans.
    tie_bidirectional: bool = False

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def resolved_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


@dataclass
class Mamba2Config:
    """Bidirectional Mamba-2 SSM mixer (Figure 1B, Layer 1).

    Defaults follow the figure: ``d_state=64``, ``d_inner=1024`` (expand=2 over
    ``d_model=512``), ``d_conv=4``, selective (Delta, B, C).
    """

    d_model: int = 512
    d_state: int = 64
    d_inner: int = 1024
    headdim: int = 64  # -> nheads = d_inner // headdim = 16
    ngroups: int = 1
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    A_init_min: float = 1.0
    A_init_max: float = 16.0
    conv_bias: bool = True
    bias: bool = False
    # Try the CUDA `mamba_ssm` kernels when available; otherwise use the
    # pure-PyTorch reference scan (CPU/MPS friendly).
    use_fast_path: bool = True
    # Share parameters between forward and reverse scans.
    tie_bidirectional: bool = False

    def __post_init__(self) -> None:
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
            )


@dataclass
class AttentionConfig:
    """Multi-head self-attention (Figure 1B, Layer 2 / MambaFormer attention).

    Defaults follow the figure: 8 heads, ``d_head=64`` (8*64 = 512 = d_model).
    Attention is bidirectional by default (``causal=False``) because alignment
    uses both upstream and downstream context, unlike autoregressive ICL.
    """

    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    dropout: float = 0.1
    causal: bool = False
    bias: bool = True
    # Optional sliding-window attention span (None = full attention). The
    # windowed variant from Figure 1B can be enabled later via this field.
    window: int | None = None


@dataclass
class GATConfig:
    """GATv2 graph-attention layer (Figure 1B, Layer 3).

    Edge-type-aware graph attention (Brody et al. 2021, arXiv:2105.14491) over
    the pangenome graph nodes. Defaults follow the figure: 4 heads, ``d_gat=128``
    hidden, 8 discrete edge types. Message passing aggregates each node's
    neighbours (plus a self-loop) with attention scores conditioned on the
    incoming edge-type embedding produced by the Reference Graph Encoder.
    """

    d_model: int = 512
    n_heads: int = 4
    d_gat: int = 128            # per-head attention hidden -> inner = n_heads * d_gat
    d_edge: int = 128           # dim of edge-type embeddings from the graph encoder
    num_edge_types: int = 8     # 8 discrete edge types (+1 internal self-loop type)
    dropout: float = 0.1
    negative_slope: float = 0.2  # LeakyReLU slope inside the GATv2 score


@dataclass
class MambaFormerConfig:
    """MambaFormer backbone (Park et al. 2024, arXiv:2402.04248).

    Mirrors the reference repo's ``MixerModel`` layer indexing for
    ``mixed_attn == "mambaformer"``: a leading Mamba block, then ``n_layer``
    interleaved blocks where (by default) even indices (``i % 2 == 0``) are
    attention and odd indices are Mamba. With ``n_layer=12`` this yields 6
    attention + 6 Mamba layers (plus the leading Mamba), matching the original.

    The flattened chain (``attention_first=True``) is ``M A M A ... M``; setting
    ``attention_first=False`` flips the parity so even indices are Mamba
    (``M M A M A ... A``).

    ``mamba_variant`` selects the SSM mixer used by the (bidirectional) Mamba
    sub-layers:
      - ``"mamba1"`` (default): reference-port pure-PyTorch Mamba-1 (``mamba1``).
      - ``"mamba2"``: the Mamba-2 / SSD mixer (``mamba``).
    """

    d_model: int = 512
    n_layer: int = 12  # interleaved blocks (default: even=attn, odd=mamba)
    leading_mamba: bool = True  # first block is Mamba (replaces positional emb)
    # When True (default, Fig d / reference), even indices are Attention.
    attention_first: bool = True
    mamba_variant: str = "mamba1"  # "mamba1" | "mamba2"
    mamba1: Mamba1Config = field(default_factory=Mamba1Config)
    mamba: Mamba2Config = field(default_factory=Mamba2Config)
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    def __post_init__(self) -> None:
        # Keep sub-mixer d_model coherent with the backbone's d_model.
        self.mamba1.d_model = self.d_model
        self.mamba.d_model = self.d_model
        self.attention.d_model = self.d_model


@dataclass
class BlockConfig:
    """A single stacked hybrid block (Figure 1B).

    Only the Mamba mixer and the SwiGLU FFN are implemented so far. The
    ``use_attention`` / ``use_gat`` flags are extension hooks so windowed
    self-attention and GATv2 layers can be dropped in later without changing
    the block's public interface.
    """

    d_model: int = 512
    use_mamba: bool = True
    use_attention: bool = False  # Layer 2: windowed multi-head self-attention
    use_gat: bool = False  # Layer 3: GATv2 graph attention
    use_ffn: bool = True
    d_ff: int = 2048
    dropout: float = 0.1
    window: int | None = 256  # sliding-window span for Layer 2 (figure: w=256)
    mamba: Mamba2Config = field(default_factory=Mamba2Config)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gat: GATConfig = field(default_factory=GATConfig)

    def __post_init__(self) -> None:
        self.mamba.d_model = self.d_model
        self.attention.d_model = self.d_model
        self.gat.d_model = self.d_model


@dataclass
class ModelConfig:
    """Top-level GraphMambaFormer encoder configuration.

    ``backbone`` selects the sequence model applied to the encoded reads:
      - ``"mambaformer"``: leading Mamba + [Attention, Mamba] x L (default).
      - ``"hybrid"``: a stack of ``n_blocks`` GraphMambaFormerBlock layers
        (Mamba (-> attention/GAT hooks) -> FFN).
    """

    d_model: int = 512
    backbone: str = "mambaformer"  # "mambaformer" | "hybrid"
    n_blocks: int = 12  # used when backbone == "hybrid"
    read_encoder: ReadEncoderConfig = field(default_factory=ReadEncoderConfig)
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    mambaformer: MambaFormerConfig = field(default_factory=MambaFormerConfig)

    def __post_init__(self) -> None:
        # Keep every sub-config's d_model coherent with the top-level value.
        self.read_encoder.d_model = self.d_model
        self.graph_encoder.d_model = self.d_model
        self.block.d_model = self.d_model
        self.block.mamba.d_model = self.d_model
        self.block.attention.d_model = self.d_model
        self.block.gat.d_model = self.d_model
        self.mambaformer.d_model = self.d_model
        self.mambaformer.mamba.d_model = self.d_model
        self.mambaformer.mamba1.d_model = self.d_model
        self.mambaformer.attention.d_model = self.d_model
