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

    ``d_state=64`` is deliberate. ``architecture/GraphMamba_Architecture.html``
    is self-contradictory here: its forward-pass diagram says ``state_dim=128``,
    while its own ``.env`` reference says ``D_STATE=64``. Figure 1B and the
    ``.env`` block agree on 64, and 64 also lands nearer the 14.2M parameter
    budget the same document quotes (64 -> 14.9M, 128 -> 15.3M), so 64 is the
    self-consistent reading. See ``scripts/audit_architecture.py``.
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
class SequenceEncoderConfig:
    """GraphMamba SequenceEncoder (architecture: Input Encoding).

    Concatenates four views of each base and projects them down to ``d_model``::

        Base(5 -> 64) + Kmer(k=3 -> 64) + Qual(42 -> 32) + PosEncode(sin/cos -> 96)
            -> Linear(256 -> D) -> LayerNorm -> Dropout(0.1)

    The five base symbols are ``A C G T N``; ``pad_idx`` is a sixth, always-zero
    row so padded positions contribute nothing.
    """

    d_model: int = 256
    d_base: int = 64
    d_kmer: int = 64
    d_qual: int = 32
    d_pos: int = 96
    kmer_size: int = 3
    num_quality_bins: int = 42
    dropout: float = 0.1
    num_modalities: int = NUM_MODALITIES
    # Prepend a modality-conditioning token (kept from the read encoder so the
    # same modality registry drives both encoders).
    prepend_modality_token: bool = False

    @property
    def d_concat(self) -> int:
        return self.d_base + self.d_kmer + self.d_qual + self.d_pos


@dataclass
class CrossAttentionConfig:
    """CrossAttentionFusion — the read <-> graph bidirectional attention bridge.

    Read->Graph attention (Q=read, K=V=graph) and Graph->Read attention
    (Q=graph, K=V=read) run in parallel, are concatenated, and pass through an
    FFN (D -> 4D -> D) whose first half is the fused LN+Linear+GELU kernel.
    """

    d_model: int = 256
    n_heads: int = 8
    d_head: int = 32  # 8 * 32 = 256 = d_model
    d_ff_mult: int = 4
    dropout: float = 0.1
    # Pool the fused (B, L+N, D) sequence down to a single (B, D) vector for the
    # routing / mapping heads: "mean" | "max" | "attention".
    pooling: str = "attention"


@dataclass
class RouterConfig:
    """ComplexityRouter — adaptive compute routing over three cost tiers.

    A lightweight MLP scores each read's difficulty and routes it to the
    ``fast`` / ``medium`` / ``full`` compute path, which saves 30-50% of the
    FLOPs on uniquely-mapping, repeat-free reads.
    """

    d_model: int = 256
    d_hidden: int = 64
    num_routes: int = 3
    dropout: float = 0.0
    # Relative cost of each route, used by the load-balancing loss term.
    route_costs: tuple[float, ...] = (0.35, 0.65, 1.0)
    route_names: tuple[str, ...] = ("fast", "medium", "full")
    # Straight-through Gumbel sampling during training keeps the router
    # differentiable while still taking hard decisions.
    gumbel_tau: float = 1.0


@dataclass
class MappingHeadConfig:
    """MappingHead — where does this read go, and how sure are we?

    Three sibling MLPs over the fused embedding: a node classifier, a
    within-node position regressor, and a MAPQ estimator.
    """

    d_model: int = 256
    max_nodes: int = 4096  # node-classifier output width (graph is padded to this)
    max_mapq: int = 60
    dropout: float = 0.1
    #: Half-width (bp) of the window the position head regresses within. The head
    #: emits a signed ``tanh`` fraction in ``[-1, 1]`` and multiplies by this to
    #: get a base offset, matching the *local* position target the loss supervises
    #: (``TargetBuilder._local_position_target``), which uses the same scale. Kept
    #: in sync with ``TargetBuilder.position_window`` (same default).
    position_window: int = 512


@dataclass
class SeedScoringConfig:
    """Neural seed / chain scoring (Stage 4, feeding back into Stages 1-2).

    The seed scorer decides which anchors survive into chaining; the chain
    scorer re-ranks the DP chains. Both consume geometric features alongside the
    backbone's read and graph representations.
    """

    d_model: int = 256
    d_hidden: int = 128
    num_seed_features: int = 12  # matches Seed.features from the synthetic data
    num_chain_features: int = 10
    dropout: float = 0.1
    # Anchors scoring below this survive only if the chain needs them; see
    # ScoringConfig.min_anchors_kept.
    seed_keep_threshold: float = 0.5

    # AGNES-style seed-match GNN. Classical indices still generate a high-recall
    # candidate set; this network performs context-aware dynamic seed selection
    # and predicts transition confidence for the chaining DP.
    use_anchor_gnn: bool = True
    anchor_gnn_hidden: tuple[int, ...] = (64, 128, 128)
    anchor_edge_features: int = 8
    anchor_gnn_dropout: float = 0.3
    anchor_gnn_max_neighbors: int = 16
    anchor_gnn_gap_threshold: int = 500
    anchor_gnn_min_nodes: int = 5
    anchor_gnn_max_nodes: int = 1000


@dataclass
class MultiTaskConfig:
    """Which of the 10 multi-task heads to build (architecture: Multi-Task Heads).

    Heads are opt-in because each one needs its own labels; the shared backbone
    is unchanged either way, so enabling a head costs one branching MLP.
    """

    d_model: int = 256
    d_hidden: int = 128
    dropout: float = 0.1

    variant_calling: bool = False
    sv_genotyping: bool = False
    haplotype: bool = False
    hla_typing: bool = False
    bqsr: bool = False
    methylation: bool = False
    ancestry: bool = False
    copy_number: bool = False
    somatic: bool = False
    pgx: bool = False

    # Output widths for the enabled heads.
    num_genotypes: int = 3  # 0/0, 0/1, 1/1
    num_sv_types: int = 5  # DEL, INS, DUP, INV, TRA
    num_hla_alleles: int = 128
    num_quality_bins: int = 42
    num_populations: int = 5
    num_cn_states: int = 6  # CN 0-5
    num_somatic_classes: int = 4  # germline / somatic / artifact / absent
    num_pgx_alleles: int = 32

    def enabled(self) -> tuple[str, ...]:
        """Names of the heads that are switched on, in declaration order."""
        names = (
            "variant_calling",
            "sv_genotyping",
            "haplotype",
            "hla_typing",
            "bqsr",
            "methylation",
            "ancestry",
            "copy_number",
            "somatic",
            "pgx",
        )
        return tuple(n for n in names if getattr(self, n))


@dataclass
class GraphMambaConfig:
    """GraphMambaModel — the core neural model (architecture: Core Model).

    Forward pass::

        SequenceEncoder --> [BiMamba2 + WindowedSelfAttn + FFN] x 6 --.
                                                                       +--> CrossAttentionFusion
        GraphEncoder ----> GATv2Conv x 3 ------------------------------'          |
                                                                                  v
                                                   ComplexityRouter -> MappingHead
                                                                                  |
                                                                                  +-> multi-task heads

    The read tower now interleaves all three of the figure's orthogonal inductive
    biases: Mamba (sequence-state / seed chaining), windowed self-attention
    (context-dependent substitution / indel scoring, ``use_read_attention``), and
    GATv2 on the graph branch (topological reasoning), fused by cross-attention.

    Defaults: ``d_model=256``, 6 BiMamba2 layers (+ windowed self-attn), 3 GATv2
    layers, ~16M parameters.
    """

    d_model: int = 256
    n_mamba_layers: int = 6
    n_gat_layers: int = 3
    dropout: float = 0.1
    # Wrap each BiMamba2 layer in a pre-norm residual + FFN (transformer-style).
    mamba_ffn: bool = True
    d_ff: int = 1024

    sequence_encoder: SequenceEncoderConfig = field(default_factory=SequenceEncoderConfig)
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    mamba: Mamba2Config = field(default_factory=Mamba2Config)
    # Figure 1B, Layer 2: windowed multi-head self-attention interleaved into the
    # read tower (Mamba -> attention -> FFN per layer). This is the second of the
    # three orthogonal inductive biases -- context-dependent substitution / indel
    # scoring -- which the parallel-tower model was missing. Windowed (O(n*w)) so
    # it scales to long reads (ONT/HiFi up to 65k bp); ``window`` is the local span.
    read_attention: AttentionConfig = field(
        default_factory=lambda: AttentionConfig(window=256)
    )
    gat: GATConfig = field(default_factory=GATConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    mapping_head: MappingHeadConfig = field(default_factory=MappingHeadConfig)
    seed_scoring: SeedScoringConfig = field(default_factory=SeedScoringConfig)
    multi_task: MultiTaskConfig = field(default_factory=MultiTaskConfig)

    # Turn the router off to always run the full path (useful for ablations).
    use_router: bool = True
    #: Enable the read-tower windowed self-attention sublayer (Figure 1B Layer 2).
    #: On by default so training gets all three inductive biases; set False to
    #: recover the Mamba-only read tower (e.g. for an ablation or a tight budget).
    use_read_attention: bool = True
    #: Swin-style shifted windows: alternate the block phase by half a window on
    #: odd layers so a base at a block boundary in one layer is mid-block in the
    #: next, letting information cross boundaries across the stack. Only affects
    #: reads longer than the window (shorter reads take the exact full-attention
    #: path), and adds no parameters. Set False to use fixed (aligned) windows.
    read_attention_shift: bool = True
    #: Which read-tower layers carry the windowed self-attention sublayer (0-based).
    #: Recent hybrid-SSM results (Jamba, Samba, NVIDIA's Mamba study) find
    #: attention is most useful *sparse and mid-stack* rather than in every layer,
    #: so the placement is configurable independently of the Mamba stack:
    #:
    #: * ``"auto"`` (default) — a depth-adaptive sparse mid-stack schedule
    #:   (~1/3 of the layers, centered), i.e. ``(2, 3)`` for the default 6-layer
    #:   tower. This is the recommended arrangement.
    #: * ``None`` (or ``"all"``) — *every* layer carries attention, the original
    #:   1:1 Mamba:attention topology. Keeps pre-sparse checkpoints loadable
    #:   byte-for-byte; use it to restore the old behavior.
    #: * an explicit tuple of 0-based layer indices — attention only on those
    #:   layers, for hand-tuned ablations.
    #:
    #: Ignored entirely when ``use_read_attention`` is False (no attention
    #: anywhere). Resolved to a concrete tuple (or ``None``) in ``__post_init__``.
    read_attention_layers: tuple[int, ...] | str | None = "auto"

    def __post_init__(self) -> None:
        d = self.d_model
        self.sequence_encoder.d_model = d
        self.graph_encoder.d_model = d
        self.mamba.d_model = d
        self.gat.d_model = d
        self.gat.d_edge = self.graph_encoder.d_edge
        self.gat.num_edge_types = self.graph_encoder.num_edge_types
        self.cross_attention.d_model = d
        self.router.d_model = d
        self.mapping_head.d_model = d
        self.seed_scoring.d_model = d
        self.multi_task.d_model = d
        # Mamba-2 keeps expand=2 relative to d_model. Shrink headdim rather than
        # rejecting the config, so scaling d_model down for a CPU run just works.
        self.mamba.d_inner = 2 * d
        while self.mamba.headdim > 1 and self.mamba.d_inner % self.mamba.headdim != 0:
            self.mamba.headdim //= 2

        # Cross-attention must reconstruct exactly d_model, so derive d_head.
        heads = self.cross_attention.n_heads
        while heads > 1 and d % heads != 0:
            heads //= 2
        self.cross_attention.n_heads = heads
        self.cross_attention.d_head = d // heads

        # Read-tower self-attention (Figure 1B, Layer 2): keep its inner width at
        # exactly d_model (aligned with the residual stream) by deriving d_head
        # from a head count that divides d_model.
        self.read_attention.d_model = d
        rheads = self.read_attention.n_heads
        while rheads > 1 and d % rheads != 0:
            rheads //= 2
        self.read_attention.n_heads = rheads
        self.read_attention.d_head = d // rheads

        # Resolve the attention schedule. Strings are keywords: "auto" picks a
        # depth-adaptive sparse mid-stack, "all"/"every" fall back to attention in
        # every layer (encoded as None downstream). An explicit tuple is validated
        # against the stack depth so a typo fails loudly at construction rather
        # than silently dropping / duplicating an attention sublayer.
        if isinstance(self.read_attention_layers, str):
            key = self.read_attention_layers.lower()
            if key == "auto":
                self.read_attention_layers = self._auto_attention_layers(self.n_mamba_layers)
            elif key in ("all", "every"):
                self.read_attention_layers = None
            else:
                raise ValueError(
                    f"read_attention_layers string must be 'auto' or 'all', got {self.read_attention_layers!r}"
                )
        if self.read_attention_layers is not None:
            self.read_attention_layers = tuple(self.read_attention_layers)
            out_of_range = sorted(
                i for i in self.read_attention_layers
                if not 0 <= i < self.n_mamba_layers
            )
            if out_of_range:
                raise ValueError(
                    f"read_attention_layers {out_of_range} out of range for a "
                    f"{self.n_mamba_layers}-layer read tower (valid: 0..{self.n_mamba_layers - 1})"
                )

    @staticmethod
    def _auto_attention_layers(n_layers: int) -> tuple[int, ...]:
        """A sparse, mid-stack windowed-attention schedule for ``n_layers``.

        Places attention on ~1/3 of the layers, centered in the stack — the
        arrangement hybrid-SSM studies (Jamba, Samba, NVIDIA) find most effective.
        Depth-adaptive so it stays in range for shallow towers (a 1- or 2-layer
        model used in tests gets a single mid layer, never an out-of-range index).
        """
        if n_layers <= 0:
            return ()
        k = max(1, round(n_layers / 3))
        start = (n_layers - k) // 2
        return tuple(range(start, start + k))


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


# --------------------------------------------------------------------------- #
# GPU acceleration stack
# --------------------------------------------------------------------------- #
@dataclass
class AccelConfig:
    """Switches for the 5-tier GPU acceleration stack.

    Everything here degrades safely: a flag that the host cannot honour is
    ignored rather than raising, so the same config runs on an A100, A6000,
    H100, H200, or a laptop CPU. Pass ``device="cuda"`` / ``"cuda:1"`` /
    ``"mps"`` / ``"cpu"`` / ``None`` (auto).
    """

    device: str | None = None  # None / "auto" -> best available
    apply_global_switches: bool = True

    # Tier switches.
    tf32: bool = True  # TF32 matmul on Ampere+
    flash_sdp: bool = True  # Flash / mem-efficient scaled_dot_product_attention
    allow_math_sdp: bool = True  # keep the math fallback for masked attention
    cudnn_benchmark: bool = True
    cuda_rawkernels: bool = True  # CuPy RawKernel tier for the DP stages
    triton_kernels: bool = True  # fused LN+Linear+GELU
    #: GenomeWorks primitives (cudamapper/cudaaligner/cudaextender/cudapoa). When
    #: off, the classical stages skip the GenomeWorks paths (ungapped X-drop
    #: prefilter, POA consensus, …) and use their built-in torch/NumPy path.
    genomeworks: bool = True
    cuda_graphs: bool = True  # static capture for fixed-shape inference
    cuda_graph_warmup: int = 3  # eager steps before first capture

    # Mixed precision: "auto" | "bf16" | "fp16".
    amp: bool = True
    amp_dtype: str = "auto"
    #: Try TransformerEngine FP8 when the GPU has FP8 tensor cores (Ada/Hopper).
    #: On Ampere (A6000) this is ignored and BF16 AMP is used instead.
    fp8: bool = True
    #: Compile the inference forward with TensorRT / torch_tensorrt (lazy).
    tensorrt: bool = False

    # torch.compile.
    compile: bool = False
    compile_mode: str = "max-autotune"

    # ---- host-side CPU parallelism ---------------------------------------- #
    #: Worker threads for the CPU-bound stages (seeding, extension) and the
    #: supervision/prefetch feed. 0 = auto (all cores, or ``$GMF_NUM_WORKERS``).
    num_workers: int = 0
    #: Fan the per-read Stage-1/Stage-3 loops out across ``num_workers`` threads.
    stage_parallel: bool = True
    #: How many training batches to build ahead on background threads so the GPU
    #: is not starved by host-side seeding/chaining. 0 disables prefetch.
    #: 3 keeps a couple of built batches queued so a slow seeding/chaining step
    #: does not immediately stall the device; raise it if the host still cannot
    #: keep the GPU fed (costs one built batch of host memory per extra level).
    prefetch: int = 3
    #: Size torch's intra-op / BLAS thread pools to the host core count.
    set_threads: bool = True

    # Per-stage kernel override: "auto" | "torch" | "cuda_rawkernel" |
    # "genomeworks" (route the stage through the GenomeWorks primitives).
    stage_backends: dict[str, str] = field(
        default_factory=lambda: {"seeding": "auto", "chaining": "auto", "extension": "auto"}
    )


# --------------------------------------------------------------------------- #
# Alignment pipeline stages 1-3
# --------------------------------------------------------------------------- #
SEEDING_MODES: tuple[str, ...] = (
    "minimizer",
    "smem",
    "fmindex",
    "dbg",
    "fuzzy",
    "multiplex_dbg",
    "gpu_kmer",
    # GenomeWorks cudamapper: GPU-resident minimizer index; identical anchors to
    # ``gpu_kmer`` but named so a run can opt into the cudamapper mapping path.
    "cudamapper",
)


@dataclass
class SeedingConfig:
    """Stage 1 — Seeding.

    ``modes`` lists the indices to query; their anchors are merged and
    deduplicated, so combining e.g. SMEM with a fuzzy spaced-seed index recovers
    anchors in regions where exact matching fails. Defaults follow the
    architecture: ``min_seed=13``, ``max_occ=200``, minimizers at ``k=15, w=10``,
    De Bruijn at ``k=21``, MultiplexDBG over ``k=15, 21, 31``, plus spaced
    fuzzy seeds for noisy / divergent reads.
    """

    # Fuzzy spaced seeds recover anchors after contiguous exact k-mers break.
    modes: tuple[str, ...] = ("smem", "minimizer", "fuzzy")

    # Minimizer sketch.
    kmer: int = 15
    window: int = 10

    # SMEM / FM-index.
    min_seed_len: int = 13
    max_occ: int = 200  # drop k-mers occurring more often than this (repeats)
    fm_sa_sample: int = 8  # suffix-array sampling rate
    fm_occ_sample: int = 64  # rank-checkpoint spacing
    fm_stride: int = 5  # exact-k-mer FM-index query stride

    # De Bruijn / multi-k.
    dbg_kmer: int = 21
    multiplex_kmers: tuple[int, ...] = (15, 21, 31)

    # Fuzzy spaced seeds: '1' = compared position, '0' = don't-care.
    spaced_pattern: str = "111010010100110111"

    # Both strands: reads are aligned forward and reverse-complemented.
    both_strands: bool = True
    # Hard cap on anchors per read after merging (keeps chaining bounded).
    max_anchors: int = 5_000
    # Merge anchors that lie on the same diagonal within this distance.
    merge_diagonal_slack: int = 4

    # Anchor capping (``SeedingEngine._cap``) ranks anchors by diagonal-cluster
    # *support*, not by raw length. A real alignment piles many anchors onto one
    # diagonal (``ref_pos - read_pos``) while errors and repeats scatter as lonely
    # hits, so for the fixed-length k-mer anchors that dominate the set length is a
    # coin flip and capping by it can discard the true diagonal while keeping noise
    # (see ``diagonal_example``). ``diagonal_band`` is the tolerance (in diagonal
    # units == bp) for treating anchors as sharing a diagonal, which absorbs the
    # small diagonal drift a short indel introduces. ``diagonal_support_gain``
    # weights that support in the capping score ``length * (1 + gain * (support -
    # 1))``; ``0.0`` restores the legacy pure-length behaviour.
    diagonal_band: int = 12
    diagonal_support_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.fm_stride <= 0:
            raise ValueError("fm_stride must be positive")
        if self.diagonal_band < 0:
            raise ValueError("diagonal_band must be non-negative")
        if self.diagonal_support_gain < 0:
            raise ValueError("diagonal_support_gain must be non-negative")
        pattern = self.spaced_pattern
        if not pattern:
            raise ValueError("spaced_pattern must be non-empty")
        if set(pattern) - {"0", "1"}:
            raise ValueError(
                f"spaced_pattern must contain only '0' and '1', got {pattern!r}"
            )
        if "1" not in pattern:
            raise ValueError("spaced_pattern must contain at least one '1'")
        if pattern.count("1") > 31:
            raise ValueError("spaced_pattern weight must be <= 31")


@dataclass
class ChainingConfig:
    """Stage 2 — Chaining (minimap2-style affine-gap DP over anchors).

    Score of chaining anchor ``j -> i``::

        f[i] = max(w_i, max_j f[j] + advance(j, i) - penalty(j, i) + graph_bonus)
        advance = min(min(dq, dr), w_i)
        penalty = gap_open + gap_extend * gap + log_coeff * log2(gap + 1)

    ``graph_bonus`` rewards pairs whose reference nodes are close in the
    pangenome graph, and ``ref_path_bias`` additionally rewards anchors sitting
    on the graph's backbone (reference) path. ``recombination_penalty`` makes the
    DP haplotype-aware: a step between anchors with disjoint haplotype sets is
    charged a recombination cost (Li-Stephens), off by default.

    On top of the DP, ``adaptive_seed_scoring`` adds an AGNES-style confidence
    gate: when a neural seed score is available it steers the anchor weights only
    for reads whose seed-score distribution is confidently separated, and
    otherwise falls back to pure length-based chaining.
    """

    #: Stage-2 backend: ``"affine"`` (the minimap2-style DP described below, the
    #: default) or ``"agnes"`` (the standalone paper-faithful AGNES hybrid chainer
    #: in :mod:`graphmambaformer.alignment.agnes` — classical seeding, an EdgeConv
    #: GNN on pure 12-D/8-D seed-graph features, and a confidence-gated DP). With
    #: ``"agnes"`` set ``agnes_checkpoint`` to a trained classifier; left unset the
    #: AGNES chainer runs its PureDP baseline.
    chainer: str = "affine"
    agnes_checkpoint: str | None = None

    max_lookback: int = 64  # predecessors considered per anchor
    max_gap: int = 5_000  # reject anchor pairs separated by more than this
    gap_open: float = 6.0
    gap_extend: float = 0.05
    log_coeff: float = 0.5

    # Graph-distance bonus.
    graph_bonus: float = 4.0
    graph_max_hops: int = 3  # bonus decays over this many hops
    ref_path_bias: float = 1.5  # extra weight for backbone-path anchors

    # Haplotype-aware chaining (Chandra & Jain, "Haplotype-aware sequence
    # alignment to pangenome graphs", Genome Research 2024 / Minichain). When the
    # pangenome graph carries haplotype paths (GFA P- / W-lines), a step between
    # two anchors that share *no* common haplotype implies a recombination, so a
    # chain that keeps switching haplotypes is an unlikely mosaic of the known
    # panel. Following the Li-Stephens copying model, we subtract a fixed cost per
    # such switch from the chaining DP, steering chains onto self-consistent
    # haplotype mosaics and away from spurious recombinant paths.
    #
    # This is a *pairwise* relaxation of Minichain's per-anchor (anchor, haplotype)
    # state: instead of carrying the active haplotype through the DP, we penalise
    # any j -> i edge whose two anchors have disjoint haplotype sets. It captures
    # the dominant effect (reject cross-haplotype jumps) without expanding the DP
    # state by |H|, so it stays O(N * lookback) and drops straight into the
    # existing additive ``bonus`` term used by every DP backend.
    #
    # ``0.0`` = off (haplotype-agnostic, the previous behaviour); a large value
    # approaches haplotype-restricted chaining (switches effectively forbidden).
    # Only active when the reference was built with haplotype paths; on a linear
    # reference or a graph without paths it is a no-op.
    recombination_penalty: float = 0.0

    # AGNES-style adaptive (confidence-gated) seed scoring
    # (Arafat et al., 2025, "AGNES", Algorithm 1). When the Stage 4 seed head has
    # written per-anchor probabilities into ``AnchorSet.score``, the DP trusts
    # those scores only when the read's score distribution is *decisively*
    # separated; otherwise it falls back to pure length-based (geometric) weights.
    # This keeps an under-confident classifier from corrupting the chain, which is
    # exactly the regime where a fixed neural blend hurts recall in repeats and
    # low-complexity regions.
    adaptive_seed_scoring: bool = True
    # τ — minimum score separation ``(μ_high - μ_low) / σ`` required to let the
    # seed scores steer the DP. Below it, the read is chained with uniform node
    # scores (classical DP), matching AGNES's confidence-based method selection.
    confidence_threshold: float = 0.7
    high_confidence_prob: float = 0.7  # p above this = a confident true seed
    low_confidence_prob: float = 0.3  # p below this = a confident spurious seed
    # Degenerate-anchor guards: with too few anchors the confidence metric is
    # meaningless, and with too many the guidance is both unreliable and costly,
    # so both extremes fall back to pure DP (AGNES lines 3-5: |V|<5 or |V|>1000).
    min_confidence_anchors: int = 5
    max_confidence_anchors: int = 1000
    # Logit transform of the seed probability, ``log(p/(1-p))``, mapped to a
    # positive, length-preserving multiplicative gate ``1 + gain * logit`` and
    # clamped. gain scales how hard a confident seed is up-/down-weighted.
    logit_gate_gain: float = 0.25
    logit_gate_min: float = 0.1
    logit_gate_max: float = 3.0

    # Learned seed-graph transition guidance. Applied only when the same AGNES
    # confidence decision that gates node scores trusts the GNN for this read.
    # Probabilities are converted to centered logits and clipped before entering
    # the DP as an additive edge term.
    gnn_transition_bonus: float = 2.0
    gnn_transition_logit_clip: float = 3.0

    # Chain selection. A single SMEM can span an entire read, so chains are
    # filtered on score rather than anchor count; raise ``min_chain_anchors``
    # only when seeding with a fixed-k index that cannot produce long anchors.
    min_chain_score: float = 20.0
    min_chain_anchors: int = 1
    max_chains: int = 8  # candidate chains kept per read for Stage 3 / re-ranking
    # A secondary chain is dropped when its read span overlaps the primary's by
    # more than this fraction.
    secondary_overlap: float = 0.5
    # Drop chains scoring below this fraction of the best chain's score.
    secondary_score_ratio: float = 0.6


@dataclass
class ExtensionConfig:
    """Stage 3 — Extension (banded affine Smith-Waterman / WFA / cudaaligner).

    Defaults follow the architecture's WFA settings (``mismatch=4``,
    ``gap_open=6``, ``x_drop=600``). ``algorithm`` selects the DP kernel:
      - ``"banded_sw"``: banded affine Smith-Waterman with traceback (default).
      - ``"wfa"``: wavefront alignment, O(n·s) in the edit distance ``s``.
      - ``"cudaaligner"``: GenomeWorks-style global affine (Gotoh) alignment of
        the chain window, emitting a full CIGAR
        (:func:`graphmambaformer.accel.genomeworks_ops.global_align`).
    """

    algorithm: str = "banded_sw"  # "banded_sw" | "wfa" | "cudaaligner"

    match_score: float = 2.0
    mismatch_penalty: float = 4.0
    gap_open: float = 6.0
    gap_extend: float = 2.0
    x_drop: float = 600.0

    # Band half-width. The chain's diagonal spread is added on top, clipped to
    # ``max_half_band``, so a chain with large indels widens its own band.
    half_band: int = 64
    max_half_band: int = 512

    # Flank beyond the chain's first/last anchor to include in the DP window.
    flank: int = 100
    # Cap the DP window so a pathological chain cannot blow up memory.
    max_window: int = 32_768
    # WFA only: abandon a wavefront past this edit distance.
    wfa_max_distance: int = 4_096

    # ---- GenomeWorks cudaextender: ungapped X-drop seed prefilter ---------- #
    #: Run a fast ungapped X-drop extension of each chain's seed *before* the
    #: gapped DP. Chains whose ungapped score cannot clear ``ungapped_min_score``
    #: are dropped cheaply, so the expensive banded DP only runs on candidates
    #: that already show a strong gap-free core (BLAST/cudaextender two-hit
    #: philosophy). Off by default so the classical result is unchanged unless
    #: explicitly enabled.
    ungapped_prefilter: bool = False
    #: X-drop threshold for the ungapped extension (kept distinct from the DP
    #: ``x_drop`` so the prefilter can be tuned independently).
    ungapped_x_drop: float = 40.0
    #: Minimum ungapped-extension score for a chain to survive the prefilter.
    ungapped_min_score: float = 20.0


@dataclass
class ScoringConfig:
    """Stage 4 — Neural scoring.

    The backbone's read/graph representations are used three ways: to prune
    anchors before chaining, to re-rank the DP chains, and to produce a
    calibrated MAPQ. ``position_rescue`` lets the MappingHead propose a locus for
    reads that Stages 1-3 failed to place at all.
    """

    score_seeds: bool = True
    rerank_chains: bool = True
    neural_mapq: bool = True
    position_rescue: bool = True

    # Blend of DP score and neural chain score used for the final ranking.
    dp_weight: float = 0.5
    neural_weight: float = 0.5

    # Never prune below this many anchors, however low the seed scores are.
    min_anchors_kept: int = 8
    # MAPQ calibration.
    max_mapq: int = 60
    mapq_floor: int = 0


PIPELINE_MODES: tuple[str, ...] = ("hybrid", "fast", "two_pass")


@dataclass
class PipelineConfig:
    """Alignment-pipeline configuration.

    ``mode`` selects the pipeline implementation:
      - ``"hybrid"`` (default): :class:`HybridAlignmentPipeline`, the accuracy
        path — seed -> chain -> extend -> score. Resource-driven Stages 5-7 are
        composed around it by :class:`alignment.SevenStagePipeline`.
      - ``"fast"``: :class:`FastAlignmentPipeline`, the throughput path — the
        classical stages only, with MAPQ from the primary/secondary score
        margin and no neural forward pass at all. (The architecture's fast mode
        also keeps a batched model pass over precomputed graph embeddings;
        here the neural work is simply skipped.)
      - ``"two_pass"``: :class:`TwoPassAligner`, the fast path for easy reads
        with a hybrid rescue for the hard tail. (The architecture's Pass 1 is a
        runtime-compiled C extension; this one is the array-programmed Python
        path, so the reads/sec figures in the spec do not apply.)
    """

    mode: str = "hybrid"

    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    chaining: ChainingConfig = field(default_factory=ChainingConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    accel: AccelConfig = field(default_factory=AccelConfig)

    # Stage toggles (Stage 3 is skippable when only a locus is needed).
    run_extension: bool = True
    run_neural_scoring: bool = True

    # Two-pass: a read is "easy" (and skips the neural pass) when its best chain
    # covers at least this fraction of the read and its margin over the runner-up
    # is at least this large.
    easy_coverage: float = 0.80
    easy_margin: float = 0.25

    batch_size: int = 16
    max_read_len: int | None = None  # truncate reads before encoding

    def __post_init__(self) -> None:
        if self.mode not in PIPELINE_MODES:
            raise ValueError(
                f"Unknown pipeline mode {self.mode!r}. Known: {list(PIPELINE_MODES)}"
            )
        for m in self.seeding.modes:
            if m not in SEEDING_MODES:
                raise ValueError(
                    f"Unknown seeding mode {m!r}. Known: {list(SEEDING_MODES)}"
                )


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    """Alignment + multi-task loss weighting.

    With ``learnable_weights=True`` the per-task weights follow Kendall et al.
    (arXiv:1705.07115): each task carries a learned log-variance ``s_i`` and
    contributes ``exp(-s_i) * L_i + s_i``, so the network balances the tasks
    itself. The static weights below are the initial scale of each term.
    """

    learnable_weights: bool = True

    # Stage losses.
    w_seed: float = 1.0  # per-anchor true/false BCE
    w_transition: float = 0.5  # seed-graph edge BCE (both endpoints are true)
    w_chain: float = 1.0  # listwise chain-ranking cross-entropy
    w_node: float = 1.0  # MappingHead node classification
    w_position: float = 1.0  # within-node position regression
    w_mapq: float = 0.5  # MAPQ regression
    w_router: float = 0.05  # compute-cost regularizer
    w_extension: float = 0.5  # alignment-score margin

    # Multi-task head losses (only applied for enabled heads).
    w_multitask: float = 1.0

    # Class imbalance: the synthetic data has ~20-30% false seeds, so positives
    # dominate; this scales the positive term in the seed BCE.
    seed_pos_weight: float = 1.0
    # Hard-read up-weighting. The neural stage is trained to *augment* the
    # classical heuristics on every read, but its highest value is on the reads
    # the heuristics struggle with (the chainer places them nowhere, off the true
    # locus, or with a single low-coverage candidate). ``TargetBuilder`` tags each
    # read with a difficulty in ``{0.0, 0.5, 1.0}`` (``targets["read_difficulty"]``)
    # and the per-read loss terms (seed / transition / chain / node / position /
    # mapq) are scaled by ``1 + (hard_read_weight - 1) * difficulty`` — a *weighted
    # average*, so hard reads pull more gradient without inflating the loss scale
    # (keeping it stable under the Kendall weighting). ``1.0`` disables the
    # emphasis and recovers the uniform per-read objective exactly.
    hard_read_weight: float = 3.0
    # Label smoothing for the node classifier (large, noisy label space).
    node_label_smoothing: float = 0.05
    # Huber transition point for the position / MAPQ regressions.
    huber_beta: float = 0.1
    # Target compute cost for the router (fraction of the full path).
    router_target_cost: float = 0.6
    # Router objective is a Switch-Transformer-style load-balancing loss, not a
    # one-sided cost penalty: penalizing only "too expensive" made routing every
    # read to the cheapest path the global optimum, so the router collapsed to
    # 100% "fast" and medium/full were never used. ``router_balance_coef`` weights
    # the load-balance term (push usage toward all routes), ``router_entropy_coef``
    # a per-read entropy bonus that encourages early exploration (anneal toward 0
    # after ~1 epoch if desired), and ``router_cost_coef`` a gentle *two-sided*
    # nudge of the expected cost toward ``router_target_cost``. Set all three to 0
    # to recover the legacy one-sided cost penalty.
    router_balance_coef: float = 1.0
    router_entropy_coef: float = 0.01
    router_cost_coef: float = 0.02


# --------------------------------------------------------------------------- #
# Core-architecture registry
# --------------------------------------------------------------------------- #
#: Selectable core architectures. ``"graphmamba"`` is the default and the one
#: exercised first; the MambaFormer / hybrid backbones remain available for
#: ablations against the earlier Figure-1 assembly.
CORE_ARCHITECTURES: tuple[str, ...] = (
    "graphmamba",
    "multitask_graphmamba",
    "mambaformer",
    "hybrid",
)


@dataclass
class CoreModelConfig:
    """Which core architecture to build, plus each variant's config.

    ``arch`` picks the implementation:
      - ``"graphmamba"`` (default): :class:`GraphMambaModel`.
      - ``"multitask_graphmamba"``: the same backbone plus the multi-task heads.
      - ``"mambaformer"`` / ``"hybrid"``: :class:`GraphMambaFormerEncoder` with
        the corresponding backbone.
    """

    arch: str = "graphmamba"
    graphmamba: GraphMambaConfig = field(default_factory=GraphMambaConfig)
    encoder: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.arch not in CORE_ARCHITECTURES:
            raise ValueError(
                f"Unknown core architecture {self.arch!r}. "
                f"Known: {list(CORE_ARCHITECTURES)}"
            )
        if self.arch == "mambaformer":
            self.encoder.backbone = "mambaformer"
        elif self.arch == "hybrid":
            self.encoder.backbone = "hybrid"
