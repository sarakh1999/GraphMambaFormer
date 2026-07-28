"""Stacked hybrid block (Figure 1B), repeated ×N (default N = 12).

Each block contains, in order:

    Layer 1  Bidirectional Mamba-2      (sequence mixing; replaces seed chaining)
    Layer 2  Windowed multi-head attn   (context-dependent sub/indel scoring)
    Layer 3  GATv2 graph attention      (pangenome path / splice / CpG reasoning)
    FFN      SwiGLU feed-forward

Layers 1, 2 and the FFN operate on the **read** representation ``x`` and are
each wrapped in a pre-norm residual (``x + sublayer(RMSNorm(x))``). Layer 3
(GATv2) operates on the **reference-graph** node embeddings carried in the
``graph`` argument: it refines them in place with its own pre-norm residual, so
after N blocks the graph has undergone N rounds of edge-type-aware message
passing. (Reads and the refined graph are fused later by the cross-attention
alignment decoder — Figure 1C.)

The block keeps its public interface — ``forward(x, mask=None, graph=None) ->
x`` — so the top-level encoder loop is unchanged. ``use_attention`` / ``use_gat``
turn Layers 2 / 3 on; built-in factories supply the modules, or a custom
``attention_factory`` / ``gat_factory`` can be passed.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..config import AttentionConfig, BlockConfig
from ..layers.attention import MultiHeadSelfAttention
from ..layers.bimamba import BiMamba2
from ..layers.common import FeedForward, RMSNorm, Residual
from ..layers.gat import GATv2Layer

SubLayerFactory = Callable[[BlockConfig], nn.Module]


def default_attention_factory(cfg: BlockConfig) -> nn.Module:
    """Windowed multi-head self-attention (Figure 1B, Layer 2)."""
    acfg = AttentionConfig(
        d_model=cfg.d_model,
        n_heads=cfg.attention.n_heads,
        d_head=cfg.attention.d_head,
        dropout=cfg.attention.dropout,
        causal=cfg.attention.causal,
        window=cfg.window,
    )
    return MultiHeadSelfAttention(acfg)


def default_gat_factory(cfg: BlockConfig) -> nn.Module:
    """GATv2 graph-attention layer (Figure 1B, Layer 3)."""
    return GATv2Layer(cfg.gat)


class GraphMambaFormerBlock(nn.Module):
    def __init__(
        self,
        cfg: BlockConfig,
        attention_factory: SubLayerFactory | None = None,
        gat_factory: SubLayerFactory | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.sublayers = nn.ModuleList()      # read-side residual sub-layers
        self.sublayer_names: list[str] = []

        # Layer 1: bidirectional Mamba-2 (read sequence mixing).
        if cfg.use_mamba:
            self._add("mamba", BiMamba2(cfg.mamba))

        # Layer 2: windowed multi-head self-attention (read sequence).
        if cfg.use_attention:
            factory = attention_factory or default_attention_factory
            self._add("attention", factory(cfg))

        # Layer 3: GATv2 graph attention (refines the reference-graph nodes).
        self.gat: nn.Module | None = None
        if cfg.use_gat:
            factory = gat_factory or default_gat_factory
            self.gat = factory(cfg)
            self.gat_norm = RMSNorm(cfg.d_model)

        # FFN row.
        if cfg.use_ffn:
            self._add("ffn", FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout))

    def _add(self, name: str, module: nn.Module) -> None:
        self.sublayers.append(Residual(self.cfg.d_model, module))
        self.sublayer_names.append(name)

    @property
    def layer_names(self) -> list[str]:
        """Full ordered layer list including the graph-side GATv2, for reporting."""
        names = list(self.sublayer_names)
        if self.gat is not None:
            insert = names.index("ffn") if "ffn" in names else len(names)
            names = names[:insert] + ["gat"] + names[insert:]
        return names

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        graph: object | None = None,
    ) -> torch.Tensor:
        # Layer 3: refine graph node embeddings (edge-type-aware message passing).
        if self.gat is not None and graph is not None:
            ne = graph.node_embeddings
            edge_attr = getattr(graph, "edge_type_embeddings", None)
            graph.node_embeddings = ne + self.gat(
                self.gat_norm(ne), graph.edge_index, edge_attr
            )

        # Layers 1, 2, FFN: read-side residual sub-layers.
        for layer in self.sublayers:
            x = layer(x, mask=mask, graph=graph)
        return x
