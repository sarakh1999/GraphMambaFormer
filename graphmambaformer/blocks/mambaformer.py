"""MambaFormer backbone (Park et al. 2024, "Can Mamba Learn How to Learn?",
arXiv:2402.04248).

Faithful port of the ``mixed_attn == "mambaformer"`` topology in the reference
repo (krafton-ai/mambaformer-icl, ``MixerModel``). We mirror the reference's
flat layer list and layer indexing directly:

    Inputs
      -> Mamba                       (leading block, layer_idx = -1)
      -> for i in range(n_layer):    (attention if i % 2 == 0 else Mamba)
      -> LayerNorm
      -> Outputs

With ``n_layer = 12`` and ``attention_first = True`` this yields the flattened
chain ``M A M A ... M`` (leading Mamba + 6 attention + 6 Mamba). Setting
``attention_first = False`` flips the parity (even indices become Mamba).

The Mamba mixer is bidirectional (``BiMamba1`` for ``mamba_variant="mamba1"``,
the reference-port Mamba-1; ``BiMamba2`` for ``mamba_variant="mamba2"``), and
attention defaults to bidirectional and is padding-mask aware -- the alignment
adaptations vs. the autoregressive ICL original.

Each sub-layer is wrapped in a pre-norm residual (``x + sublayer(norm(x))``),
which is functionally the reference Mamba ``Block`` (Add -> Norm -> Mixer) with
the residual threaded through the stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MambaFormerConfig
from ..layers.attention import MultiHeadSelfAttention
from ..layers.bimamba import BiMamba1, BiMamba2
from ..layers.common import Residual


class MambaFormer(nn.Module):
    def __init__(self, cfg: MambaFormerConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        self.layer_types: list[str] = []

        def make_mamba() -> nn.Module:
            if cfg.mamba_variant == "mamba1":
                return BiMamba1(cfg.mamba1)
            if cfg.mamba_variant == "mamba2":
                return BiMamba2(cfg.mamba)
            raise ValueError(f"Unknown mamba_variant: {cfg.mamba_variant!r}")

        def add_mamba() -> None:
            layers.append(Residual(cfg.d_model, make_mamba()))
            self.layer_types.append("mamba")

        def add_attention() -> None:
            layers.append(Residual(cfg.d_model, MultiHeadSelfAttention(cfg.attention)))
            self.layer_types.append("attention")

        # Leading Mamba block (stands in for positional embeddings; layer_idx=-1).
        if cfg.leading_mamba:
            add_mamba()

        # Interleaved blocks mirroring MixerModel: even index -> attention
        # (when attention_first), odd index -> Mamba.
        for i in range(cfg.n_layer):
            is_attention = (i % 2 == 0) == cfg.attention_first
            if is_attention:
                add_attention()
            else:
                add_mamba()

        self.layers = nn.ModuleList(layers)
        self.norm_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm_f(x)
