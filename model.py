"""Top-level GraphMambaFormer encoder assembly.

Wires the three modules built so far:
  1. Modality-Aware Read Encoder (Figure 1A, left)
  2. Reference Graph Encoder     (Figure 1A, right)
  3. A stack of N hybrid blocks  (Figure 1B) — currently bidirectional Mamba-2.

The cross-attention alignment decoder and output heads (Figure 1C) plus the
windowed-attention / GATv2 sub-layers (Figure 1B, Layers 2-3) are deliberately
left out for now. The block stack already forwards ``mask`` and ``graph`` so
those layers slot in without touching this file's interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks.hybrid_block import GraphMambaFormerBlock, SubLayerFactory
from .blocks.mambaformer import MambaFormer
from .config import ModelConfig
from .encoders.graph_encoder import GraphEncoding, ReferenceGraphEncoder
from .encoders.read_encoder import ModalityAwareReadEncoder


class GraphMambaFormerEncoder(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig | None = None,
        attention_factory: SubLayerFactory | None = None,
        gat_factory: SubLayerFactory | None = None,
    ):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.read_encoder = ModalityAwareReadEncoder(self.cfg.read_encoder)
        self.graph_encoder = ReferenceGraphEncoder(self.cfg.graph_encoder)

        # Sequence backbone: the MambaFormer (leading Mamba + [Attn, Mamba] x L)
        # or the extensible hybrid-block stack.
        if self.cfg.backbone == "mambaformer":
            self.backbone = MambaFormer(self.cfg.mambaformer)
            self.blocks = None
        elif self.cfg.backbone == "hybrid":
            self.backbone = None
            self.blocks = nn.ModuleList(
                GraphMambaFormerBlock(
                    self.cfg.block,
                    attention_factory=attention_factory,
                    gat_factory=gat_factory,
                )
                for _ in range(self.cfg.n_blocks)
            )
        else:
            raise ValueError(f"Unknown backbone: {self.cfg.backbone!r}")

        self.final_norm = nn.LayerNorm(self.cfg.d_model)

    def encode_graph(self, **graph_kwargs) -> GraphEncoding:
        return self.graph_encoder(**graph_kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,
        modality: str | int | list,
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        graph: GraphEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of reads through the block stack.

        Returns ``(hidden, mask)``. ``graph`` is accepted and threaded through
        the blocks so future graph-attention layers can consume it; the current
        Mamba/FFN layers ignore it.
        """
        hidden, mask = self.read_encoder(
            token_ids=token_ids, modality=modality, qualities=qualities, mask=mask
        )
        if self.backbone is not None:
            hidden = self.backbone(hidden, mask=mask)
        else:
            for block in self.blocks:
                hidden = block(hidden, mask=mask, graph=graph)
        return self.final_norm(hidden), mask

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
