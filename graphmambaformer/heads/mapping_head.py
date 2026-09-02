"""MappingHead — where does this read go, and how confident are we?

Three sibling MLPs over the fused embedding::

    node classifier   Linear(D->D)   -> GELU -> Linear(D->N_nodes) -> softmax
    position          Linear(D->D)   -> GELU -> Linear(D->1)       -> sigmoid * node_len
    MAPQ              Linear(D->D/2) -> GELU -> Linear(D/2->1)     -> sigmoid * 60

The node classifier scores against the *actual* graph node embeddings rather than
a fixed output matrix, so the head transfers across graphs with different node
counts — a fixed ``Linear(D, N_nodes)`` would have to be retrained for every
pangenome. A learned fixed projection is kept as a fallback for the case where no
node embeddings are supplied.

Position is regressed as a fraction of the node length, which keeps the target in
``[0, 1]`` across nodes spanning three orders of magnitude in size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import MappingHeadConfig


class MappingHead(nn.Module):
    """Node classification + within-node position + MAPQ from a pooled embedding."""

    def __init__(self, cfg: MappingHeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Projects the read embedding into the space the node embeddings live in,
        # so node scores are a dot product against real node representations.
        self.node_query = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d)
        )
        self.node_fallback = nn.Linear(d, cfg.max_nodes)

        self.position = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1)
        )
        self.mapq = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1)
        )
        self.scale = 1.0 / math.sqrt(d)

    def forward(
        self,
        pooled: torch.Tensor,
        node_embeddings: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        node_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict the mapping for each element of ``pooled`` ``(B, D)``.

        Args:
            node_embeddings: ``(B, N, D)`` graph nodes to score against. When
                ``None``, the fixed ``max_nodes`` fallback projection is used.
            node_mask: ``(B, N)`` bool, False for padded node slots.
            node_lengths: ``(B, N)`` node lengths, used to turn the predicted
                fraction into a base offset.

        Returns ``node_logits``, ``node_probs``, ``node_id``, ``position_fraction``,
        ``position``, and ``mapq``.
        """
        if node_embeddings is not None:
            query = self.node_query(pooled)  # (B, D)
            logits = torch.einsum("bd,bnd->bn", query, node_embeddings) * self.scale
            if node_mask is not None:
                logits = logits.masked_fill(~node_mask.to(torch.bool), float("-inf"))
                empty = ~node_mask.any(dim=-1, keepdim=True)
                logits = torch.where(empty, torch.zeros_like(logits), logits)
        else:
            logits = self.node_fallback(pooled)

        probs = torch.softmax(logits, dim=-1)
        node_id = logits.argmax(dim=-1)

        fraction = torch.sigmoid(self.position(pooled)).squeeze(-1)  # (B,)
        if node_lengths is not None:
            chosen_len = node_lengths.gather(1, node_id.unsqueeze(1)).squeeze(1)
            position = fraction * chosen_len.to(fraction.dtype)
        else:
            position = fraction

        mapq = torch.sigmoid(self.mapq(pooled)).squeeze(-1) * self.cfg.max_mapq

        return {
            "node_logits": logits,
            "node_probs": torch.nan_to_num(probs),
            "node_id": node_id,
            "position_fraction": fraction,
            "position": position,
            "mapq": mapq,
        }
