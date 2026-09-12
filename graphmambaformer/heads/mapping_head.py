"""MappingHead — where does this read go, and how confident are we?

Three sibling MLPs over the fused embedding::

    node classifier   Linear(D->D)   -> GELU -> Linear(D->N_nodes) -> softmax
    position          Linear(D->D)   -> GELU -> Linear(D->1)       -> tanh  (signed)
    MAPQ              Linear(D->D/2) -> GELU -> Linear(D/2->1)     -> logit p_correct
                        -> MAPQ = -10 log10(1 - p)  (calibrated confidence)

The node classifier scores against the *actual* graph node embeddings rather than
a fixed output matrix, so the head transfers across graphs with different node
counts — a fixed ``Linear(D, N_nodes)`` would have to be retrained for every
pangenome. A learned fixed projection is kept as a fallback for the case where no
node embeddings are supplied.

Position is a **signed** offset: ``tanh`` gives ``position_fraction`` in
``[-1, 1]``, matching the training target, which is the read's true start as a
signed fraction of a window around the locus the classical chainer localized
(``TargetBuilder._local_position_target``). ``0`` means "starts exactly at that
locus"; the sign says whether the true start is up- or downstream of it. A
plain ``sigmoid`` (``[0, 1]``) could not represent the downstream half of that
range, so the regressor could only ever fit non-negative offsets.
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
            node_lengths: ``(B, N)`` node lengths. Accepted for API compatibility
                (other heads/paths use it); the position offset is now scaled by
                ``cfg.position_window`` rather than the node length.

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

        # Signed within-window offset in [-1, 1] (tanh), matching the local
        # position target the loss supervises against. ``position`` scales it to a
        # base offset in bp for the (rare) position-rescue path, which clips
        # node_start + offset into the reference, so a signed value is safe.
        # Scale by the window (the same scale the target uses), NOT the node
        # length: ``fraction`` is a fraction of the window, so multiplying by node
        # length would put ``position`` in the wrong units.
        fraction = torch.tanh(self.position(pooled)).squeeze(-1)  # (B,)
        position = fraction * float(self.cfg.position_window)

        # MAPQ as a *calibrated* confidence. The head emits a logit for
        # ``p = P(placement correct)``; the reported ``MAPQ = -10 log10(1 - p)``
        # is then calibrated by construction whenever ``p`` is (which the BCE
        # calibration loss trains it to be), clipped to ``[0, max_mapq]``. The
        # old ``sigmoid(...) * max_mapq`` learned an arbitrary monotone map to a
        # copied baseline MAPQ, which could not be calibrated. Both the logit/prob
        # (for the loss) and the derived MAPQ (for the record) are returned.
        mapq_logit = self.mapq(pooled).squeeze(-1)
        mapq_prob = torch.sigmoid(mapq_logit)
        mapq = torch.clamp(
            -10.0 * torch.log10((1.0 - mapq_prob).clamp_min(1e-6)),
            min=0.0,
            max=float(self.cfg.max_mapq),
        )

        return {
            "node_logits": logits,
            "node_probs": torch.nan_to_num(probs),
            "node_id": node_id,
            "position_fraction": fraction,
            "position": position,
            "mapq": mapq,
            "mapq_prob": mapq_prob,
            "mapq_logit": mapq_logit,
        }
