"""ComplexityRouter — adaptive compute routing.

Most reads are easy: they map uniquely, sit outside repeats, and need nothing
more than the fast path. Spending the full model on them is waste. A small MLP
scores each read's difficulty from the fused embedding and assigns it to
``fast`` / ``medium`` / ``full``, which saves 30-50% of the FLOPs on a typical
sample.

Routing decisions are discrete, so the router is trained with a
**straight-through Gumbel-softmax**: the forward pass takes a hard one-hot
decision while the backward pass sees the soft probabilities. Without that, the
argmax would have zero gradient and the router could never learn. At inference
the argmax is taken directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RouterConfig


class ComplexityRouter(nn.Module):
    """Score input complexity and pick a compute path."""

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.num_routes),
        )
        self.register_buffer(
            "costs", torch.tensor(cfg.route_costs[: cfg.num_routes], dtype=torch.float32)
        )

    def forward(self, pooled: torch.Tensor, hard: bool | None = None) -> dict[str, torch.Tensor]:
        """Route each element of ``pooled`` ``(B, D)``.

        Returns a dict with:
          - ``logits`` / ``probs``: ``(B, num_routes)``.
          - ``route``: ``(B,)`` chosen route index.
          - ``weights``: ``(B, num_routes)`` straight-through one-hot in training,
            hard one-hot at inference — usable as a differentiable gate.
          - ``cost``: ``(B,)`` expected relative compute cost, for the loss's
            compute regularizer.
        """
        logits = self.mlp(pooled)
        probs = torch.softmax(logits, dim=-1)

        use_hard = (not self.training) if hard is None else hard
        if self.training and not use_hard:
            weights = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=True)
        else:
            weights = F.one_hot(logits.argmax(-1), self.cfg.num_routes).to(logits.dtype)

        return {
            "logits": logits,
            "probs": probs,
            "route": weights.argmax(-1),
            "weights": weights,
            # Expected cost uses the soft probabilities so the gradient reaches
            # the MLP even when the forward decision is hard.
            "cost": (probs * self.costs).sum(-1),
        }

    def route_names(self, route: torch.Tensor) -> list[str]:
        """Human-readable route names for a batch of route indices."""
        names = self.cfg.route_names
        return [names[int(i)] if int(i) < len(names) else "full" for i in route]
