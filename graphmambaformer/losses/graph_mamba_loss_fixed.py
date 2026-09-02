"""Fixed drop-in loss stack — NEW name, originals untouched.

This module subclasses the existing loss classes and applies two of the fixes
diagnosed for the HG005 run, without editing any existing file:

* **Router collapse** -> :class:`FixedAlignmentLoss` replaces the router term
  with the load-balancing objective from :mod:`..training.run_fixes`, so all
  compute routes get used instead of everything collapsing to "fast".
* **Frozen Kendall weights** -> :class:`EagerKendallWeighting` materializes the
  per-task ``log_vars`` in ``__init__`` (the base class creates them lazily on
  the first forward, i.e. *after* the trainer has already built the optimizer
  from ``criterion.parameters()``, so they never get optimized). Creating them
  up front means the unmodified trainer captures and trains them.

The chain fix is already active on disk (``TargetBuilder(decoy_chains=1)``); the
position fix needs a target redefinition + validation and is left opt-in in
``run_fixes.local_position_target``.

Wire it in with ``scripts/train_fixed.py`` (which patches this class into the
trainer namespace) — no original module is changed.
"""
from __future__ import annotations

import torch

from .alignment_loss import AlignmentLoss, GraphMambaLoss, KendallWeighting


class EagerKendallWeighting(KendallWeighting):
    """Kendall uncertainty weighting that creates its log-variances eagerly.

    The base class adds a ``log_var`` parameter the first time a task appears in
    ``combine()``. That first call happens during the first forward pass, which
    is after ``Trainer.__init__`` has already snapshotted
    ``criterion.parameters()`` into the optimizer — so the log-variances are
    never optimized and the "learned" weights stay frozen at their init (== the
    static weights, exactly what plot 02 shows). Materializing them here, at
    construction, fixes that with no trainer change.
    """

    def __init__(self, initial=None, enabled: bool = True):
        super().__init__(initial, enabled=enabled)
        if self.enabled:
            device = torch.device("cpu")  # params move with .to(device) later
            for name in list(self._static.keys()):
                self._ensure(name, device)


class FixedAlignmentLoss(AlignmentLoss):
    """AlignmentLoss whose router term rewards balanced route usage.

    Everything else is inherited unchanged; we only recompute the ``router``
    entry after the base forward, using the full router output (probs + hard
    weights), which the base ``router_loss`` never sees.
    """

    def forward(self, outputs, targets, seed_scores=None, chain_scores=None):
        losses = super().forward(
            outputs, targets, seed_scores=seed_scores, chain_scores=chain_scores
        )
        router = getattr(outputs, "router", None)
        if router is not None and "router" in losses:
            # Lazy import avoids any import-order coupling with the training pkg.
            from ..training.run_fixes import router_load_balance_loss

            losses["router"] = router_load_balance_loss(
                router,
                balance_coef=1.0,
                entropy_coef=0.01,   # anneal toward 0 after ~1 epoch if desired
                cost_coef=0.02,
                target_cost=getattr(self.cfg, "router_target_cost", 0.6),
            )
        return losses


class FixedGraphMambaLoss(GraphMambaLoss):
    """GraphMambaLoss using the fixed alignment loss + eager Kendall weights."""

    def __init__(self, cfg=None, multi_task=None, max_mapq: int = 60):
        super().__init__(cfg=cfg, multi_task=multi_task, max_mapq=max_mapq)
        # Swap in the fixed components (base __init__ already validated cfg).
        self.alignment = FixedAlignmentLoss(self.cfg, max_mapq=max_mapq)
        self.weighting = EagerKendallWeighting(
            self.alignment.static_weights(), enabled=self.cfg.learnable_weights
        )
