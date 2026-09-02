"""Isolated, opt-in fixes for the training pathologies seen in the HG005 run.

Everything here is NEW and standalone. The currently running GPU job imported
the old modules at process start, so nothing in this file can affect it — the
fixes only take effect when a *fresh* process imports and wires them in (see the
INTEGRATION section at the bottom). The helpers are duck-typed and defensive so
they keep working even as the surrounding modules churn.

Diagnosed problems (evidence in plots/02_loss_weights, 03_validation,
04_model_behavior):

1. Router collapse -> 100% "fast", medium/full never used.
   Root cause: ``AlignmentLoss.router_loss`` is a one-sided penalty on expected
   compute cost (``(cost - target).clamp_min(0)**2``). Sending every read to the
   cheapest path minimizes it, and nothing rewards spending more compute, so the
   global optimum IS collapse. Fix: :func:`router_load_balance_loss` adds a
   Switch-Transformer load-balancing term (+ optional entropy) that pushes usage
   toward all routes, with only a gentle two-sided nudge toward the cost budget.

2. Frozen Kendall loss weights -> plot 02 is flat at the static defaults.
   Root cause: ``KendallWeighting`` creates its per-task ``log_vars`` lazily, on
   the first ``combine()`` call (first forward). But the trainer builds the
   optimizer from ``criterion.parameters()`` *before* any forward, when that
   ParameterDict is still empty — so the log-variances never enter the optimizer
   and can never move. Fix: :func:`materialize_loss_weights` force-creates them
   up front so the optimizer captures them.

3. Inert chain loss -> chain term ~0, chain_scorer gradient ~0.
   Root cause: listwise cross-entropy over a single candidate chain is
   identically 0. ALREADY addressed on disk by ``TargetBuilder(decoy_chains=1)``;
   :func:`recommended_decoy_chains` just documents/eases tuning it.

4. Constant position head -> mapping.position_fraction.std ~0.
   Root cause: ``position_target = read.ref_start / len(reference.ref_seq)`` is a
   *global* coordinate along the whole reference. Predicting an absolute genomic
   position from a read embedding is ill-posed, so the Huber regressor minimizes
   by emitting the mean (=> zero spread). Fix: :func:`local_position_target`
   redefines the target as a *local* offset the model can actually see, and
   :func:`position_target_is_degenerate` lets you verify before/after.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Fix 1 — router load balancing
# --------------------------------------------------------------------------- #
def router_load_balance_loss(
    router_out: Mapping[str, torch.Tensor],
    *,
    balance_coef: float = 1.0,
    entropy_coef: float = 0.0,
    cost_coef: float = 0.02,
    target_cost: float = 0.6,
) -> torch.Tensor:
    """Replacement router objective that prevents collapse to one route.

    Combines three signals, all computed on the current batch's router output:

    * **load balance** (Switch-Transformer aux loss): ``R * sum_r f_r * P_r``
      where ``f_r`` is the fraction of reads routed to ``r`` (from the hard
      one-hot ``weights``) and ``P_r`` is the mean soft probability of ``r``.
      It is minimized when both are uniform, i.e. all routes get used.
    * **entropy bonus** (optional): rewards higher per-read routing entropy so
      early training explores instead of latching onto one route. Set
      ``entropy_coef>0`` for the first epoch or two, then anneal to 0.
    * **cost nudge**: a gentle *two-sided* pull of the expected compute cost
      toward ``target_cost`` (replaces the old one-sided penalty). Keep
      ``cost_coef`` small so it shapes, rather than dominates, routing.

    Args:
        router_out: the dict returned by ``ComplexityRouter.forward`` — needs
            ``probs`` ``(B, R)`` and (ideally) ``weights`` ``(B, R)``; ``cost``
            ``(B,)`` is used for the cost nudge when present.

    Returns:
        A scalar tensor attached to the graph. Use it *instead of* the built-in
        ``router`` term (give it the same ``w_router`` slot, or fold it in via
        the integration notes).
    """
    probs = router_out["probs"]
    if probs.dim() != 2:
        probs = probs.reshape(-1, probs.shape[-1])
    n_routes = probs.shape[-1]

    importance = probs.mean(dim=0)  # P_r: mean soft prob per route

    weights = router_out.get("weights")
    if weights is not None:
        load = weights.to(probs.dtype)
        if load.dim() != 2:
            load = load.reshape(-1, load.shape[-1])
        load = load.mean(dim=0)  # f_r: fraction hard-routed per route
    else:
        load = F.one_hot(probs.argmax(dim=-1), n_routes).to(probs.dtype).mean(dim=0)

    balance = n_routes * torch.sum(importance * load)

    loss = balance_coef * balance

    if entropy_coef:
        per_read_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        loss = loss - entropy_coef * per_read_entropy  # maximize entropy

    if cost_coef and "cost" in router_out:
        cost = router_out["cost"]
        cost_pen = (cost.mean() - float(target_cost)).pow(2)
        loss = loss + cost_coef * cost_pen

    return loss


# --------------------------------------------------------------------------- #
# Fix 2 — make the Kendall log-variances trainable
# --------------------------------------------------------------------------- #
def materialize_loss_weights(criterion: torch.nn.Module, device: torch.device) -> list[str]:
    """Eagerly create every Kendall ``log_var`` so the optimizer can capture it.

    Call this once, right after ``criterion.to(device)`` and **before** the
    optimizer is constructed. It walks the static weight names the criterion was
    initialized with and forces the lazily-created parameters into existence, so
    ``criterion.parameters()`` is non-empty when the optimizer reads it.

    Returns the list of names materialized (empty if learnable weighting is off).
    """
    weighting = getattr(criterion, "weighting", None)
    if weighting is None or not getattr(weighting, "enabled", False):
        return []

    static = dict(getattr(weighting, "_static", {}) or {})
    ensure = getattr(weighting, "_ensure", None)
    created: list[str] = []
    if callable(ensure):
        for name in static:
            ensure(name, device)
            created.append(name)
    return created


def loss_has_trainable_weights(criterion: torch.nn.Module) -> bool:
    """True if the criterion currently exposes trainable weighting parameters."""
    weighting = getattr(criterion, "weighting", None)
    if weighting is None:
        return False
    return any(p.requires_grad for p in weighting.parameters())


# --------------------------------------------------------------------------- #
# Fix 3 — chain hard negatives (already implemented on disk; documented here)
# --------------------------------------------------------------------------- #
def recommended_decoy_chains() -> int:
    """The ``TargetBuilder(decoy_chains=...)`` value that de-degenerates the
    chain-ranking loss. ``1`` is enough to guarantee >=2 candidates per read;
    raise to 2-3 for a harder ranking task once it is learning."""
    return 1


# --------------------------------------------------------------------------- #
# Fix 4 — a learnable position target
# --------------------------------------------------------------------------- #
def position_target_is_degenerate(targets: Mapping[str, torch.Tensor],
                                  std_floor: float = 1e-3) -> bool:
    """True when ``position_target`` has (almost) no spread across the batch.

    A near-constant target is unlearnable in the informative sense — the head
    can only match it by predicting the mean, which is exactly the ``std~0``
    behaviour seen in plot 04. Use this as a guard/telemetry around the fix.
    """
    t = targets.get("position_target")
    if t is None or t.numel() < 2:
        return True
    return float(t.float().std()) < std_floor


def local_position_target(
    reads: Sequence,
    chains_per_read: Sequence[Sequence],
    *,
    window: float = 512.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A *local* within-window offset target the model can actually predict.

    Instead of ``read.ref_start / len(whole_reference)`` (a global coordinate),
    express the read's start as a signed fraction of a small ``window`` around
    the placement the classical chainer already found:

        target = clamp( (read.ref_start - chain.ref_start) / window, -1, 1 )

    Because the fused embedding attends to the reference in the neighbourhood the
    chain localized, this residual is in-distribution and learnable, whereas the
    absolute position is not. Reads with no chain are marked invalid so they do
    not pull the head toward a meaningless value.

    Returns ``(target (B,), valid (B,))`` as CPU float/bool tensors, matching the
    ``position_target`` / ``position_valid`` keys the loss expects.
    """
    tgt = torch.zeros(len(reads), dtype=torch.float32)
    valid = torch.zeros(len(reads), dtype=torch.bool)
    w = max(float(window), 1.0)
    for i, (read, chains) in enumerate(zip(reads, chains_per_read)):
        ref_start = getattr(read, "ref_start", None)
        if ref_start is None or not chains:
            continue
        # Prefer the highest-scoring / primary chain as the local frame.
        anchor = min((c.ref_start for c in chains), default=None)
        if anchor is None:
            continue
        offset = (float(ref_start) - float(anchor)) / w
        tgt[i] = max(-1.0, min(1.0, offset))
        valid[i] = True
    return tgt, valid


# --------------------------------------------------------------------------- #
# INTEGRATION (apply at the next restart — DOES NOT touch the running job)
# --------------------------------------------------------------------------- #
#
# All edits below are in modules the fresh process imports at start-up. None of
# them affect the job already running on the GPU.
#
# --- Fix 2 (do this first; it is the smallest and highest-value) ------------
# In graphmambaformer/training/trainer.py, in Trainer.__init__, right after
#     self.criterion = GraphMambaLoss(loss_cfg or LossConfig()).to(self.device)
# add:
#     from .run_fixes import materialize_loss_weights
#     materialize_loss_weights(self.criterion, self.device)
# so the existing line
#     params = list(self.raw_model.parameters()) + list(self.criterion.parameters())
# now actually includes the log-variances. Verify with run_fixes.loss_has_trainable_weights.
#
# --- Fix 1 (router) ---------------------------------------------------------
# In graphmambaformer/losses/alignment_loss.py, AlignmentLoss.forward, replace
#     losses["router"] = self.router_loss(router["cost"])
# with
#     from ..training.run_fixes import router_load_balance_loss
#     losses["router"] = router_load_balance_loss(
#         router, target_cost=self.cfg.router_target_cost,
#         balance_coef=1.0, entropy_coef=0.01, cost_coef=0.02)
# (pass the whole ``router`` dict, not just its cost). Consider bumping
# LossConfig.w_router from 0.05 to ~0.2 so the balance term has teeth, and
# annealing entropy_coef to 0 after ~1 epoch.
#
# --- Fix 3 (chain) ----------------------------------------------------------
# Already active on disk: TargetBuilder(decoy_chains=1). Confirm the builder is
# constructed with it (grep for TargetBuilder(); pass decoy_chains=recommended
# _decoy_chains() if any call site overrides it to 0).
#
# --- Fix 4 (position) -------------------------------------------------------
# In graphmambaformer/training/targets.py, TargetBuilder.build, the chains are
# already available as ``chains_per_read`` (post-decoy). Replace the block that
# sets targets["position_target"]/["position_valid"] with:
#     from .run_fixes import local_position_target
#     pos_t, pos_v = local_position_target(reads, chains_per_read, window=512.0)
#     targets["position_target"] = pos_t
#     targets["position_valid"] = pos_v
# Validate with run_fixes.position_target_is_degenerate(targets) == False on a
# few batches before committing to a long run; tune ``window`` to the typical
# read/window length so the offsets span most of [-1, 1].
