"""Training objectives for the alignment stages and the multi-task heads.

Two pieces live here:

:class:`AlignmentLoss`
    The supervision for Stages 1-4 — anchor classification, chain ranking, node
    classification, within-node position, MAPQ, the router's compute budget, and
    an alignment-score margin.

:class:`MultiTaskLoss`
    The ten predictive-genomics heads, each with the loss its label space calls
    for (classification, ordinal regression, or per-base sequence labelling).

Both are combined by :class:`GraphMambaLoss`. Every term is masked, and a term
whose labels are absent from the batch is skipped rather than contributing zero,
so partially-labelled batches train the heads they have labels for without
diluting the gradient of the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..config import LossConfig, MultiTaskConfig

__all__ = [
    "AlignmentLoss",
    "MultiTaskLoss",
    "GraphMambaLoss",
    "LossOutput",
]


@dataclass
class LossOutput:
    """Total loss plus the detached per-term breakdown for logging."""

    total: torch.Tensor
    terms: dict[str, float] = field(default_factory=dict)
    #: Effective weight applied to each term (after learnable balancing).
    weights: dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.total.detach())

    def as_dict(self) -> dict[str, float]:
        return {"total": float(self.total.detach()), **self.terms}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of ``values`` over ``mask``, or an exact zero when nothing is valid.

    Returning a zero that is still attached to the graph keeps the term present
    with no gradient, which avoids a shape/device dance at the call sites.
    """
    if mask is None:
        return values.mean() if values.numel() else values.sum()
    mask = mask.to(values.dtype)
    total = mask.sum()
    if float(total) == 0.0:
        return (values * 0.0).sum()
    return (values * mask).sum() / total.clamp_min(1.0)


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    read_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Masked mean of ``values`` with an optional per-read weight.

    ``read_weight`` is a ``(B,)`` non-negative multiplier broadcast over every
    non-batch axis of ``values``, so a hard read's anchors / edges / offset all
    count proportionally more. The result is a *weighted average* (normalized by
    the summed weight), which re-balances the gradient toward the up-weighted
    reads without changing the term's magnitude — so it stays comparable across
    batches and stable under the Kendall weighting. With ``read_weight=None`` this
    is exactly :func:`_masked_mean`, so the uniform objective is recovered
    byte-for-byte.
    """
    if read_weight is None:
        return _masked_mean(values, mask)
    w = read_weight.to(values.dtype)
    while w.dim() < values.dim():
        w = w.unsqueeze(-1)
    w = w.expand_as(values)
    if mask is not None:
        w = w * mask.to(values.dtype)
    total = w.sum()
    if float(total) == 0.0:
        return (values * 0.0).sum()
    return (values * w).sum() / total.clamp_min(1.0)


class KendallWeighting(nn.Module):
    """Uncertainty weighting from Kendall et al. (arXiv:1705.07115).

    Each task carries a learned log-variance ``s`` and contributes
    ``exp(-s) * L + s``. The trailing ``+ s`` is what stops the trivial solution
    of driving every weight to zero. Log-variances are created lazily, on first
    sight of a task name, so enabling a head does not need a config change here.
    """

    def __init__(self, initial: dict[str, float] | None = None, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.log_vars = nn.ParameterDict()
        self._static: dict[str, float] = dict(initial or {})
        # Materialize the known task log-variances up front (on CPU; they move
        # with the module's later ``.to(device)``). Creating them lazily on the
        # first ``combine()`` call meant they came into existence *after* the
        # trainer had already built its optimizer from ``criterion.parameters()``,
        # so they were never optimized and the "learned" weights stayed frozen at
        # their init. Eager creation lets the optimizer capture them. Task names
        # not known at construction (e.g. dynamically enabled multi-task heads)
        # are still created lazily in ``combine`` and share that caveat.
        if self.enabled:
            for name in self._static:
                self._ensure(name, torch.device("cpu"))

    def _ensure(self, name: str, device: torch.device) -> None:
        if name in self.log_vars:
            return
        # Start at the log-variance whose implied weight equals the static one:
        # exp(-s) = w  =>  s = -log(w).
        weight = max(self._static.get(name, 1.0), 1e-6)
        start = -torch.log(torch.tensor(weight, device=device))
        self.log_vars[name] = nn.Parameter(start)

    def combine(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Reduce per-task losses to a scalar, returning the applied weights."""
        if not losses:
            raise ValueError("no loss terms to combine")

        device = next(iter(losses.values())).device
        total = torch.zeros((), device=device)
        applied: dict[str, float] = {}
        for name, value in losses.items():
            if self.enabled:
                self._ensure(name, device)
                log_var = self.log_vars[name]
                total = total + torch.exp(-log_var) * value + log_var
                applied[name] = float(torch.exp(-log_var).detach())
            else:
                weight = self._static.get(name, 1.0)
                total = total + weight * value
                applied[name] = weight
        return total, applied


class AlignmentLoss(nn.Module):
    """Supervision for the alignment stages.

    Expected ``targets`` keys (all optional — a missing key skips its term):

    ``seed_labels`` (B, A)
        1.0 for a true anchor, 0.0 for a decoy. Paired with ``anchor_mask``.
    ``chain_target`` (B,)
        Index of the correct chain, or -1 when no candidate is correct. Scored
        listwise over the candidates, which is what the ranking at inference
        actually needs — the absolute scores do not matter, only the ordering.
    ``node_target`` (B,)
        Correct graph node, ``-100`` to ignore.
    ``position_target`` (B,)
        True offset within the node, as a fraction in ``[0, 1]``.
    ``mapq_target`` (B,)
        Target MAPQ in ``[0, max_mapq]``; normalized internally.
    ``best_score`` / ``decoy_score`` (B,)
        Alignment scores for the margin term.
    """

    def __init__(self, cfg: LossConfig | None = None, max_mapq: int = 60):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.max_mapq = float(max_mapq)

    # -- individual terms ---------------------------------------------------- #
    def seed_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        anchor_mask: torch.Tensor | None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-anchor true/decoy BCE, masked over the padded anchor slots."""
        pos_weight = torch.as_tensor(
            self.cfg.seed_pos_weight, device=logits.device, dtype=logits.dtype
        )
        per_anchor = F.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), pos_weight=pos_weight, reduction="none"
        )
        return _weighted_masked_mean(per_anchor, anchor_mask, read_weight)

    def transition_loss(
        self,
        logits: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        seed_labels: torch.Tensor,
        gnn_active: torch.Tensor | None = None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE for chaining edges; a positive edge joins two true seeds."""
        labels = seed_labels.to(logits.device)
        src = edge_index[..., 0].clamp(0, max(labels.shape[1] - 1, 0))
        dst = edge_index[..., 1].clamp(0, max(labels.shape[1] - 1, 0))
        src_label = labels.gather(1, src)
        dst_label = labels.gather(1, dst)
        targets = (src_label.bool() & dst_label.bool()).to(logits.dtype)
        live = edge_mask.bool()
        if gnn_active is not None:
            live = live & gnn_active.bool()[:, None]
        per_edge = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return _weighted_masked_mean(per_edge, live, read_weight)

    def chain_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        chain_mask: torch.Tensor | None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Listwise cross-entropy over each read's candidate chains.

        Padded candidates are pushed to ``-inf`` so they cannot absorb
        probability mass, and reads with no correct candidate are dropped.
        """
        # The degenerate cases below must return a zero that is still attached to
        # the graph. -inf * 0 is NaN, and ChainScoringHead already fills padded
        # candidates with -inf, so the infinities are cleared before scaling.
        zero = logits.nan_to_num(neginf=0.0, posinf=0.0).sum() * 0.0

        masked = (
            logits.masked_fill(~chain_mask.bool(), float("-inf"))
            if chain_mask is not None
            else logits
        )

        valid = target >= 0
        if not bool(valid.any()):
            return zero

        # A row whose every candidate is masked is an all -inf row, whose
        # log-softmax is NaN; requiring the target slot to be live drops those.
        rows = torch.nonzero(valid, as_tuple=True)[0]
        finite = torch.isfinite(masked[rows, target[rows]])
        if not bool(finite.any()):
            return zero
        rows = rows[finite]
        per_read = F.cross_entropy(masked[rows], target[rows], reduction="none")
        if read_weight is None:
            return per_read.mean()
        w = read_weight.to(per_read.dtype)[rows]
        denom = w.sum()
        if float(denom) == 0.0:
            return per_read.mean()
        return (per_read * w).sum() / denom.clamp_min(1.0)

    def node_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if read_weight is None:
            return F.cross_entropy(
                logits,
                target,
                ignore_index=-100,
                label_smoothing=self.cfg.node_label_smoothing,
            )
        # Per-read cross-entropy so hard reads can be up-weighted; ignored rows
        # (target == -100) already contribute exactly zero, and the valid mask
        # keeps the weighted-average normalization over the supervised rows only.
        per_read = F.cross_entropy(
            logits,
            target,
            ignore_index=-100,
            label_smoothing=self.cfg.node_label_smoothing,
            reduction="none",
        )
        valid = (target != -100).to(per_read.dtype)
        return _weighted_masked_mean(per_read, valid, read_weight)

    def position_loss(
        self,
        fraction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Huber on the within-node offset; robust to the odd mis-assigned node."""
        per_read = F.smooth_l1_loss(
            fraction, target.to(fraction.dtype), beta=self.cfg.huber_beta, reduction="none"
        )
        return _weighted_masked_mean(per_read, valid, read_weight)

    def mapq_loss(
        self,
        mapq: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Huber on MAPQ, normalized to ``[0, 1]`` to keep the scale comparable.

        Legacy path: regresses the head's MAPQ toward a *target* MAPQ (e.g. the
        baseline's). Kept for back-compat / ablation; the default objective is now
        :meth:`mapq_calibration_loss`, which calibrates instead of copies.
        """
        scale = max(self.max_mapq, 1.0)
        per_read = F.smooth_l1_loss(
            mapq / scale,
            target.to(mapq.dtype) / scale,
            beta=self.cfg.huber_beta,
            reduction="none",
        )
        return _weighted_masked_mean(per_read, valid, read_weight)

    def mapq_calibration_loss(
        self,
        logit: torch.Tensor,
        correct: torch.Tensor,
        valid: torch.Tensor | None = None,
        read_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calibration BCE: train ``p = sigmoid(logit)`` toward ``P(correct)``.

        The mapping head reports ``MAPQ = -10 log10(1 - p)``. Training ``p``
        against the binary "is this placement right?" label (start within
        ``LOCUS_TOLERANCE`` of truth) with cross-entropy makes ``p`` a calibrated
        probability -- and therefore a calibrated MAPQ -- which is what lets the
        aligner *win* on MAPQ calibration instead of merely reproducing the
        baseline's MAPQ. Masked to reads that carry a correctness label and
        re-weighted per read exactly like every other term.
        """
        per_read = F.binary_cross_entropy_with_logits(
            logit, correct.to(logit.dtype), reduction="none"
        )
        return _weighted_masked_mean(per_read, valid, read_weight)

    def router_loss(self, router: dict[str, torch.Tensor]) -> torch.Tensor:
        """Load-balancing router objective that prevents collapse to one route.

        The previous term penalized only expected compute *over* budget, so
        sending every read to the cheapest path was the global optimum and the
        router collapsed to 100% "fast". This combines three signals computed on
        the current batch's router output:

        * **load balance** (Switch-Transformer aux loss): ``R * sum_r f_r * P_r``
          where ``f_r`` is the fraction of reads hard-routed to ``r`` and ``P_r``
          the mean soft probability of ``r``; minimized when both are uniform,
          i.e. all routes get used.
        * **entropy bonus** (optional): rewards higher per-read routing entropy so
          early training explores instead of latching onto one route.
        * **cost nudge**: a gentle *two-sided* pull of the expected compute cost
          toward ``router_target_cost`` (replaces the old one-sided penalty).

        All three coefficients live on :class:`LossConfig`; set them to 0 to
        recover the legacy one-sided cost penalty.
        """
        probs = router["probs"]
        if probs.dim() != 2:
            probs = probs.reshape(-1, probs.shape[-1])
        n_routes = probs.shape[-1]

        importance = probs.mean(dim=0)  # P_r: mean soft prob per route
        weights = router.get("weights")
        if weights is not None:
            load = weights.to(probs.dtype)
            if load.dim() != 2:
                load = load.reshape(-1, load.shape[-1])
            load = load.mean(dim=0)  # f_r: fraction hard-routed per route
        else:
            load = F.one_hot(probs.argmax(dim=-1), n_routes).to(probs.dtype).mean(dim=0)

        loss = self.cfg.router_balance_coef * n_routes * torch.sum(importance * load)

        if self.cfg.router_entropy_coef:
            per_read_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            loss = loss - self.cfg.router_entropy_coef * per_read_entropy  # maximize

        if self.cfg.router_cost_coef and "cost" in router:
            cost_pen = (router["cost"].mean() - self.cfg.router_target_cost).pow(2)
            loss = loss + self.cfg.router_cost_coef * cost_pen

        return loss

    def extension_loss(
        self, best: torch.Tensor, decoy: torch.Tensor, margin: float = 1.0
    ) -> torch.Tensor:
        """Hinge pushing the true alignment's score above the best decoy's."""
        return F.relu(margin - (best - decoy)).mean()

    # -- assembly ------------------------------------------------------------ #
    def _read_weight(self, targets: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Per-read loss multiplier from difficulty and/or modality.

        Two independent, multiplicative signals feed the same ``(B,)`` weight:

        * ``targets["read_difficulty"]`` — ``(B,)`` in ``[0, 1]`` (0 = the
          heuristics placed the read cleanly, 1 = they failed it), turned into
          ``1 + (hard_read_weight - 1) * difficulty`` so easy reads keep weight 1
          and the hardest reach ``hard_read_weight``.
        * ``targets["modality_weight"]`` — ``(B,)`` inverse-frequency multiplier
          the trainer injects when modality re-weighting is on, so a rare
          modality (e.g. ONT/HiFi in an Illumina-dominated batch) counts more per
          read and cannot be drowned out by the abundant modality.

        The two are multiplied when both are present. Returns ``None`` — the
        uniform objective, recovered byte-for-byte — when neither applies (e.g.
        an inference-time call, or ``hard_read_weight == 1`` with no modality
        weight).
        """
        weight: torch.Tensor | None = None

        difficulty = targets.get("read_difficulty")
        hw = float(self.cfg.hard_read_weight)
        if difficulty is not None and hw != 1.0:
            weight = 1.0 + (hw - 1.0) * difficulty.to(torch.float32)

        modality_weight = targets.get("modality_weight")
        if modality_weight is not None:
            mw = modality_weight.to(torch.float32)
            weight = mw if weight is None else weight * mw

        return weight

    def forward(
        self,
        outputs,
        targets: dict[str, torch.Tensor],
        seed_scores: dict[str, torch.Tensor] | None = None,
        chain_scores: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Collect every applicable alignment term, keyed by name."""
        losses: dict[str, torch.Tensor] = {}

        # Per-read hard-read emphasis: reads the classical heuristics struggle
        # with (see ``TargetBuilder``) get a larger weight in every per-read term,
        # so the neural stage spends its capacity where it adds the most over the
        # heuristics. ``None`` when the feature is off or the batch carries no
        # difficulty tag, in which case every term reduces to its uniform form.
        read_weight = self._read_weight(targets)

        if seed_scores is not None and "seed_labels" in targets:
            losses["seed"] = self.seed_loss(
                seed_scores["logits"], targets["seed_labels"], targets.get("anchor_mask"),
                read_weight,
            )
            if {
                "transition_logits",
                "edge_index",
                "edge_mask",
            } <= seed_scores.keys():
                losses["transition"] = self.transition_loss(
                    seed_scores["transition_logits"],
                    seed_scores["edge_index"],
                    seed_scores["edge_mask"],
                    targets["seed_labels"],
                    seed_scores.get("gnn_active"),
                    read_weight,
                )

        if chain_scores is not None and "chain_target" in targets:
            losses["chain"] = self.chain_loss(
                chain_scores["logits"].squeeze(-1)
                if chain_scores["logits"].dim() == 3
                else chain_scores["logits"],
                targets["chain_target"],
                targets.get("chain_mask"),
                read_weight,
            )

        mapping = getattr(outputs, "mapping", None)
        if mapping is not None:
            if "node_target" in targets:
                losses["node"] = self.node_loss(
                    mapping["node_logits"], targets["node_target"], read_weight
                )
            if "position_target" in targets:
                # Only supervise the offset where the node label is known: the
                # fraction is meaningless without the node it is relative to.
                valid = targets.get("position_valid")
                if valid is None and "node_target" in targets:
                    valid = targets["node_target"] >= 0
                losses["position"] = self.position_loss(
                    mapping["position_fraction"], targets["position_target"], valid,
                    read_weight,
                )
            if "mapq_correct" in targets and "mapq_logit" in mapping:
                # Calibrated MAPQ (default): predict P(placement correct).
                losses["mapq"] = self.mapq_calibration_loss(
                    mapping["mapq_logit"], targets["mapq_correct"],
                    targets.get("mapq_valid"), read_weight,
                )
            elif "mapq_target" in targets:
                # Legacy: regress toward a target (baseline) MAPQ.
                losses["mapq"] = self.mapq_loss(
                    mapping["mapq"], targets["mapq_target"], targets.get("mapq_valid"),
                    read_weight,
                )

        router = getattr(outputs, "router", None)
        if router is not None:
            losses["router"] = self.router_loss(router)

        if "best_score" in targets and "decoy_score" in targets:
            losses["extension"] = self.extension_loss(
                targets["best_score"], targets["decoy_score"]
            )

        return losses

    def static_weights(self) -> dict[str, float]:
        return {
            "seed": self.cfg.w_seed,
            "transition": self.cfg.w_transition,
            "chain": self.cfg.w_chain,
            "node": self.cfg.w_node,
            "position": self.cfg.w_position,
            "mapq": self.cfg.w_mapq,
            "router": self.cfg.w_router,
            "extension": self.cfg.w_extension,
        }


@dataclass(frozen=True)
class TaskSpec:
    """How to score one multi-task head.

    Heads are classifiers over their leading ``n_classes`` channels, some with
    extra regression channels trailing behind (genotype quality, SV breakpoint
    offsets, continuous copy number). ``scope`` says which axis the labels live
    on, which decides how the logits are flattened before the cross-entropy.
    """

    #: ``"read"`` -> labels are ``(B,)``; ``"node"`` -> ``(B, N)``; ``"base"`` -> ``(B, L)``.
    scope: str
    #: Attribute on :class:`MultiTaskConfig` holding the number of classes.
    n_classes_field: str
    #: Trailing regression channels, supervised from ``f"{name}_aux"`` when present.
    n_regression: int = 0


#: Mirrors the head widths built by :class:`MultiTaskHeads`.
TASK_SPECS: dict[str, TaskSpec] = {
    "variant_calling": TaskSpec("node", "num_genotypes", n_regression=1),  # + GQ
    "sv_genotyping": TaskSpec("node", "num_sv_types", n_regression=2),  # + breakpoints
    "copy_number": TaskSpec("node", "num_cn_states", n_regression=1),  # + continuous CN
    "ancestry_local": TaskSpec("node", "num_populations"),
    "haplotype": TaskSpec("read", "_two"),  # phase 0 / 1
    "hla_typing": TaskSpec("read", "num_hla_alleles"),
    "ancestry": TaskSpec("read", "num_populations"),
    "somatic": TaskSpec("read", "num_somatic_classes"),
    "pgx": TaskSpec("read", "num_pgx_alleles"),
    "bqsr": TaskSpec("base", "num_quality_bins"),
    "methylation": TaskSpec("base", "_two"),  # unmethylated / methylated
}


class MultiTaskLoss(nn.Module):
    """Losses for the enabled predictive-genomics heads.

    Labels go under the head's own name; the objective is derived from
    :data:`TASK_SPECS` so callers do not choose it. Heads that emit auxiliary
    regression channels pick those up from ``f"{name}_aux"``, and any head
    without labels in the batch is skipped.

    Per-element labels (node and base scope) use ``-100`` to ignore a position,
    which is how a partially-genotyped graph or a soft-clipped read is handled.
    """

    IGNORE = -100

    def __init__(
        self, cfg: MultiTaskConfig | None = None, loss_cfg: LossConfig | None = None
    ):
        super().__init__()
        self.cfg = cfg or MultiTaskConfig()
        self.loss_cfg = loss_cfg or LossConfig()

    def _n_classes(self, spec: TaskSpec) -> int:
        if spec.n_classes_field == "_two":
            return 2
        return int(getattr(self.cfg, spec.n_classes_field))

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for name, pred in predictions.items():
            target = targets.get(name)
            if target is None:
                continue  # head enabled but unlabelled in this batch
            spec = TASK_SPECS.get(name)
            if spec is None:
                raise ValueError(f"no loss spec registered for task head {name!r}")

            n_classes = self._n_classes(spec)
            losses[f"task/{name}"] = self._classification(
                pred[..., :n_classes], target, spec
            )

            aux_target = targets.get(f"{name}_aux")
            if spec.n_regression and aux_target is not None:
                losses[f"task/{name}_aux"] = self._regression(
                    pred[..., n_classes : n_classes + spec.n_regression],
                    aux_target,
                    targets.get(f"{name}_mask"),
                )
        return losses

    def _classification(
        self, logits: torch.Tensor, target: torch.Tensor, spec: TaskSpec
    ) -> torch.Tensor:
        if spec.scope == "read":
            return F.cross_entropy(logits, target.long(), ignore_index=self.IGNORE)

        # Node / base scope: flatten the element axis so one CE covers the batch.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = target.reshape(-1).long()
        if not bool((flat_target != self.IGNORE).any()):
            return flat_logits.sum() * 0.0
        return F.cross_entropy(flat_logits, flat_target, ignore_index=self.IGNORE)

    def _regression(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        target = target.to(pred.dtype)
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(-1)
        per_element = F.smooth_l1_loss(
            pred, target, beta=self.loss_cfg.huber_beta, reduction="none"
        ).mean(-1)
        return _masked_mean(per_element, mask)


class GraphMambaLoss(nn.Module):
    """The full objective: alignment terms plus multi-task terms, balanced.

    With ``LossConfig.learnable_weights`` the balancing is Kendall uncertainty
    weighting; otherwise the static config weights are used directly.
    """

    def __init__(
        self,
        cfg: LossConfig | None = None,
        multi_task: MultiTaskConfig | None = None,
        max_mapq: int = 60,
    ):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.alignment = AlignmentLoss(self.cfg, max_mapq=max_mapq)
        self.multitask = MultiTaskLoss(multi_task, self.cfg)

        static = self.alignment.static_weights()
        self.weighting = KendallWeighting(static, enabled=self.cfg.learnable_weights)
        self._task_weight = self.cfg.w_multitask

    def forward(
        self,
        outputs,
        targets: dict[str, torch.Tensor],
        seed_scores: dict[str, torch.Tensor] | None = None,
        chain_scores: dict[str, torch.Tensor] | None = None,
    ) -> LossOutput:
        losses = self.alignment(
            outputs, targets, seed_scores=seed_scores, chain_scores=chain_scores
        )

        predictions = getattr(outputs, "multitask", None)
        if predictions:
            for name, value in self.multitask(predictions, targets).items():
                losses[name] = self._task_weight * value

        if not losses:
            raise ValueError(
                "no supervised terms found: targets carried none of the expected keys "
                "(seed_labels, chain_target, node_target, position_target, mapq_target, "
                "best_score/decoy_score) and no multi-task labels"
            )

        total, applied = self.weighting.combine(losses)
        return LossOutput(
            total=total,
            terms={k: float(v.detach()) for k, v in losses.items()},
            weights=applied,
        )
