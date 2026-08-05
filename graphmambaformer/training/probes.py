"""Instrumentation that reports what the model is doing at every step.

A falling loss curve is weak evidence on its own -- it is equally consistent
with a model that has collapsed to predicting the majority class. These probes
record the things that distinguish learning from collapse:

  * activation statistics per tower (an all-zero or exploding tower is visible
    here long before the loss reflects it);
  * gradient norms per parameter group, including the fraction that are exactly
    zero, which is how a disconnected head shows up;
  * the router's decision distribution, since a router that sends every read
    down one path has stopped routing;
  * head output spread, which separates real discrimination from a constant
    prediction that happens to sit near the mean.

Everything is captured under ``torch.no_grad`` into plain floats, so a probe
cannot alter training or hold a reference to the graph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch

__all__ = ["StepReport", "BehaviorProbe"]


def _stats(t: torch.Tensor) -> dict[str, float]:
    """Mean/std/min/max plus the dead fraction, guarded for empty tensors."""
    if t is None or t.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "zero_frac": 1.0}
    f = t.detach().float()
    return {
        "mean": float(f.mean()),
        "std": float(f.std()) if f.numel() > 1 else 0.0,
        "min": float(f.min()),
        "max": float(f.max()),
        "zero_frac": float((f == 0).float().mean()),
    }


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """First tensor inside ``obj``, whatever shape of container it arrived in.

    Towers here return a mix of bare tensors, tuples and dataclasses
    (``GraphEncoding``), so unwrapping only tuples and dicts left the
    dataclass-returning towers silently unmonitored.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(obj, dict):
        for item in obj.values():
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields:
        for name in fields:
            found = _first_tensor(getattr(obj, name, None))
            if found is not None:
                return found
    return None


@dataclass
class StepReport:
    """Everything observed about one optimizer step."""

    step: int
    epoch: int
    split: str = "train"
    total: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    activations: dict[str, dict[str, float]] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    grad_zero_frac: float = 0.0
    router_distribution: dict[str, float] = field(default_factory=dict)
    head_spread: dict[str, float] = field(default_factory=dict)
    label_balance: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0
    grad_norm_total: float = 0.0

    def one_line(self) -> str:
        """A compact human-readable line, ordered most to least diagnostic."""
        parts = [f"{self.split[:5]:5s} e{self.epoch:02d} s{self.step:04d}",
                 f"loss={self.total:.4f}"]
        if self.terms:
            parts.append("(" + " ".join(
                f"{k}={v:.3f}" for k, v in sorted(self.terms.items())
            ) + ")")
        if self.grad_norm_total:
            parts.append(f"|g|={self.grad_norm_total:.3f}")
        if self.router_distribution:
            parts.append("route=" + ",".join(
                f"{k}:{v:.0%}" for k, v in self.router_distribution.items()
            ))
        return "  ".join(parts)


class BehaviorProbe:
    """Collects per-step behavior from a model, its outputs, and its gradients."""

    def __init__(self, model: torch.nn.Module, track_activations: bool = True):
        self.model = model
        self.track_activations = track_activations
        self._acts: dict[str, dict[str, float]] = {}
        self._handles: list[Any] = []
        if track_activations:
            self._register()

    # ---- forward hooks ----------------------------------------------------- #
    #: Direct children worth watching: the towers that carry the representation.
    #: Heads are covered separately by ``head_report``.
    TOWERS = (
        "sequence_encoder",
        "mamba_tower",
        "graph_encoder",
        "gat_tower",
        "fusion",
    )

    def _register(self) -> None:
        """Hook the representation towers, matched against the model's children.

        Names are looked up via ``named_children`` so a rename shows up as a
        missing tower in the report rather than as a hook that silently never
        fires.
        """
        children = dict(self.model.named_children())
        for name in self.TOWERS:
            module = children.get(name)
            if isinstance(module, torch.nn.Module):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, label: str):
        def hook(_module, _inputs, output):
            with torch.no_grad():
                tensor = _first_tensor(output)
                if tensor is not None:
                    self._acts[label] = _stats(tensor)
        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ---- observations ------------------------------------------------------ #
    def gradient_report(self) -> tuple[dict[str, float], float, float]:
        """Per-group grad norms, the global norm, and the all-zero fraction.

        Grouping is by top-level module so a silent head shows up as its own
        zero entry instead of being averaged away in a single global number.
        """
        groups: dict[str, list[float]] = defaultdict(list)
        total_sq, n_params, n_zero = 0.0, 0, 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                n_params += 1
                n_zero += 1
                continue
            norm = float(param.grad.detach().norm())
            groups[name.split(".")[0]].append(norm)
            total_sq += norm ** 2
            n_params += 1
            n_zero += int(norm == 0.0)
        per_group = {k: float(torch.tensor(v).norm()) for k, v in groups.items()}
        return per_group, total_sq ** 0.5, n_zero / max(n_params, 1)

    @staticmethod
    def router_report(outputs, route_names: Optional[Iterable[str]] = None
                      ) -> dict[str, float]:
        """Fraction of reads assigned to each compute route."""
        router = getattr(outputs, "router", None)
        if not isinstance(router, dict):
            return {}
        logits = router.get("logits")
        if logits is None or logits.numel() == 0:
            return {}
        with torch.no_grad():
            choice = logits.detach().argmax(-1)
            n_routes = logits.shape[-1]
            names = list(route_names) if route_names else [f"r{i}" for i in range(n_routes)]
            counts = torch.bincount(choice.flatten(), minlength=n_routes).float()
            counts = counts / max(float(counts.sum()), 1.0)
        return {names[i] if i < len(names) else f"r{i}": float(counts[i])
                for i in range(n_routes)}

    @staticmethod
    def head_report(outputs, seed_scores=None, chain_scores=None) -> dict[str, float]:
        """Spread of each head's outputs; a near-zero std means a constant head."""
        out: dict[str, float] = {}
        with torch.no_grad():
            mapping = getattr(outputs, "mapping", None)
            if isinstance(mapping, dict):
                for key in ("mapq", "position_fraction", "node_logits"):
                    value = mapping.get(key)
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        out[f"mapping.{key}.std"] = float(value.detach().float().std())
            for label, scores in (("seed", seed_scores), ("chain", chain_scores)):
                if isinstance(scores, dict):
                    logits = scores.get("logits")
                    if isinstance(logits, torch.Tensor) and logits.numel() > 1:
                        out[f"{label}.logits.std"] = float(logits.detach().float().std())
        return out

    def activations(self) -> dict[str, dict[str, float]]:
        return dict(self._acts)

    # ---- assembly ---------------------------------------------------------- #
    def report(self, step: int, epoch: int, loss_output, outputs, *, split: str = "train",
               seed_scores=None, chain_scores=None, lr: float = 0.0,
               label_balance: Optional[dict[str, float]] = None,
               with_gradients: bool = True) -> StepReport:
        grads, total_norm, zero_frac = (
            self.gradient_report() if with_gradients else ({}, 0.0, 0.0)
        )
        # RouterConfig owns the canonical names; ComplexityRouter.route_names is
        # a lookup method, not the list itself.
        router = getattr(self.model, "router", None)
        route_names = getattr(getattr(router, "cfg", None), "route_names", None)
        return StepReport(
            step=step,
            epoch=epoch,
            split=split,
            total=float(loss_output),
            terms=dict(getattr(loss_output, "terms", {})),
            weights=dict(getattr(loss_output, "weights", {})),
            activations=self.activations(),
            grad_norms=grads,
            grad_zero_frac=zero_frac,
            grad_norm_total=total_norm,
            router_distribution=self.router_report(outputs, route_names),
            head_spread=self.head_report(outputs, seed_scores, chain_scores),
            label_balance=dict(label_balance or {}),
            lr=lr,
        )
