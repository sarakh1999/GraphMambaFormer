"""Plots for a training run: the losses, the validation metrics, the behavior.

matplotlib is imported lazily and the ``Agg`` backend is forced, so these work
headless (in Docker, over SSH, in CI) and importing this module never requires a
display. If matplotlib is absent, :func:`plot_all` says so and returns an empty
list instead of raising -- plotting is reporting, and a missing plotting library
should not fail a training run that already succeeded.

Each figure answers one question:

  ``losses``      is the objective going down, and which term dominates?
  ``weights``     how is Kendall balancing re-weighting the terms over time?
  ``validation``  is alignment quality improving, or just the loss?
  ``behavior``    are the towers alive, are gradients flowing, is the router routing?
  ``calibration`` does the MAPQ the model claims match the errors it makes?
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

__all__ = ["plot_all", "plotting_available"]


def _pyplot():
    """Return ``pyplot`` on the Agg backend, or ``None`` if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plotting_available() -> bool:
    return _pyplot() is not None


def _save(fig, out_dir: str, name: str, written: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    written.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_losses(plt, history, out_dir: str, written: list[str]) -> None:
    steps = history.steps
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    x = [s.step for s in steps]
    axes[0].plot(x, [s.total for s in steps], lw=1.2, color="#1f77b4",
                 label="train total")
    if history.validations:
        per_epoch = max(1, len(steps) // max(len(history.validations), 1))
        vx = [(i + 1) * per_epoch for i in range(len(history.validations))]
        axes[0].plot(vx, [v.loss for v in history.validations], "o-", lw=1.4,
                     color="#d62728", label="val total")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Total objective")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    names = sorted({k for s in steps for k in s.terms})
    for name in names:
        axes[1].plot(x, [s.terms.get(name, float("nan")) for s in steps],
                     lw=1.0, label=name)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Per-term breakdown")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    _save(fig, out_dir, "01_losses.png", written)


def _plot_weights(plt, history, out_dir: str, written: list[str]) -> None:
    steps = [s for s in history.steps if s.weights]
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in sorted({k for s in steps for k in s.weights}):
        ax.plot([s.step for s in steps],
                [s.weights.get(name, float("nan")) for s in steps], lw=1.2,
                label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("effective weight")
    ax.set_title("Kendall log-variance weighting (learned balance)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "02_loss_weights.png", written)


def _plot_validation(plt, history, out_dir: str, written: list[str]) -> None:
    vals = history.validations
    if not vals:
        return
    epochs = list(range(len(vals)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, [v.chain_accuracy for v in vals], "o-", label="chain acc")
    axes[0].plot(epochs, [v.anchor_auc for v in vals], "s-", label="anchor AUC")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Alignment quality")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [v.anchor_precision for v in vals], "o-", label="precision")
    axes[1].plot(epochs, [v.anchor_recall for v in vals], "s-", label="recall")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Seed head precision / recall")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [v.mapq_mae for v in vals], "o-", color="#9467bd")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MAE (MAPQ points)")
    axes[2].set_title("MAPQ error")
    axes[2].grid(alpha=0.3)
    _save(fig, out_dir, "03_validation.png", written)


def _plot_behavior(plt, history, out_dir: str, written: list[str]) -> None:
    # The behaviour probe is sampled every ``probe_every`` steps, so plot only
    # the steps that actually carry probe data. This keeps the curves dense
    # rather than drawing zero/NaN gaps for the un-sampled steps in between.
    steps = [
        s for s in history.steps
        if s.activations or s.grad_norms or s.router_distribution or s.head_spread
    ] or history.steps
    if not steps:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    x = [s.step for s in steps]

    towers = sorted({k for s in steps for k in s.activations})
    for tower in towers:
        axes[0][0].plot(x, [s.activations.get(tower, {}).get("std", float("nan"))
                            for s in steps], lw=1.1, label=tower)
    axes[0][0].set_title("Activation spread per tower (0 = dead)")
    axes[0][0].set_xlabel("step")
    axes[0][0].set_ylabel("std")
    axes[0][0].legend(fontsize=7)
    axes[0][0].grid(alpha=0.3)

    axes[0][1].plot(x, [s.grad_norm_total for s in steps], lw=1.1, color="#2ca02c",
                    label="global")
    for group in sorted({k for s in steps for k in s.grad_norms})[:6]:
        axes[0][1].plot(x, [s.grad_norms.get(group, float("nan")) for s in steps],
                        lw=0.9, alpha=0.8, label=group)
    axes[0][1].set_yscale("log")
    axes[0][1].set_title("Gradient norms")
    axes[0][1].set_xlabel("step")
    axes[0][1].legend(fontsize=7, ncol=2)
    axes[0][1].grid(alpha=0.3)

    routes = sorted({k for s in steps for k in s.router_distribution})
    if routes:
        axes[1][0].stackplot(
            x, *[[s.router_distribution.get(r, 0.0) for s in steps] for r in routes],
            labels=routes, alpha=0.85,
        )
        axes[1][0].set_ylim(0, 1)
        axes[1][0].legend(fontsize=7, loc="upper right")
    else:
        axes[1][0].text(0.5, 0.5, "no router in this model", ha="center")
    axes[1][0].set_title("Router: fraction of reads per compute path")
    axes[1][0].set_xlabel("step")

    spreads = sorted({k for s in steps for k in s.head_spread})
    for key in spreads:
        axes[1][1].plot(x, [s.head_spread.get(key, float("nan")) for s in steps],
                        lw=1.1, label=key)
    axes[1][1].set_title("Head output spread (0 = constant prediction)")
    axes[1][1].set_xlabel("step")
    axes[1][1].legend(fontsize=7)
    axes[1][1].grid(alpha=0.3)
    _save(fig, out_dir, "04_model_behavior.png", written)


def _plot_calibration(plt, history, out_dir: str, written: list[str]) -> None:
    vals = history.validations
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    epochs = list(range(len(vals)))
    ax.plot(epochs, [v.mapq_expected_error for v in vals], "o-",
            label="error MAPQ promises")
    ax.plot(epochs, [v.mapq_observed_error for v in vals], "s-",
            label="error actually observed")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error rate")
    ax.set_title("MAPQ calibration (gap = over/under-confidence)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "05_mapq_calibration.png", written)


def _plot_label_balance(plt, history, out_dir: str, written: list[str]) -> None:
    steps = [s for s in history.steps if s.label_balance]
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in sorted({k for s in steps for k in s.label_balance}):
        ax.plot([s.step for s in steps],
                [s.label_balance.get(key, float("nan")) for s in steps],
                lw=1.1, label=key)
    ax.set_xlabel("step")
    ax.set_title("Supervision actually available per batch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "06_label_balance.png", written)


def plot_all(history, out_dir: str = "data/training_runs/latest/plots",
             verbose: bool = True) -> list[str]:
    """Write every figure for ``history``; returns the paths written."""
    plt = _pyplot()
    if plt is None:
        if verbose:
            print("matplotlib not installed - skipping plots "
                  "(pip install matplotlib)")
        return []

    written: list[str] = []
    for fn in (_plot_losses, _plot_weights, _plot_validation, _plot_behavior,
               _plot_calibration, _plot_label_balance):
        fn(plt, history, out_dir, written)
    if verbose and written:
        print(f"wrote {len(written)} figures to {out_dir}")
        for path in written:
            print(f"   {os.path.basename(path)}")
    return written
