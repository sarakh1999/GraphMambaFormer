#!/usr/bin/env python
"""Plot the training loss curve from the first step to the latest step.

Reads the structured ``history.json`` a run writes (the source of truth for
per-step losses) and, optionally, scrapes the live training log to pick up the
few most recent steps that have not been flushed to ``history.json`` yet. The
two sources are merged by optimizer step (log wins on overlap since it is the
freshest), so the curve always runs from step 0 to the very last logged step.

Usage:
    python scripts/plot_loss.py \
        --run-dir /fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d256 \
        --log logs/hg005_local.log \
        --out logs/loss_curve.png

All arguments are optional; the defaults point at the HG005 d256 run.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: no DISPLAY on the login/compute nodes
import matplotlib.pyplot as plt

TERM_KEYS = ["seed", "transition", "chain", "position", "mapq", "router"]

# Matches the periodic log line, e.g.:
#   train e00 s0283  loss=4.8168  (chain=0.000 mapq=0.003 position=0.056 ...)
_LOG_RE = re.compile(
    r"train e(?P<epoch>\d+) s(?P<step>\d+)\s+loss=(?P<loss>[0-9.]+)\s+\((?P<terms>[^)]*)\)"
)


def load_history(run_dir: str) -> dict[int, dict]:
    """Return {step: record} parsed from history.json (empty if absent)."""
    path = os.path.join(run_dir, "history.json")
    out: dict[int, dict] = {}
    if not os.path.exists(path):
        print(f"note: {path} not found; relying on --log only")
        return out
    with open(path) as fh:
        hist = json.load(fh)
    for s in hist.get("steps", []):
        if s.get("split") not in (None, "train"):
            continue
        step = int(s["step"])
        out[step] = {
            "total": float(s["total"]),
            "terms": {k: float(s.get("terms", {}).get(k, float("nan")))
                      for k in TERM_KEYS},
        }
    return out


def scrape_log(log_path: str | None) -> dict[int, dict]:
    """Return {step: record} parsed from the text training log (may be empty)."""
    out: dict[int, dict] = {}
    if not log_path or not os.path.exists(log_path):
        return out
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            m = _LOG_RE.search(line)
            if not m:
                continue
            terms = {}
            for kv in m.group("terms").split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        terms[k] = float(v)
                    except ValueError:
                        pass
            out[int(m.group("step"))] = {
                "total": float(m.group("loss")),
                "terms": {k: terms.get(k, float("nan")) for k in TERM_KEYS},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir",
                    default="/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d256",
                    help="run directory containing history.json")
    ap.add_argument("--log", default=None,
                    help="optional training log to merge in the freshest steps")
    ap.add_argument("--out", default="logs/loss_curve.png",
                    help="output PNG path")
    ap.add_argument("--title", default=None, help="override the figure title")
    args = ap.parse_args()

    merged = load_history(args.run_dir)
    merged.update(scrape_log(args.log))  # log wins on overlapping steps
    if not merged:
        raise SystemExit("no loss data found in history.json or --log")

    steps = sorted(merged)
    total = [merged[s]["total"] for s in steps]
    terms = {k: [merged[s]["terms"][k] for s in steps] for k in TERM_KEYS}

    title = args.title or (
        f"Loss curve  (step {steps[0]} -> {steps[-1]}, {len(steps)} points)\n"
        f"{args.run_dir}"
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax[0].plot(steps, total, color="#1f77b4", lw=1.6)
    ax[0].set_title("Total training loss")
    ax[0].set_xlabel("optimizer step")
    ax[0].set_ylabel("loss")
    ax[0].grid(alpha=0.3)

    for k in TERM_KEYS:
        ax[1].plot(steps, terms[k], lw=1.1, label=k)
    ax[1].set_title("Per-term loss (unweighted)")
    ax[1].set_xlabel("optimizer step")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    print(f"steps {steps[0]}..{steps[-1]} ({len(steps)} points)  "
          f"total loss {total[0]:.3f} -> {total[-1]:.3f}  min {min(total):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
