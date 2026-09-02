#!/usr/bin/env python3
"""Combined curriculum summary across all training stages.

Reads every run dir given on the command line (each must have history.json +
run_meta.json) and renders one figure: overlaid loss / locus / anchor-AUC curves
plus a parameter+results comparison table.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

STAGE_COLORS = ["#08519c", "#e6550d", "#31a354", "#6a51a3", "#d94801"]


def load(run_dir: str):
    hist = json.load(open(os.path.join(run_dir, "history.json")))
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    return hist, meta


def region_len(meta) -> int:
    r = meta["data"]["region"]
    a, b = r.split(":")[1].split("-")
    return int(b) - int(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="stage_label=run_dir pairs, in curriculum order")
    ap.add_argument("--out", default="data/training_runs/curriculum_summary.png")
    args = ap.parse_args()

    stages = []
    for spec in args.runs:
        label, run_dir = spec.split("=", 1)
        if not (os.path.exists(os.path.join(run_dir, "history.json"))
                and os.path.exists(os.path.join(run_dir, "run_meta.json"))
                and json.load(open(os.path.join(run_dir, "history.json"))).get("validations")):
            print(f"skip {label}: run not complete yet ({run_dir})")
            continue
        hist, meta = load(run_dir)
        stages.append((label, run_dir, hist, meta))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig = plt.figure(figsize=(19, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22,
                  height_ratios=[1.0, 0.9])
    fig.suptitle("GraphMambaFormer chr21 curriculum — increasing region/graph complexity",
                 fontsize=15, fontweight="bold", y=0.98)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_locus = fig.add_subplot(gs[0, 1])
    ax_auc = fig.add_subplot(gs[0, 2])

    for i, (label, _, hist, meta) in enumerate(stages):
        c = STAGE_COLORS[i % len(STAGE_COLORS)]
        vals = hist["validations"]
        ep = list(range(len(vals)))
        vl = [v["loss"] for v in vals]
        loc = [v["locus_accuracy"] for v in vals]
        auc = [v["anchor_auc"] for v in vals]
        tag = f"{label}"
        ax_loss.plot(ep, vl, "o-", color=c, lw=2, ms=5, label=tag)
        ax_locus.plot(ep, loc, "o-", color=c, lw=2, ms=5, label=tag)
        ax_auc.plot(ep, auc, "o-", color=c, lw=2, ms=5, label=tag)

    ax_loss.set_title("Validation loss", fontweight="bold")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("val loss"); ax_loss.legend(fontsize=8)
    ax_locus.set_title("Locus accuracy (val)", fontweight="bold")
    ax_locus.set_xlabel("epoch"); ax_locus.set_ylabel("fraction")
    ax_locus.set_ylim(0, 1.08); ax_locus.legend(fontsize=8)
    ax_auc.set_title("Anchor AUC (val)  — 0.5 = single-class/degenerate", fontweight="bold")
    ax_auc.set_xlabel("epoch"); ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim(0.4, 1.02); ax_auc.axhline(0.5, color="#999", ls=":", lw=1)
    ax_auc.legend(fontsize=8)

    # comparison table
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    headers = ["Stage", "Region", "Window", "Graph (nodes/edges)", "Reads tr/val",
               "Warm-start", "Epochs", "Final val loss", "Locus", "AnchorAUC", "Time"]
    rows = []
    for label, _, hist, meta in stages:
        d = meta["data"]; pang = d["references"]["pangenome"]
        last = hist["validations"][-1]
        secs = sum(e["seconds"] for e in hist["epochs"])
        ws = region_len(meta)
        ws_s = f"{ws//1000} kb" if ws < 1_000_000 else f"{ws/1e6:.1f} Mb"
        init = meta.get("init_checkpoint")
        rows.append([
            label,
            d["region"].replace("chr21:", ""),
            ws_s,
            f"{pang['n_nodes']:,}/{pang['n_edges']:,}",
            f"{d['n_train_reads']}/{d['n_val_reads']}",
            "cold" if not init else os.path.basename(os.path.dirname(init)),
            str(len(hist["epochs"])),
            f"{last['loss']:.3f}",
            f"{last['locus_accuracy']*100:.0f}%",
            f"{last['anchor_auc']:.3f}",
            f"{secs/60:.0f}m",
        ])
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#d9d9d9"); tbl[0, j].set_text_props(fontweight="bold")
    ax.set_title("Per-stage parameters & final results", fontweight="bold", y=0.86)

    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
