#!/usr/bin/env python3
"""Overlay precision/recall curves from hap.py outputs (Fig 6a style).

Reads data/fig6/<SAMPLE>/eval/<label>/*.roc.all.csv.gz from eval_happy.sh
and draws recall (y) vs precision (x), with PASS F1 annotated.

Usage:
    .venv/bin/python scripts/fig6/plot_pr.py --sample HG002 [--type SNP|INDEL]
"""
from __future__ import annotations
import argparse, glob, os, sys

# FIG6_ROOT points at the bind-mounted repo when this script is the copy baked
# into the fig6 runner image.
ROOT = os.environ.get("FIG6_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _need(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(
            f"Missing '{mod}'. Install with:\n"
            f"  {sys.executable} -m pip install matplotlib pandas"
        )


def main() -> None:
    pd = _need("pandas")
    mpl = _need("matplotlib")
    mpl.use("Agg")
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.environ.get("SAMPLE", "HG002"))
    ap.add_argument("--chr", default=os.environ.get("CHR", "chr20"))
    ap.add_argument("--type", default="SNP", choices=["SNP", "INDEL"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--xmin", type=float, default=0.996,
                    help="Precision axis lower bound (paper-style zoom)")
    ap.add_argument("--xmax", type=float, default=1.0)
    ap.add_argument("--ymin", type=float, default=None,
                    help="Recall axis lower bound (default: auto from data)")
    ap.add_argument("--ymax", type=float, default=1.0)
    args = ap.parse_args()

    eval_dir = os.path.join(ROOT, "data", "fig6", args.sample, "eval")
    plot_dir = os.path.join(ROOT, "data", "fig6", args.sample, "plots")
    out = args.out or os.path.join(plot_dir, f"fig6a_{args.chr}.png")

    rocs = sorted(glob.glob(os.path.join(eval_dir, "*", "*.roc.all.csv.gz")))
    if not rocs:
        sys.exit(f"No hap.py ROC files under {eval_dir}. Run eval_happy.sh first.")

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Friendly legend names matching the paper
    pretty = {
        "giraffe": "HPRC Giraffe / DeepVariant",
        "bwa": "BWAMEM / DeepVariant",
        "giraffe_trio": "HPRC Giraffe / DeepTrio",
        "bwa_trio": "BWAMEM / DeepTrio",
    }

    recalls_in_view = []
    for roc in rocs:
        label = os.path.basename(os.path.dirname(roc))
        df = pd.read_csv(roc)
        subset_col = "Subset" if "Subset" in df.columns else None
        sel = df[(df["Type"] == args.type) & (df["Filter"] == "PASS")].copy()
        if subset_col:
            sel = sel[sel[subset_col] == "*"]
        if sel.empty:
            print(f"  ! {label}: no {args.type}/PASS rows, skipping")
            continue
        # Keep only points inside / near the zoomed precision window
        sel = sel[
            (sel["METRIC.Precision"] >= args.xmin - 1e-4)
            & (sel["METRIC.Precision"] <= args.xmax + 1e-4)
        ].copy()
        if sel.empty:
            print(f"  ! {label}: no points in precision [{args.xmin}, {args.xmax}]")
            continue
        sel = sel.sort_values("METRIC.Precision")
        recalls_in_view.extend(sel["METRIC.Recall"].tolist())
        f1 = None
        summ = roc.replace(".roc.all.csv.gz", ".summary.csv")
        if os.path.exists(summ):
            s = pd.read_csv(summ)
            row = s[(s["Type"] == args.type) & (s["Filter"] == "PASS")]
            if not row.empty:
                f1 = float(row["METRIC.F1_Score"].iloc[0])
        name = pretty.get(label, label)
        leg = name + (f"  (F1={f1:.4f})" if f1 is not None else "")
        ax.plot(sel["METRIC.Precision"], sel["METRIC.Recall"], lw=2, label=leg)

    ax.set_xlim(args.xmin, args.xmax)
    if args.ymin is not None:
        ymin = args.ymin
    elif recalls_in_view:
        ymin = max(0.0, min(recalls_in_view) - 0.002)
    else:
        ymin = 0.99
    ax.set_ylim(ymin, args.ymax)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(f"Fig 6a-style ({args.chr}) — {args.type} — {args.sample}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}  (precision [{args.xmin}, {args.xmax}], recall [{ymin}, {args.ymax}])")


if __name__ == "__main__":
    main()
