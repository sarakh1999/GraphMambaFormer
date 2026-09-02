#!/usr/bin/env python3
"""Compare the Giraffe and GraphMambaFormer arms of the chr21 benchmark.

Pulls together whatever each arm produced and lays them side by side:

* **Mapping** (from the sorted BAMs): read count, mapped fraction, primary
  mapped, mean/median MAPQ. Always available once ``map_*`` has run.
* **Variants** (from the DeepVariant VCFs): PASS SNP / INDEL counts.
* **Accuracy** (from hap.py summaries, when a GIAB truth set exists): SNP/INDEL
  precision, recall, F1.

Writes ``data/chr21/<SAMPLE>/compare/compare.csv`` and, if matplotlib is present,
bar charts under ``data/chr21/<SAMPLE>/compare/``. Missing inputs are skipped
with a note, so this is safe to run at any point in the pipeline.

Usage:
    python scripts/chr21/compare_giraffe_ours.py --sample HG002 --chr chr21
    python scripts/chr21/compare_giraffe_ours.py --labels giraffe ours bwa
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

# Keep matplotlib's font cache off a possibly read-only $HOME.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

_ROOT = os.environ.get("CHR21_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _bam_stats(path: str) -> dict | None:
    """Mapping stats for one sorted BAM, or ``None`` if it is missing."""
    if not os.path.exists(path):
        return None
    import pysam

    n = mapped = primary_mapped = 0
    mapqs: list[int] = []
    with pysam.AlignmentFile(path, "rb") as bam:
        for a in bam.fetch(until_eof=True):
            if a.is_secondary or a.is_supplementary:
                continue
            n += 1
            if not a.is_unmapped:
                mapped += 1
                primary_mapped += 1
                mapqs.append(a.mapping_quality)
    mapqs.sort()
    mean_mapq = sum(mapqs) / len(mapqs) if mapqs else 0.0
    median_mapq = mapqs[len(mapqs) // 2] if mapqs else 0.0
    return {
        "reads": n,
        "mapped": mapped,
        "mapped_frac": (mapped / n) if n else 0.0,
        "mean_mapq": round(mean_mapq, 2),
        "median_mapq": median_mapq,
    }


def _vcf_counts(path: str) -> dict | None:
    """PASS SNP / INDEL counts for one VCF(.gz), or ``None`` if missing."""
    if not os.path.exists(path):
        return None
    import pysam

    snp = indel = 0
    with pysam.VariantFile(path) as vcf:
        for rec in vcf:
            filt = list(rec.filter.keys())
            if filt and filt != ["PASS"] and filt != ["."]:
                continue
            alts = rec.alts or ()
            for alt in alts:
                if rec.ref and len(rec.ref) == 1 and len(alt) == 1:
                    snp += 1
                else:
                    indel += 1
    return {"snp": snp, "indel": indel}


def _happy_summary(path: str) -> dict | None:
    """SNP/INDEL precision/recall/F1 (PASS) from a hap.py summary.csv."""
    if not os.path.exists(path):
        return None
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("Filter") != "PASS":
                continue
            vtype = row.get("Type")
            if vtype not in ("SNP", "INDEL"):
                continue

            def _f(key: str) -> float:
                try:
                    return float(row.get(key, "") or "nan")
                except ValueError:
                    return float("nan")

            out[vtype] = {
                "recall": _f("METRIC.Recall"),
                "precision": _f("METRIC.Precision"),
                "f1": _f("METRIC.F1_Score"),
            }
    return out or None


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if v < 1000 else f"{v:,.0f}"
    return str(v)


def _print_table(title: str, headers: list[str], rows: list[list]) -> None:
    print(f"\n== {title} ==")
    widths = [len(h) for h in headers]
    srows = [[_fmt(c) for c in r] for r in rows]
    for r in srows:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in srows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def _bar_chart(out_png, title, labels, series: dict[str, list[float]], ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed; skipping {os.path.basename(out_png)})")
        return
    import numpy as np

    keys = list(series)
    x = np.arange(len(keys))
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(keys) + 3, 5))
    for i, lab in enumerate(labels):
        vals = [series[k][i] for k in keys]
        bars = ax.bar(x + i * width, vals, width, label=lab)
        ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.environ.get("SAMPLE", "HG002"))
    ap.add_argument("--chr", default=os.environ.get("CHR", "chr21"))
    ap.add_argument("--labels", nargs="+", default=["giraffe", "ours"])
    args = ap.parse_args()

    s, chrom = args.sample, args.chr
    run = os.path.join(_ROOT, "data", "chr21", s)
    bam_dir = os.path.join(run, "bam")
    vcf_dir = os.path.join(run, "vcf")
    eval_dir = os.path.join(run, "eval")
    out_dir = os.path.join(run, "compare")
    os.makedirs(out_dir, exist_ok=True)

    per_label: dict[str, dict] = {}
    for label in args.labels:
        per_label[label] = {
            "map": _bam_stats(os.path.join(bam_dir, f"{s}.{chrom}.{label}.sorted.bam")),
            "vcf": _vcf_counts(os.path.join(vcf_dir, f"{s}.{chrom}.{label}.dv.vcf.gz")),
            "happy": _happy_summary(
                os.path.join(eval_dir, label, f"{s}.{chrom}.{label}.summary.csv")
            ),
        }

    present = [l for l in args.labels if any(per_label[l].values())]
    if not present:
        sys.exit(
            f"No results found for {s} {chrom} under {run}. "
            "Run map_giraffe.sh / map_ours.sh (+ DeepVariant / hap.py) first."
        )

    csv_rows: list[dict] = []

    # ---- mapping -----------------------------------------------------------
    map_labels = [l for l in present if per_label[l]["map"]]
    if map_labels:
        headers = ["metric"] + map_labels
        keys = ["reads", "mapped", "mapped_frac", "mean_mapq", "median_mapq"]
        rows = [[k] + [per_label[l]["map"][k] for l in map_labels] for k in keys]
        _print_table(f"Mapping — {s} {chrom}", headers, rows)
        for k in keys:
            for l in map_labels:
                csv_rows.append({"section": "mapping", "metric": k,
                                 "label": l, "value": per_label[l]["map"][k]})
        _bar_chart(os.path.join(out_dir, f"{s}.{chrom}.mapping.png"),
                   f"Mapping — {s} {chrom}", map_labels,
                   {"mapped_frac": [per_label[l]["map"]["mapped_frac"] for l in map_labels],
                    "mean_mapq/60": [per_label[l]["map"]["mean_mapq"] / 60 for l in map_labels]},
                   "fraction (MAPQ scaled /60)")

    # ---- variants ----------------------------------------------------------
    vcf_labels = [l for l in present if per_label[l]["vcf"]]
    if vcf_labels:
        headers = ["variant"] + vcf_labels
        rows = [[k] + [per_label[l]["vcf"][k] for l in vcf_labels] for k in ("snp", "indel")]
        _print_table(f"DeepVariant PASS counts — {s} {chrom}", headers, rows)
        for k in ("snp", "indel"):
            for l in vcf_labels:
                csv_rows.append({"section": "variants", "metric": k,
                                 "label": l, "value": per_label[l]["vcf"][k]})
        _bar_chart(os.path.join(out_dir, f"{s}.{chrom}.variants.png"),
                   f"DeepVariant PASS counts — {s} {chrom}", vcf_labels,
                   {"SNP": [per_label[l]["vcf"]["snp"] for l in vcf_labels],
                    "INDEL": [per_label[l]["vcf"]["indel"] for l in vcf_labels]},
                   "variant count")

    # ---- accuracy (hap.py) -------------------------------------------------
    hp_labels = [l for l in present if per_label[l]["happy"]]
    if hp_labels:
        for vtype in ("SNP", "INDEL"):
            headers = ["metric"] + hp_labels
            metrics = ["precision", "recall", "f1"]
            rows = []
            ok = False
            for m in metrics:
                row = [m]
                for l in hp_labels:
                    v = per_label[l]["happy"].get(vtype, {}).get(m, float("nan"))
                    row.append(v)
                    ok = ok or v == v  # not NaN
                rows.append(row)
            if ok:
                _print_table(f"hap.py {vtype} (PASS) — {s} {chrom}", headers, rows)
                for m in metrics:
                    for l in hp_labels:
                        csv_rows.append({
                            "section": f"happy_{vtype}", "metric": m, "label": l,
                            "value": per_label[l]["happy"].get(vtype, {}).get(m, "")})
        _bar_chart(
            os.path.join(out_dir, f"{s}.{chrom}.happy_f1.png"),
            f"hap.py F1 (PASS) — {s} {chrom}", hp_labels,
            {vt: [per_label[l]["happy"].get(vt, {}).get("f1", 0.0) for l in hp_labels]
             for vt in ("SNP", "INDEL")},
            "F1")
    else:
        print("\n(no hap.py summaries — accuracy comparison needs a GIAB truth "
              "set; skipped for HPRC-only samples)")

    # ---- csv ---------------------------------------------------------------
    csv_path = os.path.join(out_dir, "compare.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "metric", "label", "value"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
