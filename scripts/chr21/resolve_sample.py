#!/usr/bin/env python3
"""Resolve HPRC read URLs for scripts/chr21 from data/hprc/sample_links.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINKS = ROOT / "data/hprc/sample_links.json"

GIAB_DEFAULTS = {
    "HG005": {
        "reads_aln_url": (
            "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/"
            "ChineseTrio/HG005_NA24631_son/HG005_NA24631_son_HiSeq_300x/"
            "NHGRI_Illumina300X_Chinesetrio_novoalign_bams/"
            "HG005.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam"
        ),
        "truth_base": (
            "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
            "ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh38"
        ),
        "truth_vcf": "HG005_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        "truth_bed": "HG005_GRCh38_1_22_v4.2.1_benchmark.bed",
    },
    "HG002": {
        "reads_aln_url": (
            "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/"
            "AshkenazimTrio/HG002_NA24385_son/NIST_Illumina_2x250bps/"
            "novoalign_bams/HG002.GRCh38.2x250.bam"
        ),
        "truth_base": (
            "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
            "AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38"
        ),
        "truth_vcf": "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        "truth_bed": "HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed",
    },
}


def load_links() -> dict[str, dict]:
    if not LINKS.is_file():
        return {}
    rows = json.loads(LINKS.read_text())
    return {row["sample_id"]: row for row in rows}


def pick_hifi_bam(row: dict) -> str | None:
    runs = row.get("hifi_runs") or []
    if not runs:
        return None
    # Prefer the largest HiFi run for Sniffles.
    best = max(runs, key=lambda r: float(r.get("total_gbp") or 0))
    return best["path"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument(
        "field",
        choices=[
            "reads_aln_url",
            "hifi_bam_url",
            "truth_base",
            "truth_vcf",
            "truth_bed",
            "illumina_path",
            "hifi_path",
            "raw_bucket",
        ],
    )
    args = parser.parse_args()

    giab = GIAB_DEFAULTS.get(args.sample, {})
    row = load_links().get(args.sample, {})

    if args.field == "reads_aln_url":
        value = giab.get("reads_aln_url") or (row.get("illumina") or {}).get("path")
    elif args.field == "hifi_bam_url":
        value = pick_hifi_bam(row)
    elif args.field == "illumina_path":
        value = (row.get("illumina") or {}).get("path")
    elif args.field == "hifi_path":
        value = pick_hifi_bam(row)
    elif args.field == "raw_bucket":
        value = row.get("hprc_bucket")
    else:
        value = giab.get(args.field)

    if not value:
        print(f"ERROR: no {args.field} for sample {args.sample}", file=sys.stderr)
        return 1

    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
