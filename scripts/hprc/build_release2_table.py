#!/usr/bin/env python3
"""Build HPRC **Release 2** raw-sequencing-data download tables for the
GraphMambaFormer long-read training cohort.

Pulls the three official Release-2 index CSVs and produces:
  data/hprc/release2/release2_master.csv     one row per sample (all 232)
  data/hprc/release2/hifi_revio_bam.tsv      per-file HiFi (Revio, unaligned BAM)
  data/hprc/release2/ont_r1041_bam.tsv       per-file ONT  (R1041, unaligned BAM)
  data/hprc/release2/illumina.tsv            per-file Illumina (cram/fastq)
  data/hprc/release2/RELEASE2_TABLE.md       readable summary + matched cohorts

Design choices for this project (long-read aligner, pipeline consumes UNALIGNED
BAM directly):
  * HiFi  -> keep only instrument_model == "Revio" AND filetype == "bam"
            (newest PacBio chemistry; ~196 samples the portal shows as ~200).
  * ONT   -> keep only sequencing_chemistry == "R1041" AND filetype == "bam"
            (R10.4.1 flowcells; do NOT mix with the older R941 error profile).
  * Illumina -> the per-sample WGS cram (or fastq fallback) for the short-read arm.

Direct-download HTTPS is derived from the s3:// path (public bucket, no auth).
"""
import csv
import os
import urllib.request
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(ROOT, "data", "hprc", ".cache_r2")
OUT = os.path.join(ROOT, "data", "hprc", "release2")
BASE = ("https://raw.githubusercontent.com/human-pangenomics/"
        "hprc_intermediate_assembly/main/data_tables/sequencing_data")
CSVS = {
    "hifi": "data_hifi_release2_v1.0.index.csv",
    "ont": "data_ont_release2_v1.0.index.csv",
    "illumina": "data_illumina_release2_v1.0.index.csv",
}


def s3_to_https(path: str) -> str:
    """s3://human-pangenomics/KEY -> HPRC's canonical public download URL.

    Uses the us-west-2 path-style endpoint (same host HPRC's data browser uses).
    For scripted bulk pulls prefer `aws s3 cp --no-sign-request <s3_path>`.
    """
    if path.startswith("s3://"):
        _, _, rest = path.partition("s3://")
        bucket, _, key = rest.partition("/")
        return f"https://s3-us-west-2.amazonaws.com/{bucket}/{key}"
    return path


def load(name: str):
    os.makedirs(CACHE, exist_ok=True)
    local = os.path.join(CACHE, name)
    if not os.path.exists(local):
        urllib.request.urlretrieve(f"{BASE}/{name}", local)
    with open(local) as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    os.makedirs(OUT, exist_ok=True)
    hifi = load(CSVS["hifi"])
    ont = load(CSVS["ont"])
    ill = load(CSVS["illumina"])

    # ---- filter to the modalities/chemistries this project trains on ----
    hifi_revio = [r for r in hifi
                  if r["instrument_model"] == "Revio" and r["filetype"] == "bam"]
    ont_r1041 = [r for r in ont
                 if r["sequencing_chemistry"] == "R1041" and r["filetype"] == "bam"]

    # ---- per-file TSVs (these are the actual download lists) ----
    def write_tsv(path, rows, cols, coverage_key):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["sample_id", "filename", "coverage",
                        "total_gbp", "download_url", "s3_path"] + cols)
            for r in sorted(rows, key=lambda r: (r["sample_id"], r["filename"])):
                gbp = fnum(r.get("total_bp") or r.get("total_gbp") or
                           r.get("total_gpb")) 
                if gbp > 1e6:  # total_bp in bp -> Gbp
                    gbp = gbp / 1e9
                w.writerow([r["sample_id"], r["filename"], r.get(coverage_key, ""),
                            f"{gbp:.1f}", s3_to_https(r["path"]), r["path"]]
                           + [r.get(c, "") for c in cols])

    write_tsv(os.path.join(OUT, "hifi_revio_bam.tsv"), hifi_revio,
              ["seq_plate_chemistry_version", "library_id", "instrument_model"],
              "coverage")
    write_tsv(os.path.join(OUT, "ont_r1041_bam.tsv"), ont_r1041,
              ["n50", "basecaller_model", "seq_kit"], "coverage")
    write_tsv(os.path.join(OUT, "illumina.tsv"), ill,
              ["filetype", "instrument_model", "read_length"], "coverage")

    # ---- per-sample aggregation for the master table ----
    def agg(rows, cov_key="coverage"):
        d = defaultdict(lambda: {"n": 0, "cov": 0.0, "first": None})
        for r in rows:
            s = r["sample_id"]
            d[s]["n"] += 1
            d[s]["cov"] += fnum(r.get(cov_key))
            if d[s]["first"] is None:
                d[s]["first"] = s3_to_https(r["path"])
        return d

    hR, oR = agg(hifi_revio), agg(ont_r1041)
    illmap = {r["sample_id"]: r for r in ill}

    samples = sorted(set(hR) | set(oR) | set(illmap))
    master = os.path.join(OUT, "release2_master.csv")
    with open(master, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "sample_id",
            "hifi_revio_bam_files", "hifi_revio_total_cov",
            "ont_r1041_bam_files", "ont_r1041_total_cov",
            "illumina_file", "illumina_cov", "illumina_type",
            "has_hifi_revio", "has_ont_r1041", "has_illumina", "matched_all3",
            "hifi_first_url", "ont_first_url", "illumina_url",
        ])
        for s in samples:
            h, o = hR.get(s), oR.get(s)
            i = illmap.get(s)
            has_h, has_o, has_i = bool(h), bool(o), bool(i)
            w.writerow([
                s,
                h["n"] if h else 0, f"{h['cov']:.1f}" if h else "0",
                o["n"] if o else 0, f"{o['cov']:.1f}" if o else "0",
                i["filename"] if i else "", i["coverage"] if i else "",
                i["filetype"] if i else "",
                int(has_h), int(has_o), int(has_i),
                int(has_h and has_o and has_i),
                h["first"] if h else "", o["first"] if o else "",
                s3_to_https(i["path"]) if i else "",
            ])

    # ---- markdown summary ----
    matched = [s for s in samples if s in hR and s in oR and s in illmap]
    lr_matched = [s for s in samples if s in hR and s in oR]  # both long-read
    md = [
        "# HPRC Release 2 — raw sequencing download tables",
        "",
        "Generated by `scripts/hprc/build_release2_table.py` from the official "
        "Release-2 index CSVs.",
        "",
        "Filters applied (matched to the GraphMambaFormer long-read pipeline, "
        "which consumes **unaligned BAM** directly):",
        "",
        "- **HiFi**: `instrument_model == Revio` and `filetype == bam`",
        "- **ONT**: `sequencing_chemistry == R1041` and `filetype == bam`",
        "- **Illumina**: per-sample WGS `cram` (or `fastq` fallback)",
        "",
        "## Cohort sizes",
        "",
        "| Set | Samples | Files |",
        "| --- | --- | --- |",
        f"| HiFi Revio (bam) | {len(hR)} | {len(hifi_revio)} |",
        f"| ONT R1041 (bam) | {len(oR)} | {len(ont_r1041)} |",
        f"| Illumina | {len(illmap)} | {len(ill)} |",
        f"| **Matched HiFi+ONT (both newest chem)** | **{len(lr_matched)}** | — |",
        f"| **Matched all three modalities** | **{len(matched)}** | — |",
        "",
        "Per-file download lists: `hifi_revio_bam.tsv`, `ont_r1041_bam.tsv`, "
        "`illumina.tsv`. Per-sample rollup: `release2_master.csv`.",
        "",
        "## Matched all-three-modality samples",
        "",
        "```",
        "\n".join(matched),
        "```",
    ]
    with open(os.path.join(OUT, "RELEASE2_TABLE.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"HiFi Revio bam : {len(hifi_revio)} files / {len(hR)} samples")
    print(f"ONT R1041 bam  : {len(ont_r1041)} files / {len(oR)} samples")
    print(f"Illumina       : {len(ill)} files / {len(illmap)} samples")
    print(f"matched HiFi+ONT      : {len(lr_matched)} samples")
    print(f"matched all 3         : {len(matched)} samples")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
