#!/usr/bin/env python3
"""Build a GraphMambaFormer training manifest for a cohort of HPRC samples.

The manifest is the input to ``scripts/train.py --manifest`` for joint
multi-sample / multi-modality training. One entry is emitted per
``(sample, modality)`` pair that the sample actually has (per the Release-2
master table), pointing at the truth BAM that supervises that pair:

    illumina      -> Giraffe   (short-read graph aligner)
    pacbio_hifi   -> minimap2  (-x map-hifi)
    ont           -> minimap2  (-x map-ont)

Truth-BAM paths are derived from ``--truth-template`` (see defaults below); the
files are produced by the data-prep step and do not need to exist yet — pass
``--only-existing`` once they do to drop any that are still missing.

Examples
--------
    # 16 all-3-modality samples on chr21 (preview: emits all, warns on missing)
    PYTHONPATH=. python scripts/hprc/build_train_manifest.py \
        --cohort data/hprc/cohorts/cohort_16.txt \
        --region chr21 \
        --reference-fasta data/chr21/HG002/ref/chr21.fa \
        --gfa data/chr21/HG002/chr21.gfa \
        --out data/hprc/manifests/cohort_16.chr21.json

    # scale to all 232 (each sample contributes whatever modalities it has),
    # keeping only pairs whose truth BAM is already built
    PYTHONPATH=. python scripts/hprc/build_train_manifest.py \
        --cohort data/hprc/cohorts/cohort_232.txt --only-existing \
        --reference-fasta data/chr21/HG002/ref/chr21.fa \
        --gfa data/chr21/HG002/chr21.gfa \
        --out data/hprc/manifests/cohort_232.chr21.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
MASTER_CSV = os.path.join(ROOT, "data/hprc/release2/release2_master.csv")

from graphmambaformer.progress import progress

# (train modality, short tag used in paths, master-table availability column,
#  default truth aligner)
MODALITIES = {
    "illumina": ("illumina", "illumina", "has_illumina", "giraffe"),
    "hifi": ("pacbio_hifi", "hifi", "has_hifi_revio", "minimap2"),
    "ont": ("ont", "ont", "has_ont_r1041", "minimap2"),
}


def load_master(path: str) -> dict[str, dict]:
    with open(path, newline="") as fh:
        return {row["sample_id"]: row for row in csv.DictReader(fh)}


def read_cohort(path: str) -> list[str]:
    samples: list[str] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                samples.append(s)
    return samples


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cohort", help="file with one sample id per line")
    src.add_argument("--n", type=int,
                     help="use the first N all-3-modality (matched53) samples")
    p.add_argument("--modalities", default="illumina,hifi,ont",
                   help="comma list from {illumina,hifi,ont} (default: all three)")
    p.add_argument("--region", default="chr21",
                   help="samtools-style region tag stored in each entry and used "
                        "in the truth-BAM path (default: chr21)")
    p.add_argument("--reference-fasta",
                   default="data/chr21/HG002/ref/chr21.fa",
                   help="reference FASTA (manifest global default)")
    p.add_argument("--gfa", default="data/chr21/HG002/chr21.gfa",
                   help="pangenome GFA (manifest global default; needed for "
                        "ref-mode pangenome/both)")
    p.add_argument("--ref-mode", default="both",
                   choices=("linear", "pangenome", "both"),
                   help="stored as a manifest global (train.py --ref-mode "
                        "still overrides at run time)")
    p.add_argument("--truth-root", default="data/hprc/truth",
                   help="root dir for truth BAMs")
    p.add_argument("--truth-template",
                   default="{truth_root}/{sample}/{sample}.{mod_short}.{aligner}.{region}.sorted.bam",
                   help="path template; placeholders: {truth_root} {sample} "
                        "{mod_short} {modality} {aligner} {region}")
    p.add_argument("--only-existing", action="store_true",
                   help="drop entries whose truth BAM file does not exist yet")
    p.add_argument("--master", default=MASTER_CSV,
                   help="Release-2 master CSV (modality availability per sample)")
    p.add_argument("--out", required=True, help="output manifest JSON path")
    args = p.parse_args()

    master = load_master(args.master)

    if args.n is not None:
        matched = [s for s, r in master.items() if r.get("matched_all3") == "1"]
        matched.sort()
        cohort = matched[: args.n]
    else:
        cohort = read_cohort(args.cohort)

    wanted = [m.strip() for m in args.modalities.split(",") if m.strip()]
    for m in wanted:
        if m not in MODALITIES:
            sys.exit(f"ERROR: unknown modality '{m}' (choose from "
                     f"{','.join(MODALITIES)})")

    # normalise the region into a filename-safe tag (chr21:1-2 -> chr21_1-2)
    region_tag = args.region.replace(":", "_") if args.region else "all"

    entries: list[dict] = []
    n_missing = 0
    missing_examples: list[str] = []
    skipped_samples: list[str] = []
    no_modality: list[str] = []

    for sample in progress(cohort, desc="scan cohort", unit="sample"):
        row = master.get(sample)
        if row is None:
            skipped_samples.append(sample)
            continue
        emitted_any = False
        for m in wanted:
            modality, mod_short, avail_col, aligner = MODALITIES[m]
            if row.get(avail_col) != "1":
                continue                       # sample lacks this modality
            truth_bam = args.truth_template.format(
                truth_root=args.truth_root, sample=sample, mod_short=mod_short,
                modality=modality, aligner=aligner, region=region_tag,
            )
            exists = os.path.exists(truth_bam)
            if not exists:
                n_missing += 1
                if len(missing_examples) < 8:
                    missing_examples.append(truth_bam)
                if args.only_existing:
                    continue
            entries.append({
                "sample": sample,
                "modality": modality,
                "truth_bam": truth_bam,
                "region": args.region,
            })
            emitted_any = True
        if not emitted_any:
            no_modality.append(sample)

    if not entries:
        sys.exit("ERROR: no manifest entries produced (check cohort, "
                 "--modalities, and truth-BAM paths / --only-existing).")

    manifest = {
        "reference_fasta": args.reference_fasta,
        "gfa": args.gfa,
        "region": args.region,
        "ref_mode": args.ref_mode,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    n_present = len(entries) - (0 if args.only_existing else n_missing)
    by_mod: dict[str, int] = {}
    for e in entries:
        by_mod[e["modality"]] = by_mod.get(e["modality"], 0) + 1

    print(f"wrote {args.out}")
    print(f"  samples requested : {len(cohort)}")
    print(f"  entries written   : {len(entries)}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(by_mod.items())))
    print(f"  truth BAMs present: {n_present}  missing: "
          f"{0 if args.only_existing else n_missing}")
    if skipped_samples:
        print(f"  WARNING: {len(skipped_samples)} sample(s) not in master table: "
              f"{', '.join(skipped_samples[:8])}"
              + (" ..." if len(skipped_samples) > 8 else ""))
    if no_modality:
        print(f"  WARNING: {len(no_modality)} sample(s) had none of the requested "
              f"modalities: {', '.join(no_modality[:8])}"
              + (" ..." if len(no_modality) > 8 else ""))
    if n_missing and not args.only_existing:
        print("  note: some truth BAMs do not exist yet — build them in the "
              "data-prep step, e.g.:")
        for ex in missing_examples:
            print(f"        {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
