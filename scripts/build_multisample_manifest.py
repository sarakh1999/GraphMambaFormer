#!/usr/bin/env python3
"""Build train / test manifests for ONE universal aligner over many samples.

The goal this supports is: *train a single model* across all samples and
modalities, evaluate it so we can say "matches the baseline on easy reads, wins
(or draws) on the hard fraction and on MAPQ calibration, on graph references."

Two manifests are emitted (the schema ``scripts/train.py`` / ``scripts/eval.py``
already consume: ``{"entries": [...], <global defaults>}``):

* ``train_manifest.json`` — one entry per (train-sample, modality). Feeds
  ``train.py --manifest`` for a single pooled, modality-balanced run.
* ``test_manifest.json``  — one entry per (held-out-sample, modality). Feeds
  ``eval.py --manifest`` for the final stratified report. **Held-out samples
  never appear in training**, which is what keeps the "match/win" claim honest
  (no train/test leakage across samples).

Paths are resolved from templates so this works for the HG002/HG005 chr21
bundles today and the 47 HPRC samples later without code changes::

    reads : --reads-pattern   e.g. data/hprc/reads/{sample}/{moddir}/
    truth : --truth-pattern   e.g. data/{sample}/bam/{sample}.{region_tag}.giraffe.sorted.bam

Placeholders: ``{sample} {modality} {moddir} {region} {region_tag}``.
``{moddir}`` maps modality→HPRC subdir (illumina/hifi/ont); ``{region_tag}`` is
the region with ':'/'-' replaced by '_' (BAM-name friendly).

An entry is written only when its truth BAM exists (unless ``--allow-pseudo``,
which omits the missing ``truth_bam`` so training falls back to classical
pseudo-labels). Missing reads/truth are reported, never guessed.

Example (HG005 chr21, all three modalities; hold out nothing yet)::

    python scripts/build_multisample_manifest.py \
        --reference-fasta data/chr21/HG005/ref/GRCh38.chr21.fa \
        --gfa data/chr21/HG005/chr21.gfa \
        --region chr21 \
        --train-samples HG005 \
        --modalities illumina,pacbio_hifi,ont \
        --truth-pattern 'data/chr21/{sample}/bam/{sample}.{region_tag}.giraffe.sorted.bam' \
        --reads-pattern  'data/chr21/{sample}/reads/{sample}.{region_tag}.{modality}.R1.fastq.gz' \
        --out-dir data/manifests/hg005_chr21

Example (47 HPRC samples, hold out 5 for test)::

    python scripts/build_multisample_manifest.py \
        --reference-fasta data/chr21/HG002/ref/GRCh38.chr21.fa \
        --gfa data/chr21/HG002/chr21.gfa --region chr21 \
        --train-samples @data/hprc/train_samples.txt \
        --test-samples  @data/hprc/test_samples.txt \
        --modalities illumina,pacbio_hifi,ont \
        --truth-pattern 'data/hprc/bam/{sample}.{region_tag}.{modality}.giraffe.sorted.bam' \
        --reads-root data/hprc/reads \
        --out-dir data/manifests/hprc47
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# modality -> HPRC on-disk subdir (mirrors data/real_data.py HPRC_MODALITY_DIRS)
MODDIR = {
    "illumina": "illumina",
    "pacbio_hifi": "hifi",
    "hifi": "hifi",
    "ont": "ont",
}
CANON = {"hifi": "pacbio_hifi", "pacbio_hifi": "pacbio_hifi",
         "illumina": "illumina", "ont": "ont"}


def _read_sample_list(spec: str) -> list[str]:
    """Comma list, or ``@file`` with one sample per line (``#`` comments ok)."""
    if not spec:
        return []
    if spec.startswith("@"):
        path = spec[1:]
        if not os.path.exists(path):
            sys.exit(f"ERROR: sample list file not found: {path}")
        out = []
        with open(path) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    out.append(line)
        return out
    return [s.strip() for s in spec.split(",") if s.strip()]


def _fill(template: str, *, sample: str, modality: str, region: str) -> str:
    canon = CANON.get(modality, modality)
    region_tag = (region or "full").replace(":", "_").replace("-", "_").replace(",", "")
    return template.format(
        sample=sample,
        modality=canon,
        moddir=MODDIR.get(canon, canon),
        region=region or "",
        region_tag=region_tag,
    )


def _resolve_reads(args, sample: str, modality: str) -> list[str] | None:
    """Return existing read files for (sample, modality), or None if none."""
    canon = CANON.get(modality, modality)
    if args.reads_pattern:
        # A pattern may expand to R1/R2 for Illumina: fill, then glob siblings.
        base = _fill(args.reads_pattern, sample=sample, modality=modality,
                     region=args.region)
        hits = sorted(glob.glob(base))
        # If the R1 template was given, also pick up the matching R2.
        if hits and "R1" in base:
            r2 = base.replace("R1", "R2")
            hits += [p for p in sorted(glob.glob(r2)) if p not in hits]
        return hits or None
    # Otherwise use the HPRC reads-root layout: reads_root/<sample>/<moddir>/*
    moddir = MODDIR.get(canon, canon)
    folder = os.path.join(args.reads_root, sample, moddir)
    if not os.path.isdir(folder):
        return None
    hits: list[str] = []
    for ext in ("*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq", "*.cram", "*.bam"):
        hits.extend(sorted(glob.glob(os.path.join(folder, ext))))
    return hits or None


def _build_entries(args, samples: list[str], which: str) -> list[dict]:
    entries: list[dict] = []
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    missing_truth, missing_reads = [], []
    for sample in samples:
        for modality in modalities:
            canon = CANON.get(modality, modality)
            truth = _fill(args.truth_pattern, sample=sample, modality=modality,
                          region=args.region) if args.truth_pattern else None
            has_truth = bool(truth and os.path.exists(truth))
            reads = _resolve_reads(args, sample, modality)
            tag = f"{sample}/{canon}"

            if not has_truth and not args.allow_pseudo:
                missing_truth.append((tag, truth))
                continue
            if not has_truth and not reads:
                missing_reads.append(tag)
                continue

            entry: dict = {"sample": sample, "modality": canon}
            if has_truth:
                entry["truth_bam"] = os.path.abspath(truth)
            if reads:
                # train.py/eval.py accept a single read file per entry via
                # reads_file; store the R1 (R2 auto-paired) or the lone file.
                entry["reads_file"] = os.path.abspath(reads[0])
                if len(reads) > 1:
                    entry["reads_file_2"] = os.path.abspath(reads[1])
            if args.max_reads:
                entry["max_reads"] = args.max_reads
            entries.append(entry)

    if missing_truth:
        print(f"[{which}] skipped {len(missing_truth)} (sample,modality) with no "
              f"truth BAM (use --allow-pseudo to train them on classical "
              f"pseudo-labels):", file=sys.stderr)
        for tag, path in missing_truth[:10]:
            print(f"    {tag}: {path}", file=sys.stderr)
    if missing_reads:
        print(f"[{which}] skipped {len(missing_reads)} (sample,modality) with "
              f"neither truth nor reads: {', '.join(missing_reads[:10])}",
              file=sys.stderr)
    return entries


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-fasta", required=True)
    p.add_argument("--gfa", default=None,
                   help="pangenome GFA (required for --ref-mode pangenome/both "
                        "at train/eval time; recorded as a manifest global)")
    p.add_argument("--region", default=None,
                   help="samtools-style window shared by all entries (optional)")
    p.add_argument("--contig", default=None)
    p.add_argument("--modalities", default="illumina,pacbio_hifi,ont")
    p.add_argument("--train-samples", required=True,
                   help="comma list or @file (one sample per line)")
    p.add_argument("--test-samples", default="",
                   help="held-out samples for the test manifest (comma or @file). "
                        "MUST be disjoint from --train-samples.")
    p.add_argument("--truth-pattern", default=None,
                   help="template for the per-(sample,modality) truth BAM; "
                        "placeholders {sample} {modality} {moddir} {region} "
                        "{region_tag}")
    p.add_argument("--reads-pattern", default=None,
                   help="template for reads (R1); {..} placeholders as above. "
                        "If omitted, --reads-root HPRC layout is used")
    p.add_argument("--reads-root", default="data/hprc/reads",
                   help="HPRC reads root (reads_root/<sample>/<moddir>/*) used "
                        "when --reads-pattern is not given")
    p.add_argument("--allow-pseudo", action="store_true",
                   help="emit entries even without a truth BAM (train.py then "
                        "builds classical pseudo-labels from the reads)")
    p.add_argument("--max-reads", type=int, default=0,
                   help="per-entry read cap (0 = all); applied as a manifest "
                        "global and per entry")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    train_samples = _read_sample_list(args.train_samples)
    test_samples = _read_sample_list(args.test_samples)
    if not train_samples:
        sys.exit("ERROR: no train samples resolved from --train-samples")
    overlap = set(train_samples) & set(test_samples)
    if overlap:
        sys.exit(f"ERROR: train/test sample leakage — these appear in both: "
                 f"{sorted(overlap)}. Held-out samples must be disjoint.")
    if args.truth_pattern is None and not args.allow_pseudo:
        sys.exit("ERROR: give --truth-pattern (preferred) or --allow-pseudo")

    os.makedirs(args.out_dir, exist_ok=True)

    def _globals() -> dict:
        g: dict = {"reference_fasta": os.path.abspath(args.reference_fasta)}
        if args.gfa:
            g["gfa"] = os.path.abspath(args.gfa)
        if args.region:
            g["region"] = args.region
        if args.contig:
            g["contig"] = args.contig
        if args.max_reads:
            g["max_reads"] = args.max_reads
        return g

    written = {}
    for which, samples in (("train", train_samples), ("test", test_samples)):
        if not samples:
            continue
        entries = _build_entries(args, samples, which)
        if not entries:
            print(f"[{which}] no entries built — check patterns/paths",
                  file=sys.stderr)
        manifest = {**_globals(), "entries": entries}
        path = os.path.join(args.out_dir, f"{which}_manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        written[which] = (path, len(entries))
        n_mod = len({e["modality"] for e in entries})
        n_smp = len({e["sample"] for e in entries})
        print(f"[{which}] {len(entries)} entries "
              f"({n_smp} samples x {n_mod} modalities) -> {path}")

    # Print the exact commands to train one universal model + evaluate it.
    train_path = written.get("train", (None,))[0]
    test_path = written.get("test", (None,))[0]
    ref_mode = "both" if args.gfa else "linear"
    print("\n# ---- train ONE universal model over all entries ----")
    print(
        "PYTHONPATH=. python scripts/train.py --data real \\\n"
        f"  --manifest {train_path} \\\n"
        f"  --ref-mode {ref_mode} \\\n"
        "  --balance-modalities --modality-loss-weight \\\n"
        "  --monitor macro_locus_accuracy --min-lr-frac 0.05 \\\n"
        "  --device cuda --require-gpu --devices auto \\\n"
        "  --epochs 30 --batch-size 8 --d-model 256 --workers 16 --prefetch 3 \\\n"
        "  --out data/training_runs/universal"
    )
    if test_path:
        print("\n# ---- evaluate on HELD-OUT samples (stratified easy/hard + "
              "MAPQ calibration) ----")
        print(
            "PYTHONPATH=. python scripts/eval.py --data real \\\n"
            f"  --manifest {test_path} \\\n"
            f"  --ref-mode {ref_mode} --mode hybrid \\\n"
            "  --checkpoint data/training_runs/universal/checkpoint.pt \\\n"
            "  --hard-mapq-threshold 20 --max-batches 80 \\\n"
            "  --device cuda --out data/eval_runs/universal_heldout"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
