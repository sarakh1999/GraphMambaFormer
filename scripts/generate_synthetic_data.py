"""Generate a synthetic GraphMambaFormer alignment dataset and save it to disk.

Examples
--------
    # tiny CPU dataset (default) -> data/synthetic_tiny.pt
    PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py

    # paper-scale 8-10 kb reads (a handful of samples)
    PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py --preset long

    # exact AGNES split (640/160/200) with every modality
    PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py --preset table1

The dataset is saved as a ``.pt`` (torch.save) file and, optionally, a small
human-readable summary is printed. Reload with
``graphmambaformer.data.load_dataset(path)``.
"""

from __future__ import annotations

import argparse
import os

from graphmambaformer.data import (
    export_dataset,
    generate_dataset,
    preset,
    save_dataset,
)
from graphmambaformer.data.synthetic import cigar_consumed, verify_read_alignment
from graphmambaformer.progress import progress


def summarize(dataset) -> None:
    cfg = dataset.config
    print("=" * 68)
    print("Synthetic GraphMambaFormer dataset")
    print("=" * 68)
    print(f"seed={cfg.seed}  k={cfg.kmer}  w={cfg.window}  "
          f"reads={cfg.read_len_min}-{cfg.read_len_max} bp")
    for rid, ref in dataset.references.items():
        print(f"  ref {rid}: len={len(ref.seq):,}  GC={ref.gc_content:.3f}  "
              f"repeat={ref.repeat_content:.3f}  graph: "
              f"{len(ref.graph.node_seqs)} nodes / {len(ref.graph.edge_index)} edges")
    print("-" * 68)
    print(f"{'split':<10}{'reads':>7}{'avg_len':>9}{'avg_true':>10}"
          f"{'avg_false%':>12}{'avg_mapq':>10}")
    for split, recs in dataset.splits.items():
        if not recs:
            continue
        n = len(recs)
        avg_len = sum(len(r.seq) for r in recs) / n
        avg_true = sum(r.num_true_seeds for r in recs) / n
        fracs = [r.num_false_seeds / max(1, len(r.seeds)) for r in recs]
        avg_false = 100 * sum(fracs) / n
        avg_mapq = sum(r.mapq for r in recs) / n
        print(f"{split:<10}{n:>7}{avg_len:>9.0f}{avg_true:>10.1f}"
              f"{avg_false:>12.1f}{avg_mapq:>10.1f}")
    if "edge" in dataset.splits:
        print("-" * 68)
        print("edge cases:", ", ".join(r.edge_case for r in dataset.splits["edge"]))


def integrity_check(dataset) -> int:
    problems = 0
    for recs in dataset.splits.values():
        for r in progress(recs, total=len(recs), desc="integrity check",
                          unit="read", leave=False):
            issues = verify_read_alignment(r, dataset.references[r.ref_id])
            for msg in issues:
                problems += 1
                print(f"  [!] {r.read_id}: {msg}")
    print("-" * 68)
    print(f"integrity: {problems} problem(s) found")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny", choices=["tiny", "long", "table1"],
                    help="dataset size/scale preset (default: tiny)")
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed")
    ap.add_argument("--out", default=None, help="output .pt path")
    ap.add_argument("--all-modalities", action="store_true",
                    help="add one specialised read per extra modality")
    ap.add_argument("--no-edge-cases", action="store_true",
                    help="skip the injected edge-case reads")
    ap.add_argument("--no-pt", action="store_true",
                    help="do not write the torch .pt bundle")
    ap.add_argument("--emit-dir", default=None,
                    help="also export standard genomics files (FASTA/FASTQ/SAM/GFA/JSON) "
                         "into this directory")
    ap.add_argument("--to-bam", action="store_true",
                    help="convert truth.sam -> sorted, indexed BAM (requires samtools)")
    args = ap.parse_args()

    cfg = preset(args.preset)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.all_modalities:
        cfg.include_all_modalities = True
    if args.no_edge_cases:
        cfg.include_edge_cases = False

    dataset = generate_dataset(cfg)
    summarize(dataset)
    integrity_check(dataset)

    print("-" * 68)
    if not args.no_pt:
        out = args.out or os.path.join("data", f"synthetic_{args.preset}.pt")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        save_dataset(dataset, out)
        print(f"saved torch bundle -> {out}")

    if args.emit_dir:
        paths = export_dataset(dataset, args.emit_dir, to_bam=args.to_bam)
        for name, path in paths.items():
            print(f"saved {name:<7} -> {path}")
        if args.to_bam and "bam" not in paths:
            print("note: samtools not found; wrote SAM only "
                  "(install samtools to get BAM). Convert later with:")
            print(f"      samtools sort -o {os.path.join(args.emit_dir, 'truth.bam')} "
                  f"{paths['sam']} && samtools index {os.path.join(args.emit_dir, 'truth.bam')}")


if __name__ == "__main__":
    main()
