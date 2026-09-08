#!/usr/bin/env python3
"""Train the standalone AGNES seed classifier on the synthetic dataset.

This trains only the AGNES hybrid seed chainer (classical seeding + EdgeConv GNN
+ confidence-gated DP), independently of the 15M-parameter GraphMamba backbone —
i.e. the paper's method on its own terms.

Examples
--------
    # Quick CPU smoke (tiny preset)
    PYTHONPATH=. python scripts/train_agnes.py --preset tiny --epochs 20 \
        --out data/agnes/agnes_tiny.pt

    # Paper-scale split (640/160/200 reads, 8-10 kb) — use a GPU
    PYTHONPATH=. python scripts/train_agnes.py --preset table1 --epochs 50 \
        --device cuda --out data/agnes/agnes_table1.pt

The checkpoint written to ``--out`` can be loaded with
``graphmambaformer.training.agnes_train.load_agnes_model`` and dropped into the
pipeline via ``AgnesChainer(model=...)`` or ``chaining.chainer="agnes"``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from graphmambaformer.alignment.agnes import AgnesConfig, AgnesSeedClassifier
from graphmambaformer.data.synthetic import generate_dataset, preset
from graphmambaformer.training.agnes_train import (
    AgnesTrainConfig,
    _run_epoch,
    samples_from_records,
    seed_metrics,
    train_agnes,
)
import torch.nn as nn


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="tiny", choices=["tiny", "long", "table1"],
                   help="synthetic data scale (table1 = AGNES 640/160/200 split)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32, help="graphs per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--threads", type=int, default=1,
                   help="cap torch intra-op threads (AGNES graphs are tiny; the "
                        "default 1 avoids oversubscribing cores on shared nodes; "
                        "use 0 to leave torch's default untouched)")
    p.add_argument("--gap-threshold", type=int, default=500,
                   help="max |gap_read - gap_genome| for a seed-graph edge (bp)")
    p.add_argument("--confidence-threshold", type=float, default=0.7,
                   help="use GNN-guided DP only above this confidence")
    p.add_argument("--use-dataset-features", action="store_true",
                   help="train on the dataset's rich 12-D Seed.features (GC/repeat/"
                        "quality) instead of the inference-time geometric features")
    p.add_argument("--out", default="data/agnes/agnes.pt", help="checkpoint path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = AgnesConfig(
        gap_threshold=args.gap_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    train_cfg = AgnesTrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        num_threads=(args.threads if args.threads and args.threads > 0 else None),
    )

    print(f"generating synthetic dataset (preset={args.preset}) ...")
    dataset = generate_dataset(preset(args.preset))
    refs = dataset.references
    splits = dataset.splits

    train_samples = samples_from_records(splits.get("train", []), refs, cfg,
                                         use_dataset_features=args.use_dataset_features)
    val_samples = samples_from_records(splits.get("val", []), refs, cfg,
                                       use_dataset_features=args.use_dataset_features)
    test_samples = samples_from_records(splits.get("test", []), refs, cfg,
                                        use_dataset_features=args.use_dataset_features)

    def seed_total(samples):
        return int(sum(s.labels.size for s in samples))

    def pos_rate(samples):
        if not samples:
            return 0.0
        labels = np.concatenate([s.labels for s in samples])
        return float(labels.mean()) if labels.size else 0.0

    print(
        f"reads: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}  |  "
        f"seeds: train={seed_total(train_samples)} val={seed_total(val_samples)} "
        f"test={seed_total(test_samples)}  |  true-seed rate={pos_rate(train_samples):.2f}"
    )
    if not train_samples or not val_samples:
        print("ERROR: no seed graphs to train on (dataset produced no seeds).")
        return 1

    model, history = train_agnes(train_samples, val_samples, cfg=cfg, train_cfg=train_cfg)

    device = torch.device(train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    # Final test-set report.
    test_metrics = {}
    if test_samples:
        _, labels, probs = _run_epoch(
            model, test_samples, nn.BCEWithLogitsLoss(), device, train_cfg.batch_size, None
        )
        test_metrics = seed_metrics(labels, probs)
        print(
            "TEST seed classification: "
            + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "agnes_config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_metrics": history.val_metrics,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
        },
        "test_metrics": test_metrics,
        "use_dataset_features": args.use_dataset_features,
    }
    torch.save(payload, args.out)
    print(f"checkpoint -> {args.out}  (best epoch {history.best_epoch}, "
          f"val_bce={history.best_val_loss:.4f})")

    # A sidecar JSON of the curves/metrics for quick inspection.
    sidecar = os.path.splitext(args.out)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump({k: payload[k] for k in ("history", "test_metrics", "agnes_config")}, fh, indent=2)
    print(f"metrics  -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
