#!/usr/bin/env python3
"""Train the GraphMamba alignment model, validate it, and plot everything.

Runs on any GPU (NVIDIA of any generation, AMD/ROCm, Intel XPU), on Apple MPS,
or on CPU -- the device, autocast dtype and kernel tier are all chosen by
``AccelContext``, and the chosen tier is printed so a CPU run is never mistaken
for a GPU one.

    # quick CPU smoke run on synthetic data
    PYTHONPATH=. python scripts/train.py --preset tiny --epochs 3

    # longer run, explicit device, plots to a named directory
    PYTHONPATH=. python scripts/train.py --preset small --epochs 30 \
        --device cuda --out data/training_runs/chr1

Every step prints the model's behavior, not just its loss: per-term losses,
gradient norm, and the router's split across compute paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from graphmambaformer.accel import AccelContext
from graphmambaformer.alignment.pipeline import build_pipeline
from graphmambaformer.config import (
    AccelConfig,
    CoreModelConfig,
    GraphMambaConfig,
    LossConfig,
    PipelineConfig,
)
from graphmambaformer.data.synthetic import generate_dataset, preset
from graphmambaformer.models import build_core_model
from graphmambaformer.training import TrainConfig, Trainer, plot_all


def chunk(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="tiny",
                   choices=("tiny", "long", "table1"),
                   help="synthetic dataset preset")
    p.add_argument("--reads", type=int, default=0,
                   help="override the preset's read count (0 = use the preset). "
                        "The stock presets are too small for stable metrics.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=64,
                   help="small by default so a CPU run finishes quickly")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--monitor", default="locus_accuracy",
                   help="early-stopping metric (falls back to -val_loss if the "
                        "chosen metric is unmeasurable on this data)")
    p.add_argument("--out", default="data/training_runs/latest")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)

    print("=" * 74)
    print("Building dataset and model")
    print("=" * 74)
    spec = preset(args.preset)
    if args.reads:
        # The stock presets ship a handful of reads, which makes every validation
        # metric noise. Scale the splits up so the numbers mean something.
        spec = replace(spec, n_train=args.reads, n_val=max(2, args.reads // 4),
                       n_test=max(2, args.reads // 4))
    ds = generate_dataset(spec)

    accel = AccelContext(AccelConfig(device=args.device))
    model_cfg = GraphMambaConfig(d_model=args.d_model)
    model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=model_cfg)
    ).model
    pipeline = build_pipeline(
        PipelineConfig(mode="hybrid", batch_size=args.batch_size), model=model
    )

    # One index per reference, built once and shared by every batch on it. Using
    # all references rather than just the first keeps the whole dataset in play.
    references = {
        ref_id: pipeline.build_reference(ref.seq, ref_id=ref_id)
        for ref_id, ref in sorted(ds.references.items())
    }

    def batches_for(split: str) -> list[tuple]:
        out: list[tuple] = []
        by_ref: dict[int, list] = {}
        for read in ds.splits.get(split, []):
            if read.ref_id in references:
                by_ref.setdefault(read.ref_id, []).append(read)
        for ref_id, reads in sorted(by_ref.items()):
            for group in chunk(reads, args.batch_size):
                out.append((group, references[ref_id]))
        return out

    train_batches = batches_for("train")
    val_batches = batches_for("val") or batches_for("test")
    n_train = sum(len(r) for r, _ in train_batches)
    n_val = sum(len(r) for r, _ in val_batches)
    if not n_val:
        # No held-out split at all: peel the tail off train rather than
        # validating on the training reads.
        cut = max(1, int(len(train_batches) * 0.8))
        train_batches, val_batches = train_batches[:cut], train_batches[cut:]
        n_train = sum(len(r) for r, _ in train_batches)
        n_val = sum(len(r) for r, _ in val_batches)

    print(f"references: {len(references)} "
          f"({', '.join(f'{len(r.seq):,}bp' for _, r in sorted(ds.references.items()))})")
    print(f"reads: {n_train} train / {n_val} val   "
          f"batches: {len(train_batches)} train / {len(val_batches)} val")

    print()
    print("=" * 74)
    print("Training")
    print("=" * 74)
    trainer = Trainer(
        model, pipeline,
        cfg=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, monitor=args.monitor, out_dir=args.out,
        ),
        loss_cfg=LossConfig(),
        accel=accel,
        verbose=not args.quiet,
    )
    history = trainer.fit(train_batches, val_batches)

    json_path = history.to_json(os.path.join(args.out, "history.json"))
    print(f"\nhistory -> {json_path}")

    if not args.no_plots:
        print()
        plot_all(history, os.path.join(args.out, "plots"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
