#!/usr/bin/env python3
"""Re-chunk a prebuilt dataset cache to a different batch size, cheaply.

The prebuilt-dataset cache stores ``(reads, reference_index)`` batches whose
grouping is fixed at build time by ``--batch-size`` (part of the cache key). The
reads and per-window reference indexes themselves are batch-size independent, so
a cache built for one batch size can be re-chunked into another WITHOUT re-reading
the BAMs or rebuilding the graph/FM indexes (the ~4.5h "build manifest entries"
phase).

This regroups the reads per reference (batches for the same window are pooled and
re-chunked), so every emitted batch still carries a single reference — the
invariant the trainer relies on. Train batches are reshuffled with a fixed seed
to interleave windows/modalities across an epoch, matching a fresh build.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/transfer/regroup_dataset_cache.py \
        --src /fs/ess/PCS0289/mambaformer_cache_cpu/chr21_HG005_pangenome_windows.bal.1d1443d1290bcb29.pt \
        --dst /fs/ess/PCS0289/mambaformer_cache_cpu_bs8/chr21_HG005_pangenome_windows.bal.7249afbb7ce2ba62.pt \
        --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch


def _regroup(batches, batch_size, shuffle_seed=None):
    """Pool reads per reference (by object identity) and re-chunk to batch_size.

    Reference objects are shared across batches (built once per window), so
    grouping by ``id(ref)`` keeps identical windows together and never mixes
    reads from different references into one batch.
    """
    order = []            # references in first-seen order (stable output)
    reads_by_ref = {}     # id(ref) -> [ref, [reads...]]
    for reads, ref in batches:
        key = id(ref)
        if key not in reads_by_ref:
            reads_by_ref[key] = [ref, []]
            order.append(key)
        reads_by_ref[key][1].extend(reads)

    out = []
    for key in order:
        ref, reads = reads_by_ref[key]
        for i in range(0, len(reads), batch_size):
            out.append((list(reads[i : i + batch_size]), ref))

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(out)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="source cache .pt (any batch size)")
    p.add_argument("--dst", required=True, help="destination cache .pt (target batch size)")
    p.add_argument("--batch-size", type=int, required=True, help="target batch size")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="seed for the train-batch interleave shuffle (matches a fresh build)")
    args = p.parse_args()

    if os.path.exists(args.dst):
        print(f"destination already exists, refusing to overwrite: {args.dst}")
        return 0

    t0 = time.time()
    print(f"loading source cache: {args.src}")
    payload = torch.load(args.src, weights_only=False)
    train = payload["train"]
    val = payload["val"]
    data_info = payload.get("data_info", {})
    print(f"  loaded {len(train)} train / {len(val)} val batches in {time.time() - t0:.1f}s")

    n_train_reads = sum(len(r) for r, _ in train)
    n_val_reads = sum(len(r) for r, _ in val)

    new_train = _regroup(train, args.batch_size, shuffle_seed=args.shuffle_seed)
    new_val = _regroup(val, args.batch_size, shuffle_seed=None)  # val order is irrelevant

    # Read/reference invariants must be preserved exactly — only the grouping changes.
    assert sum(len(r) for r, _ in new_train) == n_train_reads, "train read count changed"
    assert sum(len(r) for r, _ in new_val) == n_val_reads, "val read count changed"
    assert all(len(r) <= args.batch_size for r, _ in new_train), "oversized train batch"
    assert all(len(r) <= args.batch_size for r, _ in new_val), "oversized val batch"

    print(f"  regrouped -> {len(new_train)} train / {len(new_val)} val batches "
          f"(batch-size {args.batch_size}); reads unchanged "
          f"({n_train_reads} train / {n_val_reads} val)")

    if isinstance(data_info, dict):
        data_info = dict(data_info)
        data_info["regrouped_from"] = os.path.abspath(args.src)
        data_info["regrouped_batch_size"] = args.batch_size

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    tmp = f"{args.dst}.tmp.{os.getpid()}"
    t1 = time.time()
    print(f"writing destination cache: {args.dst}")
    try:
        torch.save({"train": new_train, "val": new_val, "data_info": data_info}, tmp)
        os.replace(tmp, args.dst)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    size_gb = os.path.getsize(args.dst) / 1e9
    print(f"  wrote {size_gb:.2f} GB in {time.time() - t1:.1f}s")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
