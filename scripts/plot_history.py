#!/usr/bin/env python3
"""Render training plots from a run's ``history.json`` — even a *live* run.

The trainer only redraws its figures at epoch boundaries, so a long run whose
first epoch is tens of thousands of steps shows no plots for a long time. This
script reconstructs the in-memory :class:`TrainHistory` from the on-disk
``history.json`` and calls the same :func:`plot_all`, writing the PNGs into a
directory you choose. Point ``--out`` at a *separate* directory (the default,
``<run>/plots_snapshot``) when the run is still training, so you never collide
with the trainer's own writes into ``<run>/plots``.

It is CPU-only and read-only with respect to the run: it only reads
``history.json`` and writes PNGs under ``--out``.

Examples
--------
    # Snapshot a live run into <run>/plots_snapshot/
    PYTHONPATH=. python scripts/plot_history.py \
        data/training_runs/chr21_hg002_pangenome_windows_balanced/history.json

    # Custom output directory
    PYTHONPATH=. python scripts/plot_history.py path/to/history.json \
        --out /tmp/my_snapshot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphmambaformer.training.metrics import ValidationMetrics
from graphmambaformer.training.plots import plot_all
from graphmambaformer.training.probes import StepReport
from graphmambaformer.training.trainer import TrainHistory


def _only_known_fields(cls, d: dict) -> dict:
    """Keep just the keys that are real dataclass fields of ``cls``.

    Makes reconstruction robust to schema drift: an older or newer history.json
    with extra/renamed keys still loads (unknown keys dropped, missing keys take
    the dataclass default) instead of raising ``TypeError``.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def _load_history(path: str, *, retries: int = 5, delay: float = 0.4) -> TrainHistory:
    """Load and reconstruct a :class:`TrainHistory` from ``history.json``.

    A live trainer rewrites the whole file (open ``"w"`` then ``json.dump``)
    every flush, so a read can momentarily catch a truncated file. Retry a few
    times on a decode error before giving up.
    """
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            with open(path) as fh:
                payload = json.load(fh)
            break
        except (json.JSONDecodeError, ValueError) as exc:  # partial write
            last_err = exc
            time.sleep(delay)
    else:
        raise SystemExit(
            f"could not read a complete JSON from {path} after {retries} tries "
            f"(last error: {last_err}). The run may be mid-flush; try again."
        )

    steps = [
        StepReport(**_only_known_fields(StepReport, s))
        for s in payload.get("steps", [])
    ]
    validations = [
        ValidationMetrics(**_only_known_fields(ValidationMetrics, v))
        for v in payload.get("validations", [])
    ]
    return TrainHistory(
        steps=steps,
        epochs=payload.get("epochs", []),
        validations=validations,
        device_summary=payload.get("device", ""),
        config=payload.get("config", {}),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", help="path to a run's history.json")
    p.add_argument("--out", default=None,
                   help="output directory for the PNGs (default: "
                        "<run>/plots_snapshot, a sibling of history.json so it "
                        "never collides with a live trainer's own plots/ dir)")
    args = p.parse_args()

    if not os.path.exists(args.history):
        raise SystemExit(f"no such file: {args.history}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.history)),
                                   "plots_snapshot")

    history = _load_history(args.history)
    print(f"loaded {len(history.steps)} step records, "
          f"{len(history.validations)} validation records")
    written = plot_all(history, out, verbose=True)
    if not written:
        print("no figures written (no plottable records yet, or matplotlib "
              "missing)")
        return
    print(f"\nwrote {len(written)} figure(s) to {out}:")
    for path in written:
        print(f"   {path}")


if __name__ == "__main__":
    main()
