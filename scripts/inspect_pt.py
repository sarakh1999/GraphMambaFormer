"""Inspect ``.pt`` files and print a readable summary of what's inside.

Handles both kinds of ``.pt`` in this repo:
  * the per-stage artifact dicts in ``data/stage_outputs/`` (tensors + stats), and
  * dataset bundles like ``data/small_samples/synthetic_small.pt`` (a
    ``SyntheticDataset`` object).

For every tensor it prints shape / dtype / mean / std / min / max plus a small
flattened value preview; for dicts it recurses; for dataset objects it prints a
compact record summary.

Usage
-----
    # default: inspect every .pt in data/stage_outputs/
    PYTHONPATH=. .venv/bin/python scripts/inspect_pt.py

    # inspect specific files / globs
    PYTHONPATH=. .venv/bin/python scripts/inspect_pt.py data/small_samples/synthetic_small.pt
    PYTHONPATH=. .venv/bin/python scripts/inspect_pt.py "data/stage_outputs/07_*.pt"
"""

from __future__ import annotations

import glob
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO, "data", "stage_outputs")
PREVIEW = 6  # how many leading values to show from each tensor


def _fmt_tensor(t: torch.Tensor) -> str:
    tf = t.float()
    flat = tf.flatten()
    head = [round(v, 4) for v in flat[:PREVIEW].tolist()]
    ell = " ..." if flat.numel() > PREVIEW else ""
    return (
        f"Tensor shape={list(t.shape)} dtype={t.dtype} "
        f"mean={tf.mean().item():.4f} std={tf.std().item():.4f} "
        f"min={tf.min().item():.4f} max={tf.max().item():.4f} "
        f"finite={bool(torch.isfinite(tf).all())}\n"
        f"{' ' * 8}values[:{PREVIEW}]={head}{ell}"
    )


def _show(value: object, indent: int = 2) -> None:
    pad = " " * indent
    if isinstance(value, torch.Tensor):
        print(pad + _fmt_tensor(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, torch.Tensor):
                print(f"{pad}{k}: {_fmt_tensor(v)}")
            elif isinstance(v, dict):
                print(f"{pad}{k}: (dict)")
                _show(v, indent + 4)
            else:
                print(f"{pad}{k}: {v!r}")
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{type(value).__name__} len={len(value)}: {value[:PREVIEW]!r}"
              + (" ..." if len(value) > PREVIEW else ""))
    else:
        # dataset object or other — show attributes if present
        summarize_object(value, indent)


def summarize_object(obj: object, indent: int = 2) -> None:
    pad = " " * indent
    # SyntheticDataset-like: has .splits / .references / .config
    if hasattr(obj, "splits") and hasattr(obj, "references"):
        print(f"{pad}{type(obj).__name__}: "
              f"{len(getattr(obj, 'references'))} reference(s), "
              f"splits={{"
              + ", ".join(f"{k}:{len(v)}" for k, v in obj.splits.items()) + "}")
        for split, recs in obj.splits.items():
            if not recs:
                continue
            r = recs[0]
            cigar = str(getattr(r, "cigar_string", getattr(r, "cigar", "?")))[:40]
            print(f"{pad}  e.g. {split}[0]: id={getattr(r, 'read_id', '?')} "
                  f"modality={getattr(r, 'modality', '?')} "
                  f"len={len(getattr(r, 'seq', ''))} "
                  f"cigar={cigar} "
                  f"seeds={len(getattr(r, 'seeds', []))}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj!r}")


def inspect_file(path: str) -> None:
    print("=" * 78)
    print(os.path.relpath(path, REPO), f"({os.path.getsize(path):,} bytes)")
    print("=" * 78)
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, dict):
        print(f"  dict with {len(obj)} key(s): {list(obj.keys())}")
        _show(obj, indent=2)
    else:
        _show(obj, indent=2)
    print()


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths: list[str] = []
        for a in args:
            paths.extend(sorted(glob.glob(a)) or [a])
    else:
        paths = sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.pt")))

    if not paths:
        print("no .pt files found. pass a path or run make_small_samples.py / "
              "test_pipeline_stages.py first.")
        raise SystemExit(1)

    for p in paths:
        if not os.path.exists(p):
            print(f"[skip] not found: {p}")
            continue
        inspect_file(p)


if __name__ == "__main__":
    main()
