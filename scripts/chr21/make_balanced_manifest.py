#!/usr/bin/env python3
"""Derive a *balanced* pangenome-windows manifest from a plain one.

The plain manifest caps Illumina at sequencing depth 30
(max_reads = ceil(depth * window_bp / read_len)); the long-read modalities are
uncapped and only have ~100-200 reads/window. That ~100x Illumina imbalance
makes the in-RAM dataset assembly huge (millions of short reads) and OOMs the
cache build at serialization.

This rewrites every Illumina entry's max_reads to a lower target depth
(default 0.3, matching the original slice --illumina-depth 0.3 / *.bal.json),
leaving pacbio_hifi / ont entries untouched. It does NOT touch reads, BAMs, or
the graph -- only the per-entry Illumina read cap.
"""
import argparse
import json
import math
import os


def window_bp(region: str) -> int:
    # region looks like "chr21:5000056-5555548"
    span = region.split(":", 1)[1]
    start, end = span.split("-", 1)
    return int(end) - int(start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="plain manifest json")
    ap.add_argument("--out", dest="out", required=True, help="balanced manifest json")
    ap.add_argument("--depth", type=float, default=0.3,
                    help="target Illumina depth (plain uses 30; balanced uses 0.3)")
    ap.add_argument("--read-len", type=int, default=148,
                    help="Illumina read length used by the depth formula (HG005 = 148)")
    args = ap.parse_args()

    with open(args.inp) as fh:
        man = json.load(fh)
    entries = man["entries"] if isinstance(man, dict) else man

    changed = tot_before = tot_after = 0
    for e in entries:
        if (e.get("modality") == "illumina") and ("max_reads" in e):
            bp = window_bp(e["region"])
            new_cap = max(1, math.ceil(args.depth * bp / args.read_len))
            tot_before += e["max_reads"]
            tot_after += new_cap
            e["max_reads"] = new_cap
            changed += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(man, fh)

    print(f"rebalanced {changed} Illumina entries @ depth {args.depth}")
    print(f"Illumina max_reads sum: {tot_before:,} -> {tot_after:,} "
          f"({tot_after / max(1, tot_before):.3f}x)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
