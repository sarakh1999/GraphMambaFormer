#!/usr/bin/env python3
"""Coordinate-sort and index a BAM with pysam (samtools is not installed here).

longcallD's --refine-aln output is unsorted because reads are re-aligned, so it
needs this pass before anything can fetch by region.

usage: sort_index_bam.py <in.bam> <out.sorted.bam> [threads]
"""

from __future__ import annotations

import os
import sys

import pysam


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    src, dst = argv[1], argv[2]
    threads = argv[3] if len(argv) > 3 else "1"

    if not os.path.exists(src):
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 1

    pysam.sort("-@", str(threads), "-o", dst, src)
    pysam.index("-@", str(threads), dst)
    print(f"sorted -> {dst}")
    print(f"indexed -> {dst}.bai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
