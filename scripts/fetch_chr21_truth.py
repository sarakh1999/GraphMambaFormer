#!/usr/bin/env python3
"""Stream a chr21 window of real HG002 Illumina reads from the remote GIAB
NovoAlign BAM into a small local sorted+indexed truth BAM (no Docker needed).

The remote BAM is coordinate-sorted, so reads arrive in order and the output is
already sorted. We keep mapped primary alignments in the window up to a cap.
"""
from __future__ import annotations

import argparse
import os
import sys

import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphmambaformer.progress import progress

REMOTE = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/"
    "AshkenazimTrio/HG002_NA24385_son/NIST_Illumina_2x250bps/"
    "novoalign_bams/HG002.GRCh38.2x250.bam"
)


def _keep(rec) -> bool:
    return not (rec.is_unmapped or rec.is_secondary or rec.is_supplementary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="chr21")
    ap.add_argument("--start", type=int, default=5_000_000)
    ap.add_argument("--end", type=int, default=6_000_000)
    ap.add_argument("--max-reads", type=int, default=40_000)
    ap.add_argument("--tiles", type=int, default=1,
                    help="spread the sample across N evenly-spaced sub-windows "
                         "over [start,end] so reads cover the whole span "
                         "(genome-wide sampling); 1 = one contiguous window")
    ap.add_argument("--out", default="data/chr21/HG002/bam/HG002.chr21.giab.sorted.bam")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    src = pysam.AlignmentFile(REMOTE, "rb")
    out = pysam.AlignmentFile(args.out, "wb", template=src)

    kept = 0
    seen = 0
    if args.tiles <= 1:
        for rec in progress(src.fetch(args.contig, args.start, args.end),
                            total=args.max_reads, desc="fetch truth", unit="read"):
            seen += 1
            if not _keep(rec):
                continue
            out.write(rec)
            kept += 1
            if kept >= args.max_reads:
                break
    else:
        # Tiled genome-wide sampling: visit N evenly-spaced anchor positions in
        # increasing coordinate order (keeps the output coordinate-sorted) and
        # take up to per_tile primary reads starting at each anchor.
        per_tile = max(1, args.max_reads // args.tiles)
        step = max(1, (args.end - args.start) // args.tiles)
        for t in progress(range(args.tiles), desc="fetch truth tiles", unit="tile"):
            anchor = args.start + t * step
            got = 0
            for rec in src.fetch(args.contig, anchor, min(anchor + step, args.end)):
                seen += 1
                if not _keep(rec):
                    continue
                out.write(rec)
                kept += 1
                got += 1
                if got >= per_tile or kept >= args.max_reads:
                    break
            if kept >= args.max_reads:
                break
    out.close()
    src.close()

    pysam.index(args.out)
    print(f"scanned={seen} kept={kept} tiles={args.tiles} -> {args.out}")
    print(f"index -> {args.out}.bai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
