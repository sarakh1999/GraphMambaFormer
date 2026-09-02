#!/usr/bin/env python3
"""Derive paired inference FASTQ (R1/R2, +orphan single-end) from a local BAM.

No Docker / samtools binary needed — pure pysam. Reads are emitted in original
sequencing orientation (reverse-strand alignments are reverse-complemented back),
matching `samtools fastq`. Full pairs go to R1/R2; unpaired mates go to the
single-end file so nothing is silently dropped.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys

import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphmambaformer.progress import progress

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _seq_qual(rec) -> tuple[str, str]:
    seq = rec.query_sequence or ""
    q = rec.query_qualities
    qual = "".join(chr(x + 33) for x in q) if q is not None else "I" * len(seq)
    if rec.is_reverse:
        seq = _revcomp(seq)
        qual = qual[::-1]
    return seq, qual


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", default="data/chr21/HG002/bam/HG002.chr21.giab.sorted.bam")
    ap.add_argument("--out-dir", default="data/chr21/HG002/reads")
    ap.add_argument("--prefix", default="HG002.chr21")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    r1_path = os.path.join(args.out_dir, f"{args.prefix}.R1.fastq.gz")
    r2_path = os.path.join(args.out_dir, f"{args.prefix}.R2.fastq.gz")
    se_path = os.path.join(args.out_dir, f"{args.prefix}.se.fastq.gz")

    bam = pysam.AlignmentFile(args.bam)
    mates: dict[str, dict] = {}
    singles: list = []
    for rec in progress(bam.fetch(until_eof=True), desc="scan BAM", unit="aln"):
        if rec.is_secondary or rec.is_supplementary or rec.is_unmapped:
            continue
        if not rec.is_paired:
            singles.append(rec)
            continue
        slot = mates.setdefault(rec.query_name, {})
        slot[1 if rec.is_read1 else 2] = rec

    n_pairs = n_orphan = 0
    with gzip.open(r1_path, "wt") as f1, gzip.open(r2_path, "wt") as f2, gzip.open(se_path, "wt") as fs:
        for name, slot in progress(mates.items(), total=len(mates), desc="write pairs", unit="pair"):
            if 1 in slot and 2 in slot:
                for rec, fh, tag in ((slot[1], f1, "/1"), (slot[2], f2, "/2")):
                    seq, qual = _seq_qual(rec)
                    fh.write(f"@{name}{tag}\n{seq}\n+\n{qual}\n")
                n_pairs += 1
            else:
                rec = slot.get(1) or slot.get(2)
                seq, qual = _seq_qual(rec)
                fs.write(f"@{name}\n{seq}\n+\n{qual}\n")
                n_orphan += 1
        for rec in progress(singles, desc="write singles", unit="read"):
            seq, qual = _seq_qual(rec)
            fs.write(f"@{rec.query_name}\n{seq}\n+\n{qual}\n")
            n_orphan += 1

    print(f"pairs={n_pairs} single_end={n_orphan}")
    print(f"R1 -> {r1_path}")
    print(f"R2 -> {r2_path}")
    print(f"SE -> {se_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
