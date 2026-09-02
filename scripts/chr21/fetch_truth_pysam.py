#!/usr/bin/env python3
"""Fetch GIAB HG002 benchmark VCF/BED and subset to a contig, Docker-free.

Uses htslib (via pysam) remote random access on the indexed benchmark VCF so we
stream only the target contig instead of the whole-genome file. The benchmark
BED is small enough to download whole and filter by contig.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import pysam

BASE = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38"
)
VCF = "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
BED = "HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="chr21")
    ap.add_argument("--out-dir", default="data/chr21/HG002/truth")
    ap.add_argument("--prefix", default="HG002.chr21")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_vcf = os.path.join(args.out_dir, f"{args.prefix}.benchmark.vcf.gz")
    out_bed = os.path.join(args.out_dir, f"{args.prefix}.benchmark.bed")

    url = f"{BASE}/{VCF}"
    print(f"Opening remote VCF (random access): {url}")
    vf = pysam.VariantFile(url)  # htslib fetches the remote .tbi for indexed access
    contig = args.contig
    if contig not in vf.header.contigs:
        alt = contig[3:] if contig.startswith("chr") else f"chr{contig}"
        if alt in vf.header.contigs:
            contig = alt
        else:
            raise SystemExit(
                f"contig {args.contig!r} not in VCF header; "
                f"available e.g. {list(vf.header.contigs)[:5]}"
            )

    n = 0
    tmp = out_vcf[:-3]  # write plain VCF, then bgzip+index
    with pysam.VariantFile(tmp, "w", header=vf.header) as out:
        for rec in vf.fetch(contig):
            out.write(rec)
            n += 1
    pysam.tabix_compress(tmp, out_vcf, force=True)
    os.remove(tmp)
    pysam.tabix_index(out_vcf, preset="vcf", force=True)
    print(f"VCF records ({contig}): {n} -> {out_vcf} (+.tbi)")

    bed_url = f"{BASE}/{BED}"
    print(f"Downloading benchmark BED: {bed_url}")
    kept = 0
    with urllib.request.urlopen(bed_url, timeout=120) as resp, open(out_bed, "w") as fh:
        for raw in resp:
            line = raw.decode("utf-8", "replace")
            col = line.split("\t", 1)[0]
            if col == args.contig or col == contig:
                fh.write(line)
                kept += 1
    print(f"BED rows ({args.contig}): {kept} -> {out_bed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
