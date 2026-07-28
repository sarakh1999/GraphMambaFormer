"""CLI for the pipeline's format I/O contract.

    INPUT   FASTQ | BAM/SAM/CRAM | GFA
    OUTPUT  BAM   | CRAM         | GFA | GBZ

Examples
--------
    # reads: FASTQ -> unaligned BAM
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \
        data/small_samples/reads_small.fastq out.bam

    # reads: FASTQ -> CRAM (needs a reference FASTA)
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \
        data/small_samples/reads_small.fastq out.cram --reference ref.fasta

    # graph: GFA -> GFA (normalized) or GFA -> GBZ (needs `vg`)
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \
        data/small_samples/synthetic_graph.gfa out.gfa
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \
        data/small_samples/synthetic_graph.gfa out.gbz
"""

from __future__ import annotations

import argparse

from graphmambaformer.data import convert_graph, convert_reads

READ_IN = (".fastq", ".fq", ".bam", ".sam", ".cram")
GRAPH_IN = (".gfa",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input file (FASTQ / BAM / SAM / CRAM / GFA)")
    ap.add_argument("output", help="output file (.bam / .cram / .gfa / .gbz)")
    ap.add_argument("--reference", help="reference FASTA (required for CRAM output)")
    ap.add_argument("--modality", default="pacbio_hifi",
                    help="modality stamped on reads (default: pacbio_hifi)")
    args = ap.parse_args()

    low = args.input.lower()
    try:
        if low.endswith(GRAPH_IN):
            out = convert_graph(args.input, args.output)
        elif low.endswith(READ_IN):
            out = convert_reads(args.input, args.output, reference_fasta=args.reference,
                                modality=args.modality)
        else:
            raise SystemExit(f"unrecognized input format: {args.input}")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
