"""CLI for the pipeline's format I/O contract.

    INPUT   FASTQ | BAM/SAM/CRAM | GFA (.gfa / .gfa.gz)
    OUTPUT  BAM | SAM | CRAM | GFA | GBZ | Giraffe indexes (prefix)

Examples
--------
    # reads: FASTQ -> unaligned BAM / SAM / CRAM
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/reads_small.fastq out.bam
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/reads_small.fastq out.sam
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/reads_small.fastq out.cram --reference ref.fasta

    # graph: GFA -> GFA (normalized) or GFA -> GBZ (needs `vg`)
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/synthetic_graph.gfa out.gfa
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/synthetic_graph.gfa out.gbz

    # graph: GFA -> Giraffe indexes (.giraffe.gbz / .min / .dist); needs `vg`
    PYTHONPATH=. .venv/bin/python scripts/convert_formats.py \\
        data/small_samples/synthetic_graph.gfa data/indexes/chr21
"""

from __future__ import annotations

import argparse

from graphmambaformer.data import convert_graph, convert_reads

READ_IN = (".fastq", ".fq", ".fastq.gz", ".fq.gz", ".bam", ".sam", ".cram", ".ubam")
GRAPH_IN = (".gfa", ".gfa.gz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input file (FASTQ / BAM / SAM / CRAM / GFA)")
    ap.add_argument(
        "output",
        help="output file (.bam / .sam / .cram / .gfa / .gbz) "
             "or Giraffe-index prefix (writes .giraffe.gbz/.min/.dist)",
    )
    ap.add_argument("--reference", help="reference FASTA (required for CRAM output)")
    ap.add_argument("--modality", default="pacbio_hifi",
                    help="modality stamped on reads (default: pacbio_hifi)")
    args = ap.parse_args()

    low = args.input.lower()
    try:
        if low.endswith(GRAPH_IN):
            out = convert_graph(args.input, args.output)
        elif low.endswith(READ_IN) or any(
            low.endswith(ext) for ext in (".fastq", ".fq")
        ):
            # also catch misnamed gzipped FASTQ without .gz suffix via convert_reads
            out = convert_reads(args.input, args.output, reference_fasta=args.reference,
                                modality=args.modality)
        else:
            raise SystemExit(f"unrecognized input format: {args.input}")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
