#!/usr/bin/env python3
"""Map reads with the GraphMambaFormer alignment pipeline -> sorted+indexed BAM.

This is the "our implementation" arm of the chr21 mentor benchmark. It runs the
classical/neural seed -> chain -> extend -> score pipeline
(:mod:`graphmambaformer.alignment`) against a single-contig reference (e.g.
GRCh38 ``chr21``) and writes a BAM whose ``@SQ`` name matches the reference
FASTA, so DeepVariant / hap.py accept it exactly like the Giraffe BAM.

It runs entirely in Python via pysam (bundled htslib) — no Docker, no external
samtools — so it works from environments that cannot reach the Docker socket.

Usage
-----
    python scripts/chr21/align_ours.py \
        --ref  data/chr21/HG002/ref/GRCh38.chr21.fa \
        --reads data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz \
        --reads data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz \
        --out  data/chr21/HG002/bam/HG002.chr21.ours.sorted.bam \
        --mode fast

Notes
-----
* The pipeline is a *reference* implementation in pure Python; it is correct but
  not throughput-optimized. Use ``--max-reads`` to bound runtime while iterating.
* ``--mode fast`` (default) is fully classical and needs no trained model.
  ``hybrid`` / ``two_pass`` only add neural re-ranking when a model with
  alignment heads is supplied, which this script does not load, so they degrade
  gracefully to the classical path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the repo importable when run as a plain script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _RefShim:
    """Minimal stand-in for :class:`Reference` — the BAM writer only needs ``seq``."""

    __slots__ = ("seq",)

    def __init__(self, seq: str) -> None:
        self.seq = seq


def _load_reference(path: str) -> tuple[str, str]:
    """Return ``(contig_name, sequence)`` for a single-contig FASTA."""
    import pysam

    if not os.path.exists(path + ".fai"):
        pysam.faidx(path)
    fa = pysam.FastaFile(path)
    try:
        names = list(fa.references)
        if not names:
            sys.exit(f"ERROR: no contigs in reference {path}")
        if len(names) > 1:
            print(f"note: reference has {len(names)} contigs; using the first: {names[0]}")
        name = names[0]
        seq = fa.fetch(name).upper()
    finally:
        fa.close()
    return name, seq


def _load_reads(paths: list[str], modality: str, max_reads: int | None):
    """Read FASTQ/BAM reads, give every read a unique id, optionally cap count."""
    from graphmambaformer.data import read_reads

    reads = []
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"ERROR: reads file not found: {p}")
        batch = read_reads(p, modality=modality)
        reads.extend(batch)
        if max_reads and len(reads) >= max_reads:
            reads = reads[:max_reads]
            break

    # Aligning mates single-end: ids must be unique or the writer would pair a
    # result with the wrong read's bases. Suffix with a global index.
    for i, rec in enumerate(reads):
        rec.read_id = f"{rec.read_id}#{i}"
    return reads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="single-contig reference FASTA (e.g. chr21)")
    ap.add_argument("--reads", action="append", required=True,
                    help="FASTQ(.gz)/BAM reads; repeat for R1 and R2")
    ap.add_argument("--out", required=True, help="output sorted BAM path")
    ap.add_argument("--mode", default=os.environ.get("OURS_MODE", "fast"),
                    choices=["fast", "hybrid", "two_pass"])
    ap.add_argument("--modality", default=os.environ.get("OURS_MODALITY", "illumina"))
    ap.add_argument("--contig", default=os.environ.get("OURS_CONTIG"),
                    help="override @SQ contig name (default: reference FASTA header)")
    ap.add_argument("--ref-id", type=int, default=0)
    ap.add_argument("--max-reads", type=int,
                    default=int(os.environ.get("OURS_MAX_READS", "0")) or None,
                    help="cap number of reads aligned (0/unset = all)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    t0 = time.time()
    contig, ref_seq = _load_reference(args.ref)
    contig = args.contig or contig
    print(f"[ours] reference contig={contig} len={len(ref_seq):,}  ({time.time()-t0:.1f}s)")

    reads = _load_reads(args.reads, args.modality, args.max_reads)
    print(f"[ours] reads={len(reads):,} modality={args.modality} mode={args.mode}")
    if not reads:
        sys.exit("ERROR: no reads to align")

    # Build the pipeline and the one-time reference index, then align.
    from graphmambaformer import build_pipeline
    from graphmambaformer.data import write_alignments

    pipeline = build_pipeline(args.mode)
    t1 = time.time()
    reference = pipeline.build_reference(ref_seq, ref_id=args.ref_id)
    print(f"[ours] built reference index  ({time.time()-t1:.1f}s)")

    t2 = time.time()
    results, stats = pipeline.align(reads, reference)
    print(f"[ours] aligned  ({time.time()-t2:.1f}s)")
    print(f"[ours] stats: {stats.summary()}")

    references = {args.ref_id: _RefShim(ref_seq)}
    contig_names = {args.ref_id: contig}
    out_path = args.out
    if not out_path.lower().endswith((".bam", ".ubam", ".sam", ".cram")):
        out_path = out_path + ".bam"
    out = write_alignments(
        results, reads, out_path,
        references=references,
        contig_names=contig_names,
        modality=args.modality,
        reference_fasta=args.ref if out_path.lower().endswith(".cram") else None,
    )
    print(f"[ours] wrote {out}  total {time.time()-t0:.1f}s")

    # Companion SAM when the primary output is binary.
    if out.lower().endswith((".bam", ".ubam", ".cram")):
        sam_path = os.path.splitext(out)[0] + ".sam"
        write_alignments(
            results, reads, sam_path,
            references=references, contig_names=contig_names,
            modality=args.modality,
        )
        print(f"[ours] wrote {sam_path}")


if __name__ == "__main__":
    main()
