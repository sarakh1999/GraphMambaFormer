#!/usr/bin/env python3
"""Prepare chr21 **long-read** inputs (PacBio HiFi + ONT) for joint training.

The chr21 mentor benchmark ships a real Illumina truth BAM
(``HG002.chr21.*.giab.sorted.bam``), but the matching long-read reads/alignments
are not on disk and fetching real HG002 HiFi/ONT requires Docker + minimap2 + S3
(unavailable in this environment). To let the universal aligner train on all
three modalities *now*, this script simulates modality-realistic HiFi and ONT
reads **directly off the real chr21 reference window** using a controlled edit
process, so every read carries an exact truth alignment.

For each requested modality it writes, under ``data/chr21/<SAMPLE>/``:

    reads/<SAMPLE>.chr21.<mod>.fastq.gz   single-end FASTQ (as sequenced)
    bam/<SAMPLE>.chr21.<mod>.truth.sorted.bam(.bai)   truth in chr21 coordinates

The truth BAMs use absolute ``chr21`` coordinates and the full-contig ``@SQ``
length, exactly like the real Illumina GIAB BAM, so ``scripts/train.py`` shifts
all three modalities into the same window frame when it loads them.

Swapping in **real** long reads later: drop a real HiFi/ONT FASTQ/BAM in place
and point the training manifest at it (with a truth BAM, or via the pseudo-label
path). Nothing here is load-bearing beyond producing those two file kinds.

Example
-------
    PYTHONPATH=. python scripts/chr21/prepare_long_reads.py \
        --ref data/chr21/HG002/ref/chr21.fa \
        --region chr21:20000000-20050000 \
        --sample HG002 --modalities pacbio_hifi,ont \
        --n-reads 300
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphmambaformer.data.formats import validate_modality, write_bam
from graphmambaformer.data.real_data import read_fasta_contig
from graphmambaformer.data.synthetic import READ_PROFILES, simulate_reads_from_sequence
from graphmambaformer.progress import progress


class _LenSeq:
    """A stand-in for a reference sequence that only reports its length.

    ``write_bam`` needs the ``@SQ`` contig length (``len(reference.seq)``) but
    never the bases, so this avoids materializing the whole ~46 Mb chr21 string
    just to stamp ``LN`` in the header.
    """

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


def _contig_length(fasta: str, contig: str) -> int:
    fai = fasta + ".fai"
    if not os.path.exists(fai):
        import pysam

        pysam.faidx(fasta)
    with open(fai) as fh:
        for line in fh:
            name, length, *_ = line.split("\t")
            if name == contig:
                return int(length)
    raise ValueError(f"contig {contig!r} not found in {fai}")


def _write_fastq_gz(records, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in progress(records, desc="write FASTQ", unit="read", leave=False):
            qual = "".join(chr(33 + min(93, max(0, q))) for q in r.quals)
            fh.write(f"@{r.read_id} mod={r.modality}\n{r.seq}\n+\n{qual}\n")


def prepare_modality(
    *,
    modality: str,
    seq: str,
    offset: int,
    contig: str,
    contig_length: int,
    n_reads: int,
    seed: int,
    reads_path: str,
    bam_path: str,
) -> dict:
    """Simulate one modality's reads and write the FASTQ + truth BAM."""
    recs = simulate_reads_from_sequence(
        seq, modality, n_reads, ref_id=0, seed=seed,
        id_prefix=f"{modality}_chr21",
    )
    # Shift window-local coordinates into absolute contig coordinates so the
    # truth BAM matches the real GIAB Illumina BAM's frame.
    for r in progress(recs, desc=f"shift coords {modality}", unit="read", leave=False):
        r.ref_id = 0
        r.ref_start += offset
        r.ref_end += offset
        r.ref_positions = [(p + offset) if p >= 0 else -1 for p in r.ref_positions]

    _write_fastq_gz(recs, reads_path)
    write_bam(
        recs, bam_path,
        references={0: SimpleNamespace(seq=_LenSeq(contig_length))},
        sort=True, index=True, contig_names={0: contig},
    )
    lengths = [len(r.seq) for r in recs]
    return {
        "modality": modality,
        "n_reads": len(recs),
        "median_len": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "fastq": reads_path,
        "truth_bam": bam_path,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ref", default="data/chr21/HG002/ref/chr21.fa",
                   help="chr21 reference FASTA")
    p.add_argument("--region", default="chr21:20000000-20050000",
                   help="samtools-style window to simulate reads from")
    p.add_argument("--sample", default="HG002")
    p.add_argument("--modalities", default="pacbio_hifi,ont",
                   help="comma-separated modalities to prepare "
                        f"(profiles: {sorted(READ_PROFILES)})")
    p.add_argument("--n-reads", type=int, default=300,
                   help="reads per modality")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data/chr21",
                   help="root under which <SAMPLE>/reads and <SAMPLE>/bam live")
    args = p.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"ERROR: reference FASTA not found: {args.ref}")

    contig, seq, offset = read_fasta_contig(args.ref, region=args.region)
    contig_length = _contig_length(args.ref, contig)
    reads_dir = os.path.join(args.data_root, args.sample, "reads")
    bam_dir = os.path.join(args.data_root, args.sample, "bam")
    os.makedirs(reads_dir, exist_ok=True)
    os.makedirs(bam_dir, exist_ok=True)

    print("=" * 74)
    print("Preparing chr21 long-read inputs (simulated off the real reference)")
    print("=" * 74)
    print(f"reference : {args.ref}  contig={contig} (LN={contig_length:,})")
    print(f"window    : {args.region}  offset={offset:,}  len={len(seq):,}bp")
    print(f"reads/mod : {args.n_reads}")

    summaries = []
    for raw in [m.strip() for m in args.modalities.split(",") if m.strip()]:
        modality = validate_modality(raw)
        reads_path = os.path.join(reads_dir, f"{args.sample}.chr21.{modality}.fastq.gz")
        bam_path = os.path.join(bam_dir, f"{args.sample}.chr21.{modality}.truth.sorted.bam")
        info = prepare_modality(
            modality=modality, seq=seq, offset=offset, contig=contig,
            contig_length=contig_length, n_reads=args.n_reads, seed=args.seed,
            reads_path=reads_path, bam_path=bam_path,
        )
        summaries.append(info)
        print(f"\n[{modality}] {info['n_reads']} reads  median_len={info['median_len']:,}bp")
        print(f"   FASTQ     -> {info['fastq']}")
        print(f"   truth BAM -> {info['truth_bam']}")

    print("\nDone. Use these with scripts/train.py --modality-inputs (see the "
          "chr21 all-modality manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
