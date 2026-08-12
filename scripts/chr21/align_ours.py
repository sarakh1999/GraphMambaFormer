#!/usr/bin/env python3
"""Map reads with the GraphMambaFormer alignment pipeline -> sorted+indexed BAM.

This is the "our implementation" arm of the chr21 mentor benchmark. It runs the
classical/neural seed -> chain -> extend -> score pipeline
(:mod:`graphmambaformer.alignment`) against a single-contig reference (e.g.
GRCh38 ``chr21``) and writes a BAM whose ``@SQ`` name matches the reference
FASTA, so DeepVariant / hap.py accept it exactly like the Giraffe BAM.

It runs entirely in Python via pysam (bundled htslib) — no Docker, no external
samtools — so it works from environments that cannot reach the Docker socket.

Short and long reads can be aligned in **one command**. When both are given,
the default is a single combined BAM (distinct ``@RG`` / ``XM`` tags per
modality). Pass ``--bam-mode separate`` if a caller needs homogeneous files.

Throughput knobs (combined *or* per-file runs)
----------------------------------------------
* Host threads for Stage 1/3 are auto-sized to the machine (``--workers`` /
  ``$GMF_NUM_WORKERS`` / ``$OURS_WORKERS``).
* ``--index-cache DIR`` persists the Stage-1 reference index so separate R1 /
  R2 / long invocations reuse the same build (combined runs already share one
  in-memory index).
* ``--batch-size`` controls how many reads pass through the pipeline at once.
* Companion SAM is **opt-in** (``--sam``); writing it roughly doubles I/O.

Usage
-----
    # short reads only (Illumina R1/R2)
    python scripts/chr21/align_ours.py \\
        --ref  data/chr21/HG002/ref/GRCh38.chr21.fa \\
        --reads data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz \\
        --reads data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz \\
        --out  data/chr21/HG002/bam/HG002.chr21.ours.sorted.bam \\
        --mode fast

    # short + long in one command -> one combined BAM
    python scripts/chr21/align_ours.py \\
        --ref  data/chr21/HG002/ref/GRCh38.chr21.fa \\
        --reads data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz \\
        --reads data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz \\
        --long-reads data/chr21/HG002/reads/HG002.chr21.hifi.fastq.gz \\
        --out  data/chr21/HG002/bam/HG002.chr21.ours.sorted.bam \\
        --index-cache data/chr21/HG002/index_cache

    # same inputs as three separate invocations (index reused from cache)
    python scripts/chr21/align_ours.py --ref REF --reads R1.fq \\
        --out out.R1.bam --read-layout single --index-cache CACHE
    python scripts/chr21/align_ours.py --ref REF --reads R2.fq \\
        --out out.R2.bam --read-layout single --index-cache CACHE
    python scripts/chr21/align_ours.py --ref REF --long-reads hifi.fq \\
        --out out.hifi.bam --index-cache CACHE

Notes
-----
* ``--mode fast`` (default) is fully classical and needs no trained model.
  ``hybrid`` / ``two_pass`` only add neural re-ranking when a model with
  alignment heads is supplied, which this script does not load, so they degrade
  gracefully to the classical path.
* One combined BAM is always possible when both modalities share a reference
  (they do here). Separate BAMs are optional for DeepVariant / Sniffles style
  callers that expect a single platform.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Make the repo importable when run as a plain script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _RefShim:
    """Minimal stand-in for :class:`Reference` — the BAM writer only needs ``seq``."""

    __slots__ = ("seq",)

    def __init__(self, seq: str) -> None:
        self.seq = seq


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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


def _load_reads(paths: list[str], modality: str, max_reads: int | None,
                layout: str = "auto"):
    """Load paired Illumina or single-end long reads through the shared path."""
    from graphmambaformer.data import load_real_reads

    reads, _ = load_real_reads(
        reads=paths, modality=modality, max_reads=max_reads, layout=layout
    )
    return reads


def _ensure_ext(path: str) -> str:
    if path.lower().endswith((".bam", ".ubam", ".sam", ".cram")):
        return path
    return path + ".bam"


def _stem_and_ext(path: str) -> tuple[str, str]:
    low = path.lower()
    for suffix in (".bam", ".ubam", ".sam", ".cram"):
        if low.endswith(suffix):
            return path[: -len(suffix)], suffix
    return path, ".bam"


def _write_outputs(
    *,
    results,
    reads,
    out_path: str,
    bam_mode: str,
    references: dict,
    contig_names: dict,
    modality: str,
    reference_fasta: str | None,
    also_sam: bool,
) -> list[str]:
    """Write combined or per-modality BAM (and optional companion SAM)."""
    from graphmambaformer.data import write_alignments, write_alignments_split
    from graphmambaformer.data.alignment_io import alignments_to_records
    from graphmambaformer.data.formats import write_sam

    written: list[str] = []
    modalities = {getattr(r, "modality", None) or modality for r in reads}
    mixed = len(modalities) > 1
    # Combined BAM is always valid on a shared reference; separate is opt-in.
    use_separate = bam_mode == "separate"

    if mixed and use_separate:
        paths = write_alignments_split(
            results, reads, out_path,
            references=references,
            contig_names=contig_names,
            modality=modality,
            reference_fasta=(
                reference_fasta if out_path.lower().endswith(".cram") else None
            ),
        )
        written.extend(paths.values())
        for mod, p in paths.items():
            print(f"[ours] wrote {p}  (modality={mod})")

        if also_sam:
            records = alignments_to_records(results, reads, modality=modality)
            by_mod: dict[str, list] = {}
            for rec in records:
                by_mod.setdefault(rec.modality or modality, []).append(rec)
            stem, _ = _stem_and_ext(out_path)
            for mod, group in by_mod.items():
                sam_path = f"{stem}.{mod}.sam"
                write_sam(group, sam_path, references=references,
                          contig_names=contig_names)
                written.append(sam_path)
                print(f"[ours] wrote {sam_path}")
        return written

    out = write_alignments(
        results, reads, out_path,
        references=references,
        contig_names=contig_names,
        modality=modality,
        reference_fasta=(
            reference_fasta if out_path.lower().endswith(".cram") else None
        ),
    )
    written.append(out)
    print(f"[ours] wrote {out}" + ("  (combined short+long)" if mixed else ""))
    if also_sam and out.lower().endswith((".bam", ".ubam", ".cram")):
        sam_path = os.path.splitext(out)[0] + ".sam"
        write_alignments(
            results, reads, sam_path,
            references=references, contig_names=contig_names,
            modality=modality,
        )
        written.append(sam_path)
        print(f"[ours] wrote {sam_path}")
    return written


def _build_or_load_reference(pipeline, ref_seq: str, ref_id: int,
                             ref_path: str, cache_dir: str | None):
    """Build the Stage-1 index, reusing a disk cache when available."""
    from graphmambaformer.alignment.index_cache import (
        index_cache_key,
        load_index_bundle,
        save_index_bundle,
    )
    from graphmambaformer.alignment.pipeline import ReferenceIndex

    key = index_cache_key(ref_path, pipeline.cfg.seeding, ref_id=ref_id)
    bundle = load_index_bundle(cache_dir, key)
    if bundle is not None:
        print(f"[ours] loaded reference index from cache ({cache_dir})")
        return ReferenceIndex(bundle=bundle, ref_seq=ref_seq, ref_id=ref_id)

    reference = pipeline.build_reference(ref_seq, ref_id=ref_id)
    saved = save_index_bundle(cache_dir, key, reference.bundle)
    if saved:
        print(f"[ours] cached reference index -> {saved}")
    return reference


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="single-contig reference FASTA (e.g. chr21)")
    ap.add_argument("--reads", action="append", default=None,
                    help="short-read FASTQ(.gz)/BAM; repeat for R1 and R2 "
                         "(modality from --modality, default illumina)")
    ap.add_argument("--long-reads", action="append", default=None,
                    help="long-read FASTQ(.gz)/BAM/uBAM; repeatable. "
                         "May be combined with --reads in one command")
    ap.add_argument("--out", required=True,
                    help="output BAM path (combined) or stem for --bam-mode separate")
    ap.add_argument("--mode", default=os.environ.get("OURS_MODE", "fast"),
                    choices=["fast", "hybrid", "two_pass"])
    ap.add_argument("--modality", default=os.environ.get("OURS_MODALITY", "illumina"),
                    help="modality for --reads (default: illumina)")
    ap.add_argument("--long-modality",
                    default=os.environ.get("OURS_LONG_MODALITY", "pacbio_hifi"),
                    help="modality for --long-reads (default: pacbio_hifi)")
    ap.add_argument("--read-layout", default=os.environ.get("OURS_READ_LAYOUT", "auto"),
                    choices=["auto", "single", "paired"],
                    help="layout for --reads; long reads always stay single-end")
    ap.add_argument("--bam-mode", default=os.environ.get("OURS_BAM_MODE", "combined"),
                    choices=["combined", "separate", "auto"],
                    help="when short+long are both given: one BAM with @RG tags "
                         "(combined, default) or one BAM per modality (separate). "
                         "auto currently equals combined (shared reference)")
    ap.add_argument("--contig", default=os.environ.get("OURS_CONTIG"),
                    help="override @SQ contig name (default: reference FASTA header)")
    ap.add_argument("--ref-id", type=int, default=0)
    ap.add_argument("--max-reads", type=int,
                    default=int(os.environ.get("OURS_MAX_READS", "0")) or None,
                    help="cap reads per modality group (0/unset = all)")
    ap.add_argument("--workers", type=int,
                    default=_env_int("OURS_WORKERS") or _env_int("GMF_NUM_WORKERS"),
                    help="host threads for Stage 1/3 (default: all CPUs)")
    ap.add_argument("--batch-size", type=int,
                    default=_env_int("OURS_BATCH_SIZE", 64),
                    help="reads per pipeline batch (default: 64)")
    ap.add_argument("--index-cache", default=os.environ.get("OURS_INDEX_CACHE"),
                    help="directory to reuse Stage-1 indices across separate runs")
    ap.add_argument("--device", default=os.environ.get("OURS_DEVICE", "auto"),
                    help="torch device: auto|cpu|cuda|cuda:N|mps")
    ap.add_argument("--sam", action="store_true",
                    help="also write a companion .sam (off by default for speed)")
    ap.add_argument("--no-sam", action="store_true",
                    help=argparse.SUPPRESS)  # backwards-compatible no-op
    args = ap.parse_args()

    short_paths = list(args.reads or [])
    long_paths = list(args.long_reads or [])
    if not short_paths and not long_paths:
        sys.exit("ERROR: provide --reads and/or --long-reads")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    t0 = time.time()
    contig, ref_seq = _load_reference(args.ref)
    contig = args.contig or contig
    print(f"[ours] reference contig={contig} len={len(ref_seq):,}  ({time.time()-t0:.1f}s)")

    # Overlap short/long FASTQ loading when both are present.
    reads = []
    load_jobs = []
    if short_paths:
        load_jobs.append(("short", short_paths, args.modality, args.read_layout))
    if long_paths:
        load_jobs.append(("long", long_paths, args.long_modality, "single"))

    def _load_job(job):
        kind, paths, modality, layout = job
        batch = _load_reads(paths, modality, args.max_reads, layout)
        return kind, modality, batch

    if len(load_jobs) > 1:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gmf-load") as ex:
            loaded = list(ex.map(_load_job, load_jobs))
    else:
        loaded = [_load_job(job) for job in load_jobs]

    for kind, modality, batch in loaded:
        print(f"[ours] {kind} reads={len(batch):,} modality={modality}")
        reads.extend(batch)

    modalities = sorted({getattr(r, "modality", "?") for r in reads})
    print(f"[ours] total reads={len(reads):,} modalities={modalities} "
          f"mode={args.mode} bam-mode={args.bam_mode}")
    if not reads:
        sys.exit("ERROR: no reads to align")

    from graphmambaformer import build_pipeline
    from graphmambaformer.accel.parallel import configure_torch_threads, default_worker_count
    from graphmambaformer.config import AccelConfig, PipelineConfig

    workers = default_worker_count(args.workers)
    device = None if args.device in (None, "", "auto") else args.device
    cfg = PipelineConfig(
        mode=args.mode,
        batch_size=max(1, int(args.batch_size)),
        accel=AccelConfig(
            device=device,
            num_workers=workers,
            stage_parallel=True,
            set_threads=True,
        ),
    )
    thread_info = configure_torch_threads(
        "cuda" if (device or "").startswith("cuda") else (device or "cpu"),
        workers=workers,
    )
    print(f"[ours] workers={workers} batch_size={cfg.batch_size} "
          f"threads={thread_info.get('num_threads')}")

    pipeline = build_pipeline(cfg, device=device)
    t1 = time.time()
    reference = _build_or_load_reference(
        pipeline, ref_seq, args.ref_id, args.ref, args.index_cache
    )
    print(f"[ours] reference index ready  ({time.time()-t1:.1f}s)")

    t2 = time.time()
    results, stats = pipeline.align(reads, reference)
    elapsed = time.time() - t2
    rate = (len(reads) / elapsed) if elapsed > 0 else float("inf")
    print(f"[ours] aligned  ({elapsed:.1f}s, {rate:,.1f} reads/s)")
    print(f"[ours] stats: {stats.summary()}")

    references = {args.ref_id: _RefShim(ref_seq)}
    contig_names = {args.ref_id: contig}
    out_path = _ensure_ext(args.out)
    default_mod = args.modality if short_paths else args.long_modality

    _write_outputs(
        results=results,
        reads=reads,
        out_path=out_path,
        bam_mode=args.bam_mode,
        references=references,
        contig_names=contig_names,
        modality=default_mod,
        reference_fasta=args.ref,
        also_sam=bool(args.sam),
    )
    print(f"[ours] done  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
