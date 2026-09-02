#!/usr/bin/env python3
"""Map reads with the GraphMambaFormer alignment pipeline -> sorted+indexed BAM.

Supports the three HPRC modalities and their on-disk formats:

=======  =====================  ===========================================
Modality Canonical name         Typical inputs under data/hprc/reads/<SAMPLE>/
=======  =====================  ===========================================
HiFi     ``pacbio_hifi``        ``hifi/*.fastq.gz``  (also BAM)
Illumina ``illumina``           ``illumina/*.cram``  (also FASTQ / BAM)
ONT      ``ont``                ``ont/*.bam``        (also FASTQ / CRAM)
=======  =====================  ===========================================

Aligned Illumina CRAM and ONT BAM are loaded as **sequences** (prior coordinates
stripped) so this script remaps them. CRAM decode uses ``--ref``.

Run modalities **separately** (one BAM each) or **combined** (one BAM with
per-modality ``@RG`` / ``XM`` tags). Pass ``--bam-mode separate`` to always
split a multi-modality run.

Examples
--------
    # HPRC sample — all three modalities combined
    python scripts/chr21/align_ours.py \\
        --ref data/chr21/HG002/ref/GRCh38.chr21.fa \\
        --sample HG00438 --reads-root data/hprc/reads \\
        --modalities illumina,hifi,ont \\
        --out data/hprc/bam/HG00438.ours.sorted.bam \\
        --index-cache data/hprc/index_cache

    # Separate BAMs for each modality (same inputs)
    ... --bam-mode separate

    # One modality only (auto-discovers files under the sample folder)
    python scripts/chr21/align_ours.py --ref REF --sample HG00438 \\
        --modalities hifi --out out.hifi.bam

    # Explicit files (no sample discovery)
    python scripts/chr21/align_ours.py --ref REF \\
        --illumina data/hprc/reads/HG00438/illumina/HG00438.final.cram \\
        --hifi 'data/hprc/reads/HG00438/hifi/*.fastq.gz' \\
        --ont data/hprc/reads/HG00438/ont/ \\
        --out out.combined.bam

    # Hybrid with train-then-map when no checkpoint (pseudo-labels from FASTQ)
    python scripts/chr21/align_ours.py --ref REF \\
        --illumina R1.fastq.gz --illumina R2.fastq.gz \\
        --read-layout paired --mode hybrid --device cuda \\
        --epochs 20 --batch-size 8 --d-model 256 \\
        --train-out data/training_runs/illumina_hybrid \\
        --out out.hybrid.bam
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

from graphmambaformer.progress import progress


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
            for rec in progress(records, desc="group by modality", unit="rec", leave=False):
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
    print(f"[ours] wrote {out}" + ("  (combined multi-modality)" if mixed else ""))
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


def _collect_modality_inputs(args) -> list[tuple[str, list[str], str]]:
    """Return ``[(label, paths, canonical_modality), ...]`` for requested inputs."""
    from graphmambaformer.data import discover_hprc_reads, expand_read_inputs
    from graphmambaformer.data.formats import validate_modality

    jobs: list[tuple[str, list[str], str]] = []

    def add(label: str, paths: list[str], modality: str, layout_hint: str = "single"):
        expanded = expand_read_inputs(paths)
        if not expanded:
            sys.exit(f"ERROR: no files matched for {label}: {paths}")
        jobs.append((label, expanded, validate_modality(modality)))

    # Explicit per-modality flags.
    if args.illumina:
        add("illumina", list(args.illumina), "illumina")
    if args.hifi:
        add("hifi", list(args.hifi), "pacbio_hifi")
    if args.ont:
        add("ont", list(args.ont), "ont")

    # Legacy aliases.
    if args.reads:
        add("illumina", list(args.reads), args.modality)
    if args.long_reads:
        add("hifi", list(args.long_reads), args.long_modality)

    # HPRC sample auto-discovery.
    if args.sample:
        wanted = [
            m.strip().lower()
            for m in (args.modalities or "illumina,hifi,ont").split(",")
            if m.strip()
        ]
        already = {canon for _, _, canon in jobs}
        for mod in wanted:
            paths, canon = discover_hprc_reads(
                args.sample, mod, reads_root=args.reads_root
            )
            if canon in already:
                continue
            jobs.append((mod, paths, canon))
            already.add(canon)

    if not jobs:
        sys.exit(
            "ERROR: provide --illumina / --hifi / --ont, or --sample with "
            "--modalities, or legacy --reads / --long-reads"
        )
    return jobs


def _load_checkpoint(path: str, model) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return payload if isinstance(payload, dict) else {}


def _d_model_from_checkpoint(path: str, default: int) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        cfg = payload.get("model_cfg") or {}
        if isinstance(cfg, dict) and isinstance(cfg.get("d_model"), int):
            return cfg["d_model"]
    return default


def _train_with_pseudo_labels(
    reads,
    reference,
    *,
    ref_seq: str,
    contig: str,
    ref_id: int,
    reference_fasta: str,
    modality: str,
    device,
    epochs: int,
    batch_size: int,
    d_model: int,
    train_out: str,
    workers: int | None,
) -> tuple[object, str]:
    """Pseudo-label ``reads``, train GraphMamba, return ``(model, checkpoint_path)``."""
    from graphmambaformer import build_pipeline
    from graphmambaformer.accel import AccelContext
    from graphmambaformer.config import (
        AccelConfig,
        CoreModelConfig,
        GraphMambaConfig,
        PipelineConfig,
    )
    from graphmambaformer.data import build_batches
    from graphmambaformer.data.real_data import RealReference
    from graphmambaformer.models import build_core_model
    from graphmambaformer.training import (
        TrainConfig,
        Trainer,
        pseudo_label_reads,
    )

    os.makedirs(train_out, exist_ok=True)
    accel = AccelContext(
        AccelConfig(
            device=device,
            num_workers=workers or 0,
            prefetch=2,
            set_threads=True,
        )
    )
    print(f"[ours] train device: {accel.summary()}")

    model_cfg = GraphMambaConfig(d_model=d_model)
    model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=model_cfg)
    ).model
    train_pipeline = build_pipeline(
        PipelineConfig(mode="hybrid", batch_size=batch_size),
        model=model,
        accel=accel,
    )

    print(
        "[ours] no --checkpoint — generating classical pseudo-labels then training"
    )
    pseudo_bam = os.path.join(train_out, "pseudo_truth.bam")
    labeled = pseudo_label_reads(
        reads,
        reference,
        modality=modality,
        batch_size=batch_size,
        device=device,
        write_bam=pseudo_bam,
        references={ref_id: _RefShim(ref_seq)},
        contig_names={ref_id: contig},
        reference_fasta=reference_fasta,
    )
    if not labeled:
        sys.exit(
            "ERROR: classical aligner mapped 0 reads; cannot train without labels"
        )

    real_ref = RealReference(
        reference=reference,
        ref_seq=ref_seq,
        contig=contig,
        ref_id=ref_id,
        fasta_path=reference_fasta,
        label="linear",
    )
    n_val = max(1, int(len(labeled) * 0.2))
    val_reads, train_reads = labeled[:n_val], labeled[n_val:]
    if not train_reads:
        train_reads, val_reads = labeled, labeled[:1]

    train_batches = build_batches(train_reads, real_ref, batch_size)
    val_batches = build_batches(val_reads, real_ref, batch_size)
    print(
        f"[ours] train: {len(train_reads):,} reads / {len(train_batches)} batches; "
        f"val: {len(val_reads):,} / {len(val_batches)}  epochs={epochs} d_model={d_model}"
    )

    trainer = Trainer(
        model,
        train_pipeline,
        accel=accel,
        cfg=TrainConfig(
            out_dir=train_out,
            epochs=epochs,
            batch_size=batch_size,
            patience=max(2, epochs),
            save_checkpoint=True,
            checkpoint_history=True,
        ),
        verbose=True,
    )
    trainer.fit(train_batches, val_batches)
    ckpt = os.path.join(train_out, "checkpoint.pt")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(train_out, "last.pt")
    if not os.path.isfile(ckpt):
        sys.exit(f"ERROR: training finished but no checkpoint under {train_out}")
    print(f"[ours] trained checkpoint -> {ckpt}")
    # Reload into a fresh module so mapping uses the saved weights cleanly.
    map_model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=model_cfg)
    ).model
    _load_checkpoint(ckpt, map_model)
    return map_model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference FASTA (also used to decode CRAM)")
    ap.add_argument("--sample", default=os.environ.get("OURS_SAMPLE"),
                    help="HPRC sample id (e.g. HG00438); discovers files under --reads-root")
    ap.add_argument("--reads-root", default=os.environ.get("OURS_READS_ROOT", "data/hprc/reads"),
                    help="root containing <SAMPLE>/{hifi,illumina,ont}/ (default: data/hprc/reads)")
    ap.add_argument("--modalities", default=os.environ.get("OURS_MODALITIES", "illumina,hifi,ont"),
                    help="comma list used with --sample: illumina,hifi,ont "
                         "(aliases: pacbio_hifi)")
    ap.add_argument("--illumina", action="append", default=None,
                    help="Illumina CRAM/BAM/FASTQ path, dir, or glob (repeatable)")
    ap.add_argument("--hifi", action="append", default=None,
                    help="PacBio HiFi FASTQ.gz/BAM path, dir, or glob (repeatable)")
    ap.add_argument("--ont", action="append", default=None,
                    help="ONT BAM/CRAM/FASTQ path, dir, or glob (repeatable)")
    # Back-compat aliases.
    ap.add_argument("--reads", action="append", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--long-reads", action="append", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--modality", default=os.environ.get("OURS_MODALITY", "illumina"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--long-modality",
                    default=os.environ.get("OURS_LONG_MODALITY", "pacbio_hifi"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--out", required=True,
                    help="output BAM path (combined) or stem for --bam-mode separate")
    ap.add_argument("--mode", default=os.environ.get("OURS_MODE", "fast"),
                    choices=["fast", "hybrid", "two_pass"])
    ap.add_argument("--read-layout", default=os.environ.get("OURS_READ_LAYOUT", "auto"),
                    choices=["auto", "single", "paired"],
                    help="layout for Illumina FASTQ pairs; BAM/CRAM stay single-end rows")
    ap.add_argument("--bam-mode", default=os.environ.get("OURS_BAM_MODE", "combined"),
                    choices=["combined", "separate", "auto"],
                    help="multi-modality output: one BAM with @RG tags (combined) "
                         "or one BAM per modality (separate)")
    ap.add_argument("--region", default=os.environ.get("OURS_REGION"),
                    help="optional samtools region when reading BAM/CRAM (e.g. chr21)")
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
                    help="reads per pipeline batch (default: 64); also used for "
                         "inline training when --epochs/--train-out is set")
    ap.add_argument(
        "--epochs",
        type=int,
        default=_env_int("OURS_EPOCHS"),
        help="when set with hybrid/two_pass and no --checkpoint: train this many "
             "epochs on classical pseudo-labels before mapping (accepted always; "
             "ignored for --mode fast)",
    )
    ap.add_argument(
        "--d-model",
        type=int,
        default=_env_int("OURS_D_MODEL", 256),
        help="model width for hybrid training / checkpoint build (default: 256; "
             "accepted always; ignored for --mode fast without training)",
    )
    ap.add_argument(
        "--checkpoint",
        default=os.environ.get("OURS_CHECKPOINT"),
        help="trained checkpoint.pt for hybrid/two_pass neural scoring",
    )
    ap.add_argument(
        "--train-out",
        default=os.environ.get("OURS_TRAIN_OUT"),
        help="directory for inline training checkpoints when hybrid/two_pass "
             "runs without --checkpoint (implies train-then-map)",
    )
    ap.add_argument("--index-cache", default=os.environ.get("OURS_INDEX_CACHE"),
                    help="directory to reuse Stage-1 indices across separate runs")
    ap.add_argument("--device", default=os.environ.get("OURS_DEVICE", "auto"),
                    help="torch device: auto|cpu|cuda|cuda:N|mps")
    ap.add_argument("--sam", action="store_true",
                    help="also write a companion .sam (off by default for speed)")
    ap.add_argument("--no-sam", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.bam_mode == "auto":
        args.bam_mode = "combined"

    jobs = _collect_modality_inputs(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    t0 = time.time()
    contig, ref_seq = _load_reference(args.ref)
    contig = args.contig or contig
    print(f"[ours] reference contig={contig} len={len(ref_seq):,}  ({time.time()-t0:.1f}s)")

    from graphmambaformer.data import load_real_reads

    def _load_job(job):
        label, paths, modality = job
        layout = args.read_layout if modality == "illumina" else "single"
        # Illumina CRAM / ONT BAM arrive aligned on disk; strip coords for remap.
        batch, _ = load_real_reads(
            reads=paths,
            modality=modality,
            max_reads=args.max_reads,
            layout=layout,
            region=args.region,
            reference_fasta=args.ref,
            as_sequences=True,
        )
        return label, modality, paths, batch

    if len(jobs) > 1:
        with ThreadPoolExecutor(
            max_workers=min(3, len(jobs)), thread_name_prefix="gmf-load"
        ) as ex:
            loaded = list(ex.map(_load_job, jobs))
    else:
        loaded = [_load_job(job) for job in jobs]

    reads = []
    for label, modality, paths, batch in loaded:
        print(f"[ours] {label}: {len(batch):,} reads  modality={modality}  "
              f"files={len(paths)}")
        for p in paths[:5]:
            print(f"         - {p}")
        if len(paths) > 5:
            print(f"         ... +{len(paths) - 5} more")
        reads.extend(batch)

    modalities = sorted({getattr(r, "modality", "?") for r in reads})
    print(f"[ours] total reads={len(reads):,} modalities={modalities} "
          f"mode={args.mode} bam-mode={args.bam_mode}")
    if not reads:
        sys.exit("ERROR: no reads to align")

    from graphmambaformer import build_pipeline
    from graphmambaformer.accel.parallel import configure_torch_threads, default_worker_count
    from graphmambaformer.config import AccelConfig, CoreModelConfig, GraphMambaConfig, PipelineConfig
    from graphmambaformer.models import build_core_model

    workers = default_worker_count(args.workers)
    device = None if args.device in (None, "", "auto") else args.device
    want_neural = args.mode in ("hybrid", "two_pass")
    want_train = want_neural and not args.checkpoint and (
        args.epochs is not None or bool(args.train_out)
    )

    if args.mode == "fast" and (args.epochs is not None or args.train_out or args.checkpoint):
        print(
            "[ours] note: --epochs/--d-model/--checkpoint/--train-out are accepted "
            "but ignored for --mode fast (classical mapping only)"
        )

    model = None
    if want_neural and args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
        d_model = _d_model_from_checkpoint(args.checkpoint, args.d_model)
        model = build_core_model(
            CoreModelConfig(
                arch="graphmamba",
                graphmamba=GraphMambaConfig(d_model=d_model),
            )
        ).model
        _load_checkpoint(args.checkpoint, model)
        print(f"[ours] loaded checkpoint: {args.checkpoint} (d_model={d_model})")
    elif want_train:
        # Need a reference index first (classical + hybrid share Stage-1 bundle).
        index_pipeline = build_pipeline(
            PipelineConfig(
                mode="fast",
                batch_size=max(1, int(args.batch_size)),
                accel=AccelConfig(device=device, num_workers=workers, set_threads=True),
            ),
            device=device,
        )
        reference = _build_or_load_reference(
            index_pipeline, ref_seq, args.ref_id, args.ref, args.index_cache
        )
        train_out = args.train_out or os.path.join(
            os.path.dirname(os.path.abspath(args.out)) or ".",
            "training_runs",
            "align_ours_inline",
        )
        epochs = int(args.epochs) if args.epochs is not None else 6
        model, _ckpt = _train_with_pseudo_labels(
            reads,
            reference,
            ref_seq=ref_seq,
            contig=contig,
            ref_id=args.ref_id,
            reference_fasta=args.ref,
            modality=loaded[0][1],
            device=device,
            epochs=epochs,
            batch_size=max(1, int(args.batch_size)),
            d_model=int(args.d_model),
            train_out=train_out,
            workers=workers,
        )
    elif want_neural:
        print(
            "[ours] warning: hybrid/two_pass without --checkpoint or "
            "--epochs/--train-out — neural scoring disabled (classical only). "
            "Pass --checkpoint PATH or --epochs N --train-out DIR to train."
        )

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
          f"threads={thread_info.get('num_threads')} d_model={args.d_model}")

    pipeline = build_pipeline(cfg, model=model, device=device)
    t1 = time.time()
    # Reuse index from training when we already built it; otherwise build now.
    if want_train:
        # `_train_with_pseudo_labels` used the same Stage-1 reference object;
        # rebuild via cache so mapping pipeline owns a matching bundle.
        reference = _build_or_load_reference(
            pipeline, ref_seq, args.ref_id, args.ref, args.index_cache
        )
    else:
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
    default_mod = loaded[0][1]

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
