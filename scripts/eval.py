#!/usr/bin/env python3
"""Evaluate a trained (or classical) aligner on real or synthetic references.

Two data sources, one evaluation loop (mirrors ``scripts/train.py``):

* ``--data real`` — a reference **FASTA** (+ optional pangenome **GFA**) and
  reads. Reads come from an aligned **truth BAM/SAM/CRAM** (``--truth-bam``, so
  locus / MAPQ / anchor metrics can be scored) and/or plain **FASTQ/BAM**
  (``--reads-file``, alignment + predicted BAM only). Auto-selected when
  ``--reference-fasta`` is given.
* ``--data synthetic`` (default) — the fully-labelled synthetic split.

Reports locus accuracy, mapped fraction, anchor precision/recall/AUC, chain
accuracy, MAPQ MAE (whenever truth is available) and always writes predicted
alignments as a sorted+indexed **BAM** (plus **SAM**) whose ``@SQ`` name matches
the reference contig, per ``--ref-mode`` (linear / pangenome / both).

Examples
--------
    # REAL linear: score a trained checkpoint against a GIAB truth BAM window
    PYTHONPATH=. python scripts/eval.py --data real \
        --reference-fasta data/chr21/HG002/ref/GRCh38.chr21.fa \
        --truth-bam data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam \
        --region chr21:5000000-6000000 --ref-mode linear \
        --checkpoint data/training_runs/chr21_linear/checkpoint.pt \
        --out data/eval_runs/chr21_linear

    # REAL pangenome: attach the real GFA graph
    PYTHONPATH=. python scripts/eval.py --data real \
        --reference-fasta data/chr21/HG002/ref/GRCh38.chr21.fa \
        --gfa data/chr21/HG002/chr21.gfa \
        --truth-bam data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam \
        --region chr21:5000000-6000000 --ref-mode pangenome \
        --checkpoint data/training_runs/chr21_pangenome/checkpoint.pt \
        --out data/eval_runs/chr21_pangenome

    # SYNTHETIC both, classical (no checkpoint)
    PYTHONPATH=. python scripts/eval.py --ref-mode both --mode fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from graphmambaformer.accel import AccelContext
from graphmambaformer.alignment import DualReferenceAligner
from graphmambaformer.alignment.pipeline import build_pipeline
from graphmambaformer.config import (
    AccelConfig,
    CoreModelConfig,
    GraphMambaConfig,
    LossConfig,
    PipelineConfig,
)
from graphmambaformer.data import (
    build_batches,
    build_dual_reference_from_files,
    build_reference_from_files,
    build_reference_from_synthetic,
    export_dataset,
    generate_dataset,
    load_real_reads,
    preset,
    write_alignments,
)
from graphmambaformer.models import build_core_model
from graphmambaformer.progress import progress
from graphmambaformer.training import Trainer, TrainConfig


def chunk(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ref_modes(mode: str) -> list[tuple[str, bool]]:
    if mode == "linear":
        return [("linear", False)]
    if mode == "pangenome":
        return [("pangenome", True)]
    if mode == "both":
        return [("linear", False), ("pangenome", True)]
    raise ValueError(mode)


def _load_checkpoint(path: str, model) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return payload if isinstance(payload, dict) else {}


def _d_model_from_checkpoint(path: str, default: int) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        cfg = payload.get("model_cfg") or {}
        if isinstance(cfg, dict) and isinstance(cfg.get("d_model"), int):
            return cfg["d_model"]
        conf = payload.get("config") or {}
        if isinstance(conf, dict) and isinstance(conf.get("d_model"), int):
            return conf["d_model"]
    return default


def _write_pred_bam(mode_dir, split_tag, batches, pipeline, refs_meta,
                    *, write_cram: bool = False, reference_fasta: str | None = None):
    """Align every batch and write BAM (+ SAM, optional CRAM) with real @SQ."""
    all_reads, all_results = [], []
    for reads, reference in progress(
        batches, desc=f"eval align[{split_tag}]", unit="batch", leave=False
    ):
        results, stats = pipeline.align(reads, reference)
        all_reads.extend(reads)
        all_results.extend(results)
        print(f"    align stats: {stats.summary()}")
    bam_path = os.path.join(mode_dir, f"pred.{split_tag}.bam")

    class _Ref:
        __slots__ = ("seq",)

        def __init__(self, seq):
            self.seq = seq

    references = {rid: _Ref(seq) for rid, (seq, _name) in refs_meta.items()}
    contig_names = {rid: name for rid, (_seq, name) in refs_meta.items()}
    out = write_alignments(
        all_results, all_reads, bam_path,
        references=references, contig_names=contig_names,
    )
    print(f"    predicted BAM -> {out} (+ .bai)")
    try:
        import pysam
        sam_path = os.path.join(mode_dir, f"pred.{split_tag}.sam")
        with pysam.AlignmentFile(out, "rb") as bam_in, \
                pysam.AlignmentFile(sam_path, "w", header=bam_in.header) as sam_out:
            for aln in progress(bam_in, desc="export SAM", unit="aln", leave=False):
                sam_out.write(aln)
        print(f"    predicted SAM -> {sam_path}")
    except Exception as exc:  # pragma: no cover
        print(f"    note: SAM export skipped ({exc})")
    if write_cram:
        try:
            cram_path = os.path.join(mode_dir, f"pred.{split_tag}.cram")
            write_alignments(
                all_results, all_reads, cram_path,
                references=references, contig_names=contig_names,
                reference_fasta=reference_fasta,
            )
            print(f"    predicted CRAM -> {cram_path}")
        except Exception as exc:  # pragma: no cover
            print(f"    note: CRAM export skipped ({exc})")


def _write_integrated_bam(out_dir, split_tag, dual_ctx, pipeline):
    """One pass over both references (shared read encoding) -> integrated BAM.

    Uses :class:`DualReferenceAligner` so linear + pangenome are aligned in a
    single sweep and folded into one concordance-picked primary per read, rather
    than re-running the reads against each reference independently.
    """
    reads = dual_ctx["reads"]
    references = dual_ctx["references"]
    refs_meta = dual_ctx["refs_meta"]

    aligner = DualReferenceAligner(pipeline)
    result = aligner.align(reads, references, integrate=True)
    concordant = sum(1 for c in result.integrated if c.concordant and c.primary)
    mapped = sum(1 for c in result.integrated if c.primary is not None)
    print(f"    integrated: refs={result.names} mapped={mapped}/{len(reads)} "
          f"concordant={concordant}")
    for name, stats in result.stats.items():
        print(f"      [{name}] {stats.summary()}")

    integrated_dir = os.path.join(out_dir, "integrated")
    os.makedirs(integrated_dir, exist_ok=True)
    bam_path = os.path.join(integrated_dir, f"pred.{split_tag}.bam")

    class _Ref:
        __slots__ = ("seq",)

        def __init__(self, seq):
            self.seq = seq

    ref_objs = {rid: _Ref(seq) for rid, (seq, _name) in refs_meta.items()}
    contig_names = {rid: name for rid, (_seq, name) in refs_meta.items()}
    out = write_alignments(
        result.integrated_alignments(), reads, bam_path,
        references=ref_objs, contig_names=contig_names,
    )
    print(f"    integrated BAM -> {out} (+ .bai)")


# --------------------------------------------------------------------------- #
# synthetic and real batch builders -> {label: (batches, refs_meta, has_truth)}
# --------------------------------------------------------------------------- #
def build_synthetic_eval(args, pipeline, model_cfg):
    spec = preset(args.preset)
    if args.reads:
        spec = replace(spec, n_train=args.reads, n_val=max(2, args.reads // 4),
                       n_test=max(2, args.reads // 4))
    ds = generate_dataset(spec)
    kmer_size = model_cfg.graph_encoder.kmer_size

    if args.emit_truth:
        truth_dir = os.path.join(args.out, "truth")
        paths = export_dataset(ds, truth_dir, to_bam=True)
        print(f"truth export -> {truth_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    out = {}
    for label, with_graph in _ref_modes(args.ref_mode):
        references = {
            ref_id: build_reference_from_synthetic(
                pipeline, ref, with_graph=with_graph, kmer_size=kmer_size
            )
            for ref_id, ref in sorted(ds.references.items())
        }
        by_ref: dict[int, list] = {}
        for read in ds.splits.get(args.split, []):
            by_ref.setdefault(read.ref_id, []).append(read)
        batches = []
        for ref_id, reads in sorted(by_ref.items()):
            for group in chunk(reads, args.batch_size):
                batches.append((group, references[ref_id]))
        refs_meta = {rid: (ds.references[rid].seq, f"ref{rid}") for rid in references}
        out[label] = (batches, refs_meta, True)
    return out, None


def build_real_eval(args, pipeline, model_cfg):
    if not args.reference_fasta:
        sys.exit("ERROR: --data real needs --reference-fasta")
    kmer_size = model_cfg.graph_encoder.kmer_size
    modes = _ref_modes(args.ref_mode)
    if any(w for _, w in modes) and not args.gfa:
        sys.exit("ERROR: --ref-mode pangenome/both needs --gfa (a real GFA graph)")

    # For "both" the FASTA is read once and both indexes share its ref_seq; for a
    # single mode we build just that one.
    if args.ref_mode == "both":
        references = build_dual_reference_from_files(
            pipeline, args.reference_fasta, gfa=args.gfa,
            contig=args.contig, region=args.region, kmer_size=kmer_size,
        )
    else:
        references = {}
        for label, with_graph in modes:
            references[label] = build_reference_from_files(
                pipeline, args.reference_fasta,
                gfa=args.gfa if with_graph else None,
                contig=args.contig, region=args.region,
                with_graph=with_graph, kmer_size=kmer_size,
            )
    anchor_ref = references[modes[0][0]]
    reads, has_truth = load_real_reads(
        truth_bam=args.truth_bam, reads=args.reads_files or None,
        modality=args.modality, region=args.region,
        max_reads=args.max_reads or None, reference=anchor_ref,
        require_truth=False, layout=args.read_layout,
    )
    if not reads:
        sys.exit("ERROR: no reads to evaluate (check --truth-bam/--reads-file/--region)")
    print(f"real reads={len(reads)} truth={has_truth} "
          f"contig={anchor_ref.contig} window={anchor_ref.length:,}bp")

    out = {}
    for label, _ in modes:
        rr = references[label]
        batches = build_batches(reads, rr, args.batch_size)
        refs_meta = {rr.ref_id: (rr.ref_seq, rr.contig)}
        out[label] = (batches, refs_meta, has_truth)

    # A single pass over both references, sharing the read encoding, produces
    # the integrated (concordance-picked) BAM alongside the per-mode outputs.
    dual_ctx = None
    if args.ref_mode == "both" and not args.no_integrate:
        dual_ctx = {
            "reads": reads,
            "references": {label: references[label].reference for label, _ in modes},
            "refs_meta": {anchor_ref.ref_id: (anchor_ref.ref_seq, anchor_ref.contig)},
        }
    return out, dual_ctx


def build_real_manifest_eval(args, pipeline, model_cfg):
    """Build eval batches from a training-style manifest of windows.

    Mirrors ``train.py``'s manifest handling (global defaults inherited by
    entries) but evaluates **all** reads in each entry — there is no train/val
    split — so a held-out *test* manifest is scored in full and the per-window
    metrics are aggregated into one number per ref-mode. A reference/graph is
    built once per (fasta, gfa, region) and reused across modalities.
    """
    with open(args.manifest) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        globals_, entries = {}, list(raw)
    else:
        entries = list(raw.get("entries", []))
        globals_ = {k: v for k, v in raw.items() if k != "entries"}
    if not entries:
        sys.exit(f"ERROR: manifest {args.manifest} lists no entries")
    if args.max_entries:
        entries = entries[: args.max_entries]

    kmer_size = model_cfg.graph_encoder.kmer_size
    modes = _ref_modes(args.ref_mode)

    def _resolve(entry, key, cli_attr=None, default=None):
        if entry.get(key) is not None:
            return entry[key]
        if globals_.get(key) is not None:
            return globals_[key]
        if cli_attr is not None:
            val = getattr(args, cli_attr, None)
            if val is not None:
                return val
        return default

    ref_cache: dict = {}
    per_mode: dict[str, list] = {label: [] for label, _ in modes}
    refs_meta: dict = {}
    has_truth_any = False
    n_reads_total = 0
    n_entries_ok = 0

    for entry in progress(entries, desc="build eval entries", unit="entry"):
        fasta = _resolve(entry, "reference_fasta", "reference_fasta")
        gfa = _resolve(entry, "gfa", "gfa")
        region = _resolve(entry, "region", "region")
        contig = _resolve(entry, "contig", "contig")
        modality = _resolve(entry, "modality", "modality", "illumina")
        truth_bam = _resolve(entry, "truth_bam", "truth_bam")
        layout = _resolve(entry, "read_layout", "read_layout", "auto")
        max_reads = entry.get("max_reads", args.max_reads) or None
        if not fasta:
            sys.exit("ERROR: manifest eval needs reference_fasta "
                     "(entry, manifest global, or --reference-fasta)")
        if any(w for _, w in modes) and not gfa:
            sys.exit(f"ERROR: ref-mode pangenome/both needs a GFA graph "
                     f"(entry region={region}); set the manifest 'gfa' field")

        references: dict[str, object] = {}
        for label, with_graph in modes:
            key = (os.path.abspath(fasta),
                   os.path.abspath(gfa) if (gfa and with_graph) else None,
                   region, contig, label)
            rr = ref_cache.get(key)
            if rr is None:
                rr = build_reference_from_files(
                    pipeline, fasta, gfa=gfa if with_graph else None,
                    contig=contig, region=region,
                    with_graph=with_graph, kmer_size=kmer_size,
                )
                ref_cache[key] = rr
            references[label] = rr

        anchor_ref = references[modes[0][0]]
        reads, has_truth = load_real_reads(
            truth_bam=truth_bam, reads=None,
            modality=modality, region=region,
            max_reads=max_reads, reference=anchor_ref,
            require_truth=True, layout=layout,
        )
        if not reads:
            continue
        has_truth_any = has_truth_any or has_truth
        n_reads_total += len(reads)
        n_entries_ok += 1
        for label, _ in modes:
            rr = references[label]
            per_mode[label] += build_batches(reads, rr, args.batch_size)
            refs_meta.setdefault(rr.ref_id, (rr.ref_seq, rr.contig))

    print(f"manifest eval: entries={n_entries_ok}/{len(entries)}  "
          f"reads={n_reads_total}  truth={has_truth_any}  ref-mode={args.ref_mode}")
    if n_reads_total == 0:
        sys.exit("ERROR: no reads loaded from any manifest entry "
                 "(check truth_bam paths / regions).")
    out = {label: (per_mode[label], refs_meta, has_truth_any) for label, _ in modes}
    return out, None


def _subsample_batches(batches, n):
    """Evenly sample ``n`` batches across the list (keeps window/modality mix).

    Taking the first ``n`` would bias toward whichever windows/modalities sort
    first (e.g. all Illumina), so we pick evenly-spaced indices instead.
    """
    if n <= 0 or len(batches) <= n:
        return batches
    import numpy as np
    idx = sorted(set(np.linspace(0, len(batches) - 1, n).round().astype(int).tolist()))
    return [batches[i] for i in idx]


def _sweep_checkpoints(args, model, trainer, per_mode, split_tag, accel) -> int:
    """Score every checkpoint in ``args.checkpoints`` on the same batches.

    The batches (references + reads) are built once by the caller; here we only
    swap each checkpoint's weights into the shared ``model`` and re-run
    validation, then print an accuracy-vs-checkpoint table and save the raw
    numbers. Metrics only — no predicted BAM.
    """
    cols = ("locus_accuracy", "mapped_fraction", "anchor_auc",
            "chain_accuracy", "mapq_mae", "loss")
    results: dict[str, dict] = {}
    rows: list[tuple] = []
    for ckpt in args.checkpoints:
        if not os.path.isfile(ckpt):
            print(f"  skip (not found): {ckpt}")
            continue
        meta = _load_checkpoint(ckpt, model)
        tag = os.path.basename(ckpt)
        ep = meta.get("epoch") if isinstance(meta, dict) else None
        print(f"\n=== checkpoint {tag}" + (f" (epoch={ep})" if ep is not None else "") + " ===")
        results[tag] = {"epoch": ep, "modes": {}}
        for label, (batches, _refs_meta, has_truth) in per_mode.items():
            if not batches:
                continue
            if not has_truth:
                print(f"  [{label}] no truth — skipping metrics")
                continue
            m = trainer.validate(batches)
            print(f"  [{label}] {m.one_line()}")
            d = m.as_dict()
            d["n_chain_scored"] = m.n_chain_scored
            results[tag]["modes"][label] = d
            rows.append((tag, label, *[d.get(c) for c in cols]))

    if rows:
        header = ["checkpoint", "mode", "locus", "mapped", "anchorAUC",
                  "chain", "mapqMAE", "loss"]
        print("\n" + "=" * 74)
        print("held-out sweep")
        print("=" * 74)
        print("  ".join(f"{h:>10}" for h in header))
        for tag, label, locus, mapped, auc, chain, mae, loss in rows:
            def _f(x, pct=False):
                if x is None:
                    return f"{'-':>10}"
                return f"{x*100:>9.1f}%" if pct else f"{x:>10.3f}"
            print("  ".join([
                f"{tag:>10}", f"{label:>10}",
                _f(locus, True), _f(mapped, True), _f(auc),
                _f(chain, True), _f(mae), _f(loss),
            ]))

    out_path = os.path.join(args.out, "sweep_metrics.json")
    with open(out_path, "w") as fh:
        json.dump({
            "manifest": os.path.abspath(args.manifest) if args.manifest else None,
            "ref_mode": args.ref_mode,
            "split": split_tag,
            "device": accel.summary(),
            "checkpoints": list(args.checkpoints),
            "results": results,
        }, fh, indent=2)
    print(f"\nsweep metrics -> {out_path}")
    print("Saved under", os.path.abspath(args.out))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=("synthetic", "real"), default=None,
                   help="data source (default: real if --reference-fasta given)")
    # synthetic
    p.add_argument("--preset", default="tiny", choices=("tiny", "long", "table1"))
    p.add_argument("--reads", type=int, default=0,
                   help="override synthetic preset read counts (0 = preset)")
    p.add_argument("--split", default="test", choices=("train", "val", "test"),
                   help="which synthetic split to evaluate")
    p.add_argument("--emit-truth", action="store_true",
                   help="(synthetic) also export truth FASTA/FASTQ/SAM/GFA/BAM")
    # real
    p.add_argument("--manifest", default=None,
                   help="JSON manifest of windows (same shape as train.py's "
                        "--manifest). Evaluates ALL reads in every entry (no "
                        "train/val split), so a held-out *test* manifest is "
                        "scored end to end and metrics are aggregated across "
                        "windows. Mutually exclusive with a single "
                        "--reference-fasta/--region window.")
    p.add_argument("--max-entries", type=int, default=0,
                   help="cap manifest evaluation to the first N entries "
                        "(0 = all). Useful for a quick look before the full run.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap validation to N batches per ref-mode, sampled "
                        "EVENLY across all windows (0 = all). The end-to-end "
                        "locus report runs the full aligner per batch (~1-3 "
                        "min/batch), so a whole held-out set can take days; "
                        "40-80 batches gives a representative estimate in "
                        "minutes. Strongly recommended for a sweep.")
    p.add_argument("--reference-fasta", default=None)
    p.add_argument("--gfa", default=None)
    p.add_argument("--truth-bam", default=None,
                   help="aligned truth BAM/SAM/CRAM (enables locus/MAPQ metrics)")
    p.add_argument("--reads-file", dest="reads_files", action="append", default=None,
                   help="FASTQ/BAM reads; for Illumina repeat R1 then R2")
    p.add_argument("--read-layout", default="auto",
                   choices=("auto", "single", "paired"),
                   help="auto pairs two Illumina FASTQs; long reads stay single")
    p.add_argument("--region", default=None,
                   help="samtools-style window, e.g. chr21:5000000-6000000")
    p.add_argument("--contig", default=None)
    p.add_argument("--modality", default="illumina")
    p.add_argument("--max-reads", type=int, default=0)
    p.add_argument("--hard-mapq-threshold", type=int, default=20,
                   help="baseline (truth-BAM) MAPQ at/above which a read is "
                        "'easy' for the easy/hard locus-accuracy split reported "
                        "in metrics.json; below it is the 'hard' fraction "
                        "(match the value used at training time)")
    p.add_argument("--max-read-len", type=int, default=None,
                   help="truncate each read to this many bases before encoding "
                        "(match training; required for long reads to avoid OOM)")
    # shared
    p.add_argument("--checkpoint", default=None,
                   help="path to checkpoint.pt (optional for --mode fast)")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="evaluate SEVERAL checkpoints against the same batches "
                        "and print a metrics-vs-checkpoint table (e.g. "
                        "epoch_00.pt epoch_05.pt epoch_11.pt). The eval set is "
                        "built once and each checkpoint's weights are swapped "
                        "in; metrics only (no BAM). Overrides --checkpoint. All "
                        "checkpoints must share the same d_model.")
    p.add_argument("--ref-mode", default="both",
                   choices=("linear", "pangenome", "both"))
    p.add_argument("--mode", default="hybrid",
                   choices=("fast", "hybrid", "two_pass"),
                   help="pipeline mode; hybrid/two_pass use the model when loaded")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None,
                   help="accepted for CLI compatibility with train.py; ignored "
                        "(evaluation does not train)")
    p.add_argument("--d-model", type=int, default=64,
                   help="must match the checkpoint's d_model when loading weights")
    p.add_argument("--device", default=None)
    p.add_argument("--require-gpu", action="store_true",
                   help="hard-fail instead of silently running on CPU")
    p.add_argument("--workers", type=int, default=0,
                   help="host worker threads for seeding/chaining + prefetch "
                        "(0 = all CPU cores)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="batches to build ahead on background threads (0 = off)")
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model forward (CUDA/XPU)")
    p.add_argument("--cuda-graphs", dest="cuda_graphs", action="store_true",
                   default=True,
                   help="capture fixed-shape inference into CUDA Graphs (default on)")
    p.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false",
                   help="disable CUDA Graph capture")
    p.add_argument("--fp8", dest="fp8", action="store_true", default=True,
                   help="TransformerEngine FP8 when supported; Ampere → BF16")
    p.add_argument("--no-fp8", dest="fp8", action="store_false")
    p.add_argument("--tensorrt", action="store_true",
                   help="compile inference with torch_tensorrt / TensorRT")
    p.add_argument("--out", default="data/eval_runs/latest")
    p.add_argument("--no-bam", action="store_true",
                   help="skip writing predicted BAM/SAM")
    p.add_argument("--no-integrate", action="store_true",
                   help="(--ref-mode both) skip the single-pass integrated BAM "
                        "that folds linear + pangenome into one concordance call")
    p.add_argument("--write-cram", action="store_true",
                   help="also write predicted CRAM (needs reference FASTA)")
    args = p.parse_args()

    if args.data is None:
        args.data = "real" if (args.reference_fasta or args.manifest) else "synthetic"
    if args.checkpoints:
        # A sweep swaps weights and re-scores; a merged BAM per checkpoint is
        # meaningless, so force metrics-only.
        args.no_bam = True
    if args.manifest:
        # Many windows share the same contig (chr21) at different offsets, so a
        # single merged predicted BAM would have colliding @SQ/coordinates.
        # Manifest eval therefore reports aggregated metrics only; use a single
        # --reference-fasta/--region window if you need a predicted BAM.
        if not args.no_bam:
            print("note: --manifest eval writes aggregated metrics only "
                  "(predicted BAM skipped across windows)")
        args.no_bam = True
    if args.data == "synthetic" and not args.reference_fasta:
        print("note: --data synthetic is for smoke tests; "
              "production eval uses --data real (see prepare_real_hg002.sh)")

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    d_model = args.d_model
    if args.checkpoints:
        d_model = _d_model_from_checkpoint(args.checkpoints[0], args.d_model)
    elif args.checkpoint:
        d_model = _d_model_from_checkpoint(args.checkpoint, args.d_model)
    model_cfg = GraphMambaConfig(d_model=d_model)

    model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=model_cfg)
    ).model
    ckpt_meta: dict = {}
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
        ckpt_meta = _load_checkpoint(args.checkpoint, model)
        print(f"loaded checkpoint: {args.checkpoint}"
              + (f"  (best_epoch={ckpt_meta.get('best_epoch')})"
                 if "best_epoch" in ckpt_meta else ""))
    elif args.mode != "fast":
        print("note: no --checkpoint; hybrid/two_pass degrade to classical scoring")

    try:
        accel = AccelContext(AccelConfig(
            device=args.device,
            num_workers=args.workers,
            prefetch=args.prefetch,
            compile=args.compile,
            cuda_graphs=args.cuda_graphs,
            fp8=args.fp8,
            tensorrt=args.tensorrt,
        ))
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"device: {accel.summary()}")
    if accel.caps.device.type == "cpu":
        cuda_built = bool(getattr(torch.version, "cuda", None))
        reason = ("this torch build has no CUDA support (CPU-only wheel)"
                  if not cuda_built else
                  "CUDA is built into torch but no GPU is visible "
                  "(driver missing or container started without --gpus)")
        banner = ("\n" + "!" * 74 +
                  "\n! WARNING: running on CPU — the GPU will show 0% utilisation.\n"
                  f"!   reason: {reason}.\n"
                  "!   fix: install a CUDA torch wheel and pass `--device cuda`.\n" +
                  "!" * 74 + "\n")
        if args.require_gpu:
            sys.exit(banner + "ERROR: --require-gpu was set but no GPU is usable.")
        print(banner)
    pipeline = build_pipeline(
        PipelineConfig(mode=args.mode, batch_size=args.batch_size,
                       max_read_len=args.max_read_len),
        model=model,
        accel=accel,
    )

    if args.data == "real":
        if args.manifest:
            per_mode, dual_ctx = build_real_manifest_eval(args, pipeline, model_cfg)
        else:
            per_mode, dual_ctx = build_real_eval(args, pipeline, model_cfg)
        split_tag = "real"
    else:
        per_mode, dual_ctx = build_synthetic_eval(args, pipeline, model_cfg)
        split_tag = args.split

    trainer = Trainer(
        model, pipeline,
        cfg=TrainConfig(out_dir=args.out, save_checkpoint=False,
                        save_every_epoch=False, save_last=False,
                        checkpoint_history=False,
                        hard_mapq_threshold=args.hard_mapq_threshold),
        loss_cfg=LossConfig(), accel=accel, verbose=False,
    )

    if args.max_batches:
        capped = {}
        for label, (batches, refs_meta, has_truth) in per_mode.items():
            sub = _subsample_batches(batches, args.max_batches)
            if len(sub) != len(batches):
                print(f"[{label}] locus report capped: {len(sub)}/{len(batches)} "
                      f"batches (evenly sampled; --max-batches={args.max_batches})")
            capped[label] = (sub, refs_meta, has_truth)
        per_mode = capped

    print("=" * 74)
    print(f"Evaluation  data={args.data}  split={split_tag}  "
          f"ref-mode={args.ref_mode}  pipeline={args.mode}")
    print("=" * 74)

    if args.checkpoints:
        return _sweep_checkpoints(args, model, trainer, per_mode, split_tag, accel)

    all_metrics: dict[str, dict] = {}

    for label, (batches, refs_meta, has_truth) in per_mode.items():
        if not batches:
            print(f"[{label}] no reads; skipping")
            continue
        mode_dir = os.path.join(args.out, label)
        os.makedirs(mode_dir, exist_ok=True)

        if has_truth:
            metrics = trainer.validate(batches)
            print(f"[{label}] {metrics.one_line()}")
            detail = metrics.as_dict()
            detail.update({
                "ref_mode": label, "split": split_tag, "n_batches": len(batches),
                "anchor_precision": metrics.anchor_precision,
                "anchor_recall": metrics.anchor_recall,
                "mapq_expected_error": metrics.mapq_expected_error,
                "mapq_observed_error": metrics.mapq_observed_error,
                "n_chain_scored": metrics.n_chain_scored,
                "loss_terms": metrics.terms,
            })
            all_metrics[label] = detail
            with open(os.path.join(mode_dir, "metrics.json"), "w") as fh:
                json.dump(detail, fh, indent=2)
        else:
            print(f"[{label}] no truth supplied — writing predicted BAM only "
                  f"(no locus/MAPQ metrics)")
            all_metrics[label] = {"ref_mode": label, "split": split_tag,
                                  "n_batches": len(batches), "has_truth": False}

        if not args.no_bam:
            _write_pred_bam(
                mode_dir, split_tag, batches, pipeline, refs_meta,
                write_cram=args.write_cram,
                reference_fasta=args.reference_fasta,
            )

    if dual_ctx is not None and not args.no_bam:
        print(f"[integrated] one-pass linear+pangenome concordance BAM")
        _write_integrated_bam(args.out, split_tag, dual_ctx, pipeline)

    summary_path = os.path.join(args.out, "metrics.json")
    summary = {
        "data": args.data,
        "preset": args.preset if args.data == "synthetic" else None,
        "reference_fasta": os.path.abspath(args.reference_fasta)
        if args.reference_fasta else None,
        "gfa": os.path.abspath(args.gfa) if args.gfa else None,
        "truth_bam": os.path.abspath(args.truth_bam) if args.truth_bam else None,
        "region": args.region,
        "split": split_tag,
        "ref_mode": args.ref_mode,
        "pipeline_mode": args.mode,
        "checkpoint": args.checkpoint,
        "device": accel.summary(),
        "modes": all_metrics,
        "artifacts": {
            "metrics": "metrics.json + <mode>/metrics.json",
            "pred_bam": "<mode>/pred.<split>.bam (+ .bai)",
            "integrated_bam": "integrated/pred.<split>.bam (+ .bai, --ref-mode both)",
            "pred_sam": "<mode>/pred.<split>.sam",
            "pred_cram": "<mode>/pred.<split>.cram (if --write-cram)",
            "run_meta": "run_meta.json",
        },
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsummary -> {summary_path}")
    meta_path = os.path.join(args.out, "run_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"run meta -> {meta_path}")
    print("Saved under", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
