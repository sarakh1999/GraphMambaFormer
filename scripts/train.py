#!/usr/bin/env python3
"""Train the GraphMamba alignment model on **real** or synthetic references.

Two data sources, one training loop:

* ``--data real`` — real files on disk: a reference **FASTA** (linear genome),
  an optional pangenome **GFA** graph, plus either an aligned **truth
  BAM/SAM/CRAM** *or* unaligned ``--reads-file`` FASTQ/BAM. Without
  ``--truth-bam``, the classical ``fast`` aligner generates pseudo-labels
  (locus / CIGAR / MAPQ) so training can still run on real reads. Prefer a
  real truth BAM when available. (auto-selected when ``--reference-fasta``
  is given.)
* ``--data synthetic`` (default) — the fully-labelled synthetic dataset, handy
  for a CPU smoke test.

``--ref-mode`` picks the curriculum for either source:

* ``linear``    — index the FASTA sequence only (graph towers idle).
* ``pangenome`` — also attach the GFA graph (real) / synthetic graph.
* ``both``      — train each read once on linear and once on pangenome.

Everything is saved under ``--out``: per-epoch checkpoints
(``checkpoints/epoch_XX.pt``), a rolling ``last.pt``, the best ``checkpoint.pt``,
``history.json`` (every step + validation), ``run_meta.json`` (full provenance),
and the plots.

Examples
--------
    # REAL linear: chr21 window, GIAB truth BAM, on GPU
    PYTHONPATH=. python scripts/train.py --data real \
        --reference-fasta data/chr21/HG002/ref/GRCh38.chr21.fa \
        --truth-bam data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam \
        --region chr21:5000000-6000000 --ref-mode linear \
        --device cuda --epochs 20 --batch-size 8 --d-model 256 \
        --out data/training_runs/chr21_linear

    # REAL without truth BAM: FASTQ + ref → classical pseudo-labels → train
    PYTHONPATH=. python scripts/train.py --data real \
        --reference-fasta data/chr21/HG005/ref/GRCh38.chr21.fa \
        --reads-file data/chr21/HG005/reads/HG005.chr21.R1.fastq.gz \
        --reads-file data/chr21/HG005/reads/HG005.chr21.R2.fastq.gz \
        --read-layout paired --modality illumina --ref-mode linear \
        --region chr21 --device cuda --require-gpu \
        --epochs 20 --batch-size 8 --d-model 256 \
        --out data/training_runs/hg005_illumina_pseudo

    # SYNTHETIC CPU smoke (linear + pangenome)
    PYTHONPATH=. python scripts/train.py --preset tiny --epochs 3
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
    build_reference_from_files,
    build_reference_from_synthetic,
    generate_dataset,
    load_real_reads,
    preset,
)
from graphmambaformer.models import build_core_model
from graphmambaformer.training import TrainConfig, Trainer, plot_all, pseudo_label_reads


def chunk(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ref_modes(mode: str) -> list[tuple[str, bool]]:
    """Return ``(label, with_graph)`` pairs for the chosen curriculum."""
    if mode == "linear":
        return [("linear", False)]
    if mode == "pangenome":
        return [("pangenome", True)]
    if mode == "both":
        return [("linear", False), ("pangenome", True)]
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# synthetic data path
# --------------------------------------------------------------------------- #
def build_synthetic(args, pipeline, model_cfg):
    spec = preset(args.preset)
    if args.reads:
        spec = replace(spec, n_train=args.reads, n_val=max(2, args.reads // 4),
                       n_test=max(2, args.reads // 4))
    ds = generate_dataset(spec)
    kmer_size = model_cfg.graph_encoder.kmer_size
    modes = _ref_modes(args.ref_mode)

    references: dict[tuple[str, int], object] = {}
    for label, with_graph in modes:
        for ref_id, ref in sorted(ds.references.items()):
            references[(label, ref_id)] = build_reference_from_synthetic(
                pipeline, ref, with_graph=with_graph, kmer_size=kmer_size
            )

    def batches_for(split: str) -> list[tuple]:
        out: list[tuple] = []
        by_ref: dict[int, list] = {}
        for read in ds.splits.get(split, []):
            by_ref.setdefault(read.ref_id, []).append(read)
        for label, _ in modes:
            for ref_id, reads in sorted(by_ref.items()):
                key = (label, ref_id)
                if key not in references:
                    continue
                for group in chunk(reads, args.batch_size):
                    out.append((group, references[key]))
        return out

    train_batches = batches_for("train")
    val_batches = batches_for("val") or batches_for("test")
    if not sum(len(r) for r, _ in val_batches):
        cut = max(1, int(len(train_batches) * 0.8))
        train_batches, val_batches = train_batches[:cut], train_batches[cut:]

    graph_nodes = [
        f"{len(r.graph.node_seqs)}n/{len(r.graph.edge_index)}e"
        for _, r in sorted(ds.references.items())
    ]
    info = {
        "source": "synthetic",
        "preset": args.preset,
        "references": {
            str(rid): {"length": len(ref.seq)}
            for rid, ref in sorted(ds.references.items())
        },
        "pangenome_graphs": graph_nodes,
    }
    print(f"source: synthetic  preset={args.preset}")
    print(f"references: {len(ds.references)} "
          f"({', '.join(f'{len(r.seq):,}bp' for _, r in sorted(ds.references.items()))})")
    print(f"pangenome graphs: {', '.join(graph_nodes)}")
    return train_batches, val_batches, info


# --------------------------------------------------------------------------- #
# real data path (FASTA + optional GFA + truth BAM)
# --------------------------------------------------------------------------- #
def build_real(args, pipeline, model_cfg):
    if not args.reference_fasta:
        sys.exit("ERROR: --data real needs --reference-fasta")
    kmer_size = model_cfg.graph_encoder.kmer_size
    modes = _ref_modes(args.ref_mode)
    if any(with_graph for _, with_graph in modes) and not args.gfa:
        sys.exit("ERROR: --ref-mode pangenome/both needs --gfa (a real GFA graph)")

    # Build one reference per curriculum label. They share the same FASTA window,
    # so the truth-BAM offset (below) is identical for both.
    references: dict[str, object] = {}
    for label, with_graph in modes:
        rr = build_reference_from_files(
            pipeline, args.reference_fasta,
            gfa=args.gfa if with_graph else None,
            contig=args.contig, region=args.region,
            with_graph=with_graph, kmer_size=kmer_size,
        )
        references[label] = rr
        extra = (f"  graph={rr.n_nodes}n/{rr.n_edges}e" if rr.with_graph else "")
        print(f"[{label}] reference contig={rr.contig} len={rr.length:,}bp "
              f"offset={rr.offset}{extra}")

    # Labels: prefer --truth-bam; otherwise classical pseudo-labels from FASTQ.
    anchor_ref = references[modes[0][0]]
    label_source = "truth_bam"
    pseudo_bam = None
    if args.truth_bam:
        reads, has_truth = load_real_reads(
            truth_bam=args.truth_bam, reads=args.reads_files or None,
            modality=args.modality, region=args.region,
            max_reads=args.max_reads or None, reference=anchor_ref,
            require_truth=True, layout=args.read_layout,
        )
    else:
        if not args.reads_files:
            sys.exit(
                "ERROR: real training needs --truth-bam or --reads-file "
                "(FASTQ/BAM). Without --truth-bam the classical aligner "
                "builds pseudo-labels from --reads-file."
            )
        raw_reads, _ = load_real_reads(
            reads=args.reads_files,
            modality=args.modality, region=args.region,
            max_reads=args.max_reads or None, reference=anchor_ref,
            require_truth=False, layout=args.read_layout,
            reference_fasta=args.reference_fasta,
            as_sequences=True,
        )
        if not raw_reads:
            sys.exit(
                "ERROR: no reads loaded from --reads-file inside "
                f"({args.region or args.contig or 'full contig'})."
            )
        print(
            "note: no --truth-bam — generating pseudo-labels from classical "
            "aligner (distillation, not GIAB gold truth)"
        )
        pseudo_bam = os.path.join(args.out, "pseudo_truth.bam")
        reads = pseudo_label_reads(
            raw_reads,
            anchor_ref.reference,
            modality=args.modality,
            batch_size=args.batch_size,
            write_bam=pseudo_bam,
            references={anchor_ref.ref_id: type("R", (), {"seq": anchor_ref.ref_seq})()},
            contig_names={anchor_ref.ref_id: anchor_ref.contig},
            reference_fasta=args.reference_fasta,
        )
        has_truth = True
        label_source = "pseudo_classical"

    if not reads:
        sys.exit(
            "ERROR: no labelled reads inside the reference window "
            f"({args.region or args.contig or 'full contig'}). "
            "Widen --region, check --truth-bam / --reads-file, or ensure "
            "the classical aligner can map some reads."
        )

    # Split reads into train / val once, then batch each split under every mode.
    n_val = max(1, int(len(reads) * args.val_fraction))
    val_reads, train_reads = reads[:n_val], reads[n_val:]
    if not train_reads:                      # tiny sets: keep at least one train
        train_reads, val_reads = reads, reads[:1]

    train_batches: list[tuple] = []
    val_batches: list[tuple] = []
    for label, _ in modes:
        train_batches += build_batches(train_reads, references[label], args.batch_size)
        val_batches += build_batches(val_reads, references[label], args.batch_size)

    info = {
        "source": "real",
        "reference_fasta": os.path.abspath(args.reference_fasta),
        "gfa": os.path.abspath(args.gfa) if args.gfa else None,
        "truth_bam": os.path.abspath(args.truth_bam) if args.truth_bam else None,
        "pseudo_truth_bam": os.path.abspath(pseudo_bam) if pseudo_bam else None,
        "label_source": label_source,
        "region": args.region,
        "contig": {label: references[label].contig for label, _ in modes},
        "modality": args.modality,
        "has_truth": has_truth,
        "n_reads_total": len(reads),
        "n_train_reads": len(train_reads),
        "n_val_reads": len(val_reads),
        "references": {
            label: {
                "length": references[label].length,
                "offset": references[label].offset,
                "with_graph": references[label].with_graph,
                "n_nodes": references[label].n_nodes,
                "n_edges": references[label].n_edges,
            }
            for label, _ in modes
        },
    }
    print(
        f"source: real  reads={len(reads)} truth={has_truth} "
        f"labels={label_source} "
        f"(train {len(train_reads)} / val {len(val_reads)})"
    )
    if pseudo_bam:
        print(f"pseudo truth BAM: {pseudo_bam}")
    return train_batches, val_batches, info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=("synthetic", "real"), default=None,
                   help="data source (default: real if --reference-fasta given, "
                        "else synthetic)")
    # --- synthetic knobs --- #
    p.add_argument("--preset", default="tiny", choices=("tiny", "long", "table1"),
                   help="synthetic dataset preset")
    p.add_argument("--reads", dest="reads", type=int, default=0,
                   help="override the synthetic preset's read count (0 = preset)")
    # --- real-data knobs --- #
    p.add_argument("--reference-fasta", default=None,
                   help="real reference FASTA (linear genome / windowed contig)")
    p.add_argument("--gfa", default=None,
                   help="real pangenome GFA graph (needed for pangenome/both)")
    p.add_argument("--truth-bam", default=None,
                   help="aligned truth BAM/SAM/CRAM (preferred for real training). "
                        "If omitted, pass --reads-file and classical pseudo-labels "
                        "are generated automatically")
    p.add_argument("--reads-file", dest="reads_files", action="append", default=None,
                   help="FASTQ/BAM reads; for Illumina repeat R1 then R2. "
                        "With --truth-bam optional; without it these are mapped "
                        "by the classical aligner to build training labels")
    p.add_argument("--read-layout", default="auto",
                   choices=("auto", "single", "paired"),
                   help="auto pairs two Illumina FASTQs; long reads stay single")
    p.add_argument("--region", default=None,
                   help="samtools-style window, e.g. chr21:5000000-6000000 "
                        "(strongly recommended: the reference pipeline is pure "
                        "Python and indexing a whole chromosome is slow)")
    p.add_argument("--contig", default=None,
                   help="named contig when the FASTA has several (default: first)")
    p.add_argument("--modality", default="illumina",
                   help="read modality (illumina / pacbio_hifi / ont / ...)")
    p.add_argument("--max-reads", type=int, default=0,
                   help="cap number of real reads (0 = all in the window)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction of real reads held out for validation")
    # --- training knobs --- #
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=64,
                   help="small by default so a CPU run finishes quickly")
    p.add_argument("--device", default=None,
                   help="cuda / cuda:N / mps / xpu / cpu (default: auto)")
    p.add_argument("--devices", default="auto",
                   help="multi-GPU device list: 'auto' (all visible CUDA/XPU "
                        "when count>1), 'all', '0,1,2', or 'none' for single-GPU")
    p.add_argument("--require-gpu", action="store_true",
                   help="hard-fail instead of silently training on CPU (use in "
                        "GPU jobs so a CPU-only torch wheel is caught immediately)")
    p.add_argument("--workers", type=int, default=0,
                   help="host worker threads for seeding/chaining + the batch "
                        "prefetch that keeps the GPU fed (0 = all CPU cores)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="batches to build ahead on background threads so the GPU "
                        "is not starved by host-side seeding (0 disables)")
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model forward (CUDA/XPU; big speed-up "
                        "after warmup, skipped automatically on MPS)")
    p.add_argument("--cuda-graphs", dest="cuda_graphs", action="store_true",
                   default=True,
                   help="capture fixed-shape inference into CUDA Graphs (default on)")
    p.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false",
                   help="disable CUDA Graph capture")
    p.add_argument("--fp8", dest="fp8", action="store_true", default=True,
                   help="use TransformerEngine FP8 when the GPU supports it "
                        "(Ada/Hopper); Ampere falls back to BF16")
    p.add_argument("--no-fp8", dest="fp8", action="store_false",
                   help="disable FP8 even on Hopper/Ada")
    p.add_argument("--tensorrt", action="store_true",
                   help="compile inference with torch_tensorrt / TensorRT (lazy)")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--monitor", default="locus_accuracy",
                   help="early-stopping metric (falls back to -val_loss if the "
                        "chosen metric is unmeasurable on this data)")
    p.add_argument("--ref-mode", default="both",
                   choices=("linear", "pangenome", "both"),
                   help="train on linear genome, pangenome graph, or both")
    p.add_argument("--out", default="data/training_runs/latest")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="skip writing the best checkpoint.pt (per-epoch and "
                        "last.pt are still written)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Resolve the source: real when a FASTA is supplied, else synthetic.
    if args.data is None:
        args.data = "real" if args.reference_fasta else "synthetic"

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 74)
    print("Building dataset and model")
    print("=" * 74)

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
    from graphmambaformer.accel import list_visible_gpus
    visible = list_visible_gpus()
    for g in visible:
        cc = g.get("compute_capability")
        mem = g.get("total_memory_gb")
        bits = [g["name"]]
        if cc:
            bits.append(f"sm_{cc[0]}{cc[1]}")
        if mem:
            bits.append(f"{mem} GiB")
        print(f"  gpu[{g['index']}]: {', '.join(bits)}")

    # Loud guard against the #1 cause of "GPU utilisation is 0": the process is
    # silently on the CPU (a CPU-only torch wheel, a container started without
    # --gpus, or CUDA_VISIBLE_DEVICES="") so nothing ever reaches the device.
    if accel.caps.device.type == "cpu":
        cuda_built = bool(getattr(torch.version, "cuda", None))
        reason = ("this torch build has no CUDA support (CPU-only wheel)"
                  if not cuda_built else
                  "CUDA is built into torch but no GPU is visible "
                  "(driver missing, container without --gpus, or "
                  "CUDA_VISIBLE_DEVICES is empty)")
        banner = (
            "\n" + "!" * 74 +
            "\n! WARNING: running on CPU — the GPU will show 0% utilisation.\n"
            f"!   reason: {reason}.\n"
            "!   fix: install a CUDA torch wheel and launch with `--device cuda`\n"
            "!        (in Docker: `docker run --gpus all ...`).\n" +
            "!" * 74 + "\n"
        )
        if args.require_gpu:
            sys.exit(banner + "ERROR: --require-gpu was set but no GPU is usable.")
        print(banner)
    elif args.require_gpu and accel.caps.device.type == "mps":
        print("note: --require-gpu is satisfied by Apple MPS (not CUDA).")

    model_cfg = GraphMambaConfig(d_model=args.d_model)
    model = build_core_model(
        CoreModelConfig(arch="graphmamba", graphmamba=model_cfg)
    ).model
    pipeline = build_pipeline(
        PipelineConfig(mode="hybrid", batch_size=args.batch_size),
        model=model, accel=accel,
    )

    if args.data == "real":
        train_batches, val_batches, data_info = build_real(args, pipeline, model_cfg)
    else:
        train_batches, val_batches, data_info = build_synthetic(args, pipeline, model_cfg)

    n_train = sum(len(r) for r, _ in train_batches)
    n_val = sum(len(r) for r, _ in val_batches)
    print(f"ref-mode: {args.ref_mode}")
    print(f"reads: {n_train} train / {n_val} val   "
          f"batches: {len(train_batches)} train / {len(val_batches)} val")
    if args.ref_mode == "both":
        print("note: each read is trained once on the linear index and once "
              "on the pangenome index")

    print()
    print("=" * 74)
    print("Training")
    print("=" * 74)
    trainer = Trainer(
        model, pipeline,
        cfg=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, monitor=args.monitor, out_dir=args.out,
            save_checkpoint=not args.no_checkpoint,
            devices=args.devices,
        ),
        loss_cfg=LossConfig(),
        accel=accel,
        verbose=not args.quiet,
    )
    history = trainer.fit(train_batches, val_batches)

    json_path = history.to_json(os.path.join(args.out, "history.json"))
    print(f"\nhistory -> {json_path}")

    # Record how this run was configured so eval can mirror it exactly.
    meta_path = os.path.join(args.out, "run_meta.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "data": data_info,
            "ref_mode": args.ref_mode,
            "d_model": args.d_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "monitor": args.monitor,
            "device": accel.summary(),
            "devices": list(trainer.device_ids),
            "out_dir": os.path.abspath(args.out),
            "artifacts": {
                "best": "checkpoint.pt",
                "last": "last.pt",
                "per_epoch": "checkpoints/epoch_XX.pt",
                "history": "history.json",
                "plots": "plots/",
            },
        }, fh, indent=2)
    print(f"run meta -> {meta_path}")

    if not args.no_plots:
        print()
        plot_all(history, os.path.join(args.out, "plots"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
