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
from graphmambaformer.distributed import (
    main_process_first,
    maybe_init_distributed,
    shutdown_distributed,
)
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
from graphmambaformer.progress import progress
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
def _load_manifest(path: str) -> tuple[dict, list[dict]]:
    """Read a training manifest.

    Two shapes are accepted:
      * a bare JSON list of entry dicts, or
      * ``{"entries": [...], <global defaults>}`` where any top-level key
        (``reference_fasta`` / ``gfa`` / ``region`` / ``ref_mode`` / ...) is a
        default inherited by entries that do not set it themselves.
    """
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return {}, list(raw)
    entries = list(raw.get("entries", []))
    globals_ = {k: v for k, v in raw.items() if k != "entries"}
    return globals_, entries


def _resolve(entry: dict, globals_: dict, args, key: str,
             cli_attr: str | None = None, default=None):
    """entry value → manifest global → CLI flag → default."""
    if entry.get(key) is not None:
        return entry[key]
    if globals_.get(key) is not None:
        return globals_[key]
    if cli_attr is not None:
        val = getattr(args, cli_attr, None)
        if val is not None:
            return val
    return default


def _build_real_entry(entry, globals_, args, pipeline, modes, kmer_size,
                      ref_cache: dict):
    """Build train/val batches for a single (sample, modality) entry.

    ``ref_cache`` is shared across entries so a reference/graph (e.g. the chr21
    pangenome) is built once and reused by every sample and modality that maps
    to the same FASTA/GFA/region rather than rebuilt per entry.
    """
    fasta = _resolve(entry, globals_, args, "reference_fasta", "reference_fasta")
    if not fasta:
        sys.exit("ERROR: real training needs reference_fasta "
                 "(entry, manifest global, or --reference-fasta)")
    gfa = _resolve(entry, globals_, args, "gfa", "gfa")
    region = _resolve(entry, globals_, args, "region", "region")
    contig = _resolve(entry, globals_, args, "contig", "contig")
    modality = _resolve(entry, globals_, args, "modality", "modality", "illumina")
    truth_bam = _resolve(entry, globals_, args, "truth_bam", "truth_bam")
    layout = _resolve(entry, globals_, args, "read_layout", "read_layout", "auto")
    sample = entry.get("sample", "-")
    max_reads = entry.get("max_reads", args.max_reads)
    val_fraction = float(entry.get("val_fraction", args.val_fraction))
    reads_files = entry.get("reads_file") or entry.get("reads_files")
    if isinstance(reads_files, str):
        reads_files = [reads_files]
    if reads_files is None:
        reads_files = args.reads_files
    tag = f"{sample}/{modality}"

    if any(with_graph for _, with_graph in modes) and not gfa:
        sys.exit(f"ERROR: ref-mode pangenome/both needs a GFA graph "
                 f"(entry {tag}); set --gfa or the manifest 'gfa' field")

    # Build one reference per curriculum label, reusing the shared cache. The
    # cache key excludes modality on purpose — the reference is identical across
    # modalities that share the same FASTA/GFA/region.
    references: dict[str, object] = {}
    for label, with_graph in modes:
        key = (os.path.abspath(fasta),
               os.path.abspath(gfa) if (gfa and with_graph) else None,
               region, contig, label)
        rr = ref_cache.get(key)
        if rr is None:
            rr = _load_or_build_reference(
                args, pipeline, fasta, gfa, contig, region,
                with_graph, kmer_size,
            )
            ref_cache[key] = rr
            extra = (f"  graph={rr.n_nodes}n/{rr.n_edges}e" if rr.with_graph else "")
            print(f"[ref {label}] contig={rr.contig} len={rr.length:,}bp "
                  f"offset={rr.offset}{extra}")
        references[label] = rr

    anchor_ref = references[modes[0][0]]
    label_source = "truth_bam"
    pseudo_bam = None
    if truth_bam:
        reads, has_truth = load_real_reads(
            truth_bam=truth_bam, reads=reads_files or None,
            modality=modality, region=region,
            max_reads=max_reads or None, reference=anchor_ref,
            require_truth=True, layout=layout,
        )
    else:
        if not reads_files:
            sys.exit(
                f"ERROR: entry {tag} needs truth_bam or reads_file "
                "(FASTQ/BAM). Without a truth BAM the classical aligner "
                "builds pseudo-labels from reads_file."
            )
        raw_reads, _ = load_real_reads(
            reads=reads_files,
            modality=modality, region=region,
            max_reads=max_reads or None, reference=anchor_ref,
            require_truth=False, layout=layout,
            reference_fasta=fasta,
            as_sequences=True,
        )
        if not raw_reads:
            sys.exit(
                f"ERROR: no reads loaded for {tag} inside "
                f"({region or contig or 'full contig'})."
            )
        print(f"note: {tag} has no truth_bam — generating classical "
              "pseudo-labels (distillation, not GIAB gold truth)")
        pseudo_name = f"pseudo_truth.{sample}.{modality}.bam" if args.manifest \
            else "pseudo_truth.bam"
        pseudo_bam = os.path.join(args.out, pseudo_name)
        reads = pseudo_label_reads(
            raw_reads,
            anchor_ref.reference,
            modality=modality,
            batch_size=args.batch_size,
            write_bam=pseudo_bam,
            references={anchor_ref.ref_id: type("R", (), {"seq": anchor_ref.ref_seq})()},
            contig_names={anchor_ref.ref_id: anchor_ref.contig},
            reference_fasta=fasta,
        )
        has_truth = True
        label_source = "pseudo_classical"

    if not reads:
        sys.exit(
            f"ERROR: no labelled reads for {tag} inside the reference window "
            f"({region or contig or 'full contig'}). Widen --region, check "
            "truth_bam / reads_file, or ensure the classical aligner can map."
        )

    n_val = max(1, int(len(reads) * val_fraction))
    val_reads, train_reads = reads[:n_val], reads[n_val:]
    if not train_reads:                      # tiny sets: keep at least one train
        train_reads, val_reads = reads, reads[:1]

    train_batches: list[tuple] = []
    val_batches: list[tuple] = []
    for label, _ in modes:
        train_batches += build_batches(train_reads, references[label], args.batch_size)
        val_batches += build_batches(val_reads, references[label], args.batch_size)

    info = {
        "sample": sample,
        "reference_fasta": os.path.abspath(fasta),
        "gfa": os.path.abspath(gfa) if gfa else None,
        "truth_bam": os.path.abspath(truth_bam) if truth_bam else None,
        "pseudo_truth_bam": os.path.abspath(pseudo_bam) if pseudo_bam else None,
        "label_source": label_source,
        "region": region,
        "modality": modality,
        "has_truth": has_truth,
        "n_reads_total": len(reads),
        "n_train_reads": len(train_reads),
        "n_val_reads": len(val_reads),
    }
    print(f"  [{tag}] reads={len(reads)} labels={label_source} "
          f"(train {len(train_reads)} / val {len(val_reads)})")
    return train_batches, val_batches, info


def _file_sig(path):
    """Cheap content proxy for cache keys: (abspath, size, int(mtime))."""
    if not path:
        return None
    try:
        st = os.stat(path)
        return [os.path.abspath(path), st.st_size, int(st.st_mtime)]
    except OSError:
        return [os.path.abspath(path), None, None]


def _ref_index_cache_path(args, fasta, gfa, region, contig, label, kmer_size,
                          with_graph):
    """Path for a single window's reference index (graph + FM-index).

    Keyed by the reference/graph *files* (+ size/mtime), region, contig, label
    and kmer size — i.e. everything the index depends on, but NOT the reads. So
    two runs that differ only in read depth (a rebalanced manifest) reuse the
    exact same indexes instead of rebuilding them.
    """
    import hashlib

    key = {
        "v": 1,
        "fasta": _file_sig(fasta),
        "gfa": _file_sig(gfa) if (gfa and with_graph) else None,
        "region": region,
        "contig": contig,
        "label": label,
        "kmer_size": kmer_size,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(args.dataset_cache_dir, "ref_index",
                        f"{(contig or 'ref')}_{label}_{digest}.pt")


def _save_ref_index_cache(cache_path, rr):
    """Atomically persist one built reference index (best-effort)."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    try:
        torch.save(rr, tmp)
        os.replace(tmp, cache_path)
    except Exception as exc:  # noqa: BLE001 - cache is optional, never fatal
        print(f"ref-index cache: WARNING failed to save ({exc})")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _load_or_build_reference(args, pipeline, fasta, gfa, contig, region,
                             with_graph, kmer_size):
    """Return a RealReference, loading its index from disk when available.

    Falls back to building (and then caching) the index on a miss. The on-disk
    index cache is read-independent, so it survives read-depth / manifest
    changes that only affect which reads are loaded.
    """
    use_cache = getattr(args, "dataset_cache", True)
    label = "pangenome" if with_graph else "linear"
    path = (_ref_index_cache_path(args, fasta, gfa, region, contig, label,
                                  kmer_size, with_graph)
            if use_cache else None)
    if path and os.path.exists(path) and not getattr(args, "rebuild_dataset_cache", False):
        try:
            return torch.load(path, weights_only=False)
        except Exception as exc:  # noqa: BLE001 - fall back to rebuild
            print(f"ref-index cache: WARNING failed to load {path} ({exc}); rebuilding")
    rr = build_reference_from_files(
        pipeline, fasta,
        gfa=gfa if with_graph else None,
        contig=contig, region=region,
        with_graph=with_graph, kmer_size=kmer_size,
    )
    if path:
        _save_ref_index_cache(path, rr)
    return rr


def _dataset_cache_path(args, effective_ref_mode, kmer_size):
    """Path for the prebuilt-dataset cache, keyed by manifest content + build args.

    Any change that alters the built batches (a re-sliced manifest, a different
    ref-mode / batch-size / max-read-len / val-fraction / max-reads / kmer-size)
    changes the hash, so a stale cache is never silently reused.
    """
    import hashlib

    key: dict = {
        "v": 1,
        "ref_mode": effective_ref_mode,
        "batch_size": args.batch_size,
        "max_read_len": args.max_read_len,
        "val_fraction": args.val_fraction,
        "max_reads": args.max_reads,
        "kmer_size": kmer_size,
    }
    if args.manifest:
        with open(args.manifest, "rb") as fh:
            key["manifest_sha1"] = hashlib.sha1(fh.read()).hexdigest()
        key["manifest"] = os.path.abspath(args.manifest)
        stem = os.path.splitext(os.path.basename(args.manifest))[0]
    else:
        key["cli"] = [
            os.path.abspath(args.reference_fasta) if args.reference_fasta else None,
            os.path.abspath(args.gfa) if args.gfa else None,
            args.region, args.contig, args.modality,
            os.path.abspath(args.truth_bam) if args.truth_bam else None,
            [os.path.abspath(x) for x in (args.reads_files or [])],
        ]
        stem = "cli"
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(args.dataset_cache_dir, f"{stem}.{digest}.pt")


def _save_dataset_cache(cache_path, all_train, all_val, data_info):
    """Atomically write the built dataset so future runs skip the build phase.

    Best-effort: any failure (unpicklable object, out-of-quota) only prints a
    warning — training still proceeds on the in-memory dataset.
    """
    import time as _time

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    t0 = _time.time()
    print(f"dataset cache: saving prebuilt dataset -> {cache_path} ...")
    try:
        torch.save(
            {"train": all_train, "val": all_val, "data_info": data_info}, tmp
        )
        os.replace(tmp, cache_path)
        size_gb = os.path.getsize(cache_path) / 1e9
        print(f"dataset cache: wrote {size_gb:.2f} GB in {_time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001 - cache is optional, never fatal
        print(f"dataset cache: WARNING failed to save ({exc}); "
              "continuing without a cache")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def build_real(args, pipeline, model_cfg):
    kmer_size = model_cfg.graph_encoder.kmer_size
    modes = _ref_modes(args.ref_mode)
    ref_cache: dict = {}

    if args.manifest:
        globals_, entries = _load_manifest(args.manifest)
        if globals_.get("ref_mode"):
            modes = _ref_modes(globals_["ref_mode"])
        if not entries:
            sys.exit(f"ERROR: manifest {args.manifest} lists no entries")
        print(f"manifest: {args.manifest}  entries={len(entries)}  "
              f"ref-mode={args.ref_mode if not globals_.get('ref_mode') else globals_['ref_mode']}")
    else:
        if not args.reference_fasta:
            sys.exit("ERROR: --data real needs --reference-fasta or --manifest")
        globals_, entries = {}, [{}]      # single entry built from CLI flags

    # Prebuilt-dataset cache: the build phase below (reading every window's reads
    # from the BAMs + building per-window graph/FM indexes) can take ~1h for a
    # whole-chromosome manifest and is held only in RAM, so a crash/restart pays
    # it again. Cache the built batches to disk keyed by manifest+args.
    effective_ref_mode = globals_.get("ref_mode") or args.ref_mode
    cache_path = None
    if getattr(args, "dataset_cache", True):
        cache_path = _dataset_cache_path(args, effective_ref_mode, kmer_size)
        if os.path.exists(cache_path) and not getattr(args, "rebuild_dataset_cache", False):
            import time as _time

            t0 = _time.time()
            print(f"dataset cache: loading prebuilt dataset -> {cache_path}")
            try:
                payload = torch.load(cache_path, weights_only=False)
                all_train = payload["train"]
                all_val = payload["val"]
                data_info = payload["data_info"]
                print(f"dataset cache: loaded {len(all_train)} train / "
                      f"{len(all_val)} val batches in {_time.time() - t0:.1f}s "
                      "(skipped build)")
                return all_train, all_val, data_info
            except Exception as exc:  # noqa: BLE001 - fall back to rebuild
                print(f"dataset cache: WARNING failed to load ({exc}); rebuilding")

    all_train: list[tuple] = []
    all_val: list[tuple] = []
    entry_infos: list[dict] = []
    for entry in progress(entries, desc="build manifest entries", unit="entry"):
        tb, vb, info = _build_real_entry(
            entry, globals_, args, pipeline, modes, kmer_size, ref_cache
        )
        all_train += tb
        all_val += vb
        entry_infos.append(info)

    # Batches are appended per entry, so without this a multi-modality manifest
    # would feed one modality's batches in a block before the next. A universal
    # aligner learns better when each epoch interleaves modalities, so shuffle
    # the pooled training batches with a fixed seed (order is otherwise the only
    # thing that changes; each batch keeps its own reads + reference).
    if len(entries) > 1:
        import random as _random

        _random.Random(0).shuffle(all_train)

    total_reads = sum(e["n_reads_total"] for e in entry_infos)
    print(
        f"source: real  entries={len(entry_infos)}  reads={total_reads}  "
        f"batches: {len(all_train)} train / {len(all_val)} val"
    )
    if args.ref_mode == "both":
        print("note: each read is trained once on the linear index and once "
              "on the pangenome index")

    if args.manifest:
        data_info = {
            "source": "real",
            "manifest": os.path.abspath(args.manifest),
            "ref_mode": args.ref_mode,
            "n_entries": len(entry_infos),
            "n_reads_total": total_reads,
            "entries": entry_infos,
        }
    else:                                 # single-sample: keep the flat schema
        data_info = {"source": "real", "ref_mode": args.ref_mode, **entry_infos[0]}

    if cache_path is not None:
        _save_dataset_cache(cache_path, all_train, all_val, data_info)
    return all_train, all_val, data_info


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
    p.add_argument("--manifest", default=None,
                   help="JSON training manifest for multi-sample / multi-modality "
                        "joint training. A list of entries (or {\"entries\": [...], "
                        "<global defaults>}); each entry sets sample / modality / "
                        "truth_bam and may override reference_fasta / gfa / region. "
                        "References are cached across entries. Build one with "
                        "scripts/hprc/build_train_manifest.py.")
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
    p.add_argument("--max-read-len", type=int, default=None,
                   help="truncate each read to this many bases before encoding "
                        "(default: no cap). REQUIRED in practice for long reads "
                        "(HiFi/ONT ~10-20kb): the attention/Mamba towers are "
                        "O(L^2)/O(L) in read length, so uncapped long reads OOM. "
                        "A prefix (e.g. 1024) still carries plenty of seeds for "
                        "placement while keeping memory bounded")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction of real reads held out for validation")
    # --- training knobs --- #
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="accumulate gradients over N micro-batches before each "
                        "optimizer step; effective batch = batch-size * N, at the "
                        "memory of one micro-batch (1 = step every batch)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=64,
                   help="small by default so a CPU run finishes quickly")
    p.add_argument("--device", default=None,
                   help="cuda / cuda:N / mps / xpu / cpu (default: auto)")
    p.add_argument("--devices", default="auto",
                   help="legacy nn.DataParallel device list. DataParallel is NOT "
                        "supported for GraphMamba (the model returns a custom "
                        "dataclass DataParallel cannot gather), so a >1 list here "
                        "now falls back to a single GPU with a warning. For real "
                        "multi-GPU training launch with torchrun instead: "
                        "`torchrun --standalone --nproc_per_node=<N> scripts/train.py ...` "
                        "(one process per GPU, gradients all-reduced, near-linear "
                        "scaling).")
    p.add_argument("--require-gpu", action="store_true",
                   help="hard-fail instead of silently training on CPU (use in "
                        "GPU jobs so a CPU-only torch wheel is caught immediately)")
    p.add_argument("--workers", type=int, default=0,
                   help="host worker threads for seeding/chaining + the batch "
                        "prefetch that keeps the GPU fed (0 = all CPU cores)")
    p.add_argument("--prefetch", type=int, default=3,
                   help="batches to build ahead on background threads so the GPU "
                        "is not starved by host-side seeding (0 disables)")
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model forward (CUDA/XPU; big speed-up "
                        "after warmup, skipped automatically on MPS). Most useful "
                        "here because the Mamba scan runs unfused when mamba-ssm "
                        "is absent, so inductor fusion recovers a lot of it")
    p.add_argument("--compile-mode", default="max-autotune",
                   choices=("default", "reduce-overhead", "max-autotune"),
                   help="torch.compile mode (only with --compile). 'max-autotune' "
                        "is fastest at steady state but slow to warm up on the "
                        "variable read/graph shapes here; 'default' or "
                        "'reduce-overhead' warm up much faster")
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
    p.add_argument("--no-cuda-rawkernels", dest="cuda_rawkernels",
                   action="store_false", default=True,
                   help="disable the CuPy RawKernel tier for chaining/extension/"
                        "k-mer lookup and use the portable torch/numpy stages "
                        "instead (default: on when CuPy + an NVIDIA GPU are present)")
    p.add_argument("--amp-dtype", default="auto",
                   choices=("auto", "bf16", "fp16", "off"),
                   help="mixed-precision dtype for the forward/loss: 'auto' picks "
                        "bf16 on Ampere+ (no GradScaler needed), 'off' runs fp32. "
                        "bf16 is both faster and more memory-frugal than fp32")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--monitor", default="locus_accuracy",
                   help="early-stopping metric (falls back to -val_loss if the "
                        "chosen metric is unmeasurable on this data)")
    p.add_argument("--ref-mode", default="both",
                   choices=("linear", "pangenome", "both"),
                   help="train on linear genome, pangenome graph, or both")
    p.add_argument("--init-checkpoint", default=None,
                   help="warm-start model weights from a prior run's checkpoint "
                        "(checkpoint.pt / last.pt) for curriculum / continual "
                        "training; the optimizer and LR schedule start fresh")
    p.add_argument("--resume", default=None,
                   help="resume an interrupted run: restore model + optimizer + "
                        "LR-schedule step from a checkpoint and continue toward "
                        "--epochs (the TOTAL target, so the remaining epochs are "
                        "trained). Pass a checkpoint path, or 'auto' to pick the "
                        "latest epoch_*/step_* checkpoint (or last.pt) under --out. "
                        "Appends to the existing history.json. Unlike "
                        "--init-checkpoint (weights only, fresh optimizer), this "
                        "continues the same run")
    p.add_argument("--out", default="data/training_runs/latest")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="skip writing the best checkpoint.pt (per-epoch and "
                        "last.pt are still written)")
    p.add_argument("--save-every-steps", type=int, default=0,
                   help="also write checkpoints/step_XXXXXX.pt every N optimizer "
                        "steps within an epoch (0 = per-epoch only). Useful for "
                        "long runs so nothing is lost between epoch boundaries")
    p.add_argument("--plot-every-steps", type=int, default=0,
                   help="redraw the figures into <out>/plots every N optimizer "
                        "steps within an epoch (0 = per-epoch only). Handy when "
                        "one epoch is tens of thousands of steps so curves appear "
                        "long before the first epoch boundary")
    p.add_argument("--validate-every-steps", type=int, default=0,
                   help="run a validation pass every N optimizer steps within an "
                        "epoch and append it to history (0 = per-epoch only), so "
                        "validation curves fill in mid-epoch")
    p.add_argument("--intra-val-max-batches", type=int, default=0,
                   help="cap intra-epoch validation to this many val batches "
                        "(0 = full val set). Intra-epoch validation also runs the "
                        "end-to-end aligner, so a small cap keeps it cheap")
    p.add_argument("--dataset-cache-dir", default="data/dataset_cache",
                   help="directory for the prebuilt-dataset cache. The expensive "
                        "'build manifest entries' phase (reading BAMs, building "
                        "per-window graph/FM indexes, train/val split) is saved "
                        "here keyed by manifest content + build args, so future "
                        "runs load in seconds instead of rebuilding. Point this at "
                        "roomy project storage (e.g. /fs/ess/PCS0289/gmf_cache) if "
                        "your home quota is tight")
    p.add_argument("--no-dataset-cache", dest="dataset_cache",
                   action="store_false", default=True,
                   help="disable the prebuilt-dataset cache (always rebuild, "
                        "never write it)")
    p.add_argument("--rebuild-dataset-cache", action="store_true",
                   help="ignore any existing dataset cache and rebuild it "
                        "(overwrites the cache file)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Resolve the source: real when a FASTA or a manifest is supplied, else synthetic.
    if args.data is None:
        args.data = "real" if (args.reference_fasta or args.manifest) else "synthetic"

    # Multi-GPU: when launched under `torchrun --nproc_per_node=N`, spin up one
    # process per GPU. Each rank pins itself to cuda:LOCAL_RANK, trains on its
    # own shard of the batch list, and syncs gradients via all-reduce (see
    # graphmambaformer/distributed.py + training/trainer.py). Outside torchrun
    # this returns a disabled context and everything below runs exactly as a
    # single-process run.
    dist_ctx = maybe_init_distributed(requested_device=args.device)
    if dist_ctx.enabled and dist_ctx.local_rank is not None:
        # Point this rank at its own GPU so AccelContext detects the right device
        # (only when a CUDA/auto device was requested; keep an explicit cpu/mps).
        req = (args.device or "").lower()
        if req in ("", "auto", "best", "cuda"):
            args.device = f"cuda:{dist_ctx.local_rank}"

    def rank0_print(*a, **k):
        if dist_ctx.is_main:
            print(*a, **k)

    torch.manual_seed(0)
    if dist_ctx.is_main:
        os.makedirs(args.out, exist_ok=True)

    rank0_print("=" * 74)
    rank0_print("Building dataset and model")
    rank0_print("=" * 74)

    try:
        accel = AccelContext(AccelConfig(
            device=args.device,
            num_workers=args.workers,
            prefetch=args.prefetch,
            compile=args.compile,
            compile_mode=args.compile_mode,
            cuda_graphs=args.cuda_graphs,
            cuda_rawkernels=args.cuda_rawkernels,
            fp8=args.fp8,
            tensorrt=args.tensorrt,
            amp=(args.amp_dtype != "off"),
            amp_dtype=("auto" if args.amp_dtype in ("auto", "off") else args.amp_dtype),
        ))
    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
    if dist_ctx.enabled:
        rank0_print(f"multi-GPU (torchrun): world_size={dist_ctx.world_size}, "
                    f"backend={dist_ctx.backend}")
    print(f"device[rank {dist_ctx.rank}]: {accel.summary()}"
          if dist_ctx.enabled else f"device: {accel.summary()}")
    from graphmambaformer.accel import list_visible_gpus
    if dist_ctx.is_main:
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

    # Curriculum warm-start: load a prior stage's weights so each harder region
    # continues from the last instead of relearning placement from scratch.
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        rank0_print(f"warm-start: loaded {args.init_checkpoint} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")

    pipeline = build_pipeline(
        PipelineConfig(mode="hybrid", batch_size=args.batch_size,
                       max_read_len=args.max_read_len),
        model=model, accel=accel,
    )

    # Build (or load) the dataset with rank 0 going first: it populates the
    # on-disk dataset cache once while the other ranks wait, then they run and
    # hit the warm cache instead of all rebuilding it concurrently. With caching
    # enabled (the default) every rank simply loads the same prebuilt batches.
    with main_process_first(dist_ctx):
        if args.data == "real":
            train_batches, val_batches, data_info = build_real(args, pipeline, model_cfg)
        else:
            train_batches, val_batches, data_info = build_synthetic(args, pipeline, model_cfg)

    n_train = sum(len(r) for r, _ in train_batches)
    n_val = sum(len(r) for r, _ in val_batches)
    rank0_print(f"ref-mode: {args.ref_mode}")
    rank0_print(f"reads: {n_train} train / {n_val} val   "
                f"batches: {len(train_batches)} train / {len(val_batches)} val")
    if dist_ctx.enabled:
        rank0_print(f"data-parallel: sharding {len(train_batches)} train batches "
                    f"across {dist_ctx.world_size} ranks "
                    f"(~{len(train_batches) // dist_ctx.world_size} batches/rank/epoch)")
    if args.ref_mode == "both":
        rank0_print("note: each read is trained once on the linear index and once "
                    "on the pangenome index")

    rank0_print()
    rank0_print("=" * 74)
    rank0_print("Training")
    rank0_print("=" * 74)
    trainer = Trainer(
        model, pipeline,
        cfg=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            grad_accum=args.grad_accum,
            patience=args.patience, monitor=args.monitor, out_dir=args.out,
            save_checkpoint=not args.no_checkpoint,
            save_every_steps=args.save_every_steps,
            plot_every_steps=args.plot_every_steps,
            validate_every_steps=args.validate_every_steps,
            intra_val_max_batches=args.intra_val_max_batches,
            devices=args.devices,
        ),
        loss_cfg=LossConfig(),
        accel=accel,
        verbose=not args.quiet,
        dist_ctx=dist_ctx,
    )
    history = trainer.fit(train_batches, val_batches, resume=args.resume)

    # Only rank 0 owns the run directory. Other ranks hold identical state
    # (synced gradients, same seed), so they skip every disk write.
    if dist_ctx.is_main:
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
                "world_size": dist_ctx.world_size,
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

    shutdown_distributed(dist_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
