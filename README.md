# GraphMambaFormer

Bidirectional **Graph-Mamba-2** universal alignment engine that maps sequencing
reads to linear references and pangenome graphs. Focus: long reads (PacBio HiFi,
ONT); Illumina and other modalities run end-to-end through the same pipeline.

| Knob | Default | Options |
| --- | --- | --- |
| Core architecture | `graphmamba` | `graphmamba`, `multitask_graphmamba`, `mambaformer`, `hybrid` |
| Pipeline mode (`--mode`) | `hybrid` | `hybrid`, `fast`, `two_pass` |
| Reference mode (`--ref-mode`) | — | `linear`, `pangenome`, `both` |
| Modality (`--modality`) | `illumina` | `illumina`, `pacbio_hifi`, `ont`, `rna_seq`, `bisulfite`, `single_cell`, `linked_reads` |

**Contents:** [Setup](#1-setup) · [Concepts](#2-concepts) · [Commands](#3-commands-run-everything) · [Architecture](#4-architecture) · [Code map](#5-code-map) · [Docker & GPU](#6-docker--gpu) · [Formats & data](#7-formats--data)

**Quick path (real HG002 + truth BAM):** [§3.0 step-by-step](#30-step-by-step-hg002-chr21-with-a-real-truth-bam).

---

## 1. Setup

```bash
# from repo root — Python >= 3.10
.venv/bin/pip install -r requirements.txt
# NVIDIA GPU box also: .venv/bin/pip install -r requirements-gpu.txt

# sanity checks
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/check_gpu.py
```

Apple Silicon selects MPS automatically via `graphmambaformer.get_device()`.
No local Python? Use Docker (see [§6](#6-docker--gpu)).

---

## 2. Concepts

Three knobs combine freely: **architecture** × **pipeline mode** × **reference mode**, all tagged by **modality**.

**Pipeline modes** — `hybrid` (neural, accuracy; default) · `fast` (classical, throughput) · `two_pass` (fast first, hybrid rescue for hard reads).

**Reference modes** — `linear` (FASTA only) · `pangenome` (FASTA + GFA) · `both` (each read on both; eval can emit an integrated concordance BAM).

**Architectures** — `graphmamba` / `multitask_graphmamba` have alignment heads; `mambaformer` / `hybrid` are sequence-only ablation encoders (pipeline degrades to classical scoring).

**Modality aliases** (resolved by `validate_modality`): `hifi/pacbio/ccs/revio → pacbio_hifi`, `nanopore/ont_r10 → ont`, `ngs/short_read/dnbseq → illumina`, `10x/chromium → linked_reads`, `wgbs → bisulfite`, `scrna → single_cell`.

**Inputs / outputs**

| Role | Formats |
| --- | --- |
| Reference | FASTA (`.fai` built on demand) |
| Pangenome | GFA / `.gfa.gz` |
| Reads | FASTQ(`.gz`), BAM, uBAM, SAM, CRAM |
| Supervision (train) | truth BAM/SAM/CRAM (`--truth-bam`) |
| Train out | `checkpoints/epoch_XX.pt`, `last.pt`, `checkpoint.pt`, `history.json`, `run_meta.json`, `plots/` |
| Eval/map out | `pred.*.bam` (+`.bai`), `pred.*.sam`, optional CRAM, `metrics.json` |

---

## 3. Commands: run everything

All commands run from the **repo root** with `PYTHONPATH=.` (or the Docker wrapper).

### 3.0 Step-by-step: HG002 chr21 with a real truth BAM

Prefer an **external** truth BAM (Giraffe / GIAB-aligned) over self-made
pseudo-labels for training and reported metrics. You do **not** manually
download the Giraffe truth BAM — `prepare_real_hg002.sh` builds it.

Run this from your own Terminal with **Docker Desktop running** (vg / samtools
images). The Cursor agent shell cannot drive Docker for this path.

#### Step 0 — Prerequisites

```bash
cd /path/to/GraphMambaFormer   # repo root

# Python env (once)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# NVIDIA GPU box also:
# .venv/bin/pip install -r requirements-gpu.txt

# sanity
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/check_gpu.py   # optional
```

Needs: **Docker Desktop**, `curl`. `aws` CLI is optional for the default HG002
GIAB HTTPS path (required only for some HPRC S3 CRAMs).

#### Step 1 — Build inputs (downloads + creates truth BAM)

```bash
chmod +x scripts/prepare_real_hg002.sh scripts/chr21/*.sh
./scripts/prepare_real_hg002.sh
```

What the script does:

| Step | Action | Output role |
| --- | --- | --- |
| 0 | Preflight (Docker / curl / python) | checks only |
| 1 | Fetch GRCh38 → extract chr21 FASTA | linear `--reference-fasta` |
| 2 | Fetch GIAB **VCF/BED** (variant benchmark) | hap.py later — **not** `--truth-bam` |
| 3 | Stream GIAB Illumina BAM → chr21 R1/R2 FASTQ | inference / remap inputs |
| 4 | HPRC chr21 graph → GFA + Giraffe indexes | `--gfa` for pangenome/both |
| 5 | Map R1/R2 with **vg Giraffe** → sorted BAM | **`--truth-bam` labels** |

This can take a long time (network + Giraffe index/map).

#### Step 2 — Confirm files and set path vars

```bash
ls -lh \
  data/chr21/HG002/ref/GRCh38.chr21.fa \
  data/chr21/HG002/chr21.gfa \
  data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam \
  data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz \
  data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz

REF=data/chr21/HG002/ref/GRCh38.chr21.fa
GFA=data/chr21/HG002/chr21.gfa
TRUTH=data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam
REGION=chr21:5000000-6000000   # start small; full chr21 is much slower
```

**What to do with the truth BAM:** pass it as `--truth-bam "$TRUTH"` to
`train.py` / `eval.py`. Training reads locus / CIGAR / MAPQ / strand from it as
supervision labels. Do not edit it; treat it as a frozen external baseline.

| File | Role |
| --- | --- |
| `ref/GRCh38.chr21.fa` | Reference to align to |
| `chr21.gfa` | Pangenome graph (`--gfa`) |
| `bam/*.giraffe.sorted.bam` | **`--truth-bam`** — train/eval labels |
| `reads/*.R{1,2}.fastq.gz` | Raw reads for inference |
| `truth/` GIAB VCF/BED | Variant eval (hap.py); **not** `--truth-bam` |
| `checkpoint.pt` (after train) | Model weights for hybrid eval/map |

Optional public sources (if you build labels yourself instead of Giraffe):

- GIAB NovoAlign BAM (HG002 Illumina):  
  `https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_Illumina_2x250bps/novoalign_bams/HG002.GRCh38.2x250.bam`
- GIAB release (VCF/BED):  
  `https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/`
- HPRC raw data browser:  
  `https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=working/`

#### Step 3 — Train (use the truth BAM)

First real run (linear Illumina on a 1 Mb window):

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --data real \
  --reference-fasta "$REF" \
  --truth-bam "$TRUTH" \
  --region "$REGION" \
  --ref-mode linear \
  --modality illumina \
  --device cuda --require-gpu \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/illumina_linear
```

Linear + pangenome (needs GFA):

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --data real \
  --reference-fasta "$REF" \
  --gfa "$GFA" \
  --truth-bam "$TRUTH" \
  --region "$REGION" \
  --ref-mode both \
  --modality illumina \
  --device cuda --devices auto \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/hg002_chr21_both
```

CPU: drop `--require-gpu` and use `--device cpu` (much slower).  
Checkpoint: `data/training_runs/<out>/checkpoint.pt`.

#### Step 4 — Evaluate against the same truth BAM

```bash
PYTHONPATH=. .venv/bin/python scripts/eval.py --data real \
  --reference-fasta "$REF" \
  --truth-bam "$TRUTH" \
  --region "$REGION" \
  --ref-mode linear \
  --modality illumina \
  --mode hybrid \
  --checkpoint data/training_runs/illumina_linear/checkpoint.pt \
  --device cuda \
  --out data/eval_runs/illumina_linear_hybrid
```

Scores locus / MAPQ / etc. vs Giraffe labels and writes predicted BAM under `--out`.

#### Step 5 — Optional: map FASTQ only (no truth needed at infer time)

```bash
R1=data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz
R2=data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz

PYTHONPATH=. .venv/bin/python scripts/eval.py --data real \
  --reference-fasta "$REF" \
  --region "$REGION" \
  --ref-mode linear \
  --reads-file "$R1" --reads-file "$R2" \
  --read-layout auto \
  --modality illumina \
  --mode hybrid \
  --checkpoint data/training_runs/illumina_linear/checkpoint.pt \
  --out data/eval_runs/illumina_infer
```

#### FASTQ-only / mentor HG005 (no external truth yet)

Prefer mapping once with Giraffe/BWA and freezing that BAM as `--truth-bam`.
If you only have R1/R2, classical **pseudo-labels** work as a bootstrap
(distillation — weaker for paper claims):

```bash
PYTHONPATH=. .venv/bin/python scripts/train.py --data real \
  --reference-fasta "$REF" \
  --reads-file data/chr21/HG005/reads/HG005.chr21.R1.fastq.gz \
  --reads-file data/chr21/HG005/reads/HG005.chr21.R2.fastq.gz \
  --read-layout paired --modality illumina --ref-mode linear \
  --region "$REGION" --device cuda --require-gpu \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/illumina_pseudo
# → also writes data/training_runs/illumina_pseudo/pseudo_truth.bam
```

**Minimal path:** `./scripts/prepare_real_hg002.sh` → set `REF` / `TRUTH` /
`REGION` → `train.py --truth-bam "$TRUTH"` → `eval.py --truth-bam "$TRUTH"
--checkpoint ...`.

### 3.1 Train variants (truth BAM *or* FASTQ with classical pseudo-labels)

Swap `--modality {illumina|pacbio_hifi|ont|...}` and `--ref-mode {linear|pangenome|both}` freely (`--gfa` required for `pangenome`/`both`).

Prefer `--truth-bam` when you have aligned labels. **Without a truth BAM**, pass
`--reads-file` only — the classical `fast` aligner builds pseudo-labels and
writes `pseudo_truth.bam` under `--out`.

```bash
# Illumina · linear · with truth BAM
PYTHONPATH=. python scripts/train.py --data real \
  --reference-fasta "$REF" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode linear --modality illumina \
  --device cuda --require-gpu --workers 16 --prefetch 3 \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/illumina_linear

# Illumina · linear · NO truth BAM (FASTQ → classical pseudo-labels → train)
PYTHONPATH=. python scripts/train.py --data real \
  --reference-fasta "$REF" \
  --reads-file data/chr21/HG005/reads/HG005.chr21.R1.fastq.gz \
  --reads-file data/chr21/HG005/reads/HG005.chr21.R2.fastq.gz \
  --read-layout paired --modality illumina --ref-mode linear \
  --region chr21 --device cuda --require-gpu \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/illumina_pseudo

# PacBio HiFi · pangenome
PYTHONPATH=. python scripts/train.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode pangenome --modality pacbio_hifi \
  --device cuda --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/hifi_pangenome

# ONT · both (each read trained linear + pangenome)
PYTHONPATH=. python scripts/train.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode both --modality ont \
  --device cuda --devices auto --epochs 20 --batch-size 8 --d-model 256 \
  --out data/training_runs/ont_both
```

### 3.2 Evaluate with truth (metrics + predicted BAM)

Set `--mode {hybrid|fast|two_pass}` independently of modality. `fast` needs no checkpoint.

```bash
# Illumina · hybrid · linear
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode linear --modality illumina --mode hybrid \
  --checkpoint data/training_runs/illumina_linear/checkpoint.pt \
  --device cuda --out data/eval_runs/illumina_linear_hybrid

# HiFi · fast · pangenome (no checkpoint)
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode pangenome --modality pacbio_hifi --mode fast \
  --out data/eval_runs/hifi_pangenome_fast

# ONT · two_pass · both (+ integrated concordance BAM; add --write-cram for CRAM)
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode both --modality ont --mode two_pass \
  --checkpoint data/training_runs/ont_both/checkpoint.pt \
  --device cuda --out data/eval_runs/ont_both_twopass
```

### 3.3 Inference without truth (reads → BAM)

```bash
# Illumina paired FASTQ (auto-pairs R1/R2). Swap --mode as needed.
R1=data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz
R2=data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --region "$REGION" --ref-mode linear \
  --reads-file "$R1" --reads-file "$R2" \
  --read-layout auto --modality illumina --mode hybrid \
  --checkpoint data/training_runs/illumina_linear/checkpoint.pt \
  --out data/eval_runs/illumina_infer

# PacBio HiFi single-end FASTQ.gz
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --reads-file sample.hifi.fastq.gz \
  --modality pacbio_hifi --read-layout auto --ref-mode linear --mode hybrid \
  --checkpoint data/training_runs/hifi_pangenome/checkpoint.pt \
  --out data/eval_runs/hifi_infer

# ONT BAM/uBAM (pangenome + two_pass shown)
PYTHONPATH=. python scripts/eval.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --reads-file sample.ont.bam \
  --modality ont --read-layout auto --ref-mode pangenome --mode two_pass \
  --checkpoint data/training_runs/ont_both/checkpoint.pt \
  --out data/eval_runs/ont_infer
```

### 3.4 HPRC multi-modality mapping (production wrapper)

Reads under `data/hprc/reads/<SAMPLE>/{hifi,illumina,ont}/`. Full detail: [`data/hprc/README.md`](data/hprc/README.md).

```bash
export REF=data/chr21/HG002/ref/GRCh38.chr21.fa
export OURS_INDEX_CACHE=data/hprc/index_cache

# one modality
SAMPLE=HG00438 MODALITIES=hifi ./scripts/hprc/map_sample.sh

# all three → one BAM (distinct @RG / XM per modality)
SAMPLE=HG00438 MODALITIES=illumina,hifi,ont BAM_MODE=combined ./scripts/hprc/map_sample.sh

# all three → three BAMs
SAMPLE=HG00673 BAM_MODE=separate ./scripts/hprc/map_sample.sh

# override pipeline mode (wrapper default: fast) and cap reads for a smoke run
OURS_MODE=hybrid SAMPLE=HG00438 MODALITIES=ont ./scripts/hprc/map_sample.sh
OURS_MAX_READS=2000 SAMPLE=HG00438 MODALITIES=illumina,hifi,ont ./scripts/hprc/map_sample.sh

# equivalent direct call
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample HG00438 --modalities illumina,hifi,ont \
  --bam-mode combined --mode fast \
  --out data/hprc/bam/HG00438.all.ours.sorted.bam \
  --index-cache "$OURS_INDEX_CACHE" --workers 16
```

### 3.5 Synthetic CPU smoke

```bash
PYTHONPATH=. python scripts/train.py --preset tiny --ref-mode both --epochs 6 \
  --out data/training_runs/synth_both

PYTHONPATH=. python scripts/eval.py \
  --checkpoint data/training_runs/synth_both/checkpoint.pt \
  --ref-mode both --mode hybrid --emit-truth --out data/eval_runs/synth_hybrid
# classical (no checkpoint): --mode fast ; hard-tail rescue: --mode two_pass
```

### 3.6 Tests & verification

```bash
PYTHONPATH=. .venv/bin/python tests/run_all.py                    # full suite
PYTHONPATH=. .venv/bin/python tests/run_all.py pipeline           # filter by name
PYTHONPATH=. .venv/bin/python tests/run_all.py alignment_stages
PYTHONPATH=. .venv/bin/python tests/run_all.py formats
PYTHONPATH=. .venv/bin/python tests/run_all.py accel

PYTHONPATH=. .venv/bin/python scripts/verify_stages.py            # Figure-1 encoders/backbone
PYTHONPATH=. .venv/bin/python scripts/verify_alignment_pipeline.py # stages 1-4 E2E
PYTHONPATH=. .venv/bin/python scripts/audit_architecture.py       # conformance checklist
```

### 3.7 Minimal Python API

```python
from graphmambaformer import PipelineConfig, build_pipeline, build_core_model, get_device

model = build_core_model().model.to(get_device())            # arch=graphmamba
pipeline = build_pipeline(PipelineConfig(mode="hybrid"), model=model)
reference = pipeline.build_reference(ref_seq, node_seqs=nodes, node_ref_start=starts)
results, stats = pipeline.align(reads, reference)
print(results[0].primary.cigar_string, results[0].primary.mapq, stats.summary())
```

---

## 4. Architecture

**Encoders** — read/sequence encoders map k-mer or base tokens + quality + a
modality token → `d_model` (default 512). The core model uses base space so
Stage-4 heads index loci directly. The graph encoder builds pooled k-mer node
features + Laplacian PE + 8 edge-type embeddings for GATv2.

**BiMamba** — forward+reverse fused by a learned gate `g·y_fwd + (1−g)·y_rev`;
Mamba-1 (reference port) or Mamba-2 (SSD); CUDA `mamba_ssm` when available, else
pure PyTorch.

### Seven alignment stages

| Stage | Module | What it does |
| --- | --- | --- |
| 1 Seeding | `alignment/seeding.py` | minimizer / SMEM / DBG / fuzzy / multiplex-DBG / GPU k-mer → anchors |
| 2 Chaining | `alignment/chaining.py` | affine-gap DP + graph-hop bonus |
| 3 Extension | `alignment/extension.py` | banded affine SW or WFA → CIGAR |
| 4 Scoring | `alignment/scoring.py` | neural prune / re-rank / MAPQ / rescue |
| 5 Post | `alignment/postprocessing.py` | correction, population MAPQ, liftover, concordance |
| 6 Specialized | `alignment/specialized.py` | repeats, paralogs, HLA/MHC |
| 7 Predictions | `alignment/predictions.py` | genotype, phase, ancestry, clinical, PGx |
| Orchestration | `alignment/end_to_end.py`, `pipeline.py` | `SevenStagePipeline` + hybrid/fast/two_pass |

Default Stage-1 modes `("smem","minimizer","fuzzy")`, spaced pattern `111010010100110111`. Fuzzy-only:

```python
from graphmambaformer import PipelineConfig, build_pipeline
cfg = PipelineConfig(mode="fast"); cfg.seeding.modes = ("fuzzy",)
pipeline = build_pipeline(cfg, model=None)
```

**Losses** — `GraphMambaLoss` = `AlignmentLoss` (+ optional `MultiTaskLoss`),
balanced by Kendall uncertainty weighting ([arXiv:1705.07115](https://arxiv.org/abs/1705.07115));
absent labels skip their term. **Metrics**: `locus_accuracy` (within 50 bp),
`chain_accuracy` (≥2 candidates), `anchor_auc`/P/R, `mapq_mae`/calibration.

---

## 5. Code map

### Package (`graphmambaformer/`)

| File | Role |
| --- | --- |
| `__init__.py` | public API exports |
| `config.py` | modalities, modes, all dataclasses |
| `device.py` | `get_device`, multi-GPU helpers |
| `tokenization.py` | `KmerTokenizer` |
| `model.py` | `GraphMambaFormerEncoder` (ablation backbone) |
| **`models/graph_mamba.py`** | **`GraphMambaModel`, `MultiTaskGraphMamba`, towers** |
| `models/__init__.py` | `build_core_model`, `CoreModelSpec` |
| `encoders/read_encoder.py` | modality-aware k-mer read encoder |
| `encoders/sequence_encoder.py` | base-space encoder (core model) |
| `encoders/graph_encoder.py` | node/edge encoder + Laplacian PE |
| `layers/mamba1.py` · `mamba2.py` · `bimamba.py` | Mamba-1 / Mamba-2 / bidirectional wrappers |
| `layers/attention.py` · `cross_attention.py` | windowed MHSA / read↔graph fusion |
| `layers/gat.py` · `common.py` | GATv2 / shared norm+residual |
| `blocks/mambaformer.py` · `hybrid_block.py` | MambaFormer chain / Figure-1B block ×N |
| `heads/mapping_head.py` | node / offset / MAPQ |
| `heads/scoring_heads.py` | seed + chain scoring |
| `heads/router.py` · `multitask_heads.py` | compute router / ten genomics heads |
| `alignment/types.py` | `AnchorSet`, `Chain`, `AlignmentRecord`, CIGAR utils |
| `alignment/seeding.py` … `predictions.py` | stages 1–7 (see [§4](#seven-alignment-stages)) |
| `alignment/pipeline.py` | hybrid / fast / two_pass + `build_pipeline` |
| `alignment/end_to_end.py` | `SevenStagePipeline` |
| `alignment/dual_reference.py` · `index_cache.py` | linear+pangenome concordance / on-disk index |
| `data/formats.py` | FASTQ/BAM/CRAM/GFA I/O contract |
| `data/export.py` · `synthetic.py` · `real_data.py` | export + readers / synthetic gen / real windows |
| `data/dataset.py` · `reference_build.py` · `alignment_io.py` | dataset & collate / build `ReferenceIndex` / `write_alignments` |
| `losses/alignment_loss.py` | `AlignmentLoss`, `MultiTaskLoss`, Kendall |
| `training/trainer.py` · `targets.py` · `metrics.py` · `probes.py` · `plots.py` | loop / supervision / metrics / probes / figures |
| `accel/backend.py` | `AccelContext`, vendor detection |
| `accel/cuda_kernels.py` · `triton_ops.py` · `cuda_graphs.py` | CuPy RawKernel / Triton / CUDA Graphs |
| `accel/transformer_engine.py` · `tensorrt_engine.py` · `simd_sw.py` · `parallel.py` | FP8 / TensorRT / SIMD SW / host threads |

### Scripts (`scripts/`)

| Script | Role |
| --- | --- |
| `train.py` · `eval.py` | train / evaluate-infer |
| `smoke_test.py` · `check_gpu.py` | forward+backward check / accel tier |
| `generate_synthetic_data.py` · `convert_formats.py` | make synthetic data / convert formats |
| `prepare_real_hg002.sh` | build chr21 HG002 bundle (Docker) |
| `verify_stages.py` · `verify_alignment_pipeline.py` · `audit_architecture.py` | verification |
| `chr21/align_ours.py` · `chr21/*.sh` | production mapper / chr21 benchmark |
| `hprc/map_sample.sh` | HPRC sample wrapper |
| `fig6/*` | Figure-6 BWA/Giraffe/hap.py benchmark |
| `rebuild_and_publish_images.sh` | build + push GHCR images |

### Tests (`tests/`)

`run_all.py` (runner, name filters) · `test_alignment_stages.py` · `test_pipeline.py`
· `test_downstream_stages.py` · `test_core_model.py` · `test_losses.py`
· `test_training.py` · `test_formats.py` · `test_end_to_end_formats.py`
· `test_accel.py` · `test_dual_reference.py`.

---

## 6. Docker & GPU


Same codebase and genomics stack in both tags; only the PyTorch wheel differs.

| Tag | Purpose |
| --- | --- |
| `ghcr.io/sarakh1999/graphmambaformer:latest` | full stack + **CPU** PyTorch (laptops / CI / smoke) |
| `ghcr.io/sarakh1999/graphmambaformer:gpu` | same + **CUDA** PyTorch — use this for NVIDIA train/eval |

`docker/run.sh` bind-mounts the clone at `/work`, so `data/` is read/written on the host.

### 6.1 Quickstart

Full host walkthrough (prereqs, prepare script internals, truth-BAM usage,
train/eval/infer): see [§3.0](#30-step-by-step-hg002-chr21-with-a-real-truth-bam).
Docker equivalents of the same train/eval commands:

```bash
# clone
git clone https://github.com/sarakh1999/GraphMambaFormer.git
cd GraphMambaFormer

# if GHCR package is private (403 on pull): PAT needs read:packages
echo YOUR_GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin

# pull GPU image + doctor
docker pull ghcr.io/sarakh1999/graphmambaformer:gpu
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda docker/run.sh gmf-doctor

# prepare HG002 chr21 inputs (once)
chmod +x scripts/prepare_real_hg002.sh scripts/chr21/*.sh
./scripts/prepare_real_hg002.sh

REF=data/chr21/HG002/ref/GRCh38.chr21.fa
GFA=data/chr21/HG002/chr21.gfa
TRUTH=data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam
REGION=chr21:5000000-6000000
```

**Train (GPU, linear+pangenome):**

```bash
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda \
  docker/run.sh gmf-python scripts/train.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode both --modality illumina \
  --device cuda --require-gpu --devices auto \
  --epochs 20 --batch-size 8 --d-model 256 \
  --workers 16 --prefetch 3 \
  --out /work/data/training_runs/ _both
```

Linear-only: drop `--gfa` and use `--ref-mode linear`. Swap `--modality` to
`pacbio_hifi` / `ont` / etc. as needed.

**Train without truth BAM (real FASTQ → classical pseudo-labels):**

```bash
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda \
  docker/run.sh gmf-python scripts/train.py --data real \
  --reference-fasta "$REF" \
  --reads-file /work/data/chr21/HG005/reads/HG005.chr21.R1.fastq.gz \
  --reads-file /work/data/chr21/HG005/reads/HG005.chr21.R2.fastq.gz \
  --read-layout paired --modality illumina --ref-mode linear \
  --region chr21 --device cuda --require-gpu \
  --epochs 20 --batch-size 8 --d-model 256 \
  --out /work/data/training_runs/illumina_pseudo
```

**One-shot hybrid map (train-then-align via `align_ours.py`):**

```bash
PYTHONPATH=. python scripts/chr21/align_ours.py \
  --ref "$REF" \
  --illumina data/chr21/HG005/reads/HG005.chr21.R1.fastq.gz \
  --illumina data/chr21/HG005/reads/HG005.chr21.R2.fastq.gz \
  --read-layout paired --mode hybrid --device cuda \
  --epochs 20 --batch-size 8 --d-model 256 \
  --train-out data/training_runs/illumina_hybrid \
  --out data/chr21/HG005/bam/HG005.chr21.ours.sorted.bam
```

`--epochs` / `--batch-size` / `--d-model` are accepted on `align_ours.py` in all
modes; with `--mode hybrid` and no `--checkpoint` they trigger inline training.

**Evaluate (metrics + predicted BAM):**

```bash
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda \
  docker/run.sh gmf-python scripts/eval.py --data real \
  --reference-fasta "$REF" --gfa "$GFA" --truth-bam "$TRUTH" \
  --region "$REGION" --ref-mode both --modality illumina --mode hybrid \
  --checkpoint /work/data/training_runs/ _both/checkpoint.pt \
  --device cuda --require-gpu \
  --out /work/data/eval_runs/ _both_hybrid
```

Classical only (no checkpoint): `--mode fast`. Hard-tail rescue: `--mode two_pass`.

**Synthetic smoke (no real data):**

```bash
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda \
  docker/run.sh gmf-python scripts/train.py --preset tiny --ref-mode both \
  --epochs 2 --batch-size 2 --d-model 64 \
  --out /work/data/training_runs/synth_smoke

IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu GPU=cuda \
  docker/run.sh gmf-python scripts/eval.py \
  --checkpoint /work/data/training_runs/synth_smoke/checkpoint.pt \
  --ref-mode both --mode hybrid --emit-truth \
  --out /work/data/eval_runs/synth_smoke
```

CPU-only machine: use `:latest` and omit `GPU=cuda` / `--device cuda` / `--require-gpu`.

### 6.2 Build / publish locally


```bash
docker/build.sh                          # → graphmambaformer:latest
TARGET=gpu docker/build.sh               # → graphmambaformer:gpu
./scripts/rebuild_and_publish_images.sh  # build + push both to GHCR
# or: docker/publish.sh  /  TARGET=gpu docker/publish.sh
```

### 6.3 GPU flags & accel


| Flag | Default | Meaning |
| --- | --- | --- |
| `--device` | auto | `cuda` / `cuda:N` / `mps` / `xpu` / `cpu` |
| `--devices` | `auto` | multi-GPU list (`auto`/`all`/`0,1`/`none`) |
| `--require-gpu` | off | exit if no CUDA/MPS/XPU |
| `--workers N` | `0` (all cores) | host threads for seed/chain + prefetch |
| `--prefetch N` | `2` | look-ahead batches |
| `--compile` | off | `torch.compile` the forward |
| `--cuda-graphs` | on | capture fixed-shape inference |
| `--fp8` | on | TE FP8 when supported (Ampere → BF16) |
| `--tensorrt` | off | TensorRT inference |

**Accel tiers** (auto fallback): CuPy RawKernel → PyTorch · Triton → eager ·
`mamba_ssm` → pure-PyTorch SSD · CUDA Graphs → eager · TE FP8 → BF16 · TensorRT → eager.
Vendors NVIDIA / AMD / Intel / Apple / CPU are capability-gated at runtime.

---

## 7. Formats & data


```python
from graphmambaformer.data import read_reads, read_gfa, write_alignments
reads = read_reads("sample.fastq.gz", modality="ont")   # uBAM keeps MM/ML tags
graph = read_gfa("pangenome.gfa")
results, stats = pipeline.align(reads, reference)
write_alignments(results, reads, "out.bam", references=refs)
```

```bash
# CLI conversions
PYTHONPATH=. python scripts/convert_formats.py reads.fastq out.bam
PYTHONPATH=. python scripts/convert_formats.py reads.fastq out.cram --reference ref.fa
PYTHONPATH=. python scripts/convert_formats.py graph.gfa out.gbz          # needs vg

# synthetic dataset → FASTA/FASTQ/SAM(BAM)/GFA/labels
PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py \
  --preset tiny --all-modalities --emit-dir data/synthetic_tiny --to-bam
```

Presets: `tiny` (CPU smoke), `long` (8–10 kb), `table1` (640/160/200 split).

---

## License

See [LICENSE](LICENSE).
