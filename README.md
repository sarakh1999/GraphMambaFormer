# GraphMambaFormer

A bidirectional **Graph-Mamba-2** universal alignment engine for mapping
sequencing reads to pangenome graphs, built layer by layer from the proposed
architecture in `figure1_architecture_v2.html`.

Current development focus: **long reads** — PacBio HiFi and ONT.

## Implemented so far

| Figure 1 component | Module |
| --- | --- |
| 1A · Modality-Aware Read Encoder | `graphmambaformer/encoders/read_encoder.py` |
| 1A · Reference Graph Encoder | `graphmambaformer/encoders/graph_encoder.py` |
| 1B · Bidirectional Mamba (Layer 1) | `graphmambaformer/layers/mamba1.py` (reference-port Mamba-1), `layers/mamba2.py` (Mamba-2), `layers/bimamba.py` |
| 1B · Windowed multi-head self-attention (Layer 2) | `graphmambaformer/layers/attention.py` |
| 1B · GATv2 graph attention (Layer 3) | `graphmambaformer/layers/gat.py` |
| MambaFormer backbone | `graphmambaformer/blocks/mambaformer.py` |
| 1B · Hybrid block (Mamba → attention → GATv2 → FFN, ×N) | `graphmambaformer/blocks/hybrid_block.py` |
| Top-level assembly | `graphmambaformer/model.py` |
| **Core model** · GraphMambaModel + multi-task variant | `graphmambaformer/models/graph_mamba.py` |
| **Stage 1** · Seeding (minimizer / SMEM / DBG / fuzzy / GPU) | `graphmambaformer/alignment/seeding.py` |
| **Stage 2** · Chaining (affine-gap DP + graph bonus) | `graphmambaformer/alignment/chaining.py` |
| **Stage 3** · DP extension (banded affine SW + WFA) | `graphmambaformer/alignment/extension.py` |
| **Stage 4** · Neural scoring bridge | `graphmambaformer/alignment/scoring.py` |
| Alignment pipeline (hybrid / fast / two-pass) | `graphmambaformer/alignment/pipeline.py` |
| Losses (alignment + Kendall multi-task) | `graphmambaformer/losses/alignment_loss.py` |
| Training / validation / behaviour probes / plots | `graphmambaformer/training/`, `scripts/train.py` |
| GPU acceleration stack (NVIDIA · AMD · Intel · Apple · CPU) | `graphmambaformer/accel/` |

Design details:

- **Read encoder** — k-mer tokenization + base-quality embedding + on-the-fly
  sinusoidal positional encoding + a prepended modality-conditioning token → `d_model` (default 512).
- **Graph encoder** — pooled k-mer node features + Laplacian-eigenvector
  positional encoding + 8 discrete edge-type embeddings (consumed by the GATv2 layer).
- **Bidirectional Mamba** — two interchangeable mixers, each fused forward+reverse
  by a learned gate `g·y_fwd + (1−g)·y_rev`:
  - **Mamba-1** (`layers/mamba1.py`) — a faithful pure-PyTorch port of the
    reference repo's `mamba_simple.Mamba` (`in_proj → depthwise causal conv →
    x_proj → selective scan(Δ/B/C) → SiLU gate → out_proj`, S4D real init,
    `d_state=16`, `expand=2`, `dt_rank="auto"`).
  - **Mamba-2** (`layers/mamba2.py`) — a readable pure-PyTorch SSD reference scan
    (`d_state=64`, `d_inner=1024`, `d_conv=4`, selective Δ/B/C).
  Both run on CPU/Apple Silicon and automatically delegate to CUDA `mamba_ssm`
  when a GPU + the package are available.
- **Multi-head self-attention** — bidirectional (non-causal) by default and
  padding-mask aware, via `scaled_dot_product_attention`. Supports an optional
  symmetric sliding `window` (for the windowed-attention variant) and a `causal`
  flag. The reference MambaFormer attention is causal; alignment uses both
  directions, so the default differs.

## Core model: `GraphMambaModel`

The default core architecture. A read is encoded in **base space** (not k-mer
tokens) so anchor read positions index the hidden states directly, which is what
lets the Stage 4 heads look up "the model's view of this locus":

```
reads  → SequenceEncoder → BiMamba-2 tower ─┐
                                            ├→ CrossAttentionFusion → pooled
graph  → GraphEncoder    → GATv2 tower ─────┘        │
                                                     ├→ ComplexityRouter
                                                     ├→ MappingHead (node / offset / MAPQ)
                                                     ├→ SeedScoringHead
                                                     └→ ChainScoringHead
```

`MultiTaskGraphMamba` adds ten predictive-genomics heads (variant calling, SV
genotyping, haplotype, HLA, BQSR, methylation, ancestry, copy number, somatic,
PGx) as branching MLPs over the *same* forward pass, so they cost one small MLP
each rather than a second model. Heads are opt-in via `MultiTaskConfig` because
each needs its own labels.

### Core architecture modes

`build_core_model` selects the architecture; **`"graphmamba"` is the default**.

| `arch` | Model | Alignment heads |
| --- | --- | --- |
| `"graphmamba"` | `GraphMambaModel` | yes |
| `"multitask_graphmamba"` | `+ the ten task heads` | yes |
| `"mambaformer"` | `GraphMambaFormerEncoder`, MambaFormer backbone | no (ablation baseline) |
| `"hybrid"` | `GraphMambaFormerEncoder`, hybrid block stack | no (ablation baseline) |

The two encoder baselines are sequence-only. Selecting one is reported through
`CoreModelSpec.supports_alignment_heads`, and the pipeline then runs its
classical path instead of failing — so architecture and pipeline mode vary
independently.

## Alignment pipeline

Five stages, with the neural core woven into the classical ones rather than
bolted on the end:

| Stage | What it does | Key implementation notes |
| --- | --- | --- |
| **1 · Seeding** | reference → candidate anchors | minimizer sketch, FM-index SMEMs, De Bruijn, spaced/fuzzy seeds, multiplex-DBG, GPU k-mer table. Several modes can run together; anchors are merged and collapsed on shared diagonals. |
| **2 · Chaining** | anchors → collinear chains | minimap2-style affine-gap DP, plus a graph-hop bonus and a reference-path bias from the pangenome graph. Batched DP on GPU. |
| **3 · Extension** | chains → base-level CIGARs | banded affine Smith-Waterman (band widened by the chain's own diagonal spread, so a chain containing a large indel aligns through it) or WFA for low-divergence pairs. |
| **4 · Scoring** | neural refinement | anchor pruning **before** chaining, chain re-ranking **after** the DP, then MAPQ and a rescue locus for unplaced reads. |
| **5 · Post** | records | primary/secondary selection, soft clips, `AlignmentRecord` per read. |

Stage ordering is deliberate: pruning before Stage 2 shrinks the DP input and
biases anchor weights, re-ranking after Stage 2 lets the head see complete
chains, and MAPQ comes last, once the primary/secondary margin exists.

### Pipeline modes

`build_pipeline` selects the mode; **`"hybrid"` is the default**.

| `mode` | Class | Behaviour |
| --- | --- | --- |
| `"hybrid"` | `HybridAlignmentPipeline` | Full accuracy path with all four stages plus neural scoring. |
| `"fast"` | `FastAlignmentPipeline` | Classical only, MAPQ from the score margin. The throughput baseline. |
| `"two_pass"` | `TwoPassAligner` | Fast path first, hybrid re-alignment only for reads that are not confidently resolved. |

A read is "easy" (and skips the neural pass in `two_pass`) when its best chain
covers `easy_coverage` of the read and beats the runner-up by `easy_margin`.

```python
from graphmambaformer import PipelineConfig, build_pipeline, build_core_model

model = build_core_model().model              # "graphmamba" by default
pipeline = build_pipeline(PipelineConfig(), model=model)   # "hybrid" by default

reference = pipeline.build_reference(ref_seq, node_seqs=nodes, node_ref_start=starts)
results, stats = pipeline.align(reads, reference)

print(results[0].primary.cigar_string, results[0].primary.mapq)
print(stats.summary())
```

## Losses

`GraphMambaLoss` covers both halves of the model:

- **`AlignmentLoss`** — per-anchor BCE, *listwise* chain-ranking cross-entropy
  (the ordering is what inference uses, not the absolute scores), node
  classification, within-node position and MAPQ Huber terms, a one-sided router
  compute budget, and an alignment-score margin.
- **`MultiTaskLoss`** — one term per enabled head, with the objective derived
  from the head's label space (per-read, per-node, or per-base) plus any
  auxiliary regression channels.

Terms are balanced by Kendall uncertainty weighting
([arXiv:1705.07115](https://arxiv.org/abs/1705.07115)): each task learns a
log-variance `s` and contributes `exp(-s)·L + s`. A term whose labels are absent
from the batch is **skipped**, not zeroed, so partially-labelled data trains the
heads it has labels for without diluting the others.

## Training, validation and plots

`scripts/train.py` trains the model, validates it, and writes every figure:

```bash
# quick CPU run on synthetic data
PYTHONPATH=. python scripts/train.py --reads 64 --epochs 10

# GPU run, plots into a named directory
PYTHONPATH=. python scripts/train.py --preset table1 --epochs 40 \
    --device cuda --out data/training_runs/chr1
```

Supervision is built from the dataset's **ground truth**, not invented: an anchor
is positive when its implied diagonal really matches the read's true locus, and
the chain label is the candidate that best overlaps the true span. Seeding and
chaining run for real first, so the labels describe the anchors the model is
actually asked to score.

### Validation measures alignment, not just loss

The objective is a weighted sum of seven terms whose balance shifts as the
Kendall weights learn, so its absolute value is not comparable across epochs.
`graphmambaformer/training/metrics.py` reports what the aligner is judged on:

| Metric | Question it answers |
| --- | --- |
| `locus_accuracy` | did the **whole pipeline** place the read within 50 bp? |
| `chain_accuracy` | does the re-ranker pick the correct candidate chain? |
| `anchor_auc` / precision / recall | can the seed head separate true anchors? |
| `mapq_mae` | how far off is the predicted MAPQ? |
| MAPQ **calibration** | does the claimed error rate match the observed one? |

Two deliberate refusals to flatter the model: `chain_accuracy` scores only reads
with **≥2 candidates** (picking 1 of 1 is not a measurement, and reporting it as
100% is misleading), and a monitored metric that is unmeasurable on the data
reports `None` so early stopping falls back to validation loss instead of
stopping at epoch 0 on a metric that can never move.

### Model behaviour at every step

A falling loss curve is equally consistent with a model that has collapsed, so
`graphmambaformer/training/probes.py` records what actually happened each step:
per-tower activation spread (a dead tower shows as `std=0`), gradient norms per
parameter group plus the all-zero fraction, the router's split across compute
paths, and each head's output spread (a constant head shows as `std≈0`).

```
train e02 s0014  loss=4.4033  (chain=0.000 mapq=0.009 position=0.108 seed=0.603)
                 |g|=0.581  route=fast:100%,medium:0%,full:0%
epoch 02  train=4.4589  val loss=4.4063 locus=100.0% chain=n/a(<2 candidates)
          anchorAUC=0.475 mapqMAE=3.1 mapped=100.0%
```

Six figures are written per run (`--out <dir>/plots`): total loss with the
per-term breakdown, the learned Kendall weights, validation quality, model
behaviour, MAPQ calibration, and the supervision actually available per batch.
matplotlib is optional and imported lazily on the `Agg` backend, so a headless
run works and a missing install skips the plots instead of failing the training.

## GPU acceleration stack

`AccelContext` detects what the host supports and hands each stage a backend, so
the same config runs on an H100 and on a laptop CPU:

| Tier | Used for | Fallback |
| --- | --- | --- |
| CuPy `RawKernel` | k-mer lookup, chaining DP, banded SW | batched PyTorch |
| Triton | fused LayerNorm + Linear + GELU | eager PyTorch |
| `mamba_ssm` | fused selective scan | pure-PyTorch SSD scan |
| TF32 / Flash-SDP / cuDNN autotune | matmul + attention | plain kernels |
| AMP (`bf16`/`fp16`) + CUDA graphs | model forward | full precision |

```python
from graphmambaformer import AccelContext
print(AccelContext().summary())
# tier=torch_cpu | vendor=cpu | device=cpu | arch=cpu | tf32=False | amp=off
```

### Every GPU, not just recent NVIDIA

PyTorch reports AMD GPUs through the same `torch.cuda` API as NVIDIA, so
`has_cuda` alone cannot tell them apart. Detection keys on a `vendor` field
instead, and each capability is gated on the hardware that really has it:

| Vendor | Detected via | TF32 | fp16 AMP | bf16 | Raw kernels |
| --- | --- | --- | --- | --- | --- |
| NVIDIA | `torch.cuda`, no HIP | sm_80+ | sm_70+ | sm_80+ | CuPy/NVRTC |
| AMD (ROCm) | `torch.version.hip` | no | yes | MI200+ | no — Triton instead |
| Intel | `torch.xpu` | no | yes | yes | no |
| Apple | `torch.backends.mps` | no | yes | no | no |
| CPU | fallback | no | no | no | no |

Consequences that matter in practice: a Pascal card (sm_61) is **not** given
fp16 autocast, because it has no fp16 tensor cores and would run slower than
fp32; FP8 is gated at sm_89 (Ada), not sm_90, since Ada supports it; and CuPy
raw kernels are refused on AMD even when CuPy imports, because they are compiled
with NVRTC. `tests/test_accel.py` pins this across 11 simulated device classes
from Pascal to Blackwell, so the gates are verified without needing each GPU.

## MambaFormer backbone

The sequence backbone is a faithful port of the MambaFormer topology from
[krafton-ai/mambaformer-icl](https://github.com/krafton-ai/mambaformer-icl)
(Park et al. 2024, [arXiv:2402.04248](https://arxiv.org/abs/2402.04248)),
mirroring their `mixed_attn == "mambaformer"` path in `MixerModel` — including
the flat layer list and layer indexing:

```
Inputs -> Mamba (leading) -> for i in range(n_layer): (attention if i%2==0 else Mamba)
       -> LayerNorm -> Outputs
```

With `n_layer=12` this is a leading Mamba + 6 attention + 6 Mamba layers. The
leading Mamba block stands in for positional embeddings (so
`ReadEncoderConfig.use_positional_encoding` can be set `False` for a "pure"
MambaFormer). Each sub-layer is a pre-norm residual (functionally the reference
`Block`'s Add → Norm → Mixer). Adaptations vs. the autoregressive ICL original:
our Mamba mixer is **bidirectional** and attention is **bidirectional**.

The layer parity is set by `MambaFormerConfig.attention_first`:

- `True` (default, reference): even indices are Attention → flat chain `M A M A … M`.
- `False`: even indices are Mamba → flat chain `M M A M A … A`.

The SSM mixer is chosen by `MambaFormerConfig.mamba_variant`:

- `"mamba1"` (default) — the reference-port Mamba-1 mixer (`mamba1` config).
- `"mamba2"` — the Mamba-2 / SSD mixer (`mamba` config).

Select the backbone via `ModelConfig.backbone`:

- `"mambaformer"` (default) — configured by `MambaFormerConfig` (`n_layer`,
  `attention_first`, `mamba_variant`).
- `"hybrid"` — the full Figure 1B block stack repeated ×N (`n_blocks`, default 12):
  **Bidirectional Mamba → windowed multi-head attention → GATv2 → SwiGLU FFN**.

## Hybrid block (Figure 1B, ×N)

Each block runs three orthogonal inductive biases plus an FFN:

- **Layer 1 — Bidirectional Mamba-2** — O(n) sequential state propagation over the
  read (replaces seed chaining).
- **Layer 2 — Windowed multi-head self-attention** — O(n·w) context-dependent
  substitution/indel scoring (replaces Smith–Waterman); `window` set by `BlockConfig.window`.
- **Layer 3 — GATv2 graph attention** (`layers/gat.py`) — O(|E|) edge-type-aware
  message passing over the pangenome graph. It consumes the graph encoder's node
  embeddings + edge-type embeddings (8 discrete types + a learned self-loop), and
  **refines the graph node embeddings in place** with a pre-norm residual, so after
  N blocks the graph has had N rounds of message passing. Read/graph fusion is left
  to the cross-attention alignment decoder (Figure 1C, next stage).
- **FFN** — SwiGLU feed-forward.

Layers 1, 2 and the FFN are pre-norm residual sub-layers over the read tensor `x`;
Layer 3 is a pre-norm residual over the graph nodes. `forward(x, mask, graph) -> x`
keeps the interface stable. Enable Layers 2/3 with the `use_attention` / `use_gat`
flags in `BlockConfig`; built-in factories supply the modules (or pass a custom
`callable(BlockConfig) -> nn.Module`).

Not yet implemented (future): the cross-attention alignment decoder, output heads
(CIGAR/MAPQ/etc.), LoRA adapters, training.

## Docker (nothing to install)

### Quick start — pull the prebuilt image

Anyone with Docker can run the full stack (model + vg + BWA + samtools/bcftools
1.19 + DeepVariant) without building:

```bash
docker pull ghcr.io/sarakh1999/graphmambaformer:latest

# verify tools
docker run --rm -it --platform linux/amd64 \
  ghcr.io/sarakh1999/graphmambaformer:latest gmf-doctor

# model smoke test (baked into the image — no repo clone required)
docker run --rm -it --platform linux/amd64 \
  ghcr.io/sarakh1999/graphmambaformer:latest \
  gmf-python /opt/graphmambaformer/scripts/smoke_test.py
```

With a clone of this repo, `docker/run.sh` bind-mounts the working tree at
`/work` (host edits win over the baked copy) and falls back to the GHCR image
when no local `graphmambaformer:latest` exists:

```bash
git clone https://github.com/sarakh1999/GraphMambaFormer.git
cd GraphMambaFormer
docker/run.sh gmf-doctor
docker/run.sh gmf-python scripts/smoke_test.py
docker/run.sh gmf-python scripts/train.py --reads 64
SAMPLE=HG002 CHR=chr1 docker/run.sh scripts/fig6/run_all.sh
```

Tags: `:latest` (CPU full stack), `:gpu` (CUDA/ROCm), `:fig6-1.6.1` (benchmark
only), `:arm64` / `:xpu` (model-only variants). The CPU image is `linux/amd64`
(Rosetta on Apple Silicon).

### Build from source

```bash
docker/build.sh                                  # graphmambaformer:latest (CPU)
docker/run.sh gmf-doctor                         # verify every tool
docker/run.sh gmf-python scripts/smoke_test.py   # model smoke test
docker/run.sh gmf-python scripts/train.py --reads 64   # train + write plots
SAMPLE=HG002 CHR=chr1 docker/run.sh scripts/fig6/run_all.sh
```

The repo is bind-mounted at `/work` and takes precedence over the baked copy,
so host edits apply immediately and `data/` stays on the host rather than in
the image. `TARGET=fig6 docker/build.sh` builds a smaller benchmark-only image.
See `scripts/fig6/README.md` for the one exception (hap.py stays external).

### Publish (maintainers)

After a local build, push to GHCR so others can pull:

```bash
gh auth login
gh auth refresh -h github.com -s write:packages
docker/publish.sh                    # → ghcr.io/<you>/graphmambaformer:latest
TARGET=gpu docker/publish.sh         # → …:gpu
TAG=v0.1.0 docker/publish.sh         # also tag a release
```

Or use **Actions → Docker publish → Run workflow**. After the first push, set
the package to public under GitHub → Packages if anonymous pulls fail.

### GPU images

`TARGET=gpu` swaps the CPU torch wheel for a GPU build. One image spans GPU
generations because the vendor and capability detection happens at runtime:

```bash
TARGET=gpu docker/build.sh                        # NVIDIA, cu124 (default)
TORCH_CHANNEL=cu121 TARGET=gpu docker/build.sh     # NVIDIA, older drivers
TORCH_CHANNEL=rocm6.0 TARGET=gpu docker/build.sh   # AMD
INSTALL_CUPY=1 TARGET=gpu docker/build.sh          # + NVRTC raw-kernel tier

IMAGE=graphmambaformer:gpu docker/run.sh gmf-doctor
# or the published tag:
IMAGE=ghcr.io/sarakh1999/graphmambaformer:gpu docker/run.sh gmf-doctor
```

`run.sh` adds the device flags automatically (`--gpus all` for NVIDIA,
`/dev/kfd` + `/dev/dri` for AMD) only when the host actually exposes the device,
since `--gpus all` on a host without the NVIDIA runtime makes `docker run` fail
outright. `GPU=0` forces CPU. `gmf-doctor` prints the live tier and **fails** if
an image built for GPU sees none, so a silent CPU fallback surfaces as an error
rather than as an unexplained slowdown.

The CPU image is x86-64, so it runs under Rosetta on Apple Silicon. Apple's GPU
is not reachable from any container — use the native venv below for MPS/MLX.

## Setup

Prefer the project venv (Python ≥ 3.10). On Apple Silicon this installs PyTorch
with Metal/MPS plus Apple's MLX stack:

```bash
# If needed: recreate with a 3.10+ interpreter (e.g. from conda)
# /path/to/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Apple Silicon GPU (MacBook Pro M-series)

```bash
source .venv/bin/activate
python scripts/check_gpu.py          # must show mps_avail True + mlx OK
PYTHONPATH=. python scripts/smoke_test.py
```

Scripts auto-select `mps` via `graphmambaformer.get_device()` when Metal is
available. Force a backend with `get_device("cpu")` / `get_device("mps")`.

There is no CUDA/`nvidia-smi` on Mac — that is expected.

## Quick check

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```

This exercises the implemented modules on synthetic long-read data and verifies
output shapes and a backward pass (on MPS when available).

## Minimal usage

```python
import torch
from graphmambaformer import GraphMambaFormerEncoder, ModelConfig, get_device

device = get_device()  # mps on Apple Silicon, else cuda/cpu
model = GraphMambaFormerEncoder(ModelConfig(d_model=512, n_blocks=12)).to(device)
reads = ["ACGT..." , "TTGC..."]
hidden, mask = model.read_encoder.encode_reads(reads, modality="pacbio_hifi")
```

## Synthetic test dataset (`graphmambaformer/data`)

A small, fully-labelled synthetic dataset for exercising the pipeline on CPU.
Its statistics mirror the (non-public) AGNES benchmark
([arXiv:2510.16013v3](https://arxiv.org/html/2510.16013v3), Table 1):

| Characteristic | Target |
| --- | --- |
| Reference GC content | 40–50% |
| Reference repeat content | 10–15% |
| Read error rate | 15% (5% ins / 5% del / 5% sub), **2× in homopolymers** |
| Seeds per read | 15–25 true (minimizers, `k=15, w=10`) + 20–30% false |
| Split | 640 train / 160 val / 200 test (`table1` preset) |

Every read is produced by a controlled edit process, so **all ground truth is
exact by construction**: CIGAR, reference span, strand, per-base reference
coordinates, minimizer seeds (true/false + 12-dim features), a pangenome graph
(nodes/edges/8 edge types), MAPQ, and modality-specific labels (methylation,
splice junctions, barcode/UMI, chimeric). Injected **edge cases** cover empty /
sub-`k` / all-`N` / homopolymer-only / repeat-region / reverse-strand /
chimeric reads.

Presets (`graphmambaformer.data.preset`):

- `tiny` (default) — ~1.5–2.5 kb reads (shortest length that still yields the
  paper's 15–25 true seeds), a few samples; full model forward is fast on CPU.
- `long` — paper-scale 8–10 kb reads on ~80 kb references (data / encoder checks).
- `table1` — exact 640/160/200 split at 8–10 kb with every modality (GPU-scale).

```python
from graphmambaformer import generate_dataset, preset, build_datasets, collate_reads
from graphmambaformer.tokenization import KmerTokenizer

dataset = generate_dataset(preset("tiny"))
datasets, _ = build_datasets(dataset=dataset)          # per-split AlignmentDataset
tok = KmerTokenizer(k=3)
batch = [datasets["train"][i] for i in range(4)]
inputs, targets = collate_reads(batch, tok, max_read_len=256)  # model-ready tensors
```

Generate and save a dataset:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py --preset tiny
# reload later with graphmambaformer.data.load_dataset("data/synthetic_tiny.pt")
```

### Standard genomics formats (FASTA / FASTQ / SAM·BAM / GFA / JSON)

The `.pt` file is a convenience bundle for loading straight into PyTorch. For an
aligner, the idiomatic output is a **truth BAM** (the ground-truth alignments
your model learns to reproduce) plus companion files. Use `--emit-dir`:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_synthetic_data.py \
    --preset tiny --all-modalities --emit-dir data/synthetic_tiny --to-bam
```

writes into `data/synthetic_tiny/`:

| File | Contents |
| --- | --- |
| `reference.fasta` | reference sequences |
| `reads.fastq` | reads + Phred qualities (as sequenced) |
| `truth.sam` | ground-truth alignments (FLAG / POS / MAPQ / CIGAR, `=`/`X`/`I`/`D`/`S`) |
| `graph.gfa` | pangenome graph (segments + typed links via `zt:Z:` tag) |
| `labels.json` | seeds (true/false + 12-dim features) + methylation / splice / barcode |

The truth SAM is self-consistent — every `=` column matches the reference and
every `X` differs, on both strands; reverse reads are reoriented to the forward
strand per the SAM spec. `--to-bam` converts to a sorted, indexed BAM when
`samtools` is installed; otherwise the SAM is written and the samtools command
is printed. (Seeds and per-head labels have no native BAM column, so they live
in `graph.gfa` / `labels.json`.)

## Input and output formats

`graphmambaformer/data/formats.py` is the format contract for the pipeline.

| Direction | Formats |
| --- | --- |
| **Input** | FASTQ (plain or `.gz`), BAM, **uBAM**, SAM, CRAM, GFA |
| **Output** | BAM, CRAM, GFA, GBZ |

Reads and graphs each have one entry point that dispatches on the file itself,
and pipeline results go back out through `write_alignments`:

```python
from graphmambaformer.data import (read_reads, read_gfa, write_alignments,
                                   write_gfa_graph, write_gbz)

reads = read_reads("sample.fastq.gz", modality="ont")   # or .bam / .ubam / .sam / .cram
graph = read_gfa("pangenome.gfa")

results, stats = pipeline.align(reads, reference)

write_alignments(results, reads, "out.bam", references=refs)
write_alignments(results, reads, "out.cram", references=refs,
                 reference_fasta="ref.fasta")           # reference-compressed
write_gfa_graph(graph, "out.gfa")
write_gbz("out.gfa", "out.gbz")                         # needs the `vg` binary
```

Passing `ReadRecord` objects rather than bare strings matters: they carry the
Phred qualities and modality from the source file, and the pipeline forwards
both to the encoder (quality is 32 of its 256 input dims). Plain `str` reads
still work — the encoder falls back to its defaults. `write_alignments` needs
both the results and the source reads, because an `AlignmentRecord` carries no
sequence of its own; unmapped reads are written as unmapped records rather than
dropped, so the read count out matches the count in.

BAM/CRAM go through pysam's bundled htslib, so no external `samtools` is
required. GBZ is vg's binary graph+haplotype index and has no pure-Python
writer, so `write_gbz` shells out to `vg` and raises an error naming the exact
command if it is not installed.

### Modalities

Every modality loads from every input format. The canonical keys are
`illumina`, `pacbio_hifi`, `ont`, `rna_seq`, `bisulfite`, `single_cell`, and
`linked_reads`, covering short reads, long reads (ONT and PacBio HiFi), and the
HPRC/GIAB material the benchmark scripts pull down.

`validate_modality` resolves the spellings people actually type — `nanopore`,
`ont_r10` → `ont`; `hifi`, `pacbio`, `ccs`, `revio` → `pacbio_hifi`; `dnbseq`,
`ultima`, `short_read` → `illumina`; `10x`, `chromium` → `linked_reads` — and
raises on anything else. A FASTQ header carrying `mod=<modality>` (as written by
`write_fastq`) overrides the caller's default per read.

### uBAM

ONT and PacBio deliver **unaligned BAM** natively, because it preserves per-base
tags such as MM/ML methylation that FASTQ cannot carry. A uBAM has no `@SQ`
lines and every record is unmapped, so `read_reads` detects the unaligned case
and includes unmapped records there; a mapped-only read of a uBAM would return
an empty list. `is_unaligned_bam(path)` exposes the same check.

Three of these paths were silently broken until [`tests/test_formats.py`](tests/test_formats.py)
pinned them down, and each has a named regression test: gzipped FASTQ raised
`UnicodeDecodeError` despite `.fastq.gz` being routed to the FASTQ reader, a
uBAM read back as zero records, and an unrecognized modality string rode along
on every record to fail much later in the encoder's modality embedding.

### End-to-end coverage

[`tests/test_end_to_end_formats.py`](tests/test_end_to_end_formats.py) runs the
whole chain — file in, align, file out — because the seams between those steps
were where the remaining gaps were. It asserts that all 7 modalities survive a
round trip through the pipeline and back to BAM, that every input format aligns
and writes, that all three pipeline modes accept `ReadRecord`s *and* plain
strings, that sub-batching (two-pass rescue, `batch_size` chunking) keeps
per-read metadata aligned with its rows, and that supplying qualities or a
modality measurably changes the model's output rather than being accepted and
ignored.

The reverse direction is worth stating plainly: reading an *aligned* BAM skips
unmapped records by default, matching samtools semantics. Pass
`include_unmapped=True` to get them. This only applies to files with `@SQ`
lines — for a uBAM the unmapped records are the content, and they are included
automatically.

