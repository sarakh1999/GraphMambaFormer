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
| GPU acceleration stack | `graphmambaformer/accel/` |

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
# tier=torch_cpu | device=cpu | cupy=False | triton=False | amp=off
```

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

If you only have Docker, `docker/build.sh` produces one `linux/amd64` image
with the model stack *and* the full genomics benchmark preinstalled — PyTorch
(CPU), `graphmambaformer`, vg, BWA, samtools/bcftools 1.19 and DeepVariant
1.6.1 with its models:

```bash
docker/build.sh                                  # graphmambaformer:latest
docker/run.sh gmf-doctor                         # verify every tool
docker/run.sh gmf-python scripts/smoke_test.py   # model smoke test
SAMPLE=HG002 docker/run.sh scripts/fig6/run_all.sh
```

The repo is bind-mounted at `/work` and takes precedence over the baked copy,
so host edits apply immediately and `data/` stays on the host rather than in
the image. `TARGET=fig6 docker/build.sh` builds a smaller benchmark-only image.
See `scripts/fig6/README.md` for the one exception (hap.py stays external).

The image is CPU-only and x86-64, so it runs under Rosetta on Apple Silicon and
provides no MPS/MLX acceleration — for that, use the native venv below.

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

## Per-stage verification

`scripts/verify_stages.py` runs the dataset through every implemented component
and asserts each stage's output, mapped to Figure 1 (data integrity, homopolymer
doubling, read/graph encoders, Bi-Mamba-2, attention, backbone, full
forward+backward, long-read loadability, edge-case robustness):

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_stages.py
```

It also prints a Figure-1 coverage map showing which components are implemented
vs. pending (splice / barcode heads, LoRA, RLHF), and which ground-truth labels
the dataset already provides for them.

`scripts/verify_alignment_pipeline.py` covers everything downstream — the four
alignment stages, the core model, the losses, and all three pipeline modes:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_alignment_pipeline.py
```

It asserts the properties that matter rather than just shapes: every anchor is a
real exact match, chain members stay collinear, CIGARs consume exactly the read
and the reference span they claim, WFA reproduces known edit distances, anchor
pruning honours both its threshold and its keep-floor, a short training loop
actually decreases the loss, and all three modes recover the true locus.

### Unit suite (`tests/`)

pytest is **not** a dependency, so `tests/run_all.py` discovers and runs every
`test_*` function itself, printing a pass/fail table with tracebacks. The test
modules follow pytest conventions too, so `pytest tests/` works if you have it.

```bash
PYTHONPATH=. .venv/bin/python tests/run_all.py           # everything
PYTHONPATH=. .venv/bin/python tests/run_all.py losses    # substring filter
```

| Module | Covers |
| --- | --- |
| `tests/test_alignment_stages.py` | The algorithms cross-checked against independent brute-force references: suffix array vs. Python's suffix sort, FM-index vs. naive substring scan, SMEM maximality, minimizers vs. explicit per-window minima, chaining DP vs. a textbook O(n²) loop, banded SW vs. an unbanded full-matrix affine DP, WFA vs. a Levenshtein matrix. |
| `tests/test_core_model.py` | Base-space encoding, forward/backward, head masking, multi-task scopes, and that block-diagonal graph collation equals per-graph encoding. |
| `tests/test_losses.py` | All alignment terms, NaN safety for unmatched/fully-masked chain rows, skipping of unlabelled terms, Kendall vs. static weighting, all 11 task heads at their own scope. |
| `tests/test_pipeline.py` | The three modes end to end, pruning threshold + floor, two-pass rescue firing only on hard reads, batch-size invariance, MAPQ range, classical fallback. |
| `tests/test_accel.py` | Capability detection honesty (never claims a CUDA tier without CUDA), fused Triton op equals the composed torch ops, CUDA-graph runner matches eager, CuPy tier declines cleanly. CUDA-only tiers are skipped, not failed, and the exercised tier is printed. |

`scripts/check_gpu.py` is a hardware diagnostic, not a test — it needs a real
Metal/CUDA device and fails inside a sandboxed or headless session.

### Architecture conformance (`scripts/audit_architecture.py`)

The spec in `architecture/GraphMamba_Architecture.html` states concrete numbers,
and this script asserts them against the code so drift fails loudly instead of
being spotted by eye later. Each check names the spec line it came from.

```bash
PYTHONPATH=. .venv/bin/python scripts/audit_architecture.py
```

It has two halves. **Conformance** (65 assertions, the contract) covers the
`d=256 / 6 BiMamba2 / 3 GATv2` shape of the core model, the SequenceEncoder
budget (`64 + 64 + 32 + 96 = 256`), Mamba-2's `conv_dim=4 / headdim=64 /
expand=2`, cross-attention's `8 heads x 32`, GATv2's 4 heads, the forward-pass
tensor shapes down to `(B, L+N, D)`, the stage constants (`min_seed=13`,
`max_occ=200`, DBG `k=21`, multiplex `k=15,21,31`, WFA `mismatch=4 / gap_open=6 /
x_drop=600`, `max_mapq=60`), the MappingHead's `sigmoid x 60`, all ten multi-task
heads with their class counts, Kendall log-variance weighting, and the two
wiring details the spec calls out by name — that the fusion FFN really uses the
Triton fused LN+Linear+GELU, and that route labels come from `RouterConfig`
rather than a pipeline-local copy. The parameter budget is checked at ±10% of
the quoted 14.2M (currently 14,948,647, +5.3%).

**Coverage** walks the 97-feature catalogue and marks each entry implemented /
partial / missing with the module that provides it; the script verifies that
every module it names actually exists, so the table cannot overstate itself.
Roughly 44% are fully implemented and 57% at least partial. The gaps are
unbuilt scope rather than deviations: pipeline Stages 6–7 (repeat/HLA
resolution, predictive-genomics aggregation), the ten specialized aligners, the
C fast path, and all training infrastructure bar the loss.

Where the spec contradicts itself the script prints the resolution. The only
such case today is `d_state`: the forward-pass diagram says 128 while the same
document's `.env` reference says 64, so the code follows 64 — it agrees with
Figure 1B and lands nearer the quoted parameter budget.

## chr21 mentor benchmark (Giraffe vs ours + DeepVariant + Sniffles)

End-to-end scripts for **one HPRC/GIAB individual on chr21**, then scaling to
the ~44 graph-training samples:

```bash
# GIAB eval sample with truth
SAMPLE=HG005 ./scripts/chr21/run_all.sh

# first training sample (HPRC core)
./scripts/chr21/fetch_hprc_sample.sh HG00438 links
SAMPLE=HG00438 SKIP_TRUTH=1 ./scripts/chr21/run_all.sh
```

See `scripts/chr21/README.md`, `data/hprc/SAMPLE_LINKS.md`, and
`data/hprc/graph_samples_44.txt`.
