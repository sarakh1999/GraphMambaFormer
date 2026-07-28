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
vs. pending (the decoder / heads / LoRA / RLHF), and which ground-truth labels
the dataset already provides for them.

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
