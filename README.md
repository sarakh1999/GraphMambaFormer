# GraphMambaFormer

A bidirectional **Graph-Mamba-2** universal alignment engine for mapping
sequencing reads to pangenome graphs, built layer by layer from the proposed
architecture in `figure1_architecture_v2.html`.

Current development focus: **long reads** — PacBio HiFi and ONT.

## Implemented so far

### 1. Encoders (Figure 1A)

| Piece | File(s) |
| --- | --- |
| Modality-aware read encoder | `graphmambaformer/encoders/read_encoder.py` |
| Base-space sequence encoder (core model) | `graphmambaformer/encoders/sequence_encoder.py` |
| Reference graph encoder | `graphmambaformer/encoders/graph_encoder.py` |

### 2. Layers / backbone (Figure 1B)

| Piece | File(s) |
| --- | --- |
| Mamba-1 / Mamba-2 / BiMamba | `graphmambaformer/layers/mamba1.py`, `layers/mamba2.py`, `layers/bimamba.py` |
| Windowed self-attention | `graphmambaformer/layers/attention.py` |
| Cross-attention fusion | `graphmambaformer/layers/cross_attention.py` |
| GATv2 | `graphmambaformer/layers/gat.py` |
| Shared layer utils | `graphmambaformer/layers/common.py` |
| MambaFormer block stack | `graphmambaformer/blocks/mambaformer.py` |
| Hybrid block (Mamba → attn → GAT → FFN) | `graphmambaformer/blocks/hybrid_block.py` |

### 3. Core model assembly

| Piece | File(s) |
| --- | --- |
| Top-level encoder assembly | `graphmambaformer/model.py` |
| `GraphMambaModel` + multi-task variant | `graphmambaformer/models/graph_mamba.py` |
| Config / build helpers | `graphmambaformer/config.py`, `graphmambaformer/__init__.py` |

### 4. Prediction heads

| Piece | File(s) |
| --- | --- |
| Mapping (node / offset / MAPQ) | `graphmambaformer/heads/mapping_head.py` |
| Seed / chain scoring | `graphmambaformer/heads/scoring_heads.py` |
| Complexity router | `graphmambaformer/heads/router.py` |
| Multi-task genomics heads | `graphmambaformer/heads/multitask_heads.py` |

### 5. Alignment pipeline (Stages 1–5)

| Piece | File(s) |
| --- | --- |
| Stage 1 · Seeding | `graphmambaformer/alignment/seeding.py` |
| Stage 2 · Chaining | `graphmambaformer/alignment/chaining.py` |
| Stage 3 · Extension (SW / WFA) | `graphmambaformer/alignment/extension.py` |
| Stage 4 · Neural scoring | `graphmambaformer/alignment/scoring.py` |
| Hybrid / fast / two-pass pipelines | `graphmambaformer/alignment/pipeline.py` |

### 6. Losses

| Piece | File(s) |
| --- | --- |
| Alignment + Kendall multi-task loss | `graphmambaformer/losses/alignment_loss.py` |

### 7. Training / validation / plots

| Piece | File(s) |
| --- | --- |
| Trainer | `graphmambaformer/training/trainer.py` |
| Metrics (locus/chain/MAPQ/etc.) | `graphmambaformer/training/metrics.py` |
| Behaviour probes | `graphmambaformer/training/probes.py` |
| Supervision targets | `graphmambaformer/training/targets.py` |
| Plot writers | `graphmambaformer/training/plots.py` |
| CLI entrypoint | `scripts/train.py` |

### 8. GPU acceleration

| Piece | File(s) |
| --- | --- |
| Vendor / backend detection | `graphmambaformer/accel/backend.py` |
| CuPy / CUDA kernels | `graphmambaformer/accel/cuda_kernels.py` |
| Triton fused ops | `graphmambaformer/accel/triton_ops.py` |

### 9. Data I/O & formats

| Piece | File(s) |
| --- | --- |
| FASTQ / BAM / modalities | `graphmambaformer/data/formats.py`, `data/alignment_io.py` |
| Export (BAM/CRAM) | `graphmambaformer/data/export.py` |
| Datasets / synthetic | `data/dataset.py`, `data/synthetic.py` |

### 10. Benchmarks & eval scripts

| Piece | File(s) |
| --- | --- |
| Fig 6a (Giraffe / BWA / DeepVariant) | `scripts/fig6/` (`run_all.sh`, `map_*.sh`, `plot_pr.py`, …) |
| chr21 ours vs Giraffe | `scripts/chr21/` (`align_ours.py`, `compare.sh`, `map_ours.sh`, …) |
| Smoke / verify | `scripts/smoke_test.py`, `verify_stages.py`, `verify_alignment_pipeline.py` |

### 11. Tests

| Piece | File(s) |
| --- | --- |
| Core / pipeline / losses / accel / training | `tests/test_*.py`, `tests/run_all.py` |

## Docker (nothing to install)

If you only have Docker, `docker/build.sh` produces one `linux/amd64` image
with the model stack *and* the full genomics benchmark preinstalled — PyTorch
(CPU), `graphmambaformer`, vg, BWA, samtools/bcftools 1.19 and DeepVariant
1.6.1 with its models:

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

### GPU images

`TARGET=gpu` swaps the CPU torch wheel for a GPU build. One image spans GPU
generations because the vendor and capability detection happens at runtime:

```bash
TARGET=gpu docker/build.sh                        # NVIDIA, cu124 (default)
TORCH_CHANNEL=cu121 TARGET=gpu docker/build.sh     # NVIDIA, older drivers
TORCH_CHANNEL=rocm6.0 TARGET=gpu docker/build.sh   # AMD
INSTALL_CUPY=1 TARGET=gpu docker/build.sh          # + NVRTC raw-kernel tier

IMAGE=graphmambaformer:gpu docker/run.sh gmf-doctor
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
