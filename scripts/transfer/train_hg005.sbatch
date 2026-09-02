#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=15:00:00
#SBATCH --job-name=gmf_train_hg005
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/train_hg005_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true
source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true

# HG005 = ChineseTrio son (NA24631), held out of the HPRC graph -> a valid
# independent test of the graph read-aligner. This mirrors train_bs4.sbatch
# exactly (same memory-safe B=4 recipe); only the manifest + output dir change.
#
# --batch-size 4 (half the OOMing B=8) keeps the per-batch SwiGLU FFN activation
# small; --grad-accum 8 keeps the effective batch at 32 (4 * 8).
#
# REQUIRES the HG005 batch-size-4 dataset cache. --batch-size and --max-read-len
# are part of the dataset cache key, so this loads:
#   /fs/ess/PCS0289/mambaformer_cache/chr21_HG005_pangenome_windows.bal.<digest>.pt
# Build it first (CPU-only, REUSES HG002's ref_index):
#   sbatch scripts/transfer/build_cache_hg005.sbatch
# If that cache is absent this job would rebuild it on the GPU node (~1h), so
# build it ahead of time.
#
# NO --compile: it crashes on the mamba-ssm Triton kernels here.
# expandable_segments cuts allocator fragmentation, which is what turns a
# borderline step into a hard OOM.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_bal
mkdir -p "$OUT"
echo "=== output dir: $OUT ==="

# A100 = Ampere: bf16 (no GradScaler needed).
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 \
  --grad-accum 8 \
  --amp-dtype bf16 \
  --max-read-len 65536 \
  --epochs 12 \
  --require-gpu \
  --workers 8 \
  --prefetch 4 \
  --validate-every-steps 500 \
  --plot-every-steps 250 \
  --intra-val-max-batches 12 \
  --save-every-steps 1000 \
  --out "$OUT"

echo "=== TRAIN DONE (exit $?) ==="
