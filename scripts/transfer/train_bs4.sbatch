#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=15:00:00
#SBATCH --job-name=gmf_train_bs4
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/train_bs4_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true
source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true

# Memory-safe single-GPU run: --batch-size 4 (half the OOMing B=8) keeps the
# per-batch SwiGLU FFN activation to ~2.96 GiB even for the ~387kb ONT read that
# blew up the B=8 job. --grad-accum 8 keeps the effective batch at 32 (4 * 8).
#
# REQUIRES the batch-size-4 dataset cache. --batch-size is part of the dataset
# cache key, so this loads:
#   /fs/ess/PCS0289/mambaformer_cache/chr21_HG002_pangenome_windows.bal.f1ad8270df11cf9a.pt
# Build it first (CPU-only, reuses ref_index):
#   sbatch scripts/transfer/build_cache_bs4.sbatch
# If that cache is absent this job would rebuild it on the GPU node (~1h), so
# build it ahead of time.
#
# NO --compile: it crashes on the mamba-ssm Triton kernels here.
# expandable_segments cuts allocator fragmentation, which is what turns a
# borderline step into a hard OOM.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/fs/ess/PCS0289/mambaformer_runs/chr21_hg002_bal_bs4_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== output dir: $OUT ==="

# A100 = Ampere: bf16 (no GradScaler needed).
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 \
  --grad-accum 8 \
  --amp-dtype bf16 \
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
