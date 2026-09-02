#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=gmf_ddp_hg005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/ddp_hg005_%j.log
#
# 2-GPU (single node) DDP launch of the balanced chr21 HG005 pangenome run.
# HG005 = GIAB ChineseTrio son (NA24631), held out of the HPRC graph -> a valid
# independent test of the graph read-aligner.
#
# This mirrors train_ddp.sbatch (HG002, 4-GPU) but: HG005 balanced manifest,
# 2 GPUs, and --max-read-len 65536 passed EXPLICITLY so the dataset-cache key
# matches build_cache_hg005.sbatch exactly (batch-size + max-read-len + manifest
# SHA1 form the key). If the key did not match, rank 0 would rebuild the cache
# (~1h) while rank 1 waits at the NCCL barrier and times out (~10 min) -> the
# ChildFailedError you saw. So:
#
#   REQUIRED FIRST (build once, CPU-only, reuses HG002 ref_index):
#       sbatch scripts/transfer/build_cache_hg005.sbatch
#   That writes /fs/ess/PCS0289/mambaformer_cache/chr21_HG005_pangenome_windows.bal.<digest>.pt
#   Then every rank here loads it in seconds -> no barrier timeout.
#
# TO RESUME an interrupted run, re-submit with RESUME=auto (picks the latest
# checkpoint under $OUT):
#       sbatch --export=ALL,RESUME=auto scripts/transfer/train_hg005_ddp.sbatch
# Resume with the SAME --nproc_per_node (2) so per-rank steps/epoch and the LR
# schedule line up.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.."

module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_bal_ddp}"
RESUME="${RESUME:-}"
mkdir -p "$OUT"

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

# A100 = Ampere: bf16 (no GradScaler). NO --compile (crashes mamba-ssm Triton).
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 --grad-accum 8 --amp-dtype bf16 \
  --max-read-len 65536 \
  --epochs 12 --require-gpu \
  --workers 8 --prefetch 4 \
  --validate-every-steps 500 --plot-every-steps 250 \
  --intra-val-max-batches 12 --save-every-steps 1000 \
  "${RESUME_ARGS[@]}" \
  --out "$OUT"

echo "=== DDP training done -> $OUT ==="
echo "plots:        $OUT/plots/"
echo "checkpoints:  $OUT/checkpoints/  (best -> $OUT/checkpoint.pt, last -> $OUT/last.pt)"
