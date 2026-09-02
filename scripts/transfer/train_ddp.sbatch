#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=gmf_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/ddp_%j.log
#
# 4-GPU (single node) DDP launch of the balanced chr21 HG002 pangenome run.
#
# Same run as the single-GPU launch, scaled to 4x A100 via torchrun: one process
# per GPU, gradients all-reduced per optimizer step, rank 0 writes checkpoints
# (see graphmambaformer/distributed.py + training/trainer.py). scripts/train.py
# already calls maybe_init_distributed()/shutdown_distributed() and passes the
# DistContext into the Trainer, so torchrun is all that is needed here.
#
# REQUIRES THE BATCH-SIZE-4 CACHE — build it first, no rebuild here:
#   --batch-size is part of the dataset cache key (scripts/train.py:
#   _dataset_cache_path, config.py:431), so this B=4 run loads
#   chr21_HG002_pangenome_windows.bal.f1ad8270df11cf9a.pt (NOT the B=8
#   ...5917fc7f6fbb3be7.pt). Build it once, CPU-only (reuses ref_index), with:
#       sbatch scripts/transfer/build_cache_bs4.sbatch
#   Then every rank loads the prebuilt B=4 batches in seconds. If that cache is
#   cold, main_process_first() lets rank 0 rebuild it once (~1h) while the other
#   ranks wait — so pre-building is strongly preferred.
#
# WHY B=4: the B=8 build OOMs the A100-80GB when a ~387kb ONT read pads its whole
# batch (a single SwiGLU FFN activation is 5.91 GiB at B=8). B=4 halves every
# per-batch activation; --grad-accum 8 keeps the effective batch at 32 per rank.
#
# TO RESUME an interrupted DDP run (uses the new --resume flag), re-submit this
# script with RESUME=auto so it picks the latest checkpoint under $OUT:
#     sbatch --export=ALL,RESUME=auto scripts/transfer/train_ddp.sbatch
# or point --resume at a specific checkpoint by setting RESUME=/path/to/epoch_XX.pt.
# --epochs is the TOTAL target, so resuming trains only the remaining epochs.
# NOTE: resume with the SAME --nproc_per_node as the run being resumed, so the
# per-rank steps-per-epoch (and thus the LR schedule) line up exactly.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.."

module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg002_bal_ddp}"
RESUME="${RESUME:-}"
mkdir -p "$OUT"

# Optional resume: pass --resume only when RESUME is set (default is a fresh run,
# leaving the single-GPU code path byte-for-byte unchanged).
RESUME_ARGS=()
if [ -n "$RESUME" ]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

# A100 = Ampere: bf16 (no GradScaler needed). NO --compile: it crashes on the
# mamba-ssm Triton kernels.
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 --grad-accum 8 --amp-dtype bf16 \
  --epochs 12 --require-gpu \
  --workers 8 --prefetch 4 \
  --validate-every-steps 500 --plot-every-steps 250 \
  --intra-val-max-batches 12 --save-every-steps 1000 \
  "${RESUME_ARGS[@]}" \
  --out "$OUT"

echo "DDP training done -> $OUT"
echo "plots:        $OUT/plots/"
echo "checkpoints:  $OUT/checkpoints/  (best -> $OUT/checkpoint.pt, last -> $OUT/last.pt)"
