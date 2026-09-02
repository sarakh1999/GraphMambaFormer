#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=15:00:00
#SBATCH --job-name=gmf_train
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/train_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true
source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "from graphmambaformer.accel import mamba_ssm_available; print('mamba_ssm_available:', mamba_ssm_available())"

# Reuse already-built reference indexes; cache + outputs live on roomy /fs/ess.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
cp -rn data/dataset_cache/ref_index /fs/ess/PCS0289/mambaformer_cache/ 2>/dev/null || true

OUT=/fs/ess/PCS0289/mambaformer_runs/chr21_hg002_bal_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== output dir: $OUT ==="

# Build the dataset cache (first run) then train on the GPU. Later runs load the cache.
# batch-size 4 + grad-accum 8 => effective batch 32, fits A100-40GB.
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 \
  --grad-accum 8 \
  --amp-dtype bf16 \
  --epochs 12 \
  --require-gpu \
  --compile \
  --workers 8 \
  --prefetch 4 \
  --validate-every-steps 500 \
  --plot-every-steps 250 \
  --intra-val-max-batches 12 \
  --save-every-steps 1000 \
  --out "$OUT"

echo "=== TRAIN DONE (exit $?) ==="
