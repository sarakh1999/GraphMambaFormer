#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --partition=largemem
#SBATCH --cpus-per-task=32
#SBATCH --mem=375G
#SBATCH --time=6:00:00
#SBATCH --job-name=gmf_cache_build
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/cache_build_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
source .venv/bin/activate

# Reuse the already-built reference indexes so they are not rebuilt.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
cp -rn data/dataset_cache/ref_index /fs/ess/PCS0289/mambaformer_cache/ 2>/dev/null || true

# Build + save the full dataset cache on CPU only (no GPU needed), then exit.
# --epochs 0 => cache is written before the (empty) training loop, then clean exit.
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 8 \
  --device cpu \
  --epochs 0 \
  --out /fs/ess/PCS0289/mambaformer_runs/cache_build

echo "=== BUILD SCRIPT DONE (exit $?) ==="
