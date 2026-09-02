#!/usr/bin/env bash
# CPU-ONLY HG005 dataset-cache build, for a big CPU node (e.g. a0168, 102 cores,
# no GPU). This is stage 2 of run_hg005_local.sh, split out so it can run on a
# separate node in parallel with (or ahead of) GPU work.
#
# It writes to its OWN cache dir + out dir (distinct names) so it can NEVER
# collide with a build/train running on the GPU node: the cache filename is
# derived from the key (manifest + batch-size + max-read-len), so two nodes
# building the same key into the SAME dir would race on the SAME .pt. Separate
# dirs make that impossible.
#
# Run detached so it survives an SSH drop (default batch-size=4):
#   cd ~/mambaformer
#   tmux new -s hg005build 'bash scripts/transfer/build_hg005_cache_cpu.sh 2>&1 | tee logs/hg005_cache_cpu.log'
#
# Build for a different batch size (e.g. 8) -- lands in its own cache dir so it
# never collides with the bs=4 cache/training on the GPU node:
#   BATCH_SIZE=8 tmux new -s hg005build8 \
#     'BATCH_SIZE=8 bash scripts/transfer/build_hg005_cache_cpu.sh 2>&1 | tee logs/hg005_cache_cpu_bs8.log'
#
# When it finishes, train on a GPU node off THIS cache (matching batch size):
#   bs=4:  CACHE_DIR=/fs/ess/PCS0289/mambaformer_cache_cpu \
#            tmux new -s hg005 'bash scripts/transfer/run_hg005_local.sh 2>&1 | tee logs/hg005_local.log'
#   bs=8:  CACHE_DIR=/fs/ess/PCS0289/mambaformer_cache_cpu_bs8 BATCH_SIZE=8 \
#            tmux new -s hg005 'bash scripts/transfer/run_hg005_local.sh 2>&1 | tee logs/hg005_local.log'
# (run_hg005_local.sh will see the .pt, skip the build, and go straight to GPU training.)
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root (mambaformer)
# shellcheck disable=SC1091
source .venv/bin/activate

MANIFEST=data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json
# Batch size the cache is built for. The cache KEY (and thus the .pt filename)
# depends on it, so bs=4 and bs=8 are different caches. To keep them from
# stepping on each other -- and so the "already present?" glob below can't get a
# false positive from another batch size -- each non-default batch size lands in
# its OWN cache dir. bs=4 keeps the historical dir for back-compat.
BATCH_SIZE="${BATCH_SIZE:-4}"
if [ "$BATCH_SIZE" = "4" ]; then
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu}"
else
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu_bs${BATCH_SIZE}}"
fi
OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/cache_build_hg005_cpu_bs${BATCH_SIZE}}"
# Big CPU nodes parallelize read extraction hard; override WORKERS to taste.
WORKERS="${WORKERS:-48}"
mkdir -p "$CACHE_DIR" "$OUT" logs

echo "=== node: $(hostname) | cores: $(nproc) | $(date) ==="
echo "=== batch-size: $BATCH_SIZE | cache dir: $CACHE_DIR (isolated per batch size) ==="

if [ ! -f "$MANIFEST" ]; then
  echo "FATAL: manifest not found: $MANIFEST" >&2
  exit 1
fi

# Reuse the read-independent reference-index sub-cache (copied in once) so the
# expensive per-window graph/FM indexes are loaded, not rebuilt.
cp -rn data/dataset_cache/ref_index "$CACHE_DIR"/ 2>/dev/null || true

if ls "$CACHE_DIR"/*HG005*.pt >/dev/null 2>&1; then
  echo "=== cache already present -> nothing to do ==="
  ls -lh "$CACHE_DIR"/*HG005*.pt
  exit 0
fi

echo "=== building dataset cache (CPU only, batch-size=$BATCH_SIZE, do not interrupt) ==="
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --dataset-cache-dir "$CACHE_DIR" \
  --batch-size "$BATCH_SIZE" --max-read-len 65536 \
  --device cpu --epochs 0 \
  --workers "$WORKERS" --prefetch 4 \
  --out "$OUT"

echo "=== BUILD DONE -> $CACHE_DIR ==="
ls -lh "$CACHE_DIR"/*HG005*.pt
