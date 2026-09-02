#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --partition=largemem
#SBATCH --cpus-per-task=32
#SBATCH --mem=375G
#SBATCH --time=4:00:00
#SBATCH --job-name=gmf_cache_build_bs4
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/cache_build_bs4_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
source .venv/bin/activate

# Build the BATCH-SIZE-4 dataset cache for the balanced chr21 HG002 pangenome
# manifest. This is the memory-safe re-batch of the existing batch-size-8 cache:
# the batch-size-8 build OOMs the A100-80GB when a pathological ~387kb ONT read
# pads its whole batch (a single SwiGLU FFN activation alone is 5.91 GiB at B=8;
# halving to B=4 halves every per-batch activation).
#
# Only the *batching* changes. The dataset cache key includes --batch-size
# (see scripts/train.py:_dataset_cache_path, config.py:431), so this writes a
# NEW file:
#   chr21_HG002_pangenome_windows.bal.f1ad8270df11cf9a.pt   (B=4)
# leaving the running job's B=8 cache untouched:
#   chr21_HG002_pangenome_windows.bal.5917fc7f6fbb3be7.pt   (B=8)
#
# The reference-index sub-cache (ref_index/) is read-independent — its key
# excludes batch size / read length (scripts/train.py:_ref_index_cache_path,
# lines 358-366) — so the expensive per-window FM-index / graph build is REUSED,
# not rebuilt. Watch the log for the per-window "[ref pangenome] ..." lines to
# come from the cache (fast) rather than a ~1h rebuild.

# Reuse the already-built reference indexes so they are not rebuilt.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
cp -rn data/dataset_cache/ref_index /fs/ess/PCS0289/mambaformer_cache/ 2>/dev/null || true

# Build + save the dataset cache on CPU only (no GPU needed), then exit.
# --epochs 0 => cache is written before the (empty) training loop, then clean exit.
#
# NOTE on --max-read-len: left UNCAPPED here to mirror the existing B=8 cache and
# preserve full-length ONT reads. B=4 keeps the observed worst case (~387kb read)
# to ~38 GB total, comfortably under the 80 GB card. If reads longer than ~600kb
# ever appear (ONT-UL outliers), add a safety cap by uncommenting one of these
# (each changes the cache digest again, so it becomes its own cache file):
#   --max-read-len 200000   # generous; drops only extreme UL outliers
#   --max-read-len 1024     # aggressive; matches the Python-scan fallback cap
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 \
  --device cpu \
  --epochs 0 \
  --out /fs/ess/PCS0289/mambaformer_runs/cache_build_bs4

echo "=== BUILD SCRIPT DONE (exit $?) ==="
