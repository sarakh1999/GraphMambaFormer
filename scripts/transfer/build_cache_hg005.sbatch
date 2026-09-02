#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --partition=largemem
#SBATCH --cpus-per-task=32
#SBATCH --mem=375G
#SBATCH --time=4:00:00
#SBATCH --job-name=gmf_cache_build_hg005
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/cache_build_hg005_%j.log

set -euo pipefail

cd /users/PCS0289/sarakhosravi/mambaformer
source .venv/bin/activate

# Build the BATCH-SIZE-4 dataset cache for the balanced chr21 HG005 pangenome
# manifest. HG005 is a fresh, independent GIAB sample (ChineseTrio son), held
# out of the HPRC graph. This mirrors build_cache_bs4.sbatch exactly, only the
# manifest (and therefore the dataset-cache key) changes.
#
# HG005 REUSES HG002's reference indexes. The manifest points at the SAME shared
# FASTA + per-window GFAs (data/chr21/HG002/ref/chr21.fa and
# data/chr21/HG002/pangenome_windows/*.gfa), and the reference-index sub-cache
# (ref_index/) is keyed only by {fasta, gfa, region, contig, label, kmer} —
# read-independent, batch/read-length independent
# (scripts/train.py:_ref_index_cache_path, lines 358-366) — so the expensive
# per-window FM-index / graph build is REUSED, not rebuilt. Watch the log for
# the per-window "[ref pangenome] ..." lines to come from the cache (fast)
# rather than a ~1h rebuild.
#
# The dataset cache key DOES include --batch-size + --max-read-len + the manifest
# SHA1 (scripts/train.py:_dataset_cache_path, config.py:431). HG005's manifest
# differs from HG002's, so this writes a NEW dataset blob:
#   chr21_HG005_pangenome_windows.bal.<digest>.pt
# leaving every HG002 cache (B=8 and B=4) untouched.

# Step 2b (auto): the training/build scripts consume the BALANCED manifest.
# Generate it if it is missing (fast: reuses the shared HG002 graph, only
# recomputes Illumina max_reads at depth 0.3; no re-download, no window rewrite).
BAL_MANIFEST=data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json
if [ ! -f "$BAL_MANIFEST" ]; then
  echo "=== balanced manifest missing -> generating (slice @ --illumina-depth 0.3) ==="
  .venv/bin/python scripts/chr21/slice_pangenome_windows.py \
    --sample HG005 --holdout-start 40000000 --holdout-end 46699983 \
    --illumina-depth 0.3 \
    --manifest      "$BAL_MANIFEST" \
    --test-manifest data/hprc/manifests/chr21_HG005_pangenome_windows.bal.test.json
fi

# Reuse the already-built reference indexes so they are not rebuilt.
mkdir -p /fs/ess/PCS0289/mambaformer_cache
cp -rn data/dataset_cache/ref_index /fs/ess/PCS0289/mambaformer_cache/ 2>/dev/null || true

# Build + save the dataset cache on CPU only (no GPU needed), then exit.
# --epochs 0 => cache is written before the (empty) training loop, then clean exit.
#
# --max-read-len 65536 is part of the dataset-cache key, so it MUST match
# train_hg005.sbatch exactly or the GPU job would rebuild the cache. 64k also caps
# the rare ultralong ONT outlier that can blow up a B=4 batch.
PYTHONPATH=. python scripts/train.py \
  --data real \
  --manifest data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json \
  --dataset-cache-dir /fs/ess/PCS0289/mambaformer_cache \
  --batch-size 4 \
  --max-read-len 65536 \
  --device cpu \
  --epochs 0 \
  --out /fs/ess/PCS0289/mambaformer_runs/cache_build_hg005

echo "=== BUILD SCRIPT DONE (exit $?) ==="
