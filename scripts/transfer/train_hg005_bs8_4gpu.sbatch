#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=gmf_hg005_bs8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/ess/PCS0289/mambaformer_runs/hg005_bs8_4gpu_%j.log
#
# 4x A100-80GB, single node, batch-size 8, DDP, FIXED stack — the sbatch wrapper
# around scripts/transfer/train_hg005_bs8_4gpu.sh. Submit from a login node:
#
#     sbatch scripts/transfer/train_hg005_bs8_4gpu.sbatch
#     # resume a previous fixed run under the same OUT:
#     # (safe re-run is automatic; force fresh with --export=ALL,FRESH=1)
#
# The wrapper regroups the bs4 dataset cache into a bs8 one (no BAM re-read) in a
# single process BEFORE torchrun, then launches 4-GPU DDP at bs8 with host workers
# tuned to keep all 4 GPUs fed. --mem=256G covers the ~60 GB regroup peak.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.."

# Let the wrapper size the worker pool from the allocation's CPUs/GPUs.
export NGPU="${SLURM_GPUS_ON_NODE:-4}"
exec bash scripts/transfer/train_hg005_bs8_4gpu.sh
