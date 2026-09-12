#!/usr/bin/env bash
# =============================================================================
# GPU retrain + benchmark for the NEW read-tower architecture (mentor's 2:1:1).
#
# WHY THIS EXISTS
#   The read tower now interleaves attention at layers 1,3,5 (2 Mamba : 1
#   Transformer, plus the 3-layer GAT tower => 2:1:1 Mamba:Transformer:GNN).
#   Every pre-existing checkpoint was a PURE-Mamba read tower (no attention),
#   so `load_state_dict` fails against the new model. To get a trained accuracy
#   number under the new arch you must RETRAIN, which needs a GPU. On the CPU
#   login node the Mamba scan falls back to a slow pure-Python loop (~20 min per
#   eval) and is I/O-starved on the shared filesystem.
#
# WHAT IT DOES (all on one GPU node)
#   1. Train a fresh checkpoint under the new 2:1:1 arch (d_model=256).
#   2. Eval it on the EASY linear window   -> trained-vs-untrained delta.
#   3. Eval it on a HARD pangenome window  -> where the model must earn its keep.
#   The untrained CPU baseline for step 2 was: locus 99.0%, chain 0.7% (random).
#
# HOW TO RUN
#   Batch:        sbatch scripts/gpu_hg002_train_eval.sh
#   Interactive:  salloc --account=PCS0289 --gpus-per-node=1 --time=04:00:00
#                 then: bash scripts/gpu_hg002_train_eval.sh
#   Override anything without editing the file, e.g.:
#     EPOCHS=30 REGION=chr21:5000000-6000000 sbatch scripts/gpu_hg002_train_eval.sh
# =============================================================================

#SBATCH --job-name=gmf_hg002
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
# On OSC Ascend all nodes are GPU nodes; add a partition only if your site needs
# one, e.g.  #SBATCH --partition=nextgen

set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment at submit time)
# --------------------------------------------------------------------------- #
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

D_MODEL="${D_MODEL:-256}"          # must match the new-arch default (256)
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"            # host threads for seeding/chaining
MAX_READS="${MAX_READS:-0}"        # 0 = all reads in the window

REF_FA="${REF_FA:-data/chr21/HG002/ref/chr21.fa}"
GFA="${GFA:-data/chr21/HG002/chr21.gfa}"
TRUTH_BAM="${TRUTH_BAM:-data/chr21/HG002/bam/HG002.chr21.illumina.real.sorted.bam}"
MODALITY="${MODALITY:-illumina}"

# Easy window: p-arm sequence starts ~5 Mb on GRCh38 chr21; dense Illumina cov.
REGION="${REGION:-chr21:5000000-6000000}"
# Hard window: SV-dense region (matches names under data/chr21/HG002/sv/).
HARD_REGION="${HARD_REGION:-chr21:5750000-6250000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg002_chr21_linear_2to1to1_${STAMP}}"
OUT_EVAL_EASY="${OUT_EVAL_EASY:-data/eval_runs/hg002_easy_linear_trained_${STAMP}}"
OUT_EVAL_HARD="${OUT_EVAL_HARD:-data/eval_runs/hg002_hard_pangenome_trained_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
echo "############################################################"
echo "# GraphMambaFormer HG002 GPU train+eval  ($STAMP)"
echo "#   d_model=$D_MODEL epochs=$EPOCHS batch=$BATCH"
echo "#   region(easy)=$REGION  region(hard)=$HARD_REGION"
echo "############################################################"

if [ ! -f "$ROOT/.venv/bin/activate" ]; then
  echo "ERROR: $ROOT/.venv not found. Create it / point to your env." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# Keep BLAS from oversubscribing the allocated CPUs; the GPU does the heavy math.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "== GPU visibility =="
nvidia-smi -L || { echo "ERROR: no GPU visible on this node." >&2; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: torch cannot see a CUDA device." >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1) TRAIN under the new 2:1:1 architecture (fresh weights)
# --------------------------------------------------------------------------- #
echo; echo "### [1/3] train new-arch checkpoint -> $OUT_TRAIN"
python -u scripts/train.py \
  --data real \
  --reference-fasta "$REF_FA" \
  --truth-bam "$TRUTH_BAM" \
  --region "$REGION" \
  --modality "$MODALITY" \
  --ref-mode linear \
  --d-model "$D_MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --device cuda --devices auto --require-gpu \
  --monitor locus_accuracy \
  --out "$OUT_TRAIN"

if [ ! -f "$CKPT" ]; then
  # Trainer may save best as checkpoint.pt or fall back to last.pt.
  if [ -f "$OUT_TRAIN/last.pt" ]; then CKPT="$OUT_TRAIN/last.pt"; else
    echo "ERROR: no checkpoint produced under $OUT_TRAIN" >&2; exit 1; fi
fi
echo "trained checkpoint: $CKPT"

# --------------------------------------------------------------------------- #
# 2) EVAL trained model on the EASY linear window (two-pass = operational default)
#    Compare against the untrained CPU baseline: locus 99.0%, chain 0.7%.
# --------------------------------------------------------------------------- #
echo; echo "### [2/3] eval trained (easy, linear, two_pass) -> $OUT_EVAL_EASY"
python -u scripts/eval.py \
  --data real \
  --reference-fasta "$REF_FA" \
  --truth-bam "$TRUTH_BAM" \
  --region "$REGION" \
  --modality "$MODALITY" \
  --ref-mode linear \
  --mode two_pass \
  --checkpoint "$CKPT" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  ${MAX_READS:+--max-reads "$MAX_READS"} \
  --device cuda --require-gpu \
  --no-bam \
  --out "$OUT_EVAL_EASY"

# --------------------------------------------------------------------------- #
# 3) EVAL trained model on a HARD pangenome window (graph-aware chaining + rescue)
# --------------------------------------------------------------------------- #
echo; echo "### [3/3] eval trained (hard, pangenome, two_pass) -> $OUT_EVAL_HARD"
if [ ! -f "$GFA" ]; then
  echo "WARN: GFA $GFA missing; skipping hard pangenome eval." >&2
else
  python -u scripts/eval.py \
    --data real \
    --reference-fasta "$REF_FA" \
    --gfa "$GFA" \
    --truth-bam "$TRUTH_BAM" \
    --region "$HARD_REGION" \
    --modality "$MODALITY" \
    --ref-mode pangenome \
    --mode two_pass \
    --checkpoint "$CKPT" \
    --batch-size "$BATCH" \
    --workers "$WORKERS" \
    ${MAX_READS:+--max-reads "$MAX_READS"} \
    --device cuda --require-gpu \
    --no-bam \
    --out "$OUT_EVAL_HARD"
fi

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo
echo "############################################################"
echo "# DONE"
echo "#   checkpoint : $CKPT"
echo "#   easy eval  : $OUT_EVAL_EASY/linear/metrics.json"
echo "#   hard eval  : $OUT_EVAL_HARD/pangenome/metrics.json"
echo "############################################################"
echo
echo "Compare locus_accuracy / chain_accuracy / anchor_auc / mapq_mae across:"
echo "  untrained CPU baseline : locus 99.0%  chain 0.7%  anchorAUC 0.495"
echo "  trained (easy)         : cat $OUT_EVAL_EASY/linear/metrics.json"
echo "  trained (hard)         : cat $OUT_EVAL_HARD/pangenome/metrics.json"
echo
echo "For variant-level accuracy (SNV/indel F1 vs GIAB), run hap.py on the"
echo "predicted BAM in GIAB high-confidence regions (needs the GIAB VCF/BED"
echo "under data/chr21/HG002/truth/); drop --no-bam above to emit the BAM."
