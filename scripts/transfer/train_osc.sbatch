#!/usr/bin/env bash
#SBATCH --job-name=gmf_hg002_all3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_osc_%j.out
# REQUIRED on OSC: set your project account (e.g. PAS1234). Either uncomment and
# edit the next line, or submit with:  sbatch --account=PASxxxx ...
##SBATCH --account=PASXXXX
# Ascend GPU partition name may be needed; check `sinfo` / OSC docs, e.g.:
##SBATCH --partition=gpu
#
# Hours-long hybrid (max-neural) training on all 3 modalities for HG002 chr21,
# ref-mode BOTH (linear + pangenome graph towers), A100-80GB (BF16), with a
# checkpoint every 50 steps and all plots generated at the end.
#
#   cd ~/mambaformer
#   sbatch --account=PASxxxx scripts/transfer/train_osc.sbatch
#
# Tunables via env (sbatch --export=ALL,EPOCHS=60,...):
#   EPOCHS (default 40)  BATCH (default 8)  DMODEL (default 256)
#   OUT (default data/training_runs/chr21_hg002_all3_both_osc)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
source .venv/bin/activate

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8}"
DMODEL="${DMODEL:-256}"
OUT="${OUT:-data/training_runs/chr21_hg002_all3_both_osc}"
MANIFEST="${MANIFEST:-data/hprc/manifests/chr21_HG002_all_modalities_both.json}"

# If the fused Mamba kernel is available, train FULL-LENGTH reads (no cap).
# Otherwise cap read length so the Python scan fits in 80GB.
if python -c "from graphmambaformer.accel import mamba_ssm_available; import sys; sys.exit(0 if mamba_ssm_available() else 1)" 2>/dev/null; then
  MRL_ARGS=()
  echo "mamba-ssm: available -> training full-length reads"
else
  MRL_ARGS=(--max-read-len 1024)
  echo "mamba-ssm: NOT available -> capping reads at 1024 bp"
fi

mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A100 = Ampere: BF16 (no FP8). Checkpoints every 50 steps; plots auto-generated.
PYTHONPATH=. python scripts/train.py --data real \
  --manifest "$MANIFEST" \
  --ref-mode both --device cuda --require-gpu --no-fp8 \
  --d-model "$DMODEL" --batch-size "$BATCH" --epochs "$EPOCHS" \
  --patience "$EPOCHS" --lr 3e-4 \
  --save-every-steps 50 \
  "${MRL_ARGS[@]}" \
  --out "$OUT"

echo "training done -> $OUT"
echo "plots:        $OUT/plots/"
echo "checkpoints:  $OUT/checkpoints/ (epoch_XX.pt + step_XXXXXX.pt), best -> $OUT/checkpoint.pt"

# --- per-modality evaluation (metrics + predicted BAMs) -------------------- #
CKPT="$OUT/checkpoint.pt"
if [ -f "$CKPT" ]; then
  REF=data/chr21/HG002/ref/chr21.fa
  GFA=data/chr21/HG002/chr21.q.window.gfa
  REGION=chr21:20000000-20050000
  declare -A TRUTH=(
    [illumina]=data/chr21/HG002/bam/HG002.chr21.q.giab.sorted.bam
    [pacbio_hifi]=data/chr21/HG002/bam/HG002.chr21.pacbio_hifi.truth.sorted.bam
    [ont]=data/chr21/HG002/bam/HG002.chr21.ont.truth.sorted.bam
  )
  for MOD in illumina pacbio_hifi ont; do
    echo "=== eval $MOD ==="
    PYTHONPATH=. python scripts/eval.py --data real \
      --reference-fasta "$REF" --gfa "$GFA" --region "$REGION" \
      --ref-mode both --mode hybrid --no-fp8 \
      --modality "$MOD" --truth-bam "${TRUTH[$MOD]}" \
      --checkpoint "$CKPT" --d-model "$DMODEL" \
      "${MRL_ARGS[@]}" \
      --device cuda --out "data/eval_runs/chr21_hg002_all3_both_osc/$MOD"
  done
  echo "evaluations -> data/eval_runs/chr21_hg002_all3_both_osc/<modality>/metrics.json"
fi
