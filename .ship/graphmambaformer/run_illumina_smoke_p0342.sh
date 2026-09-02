#!/usr/bin/env bash
# Setup + Illumina R1/R2 smoke on OSC GPU node (e.g. p0342)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/4] env"
hostname
nvidia-smi -L || true

if ! python3 -c 'import torch' 2>/dev/null; then
  echo "[2/4] creating .venv and installing deps"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip wheel
  pip install torch --index-url https://download.pytorch.org/whl/cu121 || pip install torch
  pip install -r requirements.txt || true
  pip install -r requirements-gpu.txt || true
else
  echo "[2/4] torch already importable"
  # shellcheck disable=SC1091
  [ -f .venv/bin/activate ] && source .venv/bin/activate || true
fi

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

echo "[3/4] python=$PY; torch check"
"$PY" - <<'PY'
import torch
print(
    "torch", torch.__version__,
    "cuda", torch.cuda.is_available(),
    "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
)
PY

REF=data/smoke_illumina/smoke.chr.fa
R1=data/smoke_illumina/smoke.R1.fastq.gz
R2=data/smoke_illumina/smoke.R2.fastq.gz
OUT=data/training_runs/illumina_smoke_p0342
mkdir -p "$OUT" data/eval_runs

echo "[4/4] Illumina R1/R2 train (pseudo-labels)"
PYTHONPATH=. "$PY" scripts/train.py --data real \
  --reference-fasta "$REF" \
  --reads-file "$R1" \
  --reads-file "$R2" \
  --read-layout paired --modality illumina --ref-mode linear \
  --region smoke_chr --device cuda --require-gpu \
  --epochs 3 --batch-size 4 --d-model 64 --max-reads 80 \
  --out "$OUT"

echo "Train done:"
ls -lh "$OUT" | head

if [ -f "$OUT/checkpoint.pt" ]; then
  echo "Hybrid eval with checkpoint"
  PYTHONPATH=. "$PY" scripts/eval.py --data real \
    --reference-fasta "$REF" \
    --reads-file "$R1" --reads-file "$R2" \
    --read-layout paired --modality illumina --ref-mode linear \
    --region smoke_chr --mode hybrid \
    --checkpoint "$OUT/checkpoint.pt" \
    --device cuda \
    --out data/eval_runs/illumina_smoke_p0342
fi

echo "DONE"
nvidia-smi
