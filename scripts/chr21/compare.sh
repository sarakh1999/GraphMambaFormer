#!/usr/bin/env bash
# Side-by-side comparison of the Giraffe and GraphMambaFormer arms on $CHR.
# Reads the BAMs / DeepVariant VCFs / hap.py summaries already produced and
# writes a table + charts under data/chr21/<SAMPLE>/compare/.
#
# usage:
#   ./scripts/chr21/compare.sh
#   ./scripts/chr21/compare.sh giraffe ours bwa
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
D="$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"
  else PYTHON="python3"; fi
fi

LABELS=("$@"); [ ${#LABELS[@]} -eq 0 ] && LABELS=(giraffe ours)

"$PYTHON" "$D/compare_giraffe_ours.py" \
  --sample "$SAMPLE" --chr "$CHR" --labels "${LABELS[@]}"
