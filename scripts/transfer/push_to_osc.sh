#!/usr/bin/env bash
# Push the mambaformer project to OSC Ascend.
#
# WHY a script you run yourself: OSC login needs your password + a Duo MFA push,
# which must be typed/approved interactively. Run this from YOUR terminal.
#
# OSC filesystem note: compute nodes (e.g. a0022) share $HOME with the login
# node, so we rsync ONCE to ascend.osc.edu; the files are then visible after
# `ssh a0022`. No second copy to the compute node is needed.
#
# usage:
#   ./scripts/transfer/push_to_osc.sh                 # default user/dest below
#   OSC_USER=sarakhosravi OSC_DEST='~/mambaformer' ./scripts/transfer/push_to_osc.sh
#   INCLUDE_VG=0 ./scripts/transfer/push_to_osc.sh    # skip the 1.1G chr21.d9.vg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OSC_USER="${OSC_USER:-sarakhosravi}"
OSC_HOST="${OSC_HOST:-ascend.osc.edu}"
OSC_DEST="${OSC_DEST:-mambaformer}"     # path relative to $HOME on OSC
INCLUDE_VG="${INCLUDE_VG:-1}"           # 1 = send chr21.d9.vg (1.1G), 0 = skip

EXCLUDE_FILE="$ROOT/scripts/transfer/osc_exclude.txt"
EXTRA=()
[ "$INCLUDE_VG" = "0" ] && EXTRA+=(--exclude 'data/chr21/HG002/chr21.d9.vg')

echo "Pushing $ROOT/  ->  ${OSC_USER}@${OSC_HOST}:${OSC_DEST}/"
echo "  (you will be prompted for your OSC password + Duo)"
# --partial/--append-verify makes the transfer resumable if Duo/network drops.
rsync -avz --human-readable --progress \
  --partial --append-verify \
  --exclude-from="$EXCLUDE_FILE" "${EXTRA[@]}" \
  ./ "${OSC_USER}@${OSC_HOST}:${OSC_DEST}/"

echo
echo "Done. Next steps on OSC:"
echo "  ssh ${OSC_USER}@${OSC_HOST}"
echo "  ssh a0022                       # GPU node (shares \$HOME)"
echo "  cd ${OSC_DEST}"
echo "  # rebuild the environment (do NOT copy .venv):"
echo "  module load python cuda         # or miniconda3, per OSC docs"
echo "  python -m venv .venv && source .venv/bin/activate"
echo "  pip install -r requirements-gpu.txt"
