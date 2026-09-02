#!/usr/bin/env bash
# Preflight check for the chr21 benchmark: report what is available and what
# each missing prerequisite would block. Informational — always exits 0 so
# run_all.sh can decide what to run.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ok()   { printf "  [ok]   %-8s %s\n" "$1" "$2"; }
miss() { printf "  [MISS] %-8s %s\n" "$1" "$2"; }

echo "Preflight — SAMPLE=$SAMPLE CHR=$CHR"
echo

DOCKER_OK=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_OK=1; ok docker "Giraffe / DeepVariant / Sniffles / hap.py arms"
else
  miss docker "blocks Giraffe, DeepVariant, Sniffles, hap.py (start Docker Desktop)"
fi

if command -v aws >/dev/null 2>&1; then
  ok aws "HPRC S3 read/CRAM download"
else
  miss aws "blocks HPRC S3 fetch (brew install awscli). GIAB https reads still work."
fi

command -v curl >/dev/null 2>&1 && ok curl "reference / truth download" \
  || miss curl "blocks reference/truth download"

# Python arm (our implementation) — needs no Docker.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"
  else PYTHON="python3"; fi
fi
if command -v "$PYTHON" >/dev/null 2>&1 \
   && PYTHONPATH="$ROOT" "$PYTHON" -c "import torch,numpy,pysam,graphmambaformer" >/dev/null 2>&1; then
  ok python "GraphMambaFormer 'ours' arm ($PYTHON)"
else
  miss python "blocks 'ours' arm — pip install -r requirements.txt (SKIP_OURS=1 to skip)"
fi

echo
if [ "$DOCKER_OK" != "1" ]; then
  cat <<EOF
NOTE: Docker is unavailable in this shell (this is expected inside the Cursor
agent). Run this pipeline from your own Terminal with Docker Desktop running.
The 'ours' arm (map_ours.sh / compare.sh) does not need Docker and can run here.
EOF
fi
exit 0
