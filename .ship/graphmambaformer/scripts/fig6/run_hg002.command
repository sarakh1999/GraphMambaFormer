#!/usr/bin/env bash
set -o pipefail
cd "/Users/skhosravi/Desktop/mambaformer" || exit 1
mkdir -p data/fig6/HG002
SAMPLE=HG002 ./scripts/fig6/run_all.sh 2>&1 | tee data/fig6/HG002/run.log
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > data/fig6/HG002/run.exit
exit "$status"
