#!/usr/bin/env bash
# Rebuild CPU + GPU images and push to GHCR.
# Run in your OWN Terminal (Docker Desktop must be running):
#   ./scripts/rebuild_and_publish_images.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GH_CONFIG_DIR="${GH_CONFIG_DIR:-$ROOT/.tools/gh-config}"
export PATH="$ROOT/.tools:${PATH}"

echo "==> build CPU (latest)"
docker/build.sh
echo "==> publish :latest"
docker/publish.sh

echo "==> build GPU"
TARGET=gpu docker/build.sh
echo "==> publish :gpu"
TARGET=gpu docker/publish.sh

echo
echo "Done. Pull with:"
echo "  docker pull ghcr.io/sarakh1999/graphmambaformer:latest"
echo "  docker pull ghcr.io/sarakh1999/graphmambaformer:gpu"
