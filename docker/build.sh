#!/usr/bin/env bash
# Build a GraphMambaFormer image (linux/amd64 — DeepVariant publishes no arm64
# build, so this runs under Rosetta on Apple Silicon).
#
#   docker/build.sh                 # graphmambaformer:latest  (everything)
#   TARGET=fig6 docker/build.sh     # fig6-runner:1.6.1        (benchmark only)
#   IMAGE=gmf:dev docker/build.sh --no-cache
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-full}"
PLATFORM="${PLATFORM:-linux/amd64}"

case "$TARGET" in
  full) DEFAULT_IMAGE="graphmambaformer:latest" ;;
  fig6) DEFAULT_IMAGE="fig6-runner:1.6.1" ;;
  *)    echo "ERROR: TARGET must be 'full' or 'fig6' (got '$TARGET')" >&2; exit 1 ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"

export DOCKER_BUILDKIT=1

echo "Building $IMAGE  (target=$TARGET, platform=$PLATFORM)"
echo "This pulls the ~10 GB DeepVariant base and compiles htslib — expect a long first run."
docker build \
  --platform "$PLATFORM" \
  --target "$TARGET" \
  -f "$ROOT/docker/Dockerfile" \
  -t "$IMAGE" \
  "$@" \
  "$ROOT"

echo
echo "Built $IMAGE:"
docker run --rm --platform "$PLATFORM" "$IMAGE" gmf-doctor
