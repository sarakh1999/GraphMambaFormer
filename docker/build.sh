#!/usr/bin/env bash
# Build a GraphMambaFormer image (linux/amd64 — DeepVariant publishes no arm64
# build, so this runs under Rosetta on Apple Silicon).
#
#   docker/build.sh                 # graphmambaformer:latest  (everything, CPU)
#   TARGET=fig6 docker/build.sh     # fig6-runner:1.6.1        (benchmark only)
#   TARGET=gpu  docker/build.sh     # graphmambaformer:gpu     (CUDA cu124)
#   IMAGE=gmf:dev docker/build.sh --no-cache
#
# GPU channel selection (target=gpu only):
#   TORCH_CHANNEL=cu124 docker/build.sh   # NVIDIA, Volta..Blackwell (default)
#   TORCH_CHANNEL=cu121 docker/build.sh   # NVIDIA, older drivers
#   TORCH_CHANNEL=rocm6.0 docker/build.sh # AMD
#   INSTALL_CUPY=1 TARGET=gpu docker/build.sh  # + NVRTC raw-kernel tier
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-full}"
PLATFORM="${PLATFORM:-linux/amd64}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
INSTALL_TRITON="${INSTALL_TRITON:-1}"
INSTALL_CUPY="${INSTALL_CUPY:-0}"

case "$TARGET" in
  full) DEFAULT_IMAGE="graphmambaformer:latest" ;;
  fig6) DEFAULT_IMAGE="fig6-runner:1.6.1" ;;
  gpu)  DEFAULT_IMAGE="graphmambaformer:gpu" ;;
  *)    echo "ERROR: TARGET must be 'full', 'fig6' or 'gpu' (got '$TARGET')" >&2; exit 1 ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"

export DOCKER_BUILDKIT=1

build_args=()
if [ "$TARGET" = "gpu" ]; then
  build_args+=(--build-arg "TORCH_CHANNEL=$TORCH_CHANNEL"
               --build-arg "INSTALL_TRITON=$INSTALL_TRITON"
               --build-arg "INSTALL_CUPY=$INSTALL_CUPY")
  echo "Building $IMAGE  (target=gpu, channel=$TORCH_CHANNEL, platform=$PLATFORM)"
else
  echo "Building $IMAGE  (target=$TARGET, platform=$PLATFORM)"
fi
echo "This pulls the ~10 GB DeepVariant base and compiles htslib — expect a long first run."

docker build \
  --platform "$PLATFORM" \
  --target "$TARGET" \
  "${build_args[@]}" \
  -f "$ROOT/docker/Dockerfile" \
  -t "$IMAGE" \
  "$@" \
  "$ROOT"

echo
echo "Built $IMAGE:"
# The GPU image needs the device passed through, or gmf-doctor correctly reports
# that it fell back to CPU.
doctor_args=(--rm --platform "$PLATFORM")
if [ "$TARGET" = "gpu" ]; then
  case "$TORCH_CHANNEL" in
    cu*)   docker info 2>/dev/null | grep -qi nvidia && doctor_args+=(--gpus all) ;;
    rocm*) [ -e /dev/kfd ] && doctor_args+=(--device /dev/kfd --device /dev/dri) ;;
  esac
fi
docker run "${doctor_args[@]}" "$IMAGE" gmf-doctor
