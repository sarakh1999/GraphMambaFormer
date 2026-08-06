#!/usr/bin/env bash
# Build a GraphMambaFormer image (linux/amd64 — DeepVariant publishes no arm64
# build, so this runs under Rosetta on Apple Silicon).
#
#   docker/build.sh                 # graphmambaformer:latest  (everything, CPU)
#   TARGET=fig6 docker/build.sh     # fig6-runner:1.6.1        (benchmark only)
#   TARGET=gpu  docker/build.sh     # graphmambaformer:gpu     (NVIDIA/AMD)
#   TARGET=arm  docker/build.sh     # graphmambaformer:latest  (native arm64, CPU)
#   TARGET=intel docker/build.sh    # graphmambaformer:xpu     (Intel GPU / XPU)
#   IMAGE=gmf:dev docker/build.sh --no-cache
#
# Publish a built image so others can pull it (no rebuild):
#   docker/publish.sh                 # → ghcr.io/<you>/graphmambaformer:latest
#
# CPU coverage: the default (amd64) image runs on both Intel and AMD x86-64
# CPUs — they are the same architecture — so "Intel CPU" needs no separate
# build. TARGET=intel is specifically for Intel GPUs (XPU: Arc / Data Center
# GPU Max), which need the torch XPU wheel and a newer Python than the
# DeepVariant base ships.
#
# Architecture / vendor targets:
#   default (amd64)  Intel + AMD x86-64 CPUs, full Fig 6a genomics stack
#   TARGET=arm       native linux/arm64 (Apple Silicon / Graviton), model-only
#   TARGET=gpu +     NVIDIA CUDA (cu124/cu121) or AMD ROCm (rocm6.0) GPUs
#     TORCH_CHANNEL
#   TARGET=intel     Intel GPU (XPU), model-only (docker/Dockerfile.xpu)
#
# TARGET=arm and TARGET=intel build model-only images: DeepVariant and vg are
# amd64/Python-3.8-only, so those images drop the Fig 6a variant-calling
# baseline and ship the model stack plus samtools/bcftools/bwa. Use the default
# (amd64, under Rosetta on Apple Silicon) for the full genomics stack.
#
# GPU channel selection (target=gpu only):
#   TORCH_CHANNEL=cu124 docker/build.sh   # NVIDIA, sm_70..sm_90 (default;
#                                         #   Blackwell needs newer args, see Dockerfile)
#   TORCH_CHANNEL=cu121 docker/build.sh   # NVIDIA, older drivers
#   TORCH_CHANNEL=rocm6.0 docker/build.sh # AMD
#   INSTALL_CUPY=1 TARGET=gpu docker/build.sh  # + NVRTC raw-kernel tier
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-full}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
INSTALL_TRITON="${INSTALL_TRITON:-1}"
INSTALL_CUPY="${INSTALL_CUPY:-0}"

# The native arm64 image lives in its own Dockerfile with its own final stage;
# everything else is a target in the multi-stage docker/Dockerfile.
DOCKERFILE="$ROOT/docker/Dockerfile"
BUILD_TARGET="$TARGET"
DEFAULT_PLATFORM="linux/amd64"

case "$TARGET" in
  full) DEFAULT_IMAGE="graphmambaformer:latest" ;;
  fig6) DEFAULT_IMAGE="fig6-runner:1.6.1" ;;
  gpu)  DEFAULT_IMAGE="graphmambaformer:gpu" ;;
  arm)  DEFAULT_IMAGE="graphmambaformer:latest"
        DOCKERFILE="$ROOT/docker/Dockerfile.arm64"
        BUILD_TARGET="full"
        DEFAULT_PLATFORM="linux/arm64" ;;
  intel|xpu) DEFAULT_IMAGE="graphmambaformer:xpu"
        DOCKERFILE="$ROOT/docker/Dockerfile.xpu"
        BUILD_TARGET="full"
        DEFAULT_PLATFORM="linux/amd64" ;;
  *)    echo "ERROR: TARGET must be 'full', 'fig6', 'gpu', 'arm' or 'intel' (got '$TARGET')" >&2; exit 1 ;;
esac
PLATFORM="${PLATFORM:-$DEFAULT_PLATFORM}"
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
case "$TARGET" in
  arm)   echo "Native arm64 model image (no DeepVariant/vg); compiles htslib — expect a long first run." ;;
  intel|xpu) echo "Intel XPU model image (no DeepVariant/vg); pulls the torch XPU wheel + compiles htslib — expect a long first run." ;;
  *)     echo "This pulls the ~10 GB DeepVariant base and compiles htslib — expect a long first run." ;;
esac

docker build \
  --platform "$PLATFORM" \
  --target "$BUILD_TARGET" \
  ${build_args[@]+"${build_args[@]}"} \
  -f "$DOCKERFILE" \
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
elif [ "$TARGET" = "intel" ] || [ "$TARGET" = "xpu" ]; then
  # Intel GPUs surface as /dev/dri; pass it through so the doctor can see the
  # XPU tier rather than reporting a CPU fallback.
  [ -d /dev/dri ] && doctor_args+=(--device /dev/dri)
fi
docker run "${doctor_args[@]}" "$IMAGE" gmf-doctor
