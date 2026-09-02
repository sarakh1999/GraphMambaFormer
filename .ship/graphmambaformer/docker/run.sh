#!/usr/bin/env bash
# Run a command inside a GraphMambaFormer image with the repo bind-mounted at
# /work, so data/ is read and written in place on the host.
#
#   docker/run.sh                                   # interactive shell
#   docker/run.sh gmf-doctor                        # smoke test
#   docker/run.sh gmf-python scripts/smoke_test.py  # model checks
#   docker/run.sh gmf-python scripts/train.py --reads 64   # train + plot
#   SAMPLE=HG002 CHR=chr1 docker/run.sh scripts/fig6/run_all.sh
#
# Prebuilt images (no local build required):
#   IMAGE=ghcr.io/sarakh1999/graphmambaformer:latest docker/run.sh gmf-doctor
# If IMAGE is unset and no local graphmambaformer:latest exists, this script
# falls back to that GHCR image automatically.
#
# GPU passthrough is automatic when the image was built with TARGET=gpu:
#   IMAGE=graphmambaformer:gpu docker/run.sh gmf-doctor
#   IMAGE=graphmambaformer:xpu GPU=intel docker/run.sh gmf-doctor
# NVIDIA needs the container toolkit on the host; AMD needs /dev/kfd; Intel
# needs /dev/dri. Set GPU=0 to force a CPU run, or GPU=rocm / GPU=intel to
# select the AMD / Intel device flags explicitly.
#
# The Docker socket is mounted so eval_happy.sh can reach the external hap.py
# image; FIG6_HOST_ROOT tells it which host path to bind into that sibling
# container. Prefix MOUNT_DOCKER_SOCK=0 to run without it (only the hap.py
# stage needs it).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${PLATFORM:-linux/amd64}"
# DeepVariant's call_variants multiprocessing dies with `_queue.Empty` on
# Docker's default 64 MB /dev/shm.
SHM_SIZE="${DV_SHM_SIZE:-8g}"
MOUNT_DOCKER_SOCK="${MOUNT_DOCKER_SOCK:-1}"

# Prefer a locally built image; otherwise pull the published GHCR image so
# clone-only users can run without a multi-GB rebuild.
LOCAL_IMAGE="graphmambaformer:latest"
REMOTE_IMAGE="${REMOTE_IMAGE:-ghcr.io/sarakh1999/graphmambaformer:latest}"
if [ -z "${IMAGE:-}" ]; then
  if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    IMAGE="$LOCAL_IMAGE"
  else
    IMAGE="$REMOTE_IMAGE"
    echo "Using published image $IMAGE (no local $LOCAL_IMAGE)" >&2
  fi
fi

args=(
  --rm
  --platform "$PLATFORM"
  --shm-size "$SHM_SIZE"
  -v "$ROOT:/work"
  -w /work
  -e "FIG6_HOST_ROOT=$ROOT"
)

for var in SAMPLE CHR THREADS DOWNSAMPLE USE_FULL_GBZ REFPATH_OVERRIDE \
           GRCH38_URL READS_ALN_URL HAPPY_IMAGE \
           SKIP_OURS SKIP_SNIFFLES SKIP_TRUTH SKIP_EVAL SKIP_COMPARE \
           OURS_MODE OURS_MAX_READS OURS_BAM OURS_FORCE OURS_MODALITY \
           VG_IMAGE DV_IMAGE MINIMAP_IMAGE SNIFFLES_IMAGE \
           SAMTOOLS_IMAGE BCFTOOLS_IMAGE \
           HIFI_FASTQ HIFI_BAM HIFI_BAM_URL; do
  if [ -n "${!var:-}" ]; then
    args+=(-e "$var=${!var}")
  fi
done

sock="${DOCKER_SOCK:-/var/run/docker.sock}"
if [ "$MOUNT_DOCKER_SOCK" = "1" ] && [ -S "$sock" ]; then
  args+=(-v "$sock:/var/run/docker.sock")
fi

# Expose the GPU when one is plausibly present. GPU=auto (the default) keeps a
# CPU host working unchanged: the flags are only added when the host actually
# shows the device, since `--gpus all` on a machine without the NVIDIA runtime
# makes `docker run` fail outright.
case "${GPU:-auto}" in
  0|no|off|cpu) ;;
  rocm|amd)
    args+=(--device /dev/kfd --device /dev/dri --group-add video) ;;
  intel|xpu)
    args+=(--device /dev/dri --group-add video) ;;
  cuda|nvidia)
    args+=(--gpus all) ;;
  auto)
    # /dev/kfd is AMD-specific; the NVIDIA runtime is named in `docker info`.
    # A bare /dev/dri (Intel/integrated) is left alone in auto mode: pass
    # GPU=intel to opt in, so a host with only display graphics is unaffected.
    if [ -e /dev/kfd ]; then
      args+=(--device /dev/kfd --device /dev/dri --group-add video)
    elif docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
      args+=(--gpus all)
    fi ;;
esac

if [ -t 0 ]; then
  args+=(-it)
fi

if [ "$#" -eq 0 ]; then
  exec docker run "${args[@]}" "$IMAGE" bash
fi
exec docker run "${args[@]}" "$IMAGE" "$@"
