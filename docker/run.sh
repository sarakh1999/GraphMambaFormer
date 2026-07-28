#!/usr/bin/env bash
# Run a command inside a GraphMambaFormer image with the repo bind-mounted at
# /work, so data/ is read and written in place on the host.
#
#   docker/run.sh                                   # interactive shell
#   docker/run.sh gmf-doctor                        # smoke test
#   docker/run.sh gmf-python scripts/smoke_test.py  # model checks
#   SAMPLE=HG002 docker/run.sh scripts/fig6/run_all.sh
#
# The Docker socket is mounted so eval_happy.sh can reach the external hap.py
# image; FIG6_HOST_ROOT tells it which host path to bind into that sibling
# container. Prefix MOUNT_DOCKER_SOCK=0 to run without it (only the hap.py
# stage needs it).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-graphmambaformer:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
# DeepVariant's call_variants multiprocessing dies with `_queue.Empty` on
# Docker's default 64 MB /dev/shm.
SHM_SIZE="${DV_SHM_SIZE:-8g}"
MOUNT_DOCKER_SOCK="${MOUNT_DOCKER_SOCK:-1}"

args=(
  --rm
  --platform "$PLATFORM"
  --shm-size "$SHM_SIZE"
  -v "$ROOT:/work"
  -w /work
  -e "FIG6_HOST_ROOT=$ROOT"
)

for var in SAMPLE CHR THREADS DOWNSAMPLE USE_FULL_GBZ REFPATH_OVERRIDE \
           GRCH38_URL READS_ALN_URL HAPPY_IMAGE; do
  if [ -n "${!var:-}" ]; then
    args+=(-e "$var=${!var}")
  fi
done

sock="${DOCKER_SOCK:-/var/run/docker.sock}"
if [ "$MOUNT_DOCKER_SOCK" = "1" ] && [ -S "$sock" ]; then
  args+=(-v "$sock:/var/run/docker.sock")
fi

if [ -t 0 ]; then
  args+=(-it)
fi

if [ "$#" -eq 0 ]; then
  exec docker run "${args[@]}" "$IMAGE" bash
fi
exec docker run "${args[@]}" "$IMAGE" "$@"
