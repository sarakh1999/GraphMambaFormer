#!/usr/bin/env bash
# Publish a local GraphMambaFormer image to GitHub Container Registry (GHCR)
# so anyone can `docker pull` it without rebuilding.
#
#   docker/publish.sh                 # push graphmambaformer:latest → :latest
#   TARGET=gpu docker/publish.sh      # push graphmambaformer:gpu → :gpu
#   TAG=v0.1.0 docker/publish.sh      # also tag :v0.1.0
#   DRY_RUN=1 docker/publish.sh       # print actions only
#
# Prerequisites:
#   - a local image (docker/build.sh)
#   - gh authenticated:  gh auth login
#   - package write scope: gh auth refresh -s write:packages
#
# After the first push, the script marks the package public so anonymous
# pulls work. If that step fails, set visibility at:
#   https://github.com/users/<user>/packages/container/graphmambaformer/settings
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-full}"
TAG="${TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
REGISTRY="${REGISTRY:-ghcr.io}"

# Prefer the repo-bundled gh if present (matches other scripts in this tree).
GH="${GH:-}"
if [ -z "$GH" ]; then
  if [ -x "$ROOT/.tools/gh" ]; then
    GH="$ROOT/.tools/gh"
    export GH_CONFIG_DIR="${GH_CONFIG_DIR:-$ROOT/.tools/gh-config}"
  else
    GH="$(command -v gh || true)"
  fi
fi
if [ -z "$GH" ]; then
  echo "ERROR: gh not found. Install GitHub CLI or place it at .tools/gh" >&2
  exit 1
fi

case "$TARGET" in
  full|latest|"")
    LOCAL_IMAGE="${IMAGE:-graphmambaformer:latest}"
    REMOTE_TAG="latest"
    ;;
  fig6)
    LOCAL_IMAGE="${IMAGE:-fig6-runner:1.6.1}"
    REMOTE_TAG="fig6-1.6.1"
    ;;
  gpu)
    LOCAL_IMAGE="${IMAGE:-graphmambaformer:gpu}"
    REMOTE_TAG="gpu"
    ;;
  arm)
    LOCAL_IMAGE="${IMAGE:-graphmambaformer:latest}"
    REMOTE_TAG="arm64"
    ;;
  intel|xpu)
    LOCAL_IMAGE="${IMAGE:-graphmambaformer:xpu}"
    REMOTE_TAG="xpu"
    ;;
  *)
    echo "ERROR: TARGET must be full|fig6|gpu|arm|intel (got '$TARGET')" >&2
    exit 1
    ;;
esac

OWNER="${OWNER:-}"
if [ -z "$OWNER" ]; then
  # Prefer the GitHub login; fall back to the origin remote owner.
  OWNER="$("$GH" api user -q .login 2>/dev/null || true)"
fi
if [ -z "$OWNER" ]; then
  remote="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
  # https://github.com/user/repo.git  or  git@github.com:user/repo.git
  OWNER="$(printf '%s\n' "$remote" | sed -nE 's#.*(github\.com[:/])([^/]+)/.*#\2#p')"
fi
if [ -z "$OWNER" ]; then
  echo "ERROR: could not determine GitHub owner. Set OWNER=youruser" >&2
  exit 1
fi
OWNER="$(printf '%s' "$OWNER" | tr '[:upper:]' '[:lower:]')"

PACKAGE="${PACKAGE:-graphmambaformer}"
REMOTE_IMAGE="${REGISTRY}/${OWNER}/${PACKAGE}"

if ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: local image '$LOCAL_IMAGE' not found." >&2
  echo "Build it first, e.g.:  docker/build.sh" >&2
  exit 1
fi

echo "Local : $LOCAL_IMAGE"
echo "Remote: ${REMOTE_IMAGE}:${REMOTE_TAG}"
[ -n "$TAG" ] && echo "Extra : ${REMOTE_IMAGE}:${TAG}"

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# Login: prefer GITHUB_TOKEN / GH_TOKEN (classic PAT with write:packages),
# otherwise a token from an authenticated gh CLI.
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [ -z "$token" ]; then
  token="$("$GH" auth token 2>/dev/null || true)"
fi
if [ -z "$token" ]; then
  cat >&2 <<EOF
ERROR: no GitHub credentials for GHCR.

Option A — use the bundled GitHub CLI (recommended):
  export GH_CONFIG_DIR=$ROOT/.tools/gh-config
  $GH auth login -h github.com -p https -w
  $GH auth refresh -h github.com -s write:packages
  docker/publish.sh

Option B — personal access token (repo + write:packages):
  GITHUB_TOKEN=ghp_... docker/publish.sh
EOF
  exit 1
fi

# Avoid macOS Keychain (osxkeychain error -25293): never call `docker login`.
# Write a throwaway DOCKER_CONFIG with the GHCR auth embedded as base64.
docker_cfg="${TMPDIR:-/tmp}/gmf-docker-config-$$"
cleanup_docker_cfg() { rm -rf "$docker_cfg"; }
trap cleanup_docker_cfg EXIT
mkdir -p "$docker_cfg"
auth_b64="$(printf '%s:%s' "$OWNER" "$token" | base64 | tr -d '\n')"
cat > "$docker_cfg/config.json" <<EOF
{
  "auths": {
    "${REGISTRY}": {
      "auth": "${auth_b64}"
    }
  }
}
EOF
export DOCKER_CONFIG="$docker_cfg"
echo "Using ephemeral Docker config (no macOS Keychain)."

if [ "$DRY_RUN" = "1" ]; then
  echo "+ docker push ${REMOTE_IMAGE}:${REMOTE_TAG}   # (auth embedded, login skipped)"
fi

run docker tag "$LOCAL_IMAGE" "${REMOTE_IMAGE}:${REMOTE_TAG}"
run docker push "${REMOTE_IMAGE}:${REMOTE_TAG}"

if [ -n "$TAG" ]; then
  run docker tag "$LOCAL_IMAGE" "${REMOTE_IMAGE}:${TAG}"
  run docker push "${REMOTE_IMAGE}:${TAG}"
fi

# Best-effort: make the package public for anonymous pulls.
if [ "$DRY_RUN" != "1" ]; then
  if "$GH" api \
      --method PUT \
      -H "Accept: application/vnd.github+json" \
      "/user/packages/container/${PACKAGE}/visibility" \
      -f visibility=public >/dev/null 2>&1; then
    echo "Package visibility set to public."
  else
    echo
    echo "NOTE: could not set package visibility automatically."
    echo "Make it public at:"
    echo "  https://github.com/users/${OWNER}/packages/container/${PACKAGE}/settings"
  fi
fi

echo
echo "Published. Anyone can run:"
echo "  docker pull ${REMOTE_IMAGE}:${REMOTE_TAG}"
echo "  docker run --rm -it ${REMOTE_IMAGE}:${REMOTE_TAG} gmf-doctor"
echo "  IMAGE=${REMOTE_IMAGE}:${REMOTE_TAG} docker/run.sh gmf-python scripts/smoke_test.py"
