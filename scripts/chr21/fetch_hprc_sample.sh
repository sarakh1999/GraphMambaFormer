#!/usr/bin/env bash
# List / download HPRC raw reads for samples in a list file.
#
# usage:
#   ./scripts/chr21/fetch_hprc_sample.sh                 # list first training sample
#   ./scripts/chr21/fetch_hprc_sample.sh HG00438         # list one sample
#   ./scripts/chr21/fetch_hprc_sample.sh HG00438 links  # print resolved URLs from manifest
#   ./scripts/chr21/fetch_hprc_sample.sh HG00438 hifi    # sync HiFi for one sample
#   ./scripts/chr21/fetch_hprc_sample.sh HG00438 illumina
#   SAMPLES_FILE=data/hprc/graph_samples_44.txt ./scripts/chr21/fetch_hprc_sample.sh ALL links
#
# Does NOT download assemblies. Prefer listing first; full HiFi is huge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESOLVE="$ROOT/scripts/chr21/resolve_sample.py"
SAMPLE="${1:-HG00438}"
MODALITY="${2:-list}"   # list | links | hifi | illumina | ont | allraw
OUT_BASE="${OUT_BASE:-$ROOT/data/hprc/reads}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing command '$1'${2:+  ($2)}" >&2
    exit 1
  }
}

bucket_prefix() {
  python3 "$RESOLVE" "$1" raw_bucket 2>/dev/null || {
    local s="$1"
    case "$s" in
      HG002|HG005|HG00733|HG01109|HG01243|HG02055|HG02080|HG02109|HG02145|HG02723|HG02818|HG03098|HG03486|HG03492|NA18906|NA19240|NA20129|NA21309)
        echo "s3://human-pangenomics/working/HPRC_PLUS/${s}/raw_data"
        ;;
      *)
        echo "s3://human-pangenomics/working/HPRC/${s}/raw_data"
        ;;
    esac
  }
}

print_links() {
  local s="$1"
  echo "======== $s ========"
  echo "manifest:  $ROOT/data/hprc/SAMPLE_LINKS.md"
  echo "bucket:    $(bucket_prefix "$s")"
  python3 "$RESOLVE" "$s" illumina_path 2>/dev/null && echo "  -> Illumina CRAM for DeepVariant / Giraffe short-read arm" || echo "  -> no Illumina path in manifest"
  python3 "$RESOLVE" "$s" hifi_bam_url 2>/dev/null && echo "  -> HiFi BAM for Sniffles long-read arm" || echo "  -> no HiFi path in manifest"
}

list_one() {
  local s="$1"
  local p; p="$(bucket_prefix "$s")"
  echo "======== $s  ($p) ========"
  if command -v aws >/dev/null 2>&1; then
    aws s3 ls --no-sign-request "${p}/" || true
  else
    echo "AWS CLI not installed. Use 'links' mode or open:"
    echo "  https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=${p#s3://human-pangenomics/}/"
    print_links "$s"
  fi
}

sync_mod() {
  local s="$1" mod="$2"
  require_cmd aws "install AWS CLI (brew install awscli) before syncing HPRC reads"
  local p; p="$(bucket_prefix "$s")"
  local sub=""
  case "$mod" in
    hifi)     sub="PacBio_HiFi" ;;
    illumina) sub="Illumina" ;;
    ont)      sub="nanopore" ;;
    allraw)   sub="" ;;
    *) echo "unknown modality $mod" >&2; exit 1 ;;
  esac
  local src="$p"
  [ -n "$sub" ] && src="${p}/${sub}"
  local dst="$OUT_BASE/${s}/${mod}"
  mkdir -p "$dst"
  echo "SYNC $src -> $dst"
  aws s3 --no-sign-request sync "$src/" "$dst/"
}

if [ "$SAMPLE" = "ALL" ]; then
  SAMPLES_FILE="${SAMPLES_FILE:-$ROOT/data/hprc/graph_samples_44.txt}"
  while read -r s; do
    [ -z "$s" ] && continue
    [[ "$s" =~ ^# ]] && continue
    case "$MODALITY" in
      list) list_one "$s" ;;
      links) print_links "$s" ;;
      *) sync_mod "$s" "$MODALITY" ;;
    esac
  done < "$SAMPLES_FILE"
  exit 0
fi

case "$MODALITY" in
  list) list_one "$SAMPLE" ;;
  links) print_links "$SAMPLE" ;;
  *) sync_mod "$SAMPLE" "$MODALITY" ;;
esac
