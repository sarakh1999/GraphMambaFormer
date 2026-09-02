#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Mentor task: one HPRC sample × chr21
#   Giraffe vs GraphMambaFormer  →  DeepVariant (small variants)
#   Long-read BAM                →  Sniffles / longcallD (SVs; longcallD also small)
#
# Defaults:
#   SAMPLE=HG002   (GIAB truth; held out of HPRC pangenome training set)
#   CHR=chr21
#
# Run from your OWN Terminal (needs Docker). Cursor agent shell cannot use
# the Docker socket.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PLATFORM="${PLATFORM:-linux/amd64}"
THREADS="${THREADS:-4}"
SAMPLE="${SAMPLE:-HG002}"
CHR="${CHR:-chr21}"

VG_IMAGE="${VG_IMAGE:-quay.io/vgteam/vg:latest}"
BWA_IMAGE="${BWA_IMAGE:-biocontainers/bwa:v0.7.17_cv1}"
BCFTOOLS_IMAGE="${BCFTOOLS_IMAGE:-quay.io/biocontainers/bcftools:1.19--h8b25389_0}"
DV_IMAGE="${DV_IMAGE:-google/deepvariant:1.6.1}"
HAPPY_IMAGE="${HAPPY_IMAGE:-jmcdani20/hap.py:v0.3.12}"
MINIMAP_IMAGE="${MINIMAP_IMAGE:-quay.io/biocontainers/minimap2:2.28--h577a1d6_4}"
SNIFFLES_IMAGE="${SNIFFLES_IMAGE:-quay.io/biocontainers/sniffles:2.5.3--pyhdfd78af_0}"
SAMTOOLS_IMAGE="${SAMTOOLS_IMAGE:-$VG_IMAGE}"

FULL_IDX_DIR="$ROOT/data/downloaded_data/Giraffe prebuilt indexes"
FULL_GBZ="$FULL_IDX_DIR/hprc-v1.1-mc-grch38.d9.gbz"

RUN_DIR="$ROOT/data/chr21/${SAMPLE}"
REF_DIR="$RUN_DIR/ref"
TRUTH_DIR="$RUN_DIR/truth"
READS_DIR="$RUN_DIR/reads"
HIFI_DIR="$RUN_DIR/hifi"
BAM_DIR="$RUN_DIR/bam"
VCF_DIR="$RUN_DIR/vcf"
SV_DIR="$RUN_DIR/sv"
EVAL_DIR="$RUN_DIR/eval"

REF_FULL_GZ="$REF_DIR/GRCh38_full.fasta.gz"
REF_FA="$REF_DIR/GRCh38.${CHR}.fa"
CHR_PREFIX="$RUN_DIR/${CHR}"
TRUTH_VCF="$TRUTH_DIR/${SAMPLE}.${CHR}.benchmark.vcf.gz"
TRUTH_BED="$TRUTH_DIR/${SAMPLE}.${CHR}.benchmark.bed"

mkdir -p "$REF_DIR" "$TRUTH_DIR" "$READS_DIR" "$HIFI_DIR" \
         "$BAM_DIR" "$VCF_DIR" "$SV_DIR" "$EVAL_DIR"

dock() {
  local img="$1"; shift
  docker run --rm --platform "$PLATFORM" -v "$ROOT:/work" -w /work "$img" "$@"
}
dockvg() { dock "$VG_IMAGE" "$@"; }
inwork() { printf '/work/%s' "${1#"$ROOT"/}"; }
require() {
  [ -f "$1" ] || { echo "ERROR: missing $1${2:+  ($2)}" >&2; exit 1; }
}
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing command '$1'${2:+  ($2)}" >&2
    exit 1
  }
}

echo "[lib] SAMPLE=$SAMPLE  CHR=$CHR  THREADS=$THREADS"
echo "[lib] outputs -> $RUN_DIR"
echo "[lib] sample links -> $ROOT/data/hprc/SAMPLE_LINKS.md"
