#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared config + helpers for reproducing Nature 2023 HPRC Figure 6a
# (precision/recall of variant calling), scoped to one chromosome so it runs
# on a laptop / CPU.
#
# Feasible CPU arms (2 curves):
#   {vg Giraffe on HPRC MC graph, BWA-MEM on GRCh38} × DeepVariant
#
# Optional later: DeepTrio (needs parents) for 4 curves.
# Skipped: DRAGEN (needs Illumina FPGA hardware).
#
# Usage:
#   SAMPLE=HG002 ./scripts/fig6/run_all.sh
#   SAMPLE=HG005 ./scripts/fig6/run_all.sh
#
# Everything runs inside official Docker images. On Apple Silicon these are
# x86-64, so we force --platform linux/amd64 (Rosetta: correct but slower).
#
# Inside the fig6 runner image (docker/Dockerfile) the tools are installed
# natively, so FIG6_NATIVE=1 makes the helpers below exec them directly instead
# of nesting a container per stage.
# ---------------------------------------------------------------------------
set -euo pipefail

# FIG6_ROOT lets the copy of these scripts baked into the runner image operate
# on the bind-mounted repo at /work instead of its own /opt/fig6 location.
ROOT="${FIG6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# ---- knobs -----------------------------------------------------------------
PLATFORM="${PLATFORM:-linux/amd64}"
# 1 when running inside the fig6 runner image (set as an ENV there).
FIG6_NATIVE="${FIG6_NATIVE:-0}"
# Host-side path of $ROOT. Differs from $ROOT only in the runner, where a
# nested `docker run` must bind the host path, not the container's /work.
FIG6_HOST_ROOT="${FIG6_HOST_ROOT:-$ROOT}"
# Interpreter for plot_pr.py (pandas + matplotlib).
PLOT_PY="${PLOT_PY:-$ROOT/.venv/bin/python}"
# Cap default threads: Apple Silicon runs these images under Rosetta.
_NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
THREADS="${THREADS:-$(( _NCPU > 8 ? 8 : _NCPU ))}"
CHR="${CHR:-chr20}"
SAMPLE="${SAMPLE:-HG002}"

# ---- tool images -----------------------------------------------------------
VG_IMAGE="${VG_IMAGE:-quay.io/vgteam/vg:latest}"
BWA_IMAGE="${BWA_IMAGE:-biocontainers/bwa:v0.7.17_cv1}"
BCFTOOLS_IMAGE="${BCFTOOLS_IMAGE:-quay.io/biocontainers/bcftools:1.19--h8b25389_0}"
# vg's samtools often lacks HTTPS; use biocontainers for remote BAM slicing
SAMTOOLS_IMAGE="${SAMTOOLS_IMAGE:-quay.io/biocontainers/samtools:1.19--h50ea8bc_0}"
DV_IMAGE="${DV_IMAGE:-google/deepvariant:1.6.1}"
HAPPY_IMAGE="${HAPPY_IMAGE:-jmcdani20/hap.py:v0.3.12}"

# ---- inputs you already downloaded ----------------------------------------
FULL_IDX_DIR="$ROOT/data/downloaded_data/Giraffe prebuilt indexes"
FULL_GBZ="$FULL_IDX_DIR/hprc-v1.1-mc-grch38.d9.gbz"

# ---- layout ----------------------------------------------------------------
FIG6_DIR="$ROOT/data/fig6"
REF_DIR="$FIG6_DIR/ref"                    # shared GRCh38
IDX_DIR="$FIG6_DIR/indexes"                # shared chr Giraffe + BWA indexes
RUN_DIR="$FIG6_DIR/${SAMPLE}"              # per-sample outputs
TRUTH_DIR="$RUN_DIR/truth"
READS_DIR="$RUN_DIR/reads"
BAM_DIR="$RUN_DIR/bam"
VCF_DIR="$RUN_DIR/vcf"
EVAL_DIR="$RUN_DIR/eval"
PLOT_DIR="$RUN_DIR/plots"

REF_FULL_GZ="$REF_DIR/GRCh38_full.fasta.gz"
REF_FA="$REF_DIR/GRCh38.${CHR}.fa"
CHR_PREFIX="$IDX_DIR/${CHR}"               # ${CHR}.giraffe.gbz / .min / .dist
TRUTH_VCF="$TRUTH_DIR/${SAMPLE}.${CHR}.benchmark.vcf.gz"
TRUTH_BED="$TRUTH_DIR/${SAMPLE}.${CHR}.benchmark.bed"

R1="$READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz"
R2="$READS_DIR/${SAMPLE}.${CHR}.R2.fastq.gz"

mkdir -p "$REF_DIR" "$IDX_DIR" "$TRUTH_DIR" "$READS_DIR" \
         "$BAM_DIR" "$VCF_DIR" "$EVAL_DIR" "$PLOT_DIR"

# ---- GIAB defaults (truth + short-read BAM for chr slice) ------------------
# Prefer GRCh38 BAMs so chromosome names match truth / ref.
case "$SAMPLE" in
  HG002)
    : "${GIAB_TRUTH_BASE:=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38}"
    : "${GIAB_TRUTH_VCF:=HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz}"
    : "${GIAB_TRUTH_BED:=HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed}"
    : "${READS_ALN_URL:=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_Illumina_2x250bps/novoalign_bams/HG002.GRCh38.2x250.bam}"
    ;;
  HG005)
    : "${GIAB_TRUTH_BASE:=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh38}"
    : "${GIAB_TRUTH_VCF:=HG005_GRCh38_1_22_v4.2.1_benchmark.vcf.gz}"
    : "${GIAB_TRUTH_BED:=HG005_GRCh38_1_22_v4.2.1_benchmark.bed}"
    : "${READS_ALN_URL:=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/ChineseTrio/HG005_NA24631_son/HG005_NA24631_son_HiSeq_300x/NHGRI_Illumina300X_Chinesetrio_novoalign_bams/HG005.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam}"
    ;;
esac

# ---- helpers ---------------------------------------------------------------
# Is the command this stage wants already installed locally?
have_native() {
  case "$1" in
    /*) [ -x "$1" ] ;;
    *)  command -v "$1" >/dev/null 2>&1 ;;
  esac
}

# Run a stage in image $1. In the runner image the tools are local, so exec
# them directly; anything still missing there (hap.py) falls back to a nested
# container against the host Docker socket.
dock() {
  local img="$1"; shift
  if [ "$FIG6_NATIVE" = "1" ] && have_native "$1"; then
    "$@"
    return
  fi
  docker run --rm --platform "$PLATFORM" -v "$FIG6_HOST_ROOT:/work" -w /work "$img" "$@"
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
if [ "$FIG6_NATIVE" = "1" ]; then
  echo "[lib] native mode (fig6 runner image)"
fi
