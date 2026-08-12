#!/usr/bin/env bash
# Call small + structural variants from long-read BAM(s) with longcallD.
#
# Joint calling/phasing of SNPs, small indels, and INS/DEL SVs from PacBio
# HiFi or ONT BAMs. Multiple BAMs of the *same sample* can be passed together
# (no merge step) via -X / --input-is-list.
#
# https://github.com/yangao07/longcallD
#
# Usage:
#   ./scripts/chr21/call_longcalld.sh
#   HIFI_BAM=/path/a.bam ./scripts/chr21/call_longcalld.sh
#   HIFI_BAM=/path/a.bam HIFI_BAM_EXTRA="/path/b.bam /path/c.bam" ./scripts/chr21/call_longcalld.sh
#   PLATFORM_LR=ont ./scripts/chr21/call_longcalld.sh          # default: hifi
#   LONGCALLD_SMOKE=1 ./scripts/chr21/call_longcalld.sh        # bundled test data
#
# Outputs (under data/chr21/$SAMPLE/sv/):
#   ${SAMPLE}.${CHR}.longcalld.vcf
#   optional: ${SAMPLE}.${CHR}.longcalld.phased.bam  (set PHASED_BAM=1)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RESOLVE="$(dirname "${BASH_SOURCE[0]}")/resolve_sample.py"

LONGCALLD_BIN="${LONGCALLD_BIN:-$ROOT/.tools/longcallD-v0.0.11_arm64-macos/longcallD}"
PLATFORM_LR="${PLATFORM_LR:-hifi}"   # hifi | ont
PHASED_BAM="${PHASED_BAM:-0}"
THREADS="${THREADS:-4}"

OUT_VCF="$SV_DIR/${SAMPLE}.${CHR}.longcalld.vcf"
OUT_BAM="$SV_DIR/${SAMPLE}.${CHR}.longcalld.phased.bam"
HIFI_BAM_LOCAL="$BAM_DIR/${SAMPLE}.${CHR}.hifi.minimap.sorted.bam"

mkdir -p "$SV_DIR"

# ---- smoke mode: bundled longcallD test BAMs (no project download) --------
if [ "${LONGCALLD_SMOKE:-0}" = "1" ]; then
  TD="$ROOT/.tools/longcallD-v0.0.11_arm64-macos/test_data"
  require "$LONGCALLD_BIN" "download longcallD into .tools/ (see script header)"
  require "$TD/chr11_2M.fa"
  require "$TD/HG002_chr11_hifi_test.bam"
  SMOKE_OUT="$ROOT/data/chr21/longcalld_smoke"
  mkdir -p "$SMOKE_OUT"
  echo "longcallD smoke (bundled HiFi test BAM) -> $SMOKE_OUT"
  "$LONGCALLD_BIN" call -t "$THREADS" --hifi \
    "$TD/chr11_2M.fa" "$TD/HG002_chr11_hifi_test.bam" \
    > "$SMOKE_OUT/HG002_chr11_hifi_test.vcf"
  if [ -f "$TD/HG002_chr11_ont_test.bam" ]; then
    echo "longcallD smoke (bundled ONT test BAM)"
    "$LONGCALLD_BIN" call -t "$THREADS" --ont \
      "$TD/chr11_2M.fa" "$TD/HG002_chr11_ont_test.bam" \
      > "$SMOKE_OUT/HG002_chr11_ont_test.vcf"
  fi
  # Exercise multi-BAM API (--input-is-list). Same sample, same platform.
  printf '%s\n' "$TD/HG002_chr11_hifi_test.bam" > "$SMOKE_OUT/hifi_bam_list.txt"
  echo "longcallD multi-BAM API (--input-is-list)"
  "$LONGCALLD_BIN" call -t "$THREADS" --hifi --input-is-list \
    "$TD/chr11_2M.fa" "$SMOKE_OUT/hifi_bam_list.txt" \
    > "$SMOKE_OUT/HG002_chr11_hifi_from_list.vcf"
  echo "Done smoke:"
  ls -lh "$SMOKE_OUT"
  wc -l "$SMOKE_OUT"/*.vcf
  exit 0
fi

require "$LONGCALLD_BIN" "install longcallD under .tools/ or set LONGCALLD_BIN"
require "$REF_FA" "run fetch_reference.sh"

# Resolve primary BAM the same way Sniffles does.
if [ -z "${HIFI_BAM_URL:-}" ] && [ -z "${HIFI_BAM:-}" ] && [ -z "${HIFI_FASTQ:-}" ]; then
  HIFI_BAM_URL="$(python3 "$RESOLVE" "$SAMPLE" hifi_bam_url 2>/dev/null || true)"
fi

PRIMARY_BAM=""
if [ -n "${HIFI_BAM:-}" ]; then
  require "$HIFI_BAM"
  PRIMARY_BAM="$HIFI_BAM"
elif [ -f "$HIFI_BAM_LOCAL" ]; then
  PRIMARY_BAM="$HIFI_BAM_LOCAL"
  echo "Using existing $PRIMARY_BAM"
elif [ -n "${HIFI_BAM_URL:-}" ] || [ -n "${HIFI_FASTQ:-}" ]; then
  echo "No local long-read BAM yet — run call_sniffles.sh first (builds $HIFI_BAM_LOCAL),"
  echo "or set HIFI_BAM=/path/to/sorted.bam"
  exit 1
else
  cat >&2 <<EOF
ERROR: no long-read BAM for longcallD.

Provide one of:
  HIFI_BAM=/path/to/sorted.bam
  HIFI_BAM_EXTRA="/path/b.bam /path/c.bam"   # extra same-sample BAMs (-X)
  LONGCALLD_SMOKE=1                           # bundled test data

Or run ./scripts/chr21/call_sniffles.sh first to produce:
  $HIFI_BAM_LOCAL
EOF
  exit 1
fi

PLAT_FLAG="--hifi"
[ "$PLATFORM_LR" = "ont" ] && PLAT_FLAG="--ont"

CMD=("$LONGCALLD_BIN" call -t "$THREADS" "$PLAT_FLAG" "$REF_FA" "$PRIMARY_BAM")

# Extra same-sample BAMs: space-separated paths in HIFI_BAM_EXTRA, or a list file.
if [ -n "${HIFI_BAM_LIST:-}" ]; then
  require "$HIFI_BAM_LIST"
  CMD=("$LONGCALLD_BIN" call -t "$THREADS" "$PLAT_FLAG" --input-is-list "$REF_FA" "$HIFI_BAM_LIST")
elif [ -n "${HIFI_BAM_EXTRA:-}" ]; then
  for extra in $HIFI_BAM_EXTRA; do
    require "$extra"
    CMD+=(-X "$extra")
  done
fi

if [ "$PHASED_BAM" = "1" ]; then
  CMD+=(-b "$OUT_BAM")
fi

echo "longcallD ${PLATFORM_LR} -> $OUT_VCF"
echo "  cmd: ${CMD[*]}"
"${CMD[@]}" > "$OUT_VCF"

echo "Done: $OUT_VCF"
ls -lh "$OUT_VCF" ${PHASED_BAM:+"$OUT_BAM"} 2>/dev/null || true
wc -l "$OUT_VCF"
