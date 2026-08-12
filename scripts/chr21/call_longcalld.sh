#!/usr/bin/env bash
# Polish an aligner's long-read alignments (and call variants) with longcallD.
#
# longcallD phases reads, builds a haplotype-aware MSA consensus per locus, and
# re-aligns each phased read against it (--refine-aln). That rewrites indel
# placement the mapper got wrong — mostly homopolymers and tandem repeats, where
# minimap2/BWA scatter or mis-position I/D ops. Output is both a phased VCF
# (SNPs, small indels, INS/DEL SVs) and a refined BAM with HP/PS tags.
#
# Refinement is on by default here since polishing indels is the point; set
# REFINE_ALN=0 for plain calling, PHASED_BAM=0 to skip the BAM entirely.
#
# https://github.com/yangao07/longcallD
#
# Usage:
#   ./scripts/chr21/call_longcalld.sh
#   HIFI_BAM=/path/a.bam ./scripts/chr21/call_longcalld.sh
#   HIFI_BAM=/path/a.bam HIFI_BAM_EXTRA="/path/b.bam /path/c.bam" ./scripts/chr21/call_longcalld.sh
#   PLATFORM_LR=ont ./scripts/chr21/call_longcalld.sh          # default: hifi
#   REFINE_ALN=0 ./scripts/chr21/call_longcalld.sh             # calling only
#   LONGCALLD_SMOKE=1 ./scripts/chr21/call_longcalld.sh        # bundled test data
#
# Outputs (under data/chr21/$SAMPLE/sv/):
#   ${SAMPLE}.${CHR}.longcalld.vcf
#   ${SAMPLE}.${CHR}.longcalld.refined.sorted.bam(.bai)   # realigned, HP/PS tagged
#   ${SAMPLE}.${CHR}.longcalld.refine_report.json         # what refinement changed
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RESOLVE="$(dirname "${BASH_SOURCE[0]}")/resolve_sample.py"

LONGCALLD_BIN="${LONGCALLD_BIN:-$ROOT/.tools/longcallD-v0.0.11_arm64-macos/longcallD}"
PLATFORM_LR="${PLATFORM_LR:-hifi}"   # hifi | ont
REFINE_ALN="${REFINE_ALN:-1}"
# Refinement is delivered through the output BAM, so it implies PHASED_BAM.
PHASED_BAM="${PHASED_BAM:-1}"
[ "$REFINE_ALN" = "1" ] && PHASED_BAM=1
THREADS="${THREADS:-4}"
# pysam stands in for samtools, which is not installed on the host.
PY="${PY:-$ROOT/.venv/bin/python}"
SORT_BAM="$(dirname "${BASH_SOURCE[0]}")/sort_index_bam.py"
REFINE_REPORT="$(dirname "${BASH_SOURCE[0]}")/refine_report.py"

OUT_VCF="$SV_DIR/${SAMPLE}.${CHR}.longcalld.vcf"
OUT_BAM="$SV_DIR/${SAMPLE}.${CHR}.longcalld.phased.bam"
OUT_BAM_SORTED="$SV_DIR/${SAMPLE}.${CHR}.longcalld.refined.sorted.bam"
OUT_REPORT="$SV_DIR/${SAMPLE}.${CHR}.longcalld.refine_report.json"
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
  # Each platform: call + refine the aligner's alignments, then sort/index and
  # diff the refined CIGARs against the input BAM.
  for plat in hifi ont; do
    IN_BAM="$TD/HG002_chr11_${plat}_test.bam"
    [ -f "$IN_BAM" ] || continue
    RAW="$SMOKE_OUT/HG002_chr11_${plat}.refined.bam"
    SORTED="$SMOKE_OUT/HG002_chr11_${plat}.refined.sorted.bam"
    echo
    echo "longcallD --refine-aln (${plat}) -> $SORTED"
    "$LONGCALLD_BIN" call -t "$THREADS" "--${plat}" --refine-aln \
      -b "$RAW" "$TD/chr11_2M.fa" "$IN_BAM" \
      > "$SMOKE_OUT/HG002_chr11_${plat}_test.vcf"
    "$PY" "$SORT_BAM" "$RAW" "$SORTED" "$THREADS"
    echo "--- refinement report (${plat}) ---"
    "$PY" "$REFINE_REPORT" --original "$IN_BAM" --refined "$SORTED" \
      --json "$SMOKE_OUT/HG002_chr11_${plat}.refine_report.json"
  done

  # Exercise multi-BAM API (--input-is-list). Same sample, same platform.
  printf '%s\n' "$TD/HG002_chr11_hifi_test.bam" > "$SMOKE_OUT/hifi_bam_list.txt"
  echo
  echo "longcallD multi-BAM API (--input-is-list)"
  "$LONGCALLD_BIN" call -t "$THREADS" --hifi --input-is-list \
    "$TD/chr11_2M.fa" "$SMOKE_OUT/hifi_bam_list.txt" \
    > "$SMOKE_OUT/HG002_chr11_hifi_from_list.vcf"
  echo
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
if [ "$REFINE_ALN" = "1" ]; then
  CMD+=(--refine-aln)
fi

echo "longcallD ${PLATFORM_LR} -> $OUT_VCF"
echo "  cmd: ${CMD[*]}"
"${CMD[@]}" > "$OUT_VCF"

if [ "$PHASED_BAM" = "1" ] && [ -s "$OUT_BAM" ]; then
  # --refine-aln output is unsorted; sort/index before anything reads regions.
  "$PY" "$SORT_BAM" "$OUT_BAM" "$OUT_BAM_SORTED" "$THREADS"
  rm -f "$OUT_BAM"
  if [ "$REFINE_ALN" = "1" ]; then
    echo "--- refinement report (indel polish vs input aligner) ---"
    "$PY" "$REFINE_REPORT" --original "$PRIMARY_BAM" --refined "$OUT_BAM_SORTED" \
      --json "$OUT_REPORT" || echo "NOTE: refine report failed (non-fatal)."
  fi
fi

echo "Done: $OUT_VCF"
ls -lh "$OUT_VCF" "$OUT_BAM_SORTED" "$OUT_REPORT" 2>/dev/null || true
wc -l "$OUT_VCF"
