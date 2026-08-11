#!/usr/bin/env bash
# Mentor task end-to-end on chr21 for one HPRC/GIAB sample:
#   Giraffe (+ optional ours) → DeepVariant
#   long reads → Sniffles
#
# Defaults: SAMPLE=HG002 CHR=chr21 (single held-out GIAB sample for inference)
# First training sample listing: SAMPLE=HG00438 ./scripts/chr21/fetch_hprc_sample.sh
#
#   ./scripts/chr21/run_all.sh
#   SAMPLE=HG002 ./scripts/chr21/run_all.sh
#   SKIP_OURS=1 SKIP_SNIFFLES=1 ./scripts/chr21/run_all.sh   # giraffe+DV only
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
D="$(dirname "${BASH_SOURCE[0]}")"
chmod +x "$D"/*.sh 2>/dev/null || true

step() { echo; echo "############################################################"; echo "# $*"; echo "############################################################"; }

step "0  preflight (tools & deps)";          "$D/preflight.sh"
step "1  reference (GRCh38 $CHR)";           "$D/fetch_reference.sh"
if [ "${SKIP_TRUTH:-0}" != "1" ]; then
  step "2  truth (GIAB $SAMPLE $CHR)";       "$D/fetch_truth.sh"
else
  echo "SKIP_TRUTH=1 — skipping GIAB truth (HPRC-only sample)"
fi
step "3  short reads ($SAMPLE $CHR FASTQ)";  "$D/fetch_reads.sh"
step "4  build $CHR Giraffe indexes";        "$D/build_giraffe.sh"
step "5  map: Giraffe";                      "$D/map_giraffe.sh"
step "6  call: DeepVariant (giraffe)";       "$D/call_deepvariant.sh" giraffe

if [ "${SKIP_TRUTH:-0}" != "1" ] && [ "${SKIP_EVAL:-0}" != "1" ]; then
  step "6b eval: hap.py (giraffe)";          "$D/eval_happy.sh" giraffe
elif [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "SKIP_EVAL=1 — skipping hap.py scoring"
fi

OURS_DONE=0
if [ "${SKIP_OURS:-0}" != "1" ]; then
  step "7  map: our implementation (GraphMambaFormer)"
  if "$D/map_ours.sh"; then
    OURS_DONE=1
    step "8  call: DeepVariant (ours)"; "$D/call_deepvariant.sh" ours
    if [ "${SKIP_TRUTH:-0}" != "1" ] && [ "${SKIP_EVAL:-0}" != "1" ]; then
      step "8b eval: hap.py (ours)"; "$D/eval_happy.sh" ours
    fi
  else
    echo "NOTE: skipping DeepVariant(ours) — 'ours' BAM not produced (see message above)."
  fi
else
  echo "SKIP_OURS=1 — skipping GraphMambaFormer arm"
fi

if [ "${SKIP_COMPARE:-0}" != "1" ]; then
  step "8c compare: Giraffe vs ours"
  "$D/compare.sh" || echo "NOTE: comparison skipped — need at least one arm's outputs."
fi

if [ "${SKIP_SNIFFLES:-0}" != "1" ]; then
  step "9  Sniffles (structural variants)"
  if "$D/call_sniffles.sh"; then
    echo "Sniffles done."
  else
    echo "NOTE: Sniffles skipped — set HIFI_FASTQ / HIFI_BAM / HIFI_BAM_URL."
  fi
else
  echo "SKIP_SNIFFLES=1 — skipping SV calling"
fi

echo
echo "Done. Outputs under: $RUN_DIR"
echo "  BAM:      $BAM_DIR"
echo "  VCF:      $VCF_DIR"
echo "  SV:       $SV_DIR"
echo "  COMPARE:  $RUN_DIR/compare  (compare.csv + charts)"
echo
echo "40-sample list:  data/hprc/graph_samples_44.txt"
echo "All links:       data/hprc/SAMPLE_LINKS.md"
echo "Portal links:    data/hprc/LINKS.md"
echo "First training:  $D/fetch_hprc_sample.sh HG00438 links"
