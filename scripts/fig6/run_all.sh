#!/usr/bin/env bash
# End-to-end single-chromosome CPU baselines for Nature HPRC Fig 6a style PR curves.
# Chromosome comes from $CHR (default chr1); CHR=chr20 ./run_all.sh reproduces the
# earlier chr20 run.
#
# Arms run here (CPU-feasible):
#   1) HPRC Giraffe + DeepVariant
#   2) BWA-MEM + DeepVariant
#
# Skipped: DRAGEN (FPGA), DeepTrio (needs parents — add later).
#
#   SAMPLE=HG002 ./scripts/fig6/run_all.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
D="$(dirname "${BASH_SOURCE[0]}")"

step() { echo; echo "############################################################"; echo "# $*"; echo "############################################################"; }
run_or_die() {
  "$@" || {
    echo "ERROR: stage failed: $*" >&2
    exit 1
  }
}

step "1/9  reference (GRCh38 $CHR)";         run_or_die "$D/fetch_reference.sh"
step "2/9  truth (GIAB $SAMPLE $CHR)";       run_or_die "$D/fetch_truth.sh"
step "3/9  reads ($SAMPLE $CHR FASTQ)";      run_or_die "$D/fetch_reads.sh"
step "4/9  build $CHR Giraffe indexes";      run_or_die "$D/build_giraffe.sh"
step "5/9  map: Giraffe (graph)";            run_or_die "$D/map_giraffe.sh"
step "6/9  map: BWA-MEM (linear)";           run_or_die "$D/index_bwa.sh"; run_or_die "$D/map_bwa.sh"
step "7/9  call: DeepVariant x2";            run_or_die "$D/call_deepvariant.sh" giraffe; run_or_die "$D/call_deepvariant.sh" bwa
step "8/9  eval: hap.py x2";                 run_or_die "$D/eval_happy.sh" giraffe; run_or_die "$D/eval_happy.sh" bwa
step "9/9  plot PR curves";                  run_or_die "$PLOT_PY" "$D/plot_pr.py" --sample "$SAMPLE" --type SNP

echo; echo "All done -> $PLOT_DIR/fig6a_${CHR}.png"
echo "Tip: if DeepVariant dies mid-run, intermediates live under data/fig6/\$SAMPLE/dv/ and will be reused."
