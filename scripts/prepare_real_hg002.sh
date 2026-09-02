#!/usr/bin/env bash
# Prepare real HG002 inputs for scripts/train.py / scripts/eval.py
# (linear FASTA + pangenome GFA + truth BAM).
#
# Produces under data/chr21/HG002/:
#   ref/GRCh38.chr21.fa          — linear reference
#   chr21.gfa                    — HPRC pangenome graph (from vg convert)
#   bam/HG002.chr21.giraffe.sorted.bam — aligned truth for supervision
#   truth/…                      — GIAB benchmark VCF/BED (for hap.py later)
#   reads/…                      — Illumina FASTQ (for inference-only eval)
#
# Requires Docker (vg / samtools images). Run from your own Terminal:
#
#   ./scripts/prepare_real_hg002.sh
#   REGION=chr21:5000000-6000000 ./scripts/prepare_real_hg002.sh   # docs only
#
# Then train / eval:
#   PYTHONPATH=. python scripts/train.py --data real \
#     --reference-fasta data/chr21/HG002/ref/GRCh38.chr21.fa \
#     --gfa data/chr21/HG002/chr21.gfa \
#     --truth-bam data/chr21/HG002/bam/HG002.chr21.giraffe.sorted.bam \
#     --region chr21:5000000-6000000 --ref-mode both --device cuda \
#     --out data/training_runs/hg002_both
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAMPLE="${SAMPLE:-HG002}"
export CHR="${CHR:-chr21}"
D="$ROOT/scripts/chr21"
chmod +x "$D"/*.sh 2>/dev/null || true

step() { echo; echo "############################################################"; echo "# $*"; echo "############################################################"; }

cd "$ROOT"
step "0  preflight";            "$D/preflight.sh" || true
step "1  GRCh38 $CHR FASTA";    "$D/fetch_reference.sh"
step "2  GIAB truth (HG002)";   "$D/fetch_truth.sh"
step "3  HG002 Illumina FASTQ"; "$D/fetch_reads.sh"
step "4  HPRC $CHR graph → GFA + Giraffe indexes"; "$D/build_giraffe.sh"
step "5  map: Giraffe → truth BAM"; "$D/map_giraffe.sh"

# Canonical GFA path used by train/eval docs (build_giraffe writes $RUN_DIR/$CHR.gfa).
source "$D/lib.sh"
if [ -f "$RUN_DIR/${CHR}.gfa" ] && [ ! -e "$RUN_DIR/chr21.gfa" ] && [ "$CHR" = "chr21" ]; then
  ln -sfn "${CHR}.gfa" "$RUN_DIR/chr21.gfa"
fi
# Also expose under the name train.py examples use when CHR != chr21.
if [ -f "$RUN_DIR/${CHR}.gfa" ]; then
  ln -sfn "${CHR}.gfa" "$RUN_DIR/${CHR}_pangenome.gfa" 2>/dev/null || true
fi

echo
echo "Ready for real-data train / eval (SAMPLE=$SAMPLE CHR=$CHR):"
echo "  FASTA : $REF_FA"
echo "  GFA   : $RUN_DIR/${CHR}.gfa"
echo "  truth : $BAM_DIR/${SAMPLE}.${CHR}.giraffe.sorted.bam"
echo "  reads : $READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz"
echo
echo "Train both linear + pangenome on a window (GPU / Docker later):"
cat <<EOF
PYTHONPATH=. python scripts/train.py --data real \\
  --reference-fasta $REF_FA \\
  --gfa             $RUN_DIR/${CHR}.gfa \\
  --truth-bam       $BAM_DIR/${SAMPLE}.${CHR}.giraffe.sorted.bam \\
  --region ${CHR}:5000000-6000000 --ref-mode both \\
  --device cuda --devices auto --epochs 20 --batch-size 8 \\
  --out data/training_runs/${SAMPLE}_${CHR}_both
EOF
