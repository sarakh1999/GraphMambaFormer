# HPRC multi-modality reads

On-disk layout for the samples we train / evaluate on
(`HG00438`, `HG00621`, `HG00673`, …):

```text
data/hprc/reads/<SAMPLE>/
├── hifi/       *.fastq.gz     PacBio HiFi (unaligned)
├── illumina/   *.cram         Illumina (aligned CRAM; remapped as sequences)
└── ont/        *.bam          ONT Dorado BAM (aligned; remapped as sequences)
```

| Modality | Folder | Input types | Canonical tag |
| --- | --- | --- | --- |
| HiFi | `hifi/` | `.fastq.gz`, `.fq.gz`, `.bam` | `pacbio_hifi` |
| Illumina | `illumina/` | `.cram`, `.bam`, FASTQ | `illumina` |
| ONT | `ont/` | `.bam`, `.cram`, FASTQ | `ont` |

Aligned Illumina CRAM and ONT BAM are loaded as **sequences** (prior
coordinates stripped) and remapped by GraphMambaFormer. CRAM decode uses the
same `--ref` FASTA you align against.

## Prerequisites

```bash
# from the repo root
PYTHONPATH=. .venv/bin/python -c "import pysam, torch, graphmambaformer"
# reference FASTA (chr21 window or full GRCh38)
ls data/chr21/HG002/ref/GRCh38.chr21.fa   # or your own --ref
```

Set a shared index cache so separate modality runs do not rebuild Stage-1 indices:

```bash
export OURS_INDEX_CACHE=data/hprc/index_cache
export REF=data/chr21/HG002/ref/GRCh38.chr21.fa   # example
export SAMPLE=HG00438
```

## Run each modality separately

One BAM per modality. Reuse `--index-cache` across the three calls.

```bash
# Illumina CRAM -> BAM
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample "$SAMPLE" --modalities illumina \
  --out "data/hprc/bam/${SAMPLE}.illumina.ours.sorted.bam" \
  --index-cache "$OURS_INDEX_CACHE" --workers 16

# PacBio HiFi FASTQ.gz -> BAM
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample "$SAMPLE" --modalities hifi \
  --out "data/hprc/bam/${SAMPLE}.hifi.ours.sorted.bam" \
  --index-cache "$OURS_INDEX_CACHE" --workers 16

# ONT BAM -> BAM
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample "$SAMPLE" --modalities ont \
  --out "data/hprc/bam/${SAMPLE}.ont.ours.sorted.bam" \
  --index-cache "$OURS_INDEX_CACHE" --workers 16
```

Or pass files explicitly (dirs and globs work):

```bash
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" \
  --illumina data/hprc/reads/HG00438/illumina/HG00438.final.cram \
  --out data/hprc/bam/HG00438.illumina.ours.sorted.bam \
  --index-cache "$OURS_INDEX_CACHE"

PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" \
  --hifi 'data/hprc/reads/HG00438/hifi/*.fastq.gz' \
  --out data/hprc/bam/HG00438.hifi.ours.sorted.bam \
  --index-cache "$OURS_INDEX_CACHE"

PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" \
  --ont data/hprc/reads/HG00438/ont/ \
  --out data/hprc/bam/HG00438.ont.ours.sorted.bam \
  --index-cache "$OURS_INDEX_CACHE"
```

Optional: slice a large CRAM/BAM with `--region chr21` while iterating.

## Run all three modalities combined

One command, one BAM, distinct `@RG` / `XM` tags per modality:

```bash
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample "$SAMPLE" \
  --modalities illumina,hifi,ont \
  --bam-mode combined \
  --out "data/hprc/bam/${SAMPLE}.all.ours.sorted.bam" \
  --index-cache "$OURS_INDEX_CACHE" --workers 16
```

Same inputs, but write **three** BAMs in one command:

```bash
PYTHONPATH=. .venv/bin/python scripts/chr21/align_ours.py \
  --ref "$REF" --sample "$SAMPLE" \
  --modalities illumina,hifi,ont \
  --bam-mode separate \
  --out "data/hprc/bam/${SAMPLE}.ours.sorted.bam" \
  --index-cache "$OURS_INDEX_CACHE"
# -> ${SAMPLE}.ours.sorted.illumina.bam
# -> ${SAMPLE}.ours.sorted.pacbio_hifi.bam
# -> ${SAMPLE}.ours.sorted.ont.bam
```

Shell wrapper (same knobs):

```bash
SAMPLE=HG00438 REF="$REF" MODALITIES=illumina,hifi,ont BAM_MODE=combined \
  ./scripts/hprc/map_sample.sh

SAMPLE=HG00621 MODALITIES=hifi BAM_MODE=combined ./scripts/hprc/map_sample.sh
```

## Quick smoke (cap reads)

```bash
OURS_MAX_READS=2000 SAMPLE=HG00438 MODALITIES=illumina,hifi,ont \
  ./scripts/hprc/map_sample.sh
```

## Notes

* Samples available under `data/hprc/reads/` today: **HG00438**, **HG00621**,
  **HG00673** (same three-modality layout).
* HiFi folders often contain several `.fastq.gz` run files; all are loaded.
* Illumina `.final.cram` needs `--ref` (or `REF=`) so pysam can decode it.
* Companion SAM is opt-in (`--sam` / `OURS_SAM=1`).
* See also `SAMPLE_LINKS.md` for Year-1 download URLs and
  `scripts/chr21/README.md` for the chr21 mentor benchmark.
