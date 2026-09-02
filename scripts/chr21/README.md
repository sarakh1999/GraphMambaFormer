# chr21 mentor benchmark

One HPRC/GIAB individual on **chr21**:

1. **Giraffe** (HPRC pangenome) vs **our GraphMambaFormer** alignments  
2. **DeepVariant** → small variants  
3. **Sniffles** → structural variants ([Sniffles repo](https://github.com/fritzsedlazeck/Sniffles))  
4. **longcallD** → joint small+SV calling **and alignment polishing** ([longcallD repo](https://github.com/yangao07/longcallD))

## What this task means

**One GIAB normal sample × chr21** for validation: **HG002** (held out of the
HPRC pangenome training set). Training-data fetch across HPRC individuals is
separate (`fetch_hprc_sample.sh` / `graph_samples_44.txt`).

| Stage | Sample choice | Why |
| --- | --- | --- |
| Validation / eval (default) | `HG002` | Sole GIAB normal truth set for this project |
| First training sample | `HG00438` | First HPRC core individual in the graph-training set (`SKIP_TRUTH=1`) |
| Scale-out | `data/hprc/graph_samples_44.txt` | ~44 individuals used to build HPRC v1.1 |

## Prerequisites

Run from your own Terminal:

- Docker Desktop running — for the Giraffe, DeepVariant, Sniffles, and hap.py arms
- AWS CLI — for HPRC S3 read/CRAM download (`brew install awscli`); GIAB https reads work without it
- Python env — for the `ours` arm and `compare.sh` (`python -m venv .venv && .venv/bin/pip install -r requirements.txt`)

The Cursor agent shell cannot access the Docker socket, so execute the
Docker-based steps locally. The `ours` arm (`map_ours.sh`) and `compare.sh` are
pure Python and need no Docker. `preflight.sh` reports exactly what is available.

## Quick start

### A. GIAB validation sample (`HG002`, default)

```bash
# Giraffe + DeepVariant + Sniffles on chr21
./scripts/chr21/run_all.sh
# equivalent: SAMPLE=HG002 ./scripts/chr21/run_all.sh

# Giraffe + DeepVariant only (skip our aligner hook + Sniffles)
SKIP_OURS=1 SKIP_SNIFFLES=1 ./scripts/chr21/run_all.sh
```

### B. First training sample (`HG00438`, HPRC-only)

```bash
# Print resolved Illumina + HiFi URLs
./scripts/chr21/fetch_hprc_sample.sh HG00438 links

# Full chr21 pipeline (no GIAB truth yet)
SAMPLE=HG00438 SKIP_TRUTH=1 ./scripts/chr21/run_all.sh
```

## First training sample + ~40 list

| Item | Path / command |
| --- | --- |
| ~44 graph samples | `data/hprc/graph_samples_44.txt` |
| All Year-1 individuals | `data/hprc/year1_samples.txt` |
| All resolved read links | `data/hprc/SAMPLE_LINKS.md` |
| Machine-readable manifest | `data/hprc/sample_links.json` |
| Portal / index links | `data/hprc/LINKS.md` |
| Regenerate manifests | `./scripts/chr21/generate_sample_links.sh` |
| List first training sample | `./scripts/chr21/fetch_hprc_sample.sh HG00438 links` |

Portal: https://humanpangenome.org/data/  
Raw S3: https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=working/

## Our implementation arm

`map_ours.sh` now **runs the GraphMambaFormer alignment pipeline**
(`graphmambaformer.alignment`, seed → chain → extend → score) on the fetched
chr21 reads and writes a sorted+indexed BAM whose `@SQ` name is `chr21`, so
DeepVariant/hap.py accept it exactly like the Giraffe BAM. It runs entirely in
Python via `pysam` — **no Docker required** for this arm.

Short and long reads can be aligned in **one command**. When both are present,
the default is a **single combined BAM** with per-modality `@RG` / `XM` tags
(Illumina vs PacBio HiFi / ONT). Use `OURS_BAM_MODE=separate` if a caller needs
homogeneous BAMs.

```bash
# run our aligner (uses the repo .venv if present)
./scripts/chr21/map_ours.sh

# quick partial pass while iterating
OURS_MAX_READS=50000 OURS_MODE=fast ./scripts/chr21/map_ours.sh

# short + long in one command -> one combined BAM
OURS_LONG_READS=/path/to/hifi.fastq.gz ./scripts/chr21/map_ours.sh

# same, but write separate BAMs per modality
OURS_LONG_READS=/path/to/hifi.fastq.gz OURS_BAM_MODE=separate ./scripts/chr21/map_ours.sh

# faster: reuse Stage-1 index across separate R1 / R2 / long runs
OURS_INDEX_CACHE=data/chr21/HG002/index_cache OURS_WORKERS=16 \
  OURS_BATCH_SIZE=128 ./scripts/chr21/map_ours.sh

# or plug in an externally produced BAM
OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
```

Direct Python equivalent:

```bash
python scripts/chr21/align_ours.py \
  --ref data/chr21/HG002/ref/GRCh38.chr21.fa \
  --reads data/chr21/HG002/reads/HG002.chr21.R1.fastq.gz \
  --reads data/chr21/HG002/reads/HG002.chr21.R2.fastq.gz \
  --long-reads /path/to/hifi.fastq.gz \
  --out data/chr21/HG002/bam/HG002.chr21.ours.sorted.bam \
  --bam-mode combined \
  --index-cache data/chr21/HG002/index_cache \
  --workers 16 --batch-size 128
```

Knobs: `OURS_MODE=fast|hybrid|two_pass` (default `fast`, fully classical, needs
no trained model), `OURS_MAX_READS=N` (cap reads for speed), `OURS_FORCE=1`
(rebuild), `OURS_LONG_READS` / `HIFI_FASTQ` (optional long reads),
`OURS_LONG_MODALITY=pacbio_hifi|ont`, `OURS_BAM_MODE=combined|separate|auto`,
`OURS_INDEX_CACHE=DIR` (reuse Stage-1 index across separate-file runs),
`OURS_WORKERS` / `GMF_NUM_WORKERS`, `OURS_BATCH_SIZE`, `OURS_DEVICE`,
`OURS_SAM=1` (also write companion SAM; off by default), `PYTHON=...`.

> Companion SAM is opt-in now (`--sam` / `OURS_SAM=1`) so default runs stay
> I/O-light. Combined short+long builds the index once; separate invocations
> should set `OURS_INDEX_CACHE` so they do not rebuild it each time.

### Compare the two arms

```bash
./scripts/chr21/compare.sh           # giraffe vs ours
```

Reads the BAMs, DeepVariant VCFs, and hap.py summaries already produced and
writes `data/chr21/<SAMPLE>/compare/compare.csv` plus bar charts (mapping rate /
MAPQ, PASS SNP/INDEL counts, and hap.py F1 when a GIAB truth set exists).

## Pipeline steps

`run_all.sh` runs:

0. `preflight.sh` — report which tools/deps are available and what each blocks  
1. `fetch_reference.sh` — GRCh38 chr21 FASTA  
2. `fetch_truth.sh` — GIAB truth (skip with `SKIP_TRUTH=1`)  
3. `fetch_reads.sh` — stream chr21 Illumina FASTQ from BAM/CRAM  
4. `build_giraffe.sh` — download prebuilt `chr21.d9.vg` (~1 GB) and build Giraffe indexes  
5. `map_giraffe.sh` — Giraffe → sorted BAM  
6. `call_deepvariant.sh giraffe` — small variants  
6b. `eval_happy.sh giraffe` — hap.py vs GIAB truth (skip with `SKIP_EVAL=1` or `SKIP_TRUTH=1`)  
7. `map_ours.sh` — GraphMambaFormer aligner → sorted BAM (Python; skip with `SKIP_OURS=1`)  
8. `call_deepvariant.sh ours` — small variants for our arm  
8b. `eval_happy.sh ours` — hap.py scoring for our arm  
8c. `compare.sh` — Giraffe vs ours table + charts (skip with `SKIP_COMPARE=1`)  
9. `call_sniffles.sh` — HiFi/ONT → minimap2 → Sniffles2 SVs  
10. `call_longcalld.sh` — longcallD: phased VCF + **refined (polished) BAM** (skip with `SKIP_LONGCALLD=1`)  

## Alignment polishing with longcallD

A mapper places indels greedily per read, so one true indel in a homopolymer or
tandem repeat ends up at slightly different coordinates in every read that spans
it. longcallD phases the reads, builds a haplotype-aware MSA consensus per locus,
and re-aligns each phased read against it (`--refine-aln`), which pulls those
scattered indels onto one breakpoint. Unphased reads pass through untouched.

Refinement is on by default in `call_longcalld.sh`:

```bash
# polish + call from an existing long-read BAM
HIFI_BAM=/path/to/sorted.bam ./scripts/chr21/call_longcalld.sh

# calling only, no realignment
REFINE_ALN=0 ./scripts/chr21/call_longcalld.sh

# bundled 2 Mb chr11 test data (HiFi + ONT), no downloads
LONGCALLD_SMOKE=1 ./scripts/chr21/call_longcalld.sh
```

`refine_report.py` diffs the original BAM against the refined one so the effect
is measurable rather than assumed. On the bundled HG002 chr11 test data:

| | HiFi | ONT |
| --- | --- | --- |
| phased reads with indel structure rewritten | 143 / 356 | 300 / 367 |
| distinct indel breakpoints (all sizes) | 8724 → 8608 | 53988 → 53641 |
| distinct **≥30 bp** indel breakpoints | **178 → 67** | **418 → 177** |
| unphased control reads changed | 0 / 6 | 0 / 215 |

Large indels are where mapper disagreement is worst, and that is where
refinement consolidates most (−62% HiFi, −58% ONT distinct breakpoints). Edit
distance (`NM`) can rise slightly on individual reads — refinement optimises
haplotype consistency, not per-read edit distance — so the report prints `NM`
neutrally instead of as a better/worse score.

Because the refined BAM is a cleaner indel representation of the same reads, it
is also usable as improved supervision for our aligner's indel heads.

## Outputs

`data/chr21/<SAMPLE>/`

- `bam/*.giraffe.sorted.bam` — Giraffe  
- `bam/*.ours.sorted.bam` — our aligner  
- `vcf/*.dv.vcf.gz` — DeepVariant  
- `sv/*.sniffles.vcf.gz` — Sniffles  
- `sv/*.longcalld.vcf` — longcallD phased small variants + SVs  
- `sv/*.longcalld.refined.sorted.bam` — polished, HP/PS-tagged alignments  
- `sv/*.longcalld.refine_report.json` — what refinement changed vs the input BAM  
- `compare/compare.csv` + `compare/*.png` — Giraffe vs ours comparison  

## Key resources

| Resource | Link |
| --- | --- |
| HPRC pangenome indexes | https://github.com/human-pangenomics/hpp_pangenome_resources |
| chr21 prebuilt graph | https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.chroms/chr21.d9.vg |
| Sniffles | https://github.com/fritzsedlazeck/Sniffles |
| longcallD | https://github.com/yangao07/longcallD |
| HG00438 Illumina CRAM | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/Illumina/child/HG00438.final.cram` |
| HG00438 HiFi BAM (example) | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/PacBio_HiFi/m64043_200710_174426.ccs.bam` |
