# chr21 mentor benchmark

One HPRC/GIAB individual on **chr21**:

1. **Giraffe** (HPRC pangenome) vs **our GraphMambaFormer** alignments  
2. **DeepVariant** → small variants  
3. **Sniffles** → structural variants ([Sniffles repo](https://github.com/fritzsedlazeck/Sniffles))

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

```bash
# run our aligner (uses the repo .venv if present)
./scripts/chr21/map_ours.sh

# quick partial pass while iterating
OURS_MAX_READS=50000 OURS_MODE=fast ./scripts/chr21/map_ours.sh

# or plug in an externally produced BAM
OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
```

Knobs: `OURS_MODE=fast|hybrid|two_pass` (default `fast`, fully classical, needs
no trained model), `OURS_MAX_READS=N` (cap reads for speed), `OURS_FORCE=1`
(rebuild), `PYTHON=...` (interpreter).

> The pipeline is a correct pure-Python reference implementation, not throughput
> optimized. `hybrid`/`two_pass` only add neural re-ranking when a model with
> alignment heads is supplied; without one they degrade to the classical path.

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

## Outputs

`data/chr21/<SAMPLE>/`

- `bam/*.giraffe.sorted.bam` — Giraffe  
- `bam/*.ours.sorted.bam` — our aligner  
- `vcf/*.dv.vcf.gz` — DeepVariant  
- `sv/*.sniffles.vcf.gz` — Sniffles  
- `compare/compare.csv` + `compare/*.png` — Giraffe vs ours comparison  

## Key resources

| Resource | Link |
| --- | --- |
| HPRC pangenome indexes | https://github.com/human-pangenomics/hpp_pangenome_resources |
| chr21 prebuilt graph | https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.chroms/chr21.d9.vg |
| Sniffles | https://github.com/fritzsedlazeck/Sniffles |
| HG00438 Illumina CRAM | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/Illumina/child/HG00438.final.cram` |
| HG00438 HiFi BAM (example) | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/PacBio_HiFi/m64043_200710_174426.ccs.bam` |
