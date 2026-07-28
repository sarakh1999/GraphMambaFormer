# chr21 mentor benchmark

One HPRC/GIAB individual on **chr21**:

1. **Giraffe** (HPRC pangenome) vs **our GraphMambaFormer** alignments  
2. **DeepVariant** → small variants  
3. **Sniffles** → structural variants ([Sniffles repo](https://github.com/fritzsedlazeck/Sniffles))

## What this task means

Yes: **one HPRC individual × chr21 first**, then scale the same workflow across the ~44 graph-training samples.

| Stage | Sample choice | Why |
| --- | --- | --- |
| Pilot / eval | `HG005` (default) | GIAB truth VCF/BED for chr21 benchmarking |
| First training sample | `HG00438` | First HPRC core individual in the graph-training set |
| Scale-out | `data/hprc/graph_samples_44.txt` | ~44 individuals used to build HPRC v1.1 |

## Prerequisites

Run from your own Terminal:

- Docker Desktop running
- AWS CLI optional but recommended for HPRC S3 listing/sync (`brew install awscli`)

The Cursor agent shell cannot access the Docker socket, so execute the pipeline locally.

## Quick start

### A. GIAB eval sample (`HG005`, has truth)

```bash
# Giraffe + DeepVariant + Sniffles on chr21
SAMPLE=HG005 ./scripts/chr21/run_all.sh

# Giraffe + DeepVariant only (skip our aligner hook + Sniffles)
SKIP_OURS=1 SKIP_SNIFFLES=1 SAMPLE=HG005 ./scripts/chr21/run_all.sh
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

GraphMambaFormer does **not** emit BAM yet (Figure 1C decoder pending). `map_ours.sh` is a hook:

```bash
OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
./scripts/chr21/call_deepvariant.sh ours
```

Until the decoder lands, Giraffe is the working aligner arm.

## Pipeline steps

`run_all.sh` runs:

1. `fetch_reference.sh` — GRCh38 chr21 FASTA  
2. `fetch_truth.sh` — GIAB truth (skip with `SKIP_TRUTH=1`)  
3. `fetch_reads.sh` — stream chr21 Illumina FASTQ from BAM/CRAM  
4. `build_giraffe.sh` — download prebuilt `chr21.d9.vg` (~1 GB) and build Giraffe indexes  
5. `map_giraffe.sh` — Giraffe → sorted BAM  
6. `call_deepvariant.sh giraffe` — small variants  
6b. `eval_happy.sh giraffe` — hap.py vs GIAB truth (skip with `SKIP_EVAL=1` or `SKIP_TRUTH=1`)  
7. `map_ours.sh` — optional external BAM  
8. `call_deepvariant.sh ours` — optional  
8b. `eval_happy.sh ours` — optional hap.py scoring  
9. `call_sniffles.sh` — HiFi/ONT → minimap2 → Sniffles2 SVs  

## Outputs

`data/chr21/<SAMPLE>/`

- `bam/*.giraffe.sorted.bam` — Giraffe  
- `bam/*.ours.sorted.bam` — our aligner (when provided)  
- `vcf/*.dv.vcf.gz` — DeepVariant  
- `sv/*.sniffles.vcf.gz` — Sniffles  

## Key resources

| Resource | Link |
| --- | --- |
| HPRC pangenome indexes | https://github.com/human-pangenomics/hpp_pangenome_resources |
| chr21 prebuilt graph | https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.chroms/chr21.d9.vg |
| Sniffles | https://github.com/fritzsedlazeck/Sniffles |
| HG00438 Illumina CRAM | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/Illumina/child/HG00438.final.cram` |
| HG00438 HiFi BAM (example) | `s3://human-pangenomics/working/HPRC/HG00438/raw_data/PacBio_HiFi/m64043_200710_174426.ccs.bam` |
