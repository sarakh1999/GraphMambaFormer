# Fig 6a CPU baselines (HG002 / HG005)

Reproduce the paper’s **precision–recall** small-variant baselines on **CPU**,
scoped to **chr20** so it fits a laptop.

Paper figure (page 9) is HG005 whole-genome; we do the same pipelines on
**HG002 chr20** for a fair local comparison.

## What we run (CPU-feasible)

| Mapper | Caller | Notes |
| --- | --- | --- |
| **HPRC Giraffe** (MC pangenome) | DeepVariant | graph arm |
| **BWA-MEM** (GRCh38) | DeepVariant | linear baseline |

**Skipped:** DRAGEN (needs FPGA), DeepTrio (needs HG003/HG004 parents).

## One command

Needs Docker Desktop running. From repo root:

```bash
SAMPLE=HG002 ./scripts/fig6/run_all.sh
```

Optional:

```bash
SAMPLE=HG005 ./scripts/fig6/run_all.sh          # paper’s sample
THREADS=8 SAMPLE=HG002 ./scripts/fig6/run_all.sh
```

## Single runner image (recommended)

`docker/Dockerfile` bakes every tool above into one `linux/amd64` image on
top of `google/deepvariant:1.6.1`, so a run pulls one image instead of six and
each stage executes natively instead of starting a container:

| Tool | Version |
| --- | --- |
| DeepVariant (+ WGS/WES/PacBio/ONT models) | 1.6.1 (base image) |
| vg (static release binary) | v1.75.1 |
| BWA | 0.7.17 (Ubuntu 20.04) |
| samtools / bcftools | 1.19, built with libcurl for HTTPS BAM slicing |
| pandas / matplotlib | 2.0.3 / 3.7.5, isolated in `/opt/plotenv` |

```bash
docker/build.sh                                 # graphmambaformer:latest
SAMPLE=HG002 docker/run.sh scripts/fig6/run_all.sh
```

For a benchmark-only image without PyTorch, build the `fig6` target and point
`run.sh` at it:

```bash
TARGET=fig6 docker/build.sh                     # fig6-runner:1.6.1
IMAGE=fig6-runner:1.6.1 SAMPLE=HG002 docker/run.sh scripts/fig6/run_all.sh
```

The repo is bind-mounted at `/work`, so `data/fig6/HG002` is read and written in
place on the host — the 16 GB of BAMs, TFRecords and FASTQs stays out of the
image, and completed stages are skipped on re-runs exactly as before.

Useful entry points:

```bash
docker/run.sh gmf-doctor                        # smoke test
docker/run.sh scripts/fig6/check_prereqs.sh     # native-aware prereq check
docker/run.sh                                   # interactive shell in /work
docker/run.sh "$PLOT_PY" scripts/fig6/plot_pr.py --sample HG002
```

**hap.py is the one exception.** It is a Python-2 stack that cannot share the
DeepVariant environment, so it stays an external image. `eval_happy.sh` notices
that `/opt/hap.py/bin/hap.py` is absent and re-dispatches to
`jmcdani20/hap.py:v0.3.12`; `run.sh` mounts the Docker socket and passes
`FIG6_HOST_ROOT` so that sibling container binds the right host path. If you
prefer not to mount the socket, run stage 8 on the host:

```bash
MOUNT_DOCKER_SOCK=0 SAMPLE=HG002 docker/run.sh scripts/fig6/call_deepvariant.sh bwa
SAMPLE=HG002 ./scripts/fig6/eval_happy.sh bwa     # on the host
```

The scripts are unchanged on the host: without `FIG6_NATIVE=1` they still drive
the six per-tool images as documented above.

## Outputs

`data/fig6/HG002/`

- `bam/*.giraffe.sorted.bam` / `*.bwa.sorted.bam`
- `vcf/*.dv.vcf.gz`
- `eval/{giraffe,bwa}/*.summary.csv` + `*.roc.all.csv.gz`
- `plots/fig6a_chr20.png` — PR curves with F1

Shared (once): `data/fig6/ref/`, `data/fig6/indexes/`

## Steps (if you want to run manually)

1. `fetch_reference.sh` — GRCh38 chr20  
2. `fetch_truth.sh` — GIAB v4.2.1  
3. `fetch_reads.sh` — stream chr20 Illumina from GIAB BAM  
4. `build_chr20_giraffe.sh` — chr20 Giraffe indexes from HPRC GBZ  
5. `map_giraffe.sh` / `index_bwa.sh` + `map_bwa.sh`  
6. `call_deepvariant.sh giraffe` + `call_deepvariant.sh bwa`  
7. `eval_happy.sh giraffe` + `eval_happy.sh bwa`  
8. `plot_pr.py --sample HG002`

On Apple Silicon, Docker runs x86 images via Rosetta — expect several hours.
