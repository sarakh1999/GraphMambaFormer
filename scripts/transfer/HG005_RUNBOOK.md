# HG005 chr21 pangenome-windows run — OSC runbook

HG005 (GIAB **ChineseTrio son**, NA24631) is **held out of the HPRC graph**, so it
is a valid independent sample for the chr21 pangenome-window read-aligner. This
runbook reproduces the HG002 pipeline for HG005 with three parameterized scripts
and two sbatch files. **It does not change any HG002 default** and **does not
touch the currently-running HG002 training, its caches, manifests or output
dirs.**

Key design point — **HG005 reuses HG002's reference indexes.** The reference and
graph (FASTA, whole-chr GFA, and the 610 per-window GFAs) are **shared**: only
the reads/BAMs and the manifest names change. `slice_pangenome_windows.py`
defaults `--graph-sample HG002`, so HG005's manifest points at the *same*
`data/chr21/HG002/ref/chr21.fa` and `data/chr21/HG002/pangenome_windows/*.gfa`
paths. The `ref_index` sub-cache is keyed only by `{fasta, gfa, region, contig,
label, kmer}` (read-independent), so the expensive per-window FM-index / graph
build is loaded from cache, not rebuilt. The **dataset** cache blob is a new key
(it includes the manifest SHA1 + batch-size + max-read-len), so HG005 gets its
own `chr21_HG005_pangenome_windows.bal.<digest>.pt` while every HG002 cache is
untouched.

Prereqs / caveats:
- Steps 0–2b need internet + local writes → run on an **OSC login node**
  (compute nodes usually have no internet). Steps 3–4 are Slurm jobs.
- `pysam` streams only the `chr21` slice of each remote BAM. This needs the
  BAM's **remote `.bai` index** to sit next to the BAM on the server (all the
  GIAB/UCSC URLs below have one). If a BAM has no server-side index, download it
  first, index locally, then pass a local path via `--illumina-url/--hifi-url/
  --ont-url`.
- The 300x HG005 Illumina is **auto-downsampled ~0.1** in the fetch script so the
  streamed depth lands near the ~30x HPRC Fig-6 regime.
- **ONT is optional for a first run.** HG005 has no single verified GIAB GRCh38
  ultralong BAM; if you cannot resolve the UCSC-panel URL in step 0, skip it and
  train Illumina+HiFi-only (the fetch step skips a missing ONT gracefully and
  slice simply emits no ONT entries).

---

## 0. Resolve the ONT URL (optional)

There is no invented ONT filename in the repo. List the UCSC nanopore panel and
pick the GRCh38 (`GRCh38_no_alt`, minimap2, ~57x) BAM:

```bash
aws s3 ls --no-sign-request \
  s3://human-pangenomics/NHGRI_UCSC_panel/HG005/nanopore/Guppy_4.2.2/

# then export the full https/s3 path to the GRCh38 bam you found, e.g.:
export ONT_URL="https://human-pangenomics.s3.amazonaws.com/NHGRI_UCSC_panel/HG005/nanopore/Guppy_4.2.2/<the-GRCh38-bam-you-found>.bam"
```

If you cannot find one, leave `ONT_URL` unset and skip `--ont-url` below.

---

## 1. Fetch real HG005 chr21 truth BAMs (login node — has internet)

```bash
cd /users/PCS0289/sarakhosravi/mambaformer

# With ONT:
.venv/bin/python scripts/chr21/fetch_giab_real.py --sample HG005 --ont-url "$ONT_URL"

# Illumina + HiFi only (skip ONT):
.venv/bin/python scripts/chr21/fetch_giab_real.py --sample HG005
```

- Illumina and HiFi need **no** URL override (verified GIAB GRCh38 URLs are built
  in). Only ONT is supplied at runtime.
- Streams only `chr21`, rewrites to single-contig `chr21` truth BAMs, and
  auto-downsamples the 300x Illumina (~0.1).
- Resumable: a modality whose sorted BAM + `.bai` already exist is skipped
  (`--force` to redo).
- Writes to `data/chr21/HG005/bam/HG005.chr21.{illumina,pacbio_hifi,ont}.real.sorted.bam`
  (+ long-read FASTQs under `data/chr21/HG005/reads/`).

## 2. Slice → HG005 manifests (login/compute node)

```bash
.venv/bin/python scripts/chr21/slice_pangenome_windows.py \
  --sample HG005 --holdout-start 40000000 --holdout-end 46699983 \
  --illumina-depth 30
```

- Reuses HG002's shared graph (`--graph-sample` defaults to `HG002`); the
  per-window GFAs are **reused as-is, never rewritten** (auto-enabled because the
  reads sample differs from the graph sample), so HG002's window dir and its
  `ref_index` cache stay valid.
- Writes `data/hprc/manifests/chr21_HG005_pangenome_windows.json` and
  `...windows.test.json` (holdout region `chr21:40,000,001-46,699,983`).

## 2b. Produce the BALANCED manifests used by training

The training sbatch consumes the **balanced** manifest. HG002's `.bal.json` is
exactly the slice output at `--illumina-depth 0.3` (verified: it equals the plain
manifest with each Illumina `max_reads` recomputed as `ceil(0.3*bp/250)`; the
long-read entries are unchanged). Reproduce it for HG005 the same way:

```bash
.venv/bin/python scripts/chr21/slice_pangenome_windows.py \
  --sample HG005 --holdout-start 40000000 --holdout-end 46699983 \
  --illumina-depth 0.3 \
  --manifest      data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json \
  --test-manifest data/hprc/manifests/chr21_HG005_pangenome_windows.bal.test.json
```

(Same window/holdout logic; only the Illumina depth target changes. The graph is
still reused, nothing under `data/chr21/HG002/` is written.)

## 3. Build the dataset cache (CPU node — reuses ref_index)

```bash
sbatch scripts/transfer/build_cache_hg005.sbatch
```

- CPU-only `largemem` job. Writes `chr21_HG005_pangenome_windows.bal.<digest>.pt`
  into `/fs/ess/PCS0289/mambaformer_cache` (a NEW key; HG002 caches untouched).
- **Watch for reuse:** the per-window `[ref pangenome] ...` lines should come
  from the cache almost instantly. If you instead see a long (~1h) per-window
  index rebuild, the manifest is not pointing at HG002's shared FASTA/GFA paths
  (check that `--graph-sample` stayed `HG002` and the paths in the manifest are
  `data/chr21/HG002/...`).

## 4. Train (A100 GPU)

```bash
sbatch scripts/transfer/train_hg005.sbatch
```

- Loads the cache from step 3 (same `--batch-size 4` + `--max-read-len 65536`
  key). No `--compile`. `expandable_segments:True` set.
- 12 epochs, intra-epoch validate/plot/save, `--out
  /fs/ess/PCS0289/mambaformer_runs/chr21_hg005_bal`.
- **Watch for:** no OOM at B=4 with the 64k read cap; steady GPU memory. If the
  job starts rebuilding the dataset cache on the GPU node, step 3's cache is
  missing or its `--batch-size`/`--max-read-len` differ.

---

## Expected footprints (rough)

- HG005 chr21 truth BAMs: Illumina (post-0.1 downsample) ~ a few hundred MB;
  HiFi ~ hundreds of MB; ONT (if fetched) ~ hundreds of MB.
- Balanced manifest sizes comparable to HG002 (`chr21_HG005_pangenome_windows.bal.json`
  a few hundred KB).
- Dataset cache blob: same order as the HG002 B=4 blob (tens of GB); it is a new
  file, so ensure free space in `/fs/ess/PCS0289/mambaformer_cache`.
- `ref_index/` sub-cache: **reused**, no new large writes.

## ONT-skip note

If step 0 yields no usable GRCh38 ONT BAM, run steps 1–4 without `--ont-url`. The
manifests will contain only Illumina + HiFi entries and training proceeds
normally; add ONT later by re-running steps 1 (`--only ont --ont-url ...`), 2, 2b
and rebuilding the cache.
