"""GPU verification + benchmark for the GenomeWorks-accelerated pipeline.

This is *not* a unit test — it is a hand-run diagnostic for a GPU node. It

  1. prints what the host can accelerate and which GenomeWorks tier is live,
  2. checks the CUDA GenomeWorks primitives against the portable NumPy tier
     (bit-exact where the pipeline scores in integers), and
  3. benchmarks the CUDA tier against the portable tier so you get real
     speedup numbers for cudaextender / cudamapper (+ the vectorised
     cudaaligner / cudapoa host primitives).

Run on a machine with a GPU:

    cd mambaformer
    PYTHONPATH=. .venv/bin/python tests/gpu_genomeworks_check.py

Exit code is non-zero only if a GPU tier is live but disagrees with the
portable reference (a real regression). Off-GPU it prints why each GPU section
was skipped and exits 0.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch

# Cap intra-op threads: on a shared/throttled node torch otherwise spawns one
# thread per visible core while the cgroup grants only a few, and the resulting
# oversubscription makes the CPU baseline (and the GPU path's host-side seeding)
# hundreds of times slower — which would both distort the A/B and risk a hang.
_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(_THREADS)

from graphmambaformer.accel import accel_summary, genomeworks_summary
from graphmambaformer.accel.backend import detect_capabilities, list_visible_gpus
from graphmambaformer.accel.cuda_kernels import kernels_available
from graphmambaformer.accel.genomeworks_ops import (
    genomeworks_align_batch_enabled,
    genomeworks_backend,
    genomeworks_bindings_available,
    genomeworks_poa_batch_enabled,
    global_align,
    global_align_batch,
    map_to_reference,
    poa_consensus,
    poa_consensus_batch,
    ungapped_extend,
    ungapped_extend_batch,
)

_BASES = "ACGT"


def _rand_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(_BASES) for _ in range(n))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    out = []
    for base in seq:
        r = rng.random()
        if r < rate * 0.6:  # substitution
            out.append(rng.choice(_BASES))
        elif r < rate * 0.8:  # deletion
            continue
        elif r < rate:  # insertion
            out.append(base)
            out.append(rng.choice(_BASES))
        else:
            out.append(base)
    return "".join(out)


def _hdr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# --------------------------------------------------------------------------- #
# 1. Environment
# --------------------------------------------------------------------------- #
def report_environment() -> bool:
    _hdr("Environment")
    print(f"torch              {torch.__version__}")
    print(f"cuda available     {torch.cuda.is_available()}")
    print(f"torch threads      {_THREADS} (of {os.cpu_count()} visible cores)")
    gpus = list_visible_gpus()
    if gpus:
        for g in gpus:
            print(f"  gpu[{g.get('index')}]        {g.get('name')}  "
                  f"cc={g.get('compute_capability')}  "
                  f"mem={g.get('total_memory_gb')}GB")
    else:
        print("  (no accelerator visible)")
    print(f"\naccel:      {accel_summary()}")
    print(f"genomeworks:{'':1}{genomeworks_summary()}")
    print(f"gw backend         {genomeworks_backend()}")
    print(f"real bindings      {genomeworks_bindings_available()}")
    print(f"cupy raw kernels   {kernels_available()}")

    gpu_live = torch.cuda.is_available() and genomeworks_backend() != "portable"
    if not gpu_live:
        print("\n[skip] GenomeWorks is on the PORTABLE tier "
              "(no CUDA GPU / no CuPy / no bindings).")
        print("       Parity + benchmark A/B need the CUDA tier; nothing GPU to test here.")
    return gpu_live


# --------------------------------------------------------------------------- #
# 2. Parity: CUDA tier vs portable NumPy reference
# --------------------------------------------------------------------------- #
def check_parity() -> int:
    _hdr("Parity — CUDA GenomeWorks vs portable reference")
    rng = random.Random(0xC0FFEE)
    failures = 0

    # cudaextender: ungapped_extend_batch(device="cuda") vs per-seed portable.
    ext_mismatch = 0
    for _ in range(200):
        L = rng.randint(40, 400)
        target = _rand_seq(L, rng)
        query = _mutate(target, 0.08, rng)
        k = min(len(query), len(target))
        seeds = sorted({(rng.randrange(k), rng.randrange(k)) for _ in range(rng.randint(1, 12))})
        gpu = ungapped_extend_batch(query, target, seeds, device="cuda")
        ref = [ungapped_extend(query, target, sq, st) for sq, st in seeds]
        for g, r in zip(gpu, ref):
            if not (g.query_start == r.query_start and g.query_end == r.query_end
                    and g.target_start == r.target_start and g.target_end == r.target_end
                    and abs(g.score - r.score) <= 1e-3):
                ext_mismatch += 1
    print(f"cudaextender  ungapped_extend_batch  mismatches={ext_mismatch}  "
          f"{'OK' if ext_mismatch == 0 else 'FAIL'}")
    failures += ext_mismatch

    # cudamapper: map_to_reference on CUDA vs portable. The end-to-end mapper
    # goes through minimizer seeding, whose CuPy/NumPy argsort can order equal
    # keys differently, so exact per-overlap equality is too brittle. Compare
    # the best (top-scoring) hit per read within a small coordinate tolerance,
    # which is the meaningful signal, and report the agreement rate.
    TOL = 25  # bp: absorbs benign seed-ordering / tie-break drift
    map_total = map_agree = map_disagree = 0
    for _ in range(25):
        ref_seq = _rand_seq(rng.randint(600, 1500), rng)
        reads = []
        for _ in range(rng.randint(2, 6)):
            start = rng.randrange(0, max(1, len(ref_seq) - 200))
            reads.append(_mutate(ref_seq[start:start + rng.randint(120, 200)], 0.05, rng))
        gpu = map_to_reference(reads, ref_seq, device="cuda")
        cpu = map_to_reference(reads, ref_seq, device="cpu")
        for g_list, c_list in zip(gpu, cpu):
            map_total += 1
            if not g_list and not c_list:
                map_agree += 1  # both correctly found nothing
                continue
            if not g_list or not c_list:
                map_disagree += 1  # one mapped, the other did not
                continue
            g, c = g_list[0], c_list[0]  # best hit (lists are score-sorted)
            if (g.strand == c.strand
                    and abs(g.target_start - c.target_start) <= TOL
                    and abs(g.target_end - c.target_end) <= TOL):
                map_agree += 1
            else:
                map_disagree += 1
    rate = map_agree / map_total if map_total else 1.0
    # Fail only on a systematic divergence, not one tie-break outlier.
    map_ok = rate >= 0.95
    print(f"cudamapper    map_to_reference       best-hit agree={map_agree}/{map_total} "
          f"({rate:.0%})  {'OK' if map_ok else 'FAIL'}")
    if not map_ok:
        failures += map_disagree

    # cudaaligner: batched GPU Gotoh (global_align_batch) vs per-pair host. The
    # GPU fills the DP + traceback pointers; the host rebuilds the CIGAR, so it
    # must match global_align exactly (score and CIGAR).
    aln_pairs = []
    for _ in range(64):
        q = _rand_seq(rng.randint(80, 300), rng)
        aln_pairs.append((q, _mutate(q, 0.12, rng)))
    gpu_aln = global_align_batch(aln_pairs)
    ref_aln = [global_align(q, t) for q, t in aln_pairs]
    aln_mismatch = sum(
        0 if (g.cigar == r.cigar and abs(g.score - r.score) <= 1e-6) else 1
        for g, r in zip(gpu_aln, ref_aln)
    )
    extra = "" if genomeworks_align_batch_enabled() else "  [fell back to host]"
    print(f"cudaaligner   global_align_batch     mismatches={aln_mismatch}  "
          f"{'OK' if aln_mismatch == 0 else 'FAIL'}{extra}")
    failures += aln_mismatch

    # cudapoa: batched GPU POA (poa_consensus_batch) vs per-group host. The GPU
    # fills each locus's graph-alignment DP; the host does traceback + fold +
    # heaviest-bundle, so every group's consensus must match poa_consensus.
    poa_groups = []
    for _ in range(48):
        backbone = _rand_seq(rng.randint(120, 240), rng)
        depth = rng.randint(2, 6)
        poa_groups.append([_mutate(backbone, 0.08, rng) for _ in range(depth)])
    gpu_poa = poa_consensus_batch(poa_groups)
    ref_poa = [poa_consensus(g) for g in poa_groups]
    poa_mismatch = sum(0 if g == r else 1 for g, r in zip(gpu_poa, ref_poa))
    extra = "" if genomeworks_poa_batch_enabled() else "  [fell back to host]"
    print(f"cudapoa       poa_consensus_batch    mismatches={poa_mismatch}  "
          f"{'OK' if poa_mismatch == 0 else 'FAIL'}{extra}")
    failures += poa_mismatch

    return failures


# --------------------------------------------------------------------------- #
# 3. Benchmark: CUDA tier vs portable tier
# --------------------------------------------------------------------------- #
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters: int) -> float:
    fn()  # warm up (JIT/NVRTC compile, index build, allocator)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def _map_prebuilt(reference: str, reads: list[str], device: str) -> tuple[float, float]:
    """Time (index_build_ms, map_ms) the way the pipeline uses cudamapper.

    The reference index is built once and reused across the read batch, instead
    of the ``map_to_reference`` facade's rebuild-per-call. ``map_ms`` is the
    per-batch cost of seeding every read against the prebuilt index and chaining
    the anchors — the steady-state mapping throughput.
    """
    from graphmambaformer.accel.backend import default_context
    from graphmambaformer.accel.genomeworks_ops import _seed_reads
    from graphmambaformer.alignment.chaining import AffineChainer
    from graphmambaformer.alignment.seeding import SeedingEngine
    from graphmambaformer.config import ChainingConfig, SeedingConfig

    seed_cfg = SeedingConfig(modes=("gpu_kmer",), kmer=15, window=10)
    seeder = SeedingEngine(seed_cfg, device=device)
    backend = default_context().kernel_backend("chaining") if device == "cuda" else "torch"
    chainer = AffineChainer(ChainingConfig(), backend=backend)

    def _build():
        return seeder.build_indices(reference)

    build_ms = _time(_build, 5)
    bundle = _build()

    def _map():
        # _seed_reads batches the whole read set into one device lookup on CUDA
        # (with a verify-then-trust fallback) and stays per-read on CPU, so this
        # is the honest GPU-batched vs CPU-per-read comparison the pipeline sees.
        anchors = _seed_reads(seeder, bundle, reads)
        return chainer.chain_batch(anchors, [None] * len(anchors), device=device)

    map_ms = _time(_map, 5)
    return build_ms, map_ms


def benchmark() -> None:
    _hdr("Benchmark — CUDA vs portable (ms/iter, lower is better)")
    rng = random.Random(7)

    # cudaextender: many seeds on one long pair, swept over batch size so the
    # GPU's throughput scaling is visible (a few thousand seeds barely occupy an
    # A100). The verify-then-trust warm-up is already spent by check_parity(),
    # so these timings reflect the steady-state kernel, not the CPU reference.
    L = 5000
    target = _rand_seq(L, rng)
    query = _mutate(target, 0.08, rng)
    for n_seeds in (2_000, 20_000, 100_000):
        seeds = [(rng.randrange(L - 1), rng.randrange(L - 1)) for _ in range(n_seeds)]
        gpu_ms = _time(lambda s=seeds: ungapped_extend_batch(query, target, s, device="cuda"), 10)
        cpu_ms = _time(lambda s=seeds: ungapped_extend_batch(query, target, s, device="cpu"), 3)
        print(f"cudaextender  {n_seeds:>7,} seeds x {L}bp  "
              f"cuda={gpu_ms:8.2f}  portable={cpu_ms:9.2f}  speedup={cpu_ms / gpu_ms:6.1f}x")

    # cudamapper: the facade rebuilds the minimizer index on every call, which
    # dominates for a small reference and hides the GPU. Measure it the way the
    # pipeline actually uses it — build the index ONCE, then map read batches —
    # and sweep the reference size so the GPU crossover is visible.
    for ref_len in (50_000, 500_000):
        ref_seq = _rand_seq(ref_len, rng)
        reads = [_mutate(ref_seq[s:s + 400], 0.05, rng)
                 for s in (rng.randrange(0, ref_len - 400) for _ in range(512))]
        g_build, g_map = _map_prebuilt(ref_seq, reads, "cuda")
        c_build, c_map = _map_prebuilt(ref_seq, reads, "cpu")
        print(f"cudamapper    ref {ref_len // 1000:>4}kbp  index-build   "
              f"cuda={g_build:8.2f}  portable={c_build:8.2f}  speedup={c_build / g_build:6.1f}x")
        print(f"cudamapper    ref {ref_len // 1000:>4}kbp  map 512 reads "
              f"cuda={g_map:8.2f}  portable={c_map:8.2f}  speedup={c_map / g_map:6.1f}x")

    from graphmambaformer.accel.genomeworks_ops import genomeworks_seed_batch_enabled
    _status = "active (verified)" if genomeworks_seed_batch_enabled() else "DISABLED (fell back to per-read)"
    print(f"cudamapper    batched GPU seeding: {_status}")

    # cudaaligner: batched global affine alignment, CUDA Gotoh kernel vs the
    # per-pair vectorised host reference, swept over batch size (one thread per
    # pair, so the GPU only pays off once many pairs are in flight — which is how
    # the extension stage feeds it).
    #
    # Spend the verify-then-trust warm-up first: until it is exhausted every call
    # also runs the full host reference, which would otherwise dominate the first
    # timed block and make it look ~1x. This mirrors steady-state pipeline use.
    _warm = [(_rand_seq(180, rng), _mutate(_rand_seq(180, rng), 0.1, rng))
             for _ in range(4)]
    for _ in range(12):
        global_align_batch(_warm)
    for n_pairs in (128, 512):
        pairs = []
        for _ in range(n_pairs):
            qp = _rand_seq(rng.randint(150, 260), rng)
            pairs.append((qp, _mutate(qp, 0.1, rng)))
        gpu_ms = _time(lambda p=pairs: global_align_batch(p), 5)
        cpu_ms = _time(
            lambda p=pairs: [global_align(q, t) for q, t in p], 3
        )
        tag = "" if genomeworks_align_batch_enabled() else " [host-fallback]"
        print(f"cudaaligner  {n_pairs:>4} pairs x ~200bp  "
              f"cuda={gpu_ms:8.2f}  portable={cpu_ms:8.2f}  "
              f"speedup={cpu_ms / gpu_ms:6.1f}x{tag}")

    # cudapoa: batched POA across many independent loci (the GPU fills each
    # locus's graph-alignment DP; graph build / traceback / fold stay on host),
    # vs the per-locus host loop. POA is a serial fold *within* a locus, so the
    # win comes entirely from running many loci at once — swept over locus count.
    # Exhaust the verify-then-trust warm-up first so timings are steady-state.
    _wg = [[_mutate(_rand_seq(150, rng), 0.08, rng) for _ in range(4)]
           for _ in range(8)]
    for _ in range(12):
        poa_consensus_batch(_wg)
    for n_loci in (64, 256):
        groups = []
        for _ in range(n_loci):
            backbone = _rand_seq(rng.randint(200, 320), rng)
            groups.append([_mutate(backbone, 0.08, rng)
                           for _ in range(rng.randint(3, 6))])
        gpu_ms = _time(lambda g=groups: poa_consensus_batch(g), 3)
        cpu_ms = _time(
            lambda g=groups: [poa_consensus(x) for x in g], 2
        )
        tag = "" if genomeworks_poa_batch_enabled() else " [host-fallback]"
        print(f"cudapoa      {n_loci:>4} loci x ~4x260bp  "
              f"cuda={gpu_ms:8.2f}  portable={cpu_ms:8.2f}  "
              f"speedup={cpu_ms / gpu_ms:6.1f}x{tag}")


def main() -> int:
    gpu_live = report_environment()
    if not gpu_live:
        return 0

    failures = check_parity()
    benchmark()

    _hdr("Summary")
    if failures:
        print(f"FAIL — {failures} CUDA/portable disagreements. GPU results are NOT trustworthy.")
        return 1
    print("PASS — CUDA GenomeWorks matches the portable reference; speedups above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
