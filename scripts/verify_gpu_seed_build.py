"""Verify the CuPy GPU seeding-index build equals the portable NumPy build.

The Stage-1 indices (minimizer, fuzzy, SMEM/FM-index, DBG, multiplex) build their
sorts / suffix arrays on the GPU when a CUDA device is supplied and the reference
clears ``GMF_GPU_SEED_BUILD_MIN`` (see :mod:`graphmambaformer.alignment.seeding`).
The result is copied back to the host, so the per-read query is unchanged. This
script proves the GPU build produces an index whose *queries agree* with the
NumPy build, and reports the build speedup on a chromosome-scale slice.

Run on a GPU node (e.g. a0015):

    cd ~/mambaformer
    source osc_gpu_env.sh && source .venv/bin/activate
    PYTHONPATH=. python scripts/verify_gpu_seed_build.py

The two GPU-build floors (``GMF_GPU_SEED_BUILD_MIN`` for the FM-index suffix
array, ``GMF_GPU_SEED_TABLE_BUILD_MIN`` for the k-mer-table sketches) are pinned
to 1 below so the correctness pass forces the GPU path even for the small cases;
the timing section uses references large enough to trigger both regardless.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types

# Force both GPU-build floors on for the correctness pass (must precede the
# seeding import, which reads these at module load).
os.environ.setdefault("GMF_GPU_SEED_BUILD_MIN", "1")
os.environ.setdefault("GMF_GPU_SEED_TABLE_BUILD_MIN", "1")

import numpy as np

# Bootstrap lightweight package stubs so we import the seeding leaf module without
# dragging in the neural stack (mamba_ssm). Works identically on login/GPU nodes.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL = os.path.join(_REPO, "graphmambaformer")
sys.path.insert(0, _REPO)


def _stub(name: str, subdir: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [os.path.join(_REAL, subdir) if subdir else _REAL]
    module.__package__ = name
    sys.modules[name] = module


for _name, _sub in (
    ("graphmambaformer", ""),
    ("graphmambaformer.accel", "accel"),
    ("graphmambaformer.alignment", "alignment"),
):
    _stub(_name, _sub)

S = importlib.import_module("graphmambaformer.alignment.seeding")

import torch  # noqa: E402  (after stubs so torch is a fresh top-level import)

assert torch.cuda.is_available(), "no CUDA device visible — run on a GPU node"
import cupy as cp  # noqa: E402

rng = np.random.default_rng(7)


def rand_seq(n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


def as_rows(query_out) -> list:
    """Anchor tuples as an order-independent sorted list for set comparison."""
    read_pos, ref_pos, length = query_out[0], query_out[1], query_out[2]
    return sorted(zip(read_pos.tolist(), ref_pos.tolist(), length.tolist()))


def table_is_cupy(idx) -> bool:
    table = getattr(idx, "table", None)
    if table is not None:
        return isinstance(table.keys, cp.ndarray)
    # multiplex holds several tables
    tables = getattr(idx, "tables", {})
    return any(isinstance(t.keys, cp.ndarray) for t in tables.values())


ok = True

print("=" * 74)
print("Correctness: GPU-built index queries == CPU-built index queries")
print("=" * 74)
for (N, k, w, mo) in [(2_000, 15, 10, 0), (60_000, 15, 10, 8), (200_000, 13, 7, 4)]:
    ref = rand_seq(N)
    read = ref[N // 3 : N // 3 + 400]
    rc, rd = S.encode_bases(ref), S.encode_bases(read)

    cases = {
        "minimizer": (
            lambda d: S.MinimizerIndex(rc, k=k, window=w, max_occ=mo, device=d),
        ),
        "fuzzy": (
            lambda d: S.FuzzySeedIndex(rc, "1101011011", max_occ=mo, device=d),
        ),
        "multiplex": (
            lambda d: S.MultiplexDBG(rc, kmers=(15, 21), max_occ=mo, device=d),
        ),
        "dbg": (
            lambda d: S.DeBruijnIndex([rc], k=21, max_occ=mo, device=d),
        ),
        "smem": (
            lambda d: S.SMEMIndex(rc, min_seed_len=13, max_occ=200, device=d),
        ),
    }
    for name, (make,) in cases.items():
        gpu = make("cuda")
        cpu = make(None)
        g = as_rows(gpu.query(rd))
        c = as_rows(cpu.query(rd))
        match = g == c
        ok &= match
        # SMEM/DBG build on the GPU but the FM sampled-SA stays host arrays; the
        # k-mer tables should be materialized on host too. We only require query
        # agreement here (host copy-back is by design), so just report matches.
        print(f"  N={N:>7} {name:>10}: anchors gpu={len(g):>4} cpu={len(c):>4}  MATCH={match}")

print()
print("=" * 74)
print("Build speedup: GPU (CuPy) vs CPU (NumPy), chromosome-scale slice")
print("=" * 74)
for N in [1_000_000, 10_000_000]:
    codes = S.encode_bases(rand_seq(N))

    # warm the JIT / allocator
    S.MinimizerIndex(codes[:100_000], k=15, window=10, device="cuda")
    torch.cuda.synchronize()

    for name, make in [
        ("minimizer", lambda d: S.MinimizerIndex(codes, k=15, window=10, device=d)),
        ("smem", lambda d: S.SMEMIndex(codes, min_seed_len=13, device=d)),
    ]:
        t0 = time.perf_counter()
        make("cuda")
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        make(None)
        cpu_ms = (time.perf_counter() - t0) * 1e3

        print(
            f"  N={N:>10} {name:>10}: gpu={gpu_ms:8.1f} ms  cpu={cpu_ms:9.1f} ms  "
            f"speedup={cpu_ms / max(gpu_ms, 1e-9):5.1f}x"
        )

print()
print("ALL MATCH" if ok else "MISMATCH DETECTED")
sys.exit(0 if ok else 1)
