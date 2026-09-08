"""Verify the batched CuPy cudaextender kernel equals the portable NumPy walk.

Stage-3's ungapped X-drop prefilter (``ExtensionConfig.ungapped_prefilter``)
now batches every chain's seed extension through
:func:`graphmambaformer.accel.genomeworks_ops.ungapped_extend_batch`, which on a
CUDA host dispatches to the batched ``cudaextender`` CuPy RawKernel
(:func:`graphmambaformer.accel.cuda_kernels.ungapped_extend`) instead of walking
each seed on the CPU. The kernel is trusted only after it agrees with the
portable reference, so results are identical by construction; this script proves
that (bypassing the trust gate to test the kernel directly) and reports the
GPU-vs-CPU speedup for a realistic batch of seeds.

Run on a GPU node (e.g. a0015):

    cd ~/mambaformer
    source osc_gpu_env.sh && source .venv/bin/activate
    PYTHONPATH=. python scripts/verify_cudaextender_prefilter.py
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types

import numpy as np

# Bootstrap lightweight package stubs so we import the accel leaf modules without
# dragging in the neural stack (mamba_ssm). Works on login/GPU nodes alike.
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
):
    _stub(_name, _sub)

GE = importlib.import_module("graphmambaformer.accel.genomeworks_ops")
K = importlib.import_module("graphmambaformer.accel.cuda_kernels")

import torch  # noqa: E402  (after stubs so torch is a fresh top-level import)

assert torch.cuda.is_available(), "no CUDA device visible — run on a GPU node"
assert K.kernels_available(), "CuPy RawKernels unavailable — check the CUDA env"

rng = np.random.default_rng(11)

MATCH, MISMATCH = 2.0, 4.0
DEV = torch.device("cuda")


def rand_codes(n: int) -> np.ndarray:
    """Random base-code array (A C G T -> 1..4)."""
    return rng.integers(1, 5, size=n, dtype=np.int8)


def mutate(codes: np.ndarray, rate: float) -> np.ndarray:
    out = codes.copy()
    mask = rng.random(len(out)) < rate
    out[mask] = rng.integers(1, 5, size=int(mask.sum()), dtype=np.int8)
    return out


def make_case(read_len: int, ref_len: int, n_seeds: int, sub_rate: float):
    """A read embedded in a reference plus on- and off-diagonal seeds."""
    ref = rand_codes(ref_len)
    offset = ref_len // 4
    read = mutate(ref[offset : offset + read_len], sub_rate)
    # On-diagonal seeds (read pos q -> ref pos q+offset) plus a few off-diagonal
    # ones so low-scoring extensions are exercised too.
    qs = rng.integers(0, read_len, size=n_seeds).astype(np.int64)
    rs = (qs + offset).astype(np.int64)
    n_off = max(1, n_seeds // 10)
    rs[:n_off] = rng.integers(0, ref_len, size=n_off)
    return read, ref, list(zip(qs.tolist(), rs.tolist()))


def kernel_rows(read, ref, seeds, x_drop):
    q_t = torch.as_tensor(read, dtype=torch.int8, device=DEV)
    t_t = torch.as_tensor(ref, dtype=torch.int8, device=DEV)
    sq = torch.as_tensor([s[0] for s in seeds], dtype=torch.int32, device=DEV)
    sr = torch.as_tensor([s[1] for s in seeds], dtype=torch.int32, device=DEV)
    qs, qe, ts, te, sc = K.ungapped_extend(
        q_t, t_t, sq, sr, match=MATCH, mismatch=MISMATCH, x_drop=x_drop
    )
    torch.cuda.synchronize()
    return (
        qs.cpu().numpy(), qe.cpu().numpy(),
        ts.cpu().numpy(), te.cpu().numpy(), sc.cpu().numpy(),
    )


def portable_rows(read, ref, seeds, x_drop):
    exts = [
        GE.ungapped_extend(read, ref, sq, sr, match=MATCH, mismatch=MISMATCH, x_drop=x_drop)
        for sq, sr in seeds
    ]
    return (
        np.array([e.query_start for e in exts]),
        np.array([e.query_end for e in exts]),
        np.array([e.target_start for e in exts]),
        np.array([e.target_end for e in exts]),
        np.array([e.score for e in exts], dtype=np.float64),
    )


ok = True

print("=" * 74)
print("Correctness: batched cudaextender kernel == portable NumPy X-drop walk")
print("=" * 74)
for (read_len, ref_len, n_seeds, sub, x_drop) in [
    (400, 2_000, 64, 0.05, 40.0),
    (800, 4_000, 256, 0.10, 40.0),
    (800, 4_000, 256, 0.10, 600.0),
    (1_200, 6_000, 512, 0.20, 600.0),
]:
    read, ref, seeds = make_case(read_len, ref_len, n_seeds, sub)

    kq0, kq1, kt0, kt1, ks = kernel_rows(read, ref, seeds, x_drop)
    pq0, pq1, pt0, pt1, ps = portable_rows(read, ref, seeds, x_drop)

    span = (
        np.array_equal(kq0, pq0) and np.array_equal(kq1, pq1)
        and np.array_equal(kt0, pt0) and np.array_equal(kt1, pt1)
    )
    score = bool(np.allclose(ks, ps, atol=1e-3, rtol=1e-4))

    # Also exercise the pipeline entry point end-to-end (device pinning): the
    # cuda batch must equal the cpu-pinned (portable) batch.
    g = GE.ungapped_extend_batch(read, ref, seeds, match=MATCH, mismatch=MISMATCH,
                                 x_drop=x_drop, device="cuda")
    c = GE.ungapped_extend_batch(read, ref, seeds, match=MATCH, mismatch=MISMATCH,
                                 x_drop=x_drop, device="cpu")
    entry = all(
        gi.query_start == ci.query_start and gi.query_end == ci.query_end
        and gi.target_start == ci.target_start and gi.target_end == ci.target_end
        and abs(gi.score - ci.score) <= 1e-3
        for gi, ci in zip(g, c)
    )

    match = span and score and entry
    ok &= match
    print(
        f"  read={read_len:>5} ref={ref_len:>6} seeds={n_seeds:>4} x_drop={x_drop:>5.0f}"
        f"  span={span!s:>5} score={score!s:>5} entry={entry!s:>5}  MATCH={match}"
    )

print()
print("=" * 74)
print("Throughput: batched GPU kernel vs portable CPU walk")
print("=" * 74)
for (read_len, ref_len, n_seeds) in [(800, 4_000, 20_000), (1_500, 8_000, 50_000)]:
    read, ref, seeds = make_case(read_len, ref_len, n_seeds, 0.12)

    # warm the JIT / allocator
    kernel_rows(read, ref, seeds[:256], 40.0)

    t0 = time.perf_counter()
    kernel_rows(read, ref, seeds, 40.0)
    gpu_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    portable_rows(read, ref, seeds, 40.0)
    cpu_ms = (time.perf_counter() - t0) * 1e3

    print(
        f"  read={read_len:>5} ref={ref_len:>6} seeds={n_seeds:>6}: "
        f"gpu={gpu_ms:8.1f} ms  cpu={cpu_ms:9.1f} ms  "
        f"speedup={cpu_ms / max(gpu_ms, 1e-9):5.1f}x"
    )

print()
print("ALL MATCH" if ok else "MISMATCH DETECTED")
sys.exit(0 if ok else 1)
