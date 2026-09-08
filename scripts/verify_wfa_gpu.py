"""Verify the optional WFA-GPU CIGAR backend against ground truth + the CPU WFA.

WFA-GPU (github.com/quim0/WFA-GPU) is the maintained, CUDA-12-capable stand-in
for the archived GenomeWorks ``cudaaligner``: it produces gap-affine CIGARs on
the GPU. The extension engine uses it (opt-in, ``GMF_WFA_GPU=1``) for the
``wfa`` algorithm with unit-cost penalties (x=1,o=0,e=1), which reproduces the
CPU :class:`WavefrontAligner`'s Levenshtein objective.

This script proves, on an actual GPU node:
  1. WFA-GPU's edit distance == the true Levenshtein distance (DP ground truth),
  2. WFA-GPU's edit distance == the in-tree CPU WavefrontAligner's distance,
  3. WFA-GPU's CIGAR converts to a valid pipeline ``=/X/I/D`` alignment that
     walks both sequences and whose implied unit distance matches (2),
and reports the batched GPU-vs-CPU throughput.

Prerequisites (run once):
    cd ~/mambaformer && source osc_gpu_env.sh && source .venv/bin/activate
    bash scripts/build_wfa_gpu.sh          # builds libgmf_wfa_gpu.so
    export GMF_WFA_GPU_LIB=.../libgmf_wfa_gpu.so   # printed by the build script
    export LD_LIBRARY_PATH=...                     # printed by the build script

Then:
    PYTHONPATH=. python scripts/verify_wfa_gpu.py
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types

import numpy as np

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

EXT = importlib.import_module("graphmambaformer.alignment.extension")
WG = importlib.import_module("graphmambaformer.accel.wfa_gpu_ops")

import torch  # noqa: E402

assert torch.cuda.is_available(), "no CUDA device visible — run on a GPU node"
if not WG.available():
    err = WG.load_error()
    if err:
        print("WFA-GPU shim is present but failed to load:")
        print(f"    {err}")
        print("Likely libcudart/libwfagpu not found — keep the CUDA module loaded and")
        print("ensure LD_LIBRARY_PATH includes the WFA-GPU build dir (see build script output).")
    else:
        print("WFA-GPU shim not found. Build it first:")
        print("    bash scripts/build_wfa_gpu.sh")
        print("    export GMF_WFA_GPU_LIB=<printed path>   # and LD_LIBRARY_PATH")
    sys.exit(2)
print(WG.summary())

rng = np.random.default_rng(17)
cfg = EXT.ExtensionConfig(wfa_max_distance=1_000_000)
wfa_cpu = EXT.WavefrontAligner(cfg)


def rand_seq(n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


def mutate(seq: str, sub: float, indel: float) -> str:
    out = []
    for ch in seq:
        r = rng.random()
        if r < indel / 2:
            continue  # deletion
        if r < indel:
            out.append(rng.choice(list("ACGT")))  # insertion
            out.append(ch)
        elif r < indel + sub:
            out.append(rng.choice([c for c in "ACGT" if c != ch]))  # substitution
        else:
            out.append(ch)
    return "".join(out) or seq[:1]


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = np.arange(n + 1)
    for i in range(1, m + 1):
        cur = np.empty(n + 1, dtype=np.int64)
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return int(prev[n])


ok = True
print("=" * 74)
print("Correctness: WFA-GPU distance == Levenshtein == CPU WFA; CIGAR valid")
print("=" * 74)
for (length, sub, indel, count) in [
    (120, 0.05, 0.02, 64),
    (250, 0.10, 0.05, 64),
    (400, 0.15, 0.08, 48),
]:
    queries = [rand_seq(length) for _ in range(count)]
    targets = [mutate(q, sub, indel) for q in queries]
    pairs = list(zip(queries, targets))

    res = WG.align_batch(pairs, x=1, o=0, e=1, compute_cigar=True, max_error=0)
    assert res is not None, "align_batch returned None on a GPU with the shim loaded"

    dist_ok = cigar_ok = cpu_ok = True
    cigar_equal = 0
    for (q, t), (err, cig) in zip(pairs, res):
        qc, tc = EXT.encode_bases(q), EXT.encode_bases(t)
        lev = levenshtein(q, t)
        if err != lev:
            dist_ok = False
        d_cpu, cpu_cigar = wfa_cpu.align(qc, tc)
        if d_cpu != err:
            cpu_ok = False
        pcig = EXT._wfa_gpu_cigar_to_pipeline(cig or "", qc, tc)
        if pcig is None:
            cigar_ok = False
        else:
            unit = sum(n for op, n in pcig if op in ("X", "I", "D"))
            if unit != err:
                cigar_ok = False
            if pcig == cpu_cigar:
                cigar_equal += 1

    match = dist_ok and cigar_ok and cpu_ok
    ok &= match
    print(
        f"  len={length:>4} sub={sub:.2f} indel={indel:.2f} n={count:>3}: "
        f"dist={dist_ok!s:>5} cpu={cpu_ok!s:>5} cigar={cigar_ok!s:>5} "
        f"(cigar==cpu {cigar_equal}/{count}, co-optimal ties expected)  MATCH={match}"
    )

print()
print("=" * 74)
print("Throughput: batched WFA-GPU (with CIGAR) vs CPU WavefrontAligner loop")
print("=" * 74)
for (length, count) in [(250, 5_000), (500, 5_000)]:
    queries = [rand_seq(length) for _ in range(count)]
    targets = [mutate(q, 0.08, 0.04) for q in queries]
    pairs = list(zip(queries, targets))
    encoded = [(EXT.encode_bases(q), EXT.encode_bases(t)) for q, t in pairs]

    WG.align_batch(pairs[:256], x=1, o=0, e=1, compute_cigar=True)  # warm up
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    WG.align_batch(pairs, x=1, o=0, e=1, compute_cigar=True, max_error=0)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for qc, tc in encoded:
        wfa_cpu.align(qc, tc)
    cpu_ms = (time.perf_counter() - t0) * 1e3

    print(
        f"  len={length:>4} pairs={count:>6}: gpu={gpu_ms:8.1f} ms  cpu={cpu_ms:9.1f} ms  "
        f"speedup={cpu_ms / max(gpu_ms, 1e-9):5.1f}x"
    )

print()
print("ALL MATCH" if ok else "MISMATCH DETECTED")
sys.exit(0 if ok else 1)
