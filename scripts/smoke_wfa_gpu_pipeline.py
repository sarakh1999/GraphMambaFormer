"""End-to-end smoke test: WFA-GPU inside the real alignment pipeline.

Unlike ``scripts/verify_wfa_gpu.py`` (which unit-tests the WFA-GPU backend in
isolation), this drives the *actual* pipeline object training uses —
``build_pipeline(mode="fast")`` on CUDA with ``extension.algorithm="wfa"`` — so
it certifies that:

  1. the ``wfa`` route in :meth:`ExtensionEngine.extend_chains` really reaches
     WFA-GPU (we instrument ``wfa_gpu_ops.align_batch`` and require calls > 0),
  2. mapping results are equivalent to the same pipeline forced onto the CPU
     WavefrontAligner (``GMF_WFA_GPU=0``) — identical mapped-flags and reference
     start positions (scores/MAPQ reported; co-optimal CIGAR ties are fine),
  3. it runs end-to-end without errors and we can see the wall-clock cost.

This is the check to run before flipping a training job to ``algorithm="wfa"``.

Prerequisites (on a GPU node, e.g. a0015):
    cd ~/mambaformer && source osc_gpu_env.sh && source .venv/bin/activate
    bash scripts/build_wfa_gpu.sh          # builds libgmf_wfa_gpu.so
    # osc_gpu_env.sh already exported GMF_WFA_GPU / GMF_WFA_GPU_LIB / LD_LIBRARY_PATH

Then:
    PYTHONPATH=. python scripts/smoke_wfa_gpu_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from graphmambaformer.accel import wfa_gpu_ops as WG  # noqa: E402
from graphmambaformer.alignment import build_pipeline  # noqa: E402
from graphmambaformer.config import (  # noqa: E402
    AccelConfig,
    ExtensionConfig,
    PipelineConfig,
)

_BASES = np.array(list("ACGT"))


def _random_reference(rng: np.random.Generator, length: int) -> str:
    return "".join(_BASES[rng.integers(0, 4, size=length)])


def _mutate(rng: np.random.Generator, seq: str, sub: float, indel: float) -> str:
    """Apply i.i.d. substitutions + small indels to a read-sized slice."""
    out: list[str] = []
    for ch in seq:
        r = rng.random()
        if r < indel / 2:  # deletion: drop this base
            continue
        if r < indel:  # insertion: emit a random base, then keep the original
            out.append(_BASES[rng.integers(0, 4)])
        if rng.random() < sub:
            alt = _BASES[rng.integers(0, 4)]
            out.append(alt if alt != ch else _BASES[(ord(ch) + 1) % 4])
        else:
            out.append(ch)
    return "".join(out)


def _make_reads(
    rng: np.random.Generator,
    reference: str,
    n_reads: int,
    read_len: int,
    sub: float,
    indel: float,
) -> tuple[list[str], list[int]]:
    reads: list[str] = []
    truth: list[int] = []
    hi = len(reference) - read_len - 1
    for _ in range(n_reads):
        pos = int(rng.integers(0, hi))
        reads.append(_mutate(rng, reference[pos : pos + read_len], sub, indel))
        truth.append(pos)
    return reads, truth


def _build(read_len: int) -> "tuple":
    """Real training-style pipeline: fast mode, CUDA, wfa extension."""
    cfg = PipelineConfig(
        mode="fast",
        extension=ExtensionConfig(algorithm="wfa", wfa_max_distance=100_000),
        accel=AccelConfig(device="cuda"),
    )
    pipe = build_pipeline(cfg, device="cuda")
    return pipe


def _primaries(results) -> list[tuple[bool, int, int, float]]:
    out: list[tuple[bool, int, int, float]] = []
    for ra in results:
        p = ra.primary
        if p is None:
            out.append((False, -1, 0, 0.0))
        else:
            out.append(
                (bool(p.is_mapped), int(p.ref_start), int(p.mapq), float(p.alignment_score))
            )
    return out


def _time_align(pipe, reads, ref_index) -> tuple[list, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results, _ = pipe.align(reads, ref_index)
    torch.cuda.synchronize()
    return results, (time.perf_counter() - t0) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reads", type=int, default=300)
    ap.add_argument("--read-len", type=int, default=400)
    ap.add_argument("--ref-len", type=int, default=60_000)
    ap.add_argument("--sub", type=float, default=0.04)
    ap.add_argument("--indel", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device visible — run on a GPU node (e.g. a0015)")
        return 2
    if os.environ.get("GMF_WFA_GPU", "0") != "1" or not WG.available():
        err = WG.load_error()
        print("WFA-GPU is not enabled/loadable — this smoke test needs it.")
        if err:
            print(f"    load error: {err}")
        print("    fix: source osc_gpu_env.sh (after building) so GMF_WFA_GPU=1 and")
        print("         GMF_WFA_GPU_LIB / LD_LIBRARY_PATH are set; keep the CUDA module loaded.")
        return 2
    print(WG.summary())
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(args.seed)
    reference = _random_reference(rng, args.ref_len)
    reads, _truth = _make_reads(
        rng, reference, args.reads, args.read_len, args.sub, args.indel
    )

    # Instrument the WFA-GPU entry point so we can prove the route was taken.
    calls = {"n": 0, "pairs": 0}
    _orig_align_batch = WG.align_batch

    def _counting_align_batch(pairs, **kw):
        pairs = list(pairs)
        calls["n"] += 1
        calls["pairs"] += len(pairs)
        return _orig_align_batch(pairs, **kw)

    WG.align_batch = _counting_align_batch  # extension.py resolves this at call time
    try:
        pipe = _build(args.read_len)
        ref_index = pipe.build_reference(reference)

        # --- GPU WFA path (GMF_WFA_GPU=1) ---------------------------------- #
        os.environ["GMF_WFA_GPU"] = "1"
        _ = pipe.align(reads[: min(16, len(reads))], ref_index)  # warm up
        calls["n"] = calls["pairs"] = 0
        gpu_results, gpu_ms = _time_align(pipe, reads, ref_index)
        gpu_calls, gpu_pairs = calls["n"], calls["pairs"]

        # --- CPU WFA baseline (GMF_WFA_GPU=0) ------------------------------ #
        os.environ["GMF_WFA_GPU"] = "0"
        calls["n"] = calls["pairs"] = 0
        cpu_results, cpu_ms = _time_align(pipe, reads, ref_index)
        cpu_calls = calls["n"]
        os.environ["GMF_WFA_GPU"] = "1"
    finally:
        WG.align_batch = _orig_align_batch

    gpu = _primaries(gpu_results)
    cpu = _primaries(cpu_results)

    gpu_mapped = sum(1 for m, *_ in gpu if m)
    cpu_mapped = sum(1 for m, *_ in cpu if m)
    mapped_flag_match = sum(1 for a, b in zip(gpu, cpu) if a[0] == b[0])
    pos_match = sum(1 for a, b in zip(gpu, cpu) if a[0] and b[0] and a[1] == b[1])
    both_mapped = sum(1 for a, b in zip(gpu, cpu) if a[0] and b[0])
    mapq_deltas = [abs(a[2] - b[2]) for a, b in zip(gpu, cpu) if a[0] and b[0]]
    score_deltas = [abs(a[3] - b[3]) for a, b in zip(gpu, cpu) if a[0] and b[0]]

    print("\n" + "=" * 74)
    print("WFA-GPU exercised inside the pipeline")
    print("=" * 74)
    print(f"  reads={args.reads} read_len={args.read_len} ref_len={args.ref_len} "
          f"sub={args.sub} indel={args.indel}")
    print(f"  GPU run: align_batch calls={gpu_calls} pairs={gpu_pairs}  "
          f"(CPU run calls={cpu_calls}, expected 0)")

    print("\n" + "=" * 74)
    print("Mapping equivalence: GPU-WFA vs CPU-WFA (same pipeline)")
    print("=" * 74)
    print(f"  mapped: gpu={gpu_mapped} cpu={cpu_mapped}")
    print(f"  mapped-flag agree: {mapped_flag_match}/{len(gpu)}")
    print(f"  ref_start agree (both mapped): {pos_match}/{both_mapped}")
    if mapq_deltas:
        print(f"  |dMAPQ| max={max(mapq_deltas)} mean={sum(mapq_deltas)/len(mapq_deltas):.3f}")
    if score_deltas:
        print(f"  |dscore| max={max(score_deltas):.3g} "
              f"mean={sum(score_deltas)/len(score_deltas):.3g}")

    print("\n" + "=" * 74)
    print("Throughput (end-to-end align of the read set; seeding+chaining shared)")
    print("=" * 74)
    print(f"  gpu-wfa={gpu_ms:.1f} ms   cpu-wfa={cpu_ms:.1f} ms   "
          f"speedup={cpu_ms / gpu_ms:.2f}x")

    # ---- verdict ---------------------------------------------------------- #
    ok = True
    if gpu_calls == 0 or gpu_pairs == 0:
        print("\nFAIL: WFA-GPU path was never taken (check algorithm=='wfa' and the "
              "extension backend is not routed to 'genomeworks'/cudaaligner).")
        ok = False
    if cpu_calls != 0:
        print(f"\nFAIL: CPU baseline still called WFA-GPU ({cpu_calls}x); GMF_WFA_GPU=0 "
              "did not disable it.")
        ok = False
    if gpu_mapped == 0:
        print("\nFAIL: no reads mapped — nothing meaningful was extended.")
        ok = False
    if mapped_flag_match != len(gpu):
        print("\nFAIL: GPU and CPU disagree on which reads mapped.")
        ok = False
    if both_mapped and pos_match / both_mapped < 0.99:
        print(f"\nFAIL: ref_start agreement {pos_match}/{both_mapped} < 99% — GPU and CPU "
              "WFA place reads differently.")
        ok = False

    print("\n" + ("ALL GOOD — WFA-GPU is training-ready on this path." if ok else "SMOKE TEST FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
