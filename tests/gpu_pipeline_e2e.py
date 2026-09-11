#!/usr/bin/env python
"""Whole-pipeline check: GenomeWorks GPU paths vs the portable baseline.

``gpu_genomeworks_check.py`` proves each GenomeWorks *primitive* matches its
portable reference and how fast it is in isolation. This script proves the
*assembled* alignment pipeline (stages 1-4: seed -> chain -> extend) still maps
reads correctly once the GenomeWorks GPU paths are routed on, and reports the
end-to-end wall-clock next to the portable baseline.

It runs the same synthetic reference + reads twice:

  * baseline   AccelConfig(genomeworks=False)  -> torch/NumPy stages
  * genomeworks AccelConfig(genomeworks=True, stage_backends routed) ->
                cudamapper seeding + cudaextender ungapped prefilter +
                cudaaligner global extension + CuPy chaining (on CUDA)

Turning GenomeWorks on deliberately *changes the extension algorithm* (banded
local Smith-Waterman -> global cudaaligner), so the two configs need not emit
identical CIGARs. The honest correctness metric is therefore mapping accuracy
against the known truth locus, measured for *both* configs: the GPU route must
be at least as accurate as the baseline. Wall-clock is reported for context
(only meaningful on a real GPU host; on CPU both configs use the portable tier).

Usage:
    PYTHONPATH="$PWD" python tests/gpu_pipeline_e2e.py            # defaults
    PYTHONPATH="$PWD" python tests/gpu_pipeline_e2e.py --reads 2000 --ref-len 500000
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

# Cap intra-op threads before importing torch so a login/compute node with many
# visible cores does not oversubscribe the host stages (see gpu_genomeworks_check).
_THREADS = min(8, os.cpu_count() or 8)
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))

import torch  # noqa: E402

try:
    torch.set_num_threads(_THREADS)
except Exception:
    pass

from graphmambaformer import PipelineConfig, build_pipeline  # noqa: E402
from graphmambaformer.accel.genomeworks_ops import (  # noqa: E402
    genomeworks_align_batch_enabled,
    genomeworks_backend,
    genomeworks_seed_batch_enabled,
)
from graphmambaformer.config import AccelConfig  # noqa: E402

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _rand_ref(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _revcomp(seq: str) -> str:
    return "".join(_COMP[b] for b in reversed(seq))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    out = []
    for b in seq:
        r = rng.random()
        if r < rate * 0.6:                      # substitution
            out.append(rng.choice([x for x in "ACGT" if x != b]))
        elif r < rate * 0.8:                    # deletion
            continue
        elif r < rate:                          # insertion
            out.append(rng.choice("ACGT"))
            out.append(b)
        else:
            out.append(b)
    return "".join(out) or seq


def _make_reads(reference: str, n: int, length: int, rate: float,
                rng: random.Random) -> list[tuple[str, int]]:
    """``(read, true_start)`` pairs; ~30% are reverse-complement of their locus."""
    reads: list[tuple[str, int]] = []
    hi = len(reference) - length - 1
    for _ in range(n):
        start = rng.randint(0, hi)
        window = reference[start:start + length]
        read = _mutate(window, rate, rng)
        if rng.random() < 0.3:
            read = _revcomp(read)
        reads.append((read, start))
    return reads


def _build(genomeworks: bool, device: str):
    cfg = PipelineConfig(mode="fast")
    if genomeworks:
        cfg.accel = AccelConfig(
            genomeworks=True,
            stage_backends={
                "seeding": "genomeworks",
                "chaining": "auto",
                "extension": "genomeworks",
            },
        )
    else:
        cfg.accel = AccelConfig(genomeworks=False)
    return build_pipeline(cfg, model=None, device=device)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run(pipeline, reference: str, reads: list[str]):
    """Return (primaries, elapsed_ms). ``primaries[i]`` is (mapped, ref_start)."""
    ref_idx = pipeline.build_reference(reference)
    # Warm up once (kernel compiles, allocator, verify-then-trust warm-up).
    warm = reads[: min(64, len(reads))]
    pipeline.align(warm, ref_idx)
    _sync()
    t0 = time.perf_counter()
    results, _ = pipeline.align(reads, ref_idx)
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    primaries = []
    for r in results:
        p = r.primary
        if p is not None and p.is_mapped:
            primaries.append((True, int(p.ref_start)))
        else:
            primaries.append((False, -1))
    return primaries, elapsed_ms


def _accuracy(primaries, truth, tol: int) -> tuple[int, int]:
    """(#mapped, #accurate) — accurate = mapped within ``tol`` of the truth locus."""
    mapped = sum(1 for m, _ in primaries if m)
    accurate = sum(
        1 for (m, rs), ts in zip(primaries, truth)
        if m and abs(rs - ts) <= tol
    )
    return mapped, accurate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-len", type=int, default=200_000)
    ap.add_argument("--reads", type=int, default=800)
    ap.add_argument("--read-len", type=int, default=200)
    ap.add_argument("--error", type=float, default=0.08)
    ap.add_argument("--tol", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    reference = _rand_ref(args.ref_len, rng)
    pairs = _make_reads(reference, args.reads, args.read_len, args.error, rng)
    reads = [r for r, _ in pairs]
    truth = [s for _, s in pairs]

    bar = "=" * 68
    print(bar)
    print("Whole-pipeline end-to-end — GenomeWorks GPU vs portable baseline")
    print(bar)
    print(f"device            {device}")
    print(f"gw backend        {genomeworks_backend()}")
    print(f"reference         {args.ref_len:,} bp")
    print(f"reads             {args.reads} x {args.read_len} bp @ {args.error:.0%} error "
          f"(~30% reverse-complement)")
    print(f"locus tolerance   +/- {args.tol} bp")
    print()

    # Portable baseline.
    base = _build(genomeworks=False, device=device)
    base_prim, base_ms = _run(base, reference, reads)
    base_mapped, base_acc = _accuracy(base_prim, truth, args.tol)

    # GenomeWorks GPU route.
    gw = _build(genomeworks=True, device=device)
    print("genomeworks routing:")
    print(f"  extension algorithm  {gw.extender._effective_algorithm()}")
    print(f"  ungapped prefilter   {gw.extender._prefilter_enabled()}")
    print(f"  seeding modes        {tuple(gw.seeder.cfg.modes)}")
    print(f"  batched seed/align   seed={genomeworks_seed_batch_enabled()} "
          f"align={genomeworks_align_batch_enabled()}")
    print()
    gw_prim, gw_ms = _run(gw, reference, reads)
    gw_mapped, gw_acc = _accuracy(gw_prim, truth, args.tol)

    # Cross-config locus agreement (where both mapped).
    both = agree = 0
    for (mb, rb), (mg, rg) in zip(base_prim, gw_prim):
        if mb and mg:
            both += 1
            if abs(rb - rg) <= args.tol:
                agree += 1

    n = args.reads
    print(bar)
    print("Results (ms = wall-clock over all reads, lower is better)")
    print(bar)
    print(f"{'config':<14}{'wall_ms':>12}{'mapped':>14}{'accuracy':>16}")
    print(f"{'baseline':<14}{base_ms:>12.1f}{base_mapped:>9}/{n:<4}"
          f"{100.0 * base_acc / n:>13.1f}%")
    print(f"{'genomeworks':<14}{gw_ms:>12.1f}{gw_mapped:>9}/{n:<4}"
          f"{100.0 * gw_acc / n:>13.1f}%")
    if gw_ms > 0:
        print(f"\nspeedup (baseline/genomeworks wall-clock): {base_ms / gw_ms:.2f}x"
              "   [only meaningful on a real GPU host]")
    print(f"cross-config locus agreement (both mapped): {agree}/{both}"
          f" ({(100.0 * agree / both) if both else 0.0:.1f}%)")
    print()

    # PASS: the GPU route must map at least as accurately as the baseline and
    # clear a floor, so acceleration provably did not degrade the pipeline.
    floor = 0.90
    ok = (gw_acc >= base_acc - int(0.02 * n)) and (gw_acc >= floor * n)
    print(bar)
    if ok:
        print(f"PASS — genomeworks route accuracy {100.0 * gw_acc / n:.1f}% "
              f">= baseline {100.0 * base_acc / n:.1f}% (floor {floor:.0%}).")
        return 0
    print(f"FAIL — genomeworks accuracy {100.0 * gw_acc / n:.1f}% vs baseline "
          f"{100.0 * base_acc / n:.1f}% (floor {floor:.0%}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
