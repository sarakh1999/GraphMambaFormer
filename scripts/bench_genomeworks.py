#!/usr/bin/env python
"""Micro-benchmark and doctor for the GenomeWorks acceleration layer.

Reports which GenomeWorks tier is active (real ``pyclaragenomics`` bindings, the
CuPy ``RawKernel`` reimplementations, or the portable NumPy reference) and times
each of the four primitives on synthetic data:

    cudaextender   ungapped X-drop seed extension
    cudaaligner    global affine alignment (+CIGAR)
    cudapoa        partial-order-alignment consensus
    cudamapper     GPU minimizer seeding + chaining (read -> reference)

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_genomeworks.py
    PYTHONPATH=. .venv/bin/python scripts/bench_genomeworks.py --device cuda --reads 512
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np


def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    out = []
    for base in seq:
        r = rng.random()
        if r < rate * 0.6:  # substitution
            out.append(rng.choice([b for b in "ACGT" if b != base]))
        elif r < rate * 0.8:  # deletion
            continue
        elif r < rate:  # insertion
            out.append(base)
            out.append(rng.choice("ACGT"))
        else:
            out.append(base)
    return "".join(out)


def _time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda / cuda:N / cpu / auto")
    ap.add_argument("--reads", type=int, default=256, help="reads for cudamapper bench")
    ap.add_argument("--read-len", type=int, default=200)
    ap.add_argument("--ref-len", type=int, default=20_000)
    ap.add_argument("--poa-depth", type=int, default=20)
    ap.add_argument("--error-rate", type=float, default=0.08)
    args = ap.parse_args()

    from graphmambaformer.accel import AccelContext
    from graphmambaformer.accel import genomeworks_ops as gw

    ctx = AccelContext(device=args.device)
    print("=" * 72)
    print("GenomeWorks acceleration doctor")
    print("=" * 72)
    print(f"  device            : {ctx.caps.device}")
    print(f"  accel tier        : {ctx.caps.tier}")
    print(f"  genomeworks       : {gw.genomeworks_summary()}")
    print(f"  real bindings     : {gw.genomeworks_bindings_available()}")
    print(f"  active backend    : {gw.genomeworks_backend()}")
    print("-" * 72)

    rng = random.Random(0)
    reference = _random_seq(args.ref_len, rng)

    # cudaextender -------------------------------------------------------- #
    q = _random_seq(args.read_len, rng)
    t = q[: args.read_len // 2] + _random_seq(args.read_len // 2, rng)
    seeds = [(i, i) for i in range(0, args.read_len // 2, 8)]
    ms = _time(lambda: gw.ungapped_extend_batch(q, t, seeds, x_drop=40))
    print(f"  cudaextender      : {len(seeds)} seeds ungapped X-drop   {ms:8.2f} ms")

    # cudaaligner --------------------------------------------------------- #
    ref_win = reference[1000 : 1000 + args.read_len]
    read = _mutate(ref_win, args.error_rate, rng)
    aln = gw.global_align(read, ref_win)
    ms = _time(lambda: gw.global_align(read, ref_win))
    print(
        f"  cudaaligner       : global {len(read)}x{len(ref_win)}          {ms:8.2f} ms"
        f"   (id={aln.n_match / max(1, len(read)):.2f}, {aln.cigar_string[:24]}…)"
    )

    # cudapoa ------------------------------------------------------------- #
    template = reference[5000 : 5000 + args.read_len]
    cluster = [_mutate(template, args.error_rate, rng) for _ in range(args.poa_depth)]
    consensus = gw.poa_consensus(cluster)
    ms = _time(lambda: gw.poa_consensus(cluster), repeats=1)
    # Consensus identity to the noise-free template.
    ident = gw.global_align(consensus, template).n_match / max(1, len(template))
    print(
        f"  cudapoa           : {args.poa_depth}-read consensus         {ms:8.2f} ms"
        f"   (consensus id vs template={ident:.3f})"
    )

    # cudamapper ---------------------------------------------------------- #
    reads = []
    for _ in range(args.reads):
        pos = rng.randint(0, args.ref_len - args.read_len)
        reads.append(_mutate(reference[pos : pos + args.read_len], args.error_rate, rng))
    device = str(ctx.caps.device)
    t0 = time.perf_counter()
    overlaps = gw.map_to_reference(reads, reference, kmer=15, window=10, device=device)
    dt = time.perf_counter() - t0
    mapped = sum(1 for o in overlaps if o)
    print(
        f"  cudamapper        : {args.reads} reads seed+chain      {dt * 1e3:8.2f} ms"
        f"   ({mapped}/{args.reads} mapped, {args.reads / max(dt, 1e-9):.0f} reads/s)"
    )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
