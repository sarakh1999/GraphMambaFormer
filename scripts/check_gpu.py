#!/usr/bin/env python3
"""Verify the acceleration stack on whatever GPU this host exposes.

Works on NVIDIA (A100 / A6000 / H100 / H200 / …), AMD ROCm, Intel XPU, Apple
Silicon (MPS), and plain CPU. Run outside a restricted sandbox when probing a
real GPU::

    source .venv/bin/activate
    python scripts/check_gpu.py
    python scripts/check_gpu.py --device cuda:0
"""

from __future__ import annotations

import argparse
import platform
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto",
                   help="cuda / cuda:N / mps / xpu / cpu / auto (default)")
    args = p.parse_args()

    print(f"python     {sys.version.split()[0]}")
    print(f"platform   {platform.platform()}")

    ok = True
    try:
        import torch

        print(f"torch      {torch.__version__}")
        print(f"cuda_built {torch.version.cuda}")
        print(f"hip_built  {getattr(torch.version, 'hip', None)}")
        print(f"cuda_avail {torch.cuda.is_available()}")
        print(f"mps_avail  {torch.backends.mps.is_available() and torch.backends.mps.is_built()}")
    except Exception as exc:  # pragma: no cover
        print(f"torch FAIL  {exc}")
        return 1

    from graphmambaformer.accel import AccelContext, list_visible_gpus
    from graphmambaformer.config import AccelConfig

    gpus = list_visible_gpus()
    if not gpus:
        print("gpus       (none visible)")
    for g in gpus:
        cc = g.get("compute_capability")
        mem = g.get("total_memory_gb")
        extra = []
        if cc:
            extra.append(f"sm_{cc[0]}{cc[1]}")
        if mem:
            extra.append(f"{mem} GiB")
        if g.get("hip_arch"):
            extra.append(g["hip_arch"])
        print(f"gpu[{g['index']}]    {g['name']}"
              + (f"  ({', '.join(extra)})" if extra else ""))

    try:
        ctx = AccelContext(AccelConfig(device=None if args.device == "auto" else args.device))
    except RuntimeError as exc:
        print(f"accel FAIL  {exc}")
        return 1

    print(f"accel      {ctx.summary()}")
    device = ctx.caps.device

    # Tiny matmul on every visible GPU — proves each card can allocate and compute.
    try:
        x = torch.randn(1024, 1024, device=device)
        with ctx.autocast():
            y = x @ x
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        print(f"matmul     OK  device={y.device}  mean={float(y.float().mean()):.4f}"
              f"  amp={ctx.autocast_dtype}")
    except Exception as exc:
        print(f"matmul FAIL  {exc}")
        ok = False

    # Probe every other visible CUDA/XPU card so multi-GPU hosts fail loudly
    # when only device 0 works.
    for g in gpus:
        idx = int(g["index"])
        if device.type == "cuda" and (
            device.index is None or int(device.index) != idx
        ):
            try:
                d = torch.device("cuda", idx)
                x = torch.randn(512, 512, device=d)
                y = x @ x
                torch.cuda.synchronize(d)
                print(f"matmul[{idx}] OK  device={y.device}  mean={float(y.mean()):.4f}")
            except Exception as exc:
                print(f"matmul[{idx}] FAIL  {exc}")
                ok = False
        elif device.type == "xpu" and (
            device.index is None or int(device.index) != idx
        ):
            try:
                d = torch.device("xpu", idx)
                x = torch.randn(512, 512, device=d)
                y = x @ x
                print(f"matmul[{idx}] OK  device={y.device}  mean={float(y.mean()):.4f}")
            except Exception as exc:
                print(f"matmul[{idx}] FAIL  {exc}")
                ok = False

    # Apple MLX is optional and only useful when Metal is actually reachable.
    if (platform.system() == "Darwin"
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()):
        try:
            import mlx.core as mx

            a = mx.random.normal((1024, 1024))
            b = a @ a
            mx.eval(b)
            print(f"mlx        OK  default_device={mx.default_device()}  "
                  f"mean={float(b.mean()):.4f}")
        except Exception as exc:
            print(f"mlx        SKIP  {exc}")
    elif platform.system() == "Darwin":
        print("mlx        SKIP  (Metal/MPS not available in this process)")

    if not ok:
        print("\nGPU check failed.")
        return 1
    if device.type == "cpu":
        print("\nNo GPU visible — CPU path is usable. For NVIDIA: install a CUDA "
              "torch wheel and run with --gpus all in Docker.")
        return 0
    print(f"\nGPU ready: {ctx.caps.arch_label} via {ctx.caps.tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
