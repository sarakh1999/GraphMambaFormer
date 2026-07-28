#!/usr/bin/env python3
"""Verify Apple Silicon GPU backends (PyTorch MPS + MLX Metal).

Run outside Cursor's sandbox (a normal Terminal.app / iTerm window)::

    source .venv/bin/activate
    python scripts/check_gpu.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python     {sys.version.split()[0]}")
    print(f"platform   {platform.platform()}")
    print(f"mac_ver    {platform.mac_ver()}")

    ok = True

    try:
        import torch

        print(f"torch      {torch.__version__}")
        print(f"mps_built  {torch.backends.mps.is_built()}")
        print(f"mps_avail  {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            x = torch.randn(2048, 2048, device="mps")
            y = x @ x
            torch.mps.synchronize()
            print(f"mps_matmul OK  device={y.device}  mean={float(y.mean()):.4f}")
        else:
            print("mps_matmul SKIP (MPS not available in this process)")
            ok = False
    except Exception as exc:  # pragma: no cover
        print(f"torch FAIL  {exc}")
        ok = False

    try:
        import mlx.core as mx

        print(f"mlx        default_device={mx.default_device()}")
        a = mx.random.normal((2048, 2048))
        b = a @ a
        mx.eval(b)
        print(f"mlx_matmul OK  mean={float(b.mean()):.4f}")
    except Exception as exc:  # pragma: no cover
        print(f"mlx FAIL   {exc}")
        ok = False

    if ok:
        print("\nGPU ready: PyTorch MPS + MLX Metal are usable.")
        return 0

    print(
        "\nGPU check failed in this process.\n"
        "If you are inside Cursor's agent sandbox, re-run this in Terminal.app:\n"
        "  cd ~/Desktop/mambaformer && source .venv/bin/activate && python scripts/check_gpu.py"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
