"""Device selection for Apple Silicon (MPS), CUDA, and CPU."""

from __future__ import annotations

import os

import torch

# Allow unsupported ops (e.g. some linalg kernels) to fall back to CPU on MPS
# instead of raising NotImplementedError. Must be set before the op runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Preference order when ``prefer`` is None / ``"auto"``:
      1. CUDA (NVIDIA)
      2. MPS (Apple Silicon / Metal)
      3. CPU

    Pass ``prefer="cpu"`` / ``"mps"`` / ``"cuda"`` to force a backend.
    """
    if prefer is None or prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(prefer)
    if device.type == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but not available on this machine")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
    return device


def device_summary() -> str:
    """One-line description of which accelerator torch will use."""
    d = get_device()
    parts = [f"device={d}"]
    if d.type == "mps":
        parts.append("backend=Metal/MPS (Apple Silicon GPU)")
    elif d.type == "cuda":
        parts.append(f"name={torch.cuda.get_device_name(0)}")
    else:
        parts.append("backend=CPU")
    return " | ".join(parts)
