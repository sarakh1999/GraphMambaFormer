"""Device selection for NVIDIA CUDA, Apple Silicon (MPS), Intel XPU, and CPU.

Also resolves multi-GPU device-id lists so training can wrap the model in
``nn.DataParallel`` and use every visible accelerator (honours
``CUDA_VISIBLE_DEVICES``).
"""

from __future__ import annotations

import contextlib
import os
from typing import Sequence

import torch

# Allow unsupported ops (e.g. some linalg kernels) to fall back to CPU on MPS
# instead of raising NotImplementedError. Must be set before the op runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _xpu_available() -> bool:
    with contextlib.suppress(AttributeError, RuntimeError):
        return bool(torch.xpu.is_available())  # type: ignore[attr-defined]
    return False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Preference order when ``prefer`` is None / ``"auto"``:
      1. CUDA (NVIDIA / ROCm — whatever ``torch.cuda`` exposes)
      2. Intel XPU
      3. MPS (Apple Silicon / Metal)
      4. CPU

    Pass ``prefer="cpu"`` / ``"mps"`` / ``"cuda"`` / ``"cuda:1"`` / ``"xpu"``
    to force a backend. Indexed CUDA devices (``cuda:N``) are validated against
    ``torch.cuda.device_count()`` (honours ``CUDA_VISIBLE_DEVICES``).
    """
    if prefer is None or prefer in ("", "auto", "best"):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if _xpu_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(prefer)
    if device.type == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but not available on this machine")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available "
                "(no driver, CPU-only torch wheel, or container without --gpus)"
            )
        n = torch.cuda.device_count()
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx < 0 or idx >= n:
            raise RuntimeError(
                f"{device} out of range — torch sees {n} CUDA device(s). "
                "Check CUDA_VISIBLE_DEVICES or pass --device cuda:0."
            )
        torch.cuda.set_device(idx)
        return torch.device("cuda", idx)
    elif device.type == "xpu":
        if not _xpu_available():
            raise RuntimeError("XPU requested but not available on this machine")
    return device


def resolve_device_ids(
    devices: str | Sequence[int] | None = "auto",
    *,
    primary: torch.device | str | None = None,
) -> list[int]:
    """Return CUDA/XPU device indices to train on.

    ``devices``:
      * ``None`` / ``\"auto\"`` — every visible CUDA (or XPU) device when count > 1,
        else just the primary index (DataParallel is a no-op on one device).
      * ``\"all\"`` — every visible CUDA/XPU device (even if only one).
      * ``\"0,1,3\"`` / ``[0, 1, 3]`` — explicit list.
      * ``\"none\"`` / ``\"off\"`` / a bare digit — single-device on that / primary.

    Honours ``CUDA_VISIBLE_DEVICES``. Returns ``[]`` on CPU/MPS (no multi-device).
    """
    primary_dev = get_device(
        None if primary in (None, "", "auto", "best") else str(primary)
    )
    if primary_dev.type == "cuda":
        available = list(range(torch.cuda.device_count()))
        default_primary = (
            primary_dev.index
            if primary_dev.index is not None
            else torch.cuda.current_device()
        )
    elif primary_dev.type == "xpu":
        available = list(range(torch.xpu.device_count()))  # type: ignore[attr-defined]
        default_primary = primary_dev.index if primary_dev.index is not None else 0
    else:
        return []

    if not available:
        return []

    if devices is None or (
        isinstance(devices, str) and devices.strip().lower() in ("", "auto")
    ):
        if len(available) <= 1:
            return [int(default_primary)]
        return available

    if isinstance(devices, (list, tuple)):
        ids = [int(x) for x in devices]
    else:
        text = str(devices).strip().lower()
        if text in ("none", "off", "single"):
            return [int(default_primary)]
        if text in ("all", "*"):
            return available
        if "," not in text and text.isdigit():
            ids = [int(text)]
        else:
            ids = [int(part.strip()) for part in text.split(",") if part.strip() != ""]

    bad = [i for i in ids if i not in available]
    if bad:
        raise RuntimeError(
            f"device id(s) {bad} not visible — torch sees {available} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})"
        )
    if not ids:
        return [int(default_primary)]
    return ids


def wrap_data_parallel(
    model: torch.nn.Module, device_ids: Sequence[int]
) -> torch.nn.Module:
    """Wrap ``model`` in ``nn.DataParallel`` when more than one device is given."""
    ids = [int(i) for i in device_ids]
    if len(ids) <= 1:
        return model
    return torch.nn.DataParallel(model, device_ids=ids, output_device=ids[0])


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped in ``DataParallel`` / DDP."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def device_summary() -> str:
    """One-line description of which accelerator torch will use."""
    from .accel import list_visible_gpus

    d = get_device()
    parts = [f"device={d}"]
    if d.type == "mps":
        parts.append("backend=Metal/MPS (Apple Silicon GPU)")
    elif d.type == "cuda":
        idx = d.index if d.index is not None else 0
        with contextlib.suppress(Exception):
            parts.append(f"name={torch.cuda.get_device_name(idx)}")
        with contextlib.suppress(Exception):
            major, minor = torch.cuda.get_device_capability(idx)
            parts.append(f"sm_{major}{minor}")
        gpus = list_visible_gpus()
        if len(gpus) > 1:
            parts.append(f"visible={len(gpus)}")
    elif d.type == "xpu":
        parts.append("backend=Intel XPU")
    else:
        parts.append("backend=CPU")
    return " | ".join(parts)
