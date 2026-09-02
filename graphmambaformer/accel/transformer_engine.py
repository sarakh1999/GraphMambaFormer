"""TransformerEngine FP8 mixed precision (Hopper/Ada) with Ampere fallback.

NVIDIA TransformerEngine accelerates Transformer-style Linear/LayerNorm paths
with FP8 on Ada (sm_89) and Hopper (sm_90+). The RTX A6000 is Ampere (sm_86)
and has **no** FP8 tensor cores, so enabling FP8 there must not hard-fail: this
module falls back to BF16/FP16 torch autocast while still exposing a uniform
``precision_context`` API.

Detection is lazy and never raises at import time.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Iterator

import torch

__all__ = [
    "transformer_engine_available",
    "transformer_engine_module",
    "fp8_available_on_device",
    "precision_context",
    "te_summary",
]


@functools.lru_cache(maxsize=None)
def transformer_engine_module() -> Any | None:
    """Return ``transformer_engine.pytorch`` when importable, else ``None``."""
    try:
        import transformer_engine.pytorch as te  # type: ignore

        return te
    except Exception:
        return None


def transformer_engine_available() -> bool:
    return transformer_engine_module() is not None


def fp8_available_on_device(
    compute_capability: tuple[int, int] | None,
    *,
    vendor: str = "nvidia",
) -> bool:
    """True only when the GPU has FP8 tensor cores (Ada sm_89+ / Hopper+)."""
    if vendor != "nvidia" or compute_capability is None:
        return False
    return compute_capability >= (8, 9)


@contextlib.contextmanager
def precision_context(
    *,
    use_fp8: bool,
    device_type: str = "cuda",
    fallback_dtype: torch.dtype | None = torch.bfloat16,
    enabled: bool = True,
) -> Iterator[str]:
    """Enter the best available mixed-precision context.

    Yields a short label describing which path was taken::

        "fp8" | "bf16" | "fp16" | "fp32"

    * ``use_fp8=True`` and TE installed → ``transformer_engine`` ``fp8_autocast``
    * otherwise → ``torch.autocast`` with ``fallback_dtype`` when set
    * ``enabled=False`` or no dtype → plain eager fp32
    """
    if not enabled:
        yield "fp32"
        return

    te = transformer_engine_module() if use_fp8 else None
    if te is not None and use_fp8 and device_type == "cuda":
        # DelayedScaling is the TE default recipe; keep kwargs minimal so older
        # TE releases still accept the call.
        try:
            with te.fp8_autocast(enabled=True):
                yield "fp8"
            return
        except Exception:
            # Recipe / recipe-unavailable on this driver: fall through to AMP.
            pass

    if fallback_dtype is None or device_type not in ("cuda", "cpu", "xpu"):
        yield "fp32"
        return

    label = "bf16" if fallback_dtype == torch.bfloat16 else (
        "fp16" if fallback_dtype == torch.float16 else "amp"
    )
    with torch.autocast(device_type=device_type, dtype=fallback_dtype):
        yield label


def te_summary(
    *,
    compute_capability: tuple[int, int] | None = None,
    vendor: str = "nvidia",
) -> str:
    """One-line status for logs / ``check_gpu``."""
    te = transformer_engine_available()
    fp8 = fp8_available_on_device(compute_capability, vendor=vendor)
    if te and fp8:
        return "transformer_engine=yes fp8=yes"
    if te and not fp8:
        return "transformer_engine=yes fp8=no (Ampere/older → BF16 fallback)"
    return "transformer_engine=no"
