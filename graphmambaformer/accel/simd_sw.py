"""Optional x86 SIMD Smith-Waterman scoring through parasail.

Parasail ships explicit AVX2 and SSE2 striped kernels. The alignment engine
already narrows each candidate to the chain-derived reference window; this tier
scores that window with the best available SIMD ISA, while the portable banded
DP retains the traceback matrices.
"""

from __future__ import annotations

import functools
import platform
from typing import Any

from ..config import ExtensionConfig


@functools.lru_cache(maxsize=1)
def _parasail() -> Any | None:
    try:
        import parasail

        return parasail
    except Exception:
        return None


def simd_available() -> bool:
    """Whether an SSE2/AVX2 parasail kernel can run on this host."""

    return _parasail() is not None and platform.machine().lower() in (
        "x86_64",
        "amd64",
    )


def simd_isa() -> str | None:
    parasail = _parasail()
    if not simd_available() or parasail is None:
        return None
    if hasattr(parasail, "sw_striped_avx2_256_16"):
        return "avx2"
    if hasattr(parasail, "sw_striped_sse2_128_16"):
        return "sse2"
    return "dispatch"


def smith_waterman_score(query: str, target: str, cfg: ExtensionConfig) -> float:
    """Affine-gap local score using the best installed SIMD implementation."""

    parasail = _parasail()
    isa = simd_isa()
    if parasail is None or isa is None:
        raise RuntimeError("SSE2/AVX2 Smith-Waterman backend unavailable")
    matrix = parasail.matrix_create(
        "ACGTN",
        int(round(cfg.match_score)),
        -int(round(cfg.mismatch_penalty)),
    )
    function = (
        getattr(parasail, "sw_striped_avx2_256_16")
        if isa == "avx2"
        else getattr(parasail, "sw_striped_sse2_128_16")
        if isa == "sse2"
        else getattr(parasail, "sw_striped_16")
    )
    result = function(
        query,
        target,
        int(round(cfg.gap_open)),
        int(round(cfg.gap_extend)),
        matrix,
    )
    return float(result.score)
