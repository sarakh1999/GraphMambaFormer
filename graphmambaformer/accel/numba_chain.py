"""Optional Numba-JIT chaining DP for the CPU chaining tier.

Stage 2's chaining DP (:func:`graphmambaformer.alignment.chaining.chain_dp_numpy`)
is, on CPU, a Python loop over anchors that vectorizes each anchor's
``max_lookback`` predecessors with NumPy. Once Stages 1 and 3 are on the Numba
tier, this per-anchor Python + tiny-array NumPy work is the single largest CPU
cost of the classical pipeline, and — being Python — it holds the GIL, so the
per-read stage thread pool cannot scale across it.

This module reimplements the *same* recurrence as one ``@njit`` scalar kernel:
identical predecessor test, ``min(dq, dr, weight)`` advance, affine+log gap
penalty, optional graph bonus, and first-wins argmax tie-break against the
``weight[i]`` baseline. It returns the same ``(f, predecessor)`` as
``chain_dp_numpy`` (verified against it in the tests), so the backtrack and every
downstream stage are unchanged.

Degrades gracefully: without Numba the caller keeps ``chain_dp_numpy``. The
kernel releases the GIL (``nogil=True``), so chaining now parallelizes with the
seeding/extension kernels. Honours the shared ``GMF_NUMBA_SW`` kill switch.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .numba_sw import numba_available  # shared probe + GMF_NUMBA_SW kill switch

__all__ = ["numba_available", "chain_dp_numba"]

_NO_PREDECESSOR = -1

_KERNEL: Any | None = None


def _build_kernel() -> Any:
    from numba import njit

    @njit(cache=True, nogil=True, fastmath=False)
    def _fill(
        read_end,   # int64 (n,)
        ref_end,    # int64 (n,)
        weight,     # float64 (n,)
        bonus,      # float64 (n, lookback) when has_bonus else (1, 1)
        has_bonus,
        lookback,
        max_gap,
        gap_open,
        gap_extend,
        log_coeff,
        no_pred,
    ):
        n = read_end.shape[0]
        f = np.zeros(n, dtype=np.float64)
        parent = np.full(n, no_pred, dtype=np.int64)
        for i in range(n):
            wi = weight[i]
            lo = i - lookback
            if lo < 0:
                lo = 0
            if lo == i:                      # i == 0: no predecessor window
                f[i] = wi
                continue
            best = wi                        # the "start a new chain" baseline
            best_p = -1
            rei = read_end[i]
            rfi = ref_end[i]
            for p in range(lo, i):
                dq = rei - read_end[p]
                dr = rfi - ref_end[p]
                if dq > 0 and dr > 0 and dq <= max_gap and dr <= max_gap:
                    av = np.float64(dq if dq < dr else dr)
                    adv = av if av < wi else wi
                    gap = dr - dq
                    if gap < 0:
                        gap = -gap
                    if gap > 0:
                        pen = gap_open + gap_extend * gap + log_coeff * np.log2(gap + 1.0)
                    else:
                        pen = 0.0
                    sc = f[p] + adv - pen
                    if has_bonus:
                        sc += bonus[i, lookback - (i - p)]
                    if sc > best:            # strict: first predecessor wins ties
                        best = sc
                        best_p = p
            f[i] = best
            if best_p >= 0:
                parent[i] = best_p
        return f, parent

    return _fill


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def chain_dp_numba(
    read_end: np.ndarray,
    ref_end: np.ndarray,
    weight: np.ndarray,
    *,
    lookback: int,
    max_gap: int,
    gap_open: float,
    gap_extend: float,
    log_coeff: float,
    bonus: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Chaining DP for one read via the Numba kernel; mirrors ``chain_dp_numpy``.

    ``read_end`` / ``ref_end`` are the anchors' reference-sorted end coordinates,
    ``weight`` their per-anchor scores, ``bonus`` the optional ``(n, lookback)``
    graph bonus (column ``t`` = predecessor ``i - lookback + t``). Returns
    ``(f, predecessor)`` as ``float64`` / ``int64``.
    """
    re = np.ascontiguousarray(read_end, dtype=np.int64)
    rf = np.ascontiguousarray(ref_end, dtype=np.int64)
    w = np.ascontiguousarray(weight, dtype=np.float64)
    if bonus is not None:
        b = np.ascontiguousarray(bonus, dtype=np.float64)
        has_bonus = True
    else:
        b = np.zeros((1, 1), dtype=np.float64)
        has_bonus = False
    return _kernel()(
        re,
        rf,
        w,
        b,
        has_bonus,
        np.int64(max(1, lookback)),
        np.int64(max_gap),
        np.float64(gap_open),
        np.float64(gap_extend),
        np.float64(log_coeff),
        np.int64(_NO_PREDECESSOR),
    )
