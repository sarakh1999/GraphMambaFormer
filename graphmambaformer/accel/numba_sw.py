"""Optional Numba-JIT banded affine Smith-Waterman for the CPU extension tier.

Stage 3's banded affine DP (:func:`graphmambaformer.alignment.extension.
banded_affine_sw_batch`) is written in ``torch`` so a single kernel serves both
the CPU and CUDA tiers. On CPU, though, the DP is a Python loop over ~*read_len*
rows, each issuing a couple of dozen tiny ``torch`` ops — millions of op
dispatches whose cost is pure per-op overhead, not arithmetic. Profiling the
classical stages puts this at ~60% of CPU alignment time.

This module reimplements the *forward fill* of that exact banded recurrence as a
single ``@njit`` scalar kernel. It fills the same band-relative ``H``/``E``/``F``
matrices the torch path produces, so the existing
:func:`graphmambaformer.alignment.extension.traceback_banded` walks them
unchanged and the CIGAR is identical. Memory stays banded (``(B, M+1, 2*hb+1)``),
so it is safe on 65 kb ONT reads where a full-matrix SIMD traceback would not be.

Everything degrades gracefully: without Numba (or off CPU) the caller keeps the
portable torch path. The kernel releases the GIL (``nogil=True``), so the
per-read stage thread pool scales across cores.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np

#: Mirror the sentinels used by the torch DP so the two paths are bit-comparable.
_NEG_INF = -1.0e30
_N_CODE = 5  # ambiguous base; never scores as a match (see seeding.N_CODE)


@functools.lru_cache(maxsize=1)
def _numba() -> Any | None:
    """Return the ``numba`` module if importable, else ``None`` (cached)."""
    try:
        import numba  # noqa: F401

        return numba
    except Exception:
        return None


def numba_available() -> bool:
    """Whether the Numba banded-SW kernel can be used.

    Off unless Numba imports cleanly and ``GMF_NUMBA_SW`` is not set to ``0`` —
    the env var is a kill switch so a run can force the portable torch path.
    """
    if os.environ.get("GMF_NUMBA_SW", "1") == "0":
        return False
    return _numba() is not None


# The kernel is compiled lazily on first use so importing this module never pays
# the JIT cost (and never requires Numba). ``_KERNEL`` caches the compiled fn.
_KERNEL: Any | None = None


def _build_kernel() -> Any:
    numba = _numba()
    assert numba is not None
    from numba import njit

    @njit(cache=True, nogil=True, fastmath=False)
    def _fill(
        query,      # int32 (B, M)
        target,     # int32 (B, N)
        qlen,       # int64 (B,)
        tlen,       # int64 (B,)
        offset,     # int64 (B,)
        hb,         # int64
        match_s,    # float64
        mismatch_p,
        gap_open,
        gap_extend,
        x_drop,
        n_code,
        neg,
    ):
        B = query.shape[0]
        M = query.shape[1]
        width = 2 * hb + 1

        H = np.zeros((B, M + 1, width), dtype=np.float32)
        E = np.full((B, M + 1, width), neg, dtype=np.float32)
        F = np.full((B, M + 1, width), neg, dtype=np.float32)
        score = np.zeros(B, dtype=np.float32)
        best_i = np.zeros(B, dtype=np.int64)
        best_j = np.zeros(B, dtype=np.int64)

        # Previous-row H / F with one guard column past the band (H=0, F=neg), so
        # the vertical predecessor at band index width-1 is always addressable.
        h_prev = np.zeros((B, width + 1), dtype=np.float32)
        f_prev = np.full((B, width + 1), neg, dtype=np.float32)
        row_max_seen = np.zeros(B, dtype=np.float32)

        # Row-local scratch, reused across (i, b) to avoid per-cell allocation.
        m_val = np.empty(width, dtype=np.float32)
        f_val = np.empty(width, dtype=np.float32)
        in_band = np.empty(width, dtype=np.bool_)

        max_rows = 0
        for b in range(B):
            if qlen[b] > max_rows:
                max_rows = qlen[b]

        for i in range(1, max_rows + 1):
            all_dropped = True
            for b in range(B):
                off = offset[b]
                ql = qlen[b]
                tl = tlen[b]

                # Pass 1: M (best score not ending in a horizontal gap) and the
                # vertical-gap term F, per band column d.
                for d in range(width):
                    j = i + off - hb + d  # target column, 1-based
                    inb = (j >= 1) and (j <= tl) and (i <= ql)
                    in_band[d] = inb
                    if inb:
                        qb = query[b, i - 1]
                        tb = target[b, j - 1]
                        is_match = (qb == tb) and (qb >= 0) and (qb != n_code)
                        sub = match_s if is_match else -mismatch_p
                        diag = h_prev[b, d] + sub
                        vert = h_prev[b, d + 1] - gap_open
                        vf = f_prev[b, d + 1] - gap_extend
                        if vf > vert:
                            vert = vf
                        mv = diag if diag > vert else vert
                        if mv < 0.0:
                            mv = 0.0
                        m_val[d] = mv
                        f_val[d] = vert
                    else:
                        m_val[d] = neg
                        f_val[d] = neg

                # Pass 2: horizontal-gap term E as an exclusive max-plus prefix
                # scan of (m + d*gap_extend); e[d] = run_{d'<d} - open - (d-1)*ext.
                run = neg
                for d in range(width):
                    e_d = run - (gap_open + (d - 1) * gap_extend)
                    E[b, i, d] = e_d if in_band[d] else neg
                    cand = m_val[d] + d * gap_extend
                    if cand > run:
                        run = cand

                # Pass 3: H = max(M, E) clamped at 0; track the row's best cell.
                row_best = np.float32(0.0)
                row_slot = -1
                for d in range(width):
                    if in_band[d]:
                        ev = E[b, i, d]
                        hv = m_val[d]
                        if ev > hv:
                            hv = ev
                        if hv < 0.0:
                            hv = 0.0
                        H[b, i, d] = hv
                        F[b, i, d] = f_val[d]
                        if hv > row_best:
                            row_best = hv
                            row_slot = d
                    else:
                        H[b, i, d] = 0.0
                        F[b, i, d] = neg

                if row_slot >= 0 and row_best > score[b]:
                    score[b] = row_best
                    best_i[b] = i
                    best_j[b] = i + off - hb + row_slot
                if row_best > row_max_seen[b]:
                    row_max_seen[b] = row_best

                for d in range(width):
                    h_prev[b, d] = H[b, i, d]
                    f_prev[b, d] = F[b, i, d]
                h_prev[b, width] = 0.0
                f_prev[b, width] = neg

                # X-drop is global: the row loop stops only once every element
                # has fallen x_drop below its own best (matches the torch path).
                if x_drop > 0.0:
                    if (row_max_seen[b] - row_best) <= x_drop:
                        all_dropped = False
                else:
                    all_dropped = False

            if x_drop > 0.0 and all_dropped:
                break

        return H, E, F, score, best_i, best_j

    return _fill


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def banded_affine_sw_fill(
    query: np.ndarray,
    target: np.ndarray,
    query_len: np.ndarray,
    target_len: np.ndarray,
    band_offset: np.ndarray,
    half_band: int,
    *,
    match_score: float,
    mismatch_penalty: float,
    gap_open: float,
    gap_extend: float,
    x_drop: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill the banded affine-SW ``H``/``E``/``F`` matrices with the Numba kernel.

    Mirrors :func:`graphmambaformer.alignment.extension.banded_affine_sw_batch`
    with ``return_matrices=True``. Returns ``(H, E, F, score, best_i, best_j)``
    where ``H``/``E``/``F`` are ``float32 (B, M+1, 2*half_band+1)`` band-relative
    matrices, ``score`` is the best local score per element, and ``best_i`` /
    ``best_j`` are the 1-based query / target ends of that best cell.
    """
    q = np.ascontiguousarray(query, dtype=np.int32)
    t = np.ascontiguousarray(target, dtype=np.int32)
    ql = np.ascontiguousarray(query_len, dtype=np.int64)
    tl = np.ascontiguousarray(target_len, dtype=np.int64)
    off = np.ascontiguousarray(band_offset, dtype=np.int64)
    return _kernel()(
        q,
        t,
        ql,
        tl,
        off,
        int(half_band),
        float(match_score),
        float(mismatch_penalty),
        float(gap_open),
        float(gap_extend),
        float(x_drop),
        int(_N_CODE),
        float(_NEG_INF),
    )
