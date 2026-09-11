"""Optional Numba-JIT FM-index query kernels for the CPU seeding tier.

Once Stage 3 is off the critical path (see :mod:`numba_sw`), Stage 1's SMEM
seeding dominates CPU time, and profiling puts essentially all of it in the
FM-index backward search: the ``Occ(c, i)`` rank primitive and the ``count`` /
``locate`` / ``smems`` loops built on it. In NumPy each of those is a Python
loop issuing hundreds of thousands of tiny array ops per batch — pure per-call
overhead — and, being Python, they hold the GIL, so the per-read stage thread
pool cannot scale across seeding.

This module reimplements those primitives as ``@njit`` scalar kernels sharing an
inlined rank helper (a checkpoint lookup plus a bounded ``< occ_sample`` BWT
scan):

* :func:`fm_rank`   — batched ``Occ`` (kept for the generic ``rank`` method),
* :func:`fm_count`  — SA interval of one exact pattern (backward search),
* :func:`fm_locate` — text offsets for SA rows (LF walk to the next sample),
* :func:`fm_smems`  — the per-end-position backward search behind ``smems``.

Each returns exactly what its NumPy counterpart does (verified against them in
the tests and the existing FM cross-checks), so anchors are unchanged. Degrades
gracefully: without Numba the caller keeps the NumPy paths. The kernels release
the GIL (``nogil=True``), so seeding now parallelizes with the chaining /
extension kernels. Honours the shared ``GMF_NUMBA_SW`` kill switch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .numba_sw import numba_available  # shared probe + GMF_NUMBA_SW kill switch

__all__ = ["numba_available", "fm_rank", "fm_count", "fm_locate", "fm_smems"]

_KERNELS: Any | None = None


def _build_kernels() -> Any:
    from numba import njit

    @njit(nogil=True, inline="always")
    def _rank1(occ, bwt, occ_sample, idx, c):
        # Occ(c, idx) = count of c in bwt[:idx] via nearest checkpoint + scan.
        cp = idx // occ_sample
        r = occ[cp, c]
        base = cp * occ_sample
        for p in range(base, idx):
            if bwt[p] == c:
                r += 1
        return r

    @njit(cache=True, nogil=True, fastmath=False)
    def _rank(occ, bwt, occ_sample, indices, chars, out):
        for t in range(indices.shape[0]):
            out[t] = _rank1(occ, bwt, occ_sample, indices[t], chars[t])

    @njit(cache=True, nogil=True, fastmath=False)
    def _count(occ, bwt, C, occ_sample, nsa, pattern, n_code, out):
        lo = 0
        hi = nsa
        for k in range(pattern.shape[0] - 1, -1, -1):
            c = pattern[k]
            if c == n_code:
                out[0] = 0
                out[1] = 0
                return
            base = C[c]
            lo = base + _rank1(occ, bwt, occ_sample, lo, c)
            hi = base + _rank1(occ, bwt, occ_sample, hi, c)
            if hi <= lo:
                out[0] = 0
                out[1] = 0
                return
        out[0] = lo
        out[1] = hi

    @njit(cache=True, nogil=True, fastmath=False)
    def _locate(occ, bwt, C, occ_sample, indices, sa_mask, sa_rank, sa_values,
                sa_sample, out):
        for t in range(indices.shape[0]):
            cur = indices[t]
            steps = 0
            for _ in range(sa_sample + 1):
                if sa_mask[cur]:
                    break
                c = bwt[cur]
                cur = C[c] + _rank1(occ, bwt, occ_sample, cur, c)
                steps += 1
            out[t] = sa_values[sa_rank[cur]] + steps

    @njit(cache=True, nogil=True, fastmath=False)
    def _smems(occ, bwt, C, occ_sample, nsa, read_codes, n_code, left, ilo, ihi):
        m = read_codes.shape[0]
        for j in range(m):
            lo = 0
            hi = nsa
            cur = j
            lj = j + 1          # i_min(j); one past the end == "no match yet"
            ilo_j = 0
            ihi_j = nsa
            while cur >= 0 and read_codes[cur] != n_code:
                c = read_codes[cur]
                base = C[c]
                nlo = base + _rank1(occ, bwt, occ_sample, lo, c)
                nhi = base + _rank1(occ, bwt, occ_sample, hi, c)
                if nhi > nlo:
                    lo = nlo
                    hi = nhi
                    ilo_j = nlo
                    ihi_j = nhi
                    lj = cur
                    cur -= 1
                else:
                    break
            left[j] = lj
            ilo[j] = ilo_j
            ihi[j] = ihi_j

    class _Kernels:
        rank = staticmethod(_rank)
        count = staticmethod(_count)
        locate = staticmethod(_locate)
        smems = staticmethod(_smems)

    return _Kernels()


def _kernels() -> Any:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _build_kernels()
    return _KERNELS


def fm_rank(occ, bwt, occ_sample, indices, chars) -> np.ndarray:
    """Batched ``Occ(chars[t], indices[t])`` via the Numba kernel."""
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    chars = np.ascontiguousarray(chars, dtype=np.int64)
    out = np.empty(indices.shape[0], dtype=np.int64)
    _kernels().rank(occ, bwt, np.int64(occ_sample), indices, chars, out)
    return out


def fm_count(occ, bwt, C, occ_sample, nsa, pattern, n_code) -> tuple[int, int]:
    """SA interval ``[lo, hi)`` of one exact ``pattern`` (backward search)."""
    pattern = np.ascontiguousarray(pattern, dtype=np.int64)
    out = np.empty(2, dtype=np.int64)
    _kernels().count(occ, bwt, C, np.int64(occ_sample), np.int64(nsa), pattern,
                     np.int64(n_code), out)
    return int(out[0]), int(out[1])


def fm_locate(occ, bwt, C, occ_sample, indices, sa_mask, sa_rank, sa_values,
              sa_sample) -> np.ndarray:
    """Text offsets for SA rows ``indices`` (LF walk to the next sample)."""
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    out = np.empty(indices.shape[0], dtype=np.int64)
    _kernels().locate(occ, bwt, C, np.int64(occ_sample), indices, sa_mask,
                      sa_rank, sa_values, np.int64(sa_sample), out)
    return out


def fm_smems(occ, bwt, C, occ_sample, nsa, read_codes, n_code):
    """Per-end-position backward search behind ``smems``.

    Returns ``(left, interval_lo, interval_hi)``: for each read end position,
    ``left`` is ``i_min`` (one past the end when no match), and the interval is
    the SA range of ``read[left..j]``. Selection/super-maximality stay in the
    caller (cheap ``O(len)`` NumPy, once per read).
    """
    read_codes = np.ascontiguousarray(read_codes, dtype=np.int64)
    m = read_codes.shape[0]
    left = np.empty(m, dtype=np.int64)
    ilo = np.empty(m, dtype=np.int64)
    ihi = np.empty(m, dtype=np.int64)
    _kernels().smems(occ, bwt, C, np.int64(occ_sample), np.int64(nsa),
                     read_codes, np.int64(n_code), left, ilo, ihi)
    return left, ilo, ihi
