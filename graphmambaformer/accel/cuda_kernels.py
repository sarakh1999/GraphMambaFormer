"""CuPy ``RawKernel`` tier for the three DP-heavy alignment stages.

These kernels are the ``cuda_rawkernel`` tier of the acceleration stack. They
are compiled lazily on first use and are only ever reached on a CUDA host with
``cupy`` installed; :mod:`graphmambaformer.alignment` always carries a
torch implementation with identical semantics, which is what runs (and is
verified) on CPU / MPS.

Three kernels, matching the architecture's GPU feature list:

``kmer_lookup``
    Binary search of a GPU-resident sorted k-mer table — one thread per query.
``chain_dp``
    Anchor chaining DP: one block per read, threads cooperate over the
    predecessor lookback window. A single launch handles the whole batch.
``banded_sw``
    Batched banded affine Smith-Waterman. One block per pair, threads span the
    band, and the within-row gap recurrence is resolved with a shared-memory
    max-plus scan (see :func:`~graphmambaformer.alignment.extension.banded_affine_sw`
    for the derivation). Score-only: traceback runs on the torch tier for the
    single surviving candidate, which is far cheaper than emitting pointers for
    every candidate.

All wrappers take and return ``torch`` CUDA tensors and exchange memory with
CuPy through DLPack, so nothing is copied across the boundary.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch

from .backend import cupy_module

# Sentinel for "no predecessor" in the chaining traceback.
NO_PREDECESSOR = -1


# --------------------------------------------------------------------------- #
# Kernel sources
# --------------------------------------------------------------------------- #
_KMER_LOOKUP_SRC = r"""
// One thread per query k-mer. `table` is sorted ascending; `offsets` gives the
// half-open [start, end) slice of the position list for each table entry.
extern "C" __global__
void kmer_lookup(const long long* __restrict__ table,
                 const int* __restrict__ offsets,
                 const long long* __restrict__ queries,
                 const int n_table,
                 const int n_queries,
                 int* __restrict__ out_start,
                 int* __restrict__ out_count) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_queries) return;

    const long long key = queries[q];
    int lo = 0, hi = n_table;              // lower_bound
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (table[mid] < key) lo = mid + 1;
        else hi = mid;
    }

    if (lo < n_table && table[lo] == key) {
        out_start[q] = offsets[lo];
        out_count[q] = offsets[lo + 1] - offsets[lo];
    } else {
        out_start[q] = 0;
        out_count[q] = 0;
    }
}
"""


_CHAIN_DP_SRC = r"""
// Minimap2-style anchor chaining DP.
//
// One block per read. For anchor i the block evaluates every predecessor j in
// [i - lookback, i) in parallel, then reduces to the best (score, index).
// Anchors must be pre-sorted by reference end position.
//
// anchors layout: (B, A, 3) int32 -> (read_pos_end, ref_pos_end, reserved)
extern "C" __global__
void chain_dp(const int* __restrict__ anchors,
              const float* __restrict__ weights,      // (B, A)
              const int* __restrict__ n_anchors,   // (B,) valid anchors per read
              const float* __restrict__ bonus,     // (B, A, K) graph bonus, K=lookback
              const int max_anchors,
              const int lookback,
              const int max_gap,
              const float gap_open,
              const float gap_extend,
              const float log_coeff,
              const int use_bonus,
              float* __restrict__ f,               // (B, A) best chain score ending at i
              int* __restrict__ p) {               // (B, A) predecessor of i
    extern __shared__ char smem[];
    float* s_score = (float*)smem;                       // blockDim.x floats
    int*   s_index = (int*)(s_score + blockDim.x);       // blockDim.x ints

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int n = n_anchors[b];
    const long long base = (long long)b * max_anchors * 3;

    for (int i = 0; i < n; ++i) {
        const int qi = anchors[base + (long long)i * 3 + 0];
        const int ri = anchors[base + (long long)i * 3 + 1];
        const float wi = weights[(long long)b * max_anchors + i];

        float best = wi;   // start a fresh chain at i
        int best_j = -1;

        const int j_lo = max(0, i - lookback);
        for (int j = j_lo + tid; j < i; j += blockDim.x) {
            const int qj = anchors[base + (long long)j * 3 + 0];
            const int rj = anchors[base + (long long)j * 3 + 1];

            const int dq = qi - qj;
            const int dr = ri - rj;
            // Anchors must advance on both axes to be chainable.
            if (dq <= 0 || dr <= 0 || dq > max_gap || dr > max_gap) continue;

            // Overlap-aware anchor weight (minimap2 `alpha`).
            const float adv = fminf((float)min(dq, dr), wi);
            const int gap = abs(dr - dq);

            float penalty = 0.0f;
            if (gap > 0) {
                penalty = gap_open + gap_extend * (float)gap
                        + log_coeff * log2f((float)gap + 1.0f);
            }
            float sc = f[(long long)b * max_anchors + j] + adv - penalty;
            if (use_bonus) {
                sc += bonus[((long long)b * max_anchors + i) * lookback
                            + (j - (i - lookback) >= 0 ? j - (i - lookback) : 0)];
            }
            if (sc > best) { best = sc; best_j = j; }
        }

        s_score[tid] = best;
        s_index[tid] = best_j;
        __syncthreads();

        // Tree reduction to the block-wide best. Ties break to the lower index
        // so the result matches the sequential torch implementation exactly.
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other = s_score[tid + stride];
                if (other > s_score[tid] ||
                    (other == s_score[tid] && s_index[tid + stride] < s_index[tid])) {
                    s_score[tid] = other;
                    s_index[tid] = s_index[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            f[(long long)b * max_anchors + i] = s_score[0];
            p[(long long)b * max_anchors + i] = s_index[0];
        }
        __syncthreads();
    }
}
"""


_BANDED_SW_SRC = r"""
// Batched banded affine Smith-Waterman, score only.
//
// One block per (query, target) pair; `blockDim.x` spans the band. Band index d
// maps to target column j = i - half_band + d, so the previous row's diagonal
// predecessor is at d and its vertical predecessor at d + 1.
//
// The within-row gap recurrence
//     E[d] = max_{d' < d} ( M[d'] - open - (d - d' - 1) * extend )
// is an exclusive max-plus prefix scan of (M[d'] + d' * extend), done here with
// a Hillis-Steele scan in shared memory: log2(band) steps instead of a serial
// sweep.
extern "C" __global__
void banded_sw(const signed char* __restrict__ query,    // (B, M)
               const signed char* __restrict__ target,   // (B, N)
               const int* __restrict__ query_len,        // (B,)
               const int* __restrict__ target_len,       // (B,)
               const int* __restrict__ band_offset,      // (B,)
               const int max_query,
               const int max_target,
               const int half_band,
               const int band_width,
               const float match_score,
               const float mismatch_penalty,
               const float gap_open,
               const float gap_extend,
               float* __restrict__ out_score,            // (B,)
               int* __restrict__ out_query_end,          // (B,)
               int* __restrict__ out_target_end) {       // (B,)
    extern __shared__ char smem[];
    const int threads = blockDim.x;
    float* h_prev = (float*)smem;              // threads + 1 (guard column)
    float* f_prev = h_prev + (threads + 1);
    float* m_cur  = f_prev + (threads + 1);
    float* scan   = m_cur  + (threads + 1);

    const int b = blockIdx.x;
    const int d = threadIdx.x;
    const int m = query_len[b];
    const int n = target_len[b];

    const bool active = d < band_width;
    h_prev[d] = 0.0f;
    f_prev[d] = -1e30f;
    if (d == 0) { h_prev[band_width] = 0.0f; f_prev[band_width] = -1e30f; }
    __syncthreads();

    float best = 0.0f;
    int best_i = 0, best_j = 0;

    for (int i = 1; i <= m; ++i) {
        const int j = i + band_offset[b] - half_band + d;

        float m_val = -1e30f;
        float f_val = -1e30f;
        if (active && j >= 1 && j <= n) {
            const signed char qb = query[(long long)b * max_query + (i - 1)];
            const signed char tb = target[(long long)b * max_target + (j - 1)];
            // Negative codes mark padding / ambiguous bases: never a match.
            const float sub = (qb >= 0 && qb != 5 && qb == tb)
                            ? match_score : -mismatch_penalty;
            const float diag = h_prev[d] + sub;
            f_val = fmaxf(h_prev[d + 1] - gap_open, f_prev[d + 1] - gap_extend);
            m_val = fmaxf(0.0f, fmaxf(diag, f_val));
        }
        __syncthreads();

        // Exclusive max-plus prefix scan over (m_val + d * gap_extend).
        scan[d] = (m_val > -1e29f) ? (m_val + (float)d * gap_extend) : -1e30f;
        __syncthreads();
        for (int stride = 1; stride < threads; stride <<= 1) {
            float left = -1e30f;
            if (d >= stride) left = scan[d - stride];
            __syncthreads();
            if (left > scan[d]) scan[d] = left;
            __syncthreads();
        }
        // scan[] is now an inclusive prefix max; shift by one for exclusivity.
        const float excl = (d > 0) ? scan[d - 1] : -1e30f;
        __syncthreads();

        float h_val = -1e30f;
        if (active && j >= 1 && j <= n) {
            const float e_val = excl - gap_open - (float)(d - 1) * gap_extend;
            h_val = fmaxf(m_val, e_val);
            h_val = fmaxf(0.0f, h_val);
            if (h_val > best) { best = h_val; best_i = i; best_j = j; }
        } else {
            h_val = 0.0f;
        }
        __syncthreads();

        h_prev[d] = h_val;
        f_prev[d] = f_val;
        if (d == 0) { h_prev[band_width] = 0.0f; f_prev[band_width] = -1e30f; }
        __syncthreads();
    }

    // Reduce the per-thread best to a per-block best.
    scan[d] = best;
    m_cur[d] = (float)(best_i * (max_target + 1) + best_j);  // packed argmax
    __syncthreads();
    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (d < stride && scan[d + stride] > scan[d]) {
            scan[d] = scan[d + stride];
            m_cur[d] = m_cur[d + stride];
        }
        __syncthreads();
    }
    if (d == 0) {
        const int packed = (int)m_cur[0];
        out_score[b] = scan[0];
        out_query_end[b] = packed / (max_target + 1);
        out_target_end[b] = packed % (max_target + 1);
    }
}
"""


_WFA_DISTANCE_SRC = r"""
// Unit-cost wavefront distance. One thread owns one sequence pair; diagonals
// within that pair are stored in caller-allocated global scratch space.
extern "C" __global__
void wfa_distance(const signed char* __restrict__ query,
                  const signed char* __restrict__ target,
                  const int* __restrict__ query_len,
                  const int* __restrict__ target_len,
                  const int batch_size,
                  const int max_query,
                  const int max_target,
                  const int max_distance,
                  int* __restrict__ previous,
                  int* __restrict__ current,
                  int* __restrict__ out_distance) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    const int n = query_len[b];
    const int m = target_len[b];
    const int width = 2 * max_distance + 3;
    const int centre = max_distance + 1;
    int* prev = previous + (long long)b * width;
    int* cur = current + (long long)b * width;
    for (int t = 0; t < width; ++t) { prev[t] = -1; cur[t] = -1; }

    int reach = 0;
    while (reach < n && reach < m
           && query[(long long)b * max_query + reach] != 5
           && query[(long long)b * max_query + reach]
              == target[(long long)b * max_target + reach]) ++reach;
    prev[centre] = reach;
    const int final_k = n - m;
    if (final_k == 0 && reach >= n) { out_distance[b] = 0; return; }

    for (int score = 1; score <= max_distance; ++score) {
        for (int t = centre - score; t <= centre + score; ++t) cur[t] = -1;
        for (int k = -score; k <= score; ++k) {
            const int slot = centre + k;
            int sub = prev[slot] >= 0 ? prev[slot] + 1 : -1;
            int ins = prev[slot - 1] >= 0 ? prev[slot - 1] + 1 : -1;
            int del = prev[slot + 1];
            if (sub < 0 || sub > n || sub - k < 0 || sub - k > m) sub = -1;
            if (ins < 0 || ins > n || ins - k < 0 || ins - k > m) ins = -1;
            if (del < 0 || del > n || del - k < 0 || del - k > m) del = -1;
            int best = max(sub, max(ins, del));
            int col = best - k;
            if (best < 0) {
                cur[slot] = -1;
                continue;
            }
            while (best < n && col < m
                   && query[(long long)b * max_query + best] != 5
                   && query[(long long)b * max_query + best]
                      == target[(long long)b * max_target + col]) {
                ++best; ++col;
            }
            cur[slot] = best;
        }
        if (final_k >= -score && final_k <= score
            && cur[centre + final_k] >= n) {
            out_distance[b] = score;
            return;
        }
        int* swap = prev; prev = cur; cur = swap;
    }
    out_distance[b] = -1;
}
"""


_UNGAPPED_EXTEND_SRC = r"""
// GenomeWorks-style ungapped seed extension with the X-drop stop rule.
//
// One thread per seed. Starting from the seed position the thread walks right
// (inclusive of the seed) and left (from the base before the seed) along the
// seed's diagonal, with no gaps, accumulating +match / -mismatch. Each walk
// stops as soon as the running score falls more than `x_drop` below the best
// score seen so far, and reports the offset at which that best was reached.
// Bases in 1..4 are A/C/G/T; 5 (N), 0 (sentinel) and negatives never match.
extern "C" __global__
void ungapped_extend(const signed char* __restrict__ query,   // (M,)
                     const signed char* __restrict__ target,  // (N,)
                     const int* __restrict__ seed_q,          // (B,)
                     const int* __restrict__ seed_r,          // (B,)
                     const int batch_size,
                     const int m,
                     const int n,
                     const float match_score,
                     const float mismatch_penalty,
                     const float x_drop,
                     int* __restrict__ out_q_start,
                     int* __restrict__ out_q_end,
                     int* __restrict__ out_t_start,
                     int* __restrict__ out_t_end,
                     float* __restrict__ out_score) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    const int sq = seed_q[b];
    const int sr = seed_r[b];

    // Right extension (seed inclusive).
    float score = 0.0f, best_r = 0.0f;
    int best_off_r = 0;
    for (int i = 0; ; ++i) {
        const int qi = sq + i;
        const int ri = sr + i;
        if (qi >= m || ri >= n) break;
        const signed char a = query[qi];
        const signed char c = target[ri];
        score += (a == c && a >= 1 && a <= 4) ? match_score : -mismatch_penalty;
        if (score > best_r) { best_r = score; best_off_r = i + 1; }
        else if (best_r - score > x_drop) break;
    }

    // Left extension (strictly before the seed).
    score = 0.0f;
    float best_l = 0.0f;
    int best_off_l = 0;
    for (int i = 1; ; ++i) {
        const int qi = sq - i;
        const int ri = sr - i;
        if (qi < 0 || ri < 0) break;
        const signed char a = query[qi];
        const signed char c = target[ri];
        score += (a == c && a >= 1 && a <= 4) ? match_score : -mismatch_penalty;
        if (score > best_l) { best_l = score; best_off_l = i; }
        else if (best_l - score > x_drop) break;
    }

    out_q_start[b] = sq - best_off_l;
    out_q_end[b]   = sq + best_off_r;
    out_t_start[b] = sr - best_off_l;
    out_t_end[b]   = sr + best_off_r;
    out_score[b]   = best_l + best_r;
}
"""


# --------------------------------------------------------------------------- #
# Lazy compilation
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _kernel(name: str) -> Any | None:
    """Compile and cache one RawKernel; ``None`` when CuPy/CUDA is unavailable."""
    cp = cupy_module()
    if cp is None:
        return None
    source = {
        "kmer_lookup": _KMER_LOOKUP_SRC,
        "chain_dp": _CHAIN_DP_SRC,
        "banded_sw": _BANDED_SW_SRC,
        "wfa_distance": _WFA_DISTANCE_SRC,
        "ungapped_extend": _UNGAPPED_EXTEND_SRC,
    }[name]
    try:
        return cp.RawKernel(source, name, options=("--use_fast_math",))
    except Exception:
        return None


def kernels_available() -> bool:
    """True when the CuPy tier compiled successfully on this host."""
    return all(
        _kernel(n) is not None
        for n in ("kmer_lookup", "chain_dp", "banded_sw", "wfa_distance",
                  "ungapped_extend")
    )


def _as_cupy(tensor: torch.Tensor) -> Any:
    """Zero-copy view of a CUDA torch tensor as a CuPy array.

    The CUDA "current device" is *per host thread*. Under DDP each rank pins its
    main thread to ``cuda:<local_rank>`` via ``torch.cuda.set_device``, but this
    conversion runs inside the prefetch/stage ``ThreadPoolExecutor`` workers, and
    freshly spawned threads default back to device 0. Two things then go wrong on
    any rank whose device is not 0:

    * torch's DLPack exporter refuses to export a ``cuda:1`` tensor while the
      thread's current device is 0 (``BufferError: Can't export tensors on a
      different CUDA device index``); and
    * CuPy would import onto -- and later launch the kernel on -- the wrong device.

    So align *both* torch's and CuPy's current device to the tensor's device
    here. The setting persists on the worker thread, so it is also correct at
    kernel-launch time (every wrapper converts all its arguments through this
    helper before launching).
    """
    cp = cupy_module()
    idx = tensor.device.index
    if idx is not None:
        if torch.cuda.current_device() != idx:
            torch.cuda.set_device(idx)
        cp.cuda.Device(idx).use()
    return cp.from_dlpack(tensor.contiguous().detach())


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
def kmer_lookup(
    table: torch.Tensor,
    offsets: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up ``queries`` in a sorted GPU k-mer table.

    Returns ``(start, count)`` slices into the table's flat position list.
    """
    kernel = _kernel("kmer_lookup")
    if kernel is None:
        raise RuntimeError("CuPy k-mer lookup kernel unavailable")

    n_q = int(queries.numel())
    start = torch.empty(n_q, dtype=torch.int32, device=queries.device)
    count = torch.empty(n_q, dtype=torch.int32, device=queries.device)

    threads = 256
    blocks = (n_q + threads - 1) // threads
    # Scalars MUST be typed NumPy values, not Python int/float: CuPy marshals a
    # bare Python ``float`` as a C ``double``, so a kernel ``float`` param silently
    # reads 0 (and Python ``int`` sizing is not guaranteed to match ``int``).
    kernel(
        (blocks,),
        (threads,),
        (
            _as_cupy(table),
            _as_cupy(offsets),
            _as_cupy(queries),
            np.int32(table.numel()),
            np.int32(n_q),
            _as_cupy(start),
            _as_cupy(count),
        ),
    )
    return start, count


def chain_dp(
    anchors: torch.Tensor,
    n_anchors: torch.Tensor,
    lookback: int,
    max_gap: int,
    gap_open: float,
    gap_extend: float,
    log_coeff: float,
    bonus: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the batched chaining DP.

    Args:
        anchors: ``(B, A, 3)`` int32 ``(read_end, ref_end, reserved)``, sorted by
            reference end position.
        weights: optional ``(B, A)`` float32 anchor weights. When omitted, the
            legacy third anchors channel is used.
        n_anchors: ``(B,)`` int32 count of valid anchors per read.
        bonus: optional ``(B, A, lookback)`` float32 graph-distance bonus.

    Returns ``(f, p)``: best score ending at each anchor and its predecessor.
    """
    kernel = _kernel("chain_dp")
    if kernel is None:
        raise RuntimeError("CuPy chaining kernel unavailable")

    B, A, _ = anchors.shape
    if weights is None:
        weights = anchors[..., 2].to(torch.float32)
    if weights.shape != (B, A):
        raise ValueError("weights must have shape (B, A)")
    f = torch.zeros((B, A), dtype=torch.float32, device=anchors.device)
    p = torch.full((B, A), NO_PREDECESSOR, dtype=torch.int32, device=anchors.device)

    use_bonus = 1 if bonus is not None else 0
    if bonus is None:
        bonus = torch.zeros((1, 1, max(lookback, 1)), dtype=torch.float32, device=anchors.device)

    threads = min(256, max(32, _next_pow2(lookback)))
    shared = threads * (4 + 4)  # one float + one int per thread
    kernel(
        (B,),
        (threads,),
        (
            _as_cupy(anchors.to(torch.int32)),
            _as_cupy(weights.to(torch.float32)),
            _as_cupy(n_anchors.to(torch.int32)),
            _as_cupy(bonus),
            np.int32(A),
            np.int32(lookback),
            np.int32(max_gap),
            np.float32(gap_open),
            np.float32(gap_extend),
            np.float32(log_coeff),
            np.int32(use_bonus),
            _as_cupy(f),
            _as_cupy(p),
        ),
        shared_mem=shared,
    )
    return f, p.to(torch.long)


def banded_sw(
    query: torch.Tensor,
    target: torch.Tensor,
    query_len: torch.Tensor,
    target_len: torch.Tensor,
    half_band: int,
    match_score: float,
    mismatch_penalty: float,
    gap_open: float,
    gap_extend: float,
    band_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched banded affine Smith-Waterman scores.

    Args:
        query / target: ``(B, M)`` / ``(B, N)`` int8 base codes; negative values
            mark padding or ambiguous bases and never match.

    Returns ``(score, query_end, target_end)`` for the best local alignment.
    """
    kernel = _kernel("banded_sw")
    if kernel is None:
        raise RuntimeError("CuPy banded SW kernel unavailable")

    B, M = query.shape
    N = target.shape[1]
    band = 2 * half_band + 1
    threads = min(1024, _next_pow2(band))
    if band > 1024:
        raise RuntimeError(f"CUDA SW band width {band} exceeds 1024")
    if band_offset is None:
        band_offset = torch.zeros(B, dtype=torch.int32, device=query.device)

    score = torch.zeros(B, dtype=torch.float32, device=query.device)
    q_end = torch.zeros(B, dtype=torch.int32, device=query.device)
    t_end = torch.zeros(B, dtype=torch.int32, device=query.device)

    # h_prev, f_prev, m_cur, scan -> each (band + 1) floats.
    shared = 4 * (threads + 1) * 4
    kernel(
        (B,),
        (threads,),
        (
            _as_cupy(query.to(torch.int8)),
            _as_cupy(target.to(torch.int8)),
            _as_cupy(query_len.to(torch.int32)),
            _as_cupy(target_len.to(torch.int32)),
            _as_cupy(band_offset.to(torch.int32)),
            np.int32(M),
            np.int32(N),
            np.int32(half_band),
            np.int32(band),
            np.float32(match_score),
            np.float32(mismatch_penalty),
            np.float32(gap_open),
            np.float32(gap_extend),
            _as_cupy(score),
            _as_cupy(q_end),
            _as_cupy(t_end),
        ),
        shared_mem=shared,
    )
    return score, q_end.to(torch.long), t_end.to(torch.long)


def wfa_distance(
    query: torch.Tensor,
    target: torch.Tensor,
    query_len: torch.Tensor,
    target_len: torch.Tensor,
    max_distance: int,
) -> torch.Tensor:
    """Batched unit-cost WFA distance; ``-1`` means the bound was exceeded."""

    kernel = _kernel("wfa_distance")
    if kernel is None:
        raise RuntimeError("CuPy WFA kernel unavailable")
    B, M = query.shape
    N = target.shape[1]
    width = 2 * int(max_distance) + 3
    previous = torch.empty((B, width), dtype=torch.int32, device=query.device)
    current = torch.empty_like(previous)
    distance = torch.full((B,), -1, dtype=torch.int32, device=query.device)
    threads = 128
    blocks = (B + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            _as_cupy(query.to(torch.int8)),
            _as_cupy(target.to(torch.int8)),
            _as_cupy(query_len.to(torch.int32)),
            _as_cupy(target_len.to(torch.int32)),
            np.int32(B),
            np.int32(M),
            np.int32(N),
            np.int32(max_distance),
            _as_cupy(previous),
            _as_cupy(current),
            _as_cupy(distance),
        ),
    )
    return distance


def ungapped_extend(
    query: torch.Tensor,
    target: torch.Tensor,
    seed_query: torch.Tensor,
    seed_target: torch.Tensor,
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    x_drop: float = 600.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ungapped X-drop seed extension (GenomeWorks ``cudaextender``).

    Args:
        query / target: 1-D int8 base-code tensors (``A C G T`` -> ``1..4``,
            ``N`` -> 5, negatives = padding; only ``1..4`` can match).
        seed_query / seed_target: ``(B,)`` int32 seed positions, one per seed.

    Returns ``(query_start, query_end, target_start, target_end, score)``, each
    ``(B,)``, describing the maximal-scoring gap-free segment through each seed.
    """
    kernel = _kernel("ungapped_extend")
    if kernel is None:
        raise RuntimeError("CuPy ungapped-extend kernel unavailable")

    B = int(seed_query.numel())
    M = int(query.numel())
    N = int(target.numel())
    dev = query.device
    q_start = torch.empty(B, dtype=torch.int32, device=dev)
    q_end = torch.empty(B, dtype=torch.int32, device=dev)
    t_start = torch.empty(B, dtype=torch.int32, device=dev)
    t_end = torch.empty(B, dtype=torch.int32, device=dev)
    score = torch.empty(B, dtype=torch.float32, device=dev)

    threads = 128
    blocks = (B + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            _as_cupy(query.to(torch.int8)),
            _as_cupy(target.to(torch.int8)),
            _as_cupy(seed_query.to(torch.int32)),
            _as_cupy(seed_target.to(torch.int32)),
            np.int32(B),
            np.int32(M),
            np.int32(N),
            np.float32(match),
            np.float32(mismatch),
            np.float32(x_drop),
            _as_cupy(q_start),
            _as_cupy(q_end),
            _as_cupy(t_start),
            _as_cupy(t_end),
            _as_cupy(score),
        ),
    )
    return (
        q_start.to(torch.long),
        q_end.to(torch.long),
        t_start.to(torch.long),
        t_end.to(torch.long),
        score,
    )
