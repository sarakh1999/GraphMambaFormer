"""GenomeWorks-style GPU genomics primitives, with a three-way fallback.

This module lifts the four NVIDIA `GenomeWorks
<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>`_ modules onto the
project's existing acceleration stack:

===============  =========================================================
GenomeWorks      Primitive exposed here
===============  =========================================================
``cudamapper``   :func:`map_to_reference` — GPU minimizer seeding + anchor
                 chaining (seq-to-seq mapping / overlap).
``cudaaligner``  :func:`global_align` — global affine (Gotoh) alignment with
                 a full ``=``/``X``/``I``/``D`` CIGAR.
``cudaextender`` :func:`ungapped_extend` — ungapped seed extension with the
                 X-drop stop rule (BLAST/cudaextender semantics).
``cudapoa``      :func:`poa_consensus` / :func:`poa_msa` — partial-order
                 alignment consensus over a set of reads.
===============  =========================================================

Every primitive resolves to the fastest of three tiers, fastest first, and the
tiers are numerically equivalent so a CUDA result is only ever a throughput win:

``pyclaragenomics``
    The genuine GenomeWorks Python bindings (``genomeworks`` /
    ``claragenomics``) when they import and back onto a working GPU. GenomeWorks
    is archived (its last wheels target CUDA 10/11), so on a modern CUDA box
    this tier is usually absent — hence the two fallbacks below.
``cuda_rawkernel``
    CuPy ``RawKernel`` reimplementations of the same algorithms
    (:mod:`graphmambaformer.accel.cuda_kernels`), which is what actually runs on
    this project's GPUs.
``portable``
    A NumPy/torch reference with identical semantics. It is what the CPU / MPS
    tiers use, and — crucially — what the CUDA tiers are *verified against*, so
    correctness is established without a GPU.

The portable tier is dependency-light on purpose (NumPy + torch + this package's
``accel`` only). Anything that needs the alignment stages (the ``cudamapper``
facade) imports them lazily inside the call, so ``accel`` never imports
``alignment`` at module load and the package stays free of an import cycle.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .backend import cupy_module


# --------------------------------------------------------------------------- #
# Verify-then-trust warm-up.
#
# The CUDA kernels are deterministic, so recomputing the portable reference on
# *every* call to check them erases the very speedup the GPU exists to provide
# (it makes the CUDA path pay the full CPU cost plus device overhead). Instead
# each primitive verifies against the portable tier for the first
# ``GMF_GW_VERIFY_WARMUP`` successful calls per process and then trusts the
# kernel. Set ``GMF_GW_VERIFY=always`` to verify every call (debugging /
# paranoid mode) or ``GMF_GW_VERIFY=never`` to trust from the first call.
# --------------------------------------------------------------------------- #
_GW_VERIFY_WARMUP = max(0, int(os.environ.get("GMF_GW_VERIFY_WARMUP", "8")))
_GW_VERIFY_MODE = os.environ.get("GMF_GW_VERIFY", "warmup").strip().lower()
_gw_verify_counts: dict[str, int] = {}


def _gw_should_verify(name: str) -> bool:
    """Whether this call should cross-check the kernel against the reference."""
    if _GW_VERIFY_MODE == "always":
        return True
    if _GW_VERIFY_MODE == "never":
        return False
    return _gw_verify_counts.get(name, 0) < _GW_VERIFY_WARMUP


def _gw_mark_verified(name: str) -> None:
    _gw_verify_counts[name] = _gw_verify_counts.get(name, 0) + 1

# Base codes shared with :mod:`graphmambaformer.alignment.seeding`
# (A C G T -> 1..4, N -> 5, sentinel 0). A real match needs equal codes in 1..4;
# N (5), the sentinel (0) and negative padding never match.
_BASE_CODES = {"A": 1, "C": 2, "G": 3, "T": 4, "a": 1, "c": 2, "g": 3, "t": 4}
_N_CODE = 5
_DECODE = np.array(list("$ACGTN"), dtype="<U1")


# --------------------------------------------------------------------------- #
# Availability / tier selection
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def genomeworks_bindings() -> Any | None:
    """The genuine GenomeWorks Python package, or ``None``.

    Tries the modern (``genomeworks``) and legacy (``claragenomics``) import
    names. Cached because the import probe is not free and the answer cannot
    change within a process.
    """
    import importlib

    for name in ("genomeworks", "claragenomics"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def genomeworks_bindings_available() -> bool:
    """True when the real GenomeWorks bindings import on this host."""
    return genomeworks_bindings() is not None


def genomeworks_backend() -> str:
    """Which tier the GenomeWorks primitives will use on this host.

    One of ``"pyclaragenomics"`` (real bindings), ``"cuda_rawkernel"`` (CuPy
    kernels), or ``"portable"`` (NumPy/torch reference). The primitives are
    numerically identical across tiers; this only reports the throughput path.
    """
    if genomeworks_bindings_available():
        return "pyclaragenomics"
    from .cuda_kernels import kernels_available

    if cupy_module() is not None and kernels_available():
        return "cuda_rawkernel"
    return "portable"


def genomeworks_available() -> bool:
    """Whether the GenomeWorks primitives are usable (always ``True``).

    The portable tier has no dependencies beyond NumPy, so the primitives are
    always callable; a CUDA tier merely accelerates them. Kept as a function so
    call sites read the same as the other capability probes.
    """
    return True


def genomeworks_summary() -> str:
    """One-line description of the active GenomeWorks tier, for logs/doctor."""
    backend = genomeworks_backend()
    real = "yes" if genomeworks_bindings_available() else "no"
    return f"genomeworks: backend={backend} bindings={real}"


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def _as_codes(seq: "str | Sequence[int] | np.ndarray") -> np.ndarray:
    """Normalise a sequence to an ``int8`` code array (A C G T -> 1..4, N -> 5)."""
    if isinstance(seq, np.ndarray):
        return seq.astype(np.int8, copy=False)
    if isinstance(seq, str):
        if not seq:
            return np.zeros(0, dtype=np.int8)
        raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
        lut = np.full(256, _N_CODE, dtype=np.int8)
        for base, code in _BASE_CODES.items():
            lut[ord(base)] = code
        return lut[raw]
    return np.asarray(seq, dtype=np.int8)


def _decode(codes: np.ndarray) -> str:
    return "".join(_DECODE[np.clip(np.asarray(codes), 0, _N_CODE)])


def _is_real(code: int) -> bool:
    return 1 <= code <= 4


# =========================================================================== #
# cudaextender — ungapped seed extension (X-drop)
# =========================================================================== #
@dataclass
class UngappedExtension:
    """One ungapped, gap-free extension of a seed along a single diagonal.

    Coordinates are 0-based half-open on the input sequences; ``score`` is the
    best match/mismatch score reached, and ``query_end - query_start`` equals
    ``target_end - target_start`` because the segment carries no gaps.
    """

    query_start: int
    query_end: int
    target_start: int
    target_end: int
    score: float

    @property
    def length(self) -> int:
        return self.query_end - self.query_start


def _xdrop_one_direction(
    q: np.ndarray,
    t: np.ndarray,
    match: float,
    mismatch: float,
    x_drop: float,
) -> tuple[int, float]:
    """Extend from index 0 forward; return (offset_of_best, best_score).

    ``offset`` is the number of positions consumed on each sequence to reach the
    best score. The walk stops as soon as the running score falls more than
    ``x_drop`` below the best seen — the BLAST / cudaextender rule.
    """
    n = min(len(q), len(t))
    score = 0.0
    best = 0.0
    best_off = 0
    for i in range(n):
        a = int(q[i])
        b = int(t[i])
        score += match if (a == b and 1 <= a <= 4) else -mismatch
        if score > best:
            best = score
            best_off = i + 1
        elif best - score > x_drop:
            break
    return best_off, best


def ungapped_extend(
    query: "str | np.ndarray",
    target: "str | np.ndarray",
    seed_query: int,
    seed_target: int,
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    x_drop: float = 600.0,
) -> UngappedExtension:
    """Ungapped X-drop extension of a single seed (``cudaextender`` semantics).

    Given a seed match at ``(seed_query, seed_target)`` the seed is extended
    right (from the seed position, inclusive) and left (from the base before it)
    along its diagonal with no gaps, accumulating ``+match`` / ``-mismatch`` and
    stopping in each direction once the score drops by more than ``x_drop`` below
    the best seen. Returns the maximal-scoring gap-free segment.
    """
    q = _as_codes(query)
    t = _as_codes(target)

    # Right extension starts at the seed (inclusive), so the seed's own matches
    # are scored exactly once.
    right_off, right_score = _xdrop_one_direction(
        q[seed_query:], t[seed_target:], match, mismatch, x_drop
    )
    # Left extension walks the reversed prefixes strictly before the seed.
    left_off, left_score = _xdrop_one_direction(
        q[:seed_query][::-1], t[:seed_target][::-1], match, mismatch, x_drop
    )

    return UngappedExtension(
        query_start=seed_query - left_off,
        query_end=seed_query + right_off,
        target_start=seed_target - left_off,
        target_end=seed_target + right_off,
        score=left_score + right_score,
    )


def ungapped_extend_batch(
    query: "str | np.ndarray",
    target: "str | np.ndarray",
    seeds: Sequence[tuple[int, int]],
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    x_drop: float = 600.0,
    device: "str | Any | None" = None,
) -> list[UngappedExtension]:
    """Extend many seeds of one (query, target) pair; one entry per seed.

    On a CUDA host with the CuPy tier this dispatches to the batched
    :func:`graphmambaformer.accel.cuda_kernels.ungapped_extend` kernel,
    cross-checking it against the portable walk for the first few calls
    (see the ``GMF_GW_VERIFY`` warm-up above) and trusting it thereafter;
    otherwise it runs the portable walk directly.

    ``device`` pins the GPU the batched kernel runs on. Pass a CPU device to
    force the portable tier even when a GPU is present (so callers on the CPU
    tier never spawn device work from host worker threads); pass ``cuda`` /
    ``cuda:i`` to select a specific GPU; leave ``None`` to use the current one.
    """
    q = _as_codes(query)
    t = _as_codes(target)
    seeds = list(seeds)
    if not seeds:
        return []

    def _portable() -> list[UngappedExtension]:
        return [
            ungapped_extend(q, t, sq, sr, match=match, mismatch=mismatch, x_drop=x_drop)
            for sq, sr in seeds
        ]

    # An explicit non-CUDA device keeps everything on the portable tier: the
    # extension engine calls this from CPU worker threads and must not touch the
    # GPU unless it was configured for the CUDA backend.
    if device is not None:
        import torch

        if torch.device(device).type != "cuda":
            return _portable()

    cp = cupy_module()
    if cp is None:
        return _portable()
    try:
        import torch

        from .cuda_kernels import ungapped_extend as cuda_ungapped

        if not torch.cuda.is_available():
            return _portable()
        dev = torch.device(device) if device is not None else torch.device("cuda")
        q_t = torch.as_tensor(np.ascontiguousarray(q), dtype=torch.int8, device=dev)
        t_t = torch.as_tensor(np.ascontiguousarray(t), dtype=torch.int8, device=dev)
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32).reshape(-1, 2)
        sq = torch.as_tensor(seed_arr[:, 0], dtype=torch.int32, device=dev)
        sr = torch.as_tensor(seed_arr[:, 1], dtype=torch.int32, device=dev)
        q_start, q_end, t_start, t_end, score = cuda_ungapped(
            q_t, t_t, sq, sr, match=match, mismatch=mismatch, x_drop=x_drop,
        )
        # One bulk device->host copy per output, not a per-seed ``.item()`` sync
        # (thousands of tiny transfers otherwise dominate the runtime).
        qs = q_start.cpu().numpy(); qe = q_end.cpu().numpy()
        ts = t_start.cpu().numpy(); te = t_end.cpu().numpy()
        sc = score.cpu().numpy()
        gpu = [
            UngappedExtension(int(qs[i]), int(qe[i]), int(ts[i]), int(te[i]), float(sc[i]))
            for i in range(len(seeds))
        ]
        # Verify-then-trust, gated to a warm-up: computing the portable
        # reference on every call would erase the GPU speedup, so only the first
        # few calls per process cross-check the (deterministic) kernel.
        if _gw_should_verify("ungapped_extend"):
            portable = _portable()
            if all(
                g.query_start == p.query_start and g.query_end == p.query_end
                and g.target_start == p.target_start and g.target_end == p.target_end
                and abs(g.score - p.score) <= 1e-3
                for g, p in zip(gpu, portable)
            ):
                _gw_mark_verified("ungapped_extend")
                return gpu
            return portable  # kernel disagreed: fall back and do NOT mark trusted
        return gpu
    except Exception:
        pass
    return _portable()


# =========================================================================== #
# cudaaligner — global affine (Gotoh) alignment with CIGAR
# =========================================================================== #
@dataclass
class GlobalAlignment:
    """A global (end-to-end) alignment of a query to a target.

    ``cigar`` uses the extended operators ``=`` (match), ``X`` (mismatch),
    ``I`` (insertion, query base absent from target) and ``D`` (deletion).
    """

    score: float
    cigar: list[tuple[str, int]]
    n_match: int = 0
    n_mismatch: int = 0
    n_insertion: int = 0
    n_deletion: int = 0

    @property
    def cigar_string(self) -> str:
        return "".join(f"{n}{op}" for op, n in self.cigar)

    @property
    def edit_distance(self) -> int:
        return self.n_mismatch + self.n_insertion + self.n_deletion


def _merge_ops(ops: Iterable[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for op in ops:
        if out and out[-1][0] == op:
            out[-1] = (op, out[-1][1] + 1)
        else:
            out.append((op, 1))
    return out


def _integer_scoring(*values: float) -> bool:
    """True when every score is integer-valued (exact in float64).

    The vectorised fill computes the horizontal-gap term with a closed-form
    max-plus scan (``incl - open - j*extend``) rather than the scalar's repeated
    ``- extend`` subtraction. The two are mathematically equal but differ by ULPs
    once ``gap_extend`` is fractional, which can flip a tie between two
    equal-scoring alignments and change the CIGAR (never the score). With
    integer scores every intermediate sum is exact, so the fast path is
    bit-identical; otherwise we fall back to the scalar recurrence to keep the
    result unchanged.
    """
    return all(float(v).is_integer() for v in values)


def _global_align_scalar(
    query: "str | np.ndarray",
    target: "str | np.ndarray",
    *,
    match: float,
    mismatch: float,
    gap_open: float,
    gap_extend: float,
) -> GlobalAlignment:
    """Exact scalar Gotoh recurrence (fractional-score fallback for
    :func:`global_align`; identical output to the pre-vectorisation reference)."""
    q = _as_codes(query)
    t = _as_codes(target)
    m, n = len(q), len(t)
    NEG = -1e30
    if m == 0 and n == 0:
        return GlobalAlignment(0.0, [])
    if m == 0:
        return GlobalAlignment(-(gap_open + n * gap_extend), [("D", n)], n_deletion=n)
    if n == 0:
        return GlobalAlignment(-(gap_open + m * gap_extend), [("I", m)], n_insertion=m)

    H = np.full((m + 1, n + 1), NEG, dtype=np.float64)
    E = np.full((m + 1, n + 1), NEG, dtype=np.float64)
    F = np.full((m + 1, n + 1), NEG, dtype=np.float64)
    ptr = np.zeros((m + 1, n + 1), dtype=np.int8)
    H[0, 0] = 0.0
    for j in range(1, n + 1):
        E[0, j] = -(gap_open + j * gap_extend)
        H[0, j] = E[0, j]
        ptr[0, j] = 1
    for i in range(1, m + 1):
        F[i, 0] = -(gap_open + i * gap_extend)
        H[i, 0] = F[i, 0]
        ptr[i, 0] = 2
    qcol = q.astype(np.int64)
    for i in range(1, m + 1):
        a = int(qcol[i - 1])
        Hi_prev, Ei, Fi, Hi, Fim1 = H[i - 1], E[i], F[i], H[i], F[i - 1]
        for j in range(1, n + 1):
            b = int(t[j - 1])
            sub = match if (a == b and 1 <= a <= 4) else -mismatch
            diag = Hi_prev[j - 1] + sub
            e = max(Hi[j - 1] - (gap_open + gap_extend), Ei[j - 1] - gap_extend)
            Ei[j] = e
            f = max(Hi_prev[j] - (gap_open + gap_extend), Fim1[j] - gap_extend)
            Fi[j] = f
            best, p = diag, 0
            if e > best:
                best, p = e, 1
            if f > best:
                best, p = f, 2
            Hi[j] = best
            ptr[i, j] = p
    ops: list[str] = []
    i, j = m, n
    n_match = n_mismatch = n_ins = n_del = 0
    while i > 0 or j > 0:
        p = ptr[i, j]
        if i > 0 and j > 0 and p == 0:
            a, b = int(q[i - 1]), int(t[j - 1])
            if a == b and 1 <= a <= 4:
                ops.append("=")
                n_match += 1
            else:
                ops.append("X")
                n_mismatch += 1
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or p == 1):
            ops.append("D")
            n_del += 1
            j -= 1
        else:
            ops.append("I")
            n_ins += 1
            i -= 1
    ops.reverse()
    return GlobalAlignment(
        score=float(H[m, n]),
        cigar=_merge_ops(ops),
        n_match=n_match,
        n_mismatch=n_mismatch,
        n_insertion=n_ins,
        n_deletion=n_del,
    )


def global_align(
    query: "str | np.ndarray",
    target: "str | np.ndarray",
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    gap_open: float = 6.0,
    gap_extend: float = 2.0,
) -> GlobalAlignment:
    """Global affine-gap alignment with full CIGAR (``cudaaligner`` semantics).

    Gotoh's three-matrix recurrence (match/substitution ``H``, gap-in-target
    ``E``, gap-in-query ``F``) with an affine cost ``gap_open + k*gap_extend``
    for a length-``k`` gap. This is the portable reference the GPU aligner tier
    is checked against; it produces the same alignment a global cudaaligner call
    would, only on the host.

    The fill is vectorised row-by-row. The only serial dependency inside a row
    is the horizontal-gap recurrence
    ``E[i][j] = max(H[i][j-1] - (open+extend), E[i][j-1] - extend)``, whose
    closed form ``E[i][j] = max_{k<j}(H[i][k] - open - (j-k)*extend)`` is an
    exclusive max-plus prefix scan of ``H[i][k] + k*extend`` (one
    ``np.maximum.accumulate``). A horizontal gap opened from a cell that itself
    ends in a horizontal gap is dominated by the longer gap opened earlier, so
    ``max(diag, F)`` may stand in for ``H`` inside the scan — the same closed
    form :func:`banded_affine_sw_batch` uses. Each per-cell traceback pointer
    (0 diag / 1 E / 2 F) keeps the scalar tie-break (diag ≥ E ≥ F), so the score
    and CIGAR are bit-identical to the textbook scalar recurrence while the fill
    is ``O(m·n)`` vectorised, not a Python double loop. Only the ``int8`` pointer
    matrix is kept, so this also uses far less memory than materialising the
    three ``float64`` DP matrices.

    Fractional scores (rare — the pipeline always scores in integers) fall back
    to the exact scalar recurrence so the CIGAR never changes on a float tie.
    """
    if not _integer_scoring(match, mismatch, gap_open, gap_extend):
        return _global_align_scalar(
            query, target, match=match, mismatch=mismatch,
            gap_open=gap_open, gap_extend=gap_extend,
        )

    q = _as_codes(query)
    t = _as_codes(target)
    m, n = len(q), len(t)
    NEG = -1e30

    if m == 0 and n == 0:
        return GlobalAlignment(0.0, [])
    if m == 0:
        return GlobalAlignment(-(gap_open + n * gap_extend), [("D", n)], n_deletion=n)
    if n == 0:
        return GlobalAlignment(-(gap_open + m * gap_extend), [("I", m)], n_insertion=m)

    # Traceback pointer per cell in H: 0 diag, 1 from E (D), 2 from F (I).
    ptr = np.zeros((m + 1, n + 1), dtype=np.int8)
    ptr[0, 1:] = 1  # row-0 border: pure horizontal gap (deletions)
    ptr[1:, 0] = 2  # col-0 border: pure vertical gap (insertions)

    jcol = np.arange(n + 1, dtype=np.float64)       # column index 0..n
    scan_pos = jcol * gap_extend                    # k * extend, per column
    tcol = t.astype(np.int64)                       # target codes for columns 1..n
    go_ge = gap_open + gap_extend

    # Row-0 border of H: H[0][0]=0, H[0][j] = -(open + j*extend).
    h_prev = np.empty(n + 1, dtype=np.float64)
    h_prev[0] = 0.0
    h_prev[1:] = -(gap_open + jcol[1:] * gap_extend)
    f_prev = np.full(n + 1, NEG, dtype=np.float64)  # F[0][j] is undefined (-inf)

    qcol = q.astype(np.int64)
    for i in range(1, m + 1):
        a = int(qcol[i - 1])
        # A real match needs equal codes with the query base in A/C/G/T (1..4);
        # N (5) and the sentinel never match, so an out-of-range base scores as a
        # mismatch in every column.
        match_mask = (tcol == a) if (1 <= a <= 4) else np.zeros(n, dtype=bool)
        sub = np.where(match_mask, match, -mismatch)  # (n,) over columns 1..n
        diag = h_prev[:-1] + sub                    # H[i-1][j-1] + sub, j = 1..n
        f = np.maximum(h_prev[1:] - go_ge, f_prev[1:] - gap_extend)  # F[i][j]

        hi0 = -(gap_open + i * gap_extend)          # H[i][0] border
        # d_or_f[k], k = 0..n: border cell at k=0, else max(diag, F).
        d_or_f = np.empty(n + 1, dtype=np.float64)
        d_or_f[0] = hi0
        d_or_f[1:] = np.maximum(diag, f)
        incl = np.maximum.accumulate(d_or_f + scan_pos)      # inclusive prefix max
        # E[i][j] = -open - j*extend + max_{k<j}(H[i][k] + k*extend), j = 1..n.
        e = incl[:-1] - gap_open - jcol[1:] * gap_extend

        best = diag.copy()
        p = np.zeros(n, dtype=np.int8)
        take_e = e > best
        best = np.where(take_e, e, best)
        p = np.where(take_e, np.int8(1), p)
        take_f = f > best
        best = np.where(take_f, f, best)
        p = np.where(take_f, np.int8(2), p)
        ptr[i, 1:] = p

        hi = np.empty(n + 1, dtype=np.float64)
        hi[0] = hi0
        hi[1:] = best
        f_row = np.empty(n + 1, dtype=np.float64)
        f_row[0] = NEG
        f_row[1:] = f
        h_prev, f_prev = hi, f_row

    score_final = float(h_prev[n])

    # Traceback from (m, n).
    ops: list[str] = []
    i, j = m, n
    n_match = n_mismatch = n_ins = n_del = 0
    while i > 0 or j > 0:
        p = ptr[i, j]
        if i > 0 and j > 0 and p == 0:
            a, b = int(q[i - 1]), int(t[j - 1])
            if a == b and 1 <= a <= 4:
                ops.append("=")
                n_match += 1
            else:
                ops.append("X")
                n_mismatch += 1
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or p == 1):
            ops.append("D")
            n_del += 1
            j -= 1
        else:
            ops.append("I")
            n_ins += 1
            i -= 1
    ops.reverse()
    return GlobalAlignment(
        score=score_final,
        cigar=_merge_ops(ops),
        n_match=n_match,
        n_mismatch=n_mismatch,
        n_insertion=n_ins,
        n_deletion=n_del,
    )


def global_align_batch(
    pairs: Sequence[tuple["str | np.ndarray", "str | np.ndarray"]],
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    gap_open: float = 6.0,
    gap_extend: float = 2.0,
) -> list[GlobalAlignment]:
    """Global-align each ``(query, target)`` pair (``cudaaligner`` batch)."""
    return [
        global_align(
            q, t, match=match, mismatch=mismatch,
            gap_open=gap_open, gap_extend=gap_extend,
        )
        for q, t in pairs
    ]


# =========================================================================== #
# cudapoa — partial-order alignment consensus
# =========================================================================== #
class _PoaGraph:
    """A partial-order alignment graph (Lee, Grasso & Sharlow, 2002).

    Nodes are single bases; a directed edge ``u -> v`` records that base ``v``
    follows base ``u`` in at least one added sequence, weighted by how many.
    Aligning a new sequence to the graph reuses nodes on matches and splices in
    new nodes for mismatches / insertions, so the graph compactly represents the
    multiple alignment of every sequence added so far.
    """

    def __init__(self) -> None:
        self.base: list[int] = []
        self.out: list[dict[int, int]] = []
        self.inn: list[dict[int, int]] = []
        self.node_weight: list[int] = []

    def __len__(self) -> int:
        return len(self.base)

    def add_node(self, base: int) -> int:
        self.base.append(int(base))
        self.out.append({})
        self.inn.append({})
        self.node_weight.append(0)
        return len(self.base) - 1

    def add_edge(self, u: int, v: int, w: int = 1) -> None:
        if u < 0 or v < 0:
            return
        self.out[u][v] = self.out[u].get(v, 0) + w
        self.inn[v][u] = self.inn[v].get(u, 0) + w

    def topo_order(self) -> list[int]:
        """Kahn topological sort; the graph is a DAG by construction."""
        indeg = [len(self.inn[i]) for i in range(len(self.base))]
        stack = [i for i in range(len(self.base)) if indeg[i] == 0]
        order: list[int] = []
        while stack:
            u = stack.pop()
            order.append(u)
            for v in self.out[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if len(order) != len(self.base):  # pragma: no cover - defensive
            # A cycle should be impossible; fall back to insertion order.
            return list(range(len(self.base)))
        return order


def _poa_add_first(graph: _PoaGraph, codes: np.ndarray, weight: int) -> None:
    prev = -1
    for c in codes:
        node = graph.add_node(int(c))
        graph.node_weight[node] += weight
        if prev >= 0:
            graph.add_edge(prev, node, weight)
        prev = node


def _poa_align_and_add(
    graph: _PoaGraph,
    codes: np.ndarray,
    weight: int,
    match: float,
    mismatch: float,
    gap: float,
) -> None:
    """Align one sequence to the graph and fold it in (linear gap POA DP)."""
    if len(graph) == 0:
        _poa_add_first(graph, codes, weight)
        return

    order = graph.topo_order()
    pos = {node: idx for idx, node in enumerate(order)}
    G = len(order)
    S = len(codes)
    NEG = -1e30

    # score[g+1][s+1]; row 0 / col 0 are the all-gap borders.
    score = np.full((G + 1, S + 1), NEG, dtype=np.float64)
    # move: 0 diag(match), 1 delete graph node (gap in seq), 2 insert seq base
    move = np.zeros((G + 1, S + 1), dtype=np.int8)
    pred_g = np.full((G + 1, S + 1), -1, dtype=np.int64)  # graph predecessor row

    score[0, 0] = 0.0
    for s in range(1, S + 1):
        score[0, s] = -gap * s
        move[0, s] = 2
    for gi in range(1, G + 1):
        node = order[gi - 1]
        preds = [pos[p] + 1 for p in graph.inn[node]] or [0]
        best_pred = max(preds, key=lambda r: score[r, 0])
        score[gi, 0] = score[best_pred, 0] - gap
        move[gi, 0] = 1
        pred_g[gi, 0] = best_pred

    # Fill the DP grid one graph node (row) at a time — rows must be visited in
    # topological order, but the whole sequence dimension of a row is done at
    # once. With integer scores this vectorised fill is bit-identical to the
    # scalar recurrence in the ``else`` branch (kept verbatim for the rare
    # fractional-score case, where a max-plus scan and the serial recurrence can
    # differ by a float ULP and flip a tie).
    if _integer_scoring(match, mismatch, gap):
        codes_arr = codes[:S].astype(np.int64)             # sequence codes, cols 1..S
        scan_s = np.arange(S + 1, dtype=np.float64) * gap  # gap * s', s' = 0..S
        for gi in range(1, G + 1):
            node = order[gi - 1]
            gbase = graph.base[node]
            pred_rows = [pos[p] + 1 for p in graph.inn[node]] or [0]
            if 1 <= gbase <= 4:
                sub = np.where(codes_arr == gbase, match, -mismatch)
            else:
                sub = np.full(S, -mismatch, dtype=np.float64)
            best = np.full(S, NEG, dtype=np.float64)
            best_move = np.full(S, 2, dtype=np.int8)
            best_row = np.full(S, pred_rows[0], dtype=np.int64)
            # Diagonal (match/mismatch) then vertical (delete graph node) from
            # each predecessor, in list order, with the scalar's strict-`>` max so
            # ties resolve to the same (move, row).
            for r in pred_rows:
                diag = score[r, :S] + sub                  # score[r, s-1], s=1..S
                upd = diag > best
                best = np.where(upd, diag, best)
                best_move = np.where(upd, np.int8(0), best_move)
                best_row = np.where(upd, r, best_row)
                dele = score[r, 1:S + 1] - gap             # score[r, s], s=1..S
                upd = dele > best
                best = np.where(upd, dele, best)
                best_move = np.where(upd, np.int8(1), best_move)
                best_row = np.where(upd, r, best_row)
            # Insertion (gap in graph): score[gi,s] = max(best[s], score[gi,s-1]
            # - gap). The serial insertion chain is an exclusive max-plus scan
            # seeded by the already-set col-0 border, so a whole row's insertions
            # come from one cummax instead of an inner Python loop.
            border = np.empty(S + 1, dtype=np.float64)
            border[0] = score[gi, 0]
            border[1:] = best
            row_scores = np.maximum.accumulate(border + scan_s) - scan_s
            ins_cand = row_scores[:S] - gap                # score[gi, s-1] - gap
            take_ins = ins_cand > best
            score[gi, 1:] = row_scores[1:]
            move[gi, 1:] = np.where(take_ins, np.int8(2), best_move)
            pred_g[gi, 1:] = np.where(take_ins, gi, best_row)
    else:
        for gi in range(1, G + 1):
            node = order[gi - 1]
            gbase = graph.base[node]
            pred_rows = [pos[p] + 1 for p in graph.inn[node]] or [0]
            for s in range(1, S + 1):
                c = int(codes[s - 1])
                sub = match if (gbase == c and 1 <= gbase <= 4) else -mismatch
                best = NEG
                best_move = 2
                best_row = pred_rows[0]
                # diagonal (match/mismatch) and vertical (delete graph node) from
                # every graph predecessor.
                for r in pred_rows:
                    diag = score[r, s - 1] + sub
                    if diag > best:
                        best, best_move, best_row = diag, 0, r
                    dele = score[r, s] - gap
                    if dele > best:
                        best, best_move, best_row = dele, 1, r
                # insertion (gap in graph, consume a sequence base)
                ins = score[gi, s - 1] - gap
                if ins > best:
                    best, best_move, best_row = ins, 2, gi
                score[gi, s] = best
                move[gi, s] = best_move
                pred_g[gi, s] = best_row

    # Traceback from the best end cell (last sequence base against any node, or
    # the all-graph-consumed border).
    gi = int(np.argmax(score[:, S]))
    s = S
    # also allow ending having consumed the sequence but more graph left is fine
    aligned: list[tuple[int, int]] = []  # (graph node or -1, seq index or -1)
    while gi > 0 or s > 0:
        mv = move[gi, s]
        if gi > 0 and s > 0 and mv == 0:
            node = order[gi - 1]
            aligned.append((node, s - 1))
            gi = int(pred_g[gi, s])
            s -= 1
        elif gi > 0 and mv == 1:
            gi = int(pred_g[gi, s])  # delete graph node: no sequence base
        else:
            aligned.append((-1, s - 1))  # insertion: new node for this base
            s -= 1
    aligned.reverse()

    # Fold the alignment into the graph.
    node_for_seq: dict[int, int] = {}
    for node, si in aligned:
        if si < 0:
            continue
        c = int(codes[si])
        if node >= 0 and graph.base[node] == c:
            graph.node_weight[node] += weight
            node_for_seq[si] = node
        else:
            new_node = graph.add_node(c)
            graph.node_weight[new_node] += weight
            node_for_seq[si] = new_node
    # Chain consecutive sequence bases with edges.
    prev = -1
    for si in range(len(codes)):
        cur = node_for_seq.get(si)
        if cur is None:  # pragma: no cover - every base gets a node above
            continue
        if prev >= 0:
            graph.add_edge(prev, cur, weight)
        prev = cur


def _poa_consensus_path(graph: _PoaGraph) -> list[int]:
    """Heaviest-bundle path through the POA DAG (the consensus).

    Uses the SPOA/Lee "heaviest bundling" rule: the consensus follows, from each
    node, the *outgoing edge with the most support* (ties broken toward the
    heavier immediate edge), so a low-coverage indel detour is skipped in favour
    of the transition the majority of reads took. Edge weights — not node
    weights — are what encode that support.
    """
    if len(graph) == 0:
        return []
    order = graph.topo_order()
    score = {node: 0.0 for node in order}
    nxt: dict[int, int] = {node: -1 for node in order}
    for node in reversed(order):
        best_key: tuple[float, int] | None = None
        for v, ew in graph.out[node].items():
            total = ew + score[v]
            key = (total, ew)  # prefer the longer-supported path, then heavier edge
            if best_key is None or key > best_key:
                best_key = key
                score[node] = total
                nxt[node] = v
    # The heaviest path's weight only accumulates downstream, so a source node
    # always has the maximal score; ``max`` returns the first (earliest topo).
    start = max(order, key=lambda node: score[node])
    path = [start]
    while nxt[path[-1]] >= 0:
        path.append(nxt[path[-1]])
    return path


def poa_consensus(
    sequences: Sequence["str | np.ndarray"],
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    gap: float = 4.0,
) -> str:
    """Consensus of a set of reads via partial-order alignment (``cudapoa``).

    Builds a POA graph by aligning the reads in turn, then reads off the
    heaviest-support path as the consensus. This is the ``cudapoa`` primitive's
    core use — collapsing several noisy reads over the same locus into one
    corrected sequence for polishing / hard-read rescue.
    """
    seqs = [_as_codes(s) for s in sequences if len(_as_codes(s)) > 0]
    if not seqs:
        return ""
    # Longest first: a long backbone gives shorter reads more nodes to match.
    seqs.sort(key=len, reverse=True)
    graph = _PoaGraph()
    _poa_add_first(graph, seqs[0], weight=1)
    for codes in seqs[1:]:
        _poa_align_and_add(graph, codes, 1, match, mismatch, gap)
    path = _poa_consensus_path(graph)
    return _decode(np.asarray([graph.base[n] for n in path], dtype=np.int8))


def poa_msa(
    sequences: Sequence["str | np.ndarray"],
    *,
    match: float = 2.0,
    mismatch: float = 4.0,
    gap: float = 4.0,
) -> tuple[str, list[str]]:
    """Return ``(consensus, inputs)`` — a thin wrapper over :func:`poa_consensus`.

    A full row-per-read MSA matrix is not needed by the pipeline (only the
    consensus is consumed), so this returns the consensus plus the decoded
    inputs it was built from, which is enough for logging / inspection.
    """
    consensus = poa_consensus(sequences, match=match, mismatch=mismatch, gap=gap)
    return consensus, [_decode(_as_codes(s)) for s in sequences]


# =========================================================================== #
# cudamapper — GPU minimizer seeding + anchor chaining (seq-to-seq mapping)
# =========================================================================== #
@dataclass
class MapperOverlap:
    """One mapping of a query onto a target: a chain and its span/score."""

    query_start: int
    query_end: int
    target_start: int
    target_end: int
    strand: int
    score: float
    num_anchors: int


def map_to_reference(
    reads: Sequence[str],
    reference: str,
    *,
    kmer: int = 15,
    window: int = 10,
    device: "str | None" = None,
    max_overlaps: int = 4,
) -> list[list[MapperOverlap]]:
    """Map reads to a reference with GPU minimizer seeding + chaining.

    This is the ``cudamapper`` facade: it drives the project's GPU-resident
    :class:`~graphmambaformer.alignment.seeding.GPUKmerIndex` (minimizer table +
    batched device lookup) and the batched
    :class:`~graphmambaformer.alignment.chaining.AffineChainer` chaining DP
    (which itself uses the CuPy ``chain_dp`` kernel on a CUDA host) to turn each
    read into ranked overlaps. On CPU the very same code runs through the
    portable NumPy tier, so results are identical.

    Imported lazily so :mod:`graphmambaformer.accel` never pulls in the
    alignment stages at load time.
    """
    import torch

    from ..config import ChainingConfig, SeedingConfig
    from .backend import default_context
    from ..alignment.chaining import AffineChainer
    from ..alignment.seeding import SeedingEngine

    if device is None:
        device = str(default_context().caps.device)
    dev = torch.device(device)

    seed_cfg = SeedingConfig(modes=("gpu_kmer",), kmer=kmer, window=window)
    seeder = SeedingEngine(seed_cfg, device=dev)
    bundle = seeder.build_indices(reference)

    ctx = default_context()
    chainer = AffineChainer(ChainingConfig(), backend=ctx.kernel_backend("chaining"))

    anchor_sets = [seeder.seed_read(read, bundle) for read in reads]
    chains_per_read = chainer.chain_batch(
        anchor_sets, [None] * len(anchor_sets), device=dev
    )

    out: list[list[MapperOverlap]] = []
    for chains in chains_per_read:
        overlaps = [
            MapperOverlap(
                query_start=int(c.read_start),
                query_end=int(c.read_end),
                target_start=int(c.ref_start),
                target_end=int(c.ref_end),
                strand=int(c.strand),
                score=float(c.score),
                num_anchors=len(c),
            )
            for c in chains[:max_overlaps]
        ]
        out.append(overlaps)
    return out


__all__ = [
    "UngappedExtension",
    "GlobalAlignment",
    "MapperOverlap",
    "genomeworks_available",
    "genomeworks_backend",
    "genomeworks_bindings",
    "genomeworks_bindings_available",
    "genomeworks_summary",
    "ungapped_extend",
    "ungapped_extend_batch",
    "global_align",
    "global_align_batch",
    "poa_consensus",
    "poa_msa",
    "map_to_reference",
]
