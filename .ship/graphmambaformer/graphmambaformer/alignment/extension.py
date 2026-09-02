"""Stage 3 — Extension: base-level dynamic programming.

Chaining says *where* a read aligns; extension says *how*, producing the CIGAR.
Two algorithms:

:func:`banded_affine_sw_batch`
    Banded affine-gap Smith-Waterman, batched and fully vectorized (default).
:class:`WavefrontAligner`
    Wavefront alignment — ``O(n·s)`` in the edit distance ``s``, so it beats DP
    outright when the two sequences are similar.

Vectorizing affine DP
---------------------
Within one DP row the horizontal (gap-in-query) recurrence

    E[j] = max( H[j-1] - open,  E[j-1] - extend )

is a serial dependency along the row, which is what normally forces affine DP to
be scalar. Writing it in closed form removes the dependency::

    E[j] = max_{j' < j} ( M[j'] - open - (j - j' - 1) * extend )
         = max_{j' < j} ( M[j'] + j' * extend ) - open - (j - 1) * extend

where ``M[j]`` is the best score at ``j`` that does *not* end in a horizontal
gap (so it depends only on the previous row). The inner term is an **exclusive
max-plus prefix scan**, i.e. a single ``cummax``. Substituting ``M`` for ``H`` is
exact: a gap opened from a cell that itself ends in a gap is the same as one
longer gap opened earlier, which the scan already considers.

Each row therefore costs a handful of elementwise ops plus one ``cummax``,
vectorized across both the band and the batch. The closed form assumes
``gap_open >= gap_extend`` (true for any sane scoring scheme), which is also what
makes the traceback recurrences below valid.

Band geometry
-------------
Band index ``d`` maps to target column ``j = i + offset - half_band + d``, where
``offset`` places the band on the chain's diagonal. Because ``i`` and ``j``
advance together along a diagonal, the previous row's *diagonal* predecessor sits
at band index ``d`` and its *vertical* predecessor at ``d + 1`` — no index
arithmetic per cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import ExtensionConfig
from .seeding import N_CODE, decode_bases, encode_bases, reverse_complement_codes
from .types import AnchorSet, Chain, ExtensionResult, merge_cigar, run_length_encode

_NEG_INF = -1e30


@dataclass
class BandedDPResult:
    """Output of :func:`banded_affine_sw_batch`.

    ``H``/``E``/``F`` are only populated when matrices were requested; they are
    band-relative, shape ``(B, M + 1, W)``, and are what :func:`traceback_banded`
    walks.
    """

    score: torch.Tensor  # (B,)
    query_end: torch.Tensor  # (B,) 1-based end of the aligned query span
    target_end: torch.Tensor  # (B,) 1-based end of the aligned target span
    half_band: int
    band_offset: torch.Tensor  # (B,)
    H: Optional[torch.Tensor] = None
    E: Optional[torch.Tensor] = None
    F: Optional[torch.Tensor] = None


def banded_affine_sw_batch(
    query: torch.Tensor,
    target: torch.Tensor,
    query_len: torch.Tensor,
    target_len: torch.Tensor,
    cfg: ExtensionConfig,
    half_band: Optional[int] = None,
    band_offset: Optional[torch.Tensor] = None,
    return_matrices: bool = False,
) -> BandedDPResult:
    """Batched banded affine-gap local (Smith-Waterman) alignment.

    Args:
        query / target: ``(B, M)`` / ``(B, N)`` integer base codes. Negative
            codes, and the ambiguous code ``N``, never score as a match.
        query_len / target_len: ``(B,)`` valid lengths.
        half_band: band half-width; defaults to ``cfg.half_band``.
        band_offset: ``(B,)`` diagonal on which to centre the band (0 = main).
        return_matrices: keep the DP matrices so a CIGAR can be traced back.

    Returns a :class:`BandedDPResult` with the best local score per pair.
    """
    from .seeding import N_CODE

    B, M = query.shape
    N = target.shape[1]
    device = query.device
    hb = int(cfg.half_band if half_band is None else half_band)
    width = 2 * hb + 1

    query = query.to(torch.int16)
    target = target.to(torch.int16)
    qlen = query_len.to(device=device, dtype=torch.long)
    tlen = target_len.to(device=device, dtype=torch.long)
    offset = (
        torch.zeros(B, dtype=torch.long, device=device)
        if band_offset is None
        else band_offset.to(device=device, dtype=torch.long)
    )

    d_index = torch.arange(width, device=device)
    # Prefix-scan constants: the +d * extend / -(d - 1) * extend pair.
    scan_add = d_index.to(torch.float32) * cfg.gap_extend
    scan_sub = cfg.gap_open + (d_index.to(torch.float32) - 1.0) * cfg.gap_extend

    # One guard column past the band so the vertical predecessor at d = W - 1 is
    # always addressable; it stays out of band, hence H = 0 / F = -inf.
    h_prev = torch.zeros((B, width + 1), dtype=torch.float32, device=device)
    f_prev = torch.full((B, width + 1), _NEG_INF, dtype=torch.float32, device=device)

    H_store = E_store = F_store = None
    if return_matrices:
        H_store = torch.zeros((B, M + 1, width), dtype=torch.float32, device=device)
        E_store = torch.full((B, M + 1, width), _NEG_INF, dtype=torch.float32, device=device)
        F_store = torch.full((B, M + 1, width), _NEG_INF, dtype=torch.float32, device=device)

    best = torch.zeros(B, dtype=torch.float32, device=device)
    best_i = torch.zeros(B, dtype=torch.long, device=device)
    best_j = torch.zeros(B, dtype=torch.long, device=device)
    row_max_seen = torch.zeros(B, dtype=torch.float32, device=device)

    max_rows = int(qlen.max().item()) if B else 0
    for i in range(1, max_rows + 1):
        j = i + offset[:, None] - hb + d_index[None, :]  # (B, W) target columns
        in_band = (j >= 1) & (j <= tlen[:, None]) & (i <= qlen[:, None])

        q_base = query[:, i - 1].unsqueeze(1)  # (B, 1)
        t_base = target.gather(1, (j - 1).clamp(0, max(N - 1, 0)))  # (B, W)
        is_match = (q_base == t_base) & (q_base >= 0) & (q_base != N_CODE)
        sub = torch.where(
            is_match,
            torch.full_like(t_base, 0, dtype=torch.float32) + cfg.match_score,
            torch.full_like(t_base, 0, dtype=torch.float32) - cfg.mismatch_penalty,
        )

        diag = h_prev[:, :width] + sub
        vertical = torch.maximum(
            h_prev[:, 1:] - cfg.gap_open, f_prev[:, 1:] - cfg.gap_extend
        )
        # M: best score at (i, j) not ending in a horizontal gap.
        m_val = torch.maximum(diag, vertical).clamp(min=0.0)
        m_val = torch.where(in_band, m_val, torch.full_like(m_val, _NEG_INF))

        # Exclusive max-plus prefix scan -> the horizontal-gap term E.
        inclusive = torch.cummax(m_val + scan_add, dim=1).values
        exclusive = torch.cat(
            [torch.full((B, 1), _NEG_INF, device=device), inclusive[:, :-1]], dim=1
        )
        e_val = exclusive - scan_sub
        e_val = torch.where(in_band, e_val, torch.full_like(e_val, _NEG_INF))

        h_val = torch.maximum(m_val, e_val).clamp(min=0.0)
        h_val = torch.where(in_band, h_val, torch.zeros_like(h_val))
        f_val = torch.where(in_band, vertical, torch.full_like(vertical, _NEG_INF))

        if return_matrices:
            H_store[:, i] = h_val
            E_store[:, i] = e_val
            F_store[:, i] = f_val

        row_best, row_slot = h_val.max(dim=1)
        improved = row_best > best
        best = torch.where(improved, row_best, best)
        best_i = torch.where(improved, torch.full_like(best_i, i), best_i)
        best_j = torch.where(improved, j.gather(1, row_slot[:, None]).squeeze(1), best_j)
        row_max_seen = torch.maximum(row_max_seen, row_best)

        h_prev = torch.cat([h_val, torch.zeros((B, 1), device=device)], dim=1)
        f_prev = torch.cat(
            [f_val, torch.full((B, 1), _NEG_INF, device=device)], dim=1
        )

        # X-drop: stop once every read in the batch has fallen far below its own
        # best score, which is where the alignment has clearly ended.
        if cfg.x_drop > 0 and bool(((row_max_seen - row_best) > cfg.x_drop).all()):
            break

    return BandedDPResult(
        score=best,
        query_end=best_i,
        target_end=best_j,
        half_band=hb,
        band_offset=offset,
        H=H_store,
        E=E_store,
        F=F_store,
    )


def traceback_banded(
    result: BandedDPResult,
    index: int,
    query: np.ndarray,
    target: np.ndarray,
    cfg: ExtensionConfig,
    tolerance: float = 1e-3,
) -> ExtensionResult:
    """Recover the CIGAR for batch element ``index`` from the stored DP matrices.

    Walks back from the best cell, at each step deciding which recurrence term
    produced it. Priority is diagonal, then vertical, then horizontal, and the
    gap states use their own recurrences (``F[i][j] = max(H[i-1][j] - open,
    F[i-1][j] - extend)`` and the horizontal analogue) to tell a gap open from a
    gap extension — valid because ``gap_open >= gap_extend``.
    """
    from .seeding import N_CODE

    if result.H is None:
        raise ValueError("traceback needs banded_affine_sw_batch(return_matrices=True)")

    H = result.H[index].cpu().numpy()
    E = result.E[index].cpu().numpy()
    F = result.F[index].cpu().numpy()
    hb = result.half_band
    offset = int(result.band_offset[index].item())

    i = int(result.query_end[index].item())
    j = int(result.target_end[index].item())
    score = float(result.score[index].item())

    def band(row_i: int, col_j: int) -> int:
        return col_j - row_i - offset + hb

    def cell(matrix: np.ndarray, row_i: int, col_j: int) -> float:
        d = band(row_i, col_j)
        if row_i < 0 or row_i >= matrix.shape[0] or d < 0 or d >= matrix.shape[1]:
            return _NEG_INF
        return float(matrix[row_i, d])

    ops: list[str] = []
    state = "H"
    while i > 0 and j > 0:
        if state == "H":
            if cell(H, i, j) <= 0.0:
                break
            matched = (
                query[i - 1] == target[j - 1]
                and query[i - 1] >= 0
                and query[i - 1] != N_CODE
            )
            sub = cfg.match_score if matched else -cfg.mismatch_penalty
            diag = max(cell(H, i - 1, j - 1), 0.0) + sub
            if abs(cell(H, i, j) - diag) <= tolerance:
                ops.append("=" if matched else "X")
                i -= 1
                j -= 1
            elif abs(cell(H, i, j) - cell(F, i, j)) <= tolerance:
                state = "F"
            else:
                state = "E"
        elif state == "F":  # vertical: consumes a query base -> insertion
            current = cell(F, i, j)
            ops.append("I")
            i -= 1
            if abs(current - (cell(H, i, j) - cfg.gap_open)) <= tolerance:
                state = "H"
            elif abs(current - (cell(F, i, j) - cfg.gap_extend)) > tolerance:
                state = "H"  # numerical drift: resume from the H state
        else:  # "E" horizontal: consumes a target base -> deletion
            current = cell(E, i, j)
            ops.append("D")
            j -= 1
            if abs(current - (cell(H, i, j) - cfg.gap_open)) <= tolerance:
                state = "H"
            elif abs(current - (cell(E, i, j) - cfg.gap_extend)) > tolerance:
                state = "H"

    ops.reverse()
    cigar = run_length_encode(ops)
    counts = {op: 0 for op in ("=", "X", "I", "D")}
    for op, n in cigar:
        counts[op] = counts.get(op, 0) + n

    return ExtensionResult(
        score=score,
        cigar=cigar,
        read_start=i,
        read_end=int(result.query_end[index].item()),
        ref_start=j,
        ref_end=int(result.target_end[index].item()),
        n_match=counts["="],
        n_mismatch=counts["X"],
        n_insertion=counts["I"],
        n_deletion=counts["D"],
    )


# --------------------------------------------------------------------------- #
# Wavefront alignment
# --------------------------------------------------------------------------- #
class WavefrontAligner:
    """Global wavefront alignment under unit edit costs — ``O(n·s)``.

    WFA inverts the DP: instead of filling a matrix and reading off the score, it
    tracks, for each score ``s``, how far along each diagonal an alignment of
    that cost can reach. When the sequences are similar the answer is found after
    a few wavefronts, so cost scales with the *edit distance* rather than with
    the product of the lengths.

    Diagonal ``k = i - j``; ``offset[k]`` is the number of query bases consumed
    at the furthest reach. Each step is one vectorized shift-and-max over the
    wavefront followed by a match extension.
    """

    def __init__(self, cfg: ExtensionConfig | None = None):
        self.cfg = cfg or ExtensionConfig()

    @staticmethod
    def _extend(
        offsets: np.ndarray,
        diagonals: np.ndarray,
        query: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Advance every diagonal through its maximal run of matching bases."""
        n, m = len(query), len(target)
        out = offsets.copy()
        for idx in range(len(out)):
            i = out[idx]
            if i < 0:
                continue
            j = i - diagonals[idx]
            while (
                i < n
                and 0 <= j < m
                and query[i] == target[j]
                and query[i] >= 0
                and query[i] != N_CODE
            ):
                i += 1
                j += 1
            out[idx] = i
        return out

    def align(
        self, query: np.ndarray, target: np.ndarray
    ) -> tuple[int, list[tuple[str, int]]]:
        """Align ``query`` to ``target`` end-to-end.

        Returns ``(edit_distance, cigar)``. Raises :class:`RuntimeError` if the
        edit distance exceeds ``cfg.wfa_max_distance``, which is the caller's cue
        to fall back to banded DP.
        """
        n, m = len(query), len(target)
        if n == 0 or m == 0:
            return max(n, m), merge_cigar([("I", n), ("D", m)])

        # The alignment is complete when diagonal n - m reaches query position n,
        # which is the matrix corner (n, m).
        final_k = n - m
        history: list[tuple[np.ndarray, np.ndarray]] = [
            (
                np.array([0], dtype=np.int64),
                self._extend(
                    np.array([0], dtype=np.int64),
                    np.array([0], dtype=np.int64),
                    query,
                    target,
                ),
            )
        ]

        score = 0
        while True:
            diagonals, offsets = history[-1]
            if diagonals[0] <= final_k <= diagonals[-1]:
                if offsets[final_k - diagonals[0]] >= n:
                    return score, self._traceback(history, final_k)

            if score >= self.cfg.wfa_max_distance:
                raise RuntimeError(
                    f"WFA exceeded wfa_max_distance={self.cfg.wfa_max_distance}"
                )
            score += 1

            new_k = np.arange(diagonals[0] - 1, diagonals[-1] + 2, dtype=np.int64)

            def predecessor(shift: int) -> np.ndarray:
                """Offsets on diagonal ``new_k + shift`` of the previous wavefront."""
                idx = new_k + shift - diagonals[0]
                out = np.full(len(new_k), -1, dtype=np.int64)
                ok = (idx >= 0) & (idx < len(offsets))
                out[ok] = offsets[idx[ok]]
                return out

            def clip(values: np.ndarray) -> np.ndarray:
                """Reject candidates that would step outside the DP matrix."""
                col = values - new_k
                bad = (values < 0) | (values > n) | (col < 0) | (col > m)
                return np.where(bad, -1, values)

            # Each candidate is clipped on its own: one illegal step must not
            # invalidate a diagonal that another predecessor can still reach.
            substitution = predecessor(0)
            substitution = clip(np.where(substitution >= 0, substitution + 1, -1))
            insertion = predecessor(-1)  # k-1 -> k, consumes a query base
            insertion = clip(np.where(insertion >= 0, insertion + 1, -1))
            deletion = clip(predecessor(1))  # k+1 -> k, consumes a target base

            new_off = np.maximum(np.maximum(substitution, insertion), deletion)
            history.append((new_k, self._extend(new_off, new_k, query, target)))

    def _traceback(
        self, history: list[tuple[np.ndarray, np.ndarray]], final_k: int
    ) -> list[tuple[str, int]]:
        """Walk the stored wavefronts back to a CIGAR.

        At score ``s`` and diagonal ``k`` the furthest reach is known, so the
        predecessor is whichever of the three candidates the forward pass took —
        the one with the largest offset. The gap between that candidate and the
        reach is exactly the run of matches the extension consumed.
        """
        ops: list[str] = []
        k = int(final_k)

        for score in range(len(history) - 1, 0, -1):
            diagonals, offsets = history[score]
            reach = int(offsets[k - diagonals[0]])
            prev_k, prev_off = history[score - 1]

            def predecessor(shift: int) -> int:
                idx = k + shift - prev_k[0]
                if 0 <= idx < len(prev_off) and prev_off[idx] >= 0:
                    return int(prev_off[idx])
                return -1

            substitution, insertion, deletion = (
                predecessor(0),
                predecessor(-1),
                predecessor(1),
            )
            candidates = [
                ("X", substitution + 1 if substitution >= 0 else -1, k),
                ("I", insertion + 1 if insertion >= 0 else -1, k - 1),
                ("D", deletion, k + 1),
            ]
            op, start, k = max(candidates, key=lambda c: c[1])

            ops.extend(["="] * max(reach - start, 0))
            ops.append(op)

        # The score-0 wavefront is a pure match run out of the origin.
        diagonals, offsets = history[0]
        ops.extend(["="] * int(offsets[k - diagonals[0]]))
        ops.reverse()
        return merge_cigar(run_length_encode(ops))


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
class ExtensionEngine:
    """Stage 3 entry point: chains -> base-level alignments with CIGARs.

    Every chain of a batch is extended in one batched DP call. The DP window is
    placed on the chain's own diagonal and the band is widened by the chain's
    internal diagonal spread, so a chain containing a large indel gets a band
    wide enough to align through it instead of being clipped.
    """

    def __init__(
        self,
        cfg: ExtensionConfig | None = None,
        device: torch.device | str | None = None,
        backend: str = "torch",
    ):
        self.cfg = cfg or ExtensionConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.backend = backend
        self.wfa = WavefrontAligner(self.cfg)

    def _window(
        self, chain: Chain, anchors: AnchorSet, read_len: int, ref_len: int
    ) -> tuple[int, int, int]:
        """DP window on the reference plus the band geometry for one chain.

        Returns ``(target_start, target_end, half_band)``.
        """
        diagonal = chain.ref_start - chain.read_start
        start = max(0, diagonal - self.cfg.flank)
        end = min(ref_len, diagonal + read_len + self.cfg.flank)
        if end - start > self.cfg.max_window:
            end = start + self.cfg.max_window

        # Spread of the chain's own diagonals bounds the indel size it implies.
        idx = chain.anchor_idx
        spread = 0
        if len(idx):
            diagonals = anchors.diagonal[idx]
            spread = int(diagonals.max() - diagonals.min())
        half_band = min(self.cfg.half_band + spread, self.cfg.max_half_band)
        return start, end, half_band

    def extend_chains(
        self,
        read_seq: str,
        chains: Sequence[Chain],
        anchors: AnchorSet,
        ref_seq: str,
    ) -> list[ExtensionResult]:
        """Extend every chain of one read; returns one result per chain."""
        if not chains:
            return []

        forward = encode_bases(read_seq)
        reverse = reverse_complement_codes(forward)
        ref_codes = encode_bases(ref_seq)
        read_len, ref_len = len(forward), len(ref_codes)

        if self.cfg.algorithm == "wfa":
            return [
                self._extend_wfa(chain, anchors, forward, reverse, ref_codes, read_len, ref_len)
                for chain in chains
            ]

        windows = [self._window(c, anchors, read_len, ref_len) for c in chains]
        half_band = max(w[2] for w in windows)
        width = max(w[1] - w[0] for w in windows)

        B = len(chains)
        query = torch.full((B, read_len), -1, dtype=torch.int16, device=self.device)
        target = torch.full((B, width), -1, dtype=torch.int16, device=self.device)
        query_len = torch.full((B,), read_len, dtype=torch.long, device=self.device)
        target_len = torch.zeros(B, dtype=torch.long, device=self.device)
        band_offset = torch.zeros(B, dtype=torch.long, device=self.device)

        for b, (chain, (start, end, _)) in enumerate(zip(chains, windows)):
            codes = forward if chain.strand > 0 else reverse
            query[b, : len(codes)] = torch.as_tensor(
                codes.astype(np.int16), device=self.device
            )
            window = ref_codes[start:end]
            target[b, : len(window)] = torch.as_tensor(
                window.astype(np.int16), device=self.device
            )
            target_len[b] = len(window)
            # Read base 0 aligns to reference offset `diagonal`; inside the
            # window that is `diagonal - start`, which is where the band goes.
            band_offset[b] = (chain.ref_start - chain.read_start) - start

        simd_scores: torch.Tensor | None = None
        if self.device.type == "cpu":
            try:
                from ..accel.simd_sw import simd_available, smith_waterman_score

                if simd_available():
                    values = []
                    for chain, (start, end, _) in zip(chains, windows):
                        codes = forward if chain.strand > 0 else reverse
                        values.append(
                            smith_waterman_score(
                                decode_bases(codes),
                                decode_bases(ref_codes[start:end]),
                                self.cfg,
                            )
                        )
                    simd_scores = torch.tensor(values, dtype=torch.float32)
            except RuntimeError:
                simd_scores = None

        cuda_result = None
        if self.backend == "cuda_rawkernel" and self.device.type == "cuda":
            try:
                from ..accel.cuda_kernels import banded_sw as cuda_banded_sw

                cuda_result = cuda_banded_sw(
                    query,
                    target,
                    query_len,
                    target_len,
                    half_band=half_band,
                    match_score=self.cfg.match_score,
                    mismatch_penalty=self.cfg.mismatch_penalty,
                    gap_open=self.cfg.gap_open,
                    gap_extend=self.cfg.gap_extend,
                    band_offset=band_offset,
                )
            except RuntimeError:
                cuda_result = None

        result = banded_affine_sw_batch(
            query,
            target,
            query_len,
            target_len,
            self.cfg,
            half_band=half_band,
            band_offset=band_offset,
            return_matrices=True,
        )
        if cuda_result is not None:
            cuda_score, cuda_q_end, cuda_t_end = cuda_result
            # Traceback matrices remain on the portable tier. Only trust the
            # accelerator result when all three observables agree.
            agrees = (
                torch.allclose(cuda_score, result.score, atol=1e-3, rtol=1e-4)
                and torch.equal(cuda_q_end, result.query_end)
                and torch.equal(cuda_t_end, result.target_end)
            )
            if agrees:
                result.score = cuda_score
                result.query_end = cuda_q_end
                result.target_end = cuda_t_end
        if simd_scores is not None and torch.allclose(
            simd_scores, result.score.cpu(), atol=1e-3, rtol=1e-4
        ):
            result.score = simd_scores.to(result.score.device)

        out: list[ExtensionResult] = []
        for b, (chain, (start, _end, _)) in enumerate(zip(chains, windows)):
            codes = forward if chain.strand > 0 else reverse
            window = ref_codes[start : start + int(target_len[b].item())]
            extension = traceback_banded(result, b, codes, window, self.cfg)
            # Lift the window-local reference span back to forward-reference
            # coordinates, and record the unaligned read flanks as soft clips.
            extension.ref_start += start
            extension.ref_end += start
            extension.cigar = merge_cigar(
                [("S", extension.read_start)]
                + extension.cigar
                + [("S", read_len - extension.read_end)]
            )
            out.append(extension)
        return out

    def _extend_wfa(
        self,
        chain: Chain,
        anchors: AnchorSet,
        forward: np.ndarray,
        reverse: np.ndarray,
        ref_codes: np.ndarray,
        read_len: int,
        ref_len: int,
    ) -> ExtensionResult:
        """Extend one chain with the wavefront aligner (global within the window)."""
        codes = forward if chain.strand > 0 else reverse
        # WFA is global, so unlike local SW it must not see the flanking search
        # bases from `_window`: those would be forced into the CIGAR as deletions.
        # Use the chain's read-zero diagonal and implied net indel to define the
        # sequence that should align end-to-end.
        start = int(np.clip(chain.ref_start - chain.read_start, 0, max(ref_len - 1, 0)))
        net_indel = chain.ref_span - chain.read_span
        target_length = max(1, read_len + net_indel)
        end = min(ref_len, start + target_length)
        window = ref_codes[start:end]

        gpu_distance: int | None = None
        if self.backend == "cuda_rawkernel" and self.device.type == "cuda":
            try:
                from ..accel.cuda_kernels import wfa_distance as cuda_wfa_distance

                query_t = torch.as_tensor(codes, dtype=torch.int8, device=self.device)[None]
                target_t = torch.as_tensor(window, dtype=torch.int8, device=self.device)[None]
                gpu_distance = int(
                    cuda_wfa_distance(
                        query_t,
                        target_t,
                        torch.tensor([len(codes)], device=self.device),
                        torch.tensor([len(window)], device=self.device),
                        self.cfg.wfa_max_distance,
                    )[0].item()
                )
                if gpu_distance < 0:
                    raise RuntimeError("GPU WFA exceeded distance bound")
            except RuntimeError:
                gpu_distance = None

        try:
            distance, cigar = self.wfa.align(codes, window)
        except RuntimeError:
            # Too divergent for the wavefront bound: fall back to banded DP.
            saved, self.cfg.algorithm = self.cfg.algorithm, "banded_sw"
            try:
                return self.extend_chains(
                    decode_bases(forward),
                    [chain],
                    anchors,
                    decode_bases(ref_codes),
                )[0]
            finally:
                self.cfg.algorithm = saved
        if gpu_distance is not None and gpu_distance != distance:
            # Never let an accelerator disagreement change the biological call.
            # The portable WFA is independently checked against Levenshtein.
            gpu_distance = None

        counts = {op: 0 for op in ("=", "X", "I", "D")}
        for op, n in cigar:
            counts[op] = counts.get(op, 0) + n
        gap_cost = sum(
            self.cfg.gap_open + max(n - 1, 0) * self.cfg.gap_extend
            for op, n in cigar
            if op in ("I", "D")
        )
        score = (
            counts["="] * self.cfg.match_score
            - counts["X"] * self.cfg.mismatch_penalty
            - gap_cost
        )
        return ExtensionResult(
            score=float(score),
            cigar=cigar,
            read_start=0,
            read_end=read_len,
            ref_start=start,
            ref_end=start + len(window),
            n_match=counts["="],
            n_mismatch=counts["X"],
            n_insertion=counts["I"],
            n_deletion=counts["D"],
        )
