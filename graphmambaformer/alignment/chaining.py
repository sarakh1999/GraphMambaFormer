"""Stage 2 — Chaining.

Anchors from Stage 1 are collinear fragments of one alignment interrupted by
mismatches and indels. Chaining recovers the best collinear subset with a
minimap2-style affine-gap dynamic program over anchors sorted by reference end::

    f[i] = max( w_i,  max_j  f[j] + advance(j, i) - penalty(j, i) + bonus(j, i) )
    advance(j, i) = min( min(dq, dr), w_i )
    penalty(j, i) = gap_open + gap_extend * gap + log_coeff * log2(gap + 1)
    gap           = | dr - dq |

with ``dq``/``dr`` the read/reference advance between anchors ``j`` and ``i``.
The ``log2`` term is what makes long gaps progressively cheaper per base, so a
genuine structural indel is preferred over abandoning the chain.

Two additions make the DP pangenome-aware rather than reference-linear:

**Graph-distance bonus** — anchors whose reference nodes are within a few hops in
the pangenome graph get a bonus decaying with hop count, so a chain that walks a
bubble is not penalised for the reference-coordinate jump the bubble causes.

**Reference-path bias** — anchors on the graph's backbone path get extra weight,
which breaks ties toward the reference allele when evidence is balanced.

Three interchangeable DP backends with identical semantics:
:func:`chain_dp_numpy` (single read, the reference implementation),
:func:`chain_dp_batched` (torch, whole batch at once — the GPU path), and the
CuPy ``chain_dp`` RawKernel. All three are verified to agree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import ChainingConfig
from .types import AnchorSet, Chain

NO_PREDECESSOR = -1
_NEG_INF = -1e30


# --------------------------------------------------------------------------- #
# Graph distance
# --------------------------------------------------------------------------- #
class GraphDistanceOracle:
    """Bounded hop distances between pangenome graph nodes.

    Distances are computed on demand for just the nodes a read's anchors touch —
    typically a few dozen — via a depth-limited BFS over a CSR adjacency. An
    all-pairs precomputation would be quadratic in the graph size for a result
    that is almost entirely unused.
    """

    def __init__(self, edge_index: np.ndarray, num_nodes: int, max_hops: int = 3):
        self.num_nodes = int(num_nodes)
        self.max_hops = int(max_hops)

        edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
        # Undirected: a chain may traverse a bubble in either orientation.
        src = np.concatenate([edge_index[0], edge_index[1]])
        dst = np.concatenate([edge_index[1], edge_index[0]])
        order = np.argsort(src, kind="stable")
        self.neighbours = dst[order]
        counts = np.bincount(src, minlength=self.num_nodes)
        self.indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])

    def _bfs(self, source: int) -> dict[int, int]:
        """Nodes within ``max_hops`` of ``source``, mapped to their hop count."""
        seen = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            depth = seen[node]
            if depth >= self.max_hops:
                continue
            for nbr in self.neighbours[self.indptr[node] : self.indptr[node + 1]]:
                nbr = int(nbr)
                if nbr not in seen:
                    seen[nbr] = depth + 1
                    queue.append(nbr)
        return seen

    def hop_matrix(self, nodes: Sequence[int]) -> tuple[np.ndarray, dict[int, int]]:
        """Dense hop matrix over the distinct nodes in ``nodes``.

        Returns ``(hops, index)`` where ``hops[a, b]`` is the hop count between
        the ``a``-th and ``b``-th distinct node (``-1`` beyond ``max_hops``) and
        ``index`` maps a node id to its row.
        """
        distinct = sorted({int(n) for n in nodes if n >= 0})
        index = {node: i for i, node in enumerate(distinct)}
        hops = np.full((len(distinct), len(distinct)), -1, dtype=np.int16)
        for node, row in index.items():
            for reached, depth in self._bfs(node).items():
                col = index.get(reached)
                if col is not None:
                    hops[row, col] = depth
        return hops, index


# --------------------------------------------------------------------------- #
# DP backends
# --------------------------------------------------------------------------- #
def _penalty(gap: np.ndarray, cfg: ChainingConfig) -> np.ndarray:
    """Affine + logarithmic gap cost; zero for a perfectly collinear step."""
    return np.where(
        gap > 0,
        cfg.gap_open + cfg.gap_extend * gap + cfg.log_coeff * np.log2(gap + 1.0),
        0.0,
    )


def chain_dp_numpy(
    read_end: np.ndarray,
    ref_end: np.ndarray,
    weight: np.ndarray,
    cfg: ChainingConfig,
    bonus: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Chaining DP for one read — the reference implementation.

    Anchors must already be sorted by reference end. Sequential in ``i`` (the DP
    is inherently so) but vectorized over the ``max_lookback`` predecessors, so
    the Python loop runs once per anchor rather than once per anchor pair.

    Args:
        bonus: optional ``(n, max_lookback)`` graph bonus, column ``t`` holding
            the bonus for predecessor ``i - max_lookback + t``.

    Returns ``(f, predecessor)``.
    """
    n = len(read_end)
    f = np.zeros(n, dtype=np.float64)
    parent = np.full(n, NO_PREDECESSOR, dtype=np.int64)
    if n == 0:
        return f, parent

    lookback = max(1, cfg.max_lookback)
    for i in range(n):
        lo = max(0, i - lookback)
        if lo == i:
            f[i] = weight[i]
            continue

        dq = read_end[i] - read_end[lo:i]
        dr = ref_end[i] - ref_end[lo:i]
        ok = (dq > 0) & (dr > 0) & (dq <= cfg.max_gap) & (dr <= cfg.max_gap)
        if not ok.any():
            f[i] = weight[i]
            continue

        advance = np.minimum(np.minimum(dq, dr), weight[i])
        score = f[lo:i] + advance - _penalty(np.abs(dr - dq), cfg)
        if bonus is not None:
            score = score + bonus[i, lookback - (i - lo) :]
        score = np.where(ok, score, _NEG_INF)

        best = int(score.argmax())
        if score[best] > weight[i]:
            f[i] = score[best]
            parent[i] = lo + best
        else:
            f[i] = weight[i]
    return f, parent


def chain_dp_batched(
    read_end: torch.Tensor,
    ref_end: torch.Tensor,
    weight: torch.Tensor,
    n_anchors: torch.Tensor,
    cfg: ChainingConfig,
    bonus: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chaining DP for a whole batch of reads in one pass — the GPU path.

    Vectorized over both the batch and the lookback window, so each of the ``A``
    sequential steps is a single set of tensor ops covering every read. That
    turns ``B * A`` kernel launches into ``A``, which is the difference between
    the DP being negligible and being the bottleneck on a GPU.

    Args:
        read_end / ref_end / weight: ``(B, A)``, right-padded, sorted by
            reference end within each row.
        n_anchors: ``(B,)`` valid anchor count per read.
        bonus: optional ``(B, A, max_lookback)`` graph bonus.

    Returns ``(f, predecessor)`` of shape ``(B, A)``; padded slots hold 0 / -1.
    """
    B, A = read_end.shape
    device = read_end.device
    lookback = max(1, cfg.max_lookback)

    read_end = read_end.to(torch.float32)
    ref_end = ref_end.to(torch.float32)
    weight = weight.to(torch.float32)

    f = torch.zeros((B, A), dtype=torch.float32, device=device)
    parent = torch.full((B, A), NO_PREDECESSOR, dtype=torch.long, device=device)
    valid_row = torch.arange(A, device=device)[None, :] < n_anchors[:, None]

    offsets = torch.arange(lookback, 0, -1, device=device)  # i-1 ... i-lookback
    for i in range(A):
        j = i - offsets  # (lookback,) predecessor indices, may be negative
        in_range = j >= 0
        j_clamped = j.clamp(min=0)

        dq = read_end[:, i : i + 1] - read_end[:, j_clamped]
        dr = ref_end[:, i : i + 1] - ref_end[:, j_clamped]
        ok = (
            in_range[None, :]
            & (dq > 0)
            & (dr > 0)
            & (dq <= cfg.max_gap)
            & (dr <= cfg.max_gap)
            & valid_row[:, j_clamped]
        )

        advance = torch.minimum(torch.minimum(dq, dr), weight[:, i : i + 1])
        gap = (dr - dq).abs()
        penalty = torch.where(
            gap > 0,
            cfg.gap_open + cfg.gap_extend * gap + cfg.log_coeff * torch.log2(gap + 1.0),
            torch.zeros_like(gap),
        )
        score = f[:, j_clamped] + advance - penalty
        if bonus is not None:
            score = score + bonus[:, i, :]
        score = torch.where(ok, score, torch.full_like(score, _NEG_INF))

        best_score, best_slot = score.max(dim=1)
        take = best_score > weight[:, i]
        f[:, i] = torch.where(take, best_score, weight[:, i])
        parent[:, i] = torch.where(
            take,
            j_clamped[best_slot],
            torch.full_like(best_slot, NO_PREDECESSOR),
        )

    f = torch.where(valid_row, f, torch.zeros_like(f))
    parent = torch.where(valid_row, parent, torch.full_like(parent, NO_PREDECESSOR))
    return f, parent


# --------------------------------------------------------------------------- #
# Chainer
# --------------------------------------------------------------------------- #
@dataclass
class ChainingContext:
    """Optional pangenome context that turns the DP graph-aware.

    Attributes:
        oracle: hop-distance source for the graph bonus.
        backbone: per-node flag marking the graph's reference path.
    """

    oracle: Optional[GraphDistanceOracle] = None
    backbone: Optional[np.ndarray] = None


class AffineChainer:
    """Stage 2 entry point: anchors -> ranked candidate chains.

    Each strand is chained independently (a chain cannot switch orientation), then
    the chains from both strands compete in a single primary/secondary selection.
    """

    def __init__(self, cfg: ChainingConfig | None = None, backend: str = "torch"):
        self.cfg = cfg or ChainingConfig()
        self.backend = backend

    # ---- anchor weights & bonus -------------------------------------------- #
    def _weights(self, anchors: AnchorSet, ctx: Optional[ChainingContext]) -> np.ndarray:
        """Per-anchor chain weight: match length, biased toward the reference path."""
        weight = anchors.length.astype(np.float64)
        if ctx is not None and ctx.backbone is not None:
            on_path = (anchors.node_id >= 0) & ctx.backbone[
                np.clip(anchors.node_id, 0, len(ctx.backbone) - 1)
            ]
            weight = weight + on_path * self.cfg.ref_path_bias
        # A neural seed score, when present, scales the anchor's contribution so
        # the DP trusts anchors the model believes in.
        if np.isfinite(anchors.score).any():
            scores = np.where(np.isfinite(anchors.score), anchors.score, 1.0)
            weight = weight * (0.5 + scores)
        return weight

    def _graph_bonus(
        self, anchors: AnchorSet, ctx: Optional[ChainingContext]
    ) -> Optional[np.ndarray]:
        """``(n, max_lookback)`` bonus for chaining graph-adjacent anchors."""
        if ctx is None or ctx.oracle is None or self.cfg.graph_bonus == 0.0:
            return None
        if not (anchors.node_id >= 0).any():
            return None

        hops, index = ctx.oracle.hop_matrix(anchors.node_id)
        if hops.size == 0:
            return None

        n = len(anchors)
        lookback = max(1, self.cfg.max_lookback)
        local = np.array(
            [index.get(int(node), -1) for node in anchors.node_id], dtype=np.int64
        )

        # Column t holds the predecessor i - lookback + t.
        rows = np.arange(n)[:, None]
        cols = rows - lookback + np.arange(lookback)[None, :]
        valid = (cols >= 0) & (local[rows] >= 0) & (local[np.clip(cols, 0, n - 1)] >= 0)

        hop = np.full((n, lookback), -1, dtype=np.int16)
        src = local[rows.repeat(lookback, axis=1)][valid]
        dst = local[np.clip(cols, 0, n - 1)][valid]
        hop[valid] = hops[src, dst]

        # Full bonus for same-node pairs, decaying linearly to zero past max_hops.
        decay = 1.0 - np.clip(hop, 0, None) / max(self.cfg.graph_max_hops, 1)
        return np.where(hop >= 0, self.cfg.graph_bonus * decay, 0.0)

    # ---- DP + traceback ----------------------------------------------------- #
    def _run_dp(
        self,
        anchors: AnchorSet,
        weight: np.ndarray,
        bonus: Optional[np.ndarray],
        device: torch.device | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(anchors) == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int64)

        if self.backend == "cuda_rawkernel":
            from ..accel.cuda_kernels import chain_dp as cuda_chain_dp

            dev = torch.device(device or "cuda")
            stacked = torch.stack(
                [
                    torch.as_tensor(anchors.read_end, device=dev),
                    torch.as_tensor(anchors.ref_end, device=dev),
                    torch.as_tensor(weight, device=dev),
                ],
                dim=-1,
            )[None].to(torch.int32)
            try:
                f, parent = cuda_chain_dp(
                    stacked,
                    torch.tensor([len(anchors)], dtype=torch.int32, device=dev),
                    lookback=max(1, self.cfg.max_lookback),
                    max_gap=self.cfg.max_gap,
                    gap_open=self.cfg.gap_open,
                    gap_extend=self.cfg.gap_extend,
                    log_coeff=self.cfg.log_coeff,
                    bonus=(
                        torch.as_tensor(bonus, dtype=torch.float32, device=dev)[None]
                        if bonus is not None
                        else None
                    ),
                )
                return f[0].cpu().numpy().astype(np.float64), parent[0].cpu().numpy()
            except RuntimeError:
                pass  # kernel unavailable; fall through to the portable path

        return chain_dp_numpy(
            anchors.read_end, anchors.ref_end, weight, self.cfg, bonus=bonus
        )

    def _backtrack(
        self,
        anchors: AnchorSet,
        f: np.ndarray,
        parent: np.ndarray,
        strand: int,
        global_idx: np.ndarray,
    ) -> list[Chain]:
        """Peel chains off the DP table, highest-scoring first.

        Following a chain into anchors already claimed by a better chain yields a
        *partial* chain whose score is the score difference, matching minimap2's
        traceback: a suffix of a better chain is not reported as its own hit.
        """
        chains: list[Chain] = []
        claimed = np.zeros(len(anchors), dtype=bool)

        for i in np.argsort(-f, kind="stable"):
            i = int(i)
            if claimed[i]:
                continue

            members: list[int] = []
            node = i
            while node >= 0 and not claimed[node]:
                claimed[node] = True
                members.append(node)
                node = int(parent[node])
            score = float(f[i] - (f[node] if node >= 0 else 0.0))

            if len(members) < self.cfg.min_chain_anchors or score < self.cfg.min_chain_score:
                continue

            members.reverse()  # DP walks backwards; report in read order
            local = np.asarray(members, dtype=np.int64)
            chains.append(
                Chain(
                    anchor_idx=global_idx[local],
                    score=score,
                    strand=strand,
                    read_start=int(anchors.read_pos[local].min()),
                    read_end=int(anchors.read_end[local].max()),
                    ref_start=int(anchors.ref_pos[local].min()),
                    ref_end=int(anchors.ref_end[local].max()),
                )
            )
        return chains

    # ---- selection ---------------------------------------------------------- #
    def select(self, chains: list[Chain]) -> list[Chain]:
        """Rank chains and mark primaries, dropping redundant secondaries.

        A chain overlapping an accepted primary by more than
        ``secondary_overlap`` of its read span describes the same alignment and is
        dropped; a non-overlapping chain is kept as a secondary candidate (it may
        be a supplementary alignment of a chimeric read).
        """
        if not chains:
            return []

        chains = sorted(chains, key=lambda c: -c.score)
        best = chains[0].score
        cutoff = best * self.cfg.secondary_score_ratio

        kept: list[Chain] = []
        for chain in chains:
            if chain.score < cutoff and kept:
                break
            overlap = max(
                (self._read_overlap(chain, other) for other in kept if other.is_primary),
                default=0.0,
            )
            if overlap > self.cfg.secondary_overlap:
                continue
            chain.is_primary = not kept
            kept.append(chain)
            if len(kept) >= self.cfg.max_chains:
                break
        return kept

    @staticmethod
    def _read_overlap(a: Chain, b: Chain) -> float:
        """Overlap of two chains' read spans as a fraction of the shorter span."""
        lo = max(a.read_start, b.read_start)
        hi = min(a.read_end, b.read_end)
        if hi <= lo:
            return 0.0
        return (hi - lo) / max(min(a.read_span, b.read_span), 1)

    # ---- entry point -------------------------------------------------------- #
    def chain(
        self,
        anchors: AnchorSet,
        ctx: Optional[ChainingContext] = None,
        device: torch.device | str | None = None,
    ) -> list[Chain]:
        """Chain one read's anchors into ranked candidate chains."""
        if len(anchors) == 0:
            return []

        chains: list[Chain] = []
        for strand in (1, -1):
            subset, indices = anchors.for_strand(strand)
            if len(subset) < self.cfg.min_chain_anchors:
                continue
            ordered, order = subset.sorted_by_ref()
            weight = self._weights(ordered, ctx)
            bonus = self._graph_bonus(ordered, ctx)
            f, parent = self._run_dp(ordered, weight, bonus, device=device)
            chains.extend(
                self._backtrack(ordered, f, parent, strand, indices[order])
            )
        return self.select(chains)

    def chain_batch(
        self,
        anchor_sets: Sequence[AnchorSet],
        contexts: Optional[Sequence[Optional[ChainingContext]]] = None,
        device: torch.device | str | None = None,
    ) -> list[list[Chain]]:
        """Chain a batch of reads, sharing one batched DP launch per strand.

        Falls back to the per-read path on CPU, where padding a batch to the
        widest anchor set costs more than it saves.
        """
        contexts = list(contexts or [None] * len(anchor_sets))
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type == "cpu" or len(anchor_sets) == 1:
            return [
                self.chain(anchors, ctx, device=dev)
                for anchors, ctx in zip(anchor_sets, contexts)
            ]

        results: list[list[Chain]] = [[] for _ in anchor_sets]
        for strand in (1, -1):
            per_read = [a.for_strand(strand) for a in anchor_sets]
            ordered_sets, index_maps = [], []
            for (subset, indices) in per_read:
                ordered, order = subset.sorted_by_ref()
                ordered_sets.append(ordered)
                index_maps.append(indices[order])

            counts = [len(o) for o in ordered_sets]
            width = max(counts, default=0)
            if width == 0:
                continue

            B = len(ordered_sets)
            read_end = torch.zeros((B, width), dtype=torch.float32, device=dev)
            ref_end = torch.zeros((B, width), dtype=torch.float32, device=dev)
            weight = torch.zeros((B, width), dtype=torch.float32, device=dev)
            bonus_batch: Optional[torch.Tensor] = None
            lookback = max(1, self.cfg.max_lookback)

            for b, (ordered, ctx) in enumerate(zip(ordered_sets, contexts)):
                n = len(ordered)
                if n == 0:
                    continue
                read_end[b, :n] = torch.as_tensor(ordered.read_end, device=dev)
                ref_end[b, :n] = torch.as_tensor(ordered.ref_end, device=dev)
                weight[b, :n] = torch.as_tensor(self._weights(ordered, ctx), device=dev)
                graph_bonus = self._graph_bonus(ordered, ctx)
                if graph_bonus is not None:
                    if bonus_batch is None:
                        bonus_batch = torch.zeros(
                            (B, width, lookback), dtype=torch.float32, device=dev
                        )
                    bonus_batch[b, :n] = torch.as_tensor(
                        graph_bonus, dtype=torch.float32, device=dev
                    )

            n_anchors = torch.tensor(counts, dtype=torch.long, device=dev)
            f, parent = chain_dp_batched(
                read_end, ref_end, weight, n_anchors, self.cfg, bonus=bonus_batch
            )
            f_np = f.cpu().numpy().astype(np.float64)
            parent_np = parent.cpu().numpy()

            for b, ordered in enumerate(ordered_sets):
                n = counts[b]
                if n < self.cfg.min_chain_anchors:
                    continue
                results[b].extend(
                    self._backtrack(
                        ordered, f_np[b, :n], parent_np[b, :n], strand, index_maps[b]
                    )
                )

        return [self.select(chains) for chains in results]
