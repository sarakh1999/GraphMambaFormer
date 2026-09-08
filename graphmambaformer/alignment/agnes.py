"""Standalone AGNES hybrid seed chainer (Arafat et al., 2025).

This is the opt-in ``chaining.chainer="agnes"`` path: a *paper-faithful*
reimplementation of AGNES that is deliberately independent of the minimap2-style
:class:`~graphmambaformer.alignment.chaining.AffineChainer`. Where the affine
chainer folds AGNES's ideas into its DP as a *pairwise* relaxation (a confidence
gate on seed scores plus additive GNN edge logits), this module keeps AGNES's
own formulation intact so it can be trained and evaluated on its own terms:

* **Seed graph (Eq. 2).** Seeds are nodes; a directed edge ``i -> j`` exists only
  when the two anchors are on the same strand, strictly forward and
  non-overlapping in *both* the read and the genome, and *gap-consistent* — their
  diagonals differ by less than ``gap_threshold``. See :func:`build_seed_graph`.
* **Node classifier.** An :class:`EdgeConv` graph network scores every seed with
  the probability that it lies on the true alignment (:class:`AgnesSeedClassifier`).
* **Chaining DP (Eq. 5).** A longest-path dynamic program over the DAG combines
  node scores and gap-discounted edge weights (:func:`chain_dynamic_program`).
* **Confidence-based method selection (Algorithm 1, lines 7-11).** The chainer
  trusts the classifier only when its score distribution is decisively separated
  (:func:`confidence_metric`); otherwise, and for degenerate graphs
  (``|V| < min_nodes``, ``|V| > max_nodes``, or ``|E| = 0``; lines 3-5), it falls
  back to a pure geometric DP (``PureDP``). See :class:`AgnesChainer`.

The per-seed node features (12-D) are exactly those of
:meth:`~graphmambaformer.alignment.types.AnchorSet.to_seed_features`, so a model
trained on the synthetic dataset's ground-truth seeds accepts anchors produced by
any Stage-1 index; edge features are 8-D geometric descriptors of the step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .types import AnchorSet, Chain

#: Per-seed node feature width — matches ``AnchorSet.to_seed_features`` and the
#: synthetic dataset's ``Seed.features``.
NODE_FEATURE_DIM = 12
#: Per-edge feature width for the seed graph (geometric step descriptors).
EDGE_FEATURE_DIM = 8


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AgnesConfig:
    """Knobs for the AGNES seed graph, classifier, and chaining DP."""

    # -- seed-graph construction (Eq. 2) ------------------------------------- #
    #: An edge is dropped when the two anchors' diagonals differ by at least this
    #: (in bp); i.e. it enforces gap-consistency ``|Δdiagonal| < gap_threshold``.
    gap_threshold: float = 300.0

    # -- degeneracy guards (Algorithm 1, lines 3-5) -------------------------- #
    min_nodes: int = 5      # |V| < min_nodes  -> PureDP fallback
    max_nodes: int = 1000   # |V| > max_nodes  -> PureDP fallback

    # -- chaining DP (Eq. 5) ------------------------------------------------- #
    match_bonus: float = 1.0     # reward for a perfectly gap-consistent edge
    gap_weight: float = 0.01     # linear discount per unit of diagonal discrepancy
    min_chain_anchors: int = 1   # shortest chain the chainer will report

    # -- confidence-based method selection (lines 7-11) --------------------- #
    confidence_threshold: float = 1.0
    high_confidence_prob: float = 0.6
    low_confidence_prob: float = 0.4

    # -- EdgeConv classifier ------------------------------------------------- #
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    # -- neural score -> node weight gate ----------------------------------- #
    logit_gate_gain: float = 0.5
    logit_gate_min: float = 0.1
    logit_gate_max: float = 3.0


# --------------------------------------------------------------------------- #
# Seed graph
# --------------------------------------------------------------------------- #
@dataclass
class SeedGraph:
    """A read's seeds as a DAG: nodes are anchors, edges are collinear steps.

    ``node_features`` is ``(N, NODE_FEATURE_DIM)``; ``edge_index`` is ``(E, 2)``
    of ``(src, dst)`` pairs indexing nodes in the *input anchor order* (so a
    ground-truth label array lines up node-for-node); ``edge_features`` is
    ``(E, EDGE_FEATURE_DIM)``; and ``gap_disc`` is the per-edge ``|Δdiagonal|``
    used by :func:`_edge_weight`.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    gap_disc: np.ndarray
    n_nodes: int
    #: Whether the graph is within the ``|V|`` guards and has at least one edge,
    #: i.e. whether the neural chainer may run (vs. an obligatory PureDP fallback).
    active: bool = True

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[0])


def node_features_from_anchors(
    anchors: AnchorSet, cfg: Optional[AgnesConfig] = None
) -> np.ndarray:
    """The 12-D per-seed node features the classifier consumes.

    Identical to :meth:`AnchorSet.to_seed_features`, so training on the synthetic
    dataset's ``Seed.features`` and inference on live anchors share a feature
    space. ``cfg`` is accepted for signature symmetry and future use.
    """
    return anchors.to_seed_features().astype(np.float32)


def build_seed_graph(
    anchors: AnchorSet, cfg: Optional[AgnesConfig] = None
) -> SeedGraph:
    """Build the AGNES seed DAG (Eq. 2) from an anchor set.

    An edge ``i -> j`` is emitted iff the anchors are (i) on the same strand,
    (ii) strictly forward and non-overlapping in the read (``read_end[i] <=
    read_pos[j]``), (iii) strictly forward and non-overlapping in the genome
    (``ref_end[i] <= ref_pos[j]``), and (iv) gap-consistent (``|diagonal[j] -
    diagonal[i]| < cfg.gap_threshold``). Nodes keep their input order.
    """
    cfg = cfg or AgnesConfig()
    n = len(anchors)
    node_features = node_features_from_anchors(anchors, cfg)

    if n <= 1:
        return SeedGraph(
            node_features=node_features,
            edge_index=np.zeros((0, 2), dtype=np.int64),
            edge_features=np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
            gap_disc=np.zeros(0, dtype=np.float64),
            n_nodes=n,
            active=False,
        )

    read_pos = anchors.read_pos.astype(np.int64)
    read_end = anchors.read_end.astype(np.int64)
    ref_pos = anchors.ref_pos.astype(np.int64)
    ref_end = anchors.ref_end.astype(np.int64)
    strand = anchors.strand.astype(np.int64)
    diag = anchors.diagonal.astype(np.int64)
    length = anchors.length.astype(np.float64)

    # Pairwise Eq. 2 constraints, vectorized over the (i, j) grid. The |V| guard
    # in AgnesChainer keeps this O(N^2) build off pathologically large graphs.
    same_strand = strand[:, None] == strand[None, :]
    forward_read = read_end[:, None] <= read_pos[None, :]   # implies read_pos[i] < read_pos[j]
    forward_ref = ref_end[:, None] <= ref_pos[None, :]
    disc = np.abs(diag[None, :] - diag[:, None])
    gap_ok = disc < cfg.gap_threshold
    mask = same_strand & forward_read & forward_ref & gap_ok

    src, dst = np.nonzero(mask)
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    gap_disc = disc[src, dst].astype(np.float64)

    edge_index = np.stack([src, dst], axis=1).astype(np.int64)
    edge_features = _edge_features(
        anchors, src, dst, read_pos, read_end, ref_pos, ref_end, length, gap_disc
    )

    active = (cfg.min_nodes <= n <= cfg.max_nodes) and src.size > 0
    return SeedGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        gap_disc=gap_disc,
        n_nodes=n,
        active=active,
    )


def _edge_features(
    anchors: AnchorSet,
    src: np.ndarray,
    dst: np.ndarray,
    read_pos: np.ndarray,
    read_end: np.ndarray,
    ref_pos: np.ndarray,
    ref_end: np.ndarray,
    length: np.ndarray,
    gap_disc: np.ndarray,
) -> np.ndarray:
    """8-D geometric descriptor of every edge (read/genome gaps, spans, lengths)."""
    if src.size == 0:
        return np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    rl = float(max(anchors.read_len, 1))
    gl = float(max(anchors.ref_len, 1))
    read_gap = (read_pos[dst] - read_end[src]).astype(np.float64)
    ref_gap = (ref_pos[dst] - ref_end[src]).astype(np.float64)
    read_span = (read_pos[dst] - read_pos[src]).astype(np.float64)
    ref_span = (ref_pos[dst] - ref_pos[src]).astype(np.float64)

    feats = np.stack(
        [
            read_gap / rl,
            ref_gap / gl,
            gap_disc / gl,
            np.log1p(gap_disc),
            length[src] / rl,
            length[dst] / rl,
            read_span / rl,
            ref_span / gl,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# Chaining DP (Eq. 5)
# --------------------------------------------------------------------------- #
def _edge_weight(gap_disc: np.ndarray, cfg: AgnesConfig) -> np.ndarray:
    """Per-edge chaining reward: ``match_bonus`` discounted by gap discrepancy."""
    gap_disc = np.asarray(gap_disc, dtype=np.float64)
    return cfg.match_bonus - cfg.gap_weight * gap_disc


def chain_dynamic_program(
    graph: SeedGraph, node_scores: np.ndarray, cfg: AgnesConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Longest-path DP over the seed DAG (AGNES Eq. 5).

    ``dp[i] = f[i] + max(0, max_{i->j} (w(i, j) + dp[j]))`` where ``f`` is
    ``node_scores`` and ``w`` is :func:`_edge_weight`. Successors are evaluated
    before their predecessors via a reverse topological order, so the recurrence
    is exact. Returns ``(dp, parent)`` where ``parent[i]`` is the chosen successor
    of node ``i`` (``-1`` when the chain ends there).
    """
    n = int(graph.n_nodes)
    f = np.asarray(node_scores, dtype=np.float64)
    dp = f.copy()
    parent = np.full(n, -1, dtype=np.int64)
    if n == 0 or graph.n_edges == 0:
        return dp, parent

    weights = _edge_weight(graph.gap_disc, cfg)
    succ: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    for (s, d), w in zip(graph.edge_index, weights):
        s, d = int(s), int(d)
        succ[s].append((d, float(w)))
        indeg[d] += 1

    # Kahn topological order (the DAG is acyclic: every edge increases read_pos).
    queue = deque(int(i) for i in np.flatnonzero(indeg == 0))
    topo: list[int] = []
    remaining = indeg.copy()
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v, _ in succ[u]:
            remaining[v] -= 1
            if remaining[v] == 0:
                queue.append(v)

    for u in reversed(topo):
        best_gain = 0.0
        best_succ = -1
        for v, w in succ[u]:
            gain = w + dp[v]
            if gain > best_gain:
                best_gain = gain
                best_succ = v
        if best_succ >= 0:
            dp[u] = f[u] + best_gain
            parent[u] = best_succ
    return dp, parent


# --------------------------------------------------------------------------- #
# Confidence-based method selection (Algorithm 1, lines 7-11)
# --------------------------------------------------------------------------- #
def confidence_metric(probs: np.ndarray, cfg: AgnesConfig) -> float:
    """Scale-free separation of the seed-score distribution ``(μ_high - μ_low) / σ``.

    ``μ_high`` / ``μ_low`` are the mean probabilities of the confidently-good
    (``> high_confidence_prob``) and confidently-bad (``< low_confidence_prob``)
    seeds; dividing their gap by the overall spread measures how decisively the
    classifier separates true from spurious seeds. Returns ``0.0`` for an empty or
    zero-variance distribution.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.size == 0:
        return 0.0
    sigma = float(probs.std())
    if sigma <= 1e-9:
        return 0.0
    high = probs[probs > cfg.high_confidence_prob]
    low = probs[probs < cfg.low_confidence_prob]
    mu_high = float(high.mean()) if high.size else 0.0
    mu_low = float(low.mean()) if low.size else 0.0
    return (mu_high - mu_low) / sigma


# --------------------------------------------------------------------------- #
# EdgeConv classifier
# --------------------------------------------------------------------------- #
class EdgeConv(nn.Module):
    """A directed EdgeConv layer that aggregates each node's predecessors.

    For every edge ``s -> d`` the message ``MLP([h_d, h_s - h_d, edge_feat])`` is
    mean-aggregated into node ``d`` and added to a learned self-transform of
    ``h_d``. Nodes with no incoming edge keep only their self-transform, so
    isolated seeds are handled gracefully.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.self_transform = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        out = self.self_transform(x)
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index[0], edge_index[1]
        msg = self.message(torch.cat([x[dst], x[src] - x[dst], edge_features], dim=-1))
        agg = torch.zeros(x.shape[0], out.shape[1], dtype=out.dtype, device=x.device)
        agg = agg.index_add(0, dst, msg)
        deg = torch.zeros(x.shape[0], dtype=out.dtype, device=x.device)
        deg = deg.index_add(0, dst, torch.ones(dst.shape[0], dtype=out.dtype, device=x.device))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return out + agg / deg


class AgnesSeedClassifier(nn.Module):
    """EdgeConv network that scores each seed as true (on-alignment) vs spurious."""

    def __init__(self, cfg: Optional[AgnesConfig] = None):
        super().__init__()
        cfg = cfg or AgnesConfig()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.input = nn.Linear(NODE_FEATURE_DIM, h)
        self.layers = nn.ModuleList(
            EdgeConv(h, h, EDGE_FEATURE_DIM, dropout=cfg.dropout)
            for _ in range(max(1, cfg.num_layers))
        )
        self.norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        if edge_features is None or edge_features.numel() == 0:
            n_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
            edge_features = torch.zeros(
                (n_edges, EDGE_FEATURE_DIM), dtype=torch.float32, device=x.device
            )
        else:
            edge_features = edge_features.to(torch.float32)
        h = torch.relu(self.input(x))
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index, edge_features))
        h = self.norm(h)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, graph: SeedGraph) -> np.ndarray:
        """Per-seed probability in ``[0, 1]`` for the seeds of ``graph``."""
        device = next(self.parameters()).device
        x = torch.as_tensor(graph.node_features, dtype=torch.float32, device=device)
        if graph.n_edges:
            edge_index = torch.as_tensor(graph.edge_index.T, dtype=torch.long, device=device)
            edge_features = torch.as_tensor(
                graph.edge_features, dtype=torch.float32, device=device
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_features = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)
        logits = self.forward(x, edge_index, edge_features)
        return torch.sigmoid(logits).detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Chainer
# --------------------------------------------------------------------------- #
@dataclass
class AgnesResult:
    """Outcome of :meth:`AgnesChainer.run`."""

    best: Optional[Chain]
    #: ``"neural"`` (classifier-guided), ``"pure_dp"`` (geometric DP), or
    #: ``"fallback"`` (degenerate graph forced to PureDP).
    mode: str
    chains: list = field(default_factory=list)
    node_scores: Optional[np.ndarray] = None
    confidence: float = 0.0
    trusted: bool = False


class AgnesChainer:
    """AGNES Algorithm 1: classify seeds, then chain with confidence-based selection.

    With ``model=None`` the chainer always runs the geometric PureDP baseline.
    With a trained :class:`AgnesSeedClassifier` it classifies the seed graph and
    trusts the scores to steer node weights only when the graph is well-formed and
    the score distribution is decisively separated; degenerate graphs and
    under-confident distributions fall back to PureDP.
    """

    def __init__(self, model: Optional[AgnesSeedClassifier] = None, cfg: Optional[AgnesConfig] = None):
        self.model = model
        self.cfg = cfg or AgnesConfig()

    def run(self, anchors: AnchorSet) -> AgnesResult:
        cfg = self.cfg
        graph = build_seed_graph(anchors, cfg)
        n = graph.n_nodes
        base = anchors.length.astype(np.float64) if n else np.zeros(0, dtype=np.float64)

        node_scores: Optional[np.ndarray] = None
        confidence = 0.0
        trusted = False

        if self.model is None:
            mode, f = "pure_dp", base
        elif n < cfg.min_nodes or n > cfg.max_nodes or graph.n_edges == 0:
            # Algorithm 1 lines 3-5: degenerate graph -> obligatory PureDP.
            mode, f = "fallback", base
        else:
            node_scores = self.model.predict_proba(graph)
            confidence = confidence_metric(node_scores, cfg)
            trusted = confidence > cfg.confidence_threshold
            if trusted:
                mode, f = "neural", base * self._logit_gate(node_scores)
            else:
                mode, f = "pure_dp", base

        best = self._best_chain(graph, f, anchors) if n else None
        return AgnesResult(
            best=best,
            mode=mode,
            chains=[best] if best is not None else [],
            node_scores=node_scores,
            confidence=confidence,
            trusted=trusted,
        )

    def _logit_gate(self, probs: np.ndarray) -> np.ndarray:
        """Length-preserving multiplicative gate from logit-transformed seed probs."""
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        logit = np.log(p / (1.0 - p))
        return np.clip(
            1.0 + self.cfg.logit_gate_gain * logit,
            self.cfg.logit_gate_min,
            self.cfg.logit_gate_max,
        )

    def _best_chain(
        self, graph: SeedGraph, node_scores: np.ndarray, anchors: AnchorSet
    ) -> Optional[Chain]:
        """Trace the highest-scoring path out of the DP into a :class:`Chain`."""
        n = graph.n_nodes
        if n == 0:
            return None
        dp, parent = chain_dynamic_program(graph, node_scores, self.cfg)
        start = int(np.argmax(dp))

        members: list[int] = []
        seen: set[int] = set()
        node = start
        while node != -1 and node not in seen:
            seen.add(node)
            members.append(node)
            node = int(parent[node])

        if len(members) < self.cfg.min_chain_anchors:
            return None
        idx = np.asarray(members, dtype=np.int64)  # already in read order (head first)
        return Chain(
            anchor_idx=idx,
            score=float(dp[start]),
            strand=int(anchors.strand[idx[0]]),
            read_start=int(anchors.read_pos[idx].min()),
            read_end=int(anchors.read_end[idx].max()),
            ref_start=int(anchors.ref_pos[idx].min()),
            ref_end=int(anchors.ref_end[idx].max()),
        )
