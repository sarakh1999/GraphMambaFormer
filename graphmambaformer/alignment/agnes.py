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
  read/genome gaps differ by less than ``gap_threshold`` (``|g_r - g_g| < tau``,
  ``tau = 500 bp`` for nanopore). See :func:`build_seed_graph`.
* **Node classifier (Eqs. 1, 6).** A three-layer EdgeConv graph network with
  progressively wider hidden dims ``(64, 128, 128)``, max neighbour aggregation,
  a per-node skip transform, BatchNorm, and a ``[128, 64, 1]`` output MLP scores
  every seed with the probability that it lies on the true alignment
  (:class:`AgnesSeedClassifier`). Node features are 12-D (Eq. 1); edge features
  are 8-D (Eq. 2).
* **Chaining DP (Eq. 5).** A longest-path dynamic program over the DAG combines
  node scores and a reciprocal gap-penalty edge weight
  ``w = 1/(1 + a|g_r-g_g| + b log(|g_r-g_g|+1))`` (:func:`chain_dynamic_program`).
* **Confidence-based method selection (Algorithm 1, lines 7-11).** The chainer
  trusts the classifier only when its score distribution is decisively separated
  (:func:`confidence_metric` ``> tau = 0.7``); otherwise, and for degenerate
  graphs (``|V| < min_nodes``, ``|V| > max_nodes``, or ``|E| = 0``; lines 3-5), it
  falls back to a pure geometric DP with uniform node scores (``f(s_i) = 1``,
  ``PureDP``). When trusted, node scores are the paper's logit
  ``f(s_i) = log(p_i/(1-p_i))``. See :class:`AgnesChainer`.

The per-seed node features (12-D) are exactly those of
:meth:`~graphmambaformer.alignment.types.AnchorSet.to_seed_features`, so a model
trained on the synthetic dataset's ground-truth seeds accepts anchors produced by
any Stage-1 index. Edge features follow Eq. 2's geometric descriptors; the three
quality/signal slots the paper fills from raw squiggle data (signal continuity,
repeat overlap, hash-quality difference) are approximated from anchor geometry or
left at zero, since those signals are not available for arbitrary Stage-1 anchors
— the same intrinsic-only trade-off :meth:`AnchorSet.to_seed_features` makes for
node features, and applied identically at train and inference time.
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
#: Per-edge feature width for the seed graph (geometric step descriptors, Eq. 2).
EDGE_FEATURE_DIM = 8


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AgnesConfig:
    """Knobs for the AGNES seed graph, classifier, and chaining DP.

    The field set mirrors the ``agnes_config`` payload written by
    ``scripts/train_agnes.py``, so a checkpoint's stored config round-trips back
    through ``AgnesConfig(**ckpt["agnes_config"])`` unchanged.
    """

    # -- seed-graph construction (Eq. 2) ------------------------------------- #
    #: An edge is dropped when the read/genome gaps differ by at least this many
    #: bp; i.e. it enforces gap-consistency ``|g_r - g_g| < gap_threshold``. The
    #: paper uses ``tau = 500`` for nanopore.
    gap_threshold: float = 500.0
    #: Cap on the number of forward successors kept per node (``0`` = unlimited).
    #: Bounds the ``O(N^2)`` edge set on pathologically dense graphs.
    max_neighbors: int = 0

    # -- degeneracy guards (Algorithm 1, lines 3-5) -------------------------- #
    min_nodes: int = 5      # |V| < min_nodes  -> PureDP fallback
    max_nodes: int = 1000   # |V| > max_nodes  -> PureDP fallback

    # -- confidence-based method selection (lines 7-11) --------------------- #
    high_confidence_prob: float = 0.7   # p_i > this  -> confidently-good seed
    low_confidence_prob: float = 0.3    # p_i < this  -> confidently-bad seed
    confidence_threshold: float = 0.7   # trust the GNN only when conf > this (tau)

    # -- chaining DP (Eq. 5): reciprocal gap penalty ------------------------- #
    #: Edge weight ``w = 1/(1 + gap_penalty_a*|d| + gap_penalty_b*log(|d|+1))``
    #: where ``d = g_r - g_g``. Bounded in ``(0, 1]`` and always positive, so a
    #: large but consistent gap is discounted rather than driven negative.
    gap_penalty_a: float = 0.01
    gap_penalty_b: float = 0.5

    # -- edge feature p_gap = exp(-p_gap_a*|d| - p_gap_b*log(g_r+1)) --------- #
    p_gap_a: float = 0.01
    p_gap_b: float = 0.5

    #: Shortest chain the chainer will report (paper ``k_min = 3``).
    min_chain_anchors: int = 3
    #: Symmetric clip on the logit node score ``log(p/(1-p))`` when trusted.
    logit_clip: float = 8.0

    # -- EdgeConv classifier (Eqs. 1, 6) ------------------------------------- #
    #: Output width of each EdgeConv layer; length = number of layers.
    hidden_dims: tuple[int, ...] = (64, 128, 128)
    #: Hidden widths of the per-node output MLP (a final ``-> 1`` is appended).
    out_mlp_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.3

    # -- chain selection ----------------------------------------------------- #
    secondary_overlap: float = 0.5
    max_chains: int = 8


# --------------------------------------------------------------------------- #
# Seed graph
# --------------------------------------------------------------------------- #
@dataclass
class SeedGraph:
    """A read's seeds as a DAG: nodes are anchors, edges are collinear steps.

    ``node_features`` is ``(N, NODE_FEATURE_DIM)``; ``edge_index`` is ``(E, 2)``
    of ``(src, dst)`` pairs indexing nodes in the *input anchor order* (so a
    ground-truth label array lines up node-for-node); ``edge_features`` is
    ``(E, EDGE_FEATURE_DIM)``; and ``gap_disc`` is the per-edge ``|g_r - g_g|``
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
    """The 12-D per-seed node features the classifier consumes (Eq. 1).

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
    (``ref_end[i] <= ref_pos[j]``), and (iv) gap-consistent (``|g_r - g_g| <
    cfg.gap_threshold``). Nodes keep their input order.
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
    length = anchors.length.astype(np.float64)

    # Pairwise Eq. 2 constraints, vectorized over the (i, j) grid. The |V| guard
    # in AgnesChainer keeps this O(N^2) build off pathologically large graphs.
    same_strand = strand[:, None] == strand[None, :]
    forward_read = read_end[:, None] <= read_pos[None, :]   # implies read_pos[i] < read_pos[j]
    forward_ref = ref_end[:, None] <= ref_pos[None, :]
    # Gap consistency uses the read/genome gaps between the anchors, |g_r - g_g|.
    read_gap = read_pos[None, :] - read_end[:, None]
    ref_gap = ref_pos[None, :] - ref_end[:, None]
    disc = np.abs(read_gap - ref_gap)
    gap_ok = disc < cfg.gap_threshold
    mask = same_strand & forward_read & forward_ref & gap_ok

    src, dst = np.nonzero(mask)
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)

    if cfg.max_neighbors and src.size:
        src, dst = _cap_out_degree(src, dst, read_pos, cfg.max_neighbors)

    gap_disc = disc[src, dst].astype(np.float64)
    edge_index = np.stack([src, dst], axis=1).astype(np.int64)
    edge_features = _edge_features(
        anchors, src, dst, read_pos, read_end, ref_pos, ref_end, length, gap_disc, cfg
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


def _cap_out_degree(
    src: np.ndarray, dst: np.ndarray, read_pos: np.ndarray, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the ``max_neighbors`` nearest (in read order) successors per node."""
    keep = np.ones(src.size, dtype=bool)
    # Stable sort by (src, read distance to dst) so the closest successors lead.
    dist = read_pos[dst] - read_pos[src]
    order = np.lexsort((dist, src))
    counts: dict[int, int] = {}
    for pos in order:
        s = int(src[pos])
        c = counts.get(s, 0)
        if c >= max_neighbors:
            keep[pos] = False
        else:
            counts[s] = c + 1
    return src[keep], dst[keep]


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
    cfg: AgnesConfig,
) -> np.ndarray:
    """8-D geometric descriptor of every edge (Eq. 2).

    Columns: ``[g_r/1000, g_g/1000, kappa_gap, p_gap, delta_dir, kappa_len, rep,
    uniq_diff]`` where ``g_r``/``g_g`` are the read/genome gaps, ``kappa_gap`` is
    gap consistency, ``p_gap`` the RawHash2-style penalty, ``delta_dir`` the
    forward-direction indicator, and the last three are geometric stand-ins for
    the paper's signal-continuity / repeat / hash-quality slots (which need
    squiggle data unavailable for arbitrary anchors): match-length continuity,
    a zero repeat flag, and a uniqueness-proxy difference.
    """
    if src.size == 0:
        return np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    g_r = (read_pos[dst] - read_end[src]).astype(np.float64)   # >= 0 (forward, non-overlap)
    g_g = (ref_pos[dst] - ref_end[src]).astype(np.float64)     # >= 0
    disc = np.asarray(gap_disc, dtype=np.float64)              # |g_r - g_g|

    kappa_gap = 1.0 - disc / np.maximum(np.maximum(g_r, g_g), 1.0)
    p_gap = np.exp(-cfg.p_gap_a * disc - cfg.p_gap_b * np.log(g_r + 1.0))
    delta_dir = (ref_pos[dst] > ref_pos[src]).astype(np.float64)

    len_s = length[src].astype(np.float64)
    len_d = length[dst].astype(np.float64)
    kappa_len = 1.0 - np.abs(len_s - len_d) / np.maximum(np.maximum(len_s, len_d), 1.0)
    uniq_diff = np.abs(1.0 / np.maximum(len_s, 1.0) - 1.0 / np.maximum(len_d, 1.0))

    feats = np.stack(
        [
            g_r / 1000.0,
            g_g / 1000.0,
            kappa_gap,
            p_gap,
            delta_dir,
            kappa_len,
            np.zeros_like(g_r),  # repeat overlap: unavailable for arbitrary anchors
            uniq_diff,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# Chaining DP (Eq. 5)
# --------------------------------------------------------------------------- #
def _edge_weight(gap_disc: np.ndarray, cfg: AgnesConfig) -> np.ndarray:
    """Reciprocal gap-penalty edge reward (AGNES PureDP, Eq. 5).

    ``w = 1/(1 + a|d| + b log(|d|+1))`` with ``d = g_r - g_g``. Bounded in
    ``(0, 1]`` and monotonically decreasing in the gap discrepancy, so a
    collinear step scores ``1`` and a large-but-consistent gap is discounted
    toward — but never below — zero.
    """
    d = np.asarray(gap_disc, dtype=np.float64)
    return 1.0 / (1.0 + cfg.gap_penalty_a * d + cfg.gap_penalty_b * np.log(d + 1.0))


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
    """Scale-free separation of the seed-score distribution ``(mu_high - mu_low)/sigma``.

    ``mu_high`` / ``mu_low`` are the mean probabilities of the confidently-good
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
# EdgeConv classifier (Eqs. 1, 6)
# --------------------------------------------------------------------------- #
class EdgeConv(nn.Module):
    """A max-aggregating EdgeConv layer over the seed graph's undirected edges.

    For every (undirected) neighbour ``j`` of node ``i`` the message
    ``MLP([h_i, h_j - h_i, e_ij])`` is computed, and the element-wise **max** over
    a node's neighbours is added to a learned self-transform ``skip(h_i)`` (paper
    Eq. 6). Nodes with no neighbour keep only ``skip(h_i)``, so isolated seeds are
    handled gracefully.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        out = self.skip(x)
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index[0], edge_index[1]
        # Undirected neighbourhood N(i) = {j : (i,j) or (j,i) in E}: each stored
        # edge contributes a message to *both* endpoints (paper Eq. 3 / Eq. 6).
        msg_into_dst = self.mlp(torch.cat([x[dst], x[src] - x[dst], edge_features], dim=-1))
        msg_into_src = self.mlp(torch.cat([x[src], x[dst] - x[src], edge_features], dim=-1))
        target = torch.cat([dst, src])
        msg = torch.cat([msg_into_dst, msg_into_src], dim=0)

        agg = torch.full(
            (x.shape[0], out.shape[1]), float("-inf"), dtype=out.dtype, device=x.device
        )
        agg.index_reduce_(0, target, msg, reduce="amax", include_self=False)
        agg = torch.where(torch.isinf(agg), torch.zeros_like(agg), agg)
        return out + agg


class AgnesSeedClassifier(nn.Module):
    """EdgeConv network that scores each seed as true (on-alignment) vs spurious.

    Three EdgeConv layers with widths ``cfg.hidden_dims`` (``(64, 128, 128)`` in
    the paper), each followed by BatchNorm and ReLU, then a per-node output MLP
    with hidden widths ``cfg.out_mlp_dims`` (``(128, 64)``) and a final ``-> 1``
    logit. The initial node embedding is the raw 12-D feature vector (Eq. 1).
    """

    def __init__(self, cfg: Optional[AgnesConfig] = None):
        super().__init__()
        cfg = cfg or AgnesConfig()
        self.cfg = cfg
        hidden = tuple(int(h) for h in cfg.hidden_dims)
        dims = [NODE_FEATURE_DIM, *hidden]
        # Input normalization. The 12-D node features (Eq. 1) and 8-D edge
        # features (Eq. 2) mix wildly different scales -- raw base gaps
        # (thousands), fractions in [0, 1], and binary flags. Feeding them raw
        # into the first EdgeConv makes the first layer slow and unstable to fit
        # (the gap columns dominate the gradient), which showed up as a borderline
        # seed-classification AUC. A BatchNorm on the inputs standardizes every
        # column to zero-mean/unit-variance before the network sees it, which is
        # the standard EdgeConv/DGCNN input treatment and makes learning robust.
        self.input_norm = nn.BatchNorm1d(NODE_FEATURE_DIM)
        self.edge_norm = nn.BatchNorm1d(EDGE_FEATURE_DIM)
        self.layers = nn.ModuleList(
            EdgeConv(dims[i], dims[i + 1], EDGE_FEATURE_DIM, dropout=cfg.dropout)
            for i in range(len(hidden))
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden)

        out_layers: list[nn.Module] = []
        prev = hidden[-1]
        for width in cfg.out_mlp_dims:
            out_layers += [nn.Linear(prev, int(width)), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = int(width)
        out_layers.append(nn.Linear(prev, 1))
        self.out = nn.Sequential(*out_layers)

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

        # Standardize inputs (see __init__). BatchNorm needs >1 row in train
        # mode; a batch is a disjoint union of many graphs so nodes/edges are
        # plentiful there, while inference runs in eval mode on running stats and
        # is safe for a single-node/edge graph. Guard the (train-mode) 0/1-edge
        # corner so an edgeless batch can't trip BatchNorm.
        h = self.input_norm(x)
        if edge_features.shape[0] > (1 if self.training else 0):
            edge_features = self.edge_norm(edge_features)
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, edge_features)
            # BatchNorm needs >1 sample in train mode; a size-1 graph only occurs
            # at inference (eval mode uses running stats), so this is safe.
            h = norm(h)
            h = torch.relu(h)
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, graph: SeedGraph) -> np.ndarray:
        """Per-seed probability in ``[0, 1]`` for the seeds of ``graph``."""
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            x = torch.as_tensor(graph.node_features, dtype=torch.float32, device=device)
            if graph.n_edges:
                edge_index = torch.as_tensor(graph.edge_index.T, dtype=torch.long, device=device)
                edge_features = torch.as_tensor(
                    graph.edge_features, dtype=torch.float32, device=device
                )
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_features = torch.zeros(
                    (0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device
                )
            logits = self.forward(x, edge_index, edge_features)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        finally:
            self.train(was_training)
        return probs


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

    With ``model=None`` the chainer always runs the geometric PureDP baseline
    (uniform node scores ``f(s_i) = 1``). With a trained
    :class:`AgnesSeedClassifier` it classifies the seed graph and, only when the
    graph is well-formed *and* the score distribution is decisively separated,
    steers the DP with the paper's logit node scores ``f(s_i) = log(p_i/(1-p_i))``;
    degenerate graphs and under-confident distributions fall back to PureDP.
    """

    def __init__(self, model: Optional[AgnesSeedClassifier] = None, cfg: Optional[AgnesConfig] = None):
        self.model = model
        self.cfg = cfg or AgnesConfig()

    def run(self, anchors: AnchorSet) -> AgnesResult:
        cfg = self.cfg
        graph = build_seed_graph(anchors, cfg)
        n = graph.n_nodes
        uniform = np.ones(n, dtype=np.float64) if n else np.zeros(0, dtype=np.float64)

        node_scores: Optional[np.ndarray] = None
        confidence = 0.0
        trusted = False

        if self.model is None:
            mode, f = "pure_dp", uniform
        elif n < cfg.min_nodes or n > cfg.max_nodes or graph.n_edges == 0:
            # Algorithm 1 lines 3-5: degenerate graph -> obligatory PureDP.
            mode, f = "fallback", uniform
        else:
            node_scores = self.model.predict_proba(graph)
            confidence = confidence_metric(node_scores, cfg)
            trusted = confidence > cfg.confidence_threshold
            if trusted:
                # Algorithm 1 line 12: node score is the (clipped) logit.
                mode, f = "neural", self._logit_scores(node_scores)
            else:
                mode, f = "pure_dp", uniform

        best = self._best_chain(graph, f, anchors) if n else None
        return AgnesResult(
            best=best,
            mode=mode,
            chains=[best] if best is not None else [],
            node_scores=node_scores,
            confidence=confidence,
            trusted=trusted,
        )

    def _logit_scores(self, probs: np.ndarray) -> np.ndarray:
        """Clipped logit node scores ``f(s_i) = log(p_i/(1-p_i))`` (Algorithm 1)."""
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        logit = np.log(p / (1.0 - p))
        return np.clip(logit, -self.cfg.logit_clip, self.cfg.logit_clip)

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
