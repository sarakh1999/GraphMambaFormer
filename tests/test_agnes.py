"""Tests for the standalone AGNES hybrid seed chainer (Arafat et al., 2025).

Verifies each piece of Algorithm 1 independently, then that the classifier
actually learns to separate true from spurious seeds on synthetic ground truth:

- seed-graph construction obeys the three spatial-consistency constraints (Eq. 2)
- the chaining DP (Eq. 5) matches a brute-force longest-path over the DAG
- the confidence metric and its gate behave as specified (lines 7-11)
- degenerate graphs (|V|<5, |V|>1000, |E|=0) fall back to PureDP (lines 3-5)
- the EdgeConv classifier has the right shapes and trains to a good AUC

Pytest-compatible but self-contained.
"""

import dataclasses

import numpy as np
import torch

# AGNES graphs are tiny; one intra-op thread avoids oversubscribing every core
# (which, on a shared/loaded node, turns sub-second work into minutes).
torch.set_num_threads(1)

from graphmambaformer.alignment.agnes import (
    AgnesChainer,
    AgnesConfig,
    AgnesSeedClassifier,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_seed_graph,
    chain_dynamic_program,
    confidence_metric,
)
from graphmambaformer.alignment.agnes import _edge_weight
from graphmambaformer.alignment.types import AnchorSet


def _anchors(read_pos, ref_pos, length, strand, read_len=200, ref_len=2000):
    return AnchorSet.from_lists(
        read_pos=read_pos,
        ref_pos=ref_pos,
        length=[length] * len(read_pos),
        strand=strand,
        read_len=read_len,
        ref_len=ref_len,
    )


# --------------------------------------------------------------------------- #
# Graph construction (Eq. 2 constraints)
# --------------------------------------------------------------------------- #
def test_build_seed_graph_respects_constraints():
    cfg = AgnesConfig(gap_threshold=50)
    # A0,A1 co-linear (edge). A2 far off-diagonal (gap inconsistent, no edge).
    # A3 overlaps A0 in read (read-order violation). A4 is reverse strand.
    anchors = _anchors(
        read_pos=[0, 20, 40, 5, 60],
        ref_pos=[100, 120, 400, 100, 130],
        length=10,
        strand=[1, 1, 1, 1, -1],
    )
    g = build_seed_graph(anchors, cfg)

    assert g.node_features.shape == (5, NODE_FEATURE_DIM)
    assert g.edge_features.shape[1] == EDGE_FEATURE_DIM
    edges = {(int(s), int(d)) for s, d in g.edge_index}

    # co-linear, non-overlapping, gap-consistent pair -> edge exists
    assert (0, 1) in edges
    # gap-consistency violation (|gap_r - gap_g| >= 50) -> no edge into A2
    assert (1, 2) not in edges and (0, 2) not in edges
    # read-order / non-overlap violation: A3 (read 5-15) overlaps A0 (read 0-10),
    # so A0 -> A3 is forbidden (A3 may still chain *forward* to a later seed).
    assert (0, 3) not in edges
    # cross-strand edges are forbidden (A4 is reverse strand)
    assert all(4 not in e for e in edges)
    # every emitted edge is forward, non-overlapping in both coordinates, and
    # gap-consistent (the three AGNES Eq. 2 constraints).
    for (s, d), disc in zip(g.edge_index, g.gap_disc):
        assert anchors.read_pos[s] < anchors.read_pos[d]
        assert anchors.read_end[s] <= anchors.read_pos[d]  # no read overlap
        assert anchors.ref_end[s] <= anchors.ref_pos[d]  # no genome overlap
        assert disc < cfg.gap_threshold


def test_build_seed_graph_tiny_is_edgeless():
    g = build_seed_graph(_anchors([0], [10], 5, [1]))
    assert g.n_nodes == 1 and g.n_edges == 0


# --------------------------------------------------------------------------- #
# Dynamic program (Eq. 5) vs brute force
# --------------------------------------------------------------------------- #
def _brute_force_best(graph, f, cfg):
    weights = _edge_weight(graph.gap_disc, cfg)
    out_edges: dict[int, list[tuple[int, float]]] = {i: [] for i in range(graph.n_nodes)}
    for (s, d), w in zip(graph.edge_index, weights):
        out_edges[int(s)].append((int(d), float(w)))
    memo: dict[int, float] = {}

    def best_from(i: int) -> float:
        if i in memo:
            return memo[i]
        best = f[i]
        for d, w in out_edges[i]:
            best = max(best, f[i] + w + best_from(d))
        memo[i] = best
        return best

    return max((best_from(i) for i in range(graph.n_nodes)), default=0.0)


def test_dp_matches_bruteforce_uniform_scores():
    cfg = AgnesConfig(gap_threshold=200)
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(4, 12))
        # A noisy but mostly-collinear anchor set so the DAG has real branching.
        read_pos = np.sort(rng.integers(0, 300, size=n))
        ref_pos = read_pos * 1 + rng.integers(0, 40, size=n) + 500
        anchors = _anchors(
            list(map(int, read_pos)),
            list(map(int, ref_pos)),
            10,
            [1] * n,
            read_len=400,
            ref_len=4000,
        )
        g = build_seed_graph(anchors, cfg)
        f = np.ones(g.n_nodes, dtype=np.float64)
        dp, _ = chain_dynamic_program(g, f, cfg)
        assert np.isclose(float(dp.max()), _brute_force_best(g, f, cfg), atol=1e-6)


def test_dp_recovers_the_collinear_chain():
    cfg = AgnesConfig(gap_threshold=100, min_chain_anchors=3)
    anchors = _anchors(
        read_pos=[0, 20, 40, 60, 80],
        ref_pos=[500, 520, 540, 560, 580],  # perfectly collinear (diagonal 500)
        length=10,
        strand=[1] * 5,
    )
    chainer = AgnesChainer(model=None, cfg=cfg)  # PureDP baseline
    result = chainer.run(anchors)
    assert result.best is not None
    assert list(result.best.anchor_idx) == [0, 1, 2, 3, 4]
    assert result.mode == "pure_dp"


# --------------------------------------------------------------------------- #
# Confidence-based method selection (lines 7-11)
# --------------------------------------------------------------------------- #
def test_confidence_metric_separates():
    cfg = AgnesConfig()
    separated = np.array([0.95, 0.9, 0.88, 0.1, 0.05, 0.12], dtype=np.float32)
    flat = np.array([0.5, 0.51, 0.49, 0.5, 0.52, 0.48], dtype=np.float32)
    assert confidence_metric(separated, cfg) > cfg.confidence_threshold
    assert confidence_metric(flat, cfg) < cfg.confidence_threshold
    assert confidence_metric(np.zeros(0), cfg) == 0.0


# --------------------------------------------------------------------------- #
# Degenerate fallback (lines 3-5)
# --------------------------------------------------------------------------- #
def test_fallback_when_too_few_nodes():
    cfg = AgnesConfig(min_nodes=5)
    model = AgnesSeedClassifier(cfg)
    anchors = _anchors([0, 20, 40], [500, 520, 540], 10, [1, 1, 1])  # n=3 < 5
    result = AgnesChainer(model=model, cfg=cfg).run(anchors)
    assert result.mode == "fallback"


def test_fallback_when_no_edges():
    cfg = AgnesConfig(min_nodes=3, gap_threshold=5)
    model = AgnesSeedClassifier(cfg)
    # Six anchors whose diagonals are all wildly inconsistent -> no valid edges.
    anchors = _anchors(
        read_pos=[0, 20, 40, 60, 80, 100],
        ref_pos=[100, 900, 200, 1500, 50, 1900],
        length=10,
        strand=[1] * 6,
    )
    g = build_seed_graph(anchors, cfg)
    assert g.n_edges == 0
    result = AgnesChainer(model=model, cfg=cfg).run(anchors)
    assert result.mode == "fallback"


# --------------------------------------------------------------------------- #
# Classifier shapes
# --------------------------------------------------------------------------- #
def test_classifier_forward_shapes():
    cfg = AgnesConfig()
    model = AgnesSeedClassifier(cfg).eval()
    anchors = _anchors(
        read_pos=[0, 20, 40, 60, 80],
        ref_pos=[500, 520, 540, 560, 580],
        length=10,
        strand=[1] * 5,
    )
    g = build_seed_graph(anchors, cfg)
    x = torch.as_tensor(g.node_features)
    ei = torch.as_tensor(g.edge_index.T)
    ef = torch.as_tensor(g.edge_features)
    logits = model(x, ei, ef)
    assert logits.shape == (5,)
    probs = model.predict_proba(g)
    assert probs.shape == (5,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


# --------------------------------------------------------------------------- #
# End-to-end: the classifier learns to separate true / spurious seeds
# --------------------------------------------------------------------------- #
def test_training_learns_to_classify_seeds():
    from graphmambaformer.data.synthetic import SyntheticConfig, generate_dataset
    from graphmambaformer.training.agnes_train import (
        AgnesTrainConfig,
        samples_from_records,
        seed_metrics,
        train_agnes,
        _run_epoch,
    )
    import torch.nn as nn

    torch.manual_seed(0)
    syn = dataclasses.replace(SyntheticConfig(), n_train=60, n_val=20, n_test=20,
                              include_edge_cases=False)
    ds = generate_dataset(syn)
    cfg = AgnesConfig()
    train = samples_from_records(ds.splits["train"], ds.references, cfg, use_dataset_features=True)
    val = samples_from_records(ds.splits["val"], ds.references, cfg, use_dataset_features=True)
    test = samples_from_records(ds.splits["test"], ds.references, cfg, use_dataset_features=True)
    assert train and val and test

    # num_threads defaults to 1 (see AgnesTrainConfig): AGNES graphs are tiny, so
    # capping intra-op threads keeps this from oversubscribing every core and
    # turning a fast CPU epoch into a multi-minute one on a shared box.
    train_cfg = AgnesTrainConfig(epochs=30, patience=6, batch_size=16, device="cpu",
                                 verbose=False, seed=0)
    model, history = train_agnes(train, val, cfg=cfg, train_cfg=train_cfg)

    # Validation loss improved over the run.
    assert history.best_val_loss < history.val_loss[0]

    _, labels, probs = _run_epoch(model, test, nn.BCEWithLogitsLoss(),
                                  torch.device("cpu"), 16, None)
    metrics = seed_metrics(labels, probs)
    # The seed classification problem is very learnable; require clear signal.
    assert metrics["auc"] > 0.75, metrics
    assert metrics["f1"] > 0.6, metrics
