"""Training for the standalone AGNES seed classifier.

Trains :class:`~graphmambaformer.alignment.agnes.AgnesSeedClassifier` to predict,
per seed, whether it is a true match or a spurious one — the supervised graph-
learning problem AGNES formalizes. Labels come straight from the synthetic
dataset's ground truth (``Seed.is_true``); a read's seeds become one graph
(nodes = seeds, edges = the spatial-consistency DAG from Eq. 2), and a batch is a
disjoint union of those graphs so the whole batch is one forward pass.

Training follows the paper: binary cross-entropy, Adam (lr 1e-3, betas
0.9/0.999), batch size 32, up to 50 epochs, early stopping with patience 5 on the
validation loss, dropout 0.3.

Feature note: by default the node features are exactly the ones the pipeline
computes at inference (:func:`~graphmambaformer.alignment.agnes.
node_features_from_anchors`), so what the model trains on is what it will see in
production. Pass ``use_dataset_features=True`` to instead train on the dataset's
richer 12-D ``Seed.features`` (which also fill local GC / repeat / base-quality),
reproducing the paper's full feature set at the cost of train/inference parity for
those three columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..alignment.agnes import (
    AgnesConfig,
    AgnesSeedClassifier,
    NODE_FEATURE_DIM,
    SeedGraph,
    build_seed_graph,
)
from ..alignment.types import AnchorSet

__all__ = [
    "AgnesTrainConfig",
    "AgnesSample",
    "graph_from_record",
    "samples_from_records",
    "seed_metrics",
    "train_agnes",
    "load_agnes_model",
]


def load_agnes_model(path: str, device=None) -> AgnesSeedClassifier:
    """Rebuild a trained :class:`AgnesSeedClassifier` from a checkpoint.

    Accepts a payload written by ``scripts/train_agnes.py`` (a dict with
    ``"model"`` and ``"agnes_config"``) or a bare ``state_dict``.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        cfg = AgnesConfig(**ckpt["agnes_config"]) if ckpt.get("agnes_config") else AgnesConfig()
        state = ckpt["model"]
    else:
        cfg, state = AgnesConfig(), ckpt
    model = AgnesSeedClassifier(cfg)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


@dataclass
class AgnesTrainConfig:
    """Optimizer / schedule knobs for AGNES training (paper defaults)."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 50
    patience: int = 5
    seed: int = 42
    device: Optional[str] = None
    verbose: bool = True
    #: Cap PyTorch intra-op (ATen/OpenMP) threads during training. AGNES graphs
    #: are tiny (tens of nodes / ~hundreds of edges), so there is no work to
    #: parallelize across; leaving this unset lets torch grab *every* core and,
    #: on a shared/oversubscribed node, the resulting thread contention makes a
    #: sub-second epoch take minutes. ``1`` is the right default here; set to
    #: ``None`` to leave torch's global thread count untouched.
    num_threads: Optional[int] = 1


@dataclass
class AgnesSample:
    """One read's seed graph plus its per-seed ground-truth labels."""

    graph: SeedGraph
    labels: np.ndarray  # (N,) float32 in {0, 1}


# --------------------------------------------------------------------------- #
# Turning ground-truth reads into labelled seed graphs
# --------------------------------------------------------------------------- #
def graph_from_record(
    record,
    reference=None,
    cfg: Optional[AgnesConfig] = None,
    use_dataset_features: bool = False,
) -> Optional[AgnesSample]:
    """Build a labelled :class:`SeedGraph` from a synthetic ``ReadRecord``.

    Returns ``None`` when the read carries no seeds (nothing to learn from). The
    nodes stay in the record's seed order, so ``labels[i]`` is the label of node
    ``i`` in the returned graph.
    """
    seeds = list(getattr(record, "seeds", []) or [])
    if not seeds:
        return None

    ref_seq = getattr(reference, "seq", None)
    ref_len = len(ref_seq) if ref_seq is not None else int(
        max((s.ref_pos + s.length for s in seeds), default=1)
    )
    anchors = AnchorSet.from_lists(
        read_pos=[int(s.read_pos) for s in seeds],
        ref_pos=[int(s.ref_pos) for s in seeds],
        length=[int(s.length) for s in seeds],
        strand=[int(s.strand) for s in seeds],
        read_len=len(record.seq),
        ref_len=ref_len,
    )
    graph = build_seed_graph(anchors, cfg)

    if use_dataset_features and all(len(s.features) == NODE_FEATURE_DIM for s in seeds):
        graph.node_features = np.asarray(
            [s.features for s in seeds], dtype=np.float32
        )

    labels = np.asarray([1.0 if s.is_true else 0.0 for s in seeds], dtype=np.float32)
    return AgnesSample(graph=graph, labels=labels)


def samples_from_records(
    records: Sequence,
    references: Optional[dict] = None,
    cfg: Optional[AgnesConfig] = None,
    use_dataset_features: bool = False,
) -> list[AgnesSample]:
    """Build labelled samples for every read that has seeds."""
    out: list[AgnesSample] = []
    for rec in records:
        ref = None
        if references is not None:
            ref = references.get(getattr(rec, "ref_id", None))
        sample = graph_from_record(
            rec, ref, cfg=cfg, use_dataset_features=use_dataset_features
        )
        if sample is not None:
            out.append(sample)
    return out


# --------------------------------------------------------------------------- #
# Batching (disjoint union of graphs)
# --------------------------------------------------------------------------- #
def _collate(samples: Sequence[AgnesSample], device: torch.device) -> dict:
    """Merge samples into one big graph with offset node indices."""
    node_feats, edge_index, edge_feats, labels = [], [], [], []
    offset = 0
    for s in samples:
        g = s.graph
        node_feats.append(g.node_features)
        labels.append(s.labels)
        if g.n_edges:
            edge_index.append(g.edge_index + offset)
            edge_feats.append(g.edge_features)
        offset += g.n_nodes

    x = torch.as_tensor(np.concatenate(node_feats, axis=0), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.concatenate(labels, axis=0), dtype=torch.float32, device=device)
    if edge_index:
        ei = torch.as_tensor(
            np.concatenate(edge_index, axis=0).T, dtype=torch.long, device=device
        )
        ef = torch.as_tensor(
            np.concatenate(edge_feats, axis=0), dtype=torch.float32, device=device
        )
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        ef = torch.zeros((0, 8), dtype=torch.float32, device=device)
    return {"x": x, "edge_index": ei, "edge_features": ef, "y": y}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so the statistic is exact.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[labels].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def seed_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-seed precision / recall / F1 / accuracy / AUC at ``threshold``."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(probs) >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auc": _auc(labels, np.asarray(probs)),
    }


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
@dataclass
class AgnesHistory:
    """Per-epoch training / validation record."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf


def _run_epoch(
    model: AgnesSeedClassifier,
    samples: list[AgnesSample],
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over ``samples``; trains when ``optimizer`` is given, else evals."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_nodes = 0.0, 0
    all_probs, all_labels = [], []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        data = _collate(batch, device)
        n = data["y"].shape[0]
        if n == 0:
            continue
        with torch.set_grad_enabled(training):
            logits = model(data["x"], data["edge_index"], data["edge_features"])
            loss = criterion(logits, data["y"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * n
        total_nodes += n
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(data["y"].detach().cpu().numpy())

    mean_loss = total_loss / max(total_nodes, 1)
    probs = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.float32)
    return mean_loss, labels, probs


def train_agnes(
    train_samples: Sequence[AgnesSample],
    val_samples: Sequence[AgnesSample],
    model: Optional[AgnesSeedClassifier] = None,
    cfg: Optional[AgnesConfig] = None,
    train_cfg: Optional[AgnesTrainConfig] = None,
) -> tuple[AgnesSeedClassifier, AgnesHistory]:
    """Train the AGNES seed classifier with early stopping; return best model.

    The returned model holds the best (lowest val-loss) weights, and ``history``
    carries per-epoch losses plus validation precision/recall/F1/AUC so a caller
    can confirm the network actually learned.
    """
    cfg = cfg or AgnesConfig()
    train_cfg = train_cfg or AgnesTrainConfig()
    # Tiny graphs don't benefit from intra-op parallelism; capping threads avoids
    # pathological oversubscription (torch otherwise spawns one thread per core).
    if train_cfg.num_threads is not None:
        try:
            torch.set_num_threads(int(train_cfg.num_threads))
        except Exception:  # pragma: no cover - best-effort, never fatal
            pass
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    device = torch.device(
        train_cfg.device
        if train_cfg.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = (model or AgnesSeedClassifier(cfg)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_samples = list(train_samples)
    val_samples = list(val_samples)
    history = AgnesHistory()
    best_state = None
    stale = 0
    rng = np.random.default_rng(train_cfg.seed)

    for epoch in range(train_cfg.epochs):
        rng.shuffle(train_samples)  # shuffle graph order each epoch
        train_loss, _, _ = _run_epoch(
            model, train_samples, criterion, device, train_cfg.batch_size, optimizer
        )
        val_loss, val_labels, val_probs = _run_epoch(
            model, val_samples, criterion, device, train_cfg.batch_size, None
        )
        metrics = seed_metrics(val_labels, val_probs) if val_labels.size else {}

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_metrics.append(metrics)

        improved = val_loss < history.best_val_loss - 1e-5
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if train_cfg.verbose:
            f1 = metrics.get("f1", float("nan"))
            auc = metrics.get("auc", float("nan"))
            print(
                f"epoch {epoch:02d}  train_bce={train_loss:.4f}  val_bce={val_loss:.4f}"
                f"  val_f1={f1:.3f}  val_auc={auc:.3f}"
                + ("  *" if improved else "")
            )
        if stale >= train_cfg.patience:
            if train_cfg.verbose:
                print(
                    f"early stop at epoch {epoch} (no val improvement for "
                    f"{stale} epochs; best epoch {history.best_epoch})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
