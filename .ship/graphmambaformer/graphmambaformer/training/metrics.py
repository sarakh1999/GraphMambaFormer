"""Validation metrics that say whether the model is actually aligning better.

Validation loss alone is a poor stopping signal here: the objective is a
weighted sum of seven terms, so it can fall while locus accuracy stagnates.
These metrics measure the things the aligner is judged on directly.

MAPQ deserves a note. A mapper's MAPQ is a claim about its own error rate --
"MAPQ 30 means I am wrong about 1 in 1000" -- so the useful measurement is
calibration, not accuracy. :func:`mapq_calibration` bins reads by predicted
MAPQ and compares the observed error rate in each bin against the rate the
MAPQ claims, which is what makes over-confidence visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

__all__ = ["ValidationMetrics", "anchor_metrics", "chain_accuracy",
           "locus_accuracy", "mapq_calibration"]

#: A predicted start within this many bases of truth counts as correct.
LOCUS_TOLERANCE = 50


@dataclass
class ValidationMetrics:
    """One validation pass, in the terms the aligner is judged on."""

    loss: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)
    locus_accuracy: float = 0.0
    chain_accuracy: float = 0.0
    anchor_auc: float = 0.0
    anchor_precision: float = 0.0
    anchor_recall: float = 0.0
    mapq_mae: float = 0.0
    mapq_expected_error: float = 0.0
    mapq_observed_error: float = 0.0
    mapped_fraction: float = 0.0
    n_reads: int = 0
    #: Reads that had >=2 candidate chains, so chain_accuracy is meaningful.
    n_chain_scored: int = 0

    def one_line(self) -> str:
        chain = (
            f"chain={self.chain_accuracy:.1%}(n={self.n_chain_scored})"
            if self.n_chain_scored else "chain=n/a(<2 candidates)"
        )
        return (
            f"val loss={self.loss:.4f} locus={self.locus_accuracy:.1%} "
            f"{chain} anchorAUC={self.anchor_auc:.3f} "
            f"mapqMAE={self.mapq_mae:.1f} mapped={self.mapped_fraction:.1%}"
        )

    def monitored(self, name: str) -> Optional[float]:
        """The value of the early-stopping metric, or ``None`` if unmeasurable.

        Distinguishing "unmeasurable" from "zero" matters: chain accuracy is
        undefined when every read has a single candidate, and treating that as
        0.0 would make early stopping fire on a metric that can never improve.
        """
        if name == "chain_accuracy" and not self.n_chain_scored:
            return None
        return self.as_dict().get(name)

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "locus_accuracy": self.locus_accuracy,
            "chain_accuracy": self.chain_accuracy,
            "anchor_auc": self.anchor_auc,
            "anchor_precision": self.anchor_precision,
            "anchor_recall": self.anchor_recall,
            "mapq_mae": self.mapq_mae,
            "mapped_fraction": self.mapped_fraction,
            **{f"term/{k}": v for k, v in self.terms.items()},
        }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank identity; 0.5 when one class is absent.

    Equivalent to the Mann-Whitney U statistic, and cheap enough to run every
    epoch without sklearn.
    """
    if scores.size == 0:
        return 0.5
    pos, neg = labels > 0.5, labels <= 0.5
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average ranks within ties so tied scores cannot inflate the statistic.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def anchor_metrics(logits: torch.Tensor, labels: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   threshold: float = 0.5) -> dict[str, float]:
    """AUC / precision / recall for the seed head over valid anchors only."""
    with torch.no_grad():
        flat_logits = logits.detach().float().flatten()
        flat_labels = labels.detach().float().flatten()
        if mask is not None:
            keep = mask.detach().bool().flatten()
            flat_logits, flat_labels = flat_logits[keep], flat_labels[keep]
        probs = torch.sigmoid(flat_logits).cpu().numpy()
        truth = flat_labels.cpu().numpy()

    auc = _auc(probs, truth)
    predicted = probs >= threshold
    positive = truth > 0.5
    tp = float((predicted & positive).sum())
    precision = tp / max(float(predicted.sum()), 1.0)
    recall = tp / max(float(positive.sum()), 1.0)
    return {"auc": auc, "precision": precision, "recall": recall}


def chain_accuracy(logits: torch.Tensor, target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> tuple[float, int]:
    """How often the top-scored candidate chain is the correct one.

    Returns ``(accuracy, n_scored)``. Only reads with at least two candidates
    count: picking the right chain out of one is not a measurement, and
    including such reads would report a flattering 100% that means nothing.
    ``n_scored`` is what tells you whether the accuracy is worth reading.
    """
    with torch.no_grad():
        scores = logits.detach().float()
        if scores.dim() == 3:
            scores = scores.squeeze(-1)
        valid = (
            mask.detach().bool() if mask is not None
            else torch.ones_like(scores, dtype=torch.bool)
        )
        scores = scores.masked_fill(~valid, float("-inf"))
        decidable = (target >= 0) & (valid.sum(-1) >= 2)
        n_scored = int(decidable.sum())
        if n_scored == 0:
            return 0.0, 0
        picked = scores.argmax(-1)
        correct = (picked[decidable] == target[decidable]).float().mean()
    return float(correct), n_scored


def locus_accuracy(predicted_starts: Sequence[float], true_starts: Sequence[float],
                   tolerance: int = LOCUS_TOLERANCE) -> float:
    """Fraction of reads placed within ``tolerance`` bases of the truth."""
    if not len(predicted_starts):
        return 0.0
    pred = np.asarray(predicted_starts, dtype=np.float64)
    true = np.asarray(true_starts, dtype=np.float64)
    n = min(pred.size, true.size)
    return float((np.abs(pred[:n] - true[:n]) <= tolerance).mean())


def mapq_calibration(mapq: Sequence[float], correct: Sequence[bool],
                     n_bins: int = 6) -> dict[str, float]:
    """Compare the error rate MAPQ promises against the rate observed.

    A MAPQ of ``q`` claims an error probability of ``10 ** (-q / 10)``. Reporting
    both numbers side by side is what exposes an over-confident mapper: equal
    values mean calibrated, observed > expected means over-confident.
    """
    if not len(mapq):
        return {"expected_error": 0.0, "observed_error": 0.0, "mae": 0.0, "bins": 0}
    q = np.asarray(mapq, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    n = min(q.size, ok.size)
    q, ok = q[:n], ok[:n]

    expected = float(np.mean(10.0 ** (-q / 10.0)))
    observed = float(1.0 - ok.mean())

    # Per-bin gap, averaged: a single global number can hide compensating bins.
    edges = np.linspace(q.min(), q.max() + 1e-9, n_bins + 1)
    gaps, used = [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (q >= lo) & (q < hi)
        if not sel.any():
            continue
        used += 1
        gaps.append(abs(float(np.mean(10.0 ** (-q[sel] / 10.0)))
                        - float(1.0 - ok[sel].mean())))
    return {
        "expected_error": expected,
        "observed_error": observed,
        "mae": float(np.mean(gaps)) if gaps else 0.0,
        "bins": used,
    }


def mapq_mae(predicted: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return float((predicted.detach().float().flatten()
                      - target.detach().float().flatten()).abs().mean())
