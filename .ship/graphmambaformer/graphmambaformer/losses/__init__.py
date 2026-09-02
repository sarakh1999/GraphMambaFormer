"""Training objectives for the alignment stages and the multi-task heads."""

from .alignment_loss import (
    AlignmentLoss,
    GraphMambaLoss,
    KendallWeighting,
    LossOutput,
    MultiTaskLoss,
)

__all__ = [
    "AlignmentLoss",
    "GraphMambaLoss",
    "KendallWeighting",
    "LossOutput",
    "MultiTaskLoss",
]
