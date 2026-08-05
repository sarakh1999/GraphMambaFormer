"""Training, validation, instrumentation and plotting for the alignment model.

Four pieces, each usable on its own:

``targets``  supervision built from the dataset's ground truth (real labels)
``trainer``  the loop, device-agnostic via ``AccelContext``, early stopping
``metrics``  validation in aligner terms -- locus/chain accuracy, MAPQ calibration
``probes``   per-step model behavior: activations, gradients, routing, head spread
``plots``    every figure, headless-safe and optional
"""

from .metrics import (
    ValidationMetrics,
    anchor_metrics,
    chain_accuracy,
    locus_accuracy,
    mapq_calibration,
)
from .plots import plot_all, plotting_available
from .probes import BehaviorProbe, StepReport
from .targets import Supervision, TargetBuilder
from .trainer import TrainConfig, Trainer, TrainHistory

__all__ = [
    "BehaviorProbe",
    "StepReport",
    "Supervision",
    "TargetBuilder",
    "TrainConfig",
    "TrainHistory",
    "Trainer",
    "ValidationMetrics",
    "anchor_metrics",
    "chain_accuracy",
    "locus_accuracy",
    "mapq_calibration",
    "plot_all",
    "plotting_available",
]
