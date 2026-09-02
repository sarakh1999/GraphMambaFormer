"""Core architectures and the registry that selects between them.

Four modes, all constructed through :func:`build_core_model`:

======================  ====================================================
``arch``                Model
======================  ====================================================
``"graphmamba"``        :class:`GraphMambaModel` — the default core model
``"multitask_graphmamba"``  :class:`MultiTaskGraphMamba` — plus the task heads
``"mambaformer"``       ``GraphMambaFormerEncoder`` with the MambaFormer backbone
``"hybrid"``            ``GraphMambaFormerEncoder`` with the hybrid block stack
======================  ====================================================

``"graphmamba"`` is the default and the mode the pipeline is exercised with; the
two ``GraphMambaFormerEncoder`` backbones are kept selectable so the earlier
Figure-1 assembly stays available as an ablation baseline. The encoder backbones
are sequence-only — they have no mapping head — so :func:`build_core_model`
reports that via :attr:`CoreModelSpec.supports_alignment_heads`, and the pipeline
degrades to purely classical scoring when they are selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from ..config import CORE_ARCHITECTURES, CoreModelConfig, GraphMambaConfig, ModelConfig
from ..model import GraphMambaFormerEncoder
from .graph_mamba import (
    BiMambaTower,
    GATv2Tower,
    GraphBatch,
    GraphMambaModel,
    GraphMambaOutput,
    MultiTaskGraphMamba,
)


@dataclass
class CoreModelSpec:
    """A constructed core model plus what the pipeline can expect from it."""

    arch: str
    model: nn.Module
    d_model: int
    #: True when the model exposes the mapping head and the Stage 4 scoring heads.
    supports_alignment_heads: bool
    #: True when the model consumes base codes (``GraphMambaModel``) rather than
    #: k-mer token ids (``GraphMambaFormerEncoder``).
    base_space_input: bool

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        return (
            f"arch={self.arch} | d_model={self.d_model} | "
            f"params={self.num_parameters:,} | "
            f"alignment_heads={self.supports_alignment_heads}"
        )


def build_core_model(cfg: CoreModelConfig | str | None = None) -> CoreModelSpec:
    """Build the core model named by ``cfg``.

    Accepts a :class:`CoreModelConfig`, a bare architecture name, or ``None`` for
    the default (``"graphmamba"``).
    """
    if cfg is None:
        cfg = CoreModelConfig()
    elif isinstance(cfg, str):
        cfg = CoreModelConfig(arch=cfg)

    if cfg.arch == "graphmamba":
        model = GraphMambaModel(cfg.graphmamba)
        return CoreModelSpec(
            arch=cfg.arch,
            model=model,
            d_model=cfg.graphmamba.d_model,
            supports_alignment_heads=True,
            base_space_input=True,
        )

    if cfg.arch == "multitask_graphmamba":
        model = MultiTaskGraphMamba(cfg.graphmamba)
        return CoreModelSpec(
            arch=cfg.arch,
            model=model,
            d_model=cfg.graphmamba.d_model,
            supports_alignment_heads=True,
            base_space_input=True,
        )

    # "mambaformer" / "hybrid": the sequence-only Figure-1 encoder baselines.
    model = GraphMambaFormerEncoder(cfg.encoder)
    return CoreModelSpec(
        arch=cfg.arch,
        model=model,
        d_model=cfg.encoder.d_model,
        supports_alignment_heads=False,
        base_space_input=False,
    )


__all__ = [
    "CORE_ARCHITECTURES",
    "CoreModelConfig",
    "CoreModelSpec",
    "build_core_model",
    "GraphMambaModel",
    "MultiTaskGraphMamba",
    "GraphMambaOutput",
    "GraphBatch",
    "GraphMambaConfig",
    "ModelConfig",
    "BiMambaTower",
    "GATv2Tower",
]
