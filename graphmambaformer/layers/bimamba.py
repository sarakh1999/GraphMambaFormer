"""Bidirectional Mamba mixers with learned gated fusion (Figure 1B, Layer 1).

Runs a forward and a reverse Mamba scan and fuses them with a learned,
per-channel gate:  ``g . y_fwd + (1 - g) . y_rev``  (as drawn in the figure).
Bidirectionality matters for alignment, where both upstream and downstream
context inform where a base belongs, unlike causal language modeling.

``BiMamba2`` wraps the Mamba-2 / SSD mixer; ``BiMamba1`` wraps the reference
Mamba-1 mixer. Both share the same forward/reverse gated-fusion structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Mamba1Config, Mamba2Config
from .mamba1 import Mamba1Mixer
from .mamba2 import Mamba2Mixer


class BiMamba2(nn.Module):
    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.fwd = Mamba2Mixer(cfg)
        # Optionally tie forward/reverse parameters to halve the parameter count.
        self.rev = self.fwd if cfg.tie_bidirectional else Mamba2Mixer(cfg)
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        y_fwd = self.fwd(x, mask=mask)

        x_rev = torch.flip(x, dims=[1])
        mask_rev = torch.flip(mask, dims=[1]) if mask is not None else None
        y_rev = self.rev(x_rev, mask=mask_rev)
        y_rev = torch.flip(y_rev, dims=[1])

        g = torch.sigmoid(self.gate(torch.cat([y_fwd, y_rev], dim=-1)))
        return g * y_fwd + (1.0 - g) * y_rev


class BiMamba1(nn.Module):
    """Bidirectional wrapper around the reference Mamba-1 mixer."""

    def __init__(self, cfg: Mamba1Config):
        super().__init__()
        self.cfg = cfg
        self.fwd = Mamba1Mixer(cfg)
        # Optionally tie forward/reverse parameters to halve the parameter count.
        self.rev = self.fwd if cfg.tie_bidirectional else Mamba1Mixer(cfg)
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        y_fwd = self.fwd(x, mask=mask)

        x_rev = torch.flip(x, dims=[1])
        mask_rev = torch.flip(mask, dims=[1]) if mask is not None else None
        y_rev = self.rev(x_rev, mask=mask_rev)
        y_rev = torch.flip(y_rev, dims=[1])

        g = torch.sigmoid(self.gate(torch.cat([y_fwd, y_rev], dim=-1)))
        return g * y_fwd + (1.0 - g) * y_rev
