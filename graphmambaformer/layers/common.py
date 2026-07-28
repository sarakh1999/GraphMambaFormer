"""Shared building blocks: norms, positional encoding, FFN, residual wrapper.

These are deliberately generic so that future sub-layers (windowed attention,
GATv2) can reuse the same normalization / residual scaffolding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RMSNormGated(nn.Module):
    """RMSNorm with an optional SiLU gate, as used inside Mamba-2.

    Computes ``rmsnorm(x * silu(z))`` when a gate ``z`` is supplied.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is not None:
            x = x * F.silu(z)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    """Additive sinusoidal positional encoding, computed on the fly.

    Computed per-call rather than cached in a big buffer so it scales to the
    very long ONT reads (100 kb+) without allocating a huge position table.
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal PE")
        self.d_model = d_model
        inv_freq = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        _, L, D = x.shape
        pos = torch.arange(offset, offset + L, device=x.device, dtype=torch.float32)
        ang = pos.unsqueeze(1) * self.inv_freq.to(x.device).unsqueeze(0)  # (L, D/2)
        pe = torch.zeros(L, D, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang)
        return x + pe.unsqueeze(0)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (Figure 1B, FFN row)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Residual(nn.Module):
    """Pre-norm residual wrapper around an arbitrary sub-layer.

    The sub-layer must accept ``(x, **kwargs)`` and return a tensor of the same
    shape. Extra kwargs (e.g. ``mask``, ``graph``) are forwarded, so the same
    wrapper works for Mamba, attention, and graph layers alike.
    """

    def __init__(self, d_model: int, sublayer: nn.Module, norm: nn.Module | None = None):
        super().__init__()
        self.norm = norm if norm is not None else RMSNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return x + self.sublayer(self.norm(x), **kwargs)
