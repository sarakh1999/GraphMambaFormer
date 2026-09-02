"""Mamba-2 SSM mixer (Figure 1B, Layer 1).

This is a readable, pure-PyTorch reference implementation of the Mamba-2 /
state-space-duality (SSD) mixer using a sequential recurrent scan. It runs on
CPU/MPS (no Triton/CUDA required), which is what we need for development on a
laptop. When the official CUDA ``mamba_ssm`` package is installed and a GPU is
available, the mixer transparently delegates to it for training-scale speed.

Shapes: input ``(B, L, d_model)`` -> output ``(B, L, d_model)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..config import Mamba2Config
from .common import RMSNormGated

try:  # Optional CUDA fast path.
    from mamba_ssm.modules.mamba2 import Mamba2 as _CudaMamba2

    _HAS_CUDA_MAMBA = True
except Exception:  # pragma: no cover - depends on environment
    _CudaMamba2 = None
    _HAS_CUDA_MAMBA = False


class Mamba2Mixer(nn.Module):
    """Unidirectional Mamba-2 mixer (selective SSD)."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d_inner
        self.d_state = cfg.d_state
        self.ngroups = cfg.ngroups
        self.headdim = cfg.headdim
        self.nheads = cfg.d_inner // cfg.headdim
        self.d_conv = cfg.d_conv

        # in_proj emits [z, xBC, dt] where xBC is later split into [x, B, C].
        self.conv_dim = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state
        d_in_proj = 2 * cfg.d_inner + 2 * cfg.ngroups * cfg.d_state + self.nheads
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=cfg.bias)

        # Depthwise causal short convolution over the xBC channels.
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=cfg.d_conv,
            groups=self.conv_dim,
            padding=cfg.d_conv - 1,
            bias=cfg.conv_bias,
        )

        # dt bias initialized so softplus(dt_bias) lands in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.nheads)
            * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

        # Per-head state-decay parameter; A = -exp(A_log) is strictly negative.
        A = torch.empty(self.nheads).uniform_(cfg.A_init_min, cfg.A_init_max)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))  # skip connection per head

        self.norm = RMSNormGated(cfg.d_inner)
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=cfg.bias)

        # Instantiate the CUDA mixer lazily only if we can actually use it.
        self._cuda_mixer: nn.Module | None = None
        if cfg.use_fast_path and _HAS_CUDA_MAMBA and torch.cuda.is_available():
            self._cuda_mixer = _CudaMamba2(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.d_inner // cfg.d_model,
                headdim=cfg.headdim,
                ngroups=cfg.ngroups,
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        if self._cuda_mixer is not None and x.is_cuda:
            return self._cuda_mixer(x)
        return self._forward_reference(x, mask)

    def _forward_reference(self, u: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, L, _ = u.shape

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )

        # Depthwise causal conv (truncate the right padding) + SiLU.
        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC)[..., :L]
        xBC = F.silu(xBC.transpose(1, 2))

        x, B_mat, C_mat = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        if mask is not None:
            # Zero out padded positions so they contribute nothing to the state.
            x = x * mask.unsqueeze(-1).to(x.dtype)

        A = -torch.exp(self.A_log)  # (nheads,)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B_mat = rearrange(B_mat, "b l (g n) -> b l g n", n=self.d_state)
        C_mat = rearrange(C_mat, "b l (g n) -> b l g n", n=self.d_state)
        rep = self.nheads // self.ngroups
        B_mat = repeat(B_mat, "b l g n -> b l (g r) n", r=rep)  # (B,L,H,N)
        C_mat = repeat(C_mat, "b l g n -> b l (g r) n", r=rep)

        dA = torch.exp(dt * A)  # (B, L, H)

        # Sequential SSD scan. State: (B, H, headdim, d_state).
        state = torch.zeros(
            B, self.nheads, self.headdim, self.d_state, device=u.device, dtype=u.dtype
        )
        ys = []
        for t in range(L):
            dBx = dt[:, t, :, None, None] * x[:, t, :, :, None] * B_mat[:, t, :, None, :]
            state = dA[:, t, :, None, None] * state + dBx
            y_t = (state * C_mat[:, t, :, None, :]).sum(-1)  # (B, H, headdim)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, H, headdim)
        y = y + self.D[None, None, :, None] * x

        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        return self.out_proj(y)
