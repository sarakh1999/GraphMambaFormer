"""Mamba-1 selective SSM mixer (reference port).

A readable, pure-PyTorch reimplementation of the original Mamba block from the
reference repo (krafton-ai/mambaformer-icl, ``mamba/mamba_ssm/modules/
mamba_simple.py::Mamba``). The forward pass mirrors the reference exactly:

    in_proj -> [x, z]
    x -> depthwise causal conv1d -> SiLU
    x -> x_proj -> [dt, B, C]        (dt is projected up by dt_proj)
    selective scan (Delta, A, B, C, D)  with dt = softplus(dt + dt_bias)
    y -> y * silu(z)                 (gated)
    out_proj

The selective scan here is a sequential recurrent reference (runs on CPU/MPS,
no Triton/CUDA required), which is what we need for laptop development. When the
official CUDA ``mamba_ssm`` package is installed and a GPU is available, the
mixer transparently delegates to it for training-scale speed.

Shapes: input ``(B, L, d_model)`` -> output ``(B, L, d_model)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..config import Mamba1Config

try:  # Optional CUDA fast path.
    from mamba_ssm.modules.mamba_simple import Mamba as _CudaMamba

    _HAS_CUDA_MAMBA = True
except Exception:  # pragma: no cover - depends on environment
    _CudaMamba = None
    _HAS_CUDA_MAMBA = False


class Mamba1Mixer(nn.Module):
    """Unidirectional Mamba-1 mixer (original selective SSM)."""

    def __init__(self, cfg: Mamba1Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv
        self.d_inner = cfg.d_inner
        self.dt_rank = cfg.resolved_dt_rank

        # in_proj emits [x, z]; z is the SiLU gate applied after the scan.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=cfg.bias)

        # Depthwise causal short convolution over the inner channels.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=cfg.d_conv,
            groups=self.d_inner,
            padding=cfg.d_conv - 1,
            bias=cfg.conv_bias,
        )
        self.act = nn.SiLU()

        # Selective projections: x -> (dt, B, C).
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj weight init to preserve variance (reference: dt_init).
        dt_init_std = self.dt_rank**-0.5 * cfg.dt_scale
        if cfg.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif cfg.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"Unknown dt_init {cfg.dt_init!r}")

        # dt bias initialized so softplus(dt_bias) lands in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization: A = -exp(A_log), A_log shape (d_inner, d_state).
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # per-channel skip

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=cfg.bias)

        # Instantiate the CUDA mixer lazily only if we can actually use it.
        self._cuda_mixer: nn.Module | None = None
        if cfg.use_fast_path and _HAS_CUDA_MAMBA and torch.cuda.is_available():
            self._cuda_mixer = _CudaMamba(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object
    ) -> torch.Tensor:
        if self._cuda_mixer is not None and x.is_cuda:
            return self._cuda_mixer(x)
        return self._forward_reference(x, mask)

    def _forward_reference(
        self, u: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        B, L, _ = u.shape

        xz = self.in_proj(u)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Depthwise causal conv (truncate right padding) + SiLU.
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]
        x = self.act(x.transpose(1, 2))  # (B, L, d_inner)

        if mask is not None:
            # Zero out padded positions so they contribute nothing to the state.
            x = x * mask.unsqueeze(-1).to(x.dtype)

        # Selective (input-dependent) dt, B, C.
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt_proj is a Linear so it already adds dt_bias; softplus once, as in
        # the reference (softplus(dt @ W + dt_bias)).
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretize: dA = exp(dt * A), dB = dt * B. State: (B, d_inner, d_state).
        # dt: (B, L, d_inner) ; A: (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dBx = (
            dt.unsqueeze(-1)  # (B, L, d_inner, 1)
            * B_mat.unsqueeze(2)  # (B, L, 1, d_state)
            * x.unsqueeze(-1)  # (B, L, d_inner, 1)
        )  # (B, L, d_inner, d_state)

        state = torch.zeros(
            B, self.d_inner, self.d_state, device=u.device, dtype=torch.float32
        )
        ys = []
        for t in range(L):
            state = dA[:, t] * state + dBx[:, t]  # (B, d_inner, d_state)
            y_t = torch.einsum("bdn,bn->bd", state, C_mat[:, t])  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + self.D.float() * x  # per-channel skip

        y = y * self.act(z)  # gated
        y = y.to(u.dtype)
        return self.out_proj(y)
