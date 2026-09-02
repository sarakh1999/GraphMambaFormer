"""Triton tier: fused LayerNorm + Linear + GELU.

The cross-attention fusion FFN is the one place in the model where a LayerNorm,
a matmul and an activation sit back to back on a large tensor, so fusing them
removes two full round trips through HBM.

LayerNorm needs whole-row statistics over the reduction dimension, which the
matmul tiles across. Rather than give that up, the pass is split in two: a
cheap reduction kernel produces per-row ``mean``/``rstd``, then the matmul
kernel normalizes each ``A`` tile as it loads it and applies GELU in the
epilogue. Accumulation is FP32 regardless of the input dtype, so the result is
numerically equivalent to the unfused torch path under AMP.

Import is safe everywhere: when ``triton`` is missing (CPU / Apple Silicon) the
kernels are simply not defined and :class:`FusedLNLinearGELU` runs the torch
reference instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Triton ships with CUDA torch builds only.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised on non-CUDA hosts
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def triton_available() -> bool:
    """True when the Triton kernels can actually run (needs Triton + CUDA)."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _ln_stats_kernel(
        x_ptr,
        mean_ptr,
        rstd_ptr,
        stride_x_row,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        """Per-row mean and reciprocal std of ``x`` (one program per row)."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_x_row

        total = tl.zeros((), dtype=tl.float32)
        total_sq = tl.zeros((), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            total += tl.sum(vals, axis=0)
            total_sq += tl.sum(vals * vals, axis=0)

        mean = total / n_cols
        var = total_sq / n_cols - mean * mean
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, 1.0 / tl.sqrt(var + eps))

    @triton.jit
    def _fused_ln_linear_gelu_kernel(
        a_ptr,
        w_ptr,
        b_ptr,
        c_ptr,
        mean_ptr,
        rstd_ptr,
        gamma_ptr,
        beta_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``C = gelu(layernorm(A) @ W^T + b)`` with FP32 accumulation.

        ``W`` is stored ``(N, K)`` to match ``nn.Linear.weight``.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Row statistics are constant across the K loop.
        mean = tl.load(mean_ptr + offs_m, mask=mask_m, other=0.0)
        rstd = tl.load(rstd_ptr + offs_m, mask=mask_m, other=1.0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            gamma = tl.load(gamma_ptr + offs_k, mask=mask_k, other=1.0)
            beta = tl.load(beta_ptr + offs_k, mask=mask_k, other=0.0)
            a = (a - mean[:, None]) * rstd[:, None] * gamma[None, :] + beta[None, :]
            # Zero the tail so it cannot pollute the accumulator.
            a = tl.where(mask_k[None, :], a, 0.0)

            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += tl.dot(a, tl.trans(w), allow_tf32=True)

        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)[None, :]

        # tanh-approximated GELU, matching nn.functional.gelu(approximate="tanh").
        inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        out = 0.5 * acc * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            out,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def fused_ln_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``gelu(linear(layer_norm(x), weight, bias))``, fused when Triton is live.

    ``x`` may have any leading shape; only the last dimension is normalized and
    contracted. Falls back to composed torch ops off CUDA.
    """
    *lead, K = x.shape
    if not triton_available() or not x.is_cuda or x.numel() == 0:
        normed = F.layer_norm(x, (K,), gamma, beta, eps)
        return F.gelu(F.linear(normed, weight, bias), approximate="tanh")

    x2d = x.reshape(-1, K).contiguous()
    M, N = x2d.shape[0], weight.shape[0]

    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    _ln_stats_kernel[(M,)](
        x2d, mean, rstd, x2d.stride(0), K, eps, BLOCK=min(1024, triton.next_power_of_2(K))
    )

    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_ln_linear_gelu_kernel[grid](
        x2d,
        weight,
        bias if bias is not None else x2d,  # unused when HAS_BIAS is False
        out,
        mean,
        rstd,
        gamma,
        beta,
        M,
        N,
        K,
        x2d.stride(0),
        x2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.reshape(*lead, N).to(x.dtype)


class FusedLNLinearGELU(nn.Module):
    """LayerNorm -> Linear -> GELU as one fused op (Triton) or three (torch).

    Drop-in for the first half of a transformer FFN. The parameters are laid out
    exactly like ``nn.LayerNorm`` + ``nn.Linear``, so checkpoints transfer
    between the fused and unfused paths.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_in))
        self.beta = nn.Parameter(torch.zeros(d_in))
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_ln_linear_gelu(
            x, self.weight, self.bias, self.gamma, self.beta, self.eps
        )
