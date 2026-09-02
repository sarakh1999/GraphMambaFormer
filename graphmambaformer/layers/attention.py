"""Multi-head self-attention (Figure 1B Layer 2 / MambaFormer attention block).

This is the attention mixer used inside the MambaFormer backbone. The reference
implementation (krafton-ai/mambaformer-icl) uses a *causal* self-attention
because it targets autoregressive in-context learning. Alignment instead needs
to look both upstream and downstream of every base, so this layer defaults to
**bidirectional** attention (``causal=False``) and is padding-mask aware.

An optional sliding ``window`` is supported so the windowed-attention variant
from Figure 1B can be enabled later without changing the call sites.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.causal = cfg.causal
        self.window = cfg.window
        self.dropout = cfg.dropout
        inner = cfg.n_heads * cfg.d_head

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * inner, bias=cfg.bias)
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=cfg.bias)

    def _build_attn_mask(
        self, mask: torch.Tensor | None, L: int, device: torch.device
    ) -> torch.Tensor | None:
        """Boolean attention mask, shape broadcastable to ``(B, H, L, L)``.

        Following ``scaled_dot_product_attention`` semantics, ``True`` marks
        positions that are *allowed* to attend.
        """
        allow = None
        if mask is not None:
            # Key-padding mask: block attention *to* padded keys. (B, 1, 1, L)
            allow = mask[:, None, None, :].to(torch.bool)

        if self.window is not None:
            idx = torch.arange(L, device=device)
            # |i - j| <= window  (symmetric band). (1, 1, L, L)
            band = (idx[None, :] - idx[:, None]).abs() <= self.window
            band = band[None, None]
            allow = band if allow is None else (allow & band)

        # Fold causality into the explicit mask when both are needed (SDPA does
        # not allow is_causal together with an attn_mask).
        if self.causal and allow is not None:
            idx = torch.arange(L, device=device)
            causal = idx[None, :] <= idx[:, None]  # query i attends to key j<=i
            allow = allow & causal[None, None]

        return allow

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: object
    ) -> torch.Tensor:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)  # (B, L, 3*inner)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d_head)

        attn_mask = self._build_attn_mask(mask, L, x.device)
        # If causality is still needed and no explicit mask was built, let SDPA
        # apply its optimized causal path.
        is_causal = self.causal and attn_mask is None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, H, L, d_head)

        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.d_head)
        return self.out_proj(out)
