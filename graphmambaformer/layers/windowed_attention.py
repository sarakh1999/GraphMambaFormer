"""Windowed (block-local) multi-head self-attention — O(n·w).

This is the read-tower realization of Figure 1B **Layer 2** (windowed multi-head
self-attention, "context-dependent substitution + indel scoring"). Reads run in
*base space* and can be very long (ONT 100 kb+), so full O(n²) attention is
infeasible — a single (L, L) score matrix for a 65 kb read is billions of
entries. Instead attention is computed inside non-overlapping blocks of
``window`` positions, which is O(n·w) in both time and memory.

To avoid the hard block boundaries that non-overlapping windows introduce,
consecutive layers alternate a half-window ``shift`` (Swin-Transformer style):
a base sitting at a block edge in one layer is mid-block in the next, so
information crosses boundaries across the stack without paying the full-attention
cost. The bidirectional Mamba mixer in the same block already carries global
context, so windowed attention only needs to add *local* pairwise refinement.

Degenerate/​safe cases:
  * ``window is None`` or ``window >= L`` → a single block == exact full
    bidirectional attention, so short reads (Illumina) get dense attention with
    no code-path change.
  * A padded query position can end up in a block whose keys are all padding;
    every query is therefore always allowed to attend to *itself* so a row is
    never fully masked (no ``softmax`` NaN), and padded query outputs are zeroed
    before returning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig


class WindowedSelfAttention(nn.Module):
    """Block-local multi-head self-attention with optional shifted windows.

    Interface mirrors :class:`~graphmambaformer.layers.attention.MultiHeadSelfAttention`
    (``forward(x, mask=None)``) with one extra ``shift`` argument used by the
    tower to alternate window phase across layers. ``mask`` is a key-padding mask
    where ``True`` marks real (non-pad) positions.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.dropout = cfg.dropout
        self.window = cfg.window
        inner = cfg.n_heads * cfg.d_head
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * inner, bias=cfg.bias)
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=cfg.bias)

    def _attend(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        """Dense attention over one axis. ``x`` is ``(Bn, w, D)``.

        ``key_valid`` is ``(Bn, w)`` boolean (``True`` = attend to this key). The
        diagonal is always allowed so a fully-padded row cannot produce a NaN.
        """
        Bn, w, _ = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(Bn, w, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (Bn, H, w, d_head)

        allow = key_valid[:, None, None, :].to(torch.bool).expand(Bn, 1, w, w)
        eye = torch.eye(w, dtype=torch.bool, device=x.device)[None, None]
        allow = allow | eye  # never leave a query row fully masked

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=allow,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (Bn, H, w, d_head)
        out = out.transpose(1, 2).reshape(Bn, w, self.n_heads * self.d_head)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        shift: int = 0,
    ) -> torch.Tensor:
        B, L, D = x.shape
        w = self.window

        # Full-attention fast path: one block covering the whole read.
        if w is None or w >= L:
            key_valid = (
                mask.to(torch.bool)
                if mask is not None
                else x.new_ones(B, L, dtype=torch.bool)
            )
            out = self._attend(x, key_valid)
            if mask is not None:
                out = out * mask[..., None].to(out.dtype)
            return out

        # Windowed path: pad to a whole number of `w`-sized blocks, offset by
        # `shift` so this layer's block grid is phase-shifted from its neighbours.
        shift = int(shift) % w
        pad_right = (w - (L + shift) % w) % w
        Lp = L + shift + pad_right
        nb = Lp // w

        xpad = F.pad(x, (0, 0, shift, pad_right))  # pad the sequence axis only
        valid = mask.to(torch.bool) if mask is not None else x.new_ones(B, L, dtype=torch.bool)
        valid_pad = x.new_zeros(B, Lp, dtype=torch.bool)
        valid_pad[:, shift:shift + L] = valid

        xb = xpad.reshape(B * nb, w, D)
        vb = valid_pad.reshape(B * nb, w)
        ob = self._attend(xb, vb).reshape(B, Lp, D)

        out = ob[:, shift:shift + L, :]
        return out * valid[..., None].to(out.dtype)
