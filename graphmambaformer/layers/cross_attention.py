"""CrossAttentionFusion — the read <-> graph bidirectional attention bridge.

The BiMamba2 stack knows the read and the GATv2 stack knows the graph, but
alignment is a *correspondence* between them. Two cross-attentions supply it:

- **Read -> Graph** (``Q=read, K=V=graph``): each read position asks which graph
  nodes it could have come from. This is the learned analogue of seeding.
- **Graph -> Read** (``Q=graph, K=V=read``): each node asks which read positions
  support it, which is what lets a node accumulate read evidence for the
  variant-calling and copy-number heads.

Both directions are concatenated with their inputs, passed through an
FFN (``D -> 4D -> D``) whose first half is the fused LayerNorm+Linear+GELU
kernel, and pooled to a single ``(B, D)`` embedding for the routing and mapping
heads. Attention runs through ``scaled_dot_product_attention``, so it picks up
the Flash / mem-efficient kernels on CUDA automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..accel.triton_ops import FusedLNLinearGELU
from ..config import CrossAttentionConfig


class MultiHeadCrossAttention(nn.Module):
    """One direction of cross-attention with separate query and key/value inputs."""

    def __init__(self, cfg: CrossAttentionConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.dropout = cfg.dropout
        inner = cfg.n_heads * cfg.d_head

        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * inner, bias=False)
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``query`` ``(B, Lq, D)`` attends over ``context`` ``(B, Lk, D)``."""
        B, Lq, _ = query.shape
        Lk = context.shape[1]

        q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(context).view(B, Lk, 2, self.n_heads, self.d_head)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            # True marks keys that may be attended, per SDPA's convention.
            attn_mask = context_mask[:, None, None, :].to(torch.bool)
            # A query with no visible keys would produce NaN from a fully-masked
            # softmax, so let such rows attend everywhere and zero them after.
            empty = ~attn_mask.any(dim=-1, keepdim=True)
            attn_mask = attn_mask | empty

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, Lq, self.n_heads * self.d_head)
        return self.out_proj(out)


class AttentionPool(nn.Module):
    """Mask-aware attention pooling to a single vector per batch element.

    A learned query scores every position, so the pooled embedding can focus on
    the informative part of a read instead of being diluted by a long,
    uninformative tail the way mean pooling would.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)  # (B, L)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(torch.bool), float("-inf"))
            # Guard against an all-padding row, which would give a NaN softmax.
            empty = ~mask.any(dim=-1)
            logits = torch.where(empty.unsqueeze(-1), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bl,bld->bd", weights, x)


class CrossAttentionFusion(nn.Module):
    """Fuse read and graph representations into per-position and pooled embeddings."""

    def __init__(self, cfg: CrossAttentionConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.read_to_graph = MultiHeadCrossAttention(cfg)
        self.graph_to_read = MultiHeadCrossAttention(cfg)
        self.read_norm = nn.LayerNorm(d)
        self.graph_norm = nn.LayerNorm(d)

        # FFN over the concatenated [own, cross-attended] representation. The
        # first half is the fused LN+Linear+GELU kernel; the projection back down
        # stays a plain Linear.
        self.ffn_in = FusedLNLinearGELU(2 * d, cfg.d_ff_mult * d)
        self.ffn_out = nn.Linear(cfg.d_ff_mult * d, d)
        self.dropout = nn.Dropout(cfg.dropout)

        self.pool = AttentionPool(d) if cfg.pooling == "attention" else None

    def forward(
        self,
        read: torch.Tensor,
        graph: torch.Tensor,
        read_mask: torch.Tensor | None = None,
        graph_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse ``(B, L, D)`` read states with ``(B, N, D)`` graph node states.

        Returns a dict with:
          - ``read``: ``(B, L, D)`` read states refined by graph context.
          - ``graph``: ``(B, N, D)`` node states refined by read evidence.
          - ``fused``: ``(B, L + N, D)`` the two concatenated.
          - ``pooled``: ``(B, D)`` single embedding for the routing / mapping heads.
        """
        read_ctx = self.read_to_graph(self.read_norm(read), self.graph_norm(graph), graph_mask)
        graph_ctx = self.graph_to_read(self.graph_norm(graph), self.read_norm(read), read_mask)

        read_out = read + self.dropout(
            self.ffn_out(self.ffn_in(torch.cat([read, read_ctx], dim=-1)))
        )
        graph_out = graph + self.dropout(
            self.ffn_out(self.ffn_in(torch.cat([graph, graph_ctx], dim=-1)))
        )

        fused = torch.cat([read_out, graph_out], dim=1)
        fused_mask = None
        if read_mask is not None or graph_mask is not None:
            rm = (
                read_mask
                if read_mask is not None
                else torch.ones(read.shape[:2], dtype=torch.bool, device=read.device)
            )
            gm = (
                graph_mask
                if graph_mask is not None
                else torch.ones(graph.shape[:2], dtype=torch.bool, device=graph.device)
            )
            fused_mask = torch.cat([rm, gm], dim=1)

        if self.pool is not None:
            pooled = self.pool(fused, fused_mask)
        elif self.cfg.pooling == "max":
            masked = (
                fused.masked_fill(~fused_mask.unsqueeze(-1), float("-inf"))
                if fused_mask is not None
                else fused
            )
            pooled = masked.max(dim=1).values
        else:  # "mean"
            if fused_mask is None:
                pooled = fused.mean(dim=1)
            else:
                weights = fused_mask.unsqueeze(-1).to(fused.dtype)
                pooled = (fused * weights).sum(1) / weights.sum(1).clamp(min=1.0)

        return {
            "read": read_out,
            "graph": graph_out,
            "fused": fused,
            "pooled": torch.nan_to_num(pooled),
        }
