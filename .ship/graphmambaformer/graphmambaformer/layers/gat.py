"""GATv2 graph-attention layer (Figure 1B, Layer 3).

An edge-type-aware GATv2 (Brody et al. 2021, "How Attentive are Graph Attention
Networks?", arXiv:2105.14491) that performs message passing over the pangenome
reference graph. Unlike the original GAT, GATv2 applies the non-linearity
*before* the attention vector, giving strictly more expressive, input-dependent
attention:

    e(i, j) = a^T · LeakyReLU( W_src h_j + W_dst h_i + W_edge edge_ij )
    alpha_ij = softmax_j( e(i, j) )                      (over neighbours j -> i)
    h_i' = concat_heads  sum_j alpha_ij · (W_src h_j)

Attention is conditioned on the **edge-type embedding** produced by the
Reference Graph Encoder (the 8 discrete pangenome edge types: ref link, SNP,
insertion, deletion, SV, splice, CpG, barcode). A learned **self-loop** term is
always added so every node — including isolated ones — attends to itself.

Complexity is O(|E|) in the number of edges. The implementation is pure
PyTorch (scatter via ``index_add`` + a segment-softmax), so it runs on CPU /
Apple Silicon without ``torch_geometric``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GATConfig


def _segment_softmax(
    scores: torch.Tensor, index: torch.Tensor, num_nodes: int, eps: float = 1e-16
) -> torch.Tensor:
    """Softmax of per-edge ``scores`` (E, H) grouped by destination ``index`` (E,)."""
    H = scores.shape[1]
    idx = index.unsqueeze(-1).expand(-1, H)  # (E, H)

    # max per destination node (numerical stability); nodes with no edges stay 0.
    max_per = scores.new_full((num_nodes, H), float("-inf"))
    max_per = max_per.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    max_per = torch.nan_to_num(max_per, neginf=0.0)

    shifted = scores - max_per.gather(0, idx)
    exp = shifted.exp()

    denom = scores.new_zeros((num_nodes, H)).index_add(0, index, exp)
    return exp / (denom.gather(0, idx) + eps)


class GATv2Layer(nn.Module):
    """Edge-type-aware, multi-head GATv2 over graph nodes -> refined node features."""

    def __init__(self, cfg: GATConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_gat
        inner = cfg.n_heads * cfg.d_gat

        # separate source / destination transforms (GATv2)
        self.W_src = nn.Linear(cfg.d_model, inner, bias=False)
        self.W_dst = nn.Linear(cfg.d_model, inner, bias=False)
        # project the graph encoder's edge-type embedding into attention space
        self.edge_proj = nn.Linear(cfg.d_edge, inner, bias=False)
        # learned self-loop edge feature (own "edge type")
        self.self_loop = nn.Parameter(torch.zeros(inner))
        # per-head attention vector a
        self.att = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_gat))
        self.out_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.leaky = nn.LeakyReLU(cfg.negative_slope)
        self.dropout = nn.Dropout(cfg.dropout)

        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        node_feat: torch.Tensor,          # (N, d_model)
        edge_index: torch.Tensor,         # (2, E) src -> dst
        edge_attr: torch.Tensor | None = None,  # (E, d_edge) edge-type embeddings
    ) -> torch.Tensor:
        N = node_feat.shape[0]
        H, hd = self.n_heads, self.head_dim
        device = node_feat.device

        hs = self.W_src(node_feat).view(N, H, hd)   # source-side features
        hd_ = self.W_dst(node_feat).view(N, H, hd)  # destination-side features

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
        else:
            src = dst = torch.empty(0, dtype=torch.long, device=device)

        # --- real edges ---------------------------------------------------- #
        if src.numel() > 0:
            e = hs[src] + hd_[dst]                       # (E, H, hd)
            if edge_attr is not None:
                e = e + self.edge_proj(edge_attr).view(-1, H, hd)
            score_e = (self.leaky(e) * self.att).sum(-1)  # (E, H)
            msg_src_e = hs[src]                           # message = transformed source
        else:
            score_e = node_feat.new_zeros((0, H))
            msg_src_e = node_feat.new_zeros((0, H, hd))

        # --- self loops (every node attends to itself) --------------------- #
        self_nodes = torch.arange(N, device=device)
        e_self = hs + hd_ + self.self_loop.view(1, H, hd)
        score_self = (self.leaky(e_self) * self.att).sum(-1)  # (N, H)

        # --- combine edges + self loops, softmax over each destination ----- #
        all_scores = torch.cat([score_e, score_self], dim=0)         # (E+N, H)
        all_dst = torch.cat([dst, self_nodes], dim=0)                # (E+N,)
        all_msg_src = torch.cat([msg_src_e, hs], dim=0)              # (E+N, H, hd)

        alpha = self._softmax_dropout(all_scores, all_dst, N)        # (E+N, H)
        weighted = all_msg_src * alpha.unsqueeze(-1)                 # (E+N, H, hd)

        out = node_feat.new_zeros((N, H, hd)).index_add(0, all_dst, weighted)
        return self.out_proj(out.reshape(N, H * hd))                 # (N, d_model)

    def _softmax_dropout(self, scores, dst, N) -> torch.Tensor:
        alpha = _segment_softmax(scores, dst, N)
        return self.dropout(alpha)
