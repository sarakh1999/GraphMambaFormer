"""Neural scoring heads for Stage 4.

Stages 1-3 are exact but scoring-blind: an exact 15-mer match inside a repeat and
one in unique sequence look identical to the chaining DP. These two heads inject
the backbone's learned view of the read and the graph back into that classical
machinery.

:class:`SeedScoringHead`
    Per-anchor true/false score. An anchor is scored from its geometry (the same
    12-dim feature vector the synthetic dataset provides as ground truth), the
    read's hidden state *at that anchor's read position*, and the embedding of the
    graph node it lands on. Because the read representation is bidirectional, the
    head sees the whole read's context when judging a single anchor — which is
    what a hand-written repeat filter cannot do.

:class:`ChainScoringHead`
    Per-chain score for re-ranking. Combines chain-level geometry with an
    attention pool over the chain's member anchors, so a chain built from many
    confidently-scored anchors outranks one with the same DP score built from
    doubtful ones.

Both heads are masked: anchor/chain counts vary per read, so batches are padded
and every reduction respects the mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import SeedScoringConfig


def _gather_positions(
    sequence: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Gather ``(B, L, D)`` states at ``(B, A)`` positions -> ``(B, A, D)``.

    Out-of-range positions are clamped (padded anchors are zeroed by ``mask``
    afterwards, so their gathered value is irrelevant).
    """
    L = sequence.shape[1]
    index = positions.clamp(0, max(L - 1, 0)).unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
    gathered = sequence.gather(1, index)
    if mask is not None:
        gathered = gathered * mask.unsqueeze(-1).to(gathered.dtype)
    return gathered


class EdgeConvLayer(nn.Module):
    """Edge-feature-aware EdgeConv used on the per-read seed-match graph.

    The message ``MLP(h_i || h_j-h_i || e_ij)`` and max aggregation follow
    AGNES Eq. 6. Edges are supplied as ``source -> destination`` pairs.
    """

    def __init__(self, d_in: int, d_out: int, d_edge: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * d_in + d_edge, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        self.skip = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, A, _ = nodes.shape
        E = edge_index.shape[1]
        if E == 0:
            return self.norm(self.skip(nodes)) * node_mask.unsqueeze(-1)

        batch = torch.arange(B, device=nodes.device)[:, None]
        src = edge_index[..., 0].clamp(0, max(A - 1, 0))
        dst = edge_index[..., 1].clamp(0, max(A - 1, 0))
        src_h = nodes[batch, src]
        dst_h = nodes[batch, dst]
        messages = self.message(torch.cat([src_h, dst_h - src_h, edge_features], dim=-1))

        flat_dst = (batch * A + dst).reshape(-1)
        flat_msg = messages.reshape(B * E, -1)
        live = edge_mask.reshape(-1)
        aggregate = nodes.new_full((B * A, messages.shape[-1]), float("-inf"))
        if bool(live.any()):
            index = flat_dst[live, None].expand(-1, messages.shape[-1])
            aggregate.scatter_reduce_(
                0, index, flat_msg[live], reduce="amax", include_self=True
            )
        aggregate = aggregate.reshape(B, A, -1)
        aggregate = torch.where(torch.isfinite(aggregate), aggregate, torch.zeros_like(aggregate))
        out = self.norm(self.skip(nodes) + aggregate)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class AnchorGraphNetwork(nn.Module):
    """Three-layer AGNES-style GNN for seed nodes and chaining transitions."""

    def __init__(self, cfg: SeedScoringConfig, d_input: int):
        super().__init__()
        hidden = tuple(cfg.anchor_gnn_hidden)
        if not hidden:
            raise ValueError("anchor_gnn_hidden must contain at least one dimension")
        self.input = nn.Sequential(nn.Linear(d_input, hidden[0]), nn.GELU())
        dims = (hidden[0],) + hidden
        self.layers = nn.ModuleList(
            EdgeConvLayer(dims[i], dims[i + 1], cfg.anchor_edge_features, cfg.anchor_gnn_dropout)
            for i in range(len(hidden))
        )
        self.node_out = nn.Sequential(
            nn.Linear(hidden[-1], max(hidden[-1] // 2, 1)),
            nn.GELU(),
            nn.Dropout(cfg.anchor_gnn_dropout),
            nn.Linear(max(hidden[-1] // 2, 1), 1),
        )
        self.edge_out = nn.Sequential(
            nn.Linear(2 * hidden[-1] + cfg.anchor_edge_features, hidden[-1]),
            nn.GELU(),
            nn.Dropout(cfg.anchor_gnn_dropout),
            nn.Linear(hidden[-1], 1),
        )

    def forward(
        self,
        node_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.input(node_input) * node_mask.unsqueeze(-1)
        # Classification uses the undirected seed neighborhood described by
        # AGNES, while edge logits below retain the forward chaining direction.
        reverse_index = edge_index.flip(-1)
        message_index = torch.cat([edge_index, reverse_index], dim=1)
        message_features = torch.cat([edge_features, edge_features], dim=1)
        message_mask = torch.cat([edge_mask, edge_mask], dim=1)
        for layer in self.layers:
            nodes = layer(nodes, message_index, message_features, message_mask, node_mask)

        node_logits = self.node_out(nodes).squeeze(-1)
        B, A, _ = nodes.shape
        batch = torch.arange(B, device=nodes.device)[:, None]
        src = edge_index[..., 0].clamp(0, max(A - 1, 0))
        dst = edge_index[..., 1].clamp(0, max(A - 1, 0))
        src_h, dst_h = nodes[batch, src], nodes[batch, dst]
        edge_logits = self.edge_out(
            torch.cat([src_h, dst_h - src_h, edge_features], dim=-1)
        ).squeeze(-1)
        return node_logits, edge_logits


class SeedScoringHead(nn.Module):
    """Score anchors as true or spurious before they reach the chaining DP."""

    def __init__(self, cfg: SeedScoringConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.feature_proj = nn.Sequential(
            nn.Linear(cfg.num_seed_features, cfg.d_hidden), nn.GELU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_hidden + 2 * d, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.d_hidden // 2, 1),
        )
        self.anchor_gnn = (
            AnchorGraphNetwork(cfg, cfg.num_seed_features + 2 * d)
            if cfg.use_anchor_gnn
            else None
        )

    def forward(
        self,
        seed_features: torch.Tensor,
        read_hidden: torch.Tensor,
        anchor_read_pos: torch.Tensor,
        graph_nodes: torch.Tensor | None = None,
        anchor_node: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        gnn_active: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score a padded batch of anchors.

        Args:
            seed_features: ``(B, A, 12)`` geometric features.
            read_hidden: ``(B, L, D)`` read states in **base space**.
            anchor_read_pos: ``(B, A)`` read offset of each anchor.
            graph_nodes: ``(B, N, D)`` node states, or ``None``.
            anchor_node: ``(B, A)`` node id per anchor (``-1`` when unknown).
            anchor_mask: ``(B, A)`` bool, False for padded anchors.

        Returns ``logits`` and ``score`` (sigmoid of the logits), both ``(B, A)``.
        """
        B, A, _ = seed_features.shape
        device = seed_features.device
        if anchor_mask is None:
            anchor_mask = torch.ones((B, A), dtype=torch.bool, device=device)

        read_context = _gather_positions(read_hidden, anchor_read_pos, anchor_mask)

        if graph_nodes is not None and anchor_node is not None:
            known = anchor_node >= 0
            node_context = _gather_positions(
                graph_nodes, anchor_node.clamp(min=0), anchor_mask & known
            )
        else:
            node_context = torch.zeros_like(read_context)

        features = self.feature_proj(seed_features)
        fallback_input = torch.cat([features, read_context, node_context], dim=-1)
        logits = self.mlp(fallback_input).squeeze(-1)

        result: dict[str, torch.Tensor] = {}
        if (
            self.anchor_gnn is not None
            and edge_index is not None
            and edge_features is not None
            and edge_mask is not None
        ):
            raw_input = torch.cat([seed_features, read_context, node_context], dim=-1)
            gnn_logits, transition_logits = self.anchor_gnn(
                raw_input, edge_index, edge_features, edge_mask, anchor_mask
            )
            active = (
                gnn_active.bool()
                if gnn_active is not None
                else edge_mask.any(dim=1)
            )
            logits = torch.where(active[:, None], gnn_logits, logits)
            transition_logits = transition_logits.masked_fill(~edge_mask, 0.0)
            result.update(
                {
                    "transition_logits": transition_logits,
                    "transition_score": torch.sigmoid(transition_logits) * edge_mask,
                    "edge_index": edge_index,
                    "edge_mask": edge_mask,
                    "gnn_active": active,
                }
            )
        logits = logits.masked_fill(~anchor_mask, 0.0)
        result.update({"logits": logits, "score": torch.sigmoid(logits) * anchor_mask})
        return result


class ChainScoringHead(nn.Module):
    """Re-rank candidate chains using anchor-level and chain-level evidence."""

    def __init__(self, cfg: SeedScoringConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.feature_proj = nn.Sequential(
            nn.Linear(cfg.num_chain_features, cfg.d_hidden), nn.GELU()
        )
        # Attention pool over a chain's member anchors.
        self.member_score = nn.Linear(d, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_hidden + d, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, 1),
        )

    def forward(
        self,
        chain_features: torch.Tensor,
        member_states: torch.Tensor,
        member_mask: torch.Tensor,
        chain_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score a padded batch of chains.

        Args:
            chain_features: ``(B, C, F)`` chain-level geometry.
            member_states: ``(B, C, A, D)`` per-chain anchor representations.
            member_mask: ``(B, C, A)`` bool, False for padded members.
            chain_mask: ``(B, C)`` bool, False for padded chains.

        Returns ``logits`` and ``score``, both ``(B, C)``.
        """
        B, C = chain_features.shape[:2]
        if chain_mask is None:
            chain_mask = torch.ones((B, C), dtype=torch.bool, device=chain_features.device)

        logits = self.member_score(member_states).squeeze(-1)  # (B, C, A)
        logits = logits.masked_fill(~member_mask, float("-inf"))
        # A chain with no members would give a NaN softmax; let it attend flatly
        # and rely on the chain mask to discard the result.
        empty = ~member_mask.any(dim=-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bca,bcad->bcd", weights, member_states)

        features = self.feature_proj(chain_features)
        out = self.mlp(torch.cat([features, pooled], dim=-1)).squeeze(-1)
        out = out.masked_fill(~chain_mask, float("-inf"))
        return {
            "logits": out,
            "score": torch.sigmoid(torch.nan_to_num(out, neginf=-30.0)) * chain_mask,
        }
