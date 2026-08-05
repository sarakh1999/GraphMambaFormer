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

    def forward(
        self,
        seed_features: torch.Tensor,
        read_hidden: torch.Tensor,
        anchor_read_pos: torch.Tensor,
        graph_nodes: torch.Tensor | None = None,
        anchor_node: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
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
        logits = self.mlp(
            torch.cat([features, read_context, node_context], dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~anchor_mask, 0.0)
        return {"logits": logits, "score": torch.sigmoid(logits) * anchor_mask}


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
