"""Reference Graph Encoder (Figure 1A, right).

Turns a GFA/rGFA pangenome graph into per-node embeddings that combine:
  - k-mer node features (each node's DNA sequence -> pooled k-mer embedding),
  - Laplacian positional encoding (graph-structural position of the node),
  - edge-type encoding (8 discrete edge types) exposed for the downstream GATv2.

This module only builds node/edge features; message passing over the graph is
the job of the future GATv2 layer, so the encoder stays a pure feature builder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import GraphEncoderConfig
from ..tokenization import KmerTokenizer


@dataclass
class GraphEncoding:
    """Container for the encoded pangenome graph."""

    node_embeddings: torch.Tensor  # (N, d_model)
    edge_index: torch.Tensor  # (2, E)
    edge_type_embeddings: torch.Tensor  # (E, d_edge)
    lap_pe: torch.Tensor  # (N, lap_pe_dim)


def laplacian_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    pe_dim: int,
    sign_flip: bool = False,
) -> torch.Tensor:
    """Compute Laplacian eigenvector positional encodings.

    Uses the symmetric normalized Laplacian ``L = I - D^-1/2 A D^-1/2`` and
    takes the eigenvectors for the smallest non-trivial eigenvalues. Suitable
    for local subgraphs (dense eigendecomposition); for whole-genome graphs the
    PE would be precomputed per snarl/window upstream.
    """
    device = edge_index.device
    adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # treat as undirected for the Laplacian

    deg = adj.sum(-1)
    dinv_sqrt = torch.where(
        deg > 0, deg.pow(-0.5), torch.zeros_like(deg)
    )
    lap = torch.eye(num_nodes, device=device) - dinv_sqrt.unsqueeze(1) * adj * dinv_sqrt.unsqueeze(0)

    # Symmetrize for numerical stability, then eigendecompose.
    # `torch.linalg.eigh` is not implemented on MPS; run it on CPU and move back.
    lap = 0.5 * (lap + lap.transpose(0, 1))
    if device.type == "mps":
        eigvals, eigvecs = torch.linalg.eigh(lap.cpu())
        eigvecs = eigvecs.to(device)
    else:
        eigvals, eigvecs = torch.linalg.eigh(lap)
    # Skip the trivial (smallest) eigenvector; take the next pe_dim.
    pe = eigvecs[:, 1 : pe_dim + 1]

    if pe.shape[1] < pe_dim:  # pad small graphs
        pad = torch.zeros(num_nodes, pe_dim - pe.shape[1], device=device)
        pe = torch.cat([pe, pad], dim=1)

    if sign_flip:
        signs = torch.randint(0, 2, (pe.shape[1],), device=device, dtype=torch.float32) * 2 - 1
        pe = pe * signs.unsqueeze(0)

    return pe


class ReferenceGraphEncoder(nn.Module):
    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = KmerTokenizer(k=cfg.kmer_size, stride=1)

        self.kmer_emb = nn.Embedding(
            self.tokenizer.vocab_size, cfg.d_kmer, padding_idx=KmerTokenizer.PAD_ID
        )
        self.node_proj = nn.Linear(cfg.d_kmer, cfg.d_model)
        self.lap_proj = nn.Linear(cfg.lap_pe_dim, cfg.d_model)
        self.edge_type_emb = nn.Embedding(cfg.num_edge_types, cfg.d_edge)

        self.node_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    # ---- convenience: build node k-mer ids from raw node sequences ---------- #
    def encode_node_sequences(
        self, node_seqs: list[str], device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer.batch_encode(node_seqs, device=device)
        return batch["token_ids"], batch["mask"]

    def _pool_node_kmers(
        self, node_kmer_ids: torch.Tensor, node_kmer_mask: torch.Tensor | None
    ) -> torch.Tensor:
        emb = self.kmer_emb(node_kmer_ids)  # (N, Lk, d_kmer)
        if node_kmer_mask is not None:
            m = node_kmer_mask.unsqueeze(-1).to(emb.dtype)
            summed = (emb * m).sum(dim=1)
            counts = m.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        else:
            pooled = emb.mean(dim=1)
        return pooled  # (N, d_kmer)

    def forward(
        self,
        node_kmer_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_kmer_mask: torch.Tensor | None = None,
        lap_pe: torch.Tensor | None = None,
    ) -> GraphEncoding:
        """Encode the graph.

        Args:
            node_kmer_ids: ``(N, Lk)`` k-mer token ids for each node's sequence.
            edge_index: ``(2, E)`` source/target node indices.
            edge_type: ``(E,)`` discrete edge-type ids in ``[0, num_edge_types)``.
            node_kmer_mask: optional ``(N, Lk)`` bool mask for node k-mers.
            lap_pe: optional precomputed ``(N, lap_pe_dim)`` Laplacian PE.
        """
        num_nodes = node_kmer_ids.shape[0]

        node_feat = self.node_proj(self._pool_node_kmers(node_kmer_ids, node_kmer_mask))

        if lap_pe is None:
            lap_pe = laplacian_positional_encoding(
                edge_index,
                num_nodes,
                self.cfg.lap_pe_dim,
                sign_flip=self.cfg.lap_pe_sign_flip and self.training,
            )

        node_emb = node_feat + self.lap_proj(lap_pe)
        node_emb = self.dropout(self.node_norm(node_emb))

        edge_type_embeddings = self.edge_type_emb(edge_type)

        return GraphEncoding(
            node_embeddings=node_emb,
            edge_index=edge_index,
            edge_type_embeddings=edge_type_embeddings,
            lap_pe=lap_pe,
        )
