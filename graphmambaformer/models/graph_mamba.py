"""GraphMambaModel — the core neural model.

Forward pass (architecture: Core Model)::

    read  -> SequenceEncoder -> BiMamba2 x 6 --.
                                               +-> CrossAttentionFusion -> pooled
    graph -> GraphEncoder ----> GATv2Conv x 3 -'          |
                                                          v
                                        ComplexityRouter -> MappingHead
                                                          |
                                                          +-> multi-task heads

Two design choices matter for the alignment pipeline that consumes this model:

**Base-space read states.** The read tower stays at one hidden state per read
*base* (not per k-mer token), so Stage 4 can index ``read_hidden[b, read_pos]``
for an anchor without a coordinate transform.

**Block-diagonal graph batching.** Reads in a batch usually share one reference
graph, so the default is to encode it once and broadcast — encoding it ``B``
times would dominate the forward pass. When reads carry *different* subgraphs,
:meth:`GraphBatch.collate` concatenates them into one disconnected graph with
offset edge indices, which lets a single GATv2 call cover the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..config import GraphMambaConfig
from ..encoders.graph_encoder import ReferenceGraphEncoder
from ..encoders.sequence_encoder import SequenceEncoder
from ..heads.mapping_head import MappingHead
from ..heads.multitask_heads import MultiTaskHeads
from ..heads.router import ComplexityRouter
from ..heads.scoring_heads import ChainScoringHead, SeedScoringHead
from ..layers.bimamba import BiMamba2
from ..layers.common import FeedForward, RMSNorm
from ..layers.cross_attention import CrossAttentionFusion
from ..layers.gat import GATv2Layer


# --------------------------------------------------------------------------- #
# Graph batching
# --------------------------------------------------------------------------- #
@dataclass
class GraphBatch:
    """A pangenome graph (or several) prepared for one model forward pass.

    ``shared=True`` means every read in the batch aligns against this one graph,
    so it is encoded once and broadcast. Otherwise the graphs are concatenated
    block-diagonally and ``node_batch`` records each node's owning read.
    """

    node_kmer_ids: torch.Tensor  # (N_total, Lk)
    edge_index: torch.Tensor  # (2, E_total)
    edge_type: torch.Tensor  # (E_total,)
    node_kmer_mask: Optional[torch.Tensor] = None  # (N_total, Lk)
    node_batch: Optional[torch.Tensor] = None  # (N_total,) owning read index
    node_lengths: Optional[torch.Tensor] = None  # (N_total,) bases per node
    num_graphs: int = 1
    shared: bool = True

    @property
    def num_nodes(self) -> int:
        return int(self.node_kmer_ids.shape[0])

    def to(self, device: torch.device | str) -> "GraphBatch":
        move = lambda t: t.to(device) if isinstance(t, torch.Tensor) else t  # noqa: E731
        return GraphBatch(
            node_kmer_ids=move(self.node_kmer_ids),
            edge_index=move(self.edge_index),
            edge_type=move(self.edge_type),
            node_kmer_mask=move(self.node_kmer_mask),
            node_batch=move(self.node_batch),
            node_lengths=move(self.node_lengths),
            num_graphs=self.num_graphs,
            shared=self.shared,
        )

    @classmethod
    def from_encoder_inputs(cls, inputs: dict, node_lengths: Optional[Sequence[int]] = None) -> "GraphBatch":
        """Wrap the dict produced by ``data.graph_to_encoder_inputs`` (one shared graph)."""
        return cls(
            node_kmer_ids=inputs["node_kmer_ids"],
            edge_index=inputs["edge_index"],
            edge_type=inputs["edge_type"],
            node_kmer_mask=inputs.get("node_kmer_mask"),
            node_lengths=(
                torch.as_tensor(node_lengths, dtype=torch.long)
                if node_lengths is not None
                else None
            ),
            num_graphs=1,
            shared=True,
        )

    @classmethod
    def collate(cls, graphs: Sequence[dict]) -> "GraphBatch":
        """Concatenate per-read graphs into one disconnected graph.

        Node k-mer rows are right-padded to a common width and edge indices are
        offset by each graph's node count, so a single GATv2 pass over the union
        is exactly equivalent to running each graph separately.
        """
        width = max(int(g["node_kmer_ids"].shape[1]) for g in graphs)
        ids, masks, edges, types, owners, lengths = [], [], [], [], [], []
        offset = 0

        for index, graph in enumerate(graphs):
            node_ids = graph["node_kmer_ids"]
            n, w = node_ids.shape
            if w < width:
                pad = torch.zeros((n, width - w), dtype=node_ids.dtype, device=node_ids.device)
                node_ids = torch.cat([node_ids, pad], dim=1)
            ids.append(node_ids)

            node_mask = graph.get("node_kmer_mask")
            if node_mask is None:
                node_mask = torch.ones((n, w), dtype=torch.bool, device=node_ids.device)
            if node_mask.shape[1] < width:
                pad = torch.zeros(
                    (n, width - node_mask.shape[1]), dtype=torch.bool, device=node_mask.device
                )
                node_mask = torch.cat([node_mask, pad], dim=1)
            masks.append(node_mask)

            edges.append(graph["edge_index"] + offset)
            types.append(graph["edge_type"])
            owners.append(torch.full((n,), index, dtype=torch.long, device=node_ids.device))
            if graph.get("node_lengths") is not None:
                lengths.append(torch.as_tensor(graph["node_lengths"], dtype=torch.long))
            offset += n

        return cls(
            node_kmer_ids=torch.cat(ids, dim=0),
            edge_index=torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.cat(types, dim=0) if types else torch.zeros(0, dtype=torch.long),
            node_kmer_mask=torch.cat(masks, dim=0),
            node_batch=torch.cat(owners, dim=0),
            node_lengths=torch.cat(lengths, dim=0) if len(lengths) == len(graphs) else None,
            num_graphs=len(graphs),
            shared=False,
        )


@dataclass
class GraphMambaOutput:
    """Everything one forward pass produces.

    ``read_hidden`` is in base space, so ``read_hidden[b, i]`` is read ``b``'s
    base ``i`` — which is what Stage 4 indexes with anchor positions.
    """

    read_hidden: torch.Tensor  # (B, L, D)
    read_mask: torch.Tensor  # (B, L)
    graph_nodes: torch.Tensor  # (B, N, D)
    graph_mask: torch.Tensor  # (B, N)
    fused: torch.Tensor  # (B, L + N, D)
    pooled: torch.Tensor  # (B, D)
    mapping: dict[str, torch.Tensor] = field(default_factory=dict)
    router: dict[str, torch.Tensor] = field(default_factory=dict)
    multitask: dict[str, torch.Tensor] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Towers
# --------------------------------------------------------------------------- #
class BiMambaTower(nn.Module):
    """``n`` bidirectional Mamba-2 layers, each a pre-norm residual (+ optional FFN)."""

    def __init__(self, cfg: GraphMambaConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_mamba_layers):
            block = nn.ModuleDict(
                {
                    "norm": RMSNorm(cfg.d_model),
                    "mixer": BiMamba2(cfg.mamba),
                }
            )
            if cfg.mamba_ffn:
                block["ffn_norm"] = RMSNorm(cfg.d_model)
                block["ffn"] = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
            self.layers.append(block)
        self.final_norm = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.layers:
            x = x + block["mixer"](block["norm"](x), mask=mask)
            if "ffn" in block:
                x = x + block["ffn"](block["ffn_norm"](x))
        return self.final_norm(x)


class GATv2Tower(nn.Module):
    """``n`` GATv2 layers refining graph node embeddings in place."""

    def __init__(self, cfg: GraphMambaConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict({"norm": RMSNorm(cfg.d_model), "conv": GATv2Layer(cfg.gat)})
            for _ in range(cfg.n_gat_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.layers:
            nodes = nodes + block["conv"](block["norm"](nodes), edge_index, edge_attr)
        return self.final_norm(nodes)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GraphMambaModel(nn.Module):
    """BiMamba2 + GATv2 + cross-attention alignment backbone."""

    def __init__(self, cfg: GraphMambaConfig | None = None):
        super().__init__()
        self.cfg = cfg or GraphMambaConfig()

        self.sequence_encoder = SequenceEncoder(self.cfg.sequence_encoder)
        self.mamba_tower = BiMambaTower(self.cfg)

        self.graph_encoder = ReferenceGraphEncoder(self.cfg.graph_encoder)
        self.gat_tower = GATv2Tower(self.cfg)

        self.fusion = CrossAttentionFusion(self.cfg.cross_attention)
        self.router = ComplexityRouter(self.cfg.router) if self.cfg.use_router else None
        self.mapping_head = MappingHead(self.cfg.mapping_head)

        self.seed_scorer = SeedScoringHead(self.cfg.seed_scoring)
        self.chain_scorer = ChainScoringHead(self.cfg.seed_scoring)

    # ---- towers ------------------------------------------------------------- #
    def encode_read(
        self,
        base_codes: torch.Tensor,
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        modality: str | int | list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read tower: SequenceEncoder -> BiMamba2 stack. Returns ``(hidden, mask)``."""
        hidden, mask = self.sequence_encoder(
            base_codes, qualities=qualities, mask=mask, modality=modality
        )
        return self.mamba_tower(hidden, mask=mask), mask

    def _laplacian_pe(self, graph: GraphBatch) -> torch.Tensor | None:
        """Laplacian PE for a collated batch, computed one graph at a time.

        The PE is an eigendecomposition of the graph Laplacian, so it is *not*
        separable across a block-diagonal union: the eigenvectors of two joined
        components differ from each component's own. Letting the encoder see the
        union would silently give each read a positional encoding that depends on
        the other reads in the batch, so each component is decomposed separately
        and the results are concatenated.
        """
        if graph.shared or graph.node_batch is None or graph.num_graphs <= 1:
            return None  # single graph: the encoder's own computation is correct

        from ..encoders.graph_encoder import laplacian_positional_encoding

        owner = graph.node_batch
        pieces = []
        for index in range(graph.num_graphs):
            local = torch.nonzero(owner == index, as_tuple=True)[0]
            n_local = int(local.numel())
            if n_local == 0:
                continue
            # Re-index this component's edges into local node numbering.
            offset = int(local.min().item())
            keep = (
                (owner[graph.edge_index[0]] == index)
                if graph.edge_index.numel()
                else torch.zeros(0, dtype=torch.bool, device=owner.device)
            )
            local_edges = graph.edge_index[:, keep] - offset if keep.any() else graph.edge_index[:, :0]
            pieces.append(
                laplacian_positional_encoding(
                    local_edges,
                    n_local,
                    self.cfg.graph_encoder.lap_pe_dim,
                    sign_flip=self.cfg.graph_encoder.lap_pe_sign_flip and self.training,
                )
            )
        return torch.cat(pieces, dim=0) if pieces else None

    def encode_graph(
        self, graph: GraphBatch, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Graph tower: GraphEncoder -> GATv2 stack, densified to ``(B, N, D)``.

        Returns ``(node_states, node_mask, node_lengths)``.
        """
        encoding = self.graph_encoder(
            node_kmer_ids=graph.node_kmer_ids,
            edge_index=graph.edge_index,
            edge_type=graph.edge_type,
            node_kmer_mask=graph.node_kmer_mask,
            lap_pe=self._laplacian_pe(graph),
        )
        nodes = self.gat_tower(
            encoding.node_embeddings, graph.edge_index, encoding.edge_type_embeddings
        )
        return self._densify(nodes, graph, batch_size)

    @staticmethod
    def _densify(
        nodes: torch.Tensor, graph: GraphBatch, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turn flat node states into a padded ``(B, N, D)`` tensor plus its mask."""
        device = nodes.device
        lengths = (
            graph.node_lengths.to(device)
            if graph.node_lengths is not None
            else torch.ones(nodes.shape[0], dtype=torch.long, device=device)
        )

        if graph.shared or graph.node_batch is None:
            dense = nodes.unsqueeze(0).expand(batch_size, -1, -1)
            mask = torch.ones(
                (batch_size, nodes.shape[0]), dtype=torch.bool, device=device
            )
            return dense, mask, lengths.unsqueeze(0).expand(batch_size, -1)

        owner = graph.node_batch.to(device)
        counts = torch.bincount(owner, minlength=batch_size)
        width = int(counts.max().item()) if counts.numel() else 1
        # Position of each node within its own graph.
        slot = torch.arange(nodes.shape[0], device=device) - torch.cumsum(
            torch.cat([torch.zeros(1, dtype=counts.dtype, device=device), counts[:-1]]), 0
        )[owner]

        dense = nodes.new_zeros((batch_size, width, nodes.shape[-1]))
        dense[owner, slot] = nodes
        mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        mask[owner, slot] = True
        dense_lengths = torch.ones((batch_size, width), dtype=torch.long, device=device)
        dense_lengths[owner, slot] = lengths
        return dense, mask, dense_lengths

    # ---- forward ------------------------------------------------------------ #
    def forward(
        self,
        base_codes: torch.Tensor,
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        graph: GraphBatch | None = None,
        modality: str | int | list | None = None,
        run_heads: bool = True,
    ) -> GraphMambaOutput:
        """Run the full model on a batch of reads against a pangenome graph."""
        batch_size = base_codes.shape[0]
        # Multi-GPU (DataParallel) scatters ``base_codes`` per replica; move the
        # shared GraphBatch onto that replica's device so GAT/fusion stay local.
        if graph is not None:
            graph = graph.to(base_codes.device)

        read_hidden, read_mask = self.encode_read(
            base_codes, qualities=qualities, mask=mask, modality=modality
        )

        if graph is not None:
            graph_nodes, graph_mask, node_lengths = self.encode_graph(graph, batch_size)
        else:
            # No graph: a single zero "node" keeps every downstream shape valid.
            graph_nodes = read_hidden.new_zeros((batch_size, 1, self.cfg.d_model))
            graph_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=read_hidden.device
            )
            node_lengths = torch.ones(
                (batch_size, 1), dtype=torch.long, device=read_hidden.device
            )

        fused = self.fusion(read_hidden, graph_nodes, read_mask, graph_mask)

        output = GraphMambaOutput(
            read_hidden=fused["read"],
            read_mask=read_mask,
            graph_nodes=fused["graph"],
            graph_mask=graph_mask,
            fused=fused["fused"],
            pooled=fused["pooled"],
        )
        if not run_heads:
            return output

        if self.router is not None:
            output.router = self.router(output.pooled)
        output.mapping = self.mapping_head(
            output.pooled,
            node_embeddings=output.graph_nodes,
            node_mask=graph_mask,
            node_lengths=node_lengths,
        )
        return output

    # ---- Stage 4 scoring ---------------------------------------------------- #
    def score_seeds(
        self,
        output: GraphMambaOutput,
        seed_features: torch.Tensor,
        anchor_read_pos: torch.Tensor,
        anchor_node: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score anchors against a completed forward pass."""
        return self.seed_scorer(
            seed_features=seed_features,
            read_hidden=output.read_hidden,
            anchor_read_pos=anchor_read_pos,
            graph_nodes=output.graph_nodes,
            anchor_node=anchor_node,
            anchor_mask=anchor_mask,
        )

    def score_chains(
        self,
        chain_features: torch.Tensor,
        member_states: torch.Tensor,
        member_mask: torch.Tensor,
        chain_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Re-rank candidate chains."""
        return self.chain_scorer(chain_features, member_states, member_mask, chain_mask)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


class MultiTaskGraphMamba(GraphMambaModel):
    """:class:`GraphMambaModel` plus the ten multi-task heads.

    The heads share the backbone, so predictions are produced by the same forward
    pass as the alignment — the "zero extra cost during alignment" property the
    predictive-genomics engine relies on.
    """

    def __init__(self, cfg: GraphMambaConfig | None = None):
        super().__init__(cfg)
        self.task_heads = MultiTaskHeads(self.cfg.multi_task)

    @property
    def enabled_tasks(self) -> tuple[str, ...]:
        return self.task_heads.enabled

    def forward(
        self,
        base_codes: torch.Tensor,
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        graph: GraphBatch | None = None,
        modality: str | int | list | None = None,
        run_heads: bool = True,
    ) -> GraphMambaOutput:
        output = super().forward(
            base_codes,
            qualities=qualities,
            mask=mask,
            graph=graph,
            modality=modality,
            run_heads=run_heads,
        )
        if run_heads:
            output.multitask = self.task_heads(
                output.pooled,
                read_states=output.read_hidden,
                node_states=output.graph_nodes,
            )
        return output
