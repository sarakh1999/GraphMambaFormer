"""Build :class:`ReferenceIndex` objects from synthetic (or similar) references.

Training and evaluation both need the same linear-vs-pangenome switch: either
index the plain sequence, or also attach the synthetic pangenome graph so the
GAT / graph-encoder towers actually see nodes and edges.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.graph_mamba import GraphBatch
from ..tokenization import KmerTokenizer
from .dataset import graph_to_encoder_inputs
from .synthetic import PangenomeGraph, Reference

__all__ = ["build_reference_from_synthetic", "pangenome_graph_batch"]


def pangenome_graph_batch(
    graph: PangenomeGraph,
    *,
    kmer_size: int = 3,
    device=None,
) -> GraphBatch:
    """Encode a :class:`PangenomeGraph` into a shared :class:`GraphBatch`."""
    tokenizer = KmerTokenizer(k=kmer_size)
    inputs = graph_to_encoder_inputs(graph, tokenizer, device=device)
    node_lengths = [len(seq) for seq in graph.node_seqs]
    return GraphBatch.from_encoder_inputs(inputs, node_lengths=node_lengths)


def build_reference_from_synthetic(
    pipeline,
    ref: Reference,
    *,
    with_graph: bool = True,
    kmer_size: int = 3,
    device=None,
):
    """Build a pipeline reference index, optionally with the pangenome graph.

    * ``with_graph=False`` — linear genome only (graph towers stay idle).
    * ``with_graph=True``  — attach node sequences, backbone, edge index, and a
      :class:`GraphBatch` so chaining + the neural model use the pangenome.
    """
    if not with_graph or ref.graph is None:
        return pipeline.build_reference(ref.seq, ref_id=ref.ref_id)

    graph = ref.graph
    edge_index = (
        np.asarray(graph.edge_index, dtype=np.int64).T
        if graph.edge_index
        else np.zeros((2, 0), dtype=np.int64)
    )
    batch = pangenome_graph_batch(graph, kmer_size=kmer_size, device=device)
    return pipeline.build_reference(
        ref.seq,
        ref_id=ref.ref_id,
        node_seqs=graph.node_seqs,
        node_ref_start=graph.node_ref_start,
        backbone_path=graph.backbone_path,
        edge_index=edge_index,
        graph=batch,
    )
