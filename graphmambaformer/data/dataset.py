"""CPU-friendly PyTorch wrappers around the synthetic dataset.

Turns :class:`ReadRecord` / :class:`Reference` objects into batches that plug
directly into ``GraphMambaFormerEncoder.forward`` and ``ReferenceGraphEncoder``:

  * :class:`AlignmentDataset` — a ``torch.utils.data.Dataset`` of reads with all
    ground-truth targets attached.
  * :func:`collate_reads` — pads a list of reads into model-ready input tensors
    (``token_ids``, ``qualities``, ``mask``, ``modality``) plus stacked targets.
  * :func:`graph_to_encoder_inputs` — converts a :class:`PangenomeGraph` into the
    tensors expected by ``ReferenceGraphEncoder.forward``.
  * :func:`save_dataset` / :func:`load_dataset` — torch-based (de)serialization.

Everything defaults to CPU and small tensors so it loads on a laptop.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from ..tokenization import KmerTokenizer
from .synthetic import (
    PangenomeGraph,
    ReadRecord,
    Reference,
    SyntheticConfig,
    SyntheticDataset,
    generate_dataset,
)


class AlignmentDataset(Dataset):
    """Reads from one split, exposing raw sequences + all ground-truth targets.

    ``__getitem__`` returns a plain dict (no tokenization) so the collate can
    batch-encode with a shared tokenizer. The ``references`` map is carried
    along so the graph for each read is available.
    """

    def __init__(self, records: list[ReadRecord], references: dict[int, Reference]):
        self.records = records
        self.references = references

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        seeds = rec.seeds
        return {
            "record": rec,
            "seq": rec.seq,
            "quals": rec.quals,
            "modality": rec.modality,
            # alignment targets
            "ref_id": rec.ref_id,
            "ref_start": rec.ref_start,
            "ref_end": rec.ref_end,
            "strand": rec.strand,
            "mapq": rec.mapq,
            "cigar": rec.cigar,
            "ref_positions": torch.tensor(rec.ref_positions, dtype=torch.long),
            # seed-chaining targets
            "seed_read_pos": torch.tensor([s.read_pos for s in seeds], dtype=torch.long),
            "seed_ref_pos": torch.tensor([s.ref_pos for s in seeds], dtype=torch.long),
            "seed_labels": torch.tensor([int(s.is_true) for s in seeds], dtype=torch.long),
            "seed_features": (
                torch.tensor([s.features for s in seeds], dtype=torch.float32)
                if seeds
                else torch.zeros((0, 12), dtype=torch.float32)
            ),
        }


def collate_reads(
    batch: list[dict],
    tokenizer: KmerTokenizer,
    device: Optional[torch.device | str] = None,
    max_read_len: Optional[int] = None,
) -> tuple[dict, dict]:
    """Collate a list of :class:`AlignmentDataset` items.

    Args:
        batch: items from :class:`AlignmentDataset`.
        tokenizer: shared :class:`KmerTokenizer` (defines vocab / k / stride).
        device: optional device to move input tensors to.
        max_read_len: optional cap on base-space read length (before k-mer
            tokenization). Useful to keep the CPU Mamba scan tractable on very
            long reads; ``None`` keeps full length.

    Returns ``(inputs, targets)`` where ``inputs`` feeds
    ``GraphMambaFormerEncoder.forward`` and ``targets`` holds ground truth.
    """
    seqs, quals, modalities = [], [], []
    for item in batch:
        s, q = item["seq"], item["quals"]
        if max_read_len is not None:
            s, q = s[:max_read_len], q[:max_read_len]
        seqs.append(s)
        quals.append(q)
        modalities.append(item["modality"])

    enc = tokenizer.batch_encode(seqs, quals=quals, device=device)

    inputs = {
        "token_ids": enc["token_ids"],
        "qualities": enc["qualities"],
        "mask": enc["mask"],
        "modality": modalities,
    }

    targets = {
        "records": [item["record"] for item in batch],
        "ref_id": torch.tensor([item["ref_id"] for item in batch], dtype=torch.long),
        "ref_start": torch.tensor([item["ref_start"] for item in batch], dtype=torch.long),
        "ref_end": torch.tensor([item["ref_end"] for item in batch], dtype=torch.long),
        "strand": torch.tensor([item["strand"] for item in batch], dtype=torch.long),
        "mapq": torch.tensor([item["mapq"] for item in batch], dtype=torch.long),
        "cigars": [item["cigar"] for item in batch],
        "ref_positions": [item["ref_positions"] for item in batch],
        "seed_read_pos": [item["seed_read_pos"] for item in batch],
        "seed_ref_pos": [item["seed_ref_pos"] for item in batch],
        "seed_labels": [item["seed_labels"] for item in batch],
        "seed_features": [item["seed_features"] for item in batch],
    }
    if device is not None:
        targets["ref_id"] = targets["ref_id"].to(device)
        targets["ref_start"] = targets["ref_start"].to(device)
        targets["ref_end"] = targets["ref_end"].to(device)
        targets["strand"] = targets["strand"].to(device)
        targets["mapq"] = targets["mapq"].to(device)
    return inputs, targets


def graph_to_encoder_inputs(
    graph: PangenomeGraph,
    tokenizer: KmerTokenizer,
    device: Optional[torch.device | str] = None,
) -> dict:
    """Convert a :class:`PangenomeGraph` into ``ReferenceGraphEncoder`` inputs."""
    enc = tokenizer.batch_encode(graph.node_seqs, device=device)
    if graph.edge_index:
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(graph.edge_type, dtype=torch.long)
    if device is not None:
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
    return {
        "node_kmer_ids": enc["token_ids"],
        "node_kmer_mask": enc["mask"],
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def save_dataset(dataset: SyntheticDataset, path: str) -> None:
    """Serialize a :class:`SyntheticDataset` to ``path`` via ``torch.save``."""
    torch.save(dataset, path)


def load_dataset(path: str) -> SyntheticDataset:
    """Load a dataset saved by :func:`save_dataset`."""
    return torch.load(path, weights_only=False)


def build_datasets(
    cfg: Optional[SyntheticConfig] = None,
    dataset: Optional[SyntheticDataset] = None,
) -> tuple[dict[str, AlignmentDataset], SyntheticDataset]:
    """Generate (or wrap) a dataset and return per-split ``AlignmentDataset``s."""
    dataset = dataset or generate_dataset(cfg)
    datasets = {
        split: AlignmentDataset(recs, dataset.references)
        for split, recs in dataset.splits.items()
    }
    return datasets, dataset
