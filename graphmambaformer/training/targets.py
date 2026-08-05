"""Turn ground-truth reads into supervision targets for the loss.

Labels here are derived from the synthetic dataset's known answer, never
invented: an anchor is positive when it actually points at the read's true
reference locus, and the chain label is the candidate that best overlaps the
true span. That distinction matters -- training against random labels would
still produce a falling loss curve while teaching the model nothing.

Terms whose labels are genuinely unavailable are simply omitted.
:class:`AlignmentLoss` skips any term it finds no target for, so a partially
labelled batch trains on exactly the terms it can justify. Which terms were
supervised is reported in :attr:`Supervision.supervised`, so a run cannot
quietly train on fewer signals than intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..alignment.scoring import chain_features, encode_read_batch
from ..alignment.types import AnchorSet, Chain

__all__ = ["Supervision", "TargetBuilder"]

#: How far an anchor's implied locus may sit from the truth and still count as
#: correct. Minimizer anchors land a few bases off under indel noise.
ANCHOR_TOLERANCE = 20


@dataclass
class Supervision:
    """One training batch: model inputs, head inputs, and matching targets."""

    base_codes: torch.Tensor
    mask: torch.Tensor
    qualities: Optional[torch.Tensor]
    modality: Optional[list[str] | str]

    targets: dict[str, torch.Tensor] = field(default_factory=dict)

    # Stage inputs the scoring heads need, already padded to a rectangle.
    seed_features: Optional[torch.Tensor] = None
    anchor_read_pos: Optional[torch.Tensor] = None
    anchor_node: Optional[torch.Tensor] = None
    anchor_mask: Optional[torch.Tensor] = None
    chain_feats: Optional[torch.Tensor] = None
    chain_mask: Optional[torch.Tensor] = None
    member_states_shape: tuple[int, ...] = ()

    #: Names of the loss terms this batch can actually supervise.
    supervised: tuple[str, ...] = ()
    n_reads: int = 0

    def to(self, device: torch.device | str) -> "Supervision":
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        self.base_codes = move(self.base_codes)
        self.mask = move(self.mask)
        self.qualities = move(self.qualities)
        for key, value in list(self.targets.items()):
            self.targets[key] = value.to(device)
        for name in ("seed_features", "anchor_read_pos", "anchor_node", "anchor_mask",
                     "chain_feats", "chain_mask"):
            setattr(self, name, move(getattr(self, name)))
        return self


def _pad(rows: Sequence[np.ndarray], width: int, dtype) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=dtype)
    for i, row in enumerate(rows):
        n = min(len(row), width)
        if n:
            out[i, :n] = row[:n]
    return out


class TargetBuilder:
    """Builds a :class:`Supervision` batch by running the classical stages.

    Seeding and chaining run for real, so the labels describe the anchors and
    chains the model will actually be asked to score, rather than a synthetic
    stand-in with different statistics.
    """

    def __init__(self, pipeline, model=None, max_anchors: int = 32,
                 max_chains: int = 8, max_members: int = 8,
                 max_read_len: Optional[int] = None):
        self.pipeline = pipeline
        self.max_anchors = max_anchors
        self.max_chains = max_chains
        self.max_members = max_members
        self.max_read_len = max_read_len

        # Feature widths come from the model's own heads so the tensors always
        # match what the heads expect, rather than a duplicated constant.
        model = model or getattr(pipeline, "model", None)
        cfg = getattr(getattr(model, "cfg", None), "seed_scoring", None)
        self.n_seed_features = getattr(cfg, "num_seed_features", 12)
        self.n_chain_features = getattr(cfg, "num_chain_features", 10)

    # ---- labelling -------------------------------------------------------- #
    @staticmethod
    def _anchor_labels(anchors: AnchorSet, ref_start: int, read_len: int) -> np.ndarray:
        """1.0 for anchors consistent with the read's true diagonal."""
        if len(anchors) == 0:
            return np.zeros(0, dtype=np.float32)
        implied = anchors.ref_pos.astype(np.int64) - anchors.read_pos.astype(np.int64)
        return (np.abs(implied - ref_start) <= ANCHOR_TOLERANCE).astype(np.float32)

    @staticmethod
    def _chain_label(chains: Sequence[Chain], ref_start: int, ref_end: int) -> int:
        """Index of the candidate overlapping the truth most, or -1 if none do."""
        best, best_overlap = -1, 0
        for i, chain in enumerate(chains):
            overlap = min(chain.ref_end, ref_end) - max(chain.ref_start, ref_start)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        return best

    # ---- assembly --------------------------------------------------------- #
    def build(self, reads: Sequence, reference) -> Supervision:
        """Run the stages over ``reads`` and label the result against truth."""
        seqs = [r.seq for r in reads]
        quals = [list(r.quals) for r in reads]
        modalities = [r.modality for r in reads]

        base_codes, mask, qual_tensor = encode_read_batch(
            seqs, "cpu", max_len=self.max_read_len, quals=quals
        )

        anchor_sets = self.pipeline.seed(seqs, reference)
        chains_per_read = self.pipeline.chain(anchor_sets, reference)

        n = len(reads)
        n_anchor = max(1, min(self.max_anchors, max((len(a) for a in anchor_sets), default=1)))
        n_chain = max(1, min(self.max_chains, max((len(c) for c in chains_per_read), default=1)))

        feat_rows, pos_rows, node_rows, amask_rows, label_rows = [], [], [], [], []
        chain_feat = np.zeros((n, n_chain, self.n_chain_features), dtype=np.float32)
        chain_mask = np.zeros((n, n_chain), dtype=bool)
        chain_target = np.full(n, -1, dtype=np.int64)
        n_features = self.n_seed_features

        for row, read in enumerate(reads):
            anchors = anchor_sets[row]
            keep = min(len(anchors), n_anchor)

            # AnchorSet.to_seed_features() is the canonical descriptor: scale-free,
            # layout-compatible with the dataset's own Seed.features, and it
            # excludes AnchorSet.score -- which is NaN until the scorer fills it.
            feats = np.zeros((n_anchor, n_features), dtype=np.float32)
            if keep:
                built = anchors.to_seed_features()[:keep]
                feats[:keep, : min(built.shape[1], n_features)] = built[
                    :, :n_features
                ]
            feat_rows.append(feats)

            pos_rows.append(anchors.read_pos[:keep] if keep else np.zeros(0, np.int64))
            # node_id is -1 on a linear reference; the head's embedding needs a
            # valid index, and column 9 of the features already flags validity.
            nodes = (
                anchors.node_id[:keep] if keep else np.zeros(0, np.int64)
            )
            node_rows.append(np.maximum(nodes, 0))
            am = np.zeros(n_anchor, dtype=bool)
            am[:keep] = True
            amask_rows.append(am)

            labels = np.zeros(n_anchor, dtype=np.float32)
            labels[:keep] = self._anchor_labels(
                anchors, read.ref_start, len(read.seq)
            )[:keep]
            label_rows.append(labels)

            chains = list(chains_per_read[row])[:n_chain]
            best_score = max((c.score for c in chains), default=1.0) or 1.0
            for j, chain in enumerate(chains):
                feats = chain_features(chain, anchors, len(read.seq), best_score)
                chain_feat[row, j, : min(len(feats), self.n_chain_features)] = feats[
                    : self.n_chain_features
                ]
                chain_mask[row, j] = True
            label = self._chain_label(chains, read.ref_start, read.ref_end)
            chain_target[row] = label

        features = np.stack(feat_rows)
        chain_feat = np.nan_to_num(chain_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(features).all():
            # A non-finite feature poisons the loss to NaN on the first step and
            # every step after, so it is scrubbed here rather than debugged later.
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        targets: dict[str, torch.Tensor] = {
            "seed_labels": torch.from_numpy(np.stack(label_rows)),
            "anchor_mask": torch.from_numpy(np.stack(amask_rows)),
            "chain_target": torch.from_numpy(chain_target),
            "chain_mask": torch.from_numpy(chain_mask),
            # MAPQ from the generator's own confidence label.
            "mapq_target": torch.tensor([float(r.mapq) for r in reads]),
            "mapq_valid": torch.ones(n, dtype=torch.bool),
        }

        # Within-node offset: the read's true start as a fraction of the
        # reference span it was drawn from.
        ref_len = max(len(reference.ref_seq), 1)
        targets["position_target"] = torch.tensor(
            [min(1.0, max(0.0, r.ref_start / ref_len)) for r in reads],
            dtype=torch.float32,
        )
        targets["position_valid"] = torch.ones(n, dtype=torch.bool)

        supervised = ("seed", "chain", "mapq", "position", "router")
        return Supervision(
            base_codes=base_codes,
            mask=mask,
            qualities=qual_tensor,
            modality=modalities[0] if len(set(modalities)) == 1 else modalities,
            targets=targets,
            seed_features=torch.from_numpy(features),
            anchor_read_pos=torch.from_numpy(
                _pad(pos_rows, n_anchor, np.int64)
            ),
            anchor_node=torch.from_numpy(_pad(node_rows, n_anchor, np.int64)),
            anchor_mask=torch.from_numpy(np.stack(amask_rows)),
            chain_feats=torch.from_numpy(chain_feat),
            chain_mask=torch.from_numpy(chain_mask),
            member_states_shape=(n, n_chain, self.max_members),
            supervised=supervised,
            n_reads=n,
        )

    @staticmethod
    def label_balance(sup: Supervision) -> dict[str, float]:
        """Positive-label rates, to catch a batch that is all-negative."""
        labels = sup.targets["seed_labels"]
        amask = sup.targets["anchor_mask"]
        n_valid = int(amask.sum())
        pos = float((labels * amask).sum())
        chain_t = sup.targets["chain_target"]
        return {
            "anchor_positive_rate": pos / max(n_valid, 1),
            "anchors_per_read": n_valid / max(sup.n_reads, 1),
            "reads_with_chain_label": float((chain_t >= 0).float().mean()),
        }
