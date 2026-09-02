"""Stage 4 — the bridge between the neural core and the classical stages.

Stages 1-3 are exact but scoring-blind: to the chaining DP, an exact 15-mer inside
an Alu repeat and one in unique sequence are the same anchor. This module is where
the backbone's learned view is pushed back into that machinery, in three places:

1. **Anchor pruning** (before Stage 2) — :meth:`NeuralScorer.score_anchors` writes
   a confidence into ``AnchorSet.score``, which ``AffineChainer`` already folds
   into its anchor weights, and :meth:`prune_anchors` drops the hopeless ones so
   the quadratic-ish chaining DP runs over a smaller set.
2. **Chain re-ranking** (after Stage 2) — :meth:`NeuralScorer.score_chains` fills
   ``Chain.neural_score``, blended with the DP score for the final ordering.
3. **MAPQ + rescue** (after Stage 3) — a calibrated confidence from the mapping
   head and the primary/secondary margin, plus a proposed locus for reads the
   classical stages placed nowhere.

Everything is batched: one forward pass per batch of reads, with anchors and
chains padded to the batch maximum, because a per-read forward pass would make
the neural stage cost more than the three classical stages combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..config import ScoringConfig
from .types import AnchorSet, Chain

__all__ = [
    "NeuralScorer",
    "ScoredBatch",
    "chain_features",
    "encode_read_batch",
]

#: Width of the chain feature vector built by :func:`chain_features`. Must match
#: ``SeedScoringConfig.num_chain_features``.
NUM_CHAIN_FEATURES = 10

# Stable identity for a candidate anchor and for a directed anchor transition.
AnchorKey = tuple[int, int, int, int]
TransitionKey = tuple[AnchorKey, AnchorKey]


def encode_read_batch(
    reads: Sequence[str],
    device: torch.device | str = "cpu",
    max_len: Optional[int] = None,
    quals: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Pack reads into ``(base_codes, mask, qualities)`` for the core model.

    Unknown bases become code 4 (``N``); padding is 0 with the mask cleared, so a
    padded column is never confused with a real ``A``.

    ``quals`` are the per-base Phred scores that FASTQ and BAM carry. They feed
    the encoder's quality embedding, which is 32 of its 256 input dims; when the
    caller has none (a bare sequence string), ``None`` is returned and the
    encoder falls back to its default.
    """
    from .seeding import encode_bases

    if not reads:
        empty = torch.zeros((0, 0), dtype=torch.long, device=device)
        return empty, empty.bool(), None

    encoded = [encode_bases(r) for r in reads]
    if max_len is not None:
        encoded = [codes[:max_len] for codes in encoded]
    width = max((len(c) for c in encoded), default=1) or 1

    codes = torch.zeros((len(encoded), width), dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, seq in enumerate(encoded):
        n = len(seq)
        if n:
            codes[row, :n] = torch.as_tensor(seq.astype(np.int64))
            mask[row, :n] = True

    qual_tensor: Optional[torch.Tensor] = None
    if quals is not None:
        qual_tensor = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, q in enumerate(quals):
            n = min(len(q), width, len(encoded[row]))
            if n:
                qual_tensor[row, :n] = torch.as_tensor(
                    np.asarray(q[:n], dtype=np.int64)
                )
        qual_tensor = qual_tensor.to(device)

    return codes.to(device), mask.to(device), qual_tensor


def chain_features(
    chain: Chain,
    anchors: AnchorSet,
    read_len: int,
    best_score: float,
    backbone: Optional[np.ndarray] = None,
) -> np.ndarray:
    """A ``(10,)`` geometric descriptor of one chain for the re-ranker.

    Scale-free by construction — spans are fractions of the read and the DP score
    is relative to the read's best chain — so the head sees the same distribution
    for a 150 bp short read and a 100 kb ONT read.
    """
    feats = np.zeros(NUM_CHAIN_FEATURES, dtype=np.float32)
    read_len = max(read_len, 1)
    idx = chain.anchor_idx

    feats[0] = chain.coverage(read_len)
    feats[1] = min(len(idx) / 32.0, 1.0)
    # Relative to the read's own best chain: the absolute DP score scales with
    # read length, the ratio is what discriminates.
    feats[2] = chain.score / best_score if best_score > 0 else 0.0
    feats[3] = chain.anchor_bases(anchors) / read_len
    # >1 means the reference span is longer than the read span (net deletion).
    feats[4] = chain.ref_span / max(chain.read_span, 1)
    feats[5] = float(chain.strand)

    if len(idx):
        diagonals = anchors.diagonal[idx]
        # Diagonal spread is the indel burden the chain implies.
        feats[6] = min(float(diagonals.max() - diagonals.min()) / read_len, 1.0)
        gaps = np.diff(np.sort(anchors.read_pos[idx]))
        feats[7] = float(gaps.mean()) / read_len if gaps.size else 0.0
        scores = anchors.score[idx]
        finite = np.isfinite(scores)
        feats[8] = float(scores[finite].mean()) if finite.any() else 0.5
        if backbone is not None:
            nodes = anchors.node_id[idx]
            known = nodes >= 0
            if known.any():
                on_path = backbone[np.clip(nodes[known], 0, len(backbone) - 1)]
                feats[9] = float(on_path.mean())
    return feats


@dataclass
class ScoredBatch:
    """Stage 4 output for a batch of reads.

    ``mapq`` is per read; ``anchor_scores`` and ``chain_scores`` are ragged, one
    entry per read matching that read's anchor / chain count.
    """

    anchor_scores: list[np.ndarray] = field(default_factory=list)
    chain_scores: list[np.ndarray] = field(default_factory=list)
    mapq: Optional[np.ndarray] = None
    node_id: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    route: Optional[np.ndarray] = None
    #: Raw head outputs, kept so a training loop can reuse this pass for the loss.
    outputs: object | None = None
    seed_head: Optional[dict] = None
    chain_head: Optional[dict] = None


class NeuralScorer:
    """Applies a core model's outputs to anchors, chains, and MAPQ.

    The scorer never mutates the model and runs under ``inference_mode`` by
    default; a training loop constructs it with ``train_mode=True`` to keep the
    graph so the same forward pass feeds :class:`~graphmambaformer.losses.GraphMambaLoss`.
    """

    def __init__(
        self,
        model,
        cfg: ScoringConfig | None = None,
        device: torch.device | str | None = None,
        train_mode: bool = False,
        amp_dtype: torch.dtype | None = None,
        infer_fn: Optional[Callable[..., object]] = None,
        precision_ctx: Optional[Callable] = None,
    ):
        self.model = model
        self.cfg = cfg or ScoringConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.train_mode = train_mode
        self.amp_dtype = amp_dtype
        #: Optional CUDA-graph / TensorRT-wrapped callable for the backbone forward.
        self.infer_fn = infer_fn if infer_fn is not None else model
        #: Optional context manager factory (e.g. AccelContext.precision).
        self.precision_ctx = precision_ctx

    # -- plumbing ------------------------------------------------------------- #
    def _grad_context(self):
        if self.train_mode:
            return torch.enable_grad()
        return torch.inference_mode()

    def _autocast(self):
        if self.precision_ctx is not None and not self.train_mode:
            return self.precision_ctx()
        if self.amp_dtype is None or self.device.type not in ("cuda", "cpu"):
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    @staticmethod
    def _pad_anchor_batch(
        anchor_sets: Sequence[AnchorSet], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad ragged anchor sets to ``(B, A_max)`` tensors for the seed head."""
        width = max((len(a) for a in anchor_sets), default=0) or 1
        B = len(anchor_sets)

        features = torch.zeros((B, width, 12), dtype=torch.float32)
        read_pos = torch.zeros((B, width), dtype=torch.long)
        node = torch.full((B, width), -1, dtype=torch.long)
        mask = torch.zeros((B, width), dtype=torch.bool)

        for row, anchors in enumerate(anchor_sets):
            n = len(anchors)
            if not n:
                continue
            features[row, :n] = torch.as_tensor(anchors.to_seed_features())
            read_pos[row, :n] = torch.as_tensor(anchors.read_pos.astype(np.int64))
            node[row, :n] = torch.as_tensor(anchors.node_id.astype(np.int64))
            mask[row, :n] = True
        return (
            features.to(device),
            read_pos.to(device),
            node.to(device),
            mask.to(device),
        )

    def _pad_seed_graph(
        self, anchor_sets: Sequence[AnchorSet], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build AGNES seed-match DAGs and pad them for one GNN pass.

        Edges obey read order, reference order, strand consistency, and the
        configured gap-discrepancy threshold. Only a bounded successor
        neighborhood is materialized, keeping construction O(A*K).
        """
        cfg = self.model.cfg.seed_scoring
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        active: list[bool] = []
        max_edges = 0

        for anchors in anchor_sets:
            n = len(anchors)
            edge_rows: list[tuple[int, int]] = []
            feature_rows: list[list[float]] = []
            valid_size = cfg.anchor_gnn_min_nodes <= n <= cfg.anchor_gnn_max_nodes
            if valid_size:
                for strand in (1, -1):
                    strand_idx = np.flatnonzero(anchors.strand == strand)
                    order = strand_idx[
                        np.lexsort((anchors.ref_end[strand_idx], anchors.read_end[strand_idx]))
                    ]
                    for slot, src in enumerate(order):
                        successors = order[slot + 1 : slot + 1 + cfg.anchor_gnn_max_neighbors]
                        if successors.size == 0:
                            continue
                        read_gap = anchors.read_pos[successors] - anchors.read_end[src]
                        ref_gap = anchors.ref_pos[successors] - anchors.ref_end[src]
                        discrepancy = np.abs(read_gap - ref_gap)
                        keep = (
                            (read_gap >= 0)
                            & (ref_gap >= 0)
                            & (discrepancy <= cfg.anchor_gnn_gap_threshold)
                        )
                        for dst, dq, dr, gap in zip(
                            successors[keep],
                            read_gap[keep],
                            ref_gap[keep],
                            discrepancy[keep],
                        ):
                            denom = max(float(max(dq, dr)), 1.0)
                            consistency = 1.0 - float(gap) / denom
                            gap_score = np.exp(
                                -0.01 * float(gap) - 0.5 * np.log(float(dq) + 1.0)
                            )
                            uniqueness_delta = abs(
                                1.0 / max(int(anchors.length[src]), 1)
                                - 1.0 / max(int(anchors.length[dst]), 1)
                            )
                            edge_rows.append((int(src), int(dst)))
                            feature_rows.append(
                                [
                                    float(dq) / 1000.0,
                                    float(dr) / 1000.0,
                                    consistency,
                                    float(gap_score),
                                    1.0,
                                    1.0,  # signal continuity unavailable without raw signal
                                    0.0,  # repeat overlap is reserved for annotated references
                                    uniqueness_delta,
                                ]
                            )

            edges = np.asarray(edge_rows, dtype=np.int64).reshape(-1, 2)
            feats = np.asarray(feature_rows, dtype=np.float32).reshape(
                -1, cfg.anchor_edge_features
            )
            rows.append((edges, feats))
            is_active = bool(valid_size and len(edges))
            active.append(is_active)
            max_edges = max(max_edges, len(edges))

        width = max(max_edges, 1)
        B = len(anchor_sets)
        edge_index = torch.zeros((B, width, 2), dtype=torch.long)
        edge_features = torch.zeros(
            (B, width, cfg.anchor_edge_features), dtype=torch.float32
        )
        edge_mask = torch.zeros((B, width), dtype=torch.bool)
        for row, (edges, feats) in enumerate(rows):
            n = len(edges)
            if n:
                edge_index[row, :n] = torch.from_numpy(edges)
                edge_features[row, :n] = torch.from_numpy(feats)
                edge_mask[row, :n] = True
        return (
            edge_index.to(device),
            edge_features.to(device),
            edge_mask.to(device),
            torch.as_tensor(active, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _anchor_key(anchors: AnchorSet, index: int) -> AnchorKey:
        return (
            int(anchors.read_pos[index]),
            int(anchors.ref_pos[index]),
            int(anchors.length[index]),
            int(anchors.strand[index]),
        )

    # -- 1. anchors ----------------------------------------------------------- #
    def score_anchors(
        self, outputs, anchor_sets: Sequence[AnchorSet]
    ) -> tuple[list[np.ndarray], dict]:
        """Score every read's anchors, writing back into ``AnchorSet.score``."""
        features, read_pos, node, mask = self._pad_anchor_batch(anchor_sets, self.device)
        edge_index = edge_features = edge_mask = gnn_active = None
        if self.model.cfg.seed_scoring.use_anchor_gnn:
            edge_index, edge_features, edge_mask, gnn_active = self._pad_seed_graph(
                anchor_sets, self.device
            )
        with self._grad_context(), self._autocast():
            head = self.model.score_seeds(
                outputs,
                seed_features=features,
                anchor_read_pos=read_pos,
                anchor_node=node,
                anchor_mask=mask,
                edge_index=edge_index,
                edge_features=edge_features,
                edge_mask=edge_mask,
                gnn_active=gnn_active,
            )

        scores = head["score"].float().detach().cpu().numpy()
        per_read: list[np.ndarray] = []
        for row, anchors in enumerate(anchor_sets):
            n = len(anchors)
            values = scores[row, :n].astype(np.float32)
            anchors.score[:] = values  # feeds AffineChainer's anchor weights
            per_read.append(values)

        # Preserve learned forward-edge probabilities in coordinate-keyed maps.
        # Coordinate keys survive pruning and the strand/ref sorting done by DP.
        guidance: list[dict[TransitionKey, float]] = [
            {} for _ in anchor_sets
        ]
        if "transition_score" in head and edge_index is not None and edge_mask is not None:
            edge_prob = head["transition_score"].float().detach().cpu().numpy()
            edge_np = edge_index.detach().cpu().numpy()
            live_np = edge_mask.detach().cpu().numpy()
            active_np = head["gnn_active"].detach().cpu().numpy()
            for row, anchors in enumerate(anchor_sets):
                if not active_np[row]:
                    continue
                for (src, dst), prob in zip(edge_np[row][live_np[row]], edge_prob[row][live_np[row]]):
                    guidance[row][
                        (self._anchor_key(anchors, int(src)), self._anchor_key(anchors, int(dst)))
                    ] = float(prob)
        head["transition_guidance"] = guidance
        return per_read, head

    def prune_anchors(self, anchors: AnchorSet) -> AnchorSet:
        """Drop low-confidence anchors, keeping at least ``min_anchors_kept``.

        The floor matters: a read over a hard locus can have every anchor scored
        low, and dropping all of them turns a recoverable alignment into an
        unmapped read. Below the floor the best-scoring anchors are kept
        regardless of the threshold.
        """
        n = len(anchors)
        if n == 0 or not self.cfg.score_seeds:
            return anchors

        scores = np.where(np.isfinite(anchors.score), anchors.score, 1.0)
        threshold = self.model.cfg.seed_scoring.seed_keep_threshold
        keep = np.flatnonzero(scores >= threshold)

        floor = min(self.cfg.min_anchors_kept, n)
        if len(keep) < floor:
            keep = np.argsort(-scores, kind="stable")[:floor]
            keep = np.sort(keep)
        if len(keep) == n:
            return anchors
        return anchors.take(keep)

    # -- 2. chains ------------------------------------------------------------ #
    def score_chains(
        self,
        outputs,
        chains_per_read: Sequence[Sequence[Chain]],
        anchor_sets: Sequence[AnchorSet],
        read_lens: Sequence[int],
        backbone: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], Optional[dict]]:
        """Re-rank each read's chains, writing back ``Chain.neural_score``."""
        n_chains = max((len(c) for c in chains_per_read), default=0)
        if n_chains == 0:
            return [np.zeros(0, dtype=np.float32) for _ in chains_per_read], None

        n_members = max(
            (len(chain) for chains in chains_per_read for chain in chains), default=1
        )
        n_members = max(n_members, 1)
        B = len(chains_per_read)
        d_model = outputs.read_hidden.shape[-1]

        features = torch.zeros((B, n_chains, NUM_CHAIN_FEATURES), dtype=torch.float32)
        member_pos = torch.zeros((B, n_chains, n_members), dtype=torch.long)
        member_mask = torch.zeros((B, n_chains, n_members), dtype=torch.bool)
        chain_mask = torch.zeros((B, n_chains), dtype=torch.bool)

        for row, chains in enumerate(chains_per_read):
            if not chains:
                continue
            best = max(c.score for c in chains)
            anchors = anchor_sets[row]
            for col, chain in enumerate(chains):
                features[row, col] = torch.as_tensor(
                    chain_features(chain, anchors, read_lens[row], best, backbone)
                )
                chain_mask[row, col] = True
                take = chain.anchor_idx[:n_members]
                if len(take):
                    member_pos[row, col, : len(take)] = torch.as_tensor(
                        anchors.read_pos[take].astype(np.int64)
                    )
                    member_mask[row, col, : len(take)] = True

        # Member representations are the read states at each member anchor's read
        # position, so the head pools the backbone's own view of those loci.
        read_hidden = outputs.read_hidden
        length = read_hidden.shape[1]
        flat = member_pos.to(self.device).clamp(0, max(length - 1, 0)).reshape(B, -1)
        gathered = read_hidden.gather(
            1, flat.unsqueeze(-1).expand(-1, -1, d_model)
        ).reshape(B, n_chains, n_members, d_model)
        gathered = gathered * member_mask.to(self.device).unsqueeze(-1).to(gathered.dtype)

        with self._grad_context(), self._autocast():
            head = self.model.score_chains(
                chain_features=features.to(self.device),
                member_states=gathered,
                member_mask=member_mask.to(self.device),
                chain_mask=chain_mask.to(self.device),
            )

        scores = head["score"].float().detach().cpu().numpy()
        per_read: list[np.ndarray] = []
        for row, chains in enumerate(chains_per_read):
            values = scores[row, : len(chains)].astype(np.float32)
            for chain, value in zip(chains, values):
                chain.neural_score = float(value)
            per_read.append(values)
        return per_read, head

    def rerank(self, chains: list[Chain]) -> list[Chain]:
        """Order chains by the DP/neural blend and re-mark the primary.

        The DP score is normalized against the read's best chain first, so the two
        terms are on a common ``[0, 1]`` scale before blending — otherwise the
        weights would mean something different for every read length.
        """
        if not chains or not self.cfg.rerank_chains:
            return chains

        best_dp = max((c.score for c in chains), default=0.0)
        w_dp, w_nn = self.cfg.dp_weight, self.cfg.neural_weight
        total = w_dp + w_nn or 1.0

        def blended(chain: Chain) -> float:
            dp = chain.score / best_dp if best_dp > 0 else 0.0
            neural = chain.neural_score
            if not np.isfinite(neural):
                return dp
            return (w_dp * dp + w_nn * float(neural)) / total

        ordered = sorted(chains, key=blended, reverse=True)
        for rank, chain in enumerate(ordered):
            chain.is_primary = rank == 0
        return ordered

    # -- 3. MAPQ -------------------------------------------------------------- #
    def mapq(
        self,
        outputs,
        chains_per_read: Sequence[Sequence[Chain]],
    ) -> np.ndarray:
        """Blend the head's confidence with the primary/secondary score margin.

        The margin alone is the classical estimate and is blind to *why* a locus
        is ambiguous; the head alone cannot see the runner-up. A read whose best
        two chains score alike is capped low however confident the head is, which
        is the property a downstream variant caller depends on.
        """
        B = len(chains_per_read)
        margins = np.zeros(B, dtype=np.float32)
        for row, chains in enumerate(chains_per_read):
            if not chains:
                continue
            ordered = sorted((c.score for c in chains), reverse=True)
            best = ordered[0]
            if best <= 0:
                continue
            runner_up = ordered[1] if len(ordered) > 1 else 0.0
            margins[row] = np.clip((best - runner_up) / best, 0.0, 1.0)

        mapping = getattr(outputs, "mapping", None)
        if mapping is not None and self.cfg.neural_mapq:
            head = mapping["mapq"].float().detach().cpu().numpy()
            # The head is trained on the MAPQ scale; bring it to [0, 1] to blend.
            confidence = np.clip(head / max(self.cfg.max_mapq, 1), 0.0, 1.0)
            combined = np.minimum(confidence, margins) * 0.5 + margins * 0.5
        else:
            combined = margins

        values = np.rint(combined * self.cfg.max_mapq)
        unmapped = np.array([not chains for chains in chains_per_read], dtype=bool)
        values = np.clip(values, self.cfg.mapq_floor, self.cfg.max_mapq).astype(np.int32)
        values[unmapped] = 0
        return values

    # -- one-shot ------------------------------------------------------------- #
    def run(
        self,
        reads: Sequence[str],
        anchor_sets: Sequence[AnchorSet],
        chains_per_read: Optional[Sequence[Sequence[Chain]]] = None,
        graph=None,
        backbone: Optional[np.ndarray] = None,
        max_len: Optional[int] = None,
        quals: Optional[Sequence[Sequence[int]]] = None,
        modality: Optional[Sequence[str] | str] = None,
    ) -> ScoredBatch:
        """Score a whole batch in a single forward pass.

        ``quals`` and ``modality`` come from the source file when the reads were
        loaded from FASTQ/BAM/uBAM/CRAM; they drive the encoder's quality
        embedding and modality-conditioning token respectively. Both are
        optional so a bare list of sequence strings still works.
        """
        base_codes, mask, qual_tensor = encode_read_batch(
            reads, self.device, max_len=max_len, quals=quals
        )
        forward = self.model if self.train_mode else self.infer_fn
        with self._grad_context(), self._autocast():
            outputs = forward(
                base_codes,
                mask=mask,
                graph=graph,
                qualities=qual_tensor,
                modality=list(modality) if isinstance(modality, (list, tuple)) else modality,
            )

        batch = ScoredBatch(outputs=outputs)
        if self.cfg.score_seeds:
            batch.anchor_scores, batch.seed_head = self.score_anchors(
                outputs, anchor_sets
            )
        if chains_per_read is not None and self.cfg.rerank_chains:
            batch.chain_scores, batch.chain_head = self.score_chains(
                outputs,
                chains_per_read,
                anchor_sets,
                [len(r) for r in reads],
                backbone=backbone,
            )
        if chains_per_read is not None:
            batch.mapq = self.mapq(outputs, chains_per_read)

        mapping = getattr(outputs, "mapping", None)
        if mapping is not None:
            batch.node_id = mapping["node_id"].detach().cpu().numpy()
            batch.position = mapping["position"].float().detach().cpu().numpy()
        router = getattr(outputs, "router", None)
        if router is not None:
            batch.route = router["route"].detach().cpu().numpy()
        return batch
