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

from ..accel.parallel import parallel_map
from ..alignment.scoring import chain_features, encode_read_batch
from ..alignment.types import AnchorSet, Chain

__all__ = ["Supervision", "TargetBuilder"]

#: How far an anchor's implied locus may sit from the truth and still count as
#: correct. Minimizer anchors land a few bases off under indel noise.
ANCHOR_TOLERANCE = 20

#: Hard-read detection thresholds (feed ``targets["read_difficulty"]``, which the
#: loss turns into a per-read weight via ``LossConfig.hard_read_weight``). A read
#: is "weakly placed" (difficulty 0.5) when the chainer left it a single candidate
#: or the best chain covers less than this fraction of the read, and "failed"
#: (difficulty 1.0) when the chainer found nothing or put the best chain's locus
#: more than ``position_window`` bases off the truth. Coverage floor only; the
#: locus tolerance is read from the builder's ``position_window`` so it tracks the
#: scale the position head is trained on.
HARD_READ_MIN_COVERAGE = 0.5


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
    seed_edge_index: Optional[torch.Tensor] = None
    seed_edge_features: Optional[torch.Tensor] = None
    seed_edge_mask: Optional[torch.Tensor] = None
    seed_gnn_active: Optional[torch.Tensor] = None
    chain_feats: Optional[torch.Tensor] = None
    chain_mask: Optional[torch.Tensor] = None
    #: ``(B, n_chain, n_members)`` read positions of each chain's member anchors,
    #: and a matching validity mask. The trainer gathers the backbone's
    #: ``read_hidden`` at these positions to build the chain head's member states,
    #: which is exactly what the inference re-ranker does
    #: (:meth:`NeuralScorer.score_chains`). Keeping them here is what lets
    #: training feed the head the *same* member representation it is served at
    #: inference, instead of the zeros that left the member branch untrained.
    chain_member_pos: Optional[torch.Tensor] = None
    chain_member_mask: Optional[torch.Tensor] = None
    member_states_shape: tuple[int, ...] = ()

    #: Names of the loss terms this batch can actually supervise.
    supervised: tuple[str, ...] = ()
    n_reads: int = 0

    def to(self, device: torch.device | str) -> "Supervision":
        """Return a *copy* on ``device`` — ``self`` is left untouched.

        Non-mutating on purpose: the trainer caches the (deterministic, model-
        independent) supervision on the host and moves it to the GPU afresh every
        epoch, so mutating in place would clobber the cached CPU copy (turning it
        into GPU tensors and re-pinning the whole dataset). Building a new object
        instead keeps the cache pageable and reusable.

        The H2D transfer still overlaps compute: each CPU source is staged
        through a *fresh* pinned buffer and copied ``non_blocking`` when the
        target is CUDA (``non_blocking`` only helps from pinned memory). The
        cached tensors themselves are never page-locked, so the whole dataset is
        not permanently pinned. Off CUDA (or if pinning is unavailable) this is a
        plain blocking copy.
        """
        dev = torch.device(device)
        use_async = dev.type == "cuda"

        def move(t):
            if t is None:
                return None
            if t.device == dev:
                return t
            if use_async and not t.is_cuda:
                src = t
                try:
                    src = t.pin_memory()  # a new pinned tensor; does not touch t
                except (RuntimeError, NotImplementedError):
                    src = t
                return src.to(dev, non_blocking=True)
            return t.to(dev)

        return Supervision(
            base_codes=move(self.base_codes),
            mask=move(self.mask),
            qualities=move(self.qualities),
            modality=self.modality,
            targets={key: move(value) for key, value in self.targets.items()},
            seed_features=move(self.seed_features),
            anchor_read_pos=move(self.anchor_read_pos),
            anchor_node=move(self.anchor_node),
            anchor_mask=move(self.anchor_mask),
            seed_edge_index=move(self.seed_edge_index),
            seed_edge_features=move(self.seed_edge_features),
            seed_edge_mask=move(self.seed_edge_mask),
            seed_gnn_active=move(self.seed_gnn_active),
            chain_feats=move(self.chain_feats),
            chain_mask=move(self.chain_mask),
            chain_member_pos=move(self.chain_member_pos),
            chain_member_mask=move(self.chain_member_mask),
            member_states_shape=self.member_states_shape,
            supervised=self.supervised,
            n_reads=self.n_reads,
        )


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
                 max_read_len: Optional[int] = None,
                 decoy_chains: int = 1,
                 position_window: float = 512.0):
        self.pipeline = pipeline
        self.max_anchors = max_anchors
        self.max_chains = max_chains
        self.max_members = max_members
        self.max_read_len = max_read_len
        # Half-width (bp) of the window the position head regresses within. The
        # position target is a *local* offset relative to the locus the classical
        # chainer already found (see ``_local_position_target``), not an absolute
        # genomic coordinate, so this sets the scale over which that offset spans
        # roughly [-1, 1].
        self.position_window = float(position_window)
        self._position_warned = False
        # Number of synthetic hard-negative chains to inject for a read that the
        # chainer collapsed to a single candidate. A listwise cross-entropy over
        # one candidate is identically zero with zero gradient (softmax of a
        # singleton is 1.0), so a normal single-locus read teaches the chain
        # re-ranker nothing. Injecting a plausible-but-worse decoy (a sub-chain
        # of, or a displaced copy of, the true chain) gives the ranking loss a
        # real ordering to learn while leaving the true chain as the target.
        self.decoy_chains = int(decoy_chains)

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

    def _heuristic_difficulty(
        self, read, chains: Sequence[Chain], read_len: int
    ) -> float:
        """How badly the classical heuristics handled this read, in ``[0, 1]``.

        * ``1.0`` — the chainer produced no chain at all, or its best chain's
          locus is more than ``position_window`` bases off the truth (a gross
          misplacement the neural stage must rescue).
        * ``0.5`` — placed but weakly: a single candidate (nothing to rank) or a
          best chain covering less than ``HARD_READ_MIN_COVERAGE`` of the read.
        * ``0.0`` — a clean, confident placement.

        Read from the *pre-decoy* chains so the "single candidate" signal reflects
        what the chainer actually returned, not the training decoys injected
        later. Feeds ``targets["read_difficulty"]``; the loss turns it into a
        per-read weight via ``LossConfig.hard_read_weight``.
        """
        if not chains:
            return 1.0
        best = chains[0]  # select() returns candidates best-score-first
        truth = getattr(read, "ref_start", None)
        if truth is not None and abs(int(best.ref_start) - int(truth)) > self.position_window:
            return 1.0
        coverage = (
            best.coverage(read_len)
            if hasattr(best, "coverage")
            else (best.read_end - best.read_start) / max(read_len, 1)
        )
        if len(chains) < 2 or coverage < HARD_READ_MIN_COVERAGE:
            return 0.5
        return 0.0

    @staticmethod
    def _local_position_target(
        reads: Sequence, chains_per_read: Sequence[Sequence[Chain]], window: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """A *local* within-window offset target the position head can predict.

        The old target, ``read.ref_start / len(whole_reference)``, is an absolute
        genomic coordinate: predicting it from a single read embedding is ill-posed,
        so the Huber regressor minimized by emitting the batch mean (position head
        variance collapsed to ~0). Instead express the read's start as a signed
        fraction of a small ``window`` around the locus the classical chainer
        already localized::

            target = clamp( (read.ref_start - chain.ref_start) / window, -1, 1 )

        The fused embedding attends to the reference in that neighbourhood, so this
        residual is in-distribution and learnable. Reads with no chain are marked
        invalid so they do not drag the head toward a meaningless value.
        """
        n = len(reads)
        tgt = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        w = max(float(window), 1.0)
        for i, (read, chains) in enumerate(zip(reads, chains_per_read)):
            ref_start = getattr(read, "ref_start", None)
            if ref_start is None or not chains:
                continue
            anchor = min((c.ref_start for c in chains), default=None)
            if anchor is None:
                continue
            offset = (float(ref_start) - float(anchor)) / w
            tgt[i] = max(-1.0, min(1.0, offset))
            valid[i] = True
        return tgt, valid

    @staticmethod
    def _subchain(parent: Chain, anchors: AnchorSet, keep: np.ndarray) -> Chain:
        """A decoy chain built from a strict subset of ``parent``'s anchors.

        Fewer anchors -> lower coverage and a shorter span, so its geometric
        features (and the score, scaled by the retained anchor fraction) are
        strictly worse than the parent's. Because its reference span is contained
        in the parent's, it never out-overlaps the truth, so the parent stays the
        ranking target. This is a genuine hard negative: the re-ranker must learn
        to prefer the more complete collinear chain.
        """
        idx = np.asarray(parent.anchor_idx, dtype=np.int64)[keep]
        read_pos = anchors.read_pos[idx]
        read_end = anchors.read_end[idx]
        ref_pos = anchors.ref_pos[idx]
        ref_end = anchors.ref_end[idx]
        frac = len(idx) / max(len(parent.anchor_idx), 1)
        return Chain(
            anchor_idx=idx,
            score=float(parent.score) * frac,
            strand=parent.strand,
            read_start=int(read_pos.min()),
            read_end=int(read_end.max()),
            ref_start=int(ref_pos.min()),
            ref_end=int(ref_end.max()),
        )

    def _make_decoys(
        self, chains: Sequence[Chain], anchors: AnchorSet, read_len: int
    ) -> list[Chain]:
        """Synthesize up to ``decoy_chains`` hard negatives for a lone chain.

        Prefers strict sub-chains of the primary (real, collinear, but less
        complete). A single-anchor primary cannot be sub-chained, so it falls
        back to a displaced, down-scored copy sitting a full read-length off the
        true locus -- a wrong placement the re-ranker should rank below the true
        chain. Returns an empty list when no decoy can be built.
        """
        if not chains or self.decoy_chains <= 0:
            return []
        primary = chains[0]
        k = len(primary.anchor_idx)
        decoys: list[Chain] = []

        if k >= 2:
            # Progressively shorter prefixes of the primary: dropping the last
            # anchor is the hardest negative, half-length a clearer one.
            drops = [1]
            if k >= 4:
                drops.append(k - max(1, k // 2))
            for d in drops:
                keep_n = k - d
                if keep_n < 1:
                    continue
                decoys.append(
                    self._subchain(primary, anchors, np.arange(keep_n, dtype=np.int64))
                )
                if len(decoys) >= self.decoy_chains:
                    break
        else:
            shift = max(read_len, 1)
            decoys.append(
                Chain(
                    anchor_idx=np.asarray(primary.anchor_idx, dtype=np.int64).copy(),
                    score=float(primary.score) * 0.5,
                    strand=primary.strand,
                    read_start=primary.read_start,
                    read_end=primary.read_end,
                    ref_start=primary.ref_start + shift,
                    ref_end=primary.ref_end + shift,
                )
            )
        return decoys[: self.decoy_chains]

    def _augment_with_decoys(
        self, chains: Sequence[Chain], anchors: AnchorSet, read_len: int
    ) -> list[Chain]:
        """Append synthetic decoys when the chainer returned a single candidate."""
        chains = list(chains)
        if len(chains) >= 2 or not chains:
            return chains
        return chains + self._make_decoys(chains, anchors, read_len)

    def _build_row(
        self,
        read,
        anchors: AnchorSet,
        chains_in: Sequence[Chain],
        n_anchor: int,
        n_chain: int,
        n_features: int,
    ) -> tuple:
        """Assemble one read's padded seed/chain features and labels.

        Pure per-read work (no shared mutable state), so it is safe to run
        concurrently across the host thread pool. Returns everything the caller
        stacks into the batch tensors, in a fixed tuple order.
        """
        keep = min(len(anchors), n_anchor)

        # AnchorSet.to_seed_features() is the canonical descriptor: scale-free,
        # layout-compatible with the dataset's own Seed.features, and it
        # excludes AnchorSet.score -- which is NaN until the scorer fills it.
        feats = np.zeros((n_anchor, n_features), dtype=np.float32)
        if keep:
            built = anchors.to_seed_features()[:keep]
            feats[:keep, : min(built.shape[1], n_features)] = built[:, :n_features]

        pos = anchors.read_pos[:keep] if keep else np.zeros(0, np.int64)
        # node_id is -1 on a linear reference; the head's embedding needs a
        # valid index, and column 9 of the features already flags validity.
        nodes = anchors.node_id[:keep] if keep else np.zeros(0, np.int64)
        nodes = np.maximum(nodes, 0)

        amask = np.zeros(n_anchor, dtype=bool)
        amask[:keep] = True

        labels = np.zeros(n_anchor, dtype=np.float32)
        labels[:keep] = self._anchor_labels(anchors, read.ref_start, len(read.seq))[:keep]

        chains = list(chains_in)[:n_chain]
        best_score = max((c.score for c in chains), default=1.0) or 1.0
        chain_feat_row = np.zeros((n_chain, self.n_chain_features), dtype=np.float32)
        chain_mask_row = np.zeros(n_chain, dtype=bool)
        # Per-chain member read positions + validity, so the trainer can gather
        # the SAME per-anchor backbone states the inference re-ranker pools over
        # (NeuralScorer.score_chains gathers ``read_hidden`` at these positions).
        # Members are the chain's anchors, indexed into this read's AnchorSet.
        n_members = self.max_members
        n_anchors_total = len(anchors)
        member_pos_row = np.zeros((n_chain, n_members), dtype=np.int64)
        member_mask_row = np.zeros((n_chain, n_members), dtype=bool)
        for j, chain in enumerate(chains):
            cfeats = chain_features(chain, anchors, len(read.seq), best_score)
            chain_feat_row[j, : min(len(cfeats), self.n_chain_features)] = cfeats[
                : self.n_chain_features
            ]
            chain_mask_row[j] = True
            idx = np.asarray(chain.anchor_idx, dtype=np.int64)[:n_members]
            idx = idx[(idx >= 0) & (idx < n_anchors_total)]
            if idx.size:
                member_pos_row[j, : idx.size] = anchors.read_pos[idx].astype(np.int64)
                member_mask_row[j, : idx.size] = True
        chain_target = self._chain_label(chains, read.ref_start, read.ref_end)

        return (feats, pos, nodes, amask, labels, chain_feat_row, chain_mask_row,
                chain_target, member_pos_row, member_mask_row)

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

        # Per-read heuristic difficulty (from the *pre-decoy* chains), used by the
        # loss to up-weight the reads the classical stages struggled with. Scored
        # here so it sees exactly what seeding+chaining produced for this read.
        read_difficulty = np.array(
            [
                self._heuristic_difficulty(read, chains, len(read.seq))
                for read, chains in zip(reads, chains_per_read)
            ],
            dtype=np.float32,
        )

        # Inject hard-negative decoy chains for reads the chainer collapsed to a
        # single candidate, so the listwise chain-ranking loss has >=2 candidates
        # to order (a singleton list makes that term identically zero). Done
        # before n_chain is sized so the decoys fit in the padded chain tensors.
        if self.decoy_chains > 0:
            chains_per_read = [
                self._augment_with_decoys(chains, anchors, len(read.seq))
                for read, anchors, chains in zip(reads, anchor_sets, chains_per_read)
            ]

        n = len(reads)
        n_anchor = max(1, min(self.max_anchors, max((len(a) for a in anchor_sets), default=1)))
        n_chain = max(1, min(self.max_chains, max((len(c) for c in chains_per_read), default=1)))
        n_features = self.n_seed_features

        # Per-read supervision assembly is independent across reads and dominated
        # by NumPy work that releases the GIL, so it is fanned across the host
        # threads that feed the GPU (the same pool that seeds/chains). This is the
        # last serial CPU stage in the supervision builder the module docstring
        # flags as GPU-starving; threading it keeps the accelerator fed. The
        # thread budget mirrors the pipeline's stage workers so the pools stay
        # consistent, and ``parallel_map`` degrades to a serial pass for tiny
        # batches, so correctness never depends on the thread count.
        rows = parallel_map(
            lambda item: self._build_row(
                item[0], item[1], item[2], n_anchor, n_chain, n_features
            ),
            list(zip(reads, anchor_sets, chains_per_read)),
            workers=getattr(self.pipeline, "_stage_workers", None),
        )

        feat_rows = [r[0] for r in rows]
        pos_rows = [r[1] for r in rows]
        node_rows = [r[2] for r in rows]
        amask_rows = [r[3] for r in rows]
        label_rows = [r[4] for r in rows]
        chain_feat = np.stack([r[5] for r in rows]) if rows else np.zeros(
            (0, n_chain, self.n_chain_features), dtype=np.float32
        )
        chain_mask = np.stack([r[6] for r in rows]) if rows else np.zeros(
            (0, n_chain), dtype=bool
        )
        chain_target = np.array([r[7] for r in rows], dtype=np.int64) if rows else np.zeros(
            0, dtype=np.int64
        )
        member_pos = np.stack([r[8] for r in rows]) if rows else np.zeros(
            (0, n_chain, self.max_members), dtype=np.int64
        )
        member_mask = np.stack([r[9] for r in rows]) if rows else np.zeros(
            (0, n_chain, self.max_members), dtype=bool
        )

        features = np.stack(feat_rows)
        chain_feat = np.nan_to_num(chain_feat, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(features).all():
            # A non-finite feature poisons the loss to NaN on the first step and
            # every step after, so it is scrubbed here rather than debugged later.
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        seed_edge_index = seed_edge_features = seed_edge_mask = seed_gnn_active = None
        scorer = getattr(self.pipeline, "scorer", None)
        if scorer is not None and scorer.model.cfg.seed_scoring.use_anchor_gnn:
            truncated = [
                anchors.take(np.arange(min(len(anchors), n_anchor), dtype=np.int64))
                for anchors in anchor_sets
            ]
            (
                seed_edge_index,
                seed_edge_features,
                seed_edge_mask,
                seed_gnn_active,
            ) = scorer._pad_seed_graph(truncated, torch.device("cpu"))

        targets: dict[str, torch.Tensor] = {
            "seed_labels": torch.from_numpy(np.stack(label_rows)),
            "anchor_mask": torch.from_numpy(np.stack(amask_rows)),
            "chain_target": torch.from_numpy(chain_target),
            "chain_mask": torch.from_numpy(chain_mask),
            # MAPQ from the generator's own confidence label.
            "mapq_target": torch.tensor([float(r.mapq) for r in reads]),
            "mapq_valid": torch.ones(n, dtype=torch.bool),
            # Per-read heuristic difficulty -> per-read loss weight (see the loss).
            "read_difficulty": torch.from_numpy(read_difficulty),
        }

        # Within-window offset relative to the locus the chainer already found.
        # A *local* residual is learnable from the fused embedding; the absolute
        # ``ref_start / len(reference)`` coordinate it replaced was not (the head
        # degenerated to predicting the batch mean). Computed from the same
        # ``chains_per_read`` used above, so no extra stage runs.
        pos_target, pos_valid = self._local_position_target(
            reads, chains_per_read, self.position_window
        )
        targets["position_target"] = torch.from_numpy(pos_target)
        targets["position_valid"] = torch.from_numpy(pos_valid)
        if not self._position_warned and (
            pos_valid.sum() < 2 or float(pos_target[pos_valid].std() if pos_valid.any() else 0.0) < 1e-3
        ):
            self._position_warned = True
            print(
                "[targets] WARNING: local position_target has ~0 spread on this "
                f"batch; the position head has little to learn. Revisit "
                f"position_window (={self.position_window}) or confirm chain.ref_start "
                "differs from read.ref_start."
            )

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
            seed_edge_index=seed_edge_index,
            seed_edge_features=seed_edge_features,
            seed_edge_mask=seed_edge_mask,
            seed_gnn_active=seed_gnn_active,
            chain_feats=torch.from_numpy(chain_feat),
            chain_mask=torch.from_numpy(chain_mask),
            chain_member_pos=torch.from_numpy(member_pos),
            chain_member_mask=torch.from_numpy(member_mask),
            member_states_shape=(n, n_chain, self.max_members),
            supervised=supervised + (("transition",) if seed_edge_mask is not None else ()),
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
        # Mean live candidate chains per read. The listwise chain-ranking loss is
        # identically zero for any read with a single candidate (softmax of one
        # element is 1.0, zero gradient); decoy injection pushes this above 1 so
        # the term actually trains. Surfacing it here makes the chain fix visible
        # in history/plot 06 rather than something to infer from the loss curve.
        chain_mask = sup.targets.get("chain_mask")
        cand_per_read = (
            float(chain_mask.float().sum(dim=1).mean())
            if chain_mask is not None and chain_mask.numel()
            else 0.0
        )
        return {
            "anchor_positive_rate": pos / max(n_valid, 1),
            "anchors_per_read": n_valid / max(sup.n_reads, 1),
            "reads_with_chain_label": float((chain_t >= 0).float().mean()),
            "chain_candidates_per_read": cand_per_read,
        }
