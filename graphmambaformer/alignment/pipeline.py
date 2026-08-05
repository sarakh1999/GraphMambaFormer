"""The alignment pipeline and its selectable modes.

Three modes share the same stages and differ only in how much work they spend:

:class:`HybridAlignmentPipeline` (``"hybrid"``, the default)
    The full accuracy path: seed -> neural anchor pruning -> chain -> extend ->
    neural re-rank + MAPQ -> post-process.

:class:`FastAlignmentPipeline` (``"fast"``)
    Throughput path. Classical stages only, no re-ranking, MAPQ from the
    primary/secondary margin. This is the baseline the hybrid mode is measured
    against, and the fallback when the core model has no alignment heads.

:class:`TwoPassAligner` (``"two_pass"``)
    Runs the fast path first, then re-aligns only the reads it could not resolve
    confidently through the hybrid path. Most reads are easy, so this buys most of
    the hybrid accuracy for close to the fast cost.

All three take a :class:`ReferenceIndex` built once per reference and reused
across batches, since index construction dominates single-batch cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..accel import AccelContext
from ..config import PIPELINE_MODES, PipelineConfig
from .chaining import AffineChainer, ChainingContext, GraphDistanceOracle
from .extension import ExtensionEngine
from .scoring import NeuralScorer
from .seeding import SeedIndexBundle, SeedingEngine
from .types import AlignmentRecord, AnchorSet, Chain, ReadAlignments

__all__ = [
    "AlignmentPipeline",
    "FastAlignmentPipeline",
    "HybridAlignmentPipeline",
    "PipelineStats",
    "ReferenceIndex",
    "TwoPassAligner",
    "build_pipeline",
]


@dataclass
class ReferenceIndex:
    """Everything the stages need about one reference, built once.

    Holds the Stage 1 indices, the linear reference used by Stage 3, and the
    optional graph context (hop oracle + backbone flags) that makes Stage 2
    graph-aware.
    """

    bundle: SeedIndexBundle
    ref_seq: str
    ref_id: int = 0
    oracle: Optional[GraphDistanceOracle] = None
    backbone: Optional[np.ndarray] = None
    #: Encoded graph for the core model, shared across reads.
    graph: object | None = None

    @property
    def chaining_context(self) -> ChainingContext:
        return ChainingContext(oracle=self.oracle, backbone=self.backbone)


@dataclass
class PipelineStats:
    """Counters for one pipeline call, for cost accounting across modes."""

    n_reads: int = 0
    n_mapped: int = 0
    n_anchors: int = 0
    n_anchors_pruned: int = 0
    n_chains: int = 0
    n_extended: int = 0
    n_neural_batches: int = 0
    n_rescued: int = 0
    #: Reads handled by each pass (``two_pass`` splits into fast / hybrid).
    per_pass: dict = field(default_factory=dict)

    def merge(self, other: "PipelineStats") -> None:
        for name in (
            "n_reads",
            "n_mapped",
            "n_anchors",
            "n_anchors_pruned",
            "n_chains",
            "n_extended",
            "n_neural_batches",
            "n_rescued",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key, value in other.per_pass.items():
            self.per_pass[key] = self.per_pass.get(key, 0) + value

    def summary(self) -> str:
        passes = ", ".join(f"{k}={v}" for k, v in sorted(self.per_pass.items()))
        return (
            f"reads={self.n_reads} mapped={self.n_mapped} "
            f"anchors={self.n_anchors} (pruned {self.n_anchors_pruned}) "
            f"chains={self.n_chains} extended={self.n_extended} "
            f"neural_batches={self.n_neural_batches} rescued={self.n_rescued}"
            + (f" | passes: {passes}" if passes else "")
        )


class AlignmentPipeline:
    """Shared stage plumbing; subclasses decide which stages run.

    Subclasses override :meth:`align_batch`. The stage helpers here are written so
    a mode is a short composition of them rather than a copy of the whole flow.
    """

    #: Name reported in ``AlignmentRecord.pass_name``.
    mode: str = "base"

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        model=None,
        accel: AccelContext | None = None,
        device: torch.device | str | None = None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.accel = accel or AccelContext(self.cfg.accel, device=device)
        self.device = self.accel.caps.device

        self.seeder = SeedingEngine(self.cfg.seeding, device=self.device)
        self.chainer = AffineChainer(
            self.cfg.chaining, backend=self.accel.kernel_backend("chaining")
        )
        self.extender = ExtensionEngine(self.cfg.extension, device=self.device)

        self.model = model
        self.scorer: Optional[NeuralScorer] = None
        if model is not None and self._model_has_heads(model):
            self.scorer = NeuralScorer(
                model,
                self.cfg.scoring,
                device=self.device,
                amp_dtype=self.accel.autocast_dtype,
            )

    @staticmethod
    def _model_has_heads(model) -> bool:
        """True when the model exposes the Stage 4 heads.

        The MambaFormer / hybrid encoder baselines are sequence-only, so selecting
        one of those degrades every mode to classical scoring rather than failing.
        """
        return all(
            hasattr(model, name) for name in ("score_seeds", "score_chains", "mapping_head")
        )

    @property
    def uses_neural_scoring(self) -> bool:
        return self.scorer is not None and self.cfg.run_neural_scoring

    # ---- reference ---------------------------------------------------------- #
    def build_reference(
        self,
        ref_seq: str,
        ref_id: int = 0,
        node_seqs: Optional[Sequence[str]] = None,
        node_ref_start: Optional[Sequence[int]] = None,
        backbone_path: Optional[Sequence[int]] = None,
        edge_index: Optional[np.ndarray] = None,
        graph=None,
    ) -> ReferenceIndex:
        """Build the Stage 1 indices and graph context for one reference."""
        bundle = self.seeder.build_indices(
            ref_seq,
            ref_id=ref_id,
            node_seqs=node_seqs,
            node_ref_start=node_ref_start,
            backbone_path=backbone_path,
        )
        oracle = None
        if edge_index is not None and node_seqs is not None:
            oracle = GraphDistanceOracle(
                edge_index, len(node_seqs), max_hops=self.cfg.chaining.graph_max_hops
            )
        return ReferenceIndex(
            bundle=bundle,
            ref_seq=ref_seq,
            ref_id=ref_id,
            oracle=oracle,
            backbone=bundle.backbone,
            graph=graph,
        )

    # ---- stages ------------------------------------------------------------- #
    def seed(self, reads: Sequence[str], reference: ReferenceIndex) -> list[AnchorSet]:
        return [self.seeder.seed_read(read, reference.bundle) for read in reads]

    def chain(
        self, anchor_sets: Sequence[AnchorSet], reference: ReferenceIndex
    ) -> list[list[Chain]]:
        ctx = reference.chaining_context
        return self.chainer.chain_batch(
            anchor_sets, [ctx] * len(anchor_sets), device=self.device
        )

    def extend(
        self,
        reads: Sequence[str],
        chains_per_read: Sequence[Sequence[Chain]],
        anchor_sets: Sequence[AnchorSet],
        reference: ReferenceIndex,
    ) -> list[list]:
        """Extend every read's chains, or return empty lists when Stage 3 is off."""
        if not self.cfg.run_extension:
            return [[] for _ in reads]
        return [
            self.extender.extend_chains(read, chains, anchors, reference.ref_seq)
            for read, chains, anchors in zip(reads, chains_per_read, anchor_sets)
        ]

    # ---- record assembly ---------------------------------------------------- #
    def _records(
        self,
        read_id: str,
        read_len: int,
        anchors: AnchorSet,
        chains: Sequence[Chain],
        extensions: Sequence,
        reference: ReferenceIndex,
        mapq: int,
        route: str = "full",
    ) -> list[AlignmentRecord]:
        """Turn one read's chains (+ extensions) into output records."""
        stages = ("seed", "chain") + (("extend",) if extensions else ())
        if self.uses_neural_scoring:
            stages = stages + ("score",)

        records: list[AlignmentRecord] = []
        for rank, chain in enumerate(chains):
            extension = extensions[rank] if rank < len(extensions) else None
            ref_start = extension.ref_start if extension else chain.ref_start
            ref_end = extension.ref_end if extension else chain.ref_end
            node = reference.bundle.node_of(np.asarray([ref_start], dtype=np.int64))
            records.append(
                AlignmentRecord(
                    read_id=read_id,
                    read_len=read_len,
                    ref_id=reference.ref_id,
                    ref_start=int(ref_start),
                    ref_end=int(ref_end),
                    strand=int(chain.strand),
                    # Only the primary carries the read's MAPQ; a secondary
                    # alignment of an ambiguous locus is by definition uncertain.
                    mapq=int(mapq) if chain.is_primary else 0,
                    chain_score=float(chain.score),
                    alignment_score=float(extension.score) if extension else 0.0,
                    cigar=list(extension.cigar) if extension else [],
                    node_id=int(node[0]),
                    is_mapped=True,
                    is_primary=chain.is_primary,
                    is_supplementary=not chain.is_primary,
                    route=route,
                    pass_name=self.mode,
                    n_anchors=len(anchors),
                    n_chains=len(chains),
                    stages=stages,
                    extension=extension,
                )
            )
        return records

    def _margin_mapq(self, chains: Sequence[Chain]) -> int:
        """Classical MAPQ from the primary/secondary score margin."""
        if not chains:
            return 0
        ordered = sorted((c.score for c in chains), reverse=True)
        best = ordered[0]
        if best <= 0:
            return 0
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        margin = float(np.clip((best - runner_up) / best, 0.0, 1.0))
        return int(np.clip(round(margin * self.cfg.scoring.max_mapq), 0, self.cfg.scoring.max_mapq))

    def align_batch(
        self,
        reads: Sequence[str],
        reference: ReferenceIndex,
        read_ids: Optional[Sequence[str]] = None,
    ) -> tuple[list[ReadAlignments], PipelineStats]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- public entry point ------------------------------------------------- #
    def align(
        self,
        reads: Sequence[str],
        reference: ReferenceIndex,
        read_ids: Optional[Sequence[str]] = None,
    ) -> tuple[list[ReadAlignments], PipelineStats]:
        """Align reads in batches of ``cfg.batch_size``."""
        ids = list(read_ids) if read_ids is not None else [f"read{i}" for i in range(len(reads))]
        results: list[ReadAlignments] = []
        stats = PipelineStats()
        for start in range(0, len(reads), self.cfg.batch_size):
            chunk = list(reads[start : start + self.cfg.batch_size])
            chunk_ids = ids[start : start + self.cfg.batch_size]
            batch_results, batch_stats = self.align_batch(chunk, reference, chunk_ids)
            results.extend(batch_results)
            stats.merge(batch_stats)
        return results, stats


class FastAlignmentPipeline(AlignmentPipeline):
    """Classical throughput path: seed -> chain -> extend, margin-based MAPQ."""

    mode = "fast"

    def align_batch(self, reads, reference, read_ids=None):
        ids = list(read_ids) if read_ids is not None else [f"read{i}" for i in range(len(reads))]
        stats = PipelineStats(n_reads=len(reads))

        anchor_sets = self.seed(reads, reference)
        stats.n_anchors = sum(len(a) for a in anchor_sets)
        chains_per_read = self.chain(anchor_sets, reference)
        stats.n_chains = sum(len(c) for c in chains_per_read)
        extensions = self.extend(reads, chains_per_read, anchor_sets, reference)
        stats.n_extended = sum(len(e) for e in extensions)

        results = []
        for row, read in enumerate(reads):
            chains = chains_per_read[row]
            records = self._records(
                ids[row],
                len(read),
                anchor_sets[row],
                chains,
                extensions[row],
                reference,
                mapq=self._margin_mapq(chains),
                route="fast",
            )
            stats.n_mapped += bool(records)
            results.append(
                ReadAlignments(
                    read_id=ids[row],
                    read_len=len(read),
                    anchors=anchor_sets[row],
                    chains=list(chains),
                    records=records
                    or [AlignmentRecord.unmapped(ids[row], len(read), pass_name=self.mode)],
                )
            )
        stats.per_pass["fast"] = len(reads)
        return results, stats


class HybridAlignmentPipeline(AlignmentPipeline):
    """The full accuracy path, with the neural stage woven through Stages 1-4.

    Ordering matters. Anchor scoring happens *before* chaining so pruning shrinks
    the DP input and the surviving scores bias the anchor weights; chain
    re-ranking happens *after* the DP so the head sees complete chains; MAPQ comes
    last, once the primary/secondary margin is known.

    With a model that has no alignment heads (the encoder baselines) this degrades
    to the classical path rather than failing, so mode and architecture can be
    varied independently.
    """

    mode = "hybrid"

    def align_batch(self, reads, reference, read_ids=None):
        ids = list(read_ids) if read_ids is not None else [f"read{i}" for i in range(len(reads))]
        stats = PipelineStats(n_reads=len(reads))

        # Stage 1.
        anchor_sets = self.seed(reads, reference)
        stats.n_anchors = sum(len(a) for a in anchor_sets)

        if not self.uses_neural_scoring:
            fallback = FastAlignmentPipeline.align_batch(self, reads, reference, ids)
            for read_alignments in fallback[0]:
                for record in read_alignments.records:
                    record.pass_name = self.mode
                    record.route = "classical"
            fallback[1].per_pass = {"hybrid_classical": len(reads)}
            return fallback

        scorer = self.scorer
        assert scorer is not None

        # Stage 4a — one forward pass, reused for anchors, chains, and MAPQ.
        base_codes, mask = _encode(reads, self.device, self.cfg.max_read_len)
        with torch.inference_mode(), self.accel.autocast():
            outputs = self.model(base_codes, mask=mask, graph=reference.graph)
        stats.n_neural_batches = 1

        scorer.score_anchors(outputs, anchor_sets)
        pruned = [scorer.prune_anchors(a) for a in anchor_sets]
        stats.n_anchors_pruned = stats.n_anchors - sum(len(a) for a in pruned)

        # Stage 2 over the surviving anchors.
        chains_per_read = self.chain(pruned, reference)
        stats.n_chains = sum(len(c) for c in chains_per_read)

        # Stage 4b — re-rank, then Stage 3 extends the chains in final order.
        scorer.score_chains(
            outputs, chains_per_read, pruned, [len(r) for r in reads], reference.backbone
        )
        chains_per_read = [scorer.rerank(list(chains)) for chains in chains_per_read]

        extensions = self.extend(reads, chains_per_read, pruned, reference)
        stats.n_extended = sum(len(e) for e in extensions)

        # Stage 4c — MAPQ, and a proposed locus for reads placed nowhere.
        mapqs = scorer.mapq(outputs, chains_per_read)
        routes = _route_names(self.model, outputs)

        results = []
        for row, read in enumerate(reads):
            chains = chains_per_read[row]
            records = self._records(
                ids[row],
                len(read),
                pruned[row],
                chains,
                extensions[row],
                reference,
                mapq=int(mapqs[row]),
                route=routes[row],
            )
            if not records and self.cfg.scoring.position_rescue:
                rescued = self._rescue(ids[row], read, outputs, row, reference, routes[row])
                if rescued is not None:
                    records = [rescued]
                    stats.n_rescued += 1
            stats.n_mapped += bool(records)
            results.append(
                ReadAlignments(
                    read_id=ids[row],
                    read_len=len(read),
                    anchors=pruned[row],
                    chains=list(chains),
                    records=records
                    or [AlignmentRecord.unmapped(ids[row], len(read), pass_name=self.mode)],
                )
            )
        stats.per_pass["hybrid"] = len(reads)
        return results, stats

    def _rescue(
        self,
        read_id: str,
        read: str,
        outputs,
        row: int,
        reference: ReferenceIndex,
        route: str,
    ) -> Optional[AlignmentRecord]:
        """Propose a locus from the mapping head for an unplaced read.

        Seeding fails outright on reads whose every k-mer is repetitive or
        error-ridden. The head still predicts a node and an offset within it, so
        the read is reported at low MAPQ instead of being dropped — a candidate
        locus a caller can revisit, explicitly marked as unextended.
        """
        mapping = getattr(outputs, "mapping", None)
        if mapping is None:
            return None

        node = int(mapping["node_id"][row].item())
        starts = reference.bundle.node_starts
        node_ids = reference.bundle.node_ids
        if starts is None or len(starts) == 0:
            return None

        slot = int(np.flatnonzero(np.asarray(node_ids) == node)[0]) if node_ids is not None else node
        if not 0 <= slot < len(starts):
            return None

        offset = int(mapping["position"][row].item())
        ref_start = int(np.clip(starts[slot] + offset, 0, max(len(reference.ref_seq) - 1, 0)))
        return AlignmentRecord(
            read_id=read_id,
            read_len=len(read),
            ref_id=reference.ref_id,
            ref_start=ref_start,
            ref_end=min(ref_start + len(read), len(reference.ref_seq)),
            strand=1,
            mapq=self.cfg.scoring.mapq_floor,
            node_id=node,
            is_mapped=True,
            is_primary=True,
            route=route,
            pass_name=self.mode,
            stages=("seed", "score", "rescue"),
        )


class TwoPassAligner(AlignmentPipeline):
    """Fast path for the easy majority, hybrid rescue for the hard tail.

    A read is "easy" when its best chain covers enough of the read and clearly
    beats the runner-up; those keep the fast result. Everything else is re-aligned
    through the hybrid path, so the neural cost is paid only where it changes the
    answer.
    """

    mode = "two_pass"

    def __init__(self, cfg=None, model=None, accel=None, device=None):
        super().__init__(cfg, model=model, accel=accel, device=device)
        self.fast = FastAlignmentPipeline(self.cfg, model=None, accel=self.accel)
        self.hybrid = HybridAlignmentPipeline(
            self.cfg, model=model, accel=self.accel
        )
        # Share the built engines so the two passes do not duplicate state.
        for stage in (self.fast, self.hybrid):
            stage.seeder = self.seeder
            stage.chainer = self.chainer
            stage.extender = self.extender

    def _is_easy(self, alignments: ReadAlignments) -> bool:
        chains = alignments.chains
        if not chains:
            return False  # nothing found: exactly the case the hybrid pass is for
        best = chains[0]
        if best.coverage(alignments.read_len) < self.cfg.easy_coverage:
            return False
        if len(chains) == 1:
            return True
        margin = (best.score - chains[1].score) / max(best.score, 1e-6)
        return margin >= self.cfg.easy_margin

    def align_batch(self, reads, reference, read_ids=None):
        ids = list(read_ids) if read_ids is not None else [f"read{i}" for i in range(len(reads))]

        first, stats = self.fast.align_batch(reads, reference, ids)
        stats.per_pass = {"fast": len(reads)}

        hard = [i for i, alignments in enumerate(first) if not self._is_easy(alignments)]
        if not hard or not self.hybrid.uses_neural_scoring:
            for alignments in first:
                for record in alignments.records:
                    record.pass_name = self.mode
            return first, stats

        second, hard_stats = self.hybrid.align_batch(
            [reads[i] for i in hard], reference, [ids[i] for i in hard]
        )
        # The fast pass already counted these reads; keep only the extra work.
        hard_stats.n_reads = 0
        hard_stats.n_mapped -= sum(bool(first[i].records[0].is_mapped) for i in hard)
        hard_stats.per_pass = {"hybrid_rescue": len(hard)}
        stats.merge(hard_stats)

        for slot, index in enumerate(hard):
            first[index] = second[slot]
        for alignments in first:
            for record in alignments.records:
                record.pass_name = self.mode
        return first, stats


def _encode(reads, device, max_len):
    from .scoring import encode_read_batch

    return encode_read_batch(reads, device=device, max_len=max_len)


def _route_names(model, outputs) -> list[str]:
    """Per-read route label, defaulting to ``"full"`` when there is no router.

    The names come from the router itself rather than a local copy, so the
    pipeline cannot drift from ``RouterConfig.route_names``.
    """
    router = getattr(outputs, "router", None)
    if router is None:
        return ["full"] * len(outputs.read_hidden)
    return model.router.route_names(router["route"])


#: Pipeline mode -> implementation. ``"hybrid"`` is the default.
PIPELINE_REGISTRY: dict[str, type[AlignmentPipeline]] = {
    "hybrid": HybridAlignmentPipeline,
    "fast": FastAlignmentPipeline,
    "two_pass": TwoPassAligner,
}


def build_pipeline(
    cfg: PipelineConfig | str | None = None,
    model=None,
    accel: AccelContext | None = None,
    device: torch.device | str | None = None,
) -> AlignmentPipeline:
    """Build the pipeline named by ``cfg`` (a config, a mode name, or ``None``).

    ``None`` gives the default hybrid pipeline.
    """
    if cfg is None:
        cfg = PipelineConfig()
    elif isinstance(cfg, str):
        cfg = PipelineConfig(mode=cfg)

    if cfg.mode not in PIPELINE_REGISTRY:  # pragma: no cover - PipelineConfig validates
        raise ValueError(
            f"Unknown pipeline mode {cfg.mode!r}. Known: {sorted(PIPELINE_REGISTRY)}"
        )
    return PIPELINE_REGISTRY[cfg.mode](cfg, model=model, accel=accel, device=device)


assert set(PIPELINE_REGISTRY) == set(PIPELINE_MODES), (
    "PIPELINE_REGISTRY and config.PIPELINE_MODES have drifted apart"
)
