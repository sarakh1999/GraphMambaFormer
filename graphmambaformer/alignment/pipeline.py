"""The alignment pipeline and its selectable modes.

Three modes share the same stages and differ only in how much work they spend:

:class:`HybridAlignmentPipeline` (``"hybrid"``, the default)
    The full accuracy path: seed -> seed-graph GNN -> adaptive chain -> extend ->
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

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from ..accel import AccelContext, default_worker_count, parallel_map
from ..config import PIPELINE_MODES, PipelineConfig
from ..progress import progress
from .chaining import (
    AffineChainer,
    ChainingContext,
    GraphDistanceOracle,
    pack_node_haplotypes,
)
from .extension import ExtensionEngine
from .scoring import NeuralScorer
from .seeding import SeedIndexBundle, SeedingEngine
from .types import AlignmentRecord, AnchorSet, Chain, ReadAlignments

__all__ = [
    "AlignmentPipeline",
    "FastAlignmentPipeline",
    "HybridAlignmentPipeline",
    "PipelineStats",
    "ReadBatch",
    "ReferenceIndex",
    "TwoPassAligner",
    "build_pipeline",
]


@dataclass
class ReadBatch:
    """Reads plus the per-read metadata the encoder can condition on.

    The pipeline accepts either bare sequence strings or the ``ReadRecord``
    objects the format readers produce. Records carry Phred qualities and a
    modality, which feed the encoder's quality embedding and modality token;
    plain strings carry neither, and the encoder falls back to its defaults.
    """

    seqs: list[str]
    ids: list[str]
    quals: Optional[list[list[int]]] = None
    modalities: Optional[list[str]] = None

    def __len__(self) -> int:
        return len(self.seqs)

    def slice(self, start: int, stop: int) -> "ReadBatch":
        return self.select(range(*slice(start, stop).indices(len(self.seqs))))

    def select(self, rows: Iterable[int]) -> "ReadBatch":
        """A sub-batch keeping per-read metadata aligned with the chosen rows."""
        rows = list(rows)
        return ReadBatch(
            seqs=[self.seqs[i] for i in rows],
            ids=[self.ids[i] for i in rows],
            quals=None if self.quals is None else [self.quals[i] for i in rows],
            modalities=(
                None if self.modalities is None else [self.modalities[i] for i in rows]
            ),
        )

    @property
    def modality(self) -> Optional[list[str] | str]:
        """A single modality when the batch is homogeneous, else one per read."""
        if not self.modalities:
            return None
        unique = set(self.modalities)
        return self.modalities[0] if len(unique) == 1 else self.modalities


def as_read_batch(reads, read_ids: Optional[Sequence[str]] = None) -> ReadBatch:
    """Normalize ``Sequence[str] | Sequence[ReadRecord]`` into a :class:`ReadBatch`.

    Detection is duck-typed on ``.seq`` so this does not import the data layer
    (which would be a circular import). Passing an existing :class:`ReadBatch`
    through is a no-op, so the public entry points can normalize defensively.
    """
    if isinstance(reads, ReadBatch):
        return reads
    reads = list(reads)
    if reads and not isinstance(reads[0], str) and hasattr(reads[0], "seq"):
        seqs = [r.seq for r in reads]
        quals = [list(getattr(r, "quals", []) or []) for r in reads]
        modalities = [getattr(r, "modality", None) for r in reads]
        ids = (
            list(read_ids)
            if read_ids is not None
            else [getattr(r, "read_id", f"read{i}") for i, r in enumerate(reads)]
        )
        return ReadBatch(
            seqs=seqs,
            ids=ids,
            quals=quals if any(quals) else None,
            modalities=modalities if all(m for m in modalities) else None,
        )

    seqs = [str(r) for r in reads]
    ids = list(read_ids) if read_ids is not None else [f"read{i}" for i in range(len(seqs))]
    return ReadBatch(seqs=seqs, ids=ids)


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
    #: ``(num_nodes, W)`` uint64 haplotype bitset for haplotype-aware chaining
    #: (packed from the graph's P-/W-line paths); ``None`` disables the term.
    node_haplotypes: Optional[np.ndarray] = None

    @property
    def chaining_context(self) -> ChainingContext:
        return ChainingContext(
            oracle=self.oracle,
            backbone=self.backbone,
            node_haplotypes=self.node_haplotypes,
        )


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

        # Host-side thread budget for the CPU-bound per-read loops (Stage 1/3).
        # Threads (not processes) so the FM-index / reference bundle are shared
        # rather than pickled; the NumPy index queries release the GIL.
        acfg = self.accel.cfg
        self._stage_workers = (
            default_worker_count(acfg.num_workers)
            if getattr(acfg, "stage_parallel", True)
            else 1
        )

        # GenomeWorks routing (AccelConfig). The master switch propagates to the
        # extension engine (its kill switch), and routing the seeding stage to
        # GenomeWorks selects the cudamapper minimizer index. Keep these here so
        # ``genomeworks`` / ``stage_backends`` stay effective end to end.
        _stage_backends = getattr(acfg, "stage_backends", {}) or {}
        _gw_master = bool(getattr(acfg, "genomeworks", True))
        seeding_cfg = self.cfg.seeding
        if _gw_master and _stage_backends.get("seeding") == "genomeworks":
            # Route seeding through the cudamapper GPU minimizer index, but keep
            # the complementary modes (SMEM, fuzzy, …) so the GPU route stays as
            # sensitive as the default seeder: cudamapper *replaces the minimizer
            # / gpu_kmer mode*, it does not replace the whole multi-mode seeder.
            # Minimizer-only seeding otherwise leaves ~5% of hard / reverse-
            # complement reads unmapped (see tests/gpu_pipeline_e2e.py).
            kept = tuple(
                m for m in seeding_cfg.modes
                if m not in ("minimizer", "gpu_kmer", "cudamapper")
            )
            seeding_cfg = replace(seeding_cfg, modes=("cudamapper",) + kept)

        self.seeder = SeedingEngine(
            seeding_cfg, device=self.device, workers=self._stage_workers
        )
        self.chainer = AffineChainer(
            self.cfg.chaining,
            backend=self.accel.kernel_backend("chaining"),
            workers=self._stage_workers,
        )
        self.extender = ExtensionEngine(
            self.cfg.extension,
            device=self.device,
            backend=self.accel.kernel_backend("extension"),
            genomeworks=_gw_master,
        )

        # The pipeline owns the compute device, so the model must live on it too;
        # otherwise inputs encoded on ``self.device`` mismatch a model left on CPU.
        if model is not None and hasattr(model, "to"):
            model = model.to(self.device)
        self.model = model
        self.scorer: Optional[NeuralScorer] = None
        # Inference wrappers: TensorRT (optional) then CUDA Graphs around the
        # resulting callable. Training still uses ``self.model`` directly.
        self._infer_model = model
        if model is not None:
            wrapped = self.accel.wrap_tensorrt(model)
            self._infer_model = self.accel.wrap_cuda_graphs(wrapped)
        if model is not None and self._model_has_heads(model):
            self.scorer = NeuralScorer(
                model,
                self.cfg.scoring,
                device=self.device,
                amp_dtype=self.accel.autocast_dtype,
                infer_fn=self._infer_model,
                precision_ctx=self.accel.precision,
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
        haplotype_paths: Optional[Sequence[Sequence[int]]] = None,
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
        node_haplotypes = None
        if edge_index is not None and node_seqs is not None:
            oracle = GraphDistanceOracle(
                edge_index, len(node_seqs), max_hops=self.cfg.chaining.graph_max_hops
            )
        if node_seqs is not None:
            node_haplotypes = pack_node_haplotypes(len(node_seqs), haplotype_paths)
        return ReferenceIndex(
            bundle=bundle,
            ref_seq=ref_seq,
            ref_id=ref_id,
            oracle=oracle,
            backbone=bundle.backbone,
            graph=graph,
            node_haplotypes=node_haplotypes,
        )

    # ---- stages ------------------------------------------------------------- #
    def seed(self, reads: Sequence[str], reference: ReferenceIndex) -> list[AnchorSet]:
        """Seed every read against the reference bundle, one thread per read.

        Seeding is independent per read and dominated by NumPy index queries
        (which release the GIL), so fanning it across the host cores is a clean
        throughput win and keeps the GPU from waiting on Stage 1.
        """
        return parallel_map(
            lambda read: self.seeder.seed_read(read, reference.bundle),
            reads,
            workers=self._stage_workers,
            pbar="seed" if len(reads) >= 32 else None,
        )

    def chain(
        self, anchor_sets: Sequence[AnchorSet], reference: ReferenceIndex,
        trust_neural: Optional[Sequence[Optional[bool]]] = None,
        learned_transitions: Optional[Sequence[Optional[dict]]] = None,
    ) -> list[list[Chain]]:
        """Chain each read, optionally with a per-read AGNES trust decision.

        ``trust_neural[i]`` is the Algorithm-1 decision for read ``i`` (True /
        False / None), computed on the *full* scored anchor set before pruning.
        """
        base = reference.chaining_context
        if trust_neural is None and learned_transitions is None:
            contexts: list[Optional[ChainingContext]] = [base] * len(anchor_sets)
        else:
            flags = list(trust_neural or [None] * len(anchor_sets))
            transitions = list(learned_transitions or [None] * len(anchor_sets))
            contexts = [
                ChainingContext(
                    oracle=base.oracle,
                    backbone=base.backbone,
                    node_haplotypes=base.node_haplotypes,
                    trust_neural=flag,
                    learned_transitions=edge_scores,
                )
                for flag, edge_scores in zip(flags, transitions)
            ]
        return self.chainer.chain_batch(anchor_sets, contexts, device=self.device)

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
        work = list(zip(reads, chains_per_read, anchor_sets))
        # On CUDA the batched banded-SW already runs on the device; threading the
        # host loop there only helps the Python glue, so keep it to the CPU path
        # where the DP itself runs on the host.
        workers = self._stage_workers if self.device.type == "cpu" else 1
        return parallel_map(
            lambda item: self.extender.extend_chains(
                item[0], item[1], item[2], reference.ref_seq
            ),
            work,
            workers=workers,
            pbar="extend" if len(work) >= 32 else None,
        )

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
        encoded: Optional[tuple] = None,
    ) -> tuple[list[ReadAlignments], PipelineStats]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- public entry point ------------------------------------------------- #
    def align(
        self,
        reads: Sequence[str] | Sequence[object],
        reference: ReferenceIndex,
        read_ids: Optional[Sequence[str]] = None,
    ) -> tuple[list[ReadAlignments], PipelineStats]:
        """Align reads in batches of ``cfg.batch_size``.

        ``reads`` may be plain sequence strings or the ``ReadRecord`` objects
        returned by the format readers, in which case their Phred qualities and
        modality are carried through to the model.
        """
        batch = as_read_batch(reads, read_ids)
        results: list[ReadAlignments] = []
        stats = PipelineStats()
        steps = range(0, len(batch), self.cfg.batch_size)
        for start in progress(
            steps,
            desc=f"align[{self.mode}]",
            unit="batch",
            leave=False,
        ):
            chunk = batch.slice(start, start + self.cfg.batch_size)
            batch_results, batch_stats = self.align_batch(chunk, reference, chunk.ids)
            results.extend(batch_results)
            stats.merge(batch_stats)
        return results, stats


class FastAlignmentPipeline(AlignmentPipeline):
    """Classical throughput path: seed -> chain -> extend, margin-based MAPQ."""

    mode = "fast"

    def align_batch(self, reads, reference, read_ids=None, encoded=None):
        # ``encoded`` (a precomputed read encoding) is accepted for a uniform
        # signature with the neural modes but ignored: the fast path never runs
        # the core model, so there is nothing to reuse.
        batch = as_read_batch(reads, read_ids)
        reads, ids = batch.seqs, batch.ids
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

    Ordering matters. Seed-graph scoring happens *before* chaining so the AGNES
    confidence decision can reweight the DP; chain re-ranking happens *after*
    the DP so the head sees complete chains; MAPQ comes last, once the
    primary/secondary margin is known.

    With adaptive seed scoring (default), trust is decided on the full scored
    graph: confident reads use GNN node/edge guidance, while under-confident or
    degenerate graphs run pure geometric DP over the unchanged candidate set.
    Pruning remains available only for the non-adaptive ablation path.

    With a model that has no alignment heads (the encoder baselines) this degrades
    to the classical path rather than failing, so mode and architecture can be
    varied independently.
    """

    mode = "hybrid"

    def _prepare_anchors_agnes(
        self, anchor_sets: Sequence[AnchorSet], seed_head: dict
    ) -> tuple[list[AnchorSet], list[Optional[bool]]]:
        """Apply AGNES Algorithm 1 without changing the PureDP candidate graph."""
        scorer = self.scorer
        assert scorer is not None
        adaptive = self.cfg.chaining.adaptive_seed_scoring
        gnn_active = seed_head.get("gnn_active")
        active_rows = (
            gnn_active.detach().cpu().numpy().astype(bool)
            if isinstance(gnn_active, torch.Tensor)
            else None
        )
        prepared: list[AnchorSet] = []
        trust_flags: list[Optional[bool]] = []
        for row, anchors in enumerate(anchor_sets):
            if adaptive and np.isfinite(anchors.score).any():
                trust = self.chainer.trust_neural_scores(anchors)
                if active_rows is not None:
                    trust = trust and bool(active_rows[row])
                trust_flags.append(trust)
                prepared.append(anchors)
            else:
                trust_flags.append(None)
                prepared.append(scorer.prune_anchors(anchors))
        return prepared, trust_flags

    def align_batch(self, reads, reference, read_ids=None, encoded=None):
        batch = as_read_batch(reads, read_ids)
        reads, ids = batch.seqs, batch.ids
        stats = PipelineStats(n_reads=len(reads))

        # Stage 1.
        anchor_sets = self.seed(reads, reference)
        stats.n_anchors = sum(len(a) for a in anchor_sets)

        if not self.uses_neural_scoring:
            fallback = FastAlignmentPipeline.align_batch(self, batch, reference, ids)
            for read_alignments in fallback[0]:
                for record in read_alignments.records:
                    record.pass_name = self.mode
                    record.route = "classical"
            fallback[1].per_pass = {"hybrid_classical": len(reads)}
            return fallback

        scorer = self.scorer
        assert scorer is not None

        # Stage 4a — one forward pass, reused for anchors, chains, and MAPQ.
        # Qualities and modality ride along when the reads came from a file that
        # carries them (FASTQ/BAM/uBAM/CRAM); they drive the encoder's quality
        # embedding and modality token.
        #
        # ``encoded`` lets a caller aligning the same reads against several
        # references (e.g. linear + pangenome) build this read tensor once and
        # reuse it here; only the graph-dependent forward pass below must rerun.
        if encoded is not None:
            base_codes, mask, qual_tensor = encoded
        else:
            base_codes, mask, qual_tensor = _encode(
                reads, self.device, self.cfg.max_read_len, quals=batch.quals
            )
        # CUDA Graphs + TensorRT wrap ``_infer_model``; TE FP8 / AMP via precision().
        infer = getattr(self, "_infer_model", self.model)
        with torch.inference_mode(), self.accel.precision():
            outputs = infer(
                base_codes,
                mask=mask,
                graph=reference.graph,
                qualities=qual_tensor,
                modality=batch.modality,
            )
        stats.n_neural_batches = 1

        _, seed_head = scorer.score_anchors(outputs, anchor_sets)
        learned_transitions = seed_head.get("transition_guidance")

        # AGNES Algorithm 1: classify the full seed graph, then either guide DP
        # with logits or run PureDP over the unchanged candidate graph. It does
        # not prune the low-confidence branch. Graphs outside the |V| guards or
        # with |E|=0 are marked inactive by the seed-graph builder and must also
        # take PureDP, regardless of any fallback MLP score distribution.
        prepared, trust_flags = self._prepare_anchors_agnes(anchor_sets, seed_head)
        stats.n_anchors_pruned = stats.n_anchors - sum(len(a) for a in prepared)

        # Stage 2 over the prepared anchors, with the per-read trust decision.
        chains_per_read = self.chain(
            prepared,
            reference,
            trust_neural=trust_flags,
            learned_transitions=learned_transitions,
        )
        stats.n_chains = sum(len(c) for c in chains_per_read)

        # Stage 4b — re-rank, then Stage 3 extends the chains in final order.
        scorer.score_chains(
            outputs, chains_per_read, prepared, [len(r) for r in reads], reference.backbone
        )
        chains_per_read = [scorer.rerank(list(chains)) for chains in chains_per_read]

        extensions = self.extend(reads, chains_per_read, prepared, reference)
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
                prepared[row],
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
                    anchors=prepared[row],
                    chains=list(chains),
                    records=records
                    or [AlignmentRecord.unmapped(ids[row], len(read), pass_name=self.mode)],
                    signals=_signals_for_row(outputs, row),
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

        if node_ids is not None:
            matches = np.flatnonzero(np.asarray(node_ids) == node)
            if matches.size == 0:
                return None
            slot = int(matches[0])
        else:
            slot = node
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

    def align_batch(self, reads, reference, read_ids=None, encoded=None):
        # ``encoded`` is ignored here: the hybrid rescue only ever encodes the
        # hard-read *subset*, so a full-batch encoding would not line up. Reuse
        # across references is offered by the pure hybrid mode instead.
        batch = as_read_batch(reads, read_ids)
        ids = batch.ids

        first, stats = self.fast.align_batch(batch, reference, ids)
        stats.per_pass = {"fast": len(batch)}

        hard = [i for i, alignments in enumerate(first) if not self._is_easy(alignments)]
        if not hard or not self.hybrid.uses_neural_scoring:
            for alignments in first:
                for record in alignments.records:
                    record.pass_name = self.mode
            return first, stats

        # Subset via select() so the hard reads keep their qualities and modality.
        second, hard_stats = self.hybrid.align_batch(
            batch.select(hard), reference, [ids[i] for i in hard]
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


def _encode(reads, device, max_len, quals=None):
    from .scoring import encode_read_batch

    return encode_read_batch(reads, device=device, max_len=max_len, quals=quals)


def _route_names(model, outputs) -> list[str]:
    """Per-read route label, defaulting to ``"full"`` when there is no router.

    The names come from the router itself rather than a local copy, so the
    pipeline cannot drift from ``RouterConfig.route_names``.
    """
    router = getattr(outputs, "router", None)
    if router is None:
        return ["full"] * len(outputs.read_hidden)
    return model.router.route_names(router["route"])


def _signals_for_row(outputs, row: int) -> dict[str, np.ndarray]:
    """Detach one read's enabled multi-task outputs for Stages 5–7."""

    multitask = getattr(outputs, "multitask", None)
    if not multitask:
        return {}
    signals: dict[str, np.ndarray] = {}
    for name, value in multitask.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and row < value.shape[0]:
            signals[name] = value[row].float().detach().cpu().numpy()
    return signals


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
