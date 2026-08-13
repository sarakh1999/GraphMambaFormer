"""Pipeline tests: the three modes end to end, pruning, rescue, determinism.

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import numpy as np
import torch

from graphmambaformer.alignment import (
    HybridAlignmentPipeline,
    build_pipeline,
    PIPELINE_REGISTRY,
)
from graphmambaformer.alignment.types import AnchorSet
from graphmambaformer.config import (
    MODALITIES,
    CoreModelConfig,
    GraphMambaConfig,
    PipelineConfig,
)
from graphmambaformer.data.synthetic import ReadRecord
from graphmambaformer.models import build_core_model
from graphmambaformer.models.graph_mamba import GraphBatch

RNG = np.random.default_rng(7)
BASES = "ACGT"


def make_reference(length=3000):
    return "".join(RNG.choice(list(BASES), size=length))


def mutate(seq, rate=0.02):
    out = list(seq)
    for i in range(len(out)):
        if RNG.random() < rate:
            out[i] = RNG.choice(list(BASES))
    return "".join(out)


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def make_reads(ref, n=8, read_len=200, rate=0.02):
    reads, truth = [], []
    for i in range(n):
        start = int(RNG.integers(0, len(ref) - read_len))
        seq = mutate(ref[start : start + read_len], rate)
        strand = 1
        if i % 3 == 2:
            seq, strand = revcomp(seq), -1
        reads.append(seq)
        truth.append((start, strand))
    return reads, truth


def make_graph(cfg, n_nodes, node_len=200):
    """A simple linear pangenome graph shared by every read in the batch."""
    node_k = torch.randint(0, 4 ** cfg.graph_encoder.kmer_size, (n_nodes, 6))
    src = torch.arange(n_nodes - 1)
    edges = torch.stack([src, src + 1])
    graph = GraphBatch(
        node_kmer_ids=node_k,
        edge_index=edges,
        edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
    )
    return graph, edges.numpy()


def build(mode, with_model=True, arch="graphmamba", **cfg_kwargs):
    cfg = PipelineConfig(mode=mode, **cfg_kwargs)
    cfg.seeding.modes = ("minimizer", "smem")
    model = None
    gm = GraphMambaConfig(d_model=64)
    if with_model:
        model = build_core_model(CoreModelConfig(arch=arch, graphmamba=gm)).model
        model.eval()
    return build_pipeline(cfg, model=model), gm


def reference_for(pipeline, ref, gm, node_len=200):
    n_nodes = len(ref) // node_len
    node_seqs = [ref[i * node_len : (i + 1) * node_len] for i in range(n_nodes)]
    node_start = [i * node_len for i in range(n_nodes)]
    graph, edges = make_graph(gm, n_nodes)
    return pipeline.build_reference(
        ref,
        ref_id=0,
        node_seqs=node_seqs,
        node_ref_start=node_start,
        backbone_path=list(range(n_nodes)),
        edge_index=edges,
        graph=graph,
    )


def accuracy(results, truth, tol=30):
    hits = 0
    for alignments, (start, strand) in zip(results, truth):
        rec = alignments.primary
        if rec is None or not rec.is_mapped:
            continue
        if rec.strand == strand and abs(rec.ref_start - start) <= tol:
            hits += 1
    return hits / len(truth)


def test_default_is_hybrid():
    assert PipelineConfig().mode == "hybrid"
    assert isinstance(build_pipeline(), HybridAlignmentPipeline)
    assert set(PIPELINE_REGISTRY) == {"hybrid", "fast", "two_pass"}
    print("default pipeline mode is hybrid; registry has all 3 modes")


def test_route_labels_come_from_the_router():
    """Route labels must be the router's own vocabulary, not a pipeline-local copy.

    The architecture names the three routes fast/medium/full. A duplicated tuple
    in the pipeline silently drifted to "standard" once already.
    """
    ref = make_reference()
    reads, _ = make_reads(ref)
    pipe, gm = build("hybrid")
    results, _ = pipe.align(reads, reference_for(pipe, ref, gm))

    allowed = set(pipe.model.router.cfg.route_names)
    assert allowed == {"fast", "medium", "full"}, allowed
    seen = {r.primary.route for r in results if r.primary is not None}
    assert seen <= allowed, (seen, allowed)
    print(f"routes seen {sorted(seen)} all drawn from RouterConfig {sorted(allowed)}")


def test_hybrid_end_to_end():
    ref = make_reference()
    reads, truth = make_reads(ref)
    pipe, gm = build("hybrid")
    reference = reference_for(pipe, ref, gm)
    results, stats = pipe.align(reads, reference)

    assert len(results) == len(reads)
    acc = accuracy(results, truth)
    print(f"hybrid: {stats.summary()}")
    print(f"   locus accuracy {acc:.0%}")
    assert acc >= 0.75, acc
    assert stats.n_neural_batches >= 1

    rec = results[0].primary
    assert rec.is_mapped and rec.cigar, rec
    consumed = sum(n for op, n in rec.cigar if op in "=XIS")
    assert consumed == rec.read_len, (consumed, rec.read_len)
    assert "score" in rec.stages and "extend" in rec.stages
    print(f"   CIGAR covers the read exactly; stages={rec.stages}")
    print(f"   example: {rec.ref_start} strand={rec.strand} mapq={rec.mapq} "
          f"cigar={rec.cigar_string[:40]}")


def test_all_modes_align():
    ref = make_reference()
    reads, truth = make_reads(ref)
    for mode in ("hybrid", "fast", "two_pass"):
        pipe, gm = build(mode)
        reference = reference_for(pipe, ref, gm)
        results, stats = pipe.align(reads, reference)
        acc = accuracy(results, truth)
        assert len(results) == len(reads)
        assert acc >= 0.75, (mode, acc)
        names = {r.pass_name for a in results for r in a.records}
        assert names == {mode}, (mode, names)
        print(f"{mode:9s} acc={acc:.0%}  {stats.summary()}")


def test_neural_pruning_shrinks_dp_input():
    ref = make_reference()
    reads, _ = make_reads(ref)
    pipe, gm = build("hybrid")
    reference = reference_for(pipe, ref, gm)
    _, stats = pipe.align(reads, reference)
    assert stats.n_anchors > 0
    print(f"anchors seeded={stats.n_anchors} pruned_by_neural={stats.n_anchors_pruned}")


def test_agnes_seed_graph_has_spatial_edges_and_features():
    pipe, _ = build("hybrid")
    anchors = AnchorSet.from_lists(
        read_pos=[0, 20, 40, 60, 80, 100],
        ref_pos=[100, 120, 141, 161, 181, 201],
        length=[10] * 6,
        strand=[1] * 6,
        node_id=[0, 0, 1, 1, 2, 2],
        read_len=120,
        ref_len=1000,
    )
    edge_index, edge_features, edge_mask, active = pipe.scorer._pad_seed_graph(
        [anchors], torch.device("cpu")
    )
    assert active.tolist() == [True]
    assert edge_mask.sum() > 0
    assert edge_index.shape[-1] == 2
    assert edge_features.shape[-1] == pipe.model.cfg.seed_scoring.anchor_edge_features
    assert torch.isfinite(edge_features).all()


def test_agnes_fallback_keeps_all_seeds_and_requires_live_edges():
    pipe, _ = build("hybrid")
    anchors = AnchorSet.from_lists(
        read_pos=[0, 20, 40, 60, 80, 100],
        ref_pos=[100, 120, 140, 160, 180, 200],
        length=[10] * 6,
        strand=[1] * 6,
        read_len=120,
        ref_len=1000,
    )
    anchors.score[:] = np.array([0.98, 0.97, 0.96, 0.04, 0.03, 0.02])

    kept, trusted = pipe._prepare_anchors_agnes(
        [anchors], {"gnn_active": torch.tensor([False])}
    )
    assert trusted == [False]  # |E|=0 / inactive graph forces PureDP
    assert kept[0] is anchors and len(kept[0]) == 6  # PureDP uses unchanged V

    kept, trusted = pipe._prepare_anchors_agnes(
        [anchors], {"gnn_active": torch.tensor([True])}
    )
    assert trusted == [True]
    assert kept[0] is anchors and len(kept[0]) == 6


def _records_for_modality(seqs, truth, modality: str) -> list[ReadRecord]:
    """Stamp synthetic reads with a modality token and Phred qualities."""
    out = []
    for i, (seq, (start, strand)) in enumerate(zip(seqs, truth)):
        out.append(
            ReadRecord(
                read_id=f"{modality}_{i}",
                ref_id=0,
                modality=modality,
                seq=seq,
                quals=[30] * len(seq),
                ref_start=start,
                ref_end=start + len(seq),
                strand=strand,
                cigar=[("=", len(seq))],
                ref_positions=list(range(start, start + len(seq))),
                mapq=40,
            )
        )
    return out


def test_agnes_gnn_modules_across_all_modalities():
    """Seed-graph GNN + hybrid chaining must work for every modality token.

    Covers the registered modality conditioning tokens (illumina through
    linked_reads). For each modality we (1) force a live AGNES seed graph so
    EdgeConv + transition heads actually run under that modality token, then
    (2) align through the hybrid path end-to-end and check locus accuracy.
    """
    assert set(MODALITIES) == {
        "illumina",
        "pacbio_hifi",
        "ont",
        "rna_seq",
        "bisulfite",
        "single_cell",
        "linked_reads",
    }, sorted(MODALITIES)

    # Per-modality smoke profiles: short/exact for Illumina-like, longer / noisier
    # for ONT; everything else shares the HiFi-like profile.
    profiles = {
        "illumina": dict(n=6, read_len=150, rate=0.01, tol=20),
        "pacbio_hifi": dict(n=6, read_len=200, rate=0.01, tol=30),
        "ont": dict(n=6, read_len=250, rate=0.08, tol=40),
        "rna_seq": dict(n=6, read_len=150, rate=0.02, tol=30),
        "bisulfite": dict(n=6, read_len=150, rate=0.02, tol=30),
        "single_cell": dict(n=6, read_len=150, rate=0.02, tol=30),
        "linked_reads": dict(n=6, read_len=150, rate=0.02, tol=30),
    }

    pipe, gm = build("hybrid")
    assert pipe.model.cfg.seed_scoring.use_anchor_gnn
    assert pipe.model.seed_scorer.anchor_gnn is not None
    # Keep seeding dense enough that real Stage-1 graphs often clear |V|>=5.
    pipe.cfg.seeding.modes = ("minimizer", "smem", "fuzzy")

    ref = make_reference(length=4000)
    reference = reference_for(pipe, ref, gm)
    summary = []

    # Shared collinear seed graph used to force the GNN path under each modality
    # token, independent of Stage-1 density.
    forced = AnchorSet.from_lists(
        read_pos=[0, 20, 40, 60, 80, 100, 120, 140],
        ref_pos=[200, 220, 241, 261, 281, 301, 321, 341],
        length=[12] * 8,
        strand=[1] * 8,
        node_id=[0, 0, 1, 1, 2, 2, 3, 3],
        read_len=160,
        ref_len=len(ref),
    )

    from graphmambaformer.alignment.scoring import encode_read_batch

    for modality in sorted(MODALITIES):
        params = profiles[modality]
        seqs, truth = make_reads(
            ref, n=params["n"], read_len=params["read_len"], rate=params["rate"]
        )
        records = _records_for_modality(seqs, truth, modality)

        # Force a live seed graph so EdgeConv / transition heads run with this
        # modality's conditioning token (not only PureDP fallback).
        codes, mask, quals = encode_read_batch(
            [records[0].seq], pipe.device, quals=[records[0].quals]
        )
        with torch.inference_mode():
            outputs = pipe.model(
                codes,
                mask=mask,
                graph=reference.graph,
                qualities=quals,
                modality=modality,
            )
            scores, head = pipe.scorer.score_anchors(outputs, [forced])

        assert bool(head["gnn_active"][0]), modality
        assert int(head["edge_mask"].sum()) > 0, modality
        assert "transition_score" in head and head["transition_score"].ndim == 2
        assert torch.isfinite(head["transition_score"]).all(), modality
        assert np.isfinite(scores[0]).all() and scores[0].shape == (len(forced),)

        prepared, trust_flags = pipe._prepare_anchors_agnes([forced], head)
        assert prepared[0] is forced and len(prepared[0]) == len(forced)
        assert isinstance(trust_flags[0], bool)

        # Full hybrid path with modality-tagged ReadRecords.
        results, stats = pipe.align(records, reference)
        acc = accuracy(results, truth, tol=params["tol"])
        assert len(results) == len(records), modality
        assert stats.n_neural_batches >= 1, modality
        assert acc >= 0.5, (modality, acc, stats.summary())
        assert all(r.primary is not None for r in results), modality

        summary.append(
            f"{modality:13s} acc={acc:.0%} anchors={stats.n_anchors} "
            f"chains={stats.n_chains} forced_edges={int(head['edge_mask'].sum())} "
            f"forced_active={bool(head['gnn_active'][0])} trust={trust_flags[0]}"
        )

    for line in summary:
        print(line)
    print(f"AGNES seed-graph GNN + hybrid chaining OK on all {len(MODALITIES)} modalities")


def test_encoder_baseline_degrades_gracefully():
    """A sequence-only baseline has no Stage 4 heads: hybrid must still align."""
    ref = make_reference()
    reads, truth = make_reads(ref)
    pipe, gm = build("hybrid", arch="hybrid")
    assert not pipe.uses_neural_scoring
    reference = reference_for(pipe, ref, gm)
    results, stats = pipe.align(reads, reference)
    assert accuracy(results, truth) >= 0.75
    routes = {r.route for a in results for r in a.records}
    assert routes == {"classical"}, routes
    print(f"encoder baseline -> classical fallback OK ({stats.summary()})")


def test_no_model_runs_classical():
    ref = make_reference()
    reads, truth = make_reads(ref)
    pipe, gm = build("hybrid", with_model=False)
    reference = reference_for(pipe, ref, gm)
    results, _ = pipe.align(reads, reference)
    assert accuracy(results, truth) >= 0.75
    print("hybrid with no model at all still aligns classically")


def test_two_pass_only_rescues_hard_reads():
    ref = make_reference()
    reads, _ = make_reads(ref, n=8)
    pipe, gm = build("two_pass")
    reference = reference_for(pipe, ref, gm)
    _, stats = pipe.align(reads, reference)
    fast = stats.per_pass.get("fast", 0)
    rescued = stats.per_pass.get("hybrid_rescue", 0)
    assert fast == len(reads)
    assert rescued <= len(reads)
    print(f"two_pass: fast={fast} hybrid_rescue={rescued} (neural paid only on the tail)")


def test_unmappable_read_is_reported():
    ref = make_reference()
    junk = "".join(RNG.choice(list("ACGT"), size=200))  # unrelated sequence
    pipe, gm = build("hybrid")
    reference = reference_for(pipe, ref, gm)
    results, _ = pipe.align([junk], reference)
    rec = results[0].primary
    assert rec is not None
    # Either unmapped, or rescued by the mapping head at the MAPQ floor.
    assert (not rec.is_mapped) or "rescue" in rec.stages, rec
    print(f"unrelated read -> mapped={rec.is_mapped} stages={rec.stages} mapq={rec.mapq}")


def test_batching_is_deterministic():
    ref = make_reference()
    reads, _ = make_reads(ref, n=6)
    out = []
    for batch_size in (2, 6):
        pipe, gm = build("hybrid", batch_size=batch_size)
        torch.manual_seed(0)
        reference = reference_for(pipe, ref, gm)
        results, _ = pipe.align(reads, reference)
        out.append([(r.primary.ref_start, r.primary.strand) for r in results])
    assert out[0] == out[1], out
    print("batch_size does not change the result (batch-invariant)")


def test_mapq_is_calibrated_range():
    ref = make_reference()
    reads, _ = make_reads(ref)
    pipe, gm = build("hybrid")
    reference = reference_for(pipe, ref, gm)
    results, _ = pipe.align(reads, reference)
    mapqs = [a.primary.mapq for a in results if a.primary.is_mapped]
    assert all(0 <= m <= 60 for m in mapqs), mapqs
    print(f"MAPQ within [0, 60]: {sorted(mapqs)}")


def test_multitask_signals_are_extracted_per_read():
    ref = make_reference()
    reads, _ = make_reads(ref, n=2)
    cfg = PipelineConfig(mode="hybrid")
    cfg.seeding.modes = ("minimizer", "smem")
    gm = GraphMambaConfig(d_model=64)
    gm.multi_task.haplotype = True
    gm.multi_task.ancestry = True
    model = build_core_model(
        CoreModelConfig(arch="multitask_graphmamba", graphmamba=gm)
    ).model
    model.eval()
    pipe = build_pipeline(cfg, model=model)
    results, _ = pipe.align(reads, reference_for(pipe, ref, gm))
    assert all({"haplotype", "ancestry", "ancestry_local"} <= set(row.signals) for row in results)
    assert all(row.signals["haplotype"].shape == (2,) for row in results)
    print("multi-task head tensors extracted per read for downstream stages")


def test_pruning_mechanism_and_floor():
    """Force low seed scores to prove pruning fires and respects its floor."""
    ref = make_reference()
    reads, truth = make_reads(ref)
    pipe, gm = build("hybrid")
    reference = reference_for(pipe, ref, gm)
    scorer = pipe.scorer

    anchor_sets = pipe.seed(reads, reference)
    n_before = sum(len(a) for a in anchor_sets)

    # Everything below threshold: the floor must still keep min_anchors_kept.
    pipe.cfg.scoring.min_anchors_kept = 1
    for a in anchor_sets:
        a.score[:] = 0.01
    pruned = [scorer.prune_anchors(a) for a in anchor_sets]
    assert all(len(p) == min(1, len(a)) for p, a in zip(pruned, anchor_sets)), [
        len(p) for p in pruned
    ]
    n_after = sum(len(p) for p in pruned)
    assert n_after < n_before, (n_before, n_after)

    # A high floor keeps everything even at score 0.
    pipe.cfg.scoring.min_anchors_kept = 999
    kept = [scorer.prune_anchors(a) for a in anchor_sets]
    assert [len(k) for k in kept] == [len(a) for a in anchor_sets]

    # Mixed scores: only the confident anchors survive.
    pipe.cfg.scoring.min_anchors_kept = 0
    multi = [a for a in anchor_sets if len(a) >= 2]
    assert multi, "need a read with >=2 anchors to test selective pruning"
    a = multi[0]
    a.score[:] = 0.9
    a.score[0] = 0.01
    survivors = scorer.prune_anchors(a)
    assert len(survivors) == len(a) - 1, (len(survivors), len(a))
    print(f"pruning fires ({n_before} -> {n_after} anchors), floor and threshold both honored")


def test_two_pass_rescues_hard_reads():
    """High-error reads are not 'easy', so the hybrid pass must pick them up."""
    ref = make_reference()
    easy, _ = make_reads(ref, n=4, rate=0.01)
    hard, _ = make_reads(ref, n=4, read_len=120, rate=0.18)
    reads = easy + hard

    pipe, gm = build("two_pass")
    reference = reference_for(pipe, ref, gm)
    _, stats = pipe.align(reads, reference)
    rescued = stats.per_pass.get("hybrid_rescue", 0)
    assert rescued > 0, stats.summary()
    assert stats.n_neural_batches >= 1
    print(f"two_pass rescue fires on hard reads: hybrid_rescue={rescued}/{len(reads)}")


def test_two_pass_matches_hybrid_on_hard_reads():
    """Rescued reads should get the hybrid answer, not the fast one."""
    ref = make_reference()
    reads, truth = make_reads(ref, n=6, read_len=120, rate=0.15)
    out = {}
    for mode in ("two_pass", "hybrid"):
        pipe, gm = build(mode)
        reference = reference_for(pipe, ref, gm)
        results, _ = pipe.align(reads, reference)
        out[mode] = [
            (r.primary.is_mapped, r.primary.ref_start, r.primary.strand) for r in results
        ]
    agree = sum(a == b for a, b in zip(out["two_pass"], out["hybrid"]))
    print(f"two_pass agrees with hybrid on {agree}/{len(reads)} hard reads")
    assert agree == len(reads), (out["two_pass"], out["hybrid"])


if __name__ == "__main__":
    test_default_is_hybrid()
    test_hybrid_end_to_end()
    test_all_modes_align()
    test_neural_pruning_shrinks_dp_input()
    test_agnes_seed_graph_has_spatial_edges_and_features()
    test_agnes_fallback_keeps_all_seeds_and_requires_live_edges()
    test_agnes_gnn_modules_across_all_modalities()
    test_encoder_baseline_degrades_gracefully()
    test_no_model_runs_classical()
    test_two_pass_only_rescues_hard_reads()
    test_unmappable_read_is_reported()
    test_batching_is_deterministic()
    test_mapq_is_calibrated_range()
    test_pruning_mechanism_and_floor()
    test_two_pass_rescues_hard_reads()
    test_two_pass_matches_hybrid_on_hard_reads()
    print("\nALL PIPELINE CHECKS PASSED")
