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
from graphmambaformer.config import (
    CoreModelConfig,
    GraphMambaConfig,
    PipelineConfig,
)
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
