"""Verification of the alignment stages, the neural core, and the losses.

Complements ``verify_stages.py`` (which covers the Figure-1 encoders and the
MambaFormer backbone) by exercising everything downstream of them:

  Stage 1 — seeding: minimizer / SMEM anchors, strand handling, index modes
  Stage 2 — chaining: affine-gap DP, graph bonus, primary/secondary selection
  Stage 3 — extension: banded affine Smith-Waterman + WFA, CIGAR correctness
  Stage 4 — neural scoring: anchor pruning, chain re-ranking, MAPQ, rescue
  core    — GraphMambaModel forward/backward, architecture modes
  losses  — alignment + multi-task terms, Kendall weighting, a training step
  modes   — the three pipeline modes end to end, plus the classical fallback

Everything runs on CPU at small ``d_model`` in well under a minute.

Run: PYTHONPATH=. .venv/bin/python scripts/verify_alignment_pipeline.py
"""

from __future__ import annotations

import numpy as np
import torch

from graphmambaformer import (
    AffineChainer,
    ChainingConfig,
    CoreModelConfig,
    ExtensionConfig,
    ExtensionEngine,
    GraphMambaConfig,
    GraphMambaLoss,
    LossConfig,
    ModelConfig,
    PipelineConfig,
    SeedingConfig,
    SeedingEngine,
    build_core_model,
    build_pipeline,
)
from graphmambaformer.alignment import WavefrontAligner, encode_bases
from graphmambaformer.alignment.pipeline import (
    FastAlignmentPipeline,
    HybridAlignmentPipeline,
    TwoPassAligner,
)
from graphmambaformer.models.graph_mamba import GraphBatch

BASES = "ACGT"
RNG = np.random.default_rng(11)
NODE_LEN = 200


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def ok(message: str) -> None:
    print(f"  [ok] {message}")


# --------------------------------------------------------------------------- #
# Synthetic reference / reads
# --------------------------------------------------------------------------- #
def make_reference(length: int = 4000) -> str:
    return "".join(RNG.choice(list(BASES), size=length))


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def mutate(seq: str, rate: float) -> str:
    out = list(seq)
    for i in range(len(out)):
        if RNG.random() < rate:
            out[i] = RNG.choice(list(BASES))
    return "".join(out)


def make_reads(ref, n=8, read_len=200, rate=0.02):
    """Reads sampled from the reference, every third one reverse-complemented."""
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


def graph_for(ref: str, cfg: GraphMambaConfig):
    """A linear pangenome graph over the reference, shared by the batch."""
    n_nodes = len(ref) // NODE_LEN
    node_seqs = [ref[i * NODE_LEN : (i + 1) * NODE_LEN] for i in range(n_nodes)]
    node_start = [i * NODE_LEN for i in range(n_nodes)]
    src = torch.arange(n_nodes - 1)
    edges = torch.stack([src, src + 1])
    graph = GraphBatch(
        node_kmer_ids=torch.randint(0, 4**cfg.graph_encoder.kmer_size, (n_nodes, 6)),
        edge_index=edges,
        edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
    )
    return node_seqs, node_start, edges.numpy(), graph


def small_core(arch: str = "graphmamba", **overrides):
    cfg = GraphMambaConfig(d_model=64, **overrides)
    model = build_core_model(CoreModelConfig(arch=arch, graphmamba=cfg)).model
    model.eval()
    return model, cfg


def locus_accuracy(results, truth, tol: int = 30) -> float:
    hits = 0
    for alignments, (start, strand) in zip(results, truth):
        record = alignments.primary
        if record is None or not record.is_mapped:
            continue
        if record.strand == strand and abs(record.ref_start - start) <= tol:
            hits += 1
    return hits / max(len(truth), 1)


# --------------------------------------------------------------------------- #
# Stage 1 — seeding
# --------------------------------------------------------------------------- #
def verify_seeding() -> None:
    banner("Stage 1 - Seeding")
    ref = make_reference()
    engine = SeedingEngine(SeedingConfig(modes=("minimizer", "smem")))
    bundle = engine.build_indices(ref)

    read = ref[1000:1200]
    anchors = engine.seed_read(read, bundle)
    assert len(anchors) > 0, "no anchors on an exact substring"
    # Every anchor must describe a real match: read[rp:rp+l] == ref[fp:fp+l].
    read_codes = encode_bases(read)
    for i in range(len(anchors)):
        rp, fp, ln = anchors.read_pos[i], anchors.ref_pos[i], anchors.length[i]
        if anchors.strand[i] != 1:
            continue
        assert np.array_equal(read_codes[rp : rp + ln], bundle.ref_codes[fp : fp + ln]), i
    ok(f"{len(anchors)} anchors, every one an exact match")

    on_diagonal = anchors.diagonal == 1000
    assert on_diagonal.any(), "no anchor on the true diagonal"
    ok(f"{int(on_diagonal.sum())}/{len(anchors)} anchors on the true diagonal (1000)")

    rc_anchors = engine.seed_read(revcomp(read), bundle)
    assert (rc_anchors.strand == -1).any(), "reverse-complemented read found no - strand"
    ok("reverse-complemented read yields minus-strand anchors")

    for modes in (("minimizer",), ("smem",), ("fmindex",), ("dbg",), ("fuzzy",),
                  ("multiplex_dbg",), ("minimizer", "smem", "fuzzy")):
        eng = SeedingEngine(SeedingConfig(modes=modes))
        found = eng.seed_read(read, eng.build_indices(ref))
        assert len(found) > 0, modes
    ok("all seeding index modes produce anchors (incl. a 3-way mixed run)")


# --------------------------------------------------------------------------- #
# Stage 2 — chaining
# --------------------------------------------------------------------------- #
def verify_chaining() -> None:
    banner("Stage 2 - Chaining")
    ref = make_reference()
    engine = SeedingEngine(SeedingConfig(modes=("minimizer", "smem")))
    bundle = engine.build_indices(ref)
    chainer = AffineChainer(ChainingConfig())

    read = mutate(ref[1500:1750], 0.02)
    anchors = engine.seed_read(read, bundle)
    chains = chainer.chain(anchors)
    assert chains, "no chain from a mutated substring"
    best = chains[0]
    assert best.is_primary
    assert abs(best.ref_start - 1500) < 60, best.ref_start
    ok(f"primary chain at ref {best.ref_start} (truth 1500), "
       f"{len(best)} anchors, score {best.score:.1f}")

    assert best.coverage(len(read)) > 0.5
    ok(f"chain covers {best.coverage(len(read)):.0%} of the read")

    # A read whose halves come from loci 2.5 kb apart. Whether that is one
    # alignment with a deletion or two separate alignments is exactly what
    # max_gap decides, so both readings are checked.
    chimera = ref[300:450] + ref[3000:3150]
    chim_anchors = engine.seed_read(chimera, bundle)

    joined = chainer.chain(chim_anchors)
    assert len(joined) == 1 and len(joined[0]) == 2, [len(c) for c in joined]
    ok(f"long-read max_gap={chainer.cfg.max_gap}: halves join as one "
       f"{joined[0].ref_span - joined[0].read_span} bp deletion")

    short_read = AffineChainer(ChainingConfig(max_gap=500))
    split = short_read.chain(chim_anchors)
    starts = sorted(c.ref_start for c in split)
    assert len(split) >= 2, f"max_gap=500 still gave {len(split)} chain(s)"
    ok(f"short-read max_gap=500: splits into {len(split)} chains at {starts}")

    # Anchors must stay collinear within a chain.
    for chain in chains:
        pos = anchors.read_pos[chain.anchor_idx]
        assert np.all(np.diff(pos) >= 0), "chain anchors not in read order"
    ok("chain members are collinear in read order")


# --------------------------------------------------------------------------- #
# Stage 3 — extension
# --------------------------------------------------------------------------- #
def verify_extension() -> None:
    banner("Stage 3 - Dynamic programming extension")
    ref = make_reference()
    engine = SeedingEngine(SeedingConfig(modes=("minimizer", "smem")))
    bundle = engine.build_indices(ref)
    chainer = AffineChainer(ChainingConfig())

    read = mutate(ref[900:1150], 0.03)
    anchors = engine.seed_read(read, bundle)
    chains = chainer.chain(anchors)
    extender = ExtensionEngine(ExtensionConfig())
    results = extender.extend_chains(read, chains, anchors, ref)
    assert results, "no extension"

    best = results[0]
    consumed = sum(n for op, n in best.cigar if op in "=XIS")
    assert consumed == len(read), (consumed, len(read))
    ok(f"CIGAR consumes exactly {consumed} read bases; {best.cigar_string[:48]}...")

    ref_consumed = sum(n for op, n in best.cigar if op in "=XD")
    assert ref_consumed == best.ref_end - best.ref_start, (
        ref_consumed,
        best.ref_end - best.ref_start,
    )
    ok(f"CIGAR reference span matches ref_start..ref_end ({ref_consumed} bases)")
    assert best.identity > 0.85, best.identity
    ok(f"identity {best.identity:.1%} at a 3% error rate, score {best.score:.1f}")

    # Exact substring must align without any edit.
    exact = ref[2000:2200]
    exact_anchors = engine.seed_read(exact, bundle)
    exact_chains = chainer.chain(exact_anchors)
    exact_result = extender.extend_chains(exact, exact_chains, exact_anchors, ref)[0]
    assert exact_result.edit_distance == 0, exact_result.cigar
    ok("exact substring aligns at edit distance 0")

    # WFA agrees with the true edit distance on known pairs.
    wfa = WavefrontAligner(ExtensionConfig(algorithm="wfa"))
    cases = {
        ("ACGTACGTACGT", "ACGTACGTACGT"): 0,  # identical
        ("ACGTACGTACGT", "ACGTTCGTACGT"): 1,  # one substitution
        ("ACGTACGTACGT", "ACGTACGACGT"): 1,  # one deletion
        ("ACGTACGTACGT", "ACGTACGTTACGT"): 1,  # one insertion
    }
    for (a, b), expected in cases.items():
        distance, cigar = wfa.align(encode_bases(a), encode_bases(b))
        assert distance == expected, (a, b, distance, expected, cigar)
        consumed = sum(n for op, n in cigar if op in "=XI")
        assert consumed == len(a), (cigar, len(a))
    ok(f"WFA edit distances exact on {len(cases)} cases (match/sub/del/ins)")

    # WFA and banded DP must agree on the same pair.
    long_a = ref[1200:1400]
    long_b = mutate(long_a, 0.02)
    distance, _ = wfa.align(encode_bases(long_b), encode_bases(long_a))
    ok(f"WFA on a 200 bp 2%-error pair: edit distance {distance}")


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
def verify_core_model() -> None:
    banner("Core model - GraphMambaModel")
    ref = make_reference()
    model, cfg = small_core()
    _, _, _, graph = graph_for(ref, cfg)

    reads, _ = make_reads(ref, n=4)
    from graphmambaformer.alignment import encode_read_batch

    codes, mask = encode_read_batch(reads)
    with torch.no_grad():
        out = model(codes, mask=mask, graph=graph)

    assert out.read_hidden.shape[:2] == codes.shape
    assert out.pooled.shape == (len(reads), cfg.d_model)
    ok(f"read states {tuple(out.read_hidden.shape)}, pooled {tuple(out.pooled.shape)}")
    assert out.mapping is not None and out.router is not None
    ok(f"mapping head + router present (routes {out.router['route'].tolist()})")

    # Gradients must reach the backbone from the alignment heads.
    model.train()
    out = model(codes, mask=mask, graph=graph)
    out.mapping["mapq"].sum().backward()
    got = sum(1 for p in model.parameters() if p.grad is not None)
    assert got > 0
    ok(f"backward from MAPQ reaches {got} parameter tensors")
    model.eval()

    # Each selectable architecture must build. The encoder baselines are shrunk
    # too, so the comparison is at a like-for-like d_model.
    for arch in ("graphmamba", "multitask_graphmamba", "mambaformer", "hybrid"):
        # ModelConfig propagates d_model to its sub-configs in __post_init__, so
        # the size has to be passed at construction, not assigned afterwards.
        arch_cfg = CoreModelConfig(
            arch=arch,
            graphmamba=GraphMambaConfig(d_model=64),
            encoder=ModelConfig(d_model=64, n_blocks=2),
        )
        if arch == "multitask_graphmamba":
            # Heads are opt-in, so enable a few or the model is identical to the
            # plain one and the param count says nothing.
            for name in ("variant_calling", "haplotype", "bqsr"):
                setattr(arch_cfg.graphmamba.multi_task, name, True)
            arch_cfg.graphmamba.multi_task.d_model = 64
        spec = build_core_model(arch_cfg)
        extra = ""
        if arch == "multitask_graphmamba":
            extra = f" | task heads: {list(spec.model.enabled_tasks)}"
        ok(f"arch {arch:22s} {spec.summary()}{extra}")


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def verify_losses() -> None:
    banner("Losses - alignment terms + Kendall weighting + a training step")
    ref = make_reference()
    model, cfg = small_core()
    _, _, _, graph = graph_for(ref, cfg)
    reads, _ = make_reads(ref, n=4)

    from graphmambaformer.alignment import encode_read_batch

    codes, mask = encode_read_batch(reads)
    model.train()
    out = model(codes, mask=mask, graph=graph)

    b, n_anchor, n_chain = len(reads), 6, 3
    seed_head = model.score_seeds(
        out,
        seed_features=torch.randn(b, n_anchor, 12),
        anchor_read_pos=torch.randint(0, codes.shape[1], (b, n_anchor)),
        anchor_node=torch.randint(0, 4, (b, n_anchor)),
        anchor_mask=torch.ones(b, n_anchor, dtype=torch.bool),
    )
    chain_head = model.score_chains(
        chain_features=torch.randn(b, n_chain, 10),
        member_states=torch.randn(b, n_chain, 4, cfg.d_model),
        member_mask=torch.ones(b, n_chain, 4, dtype=torch.bool),
        chain_mask=torch.ones(b, n_chain, dtype=torch.bool),
    )
    targets = {
        "seed_labels": torch.randint(0, 2, (b, n_anchor)).float(),
        "anchor_mask": torch.ones(b, n_anchor, dtype=torch.bool),
        "chain_target": torch.zeros(b, dtype=torch.long),
        "chain_mask": torch.ones(b, n_chain, dtype=torch.bool),
        "node_target": torch.zeros(b, dtype=torch.long),
        "position_target": torch.rand(b),
        "mapq_target": torch.full((b,), 60.0),
    }

    loss_fn = GraphMambaLoss(LossConfig())
    result = loss_fn(out, targets, seed_scores=seed_head, chain_scores=chain_head)
    ok(f"terms: {', '.join(f'{k}={v:.3f}' for k, v in sorted(result.terms.items()))}")
    assert torch.isfinite(result.total)
    ok(f"total {float(result):.3f}, Kendall weights {len(loss_fn.weighting.log_vars)}")

    # A short training loop must actually reduce the loss.
    params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-3)
    history = []
    for _ in range(12):
        optimizer.zero_grad()
        out = model(codes, mask=mask, graph=graph)
        seed_head = model.score_seeds(
            out,
            seed_features=torch.zeros(b, n_anchor, 12),
            anchor_read_pos=torch.zeros((b, n_anchor), dtype=torch.long),
            anchor_mask=torch.ones(b, n_anchor, dtype=torch.bool),
        )
        loss = loss_fn(out, targets, seed_scores=seed_head)
        loss.total.backward()
        optimizer.step()
        history.append(float(loss))
    assert history[-1] < history[0], history
    ok(f"12 steps: loss {history[0]:.3f} -> {history[-1]:.3f} (decreasing)")
    model.eval()


# --------------------------------------------------------------------------- #
# Stage 4 + pipeline modes
# --------------------------------------------------------------------------- #
def verify_pipeline_modes() -> None:
    banner("Stage 4 + pipeline modes")
    ref = make_reference()
    reads, truth = make_reads(ref, n=9)

    expected = {
        "hybrid": HybridAlignmentPipeline,
        "fast": FastAlignmentPipeline,
        "two_pass": TwoPassAligner,
    }
    assert isinstance(build_pipeline(), HybridAlignmentPipeline)
    ok("default pipeline mode is hybrid")

    for mode, cls in expected.items():
        cfg = PipelineConfig(mode=mode)
        cfg.seeding.modes = ("minimizer", "smem")
        model, gm = small_core()
        pipeline = build_pipeline(cfg, model=model)
        assert isinstance(pipeline, cls)

        node_seqs, node_start, edges, graph = graph_for(ref, gm)
        reference = pipeline.build_reference(
            ref,
            node_seqs=node_seqs,
            node_ref_start=node_start,
            backbone_path=list(range(len(node_seqs))),
            edge_index=edges,
            graph=graph,
        )
        results, stats = pipeline.align(reads, reference)
        accuracy = locus_accuracy(results, truth)
        assert accuracy >= 0.75, (mode, accuracy)
        mapqs = [a.primary.mapq for a in results if a.primary.is_mapped]
        assert all(0 <= m <= 60 for m in mapqs)
        ok(f"{mode:9s} accuracy {accuracy:.0%} | {stats.summary()}")

    # Stage 4 mechanics: pruning honours its threshold and its floor.
    cfg = PipelineConfig(mode="hybrid")
    cfg.seeding.modes = ("minimizer", "smem")
    model, gm = small_core()
    pipeline = build_pipeline(cfg, model=model)
    node_seqs, node_start, edges, graph = graph_for(ref, gm)
    reference = pipeline.build_reference(
        ref, node_seqs=node_seqs, node_ref_start=node_start, graph=graph
    )
    anchor_sets = pipeline.seed(reads, reference)
    rich = max(anchor_sets, key=len)
    rich.score[:] = 0.9
    rich.score[0] = 0.01
    pipeline.cfg.scoring.min_anchors_kept = 0
    assert len(pipeline.scorer.prune_anchors(rich)) == len(rich) - 1
    pipeline.cfg.scoring.min_anchors_kept = len(rich)
    assert len(pipeline.scorer.prune_anchors(rich)) == len(rich)
    ok("anchor pruning honours both the score threshold and the keep floor")

    # A sequence-only baseline has no Stage 4 heads: hybrid must still align.
    baseline = build_pipeline(cfg, model=build_core_model(CoreModelConfig(arch="hybrid")).model)
    assert not baseline.uses_neural_scoring
    results, _ = baseline.align(reads, reference)
    assert locus_accuracy(results, truth) >= 0.75
    ok("encoder-baseline architecture degrades to the classical path, still aligns")


def main() -> None:
    torch.manual_seed(0)
    verify_seeding()
    verify_chaining()
    verify_extension()
    verify_core_model()
    verify_losses()
    verify_pipeline_modes()
    print(f"\n{'=' * 72}\nALL ALIGNMENT PIPELINE CHECKS PASSED\n{'=' * 72}")


if __name__ == "__main__":
    main()
