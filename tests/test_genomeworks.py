"""Tests for the GenomeWorks acceleration layer (``accel.genomeworks_ops``).

These exercise the *portable* tier, which is what runs on CPU and what the CUDA
tiers are verified against, so they pass without a GPU. Each of the four
GenomeWorks primitives (cudaextender / cudaaligner / cudapoa / cudamapper) is
checked for biological correctness on small, hand-verifiable inputs.
"""

from __future__ import annotations

import random

from graphmambaformer.accel import genomeworks_ops as gw
from graphmambaformer.alignment.types import cigar_read_length, cigar_ref_length


def _random_seq(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# --------------------------------------------------------------------------- #
# Availability plumbing
# --------------------------------------------------------------------------- #
def test_availability_and_summary():
    assert gw.genomeworks_available() is True
    assert gw.genomeworks_backend() in ("pyclaragenomics", "cuda_rawkernel", "portable")
    summary = gw.genomeworks_summary()
    assert "genomeworks" in summary and "backend=" in summary
    # On a box without the archived bindings this is the honest answer.
    assert isinstance(gw.genomeworks_bindings_available(), bool)


# --------------------------------------------------------------------------- #
# cudaextender — ungapped X-drop seed extension
# --------------------------------------------------------------------------- #
def test_ungapped_extend_perfect_match():
    seq = "ACGTACGTACGT"
    ext = gw.ungapped_extend(seq, seq, seed_query=4, seed_target=4, match=2, mismatch=4)
    # A perfect diagonal extends across the whole sequence.
    assert ext.query_start == 0 and ext.query_end == len(seq)
    assert ext.target_start == 0 and ext.target_end == len(seq)
    assert ext.score == 2 * len(seq)
    assert ext.length == len(seq)


def test_ungapped_extend_xdrop_stops():
    # A run of matches, then a wall of mismatches. With a small x_drop the walk
    # must stop at the last match rather than paying through the mismatches.
    query = "AAAAAA" + "CCCCCC"
    target = "AAAAAA" + "GGGGGG"
    ext = gw.ungapped_extend(query, target, seed_query=0, seed_target=0,
                             match=2, mismatch=4, x_drop=3)
    assert ext.query_end == 6  # stops right after the 6 matches
    assert ext.score == 12


def test_ungapped_extend_left_and_right():
    #             seed here v
    query = "TTTTACGTACGTGGGG"
    target = "xxxxACGTACGTyyyy".replace("x", "A").replace("y", "A")
    # Only the central ACGTACGT block matches; flanks differ.
    ext = gw.ungapped_extend(query, target, seed_query=6, seed_target=6,
                             match=2, mismatch=4, x_drop=3)
    assert query[ext.query_start:ext.query_end] == "ACGTACGT"
    assert ext.score == 16


def test_ungapped_extend_batch_matches_single():
    query = "ACGTACGTACGTACGT"
    target = "ACGTACGAACGTACGT"  # one mismatch at index 7
    seeds = [(0, 0), (8, 8), (4, 4)]
    batch = gw.ungapped_extend_batch(query, target, seeds, match=2, mismatch=4, x_drop=10)
    singles = [gw.ungapped_extend(query, target, sq, sr, match=2, mismatch=4, x_drop=10)
               for sq, sr in seeds]
    assert batch == singles


# --------------------------------------------------------------------------- #
# cudaaligner — global affine alignment with CIGAR
# --------------------------------------------------------------------------- #
def test_global_align_identity():
    aln = gw.global_align("ACGTACGT", "ACGTACGT", match=2, mismatch=4)
    assert aln.cigar == [("=", 8)]
    assert aln.score == 16
    assert aln.n_match == 8 and aln.edit_distance == 0


def test_global_align_substitution():
    aln = gw.global_align("ACGT", "ACTT", match=2, mismatch=4)
    assert aln.cigar == [("=", 2), ("X", 1), ("=", 1)]
    assert aln.n_mismatch == 1
    assert aln.score == 2 + 2 - 4 + 2


def test_global_align_insertion_and_lengths():
    # Query has an extra base relative to the target -> one insertion.
    aln = gw.global_align("ACGTT", "ACGT", match=2, mismatch=4, gap_open=6, gap_extend=2)
    assert aln.n_insertion == 1
    assert cigar_read_length(aln.cigar) == 5   # consumes all query bases
    assert cigar_ref_length(aln.cigar) == 4    # consumes all target bases


def test_global_align_deletion():
    # Target has an extra base relative to the query -> one deletion.
    aln = gw.global_align("ACGT", "ACGTT", match=2, mismatch=4)
    assert aln.n_deletion == 1
    assert cigar_read_length(aln.cigar) == 4
    assert cigar_ref_length(aln.cigar) == 5


def test_global_align_empty():
    assert gw.global_align("", "").cigar == []
    assert gw.global_align("ACGT", "").cigar == [("I", 4)]
    assert gw.global_align("", "ACGT").cigar == [("D", 4)]


# --------------------------------------------------------------------------- #
# cudapoa — partial-order alignment consensus
# --------------------------------------------------------------------------- #
def test_poa_consensus_identical():
    assert gw.poa_consensus(["ACGTACGT", "ACGTACGT", "ACGTACGT"]) == "ACGTACGT"


def test_poa_consensus_majority_substitution():
    # Two reads agree, one disagrees at the 3rd base -> majority base wins.
    reads = ["ACGTACGT", "ACGTACGT", "ACTTACGT"]
    assert gw.poa_consensus(reads) == "ACGTACGT"


def test_poa_consensus_tolerates_indel():
    # One read carries a single-base insertion; the consensus should recover the
    # length-8 backbone the majority supports.
    reads = ["ACGTACGT", "ACGTACGT", "ACGTAACGT"]
    consensus = gw.poa_consensus(reads)
    assert consensus == "ACGTACGT"


def test_poa_consensus_single_read():
    assert gw.poa_consensus(["ACGTACGT"]) == "ACGTACGT"
    assert gw.poa_consensus([]) == ""


def test_poa_msa_returns_consensus_and_inputs():
    consensus, inputs = gw.poa_msa(["ACGT", "ACGT"])
    assert consensus == "ACGT"
    assert inputs == ["ACGT", "ACGT"]


# --------------------------------------------------------------------------- #
# cudamapper — GPU seeding + chaining facade (runs on the portable tier on CPU)
# --------------------------------------------------------------------------- #
def test_map_to_reference_places_read():
    reference = _random_seq(800, seed=1)
    read = reference[300:300 + 120]
    overlaps = gw.map_to_reference([read], reference, kmer=13, window=5, device="cpu")
    assert len(overlaps) == 1
    assert overlaps[0], "expected at least one overlap for a substring read"
    top = overlaps[0][0]
    # The chain should land on/near the true locus (300) on the forward strand.
    assert top.strand == 1
    assert abs(top.target_start - 300) <= 20


# --------------------------------------------------------------------------- #
# End-to-end wiring: the primitives plugged into the alignment pipeline
# --------------------------------------------------------------------------- #
def _fast_pipeline(**extension_overrides):
    from graphmambaformer import PipelineConfig, build_pipeline

    cfg = PipelineConfig(mode="fast")
    for key, value in extension_overrides.items():
        setattr(cfg.extension, key, value)
    return build_pipeline(cfg, model=None, device="cpu")


def test_extension_cudaaligner_algorithm_maps_read():
    reference = _random_seq(900, seed=7)
    read = reference[400:400 + 150]
    pipeline = _fast_pipeline(algorithm="cudaaligner")
    reference_index = pipeline.build_reference(reference)
    results, _ = pipeline.align([read], reference_index)
    primary = results[0].primary
    assert primary is not None and primary.is_mapped
    assert abs(primary.ref_start - 400) <= 20
    # A clean substring aligns with a CIGAR that consumes the whole read.
    assert cigar_read_length(primary.cigar) == len(read)
    assert primary.cigar_string.endswith("=") or "=" in primary.cigar_string


def test_extension_ungapped_prefilter_keeps_true_chain():
    reference = _random_seq(900, seed=11)
    read = reference[250:250 + 150]
    pipeline = _fast_pipeline(ungapped_prefilter=True, ungapped_min_score=20.0)
    reference_index = pipeline.build_reference(reference)
    results, _ = pipeline.align([read], reference_index)
    primary = results[0].primary
    assert primary is not None and primary.is_mapped
    assert abs(primary.ref_start - 250) <= 20


def test_extension_prefilter_matches_default_for_easy_read():
    # For a clean substring the prefilter must not change the locus the default
    # path finds — it only saves work on hopeless chains.
    reference = _random_seq(900, seed=13)
    read = reference[500:500 + 150]

    base = _fast_pipeline()
    ref_idx = base.build_reference(reference)
    base_results, _ = base.align([read], ref_idx)
    base_primary = base_results[0].primary

    filt = _fast_pipeline(ungapped_prefilter=True, ungapped_min_score=20.0)
    ref_idx2 = filt.build_reference(reference)
    filt_results, _ = filt.align([read], ref_idx2)
    filt_primary = filt_results[0].primary

    assert base_primary.is_mapped and filt_primary.is_mapped
    assert base_primary.ref_start == filt_primary.ref_start


# --------------------------------------------------------------------------- #
# cudapoa consensus polisher (Stage 5 wiring)
# --------------------------------------------------------------------------- #
def test_consensus_polisher_majority():
    from graphmambaformer.alignment import ConsensusPolisher

    polisher = ConsensusPolisher(min_depth=2)
    result = polisher.consensus(["ACGTACGTAC", "ACGTACGTAC", "ACGTACGTAC"])
    assert result.depth == 3
    assert result.consensus == "ACGTACGTAC"
    assert result.backend in ("pyclaragenomics", "cuda_rawkernel", "portable")


def test_consensus_polisher_below_min_depth():
    from graphmambaformer.alignment import ConsensusPolisher

    polisher = ConsensusPolisher(min_depth=3)
    result = polisher.consensus(["ACGTACGT"])
    # Not enough depth: echo the single read unchanged rather than fabricate one.
    assert result.consensus == "ACGTACGT"
    assert result.backend == "none"


# --------------------------------------------------------------------------- #
# Pipeline-wide control surface: master switch + per-stage GenomeWorks routing
# --------------------------------------------------------------------------- #
def _pipeline_with_accel(accel, **ext):
    from graphmambaformer import PipelineConfig, build_pipeline

    cfg = PipelineConfig(mode="fast")
    cfg.accel = accel
    for key, value in ext.items():
        setattr(cfg.extension, key, value)
    return build_pipeline(cfg, model=None, device="cpu")


def test_extension_genomeworks_backend_forces_cudaaligner():
    from graphmambaformer.config import AccelConfig

    accel = AccelConfig(stage_backends={
        "seeding": "auto", "chaining": "auto", "extension": "genomeworks"})
    pipeline = _pipeline_with_accel(accel)
    # The route flips the effective algorithm to cudaaligner and turns the
    # cudaextender ungapped prefilter on, without any per-stage flag.
    assert pipeline.extender._effective_algorithm() == "cudaaligner"
    assert pipeline.extender._prefilter_enabled() is True

    reference = _random_seq(900, seed=21)
    read = reference[350:500]
    ref_idx = pipeline.build_reference(reference)
    primary = pipeline.align([read], ref_idx)[0][0].primary
    assert primary is not None and primary.is_mapped
    assert abs(primary.ref_start - 350) <= 20
    assert cigar_read_length(primary.cigar) == len(read)


def test_master_switch_off_downgrades_genomeworks_paths():
    from graphmambaformer.config import AccelConfig

    # Both GW knobs requested, but the master switch vetoes them.
    pipeline = _pipeline_with_accel(
        AccelConfig(genomeworks=False),
        algorithm="cudaaligner",
        ungapped_prefilter=True,
    )
    assert pipeline.extender._effective_algorithm() == "banded_sw"
    assert pipeline.extender._prefilter_enabled() is False

    reference = _random_seq(900, seed=23)
    read = reference[300:450]
    ref_idx = pipeline.build_reference(reference)
    primary = pipeline.align([read], ref_idx)[0][0].primary
    assert primary is not None and primary.is_mapped


def test_seeding_genomeworks_backend_uses_cudamapper_index():
    from graphmambaformer.config import AccelConfig

    accel = AccelConfig(stage_backends={
        "seeding": "genomeworks", "chaining": "genomeworks", "extension": "auto"})
    pipeline = _pipeline_with_accel(accel)
    assert pipeline.seeder.cfg.modes == ("cudamapper",)

    reference = _random_seq(900, seed=25)
    read = reference[420:560]
    ref_idx = pipeline.build_reference(reference)
    primary = pipeline.align([read], ref_idx)[0][0].primary
    assert primary is not None and primary.is_mapped
    assert abs(primary.ref_start - 420) <= 20


# --------------------------------------------------------------------------- #
# Stage-5 cudapoa wiring: SevenStagePipeline locus consensus
# --------------------------------------------------------------------------- #
def _seven_stage(aligner):
    from graphmambaformer.alignment import (
        ConsensusPolisher,
        SevenStagePipeline,
        SevenStageResources,
    )

    ref = _random_seq(400, seed=31)
    ref_index = aligner.build_reference(ref)
    resources = SevenStageResources(
        reference_sequences={"linear": ref},
        consensus_polisher=ConsensusPolisher(min_depth=2),
    )
    pipe = SevenStagePipeline(aligner, resources, strict=False)
    reads = [ref[120:260], ref[120:260], ref[120:260]]
    result = pipe.run(reads, {"linear": ref_index}, read_ids=["r0", "r1", "r2"])
    return ref, result


def test_seven_stage_polishes_locus_cluster_with_cudapoa():
    from graphmambaformer import PipelineConfig, build_pipeline

    aligner = build_pipeline(PipelineConfig(mode="fast"), model=None, device="cpu")
    ref, result = _seven_stage(aligner)
    assert result.locus_consensus, "reads clustered at one locus should be polished"
    lc = result.locus_consensus[0]
    assert lc.depth >= 2
    assert lc.backend in ("pyclaragenomics", "cuda_rawkernel", "portable")
    # Three identical reads collapse to exactly that reference window.
    assert lc.consensus == ref[120:260]


def test_seven_stage_consensus_respects_master_switch():
    from graphmambaformer import PipelineConfig, build_pipeline
    from graphmambaformer.config import AccelConfig

    cfg = PipelineConfig(mode="fast")
    cfg.accel = AccelConfig(genomeworks=False)
    aligner = build_pipeline(cfg, model=None, device="cpu")
    _, result = _seven_stage(aligner)
    # Master switch off: no POA runs even though a polisher was supplied.
    assert result.locus_consensus == []


# --------------------------------------------------------------------------- #
# Capability plumbing reaches AccelContext
# --------------------------------------------------------------------------- #
def test_accel_context_reports_genomeworks():
    from graphmambaformer.accel import AccelContext

    caps = AccelContext(device="cpu").caps
    assert hasattr(caps, "has_genomeworks")
    assert caps.genomeworks_backend in ("pyclaragenomics", "cuda_rawkernel", "portable")
    assert "genomeworks=" in caps.summary()


# --------------------------------------------------------------------------- #
# Regression: the CuPy chaining kernel must not be fed CPU tensors.
#
# ``kernel_backend("chaining")`` returns ``cuda_rawkernel``/``genomeworks`` on a
# GPU host regardless of the requested device, so a caller that pins a CPU
# device (e.g. ``map_to_reference(..., device="cpu")`` to force the portable
# reference on a GPU box) used to hand host tensors to ``_as_cupy`` and crash
# with ``TypeError: CPU arrays cannot be directly imported to CuPy``. The
# chainer must instead run the NumPy path for any non-CUDA device. This
# reproduces the failure on CPU (no GPU needed) by forcing the GW backend.
# --------------------------------------------------------------------------- #
def _diagonal_anchors():
    import numpy as np

    from graphmambaformer.alignment.types import AnchorSet

    read_pos = np.array([0, 40, 80, 120], dtype=np.int64)
    ref_pos = np.array([100, 140, 180, 220], dtype=np.int64)
    return AnchorSet.from_lists(
        read_pos=read_pos,
        ref_pos=ref_pos,
        length=np.full(4, 15, dtype=np.int64),
        strand=np.ones(4, dtype=np.int8),
        read_len=200,
        ref_len=400,
    )


def test_genomeworks_chaining_on_cpu_device_falls_back_to_numpy():
    from graphmambaformer.alignment.chaining import AffineChainer
    from graphmambaformer.config import ChainingConfig

    cfg = ChainingConfig()
    reference = AffineChainer(cfg, backend="torch").chain(_diagonal_anchors())

    for backend in ("cuda_rawkernel", "genomeworks"):
        chainer = AffineChainer(cfg, backend=backend)
        # A CPU device must not attempt the CuPy kernel — no TypeError, and the
        # result is identical to the portable NumPy chainer.
        chains = chainer.chain(_diagonal_anchors(), device="cpu")
        batched = chainer.chain_batch(
            [_diagonal_anchors(), _diagonal_anchors()], device="cpu"
        )
        assert len(chains) == len(reference)
        assert [c.score for c in chains] == [c.score for c in reference]
        assert all(cb for cb in batched)
