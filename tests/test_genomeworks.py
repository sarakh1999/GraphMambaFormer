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


def _random_poa_groups(seed: int, n_groups: int = 8):
    """A handful of independent locus pile-ups: a backbone per group plus a few
    reads carrying random substitutions / short indels."""
    rng = random.Random(seed)

    def _mut(base: str) -> str:
        chars = list(base)
        for _ in range(rng.randint(0, 3)):
            p = rng.randrange(len(chars))
            roll = rng.random()
            if roll < 0.6:                       # substitution
                chars[p] = rng.choice("ACGT")
            elif roll < 0.8:                     # insertion
                chars.insert(p, rng.choice("ACGT"))
            else:                                # deletion
                del chars[p]
        return "".join(chars) or base

    groups = []
    for _ in range(n_groups):
        base = "".join(rng.choice("ACGT") for _ in range(rng.randint(20, 60)))
        depth = rng.randint(1, 5)
        groups.append([_mut(base) for _ in range(depth)])
    return groups


def test_poa_consensus_batch_matches_per_group():
    """The batched entry point returns exactly per-group ``poa_consensus``.

    On CPU this exercises the host fallback path (no CuPy), which must be the
    plain per-group reference in order.
    """
    groups = _random_poa_groups(seed=17)
    groups += [[], ["ACGTACGT"]]  # empty group and a single-read group
    assert gw.poa_consensus_batch(groups) == [gw.poa_consensus(g) for g in groups]


def _poa_fill_scalar(spec, match, mismatch, gap):
    """NumPy scalar reproduction of the ``poa_fill`` CUDA kernel recurrence.

    Same border init, same predecessor iteration order and strict-``>`` first-
    wins tie-break, same insertion-last comparison. Used to validate the GPU
    fill path (spec construction + traceback + fold + orchestration) end-to-end
    on CPU, without a device — the analog of the cudaaligner traceback test.
    """
    import numpy as np

    G, S = spec.G, spec.S
    NEG = -1e30
    score = np.full((G + 1, S + 1), NEG, dtype=np.float64)
    move = np.zeros((G + 1, S + 1), dtype=np.int8)
    pred_g = np.full((G + 1, S + 1), -1, dtype=np.int64)
    off, flat, base, codes = spec.pred_off, spec.pred_flat, spec.base_of_row, spec.codes

    score[0, 0] = 0.0
    for s in range(1, S + 1):
        score[0, s] = -gap * s
        move[0, s] = 2
    for gi in range(1, G + 1):
        preds = [int(r) for r in flat[int(off[gi - 1]):int(off[gi])]]
        best_r, best_v = preds[0], score[preds[0], 0]
        for r in preds[1:]:
            if score[r, 0] > best_v:
                best_v, best_r = score[r, 0], r
        score[gi, 0] = best_v - gap
        move[gi, 0] = 1
        pred_g[gi, 0] = best_r
    for gi in range(1, G + 1):
        gbase = int(base[gi - 1])
        preds = [int(r) for r in flat[int(off[gi - 1]):int(off[gi])]]
        for s in range(1, S + 1):
            c = int(codes[s - 1])
            sub = match if (gbase == c and 1 <= gbase <= 4) else -mismatch
            best, bmove, brow = NEG, 2, preds[0]
            for r in preds:
                diag = score[r, s - 1] + sub
                if diag > best:
                    best, bmove, brow = diag, 0, r
                dele = score[r, s] - gap
                if dele > best:
                    best, bmove, brow = dele, 1, r
            ins = score[gi, s - 1] - gap
            if ins > best:
                best, bmove, brow = ins, 2, gi
            score[gi, s], move[gi, s], pred_g[gi, s] = best, bmove, brow
    return score, move, pred_g


def test_poa_fill_batch_orchestration_reproduces_consensus():
    """spec + kernel-recurrence fill + fold + round orchestration == host POA.

    This is the GPU cudapoa path with the CUDA fill swapped for its exact NumPy
    twin, so a green result means the device path is bit-identical to
    ``poa_consensus`` for integer scores (float32 stays exact) — everything but
    the RawKernel launch itself is covered here.
    """
    import numpy as np

    match, mismatch, gap = 2.0, 4.0, 4.0
    groups = _random_poa_groups(seed=23, n_groups=10)

    for group in groups:
        # Mirror _poa_consensus_batch_gpu's host orchestration exactly.
        seqs = [c for c in (gw._as_codes(s) for s in group) if len(c) > 0]
        seqs.sort(key=len, reverse=True)
        graph = gw._PoaGraph()
        if seqs:
            gw._poa_add_first(graph, seqs[0], weight=1)
        for r in range(1, len(seqs)):
            spec = gw._poa_fill_spec(graph, seqs[r])
            fill = _poa_fill_scalar(spec, match, mismatch, gap)
            gw._poa_fold_from_fill(graph, spec, fill, weight=1)
        if len(graph) == 0:
            got = ""
        else:
            path = gw._poa_consensus_path(graph)
            got = gw._decode(np.asarray([graph.base[n] for n in path], dtype=np.int8))
        assert got == gw.poa_consensus(group, match=match, mismatch=mismatch, gap=gap)


def test_consensus_batch_matches_per_group():
    """ConsensusPolisher.consensus_batch == per-group consensus (the seam
    _polish_loci relies on), including below-min-depth echo and empty groups."""
    from graphmambaformer.alignment import ConsensusPolisher

    polisher = ConsensusPolisher(min_depth=2)
    groups = [
        ["ACGTACGTAC", "ACGTACGTAC", "ACGTTCGTAC"],
        ["TTGGCCAA"],                 # below min_depth -> echo unchanged
        ["GGGGCCCCAAAA", "GGGGCCTCAAAA", "GGGGCCCCAAAA", "GGGACCCCAAAA"],
        [],                          # empty -> ""
    ]
    batch = polisher.consensus_batch(groups)
    per = [polisher.consensus(g) for g in groups]
    assert len(batch) == len(groups)
    for b, p in zip(batch, per):
        assert (b.consensus, b.depth, b.backend) == (p.consensus, p.depth, p.backend)


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


def test_gpu_kmer_query_batch_matches_per_read_query():
    """Batched cudamapper seeding must reproduce per-read ``query()`` exactly.

    This exercises the risky segment-and-split math on CPU: ``GPUKmerIndex``
    runs on CPU too (only the ``_seed_reads`` dispatcher gates on CUDA), so the
    GPU batched lookup is validated without a GPU. On the A100 the same code is
    additionally cross-checked at runtime by the verify-then-trust warm-up.
    """
    import numpy as np

    from graphmambaformer.alignment.seeding import (
        SeedingEngine,
        encode_bases,
        reverse_complement_codes,
    )
    from graphmambaformer.config import SeedingConfig

    reference = _random_seq(1200, seed=11)
    seeder = SeedingEngine(
        SeedingConfig(modes=("gpu_kmer",), kmer=13, window=5), device="cpu"
    )
    bundle = seeder.build_indices(reference)
    index = bundle.indices["gpu_kmer"]

    # Exact substrings, their reverse-complements, a near-miss random read, and
    # an empty input — covering the zero-length and no-hit branches.
    reads = [reference[100:260], reference[700:830], _random_seq(150, seed=99)]
    code_arrays = []
    for r in reads:
        fwd = encode_bases(r)
        code_arrays.append(fwd)
        code_arrays.append(reverse_complement_codes(fwd))
    code_arrays.append(encode_bases(""))

    batched = gw._gpu_kmer_query_batch(index, code_arrays)
    assert len(batched) == len(code_arrays)
    for codes, (rp, fp, ln) in zip(code_arrays, batched):
        e_rp, e_fp, e_ln = index.query(codes)
        got = sorted(zip(rp.tolist(), fp.tolist(), ln.tolist()))
        exp = sorted(
            zip(
                np.asarray(e_rp).tolist(),
                np.asarray(e_fp).tolist(),
                np.asarray(e_ln).tolist(),
            )
        )
        assert got == exp, "batched lookup diverged from per-read query()"


def test_global_align_batch_cpu_matches_per_pair():
    """On CPU, global_align_batch must equal per-pair global_align (incl. empties)."""
    rng = random.Random(4)
    pairs = []
    for _ in range(24):
        q = _random_seq(rng.randint(0, 60), rng.randint(0, 10_000))
        t = _random_seq(rng.randint(0, 60), rng.randint(0, 10_000))
        pairs.append((q, t))
    got = gw.global_align_batch(pairs)
    exp = [gw.global_align(q, t) for q, t in pairs]
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert g.cigar == e.cigar and abs(g.score - e.score) <= 1e-6


def test_traceback_ptr_reproduces_global_align():
    """The GPU path's host traceback + flat ptr layout reproduce global_align.

    The CUDA ``global_affine`` kernel only fills the ``(m+1, n+1)`` pointer matrix
    (flattened with stride ``maxN+1``); the CIGAR is rebuilt on the host by
    ``_traceback_ptr``. Here the *exact same* integer-scored Gotoh recurrence the
    kernel runs is reproduced in NumPy, then handed to ``_traceback_ptr`` — so
    the pointer convention, flat indexing, tie-break, and traceback are all
    validated on CPU (only the float32-vs-float64 arithmetic differs on-device,
    and integer scores are exact in both).
    """
    import numpy as np

    match, mismatch, go, ge = 2.0, 4.0, 6.0, 2.0
    go_ge = go + ge
    NEG = -1e30
    rng = random.Random(9)
    for trial in range(60):
        q = _random_seq(rng.randint(1, 45), 1000 + trial)
        t = _random_seq(rng.randint(1, 45), 5000 + trial)
        qc, tc = gw._as_codes(q), gw._as_codes(t)
        m, n = len(qc), len(tc)
        rowW = n + 1
        ptr = np.zeros((m + 1) * rowW, dtype=np.int8)
        h_prev = np.empty(rowW)
        f_prev = np.empty(rowW)
        h_prev[0] = 0.0
        f_prev[0] = NEG
        for j in range(1, n + 1):
            h_prev[j] = -(go + j * ge)
            f_prev[j] = NEG
            ptr[j] = 1
        for i in range(1, m + 1):
            a = int(qc[i - 1])
            h_cur = np.empty(rowW)
            f_cur = np.empty(rowW)
            h_cur[0] = -(go + i * ge)
            f_cur[0] = NEG
            ptr[i * rowW] = 2
            e_prev = NEG
            for j in range(1, n + 1):
                b = int(tc[j - 1])
                sub = match if (a == b and 1 <= a <= 4) else -mismatch
                diag = h_prev[j - 1] + sub
                e = max(h_cur[j - 1] - go_ge, e_prev - ge)
                f = max(h_prev[j] - go_ge, f_prev[j] - ge)
                best, p = diag, 0
                if e > best:
                    best, p = e, 1
                if f > best:
                    best, p = f, 2
                h_cur[j] = best
                f_cur[j] = f
                ptr[i * rowW + j] = p
                e_prev = e
            h_prev, f_prev = h_cur, f_cur

        ops, nm, nx, ni, nd = gw._traceback_ptr(ptr, qc, tc, m, n, n)
        exp = gw.global_align(q, t)
        assert gw._merge_ops(ops) == exp.cigar, (q, t)
        assert abs(float(h_prev[n]) - exp.score) <= 1e-6
        assert (nm, nx, ni, nd) == (
            exp.n_match, exp.n_mismatch, exp.n_insertion, exp.n_deletion
        )


def test_seed_reads_cpu_falls_back_to_per_read():
    """On CPU (no CUDA index) ``_seed_reads`` yields identical per-read anchors."""
    from graphmambaformer.alignment.seeding import SeedingEngine
    from graphmambaformer.config import SeedingConfig

    reference = _random_seq(900, seed=5)
    reads = [reference[200:340], reference[500:600]]
    seeder = SeedingEngine(
        SeedingConfig(modes=("gpu_kmer",), kmer=13, window=5), device="cpu"
    )
    bundle = seeder.build_indices(reference)

    got = gw._seed_reads(seeder, bundle, reads)
    ref = [seeder.seed_read(r, bundle) for r in reads]
    assert gw._anchor_sets_equal(got, ref)


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


def test_extend_cudaaligner_batch_matches_per_chain():
    """The batched extension path (what the pipeline now calls) must return
    exactly the per-chain ``_extend_cudaaligner`` results, in order.

    Validates the wiring on CPU, where ``global_align_batch`` degrades to the
    per-pair host reference; on the CUDA tier the same call is one GPU Gotoh
    launch, verified bit-for-bit against this reference.
    """
    import numpy as np

    from graphmambaformer.alignment.seeding import (
        encode_bases,
        reverse_complement_codes,
    )
    from graphmambaformer.alignment.types import Chain

    reference = _random_seq(1200, seed=101)
    read = reference[300:300 + 180]
    engine = _fast_pipeline(algorithm="cudaaligner").extender

    forward = encode_bases(read)
    reverse = reverse_complement_codes(forward)
    ref_codes = encode_bases(reference)
    read_len, ref_len = len(forward), len(ref_codes)

    # Synthetic chains at different loci / strands / net-indels. Only the
    # coordinates and strand feed the cudaaligner window; anchor_idx/score do
    # not, so an empty index array is fine here.
    def _chain(strand, rs, re_, fs, fe):
        return Chain(anchor_idx=np.array([], dtype=np.int64), score=0.0,
                     strand=strand, read_start=rs, read_end=re_,
                     ref_start=fs, ref_end=fe)

    chains = [
        _chain(1, 0, 180, 300, 480),    # net indel 0
        _chain(1, 10, 170, 305, 470),   # target shorter than read
        _chain(-1, 0, 180, 800, 995),   # reverse strand, target longer
        _chain(1, 0, 180, 1150, 1200),  # window clipped by ref end
    ]

    batched = engine._extend_cudaaligner_batch(
        chains, forward, reverse, ref_codes, read_len, ref_len
    )
    per_chain = [
        engine._extend_cudaaligner(c, forward, reverse, ref_codes, read_len, ref_len)
        for c in chains
    ]

    assert len(batched) == len(per_chain) == len(chains)
    for b, s in zip(batched, per_chain):
        assert b.cigar == s.cigar
        assert b.score == s.score
        assert (b.read_start, b.read_end, b.ref_start, b.ref_end) == (
            s.read_start, s.read_end, s.ref_start, s.ref_end)
        assert (b.n_match, b.n_mismatch, b.n_insertion, b.n_deletion) == (
            s.n_match, s.n_mismatch, s.n_insertion, s.n_deletion)

    # Empty chain list is a no-op (matches the per-chain comprehension).
    assert engine._extend_cudaaligner_batch(
        [], forward, reverse, ref_codes, read_len, ref_len
    ) == []


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
