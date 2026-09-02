"""Algorithmic cross-checks against independent brute-force references.

The other suites verify that the stages are *self-consistent* (a CIGAR consumes
the bases it claims, a chain is collinear). These tests verify they are
*correct*, by recomputing the same answer a second, obviously-right way:

- suffix array vs. Python's own sort of the suffixes
- FM-index counting / locating / SMEMs vs. naive substring scanning
- minimizer selection vs. an explicit per-window minimum
- chaining DP vs. a textbook O(n^2) loop
- banded Smith-Waterman vs. an unbanded full-matrix affine DP
- WFA vs. a Levenshtein edit-distance matrix

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import numpy as np
import torch

from graphmambaformer.alignment.chaining import (
    AffineChainer,
    ChainingContext,
    GraphDistanceOracle,
    chain_dp_batched,
    chain_dp_numpy,
)
from graphmambaformer.alignment.extension import (
    WavefrontAligner,
    banded_affine_sw_batch,
    traceback_banded,
)
from graphmambaformer.alignment.seeding import (
    DeBruijnIndex,
    FMIndex,
    FuzzySeedIndex,
    GPUKmerIndex,
    MinimizerIndex,
    MultiplexDBG,
    encode_bases,
    minimizer_mask,
    pack_kmers,
    reverse_complement_codes,
    suffix_array,
)
from graphmambaformer.alignment.types import AnchorSet, Chain
from graphmambaformer.config import ChainingConfig, ExtensionConfig

RNG = np.random.default_rng(3)
BASES = "ACGT"


def random_seq(n: int) -> str:
    return "".join(RNG.choice(list(BASES), size=n))


# --------------------------------------------------------------------------- #
# Stage 1 primitives
# --------------------------------------------------------------------------- #
def _sa_with_sentinel(seq: str) -> np.ndarray:
    """Suffix array of ``seq``. The caller owns appending the sentinel."""
    codes = np.concatenate([encode_bases(seq), [0]]).astype(np.int8)
    return suffix_array(codes)


def test_suffix_array_matches_python_sort():
    for length in (1, 2, 17, 200):
        seq = random_seq(length)
        got = _sa_with_sentinel(seq)
        # "\x00" stands in for the sentinel, which sorts below every base, so the
        # empty suffix at index `length` comes first.
        expected = sorted(range(length + 1), key=lambda i: seq[i:] + "\x00")
        assert list(got) == expected, (seq, list(got), expected)

    # A highly repetitive string is where naive suffix sorting degenerates.
    repeat = "ACGT" * 25
    expected = sorted(range(len(repeat) + 1), key=lambda i: repeat[i:] + "\x00")
    assert list(_sa_with_sentinel(repeat)) == expected
    # A homopolymer is the worst case for prefix doubling.
    poly = "A" * 40
    expected = sorted(range(len(poly) + 1), key=lambda i: poly[i:] + "\x00")
    assert list(_sa_with_sentinel(poly)) == expected
    print("suffix array matches Python's suffix sort (random, repetitive, homopolymer)")


def test_fmindex_count_and_locate_match_naive_search():
    text = random_seq(400)
    fm = FMIndex(encode_bases(text))

    def naive(pattern: str) -> list[int]:
        return [i for i in range(len(text) - len(pattern) + 1)
                if text[i : i + len(pattern)] == pattern]

    patterns = [text[100:110], text[0:5], text[-7:], "ACGTACGTACGTACGT", "AAAAAAAA"]
    patterns += [random_seq(6) for _ in range(10)]
    for pattern in patterns:
        codes = encode_bases(pattern)
        expected = naive(pattern)
        lo, hi = fm.count(codes)
        assert hi - lo == len(expected), (pattern, hi - lo, len(expected))
        got = sorted(int(x) for x in fm.occurrences(codes))
        assert got == expected, (pattern, got, expected)
    print(f"FM-index count + locate match naive search on {len(patterns)} patterns")


def test_fmindex_smems_are_maximal_and_real():
    text = random_seq(500)
    fm = FMIndex(encode_bases(text))
    # A read with two exact blocks separated by a mismatch region.
    read = text[50:110] + random_seq(20) + text[300:360]
    read_codes = encode_bases(read)

    starts, lengths, _ = fm.smems(read_codes, min_len=12)
    assert len(starts) > 0, "no SMEMs found"

    for start, length in zip(starts, lengths):
        block = read[start : start + length]
        assert block in text, (start, length, block)
        # Right-maximal: extending one base to the right must break the match.
        if start + length < len(read):
            assert read[start : start + length + 1] not in text
        # Left-maximal: extending one base to the left must break it too.
        if start > 0:
            assert read[start - 1 : start + length] not in text
    print(f"{len(starts)} SMEMs, each a real match and maximal on both sides")


def test_minimizer_mask_matches_explicit_windows():
    for window in (1, 3, 5, 10):
        hashes = RNG.integers(0, 1000, size=60)
        mask = minimizer_mask(hashes, window)
        # A position is selected iff it is the minimum of some window of `window`
        # consecutive positions containing it.
        expected = np.zeros(len(hashes), dtype=bool)
        for start in range(len(hashes) - window + 1):
            block = hashes[start : start + window]
            expected[start + int(block.argmin())] = True
        assert np.array_equal(mask, expected), window
    print("minimizer selection matches explicit per-window minima (w=1,3,5,10)")


def test_kmer_packing_round_trips():
    seq = random_seq(50)
    k = 8
    packed, valid = pack_kmers(encode_bases(seq), k)
    assert len(packed) == len(valid) == len(seq) - k + 1
    assert valid.all(), "no ambiguous bases, so every window is valid"

    # Unpack each 2-bit-encoded k-mer and compare to the source substring.
    for pos, value in enumerate(packed):
        digits = [(int(value) >> (2 * (k - 1 - j))) & 3 for j in range(k)]
        assert "".join(BASES[d] for d in digits) == seq[pos : pos + k], pos

    # A window containing N must be flagged invalid rather than silently packed.
    with_n = seq[:10] + "N" + seq[11:]
    _, valid_n = pack_kmers(encode_bases(with_n), k)
    assert not valid_n[max(0, 10 - k + 1) : 11].any(), "N windows must be invalid"
    assert valid_n[:3].all(), "windows clear of the N stay valid"
    print("2-bit k-mer packing round-trips; N-containing windows flagged invalid")


def test_reverse_complement_is_an_involution():
    seq = random_seq(64)
    codes = encode_bases(seq)
    once = reverse_complement_codes(codes)
    assert np.array_equal(reverse_complement_codes(once), codes)
    expected = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    assert np.array_equal(once, encode_bases(expected))
    print("reverse complement matches the string translation and is an involution")


def test_dbg_projects_node_offsets_and_skips_unprojected_nodes():
    nodes = [encode_bases("AACCGGTT"), encode_bases("TTTTAAAA")]
    dbg = DeBruijnIndex(nodes, k=4, node_ref_start=[100, -1])
    read_pos, ref_pos, length, node = dbg.query(encode_bases("AACCGGTTTTTTAAAA"))
    assert len(read_pos) > 0
    assert (node == 0).all(), node
    assert (ref_pos >= 100).all() and (ref_pos < 108).all(), ref_pos
    assert (length == 4).all()


def test_fuzzy_spaced_seed_ignores_dont_care_mismatch():
    index = FuzzySeedIndex(encode_bases("ACG"), pattern="101")
    read_pos, ref_pos, length = index.query(encode_bases("ATG"))
    assert np.array_equal(read_pos, [0])
    assert np.array_equal(ref_pos, [0])
    assert np.array_equal(length, [3])


def test_fuzzy_pack_spaced_kmers_care_bits_and_n_bases():
    from graphmambaformer.alignment.seeding import pack_spaced_kmers, validate_spaced_pattern

    pattern = "111010010100110111"
    care = [i for i, ch in enumerate(pattern) if ch == "1"]
    ref = "ACGTACGTACGTACGTAC"
    packed, valid = pack_spaced_kmers(encode_bases(ref), pattern)
    assert valid.tolist() == [True]
    manual = 0
    for i in care:
        manual = manual * 4 + "ACGT".index(ref[i])
    assert int(packed[0]) == manual

    noisy = list(ref)
    for i, ch in enumerate(pattern):
        if ch == "0":
            noisy[i] = "A" if noisy[i] != "A" else "T"
    assert pack_spaced_kmers(encode_bases("".join(noisy)), pattern)[0][0] == packed[0]

    broken = list(ref)
    broken[care[0]] = "A" if broken[care[0]] != "A" else "T"
    assert pack_spaced_kmers(encode_bases("".join(broken)), pattern)[0][0] != packed[0]

    n_care = list(ref)
    n_care[care[2]] = "N"
    assert not pack_spaced_kmers(encode_bases("".join(n_care)), pattern)[1][0]
    n_dont = list(ref)
    n_dont[pattern.index("0")] = "N"
    assert pack_spaced_kmers(encode_bases("".join(n_dont)), pattern)[1][0]

    for bad in ("", "000", "121", "11x"):
        try:
            validate_spaced_pattern(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_fuzzy_recovers_when_exact_contiguous_kmers_fail():
    """Spaced seeds hit through don't-care mismatches; exact 15-mers do not."""
    from graphmambaformer.config import SeedingConfig
    from graphmambaformer.alignment.seeding import SeedingEngine

    pattern = "111010010100110111"
    care = {i for i, ch in enumerate(pattern) if ch == "1"}
    span = len(pattern)
    ref = random_seq(240)
    start = 40
    truth = ref[start : start + 120]
    read = list(truth)
    # Keep only the compared bases of one spaced seed at offset 0; mutate the rest
    # so every contiguous 15-mer on the true diagonal is broken.
    for pos in range(len(read)):
        if pos in care:
            continue
        read[pos] = next(base for base in BASES if base != read[pos])
    read = "".join(read)
    assert sum(a != b for a, b in zip(read, truth)) >= span - len(care)

    exact = SeedingEngine(SeedingConfig(modes=("minimizer",), kmer=15, window=1, max_occ=50))
    fuzzy = SeedingEngine(
        SeedingConfig(modes=("fuzzy",), spaced_pattern=pattern, max_occ=50)
    )
    exact_anchors = exact.seed_read(read, exact.build_indices(ref))
    fuzzy_anchors = fuzzy.seed_read(read, fuzzy.build_indices(ref))

    exact_diag = int(
        ((exact_anchors.strand == 1) & (np.abs(exact_anchors.diagonal - start) <= 2)).sum()
    ) if len(exact_anchors) else 0
    fuzzy_diag = int(
        ((fuzzy_anchors.strand == 1) & (np.abs(fuzzy_anchors.diagonal - start) <= 2)).sum()
    ) if len(fuzzy_anchors) else 0
    assert exact_diag == 0, exact_diag
    assert fuzzy_diag > 0, fuzzy_diag

    # Surviving fuzzy anchors match only the compared ('1') positions.
    rc, fc = encode_bases(read), encode_bases(ref)
    matched = False
    for i in range(len(fuzzy_anchors)):
        if fuzzy_anchors.strand[i] != 1:
            continue
        if abs(int(fuzzy_anchors.diagonal[i]) - start) > 2:
            continue
        rp = int(fuzzy_anchors.read_pos[i])
        fp = int(fuzzy_anchors.ref_pos[i])
        ln = int(fuzzy_anchors.length[i])
        assert ln == span
        assert not np.array_equal(rc[rp : rp + ln], fc[fp : fp + ln])
        assert all(rc[rp + off] == fc[fp + off] for off in care if off < ln)
        matched = True
    assert matched


def test_fuzzy_seeds_are_not_merged_into_pseudo_mems():
    """Overlapping spaced seeds must keep pattern span, not grow like exact MEMs."""
    from graphmambaformer.config import SeedingConfig
    from graphmambaformer.alignment.seeding import SeedingEngine, source_id

    pattern = "111010010100110111"
    ref = random_seq(300)
    read = ref[20:120]
    eng = SeedingEngine(
        SeedingConfig(modes=("fuzzy",), spaced_pattern=pattern, both_strands=False, max_occ=50)
    )
    anchors = eng.seed_read(read, eng.build_indices(ref))
    assert len(anchors) > 1
    assert (anchors.length == len(pattern)).all(), set(map(int, anchors.length))
    assert (anchors.source == source_id("fuzzy")).all()


def test_fuzzy_reverse_strand_and_default_modes():
    from graphmambaformer.config import SeedingConfig
    from graphmambaformer.alignment.seeding import SeedingEngine

    def revcomp(seq: str) -> str:
        return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    ref = random_seq(400)
    start = 100
    eng = SeedingEngine(SeedingConfig(modes=("fuzzy",), max_occ=50))
    bundle = eng.build_indices(ref)
    fwd = eng.seed_read(ref[start : start + 80], bundle)
    rev = eng.seed_read(revcomp(ref[start : start + 80]), bundle)
    assert ((fwd.strand == 1) & (np.abs(fwd.diagonal - start) <= 2)).any()
    assert ((rev.strand == -1) & (np.abs(rev.diagonal - start) <= 2)).any()
    assert "fuzzy" in SeedingConfig().modes


def test_fuzzy_end_to_end_pipeline_locus():
    from graphmambaformer.config import PipelineConfig
    from graphmambaformer.alignment import build_pipeline

    rng = np.random.default_rng(3)
    ref = "".join(rng.choice(list(BASES), size=2500))
    cfg = PipelineConfig(mode="fast")
    cfg.seeding.modes = ("fuzzy",)
    cfg.seeding.max_occ = 50
    cfg.chaining.min_chain_score = 1.0
    pipe = build_pipeline(cfg, model=None, device="cpu")
    reference = pipe.build_reference(ref)
    hits = 0
    n = 16
    for i in range(n):
        start = int(rng.integers(0, len(ref) - 160))
        read = list(ref[start : start + 160])
        for j in range(len(read)):
            if rng.random() < 0.12:
                read[j] = next(base for base in BASES if base != read[j])
        alignments, _ = pipe.align(["".join(read)], reference, [f"r{i}"])
        record = alignments[0].primary
        if record and record.is_mapped and abs(record.ref_start - start) <= 40:
            hits += 1
    assert hits / n >= 0.8, hits


def test_multiplex_dbg_falls_back_to_shorter_k_near_errors():
    ref = random_seq(80)
    read = list(ref)
    read[40] = next(base for base in BASES if base != read[40])
    index = MultiplexDBG(encode_bases(ref), kmers=(3, 7), max_occ=20)
    _, _, lengths = index.query(encode_bases("".join(read)))
    assert (lengths == 7).any(), lengths
    assert (lengths == 3).any(), lengths


def test_gpu_kmer_cpu_fallback_matches_minimizer_index():
    ref = random_seq(120)
    read = ref[15:95]
    expected = MinimizerIndex(encode_bases(ref), k=7, window=4).query(
        encode_bases(read)
    )
    got = GPUKmerIndex(
        encode_bases(ref), k=7, window=4, device="cpu"
    ).query(encode_bases(read))
    expected_rows = sorted(zip(*[values.tolist() for values in expected]))
    got_rows = sorted(zip(*[values.tolist() for values in got]))
    assert got_rows == expected_rows


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def _chain_dp_bruteforce(read_end, ref_end, weight, cfg):
    """Textbook O(n^2) chaining DP, written for clarity rather than speed."""
    n = len(read_end)
    f = np.zeros(n)
    parent = np.full(n, -1, dtype=np.int64)
    lookback = max(1, cfg.max_lookback)
    for i in range(n):
        best_score, best_j = weight[i], -1
        for j in range(max(0, i - lookback), i):
            dq = read_end[i] - read_end[j]
            dr = ref_end[i] - ref_end[j]
            if dq <= 0 or dr <= 0 or dq > cfg.max_gap or dr > cfg.max_gap:
                continue
            gap = abs(dr - dq)
            penalty = 0.0
            if gap > 0:
                penalty = (
                    cfg.gap_open
                    + cfg.gap_extend * gap
                    + cfg.log_coeff * np.log2(gap + 1.0)
                )
            score = f[j] + min(dq, dr, weight[i]) - penalty
            if score > best_score:
                best_score, best_j = score, j
        f[i], parent[i] = best_score, best_j
    return f, parent


def test_chaining_dp_matches_bruteforce():
    cfg = ChainingConfig()
    for trial in range(6):
        n = int(RNG.integers(4, 25))
        read_end = np.sort(RNG.integers(1, 400, size=n)).astype(np.int64)
        ref_end = np.sort(RNG.integers(1, 400, size=n)).astype(np.int64)
        weight = RNG.integers(5, 40, size=n).astype(np.float64)

        f, parent = chain_dp_numpy(read_end, ref_end, weight, cfg)
        exp_f, exp_parent = _chain_dp_bruteforce(read_end, ref_end, weight, cfg)
        assert np.allclose(f, exp_f), (trial, f, exp_f)
        # Ties can pick a different but equally-scoring predecessor, so compare
        # the achieved score rather than the identity of the parent.
        for i in range(n):
            if parent[i] != exp_parent[i]:
                assert np.isclose(f[i], exp_f[i]), (trial, i)
    print("chaining DP scores match the brute-force O(n^2) reference on 6 cases")


def test_chaining_prefers_collinear_anchors():
    """Collinear anchors must outscore the same anchors with a large indel."""
    cfg = ChainingConfig()
    weight = np.array([30.0, 30.0])

    collinear_f, _ = chain_dp_numpy(
        np.array([30, 60]), np.array([30, 60]), weight, cfg
    )
    shifted_f, _ = chain_dp_numpy(
        np.array([30, 60]), np.array([30, 300]), weight, cfg
    )
    assert collinear_f[-1] > shifted_f[-1], (collinear_f, shifted_f)
    print(f"collinear chain scores {collinear_f[-1]:.1f} > "
          f"{shifted_f[-1]:.1f} with a 240 bp gap")


def test_graph_distance_oracle_and_bonus_decay():
    cfg = ChainingConfig(max_lookback=4, graph_bonus=6.0, graph_max_hops=3)
    oracle = GraphDistanceOracle(
        np.array([[0, 1], [1, 2]], dtype=np.int64), num_nodes=3, max_hops=3
    )
    hops, index = oracle.hop_matrix([0, 1, 2])
    assert hops[index[0], index[0]] == 0
    assert hops[index[0], index[1]] == 1
    assert hops[index[0], index[2]] == 2

    anchors = AnchorSet.from_lists(
        read_pos=[0, 20, 40],
        ref_pos=[0, 20, 40],
        length=[10, 10, 10],
        strand=[1, 1, 1],
        node_id=[0, 1, 2],
        read_len=60,
        ref_len=60,
    )
    bonus = AffineChainer(cfg)._graph_bonus(
        anchors, ChainingContext(oracle=oracle)
    )
    assert bonus is not None
    assert np.isclose(bonus[1, 3], 4.0)  # one hop: 6 * (1 - 1/3)
    assert np.isclose(bonus[2, 2], 2.0)  # two hops
    assert np.isclose(bonus[2, 3], 4.0)  # one hop


def test_learned_transition_guidance_is_confidence_gated():
    cfg = ChainingConfig(
        max_lookback=4,
        graph_bonus=0.0,
        gnn_transition_bonus=2.0,
    )
    anchors = AnchorSet.from_lists(
        read_pos=[0, 20, 40],
        ref_pos=[0, 20, 40],
        length=[10, 10, 10],
        strand=[1, 1, 1],
        read_len=60,
        ref_len=60,
    )

    def key(i):
        return (
            int(anchors.read_pos[i]),
            int(anchors.ref_pos[i]),
            int(anchors.length[i]),
            int(anchors.strand[i]),
        )

    transitions = {(key(0), key(1)): 0.9, (key(1), key(2)): 0.1}
    chainer = AffineChainer(cfg)
    guided = chainer._graph_bonus(
        anchors,
        ChainingContext(trust_neural=True, learned_transitions=transitions),
    )
    fallback = chainer._graph_bonus(
        anchors,
        ChainingContext(trust_neural=False, learned_transitions=transitions),
    )
    assert guided is not None
    assert guided[1, 3] > 0.0
    assert guided[2, 3] < 0.0
    assert fallback is None


def test_batched_chaining_matches_numpy_with_ragged_rows():
    cfg = ChainingConfig(max_lookback=4)
    read_end = torch.tensor([[10, 20, 30], [8, 18, 0]], dtype=torch.float32)
    ref_end = torch.tensor([[10, 21, 32], [8, 20, 0]], dtype=torch.float32)
    weight = torch.tensor([[10.5, 9.25, 8.75], [7.5, 8.25, 0.0]])
    counts = torch.tensor([3, 2])
    got, parent = chain_dp_batched(read_end, ref_end, weight, counts, cfg)
    for row, count in enumerate(counts.tolist()):
        expected, expected_parent = chain_dp_numpy(
            read_end[row, :count].numpy(),
            ref_end[row, :count].numpy(),
            weight[row, :count].numpy(),
            cfg,
        )
        assert np.allclose(got[row, :count].numpy(), expected)
        assert np.array_equal(parent[row, :count].numpy(), expected_parent)


# --------------------------------------------------------------------------- #
# Stage 2 — AGNES adaptive (confidence-gated) seed scoring
# --------------------------------------------------------------------------- #
def _anchor_set_with_scores(scores: np.ndarray) -> AnchorSet:
    """A minimal forward-strand AnchorSet carrying the given neural seed scores."""
    n = len(scores)
    anchors = AnchorSet.from_lists(
        read_pos=np.arange(n, dtype=np.int64) * 20,
        ref_pos=np.arange(n, dtype=np.int64) * 20,
        length=np.full(n, 15, dtype=np.int64),
        strand=np.ones(n, dtype=np.int8),
        read_len=n * 20 + 15,
        ref_len=n * 20 + 15,
    )
    anchors.score[:] = np.asarray(scores, dtype=np.float32)
    return anchors


def test_confidence_gate_falls_back_when_scores_are_flat():
    """A flat / undecisive score distribution must leave the weights untouched."""
    chainer = AffineChainer(ChainingConfig())
    anchors = _anchor_set_with_scores(np.full(12, 0.5, dtype=np.float32))

    conf = chainer._seed_confidence(anchors.score)
    gate = chainer._confidence_gate(anchors.score, np.isfinite(anchors.score))
    weights = chainer._weights(anchors, None)

    assert conf <= ChainingConfig().confidence_threshold, conf
    assert np.allclose(gate, 1.0), gate
    # Pure length-based chaining is recovered exactly (length == 15 per anchor).
    assert np.allclose(weights, anchors.length.astype(np.float64)), weights
    print(f"flat seed scores -> confidence {conf:.2f} <= tau, gate is identity")


def test_confidence_gate_trusts_a_decisively_separated_distribution():
    """When the classifier is decisive, good seeds are up-weighted and bad ones down."""
    chainer = AffineChainer(ChainingConfig())
    scores = np.array([0.97, 0.95, 0.96, 0.98, 0.03, 0.05, 0.02, 0.04,
                       0.96, 0.97, 0.02, 0.03], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)

    conf = chainer._seed_confidence(anchors.score)
    gate = chainer._confidence_gate(anchors.score, np.isfinite(anchors.score))

    assert conf > ChainingConfig().confidence_threshold, conf
    good, bad = scores > 0.5, scores < 0.5
    assert (gate[good] > 1.0).all(), gate[good]
    assert (gate[bad] < 1.0).all(), gate[bad]
    print(f"separated seed scores -> confidence {conf:.2f} > tau; "
          f"good gate ~{gate[good].mean():.2f}, bad gate ~{gate[bad].mean():.2f}")


def test_confidence_gate_ignores_degenerate_anchor_counts():
    """Below the minimum anchor count the metric is meaningless -> pure DP."""
    cfg = ChainingConfig()
    chainer = AffineChainer(cfg)
    # Decisively separated, but too few anchors to be trusted.
    scores = np.array([0.97, 0.02, 0.96], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    assert len(anchors) < cfg.min_confidence_anchors

    gate = chainer._confidence_gate(anchors.score, np.isfinite(anchors.score))
    assert np.allclose(gate, 1.0), gate
    print(f"{len(anchors)} anchors (< {cfg.min_confidence_anchors}) -> gate is identity")


def test_adaptive_scoring_changes_chain_choice_when_confident():
    """End to end: a confident seed score should pull the chain to the right locus.

    Two collinear diagonals of equal geometric weight compete. Without scores the
    DP cannot separate them; with a decisively separated score distribution that
    favours the second diagonal, the adaptive gate makes it the primary chain.
    """
    cfg = ChainingConfig(min_chain_score=1.0, min_confidence_anchors=4)
    chainer = AffineChainer(cfg)

    # Diagonal A at ref==read; diagonal B shifted by +500 bp. Four anchors each.
    read_pos = np.array([0, 40, 80, 120, 0, 40, 80, 120], dtype=np.int64)
    ref_pos = np.array([0, 40, 80, 120, 500, 540, 580, 620], dtype=np.int64)
    anchors = AnchorSet.from_lists(
        read_pos=read_pos,
        ref_pos=ref_pos,
        length=np.full(8, 15, dtype=np.int64),
        strand=np.ones(8, dtype=np.int8),
        read_len=200,
        ref_len=700,
    )
    # Confidently favour diagonal B (the second four anchors).
    anchors.score[:] = np.array([0.05, 0.03, 0.04, 0.02,
                                 0.97, 0.98, 0.96, 0.97], dtype=np.float32)

    chains = chainer.chain(anchors)
    assert chains, "expected at least one chain"
    primary = chains[0]
    ref_starts = anchors.ref_pos[primary.anchor_idx]
    assert ref_starts.min() >= 500, (ref_starts, "primary should be diagonal B")
    print(f"adaptive gate steered the primary chain to ref {ref_starts.min()}-"
          f"{anchors.ref_end[primary.anchor_idx].max()}")


def test_seed_confidence_matches_hand_computed_formula():
    """``(μ_high - μ_low) / σ`` must match a direct NumPy evaluation of AGNES."""
    cfg = ChainingConfig()
    chainer = AffineChainer(cfg)
    scores = np.array([0.9, 0.85, 0.1, 0.15, 0.92, 0.08, 0.5, 0.55], dtype=np.float64)

    high = scores[scores > cfg.high_confidence_prob]
    low = scores[scores < cfg.low_confidence_prob]
    expected = (high.mean() - low.mean()) / scores.std()
    got = chainer._seed_confidence(scores)
    assert np.isclose(got, expected), (got, expected)
    assert got > cfg.confidence_threshold
    print(f"confidence formula matches hand compute: {got:.4f} == {expected:.4f}")


def test_seed_confidence_empty_or_zero_spread_is_zero():
    chainer = AffineChainer(ChainingConfig())
    assert chainer._seed_confidence(np.zeros(0)) == 0.0
    assert chainer._seed_confidence(np.full(8, 0.8)) == 0.0  # σ ~ 0
    # Only mid-band scores: μ_high=μ_low=0 by convention when those sets are empty.
    mid = np.array([0.4, 0.5, 0.55, 0.45, 0.6, 0.35], dtype=np.float64)
    assert chainer._seed_confidence(mid) == 0.0
    print("empty / zero-spread / mid-only scores all report confidence 0")


def test_seed_confidence_one_sided_high_or_low():
    """Missing one tail zeros that mean; the other side still drives the metric."""
    cfg = ChainingConfig()
    chainer = AffineChainer(cfg)
    only_high = np.array([0.8, 0.9, 0.85, 0.95, 0.5, 0.55], dtype=np.float64)
    only_low = np.array([0.1, 0.2, 0.05, 0.15, 0.5, 0.55], dtype=np.float64)

    conf_high = chainer._seed_confidence(only_high)
    conf_low = chainer._seed_confidence(only_low)
    # μ_low = 0 when no low seeds -> (μ_high - 0) / σ > 0
    assert conf_high > 0.0, conf_high
    # μ_high = 0 when no high seeds -> (0 - μ_low) / σ < 0 (never clears τ)
    assert conf_low < 0.0, conf_low
    print(f"one-sided high conf={conf_high:.2f}; one-sided low conf={conf_low:.2f}")


def test_confidence_gate_logit_math_and_clamping():
    """Trusted gates must equal clip(1 + gain * logit(p)), including extremes."""
    cfg = ChainingConfig(
        confidence_threshold=0.0,  # force the gate open once count clears
        min_confidence_anchors=4,
        logit_gate_gain=0.25,
        logit_gate_min=0.1,
        logit_gate_max=3.0,
    )
    chainer = AffineChainer(cfg)
    # Mix mid + extremes so the gate is exercised at both clamps and the interior.
    scores = np.array([0.5, 0.9, 1e-8, 1.0 - 1e-8, 0.7, 0.3], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    gate = chainer._confidence_gate(anchors.score, np.isfinite(anchors.score))

    p = np.clip(scores.astype(np.float64), 1e-4, 1.0 - 1e-4)
    expected = np.clip(1.0 + cfg.logit_gate_gain * np.log(p / (1.0 - p)),
                       cfg.logit_gate_min, cfg.logit_gate_max)
    assert np.allclose(gate, expected), (gate, expected)
    assert gate.min() >= cfg.logit_gate_min - 1e-12
    assert gate.max() <= cfg.logit_gate_max + 1e-12
    # p=0.5 -> logit 0 -> gate exactly 1.
    assert np.isclose(gate[0], 1.0)
    print(f"logit gate matches hand compute; range [{gate.min():.3f}, {gate.max():.3f}]")


def test_confidence_gate_falls_back_at_exact_threshold():
    """AGNES uses ``conf > τ``; equality must fall back to the identity gate."""
    cfg = ChainingConfig(min_confidence_anchors=4)
    chainer = AffineChainer(cfg)
    # Keep one dtype end-to-end: float32 scores are what the seed head writes.
    scores = np.array([0.9, 0.1, 0.85, 0.15, 0.88, 0.12], dtype=np.float32)
    conf = chainer._seed_confidence(scores)
    assert conf > 0.0
    # Re-point τ to the measured confidence so the comparison is exactly ``<=``.
    chainer.cfg.confidence_threshold = float(conf)
    gate = chainer._confidence_gate(scores, np.ones(len(scores), dtype=bool))
    assert np.allclose(gate, 1.0), (conf, gate)
    # Bumping τ just below conf must open the gate.
    chainer.cfg.confidence_threshold = float(conf) - 1e-6
    gate_open = chainer._confidence_gate(scores, np.ones(len(scores), dtype=bool))
    assert not np.allclose(gate_open, 1.0), gate_open
    print(f"conf == τ ({conf:.4f}) falls back; conf > τ opens the gate")


def test_confidence_gate_counts_only_finite_scores():
    """NaN slots must not inflate the finite count past the degenerate guards."""
    cfg = ChainingConfig(min_confidence_anchors=5, max_confidence_anchors=1000)
    chainer = AffineChainer(cfg)
    # Four finite + many NaN: finite count is below the floor -> identity.
    scores = np.array([0.97, 0.02, 0.96, 0.03, np.nan, np.nan, np.nan, np.nan],
                      dtype=np.float32)
    finite = np.isfinite(scores)
    assert int(finite.sum()) < cfg.min_confidence_anchors
    gate = chainer._confidence_gate(scores, finite)
    assert np.allclose(gate, 1.0), gate
    # Non-finite positions stay at 1 even when the finite subset is trusted.
    scores_ok = np.array([0.97, 0.02, 0.96, 0.03, 0.95, 0.04, np.nan, np.nan],
                         dtype=np.float32)
    finite_ok = np.isfinite(scores_ok)
    gate_ok = chainer._confidence_gate(scores_ok, finite_ok)
    assert np.allclose(gate_ok[~finite_ok], 1.0), gate_ok
    assert not np.allclose(gate_ok[finite_ok], 1.0), gate_ok[finite_ok]
    print("NaN slots ignored for the count; unscored positions keep gate=1")


def test_confidence_gate_falls_back_above_max_anchors():
    """AGNES |V| > 1000 guard: too many anchors -> classical DP."""
    cfg = ChainingConfig(max_confidence_anchors=10, min_confidence_anchors=5)
    chainer = AffineChainer(cfg)
    # Decisively separated, but over the cap.
    n = 12
    scores = np.array([0.95 if i % 2 == 0 else 0.05 for i in range(n)], dtype=np.float32)
    assert n > cfg.max_confidence_anchors
    gate = chainer._confidence_gate(scores, np.ones(n, dtype=bool))
    assert np.allclose(gate, 1.0), gate
    print(f"{n} anchors (> {cfg.max_confidence_anchors}) -> gate is identity")


def test_weights_without_neural_scores_are_pure_length():
    """Unscored anchors (NaN) must never enter the adaptive path."""
    chainer = AffineChainer(ChainingConfig())
    anchors = AnchorSet.from_lists(
        read_pos=[0, 20, 40],
        ref_pos=[0, 20, 40],
        length=[10, 20, 30],
        strand=[1, 1, 1],
        read_len=100,
        ref_len=100,
    )
    assert not np.isfinite(anchors.score).any()
    weights = chainer._weights(anchors, None)
    assert np.allclose(weights, [10.0, 20.0, 30.0]), weights
    print("no neural scores -> pure length weights")


def test_legacy_linear_blend_when_adaptive_disabled():
    """``adaptive_seed_scoring=False`` keeps the historical ``0.5 + score`` blend."""
    cfg = ChainingConfig(adaptive_seed_scoring=False)
    chainer = AffineChainer(cfg)
    scores = np.array([0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    weights = chainer._weights(anchors, None)
    expected = anchors.length.astype(np.float64) * (0.5 + scores)
    assert np.allclose(weights, expected), (weights, expected)
    # Flat scores still blend under the legacy path (unlike adaptive fallback).
    flat = _anchor_set_with_scores(np.full(6, 0.5, dtype=np.float32))
    flat_w = chainer._weights(flat, None)
    assert np.allclose(flat_w, flat.length * 1.0)  # 0.5 + 0.5
    # And they differ from the adaptive identity-on-flat behaviour.
    adaptive = AffineChainer(ChainingConfig(adaptive_seed_scoring=True))
    adaptive_w = adaptive._weights(flat, None)
    assert np.allclose(adaptive_w, flat.length.astype(np.float64))
    print("legacy blend = length*(0.5+score); adaptive falls back on flat scores")


def test_adaptive_and_legacy_weights_diverge_when_confident():
    """On a decisive distribution the two scoring paths must not agree."""
    scores = np.array([0.97, 0.95, 0.96, 0.02, 0.03, 0.04, 0.98, 0.01],
                      dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    adaptive_w = AffineChainer(ChainingConfig(adaptive_seed_scoring=True))._weights(
        anchors, None
    )
    legacy_w = AffineChainer(ChainingConfig(adaptive_seed_scoring=False))._weights(
        anchors, None
    )
    assert not np.allclose(adaptive_w, legacy_w), (adaptive_w, legacy_w)
    # Adaptive up-weights the good seeds harder than the linear blend near p~1.
    good = scores > 0.5
    assert adaptive_w[good].mean() > legacy_w[good].mean()
    print(f"adaptive/legacy diverge; good-seed means "
          f"{adaptive_w[good].mean():.2f} vs {legacy_w[good].mean():.2f}")


def test_backbone_bias_composes_with_confidence_gate():
    """Reference-path bias is applied before the gate, so both effects stack."""
    cfg = ChainingConfig(ref_path_bias=2.0, confidence_threshold=0.0,
                         min_confidence_anchors=4)
    chainer = AffineChainer(cfg)
    scores = np.array([0.9, 0.1, 0.85, 0.15], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    anchors.node_id[:] = np.array([0, 1, 0, 1], dtype=np.int64)
    backbone = np.array([True, False], dtype=bool)
    from graphmambaformer.alignment.chaining import ChainingContext
    ctx = ChainingContext(backbone=backbone)

    weights = chainer._weights(anchors, ctx)
    gate = chainer._confidence_gate(anchors.score, np.isfinite(anchors.score))
    base = anchors.length.astype(np.float64) + np.array([2.0, 0.0, 2.0, 0.0])
    assert np.allclose(weights, base * gate), (weights, base * gate)
    print("backbone bias stacks with the confidence gate")


def test_flat_scores_do_not_change_primary_chain():
    """Under-confident scores must leave the classical primary undisturbed."""
    cfg = ChainingConfig(min_chain_score=1.0, min_confidence_anchors=4)
    # Diagonal A is geometrically stronger (more anchors on a clean diagonal).
    read_pos = np.array([0, 30, 60, 90, 0, 40], dtype=np.int64)
    ref_pos = np.array([0, 30, 60, 90, 400, 440], dtype=np.int64)
    anchors = AnchorSet.from_lists(
        read_pos=read_pos,
        ref_pos=ref_pos,
        length=np.full(6, 20, dtype=np.int64),
        strand=np.ones(6, dtype=np.int8),
        read_len=150,
        ref_len=500,
    )

    classical = AffineChainer(cfg).chain(anchors)
    assert classical, "expected a classical chain"
    classical_ref = int(anchors.ref_pos[classical[0].anchor_idx].min())

    anchors.score[:] = 0.5  # flat -> adaptive falls back
    adaptive = AffineChainer(cfg).chain(anchors)
    assert adaptive, "expected an adaptive (fallback) chain"
    adaptive_ref = int(anchors.ref_pos[adaptive[0].anchor_idx].min())
    assert adaptive_ref == classical_ref == 0, (adaptive_ref, classical_ref)
    print(f"flat scores preserve classical primary at ref {classical_ref}")


def test_adaptive_scoring_can_override_weaker_geometric_diagonal():
    """A shorter but confidently-scored diagonal must beat a longer unscored one."""
    cfg = ChainingConfig(min_chain_score=1.0, min_confidence_anchors=4,
                         secondary_score_ratio=0.0, max_chains=4)
    # Diagonal A: 5 long anchors, low neural score.
    # Diagonal B: 4 shorter anchors, high neural score — should win under adaptive.
    read_a = np.array([0, 40, 80, 120, 160], dtype=np.int64)
    ref_a = np.array([0, 40, 80, 120, 160], dtype=np.int64)
    read_b = np.array([0, 40, 80, 120], dtype=np.int64)
    ref_b = np.array([500, 540, 580, 620], dtype=np.int64)
    anchors = AnchorSet.from_lists(
        read_pos=np.concatenate([read_a, read_b]),
        ref_pos=np.concatenate([ref_a, ref_b]),
        length=np.concatenate([np.full(5, 25), np.full(4, 15)]).astype(np.int64),
        strand=np.ones(9, dtype=np.int8),
        read_len=220,
        ref_len=700,
    )
    anchors.score[:] = np.array(
        [0.05, 0.04, 0.03, 0.02, 0.06, 0.97, 0.98, 0.96, 0.95], dtype=np.float32
    )

    classical_cfg = ChainingConfig(
        min_chain_score=1.0, adaptive_seed_scoring=True, min_confidence_anchors=4,
        # Force classical by wiping scores.
    )
    no_score = AnchorSet.from_lists(
        read_pos=anchors.read_pos, ref_pos=anchors.ref_pos, length=anchors.length,
        strand=anchors.strand, read_len=anchors.read_len, ref_len=anchors.ref_len,
    )
    classical_primary = AffineChainer(classical_cfg).chain(no_score)[0]
    classical_ref = int(no_score.ref_pos[classical_primary.anchor_idx].min())
    assert classical_ref == 0, "without scores the longer diagonal A should win"

    adaptive_primary = AffineChainer(cfg).chain(anchors)[0]
    adaptive_ref = int(anchors.ref_pos[adaptive_primary.anchor_idx].min())
    assert adaptive_ref >= 500, (adaptive_ref, "confident shorter diagonal B must win")
    print(f"classical picked ref {classical_ref}; adaptive overrode to ref {adaptive_ref}")


def test_weights_are_strictly_positive_under_gate():
    """Clamped gates must keep every weight > 0 so the DP never sees a dead anchor."""
    cfg = ChainingConfig(confidence_threshold=0.0, min_confidence_anchors=4,
                         logit_gate_min=0.1, logit_gate_max=3.0)
    chainer = AffineChainer(cfg)
    scores = np.array([1e-9, 1.0, 0.0, 0.999999, 0.5, 0.01], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    weights = chainer._weights(anchors, None)
    assert (weights > 0).all(), weights
    assert weights.min() >= anchors.length.min() * cfg.logit_gate_min - 1e-9
    print(f"all weights positive under extreme scores; min={weights.min():.3f}")


def test_agnes_algorithm1_decision_table():
    """AGNES Algorithm 1 decision table: degenerate / flat / separated."""
    cfg = ChainingConfig(confidence_threshold=0.7, min_confidence_anchors=5,
                         max_confidence_anchors=1000)
    chainer = AffineChainer(cfg)

    # |V| < 5 → PureDP even if scores look perfect.
    tiny = _anchor_set_with_scores(np.array([0.99, 0.01, 0.98, 0.02], dtype=np.float32))
    assert chainer.trust_neural_scores(tiny) is False

    # Flat mid scores → PureDP.
    flat = _anchor_set_with_scores(np.full(12, 0.5, dtype=np.float32))
    assert chainer.trust_neural_scores(flat) is False

    # Decisively separated → GNN-guided.
    sep = _anchor_set_with_scores(np.array(
        [0.95, 0.97, 0.04, 0.03, 0.96, 0.02, 0.94, 0.05, 0.98, 0.01], dtype=np.float32
    ))
    assert chainer.trust_neural_scores(sep) is True

    # |V| > max → PureDP.
    big_cfg = ChainingConfig(max_confidence_anchors=8, min_confidence_anchors=5)
    big = _anchor_set_with_scores(np.array(
        [0.95 if i % 2 == 0 else 0.05 for i in range(12)], dtype=np.float32
    ))
    assert AffineChainer(big_cfg).trust_neural_scores(big) is False
    print("Algorithm 1 decision table: tiny/flat/big=False, separated=True")


def test_pruned_scores_cannot_reopen_gate_without_trust_flag():
    """Regression: after pruning away p<0.3 seeds, auto-confidence would lie.

    The survivors are all high-scoring, so ``(μ_high - 0) / σ`` looks decisive
    even though the classifier was never trusted on the full set. ``trust_neural=
    False`` must force classical weights; that is what the hybrid pipeline pins
    after an under-confident Algorithm-1 decision.
    """
    from graphmambaformer.alignment.chaining import ChainingContext

    chainer = AffineChainer(ChainingConfig(min_confidence_anchors=5))
    # Full set is flat / under-confident.
    full_scores = np.full(12, 0.55, dtype=np.float32)
    full = _anchor_set_with_scores(full_scores)
    assert chainer.trust_neural_scores(full) is False

    # Simulate prune-to-top-k: only the "best" (still ~0.55) remain — or worse,
    # a pruned set that is artificially all-high.
    pruned = _anchor_set_with_scores(np.array(
        [0.92, 0.91, 0.93, 0.90, 0.94, 0.89], dtype=np.float32
    ))
    # Auto path on the pruned set would *incorrectly* trust them:
    assert chainer.trust_neural_scores(pruned) is True

    classical = chainer._weights(pruned, ChainingContext(trust_neural=False))
    assert np.allclose(classical, pruned.length.astype(np.float64)), classical

    guided = chainer._weights(pruned, ChainingContext(trust_neural=True))
    assert not np.allclose(guided, classical), (guided, classical)
    print("trust_neural=False blocks reopen after prune; True still applies gate")


def test_forced_trust_applies_logit_without_rechecking_confidence():
    """``trust_neural=True`` must apply the logit gate even on a tiny set."""
    from graphmambaformer.alignment.chaining import ChainingContext

    cfg = ChainingConfig(min_confidence_anchors=5)  # tiny set would fail auto
    chainer = AffineChainer(cfg)
    scores = np.array([0.9, 0.1, 0.85], dtype=np.float32)
    anchors = _anchor_set_with_scores(scores)
    assert chainer.trust_neural_scores(anchors) is False  # |V|<5

    auto = chainer._weights(anchors, None)
    forced = chainer._weights(anchors, ChainingContext(trust_neural=True))
    assert np.allclose(auto, anchors.length.astype(np.float64))
    expected = anchors.length * chainer._logit_gate(anchors.score, np.isfinite(anchors.score))
    assert np.allclose(forced, expected), (forced, expected)
    print("forced trust bypasses the |V| guard and applies the logit gate")


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #
def _full_affine_sw(query, target, cfg):
    """Unbanded affine-gap Smith-Waterman max score (no traceback)."""
    n, m = len(query), len(target)
    neg = float("-inf")
    H = np.zeros((n + 1, m + 1))
    E = np.full((n + 1, m + 1), neg)  # gap in query (deletion)
    F = np.full((n + 1, m + 1), neg)  # gap in target (insertion)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i][j] = max(H[i][j - 1] - cfg.gap_open, E[i][j - 1] - cfg.gap_extend)
            F[i][j] = max(H[i - 1][j] - cfg.gap_open, F[i - 1][j] - cfg.gap_extend)
            score = (
                cfg.match_score
                if query[i - 1] == target[j - 1]
                else -cfg.mismatch_penalty
            )
            H[i][j] = max(0.0, H[i - 1][j - 1] + score, E[i][j], F[i][j])
            best = max(best, H[i][j])
    return best


def test_banded_sw_matches_unbanded_dp():
    """With a band wide enough to cover the matrix, banded == full DP."""
    cfg = ExtensionConfig()
    for trial in range(4):
        length = int(RNG.integers(20, 45))
        a = random_seq(length)
        # Mutate a couple of bases so the optimum is not trivially the diagonal.
        b_list = list(a)
        for _ in range(2):
            pos = int(RNG.integers(0, length))
            b_list[pos] = RNG.choice(list(BASES))
        b = "".join(b_list)

        q = encode_bases(a).astype(np.int16)
        t = encode_bases(b).astype(np.int16)
        result = banded_affine_sw_batch(
            torch.as_tensor(q)[None],
            torch.as_tensor(t)[None],
            torch.tensor([len(a)]),
            torch.tensor([len(b)]),
            cfg,
            half_band=length,  # band covers everything -> exact
            return_matrices=False,
        )
        got = float(result.score[0])
        expected = _full_affine_sw(q, t, cfg)
        assert abs(got - expected) < 1e-4, (trial, got, expected)
    print("banded SW equals the unbanded affine DP when the band is full-width")


def test_banded_sw_identical_sequences_score_perfectly():
    cfg = ExtensionConfig()
    seq = random_seq(40)
    codes = encode_bases(seq).astype(np.int16)
    result = banded_affine_sw_batch(
        torch.as_tensor(codes)[None],
        torch.as_tensor(codes)[None],
        torch.tensor([len(seq)]),
        torch.tensor([len(seq)]),
        cfg,
        half_band=8,
        return_matrices=True,
    )
    perfect = cfg.match_score * len(seq)
    assert abs(float(result.score[0]) - perfect) < 1e-4, result.score
    extension = traceback_banded(result, 0, encode_bases(seq), encode_bases(seq), cfg)
    assert extension.cigar == [("=", len(seq))], extension.cigar
    assert extension.edit_distance == 0
    print(f"identical {len(seq)} bp sequences: score {perfect}, "
          f"CIGAR {extension.cigar_string}")


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def test_wfa_matches_levenshtein():
    wfa = WavefrontAligner(ExtensionConfig(algorithm="wfa"))
    cases = [("ACGTACGTACGT", "ACGTACGTACGT"), ("ACGT", "TGCA")]
    for _ in range(8):
        a = random_seq(int(RNG.integers(8, 30)))
        b_list = list(a)
        for _ in range(int(RNG.integers(0, 4))):
            pos = int(RNG.integers(0, len(b_list)))
            action = RNG.integers(0, 3)
            if action == 0:
                b_list[pos] = RNG.choice(list(BASES))
            elif action == 1:
                b_list.pop(pos)
            else:
                b_list.insert(pos, RNG.choice(list(BASES)))
        cases.append((a, "".join(b_list)))

    for a, b in cases:
        distance, cigar = wfa.align(encode_bases(a), encode_bases(b))
        expected = _levenshtein(a, b)
        assert distance == expected, (a, b, distance, expected)
        # The CIGAR must account for exactly the query it consumed.
        consumed = sum(n for op, n in cigar if op in "=XI")
        assert consumed == len(a), (a, b, cigar)
        target_consumed = sum(n for op, n in cigar if op in "=XD")
        assert target_consumed == len(b), (a, b, cigar)
        edits = sum(n for op, n in cigar if op in "XID")
        assert edits == distance, (a, b, distance, cigar)
    print(f"WFA distance == Levenshtein on {len(cases)} random indel/sub cases")


def test_wfa_reports_when_over_budget():
    """Beyond wfa_max_distance the aligner must signal, not silently truncate."""
    wfa = WavefrontAligner(ExtensionConfig(algorithm="wfa", wfa_max_distance=2))
    a, b = random_seq(60), random_seq(60)  # unrelated: distance >> 2
    try:
        wfa.align(encode_bases(a), encode_bases(b))
    except RuntimeError as exc:
        print(f"WFA over budget raises for the caller to fall back: {exc}")
        return
    raise AssertionError("expected RuntimeError once past wfa_max_distance")


def test_wfa_does_not_treat_ambiguous_bases_as_matches():
    wfa = WavefrontAligner(ExtensionConfig(algorithm="wfa"))
    distance, cigar = wfa.align(encode_bases("AN"), encode_bases("AN"))
    assert distance == 1
    assert cigar == [("=", 1), ("X", 1)], cigar


def test_wfa_extension_does_not_force_search_flanks_into_cigar():
    from graphmambaformer.alignment.extension import ExtensionEngine

    read = "ACGTACGT"
    reference = "T" * 50 + read + "G" * 50
    anchors = AnchorSet.from_lists(
        read_pos=[0],
        ref_pos=[50],
        length=[8],
        strand=[1],
        read_len=8,
        ref_len=len(reference),
    )
    chain = Chain(
        anchor_idx=np.array([0]),
        score=8.0,
        strand=1,
        read_start=0,
        read_end=8,
        ref_start=50,
        ref_end=58,
        is_primary=True,
    )
    result = ExtensionEngine(ExtensionConfig(algorithm="wfa", flank=25)).extend_chains(
        read, [chain], anchors, reference
    )[0]
    assert result.cigar == [("=", 8)], result.cigar
    assert (result.ref_start, result.ref_end) == (50, 58)
