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

from graphmambaformer.alignment.chaining import chain_dp_numpy
from graphmambaformer.alignment.extension import (
    WavefrontAligner,
    banded_affine_sw_batch,
    traceback_banded,
)
from graphmambaformer.alignment.seeding import (
    FMIndex,
    encode_bases,
    minimizer_mask,
    pack_kmers,
    reverse_complement_codes,
    suffix_array,
)
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
