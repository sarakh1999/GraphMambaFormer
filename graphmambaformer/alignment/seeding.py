"""Stage 1 — Seeding.

Six index types, one interface. Each index is built once over a reference (or a
pangenome graph's node sequences) and then queried with a read to produce an
:class:`~graphmambaformer.alignment.types.AnchorSet`:

===================  ========================================================
Index                Method
===================  ========================================================
:class:`MinimizerIndex`   ``(w, k)`` minimizer sketch — the default fast path
:class:`SMEMIndex`        super-maximal exact matches over an FM-index
:class:`FMIndex`          BWT + sampled suffix array; exact k-mer occurrences
:class:`DeBruijnIndex`    de Bruijn k-mer index (``k=21``), graph-node aware
:class:`FuzzySeedIndex`   spaced seeds — tolerates a mismatch inside the seed
:class:`MultiplexDBG`     several ``k`` at once, adaptive to repeat scale
:class:`GPUKmerIndex`     GPU-resident table; batched lookup for many reads
===================  ========================================================

Everything is array-programmed rather than looped: k-mer packing is a Horner
sweep, minimizer selection is ``w`` vectorized comparisons, and the SMEM search
extends *every* read end position simultaneously so the FM-index walk costs one
batched rank query per extension step instead of one per position. That is what
makes the accurate seeding modes usable on long reads, and it means the same
code lifts onto CuPy by swapping the array namespace.

:class:`SeedingEngine` is the stage entry point: it runs the configured modes,
merges their anchors, collapses near-duplicates on the same diagonal, and maps
each anchor onto a pangenome graph node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import SeedingConfig
from .types import AnchorSet, source_id

# Base alphabet. 0 is reserved as the FM-index sentinel, which must sort below
# every real symbol; ``N`` gets its own code so it can never match a real base.
SENTINEL = 0
BASE_CODES = {"A": 1, "C": 2, "G": 3, "T": 4}
N_CODE = 5
ALPHABET_SIZE = 6

_COMPLEMENT = np.array([0, 4, 3, 2, 1, 5], dtype=np.int8)  # $ A C G T N -> $ T G C A N

_ENCODE_LUT = np.full(256, N_CODE, dtype=np.int8)
for _base, _code in BASE_CODES.items():
    _ENCODE_LUT[ord(_base)] = _code
    _ENCODE_LUT[ord(_base.lower())] = _code

_MAX_K = 31  # 2 bits per base must fit in a signed 64-bit integer


# --------------------------------------------------------------------------- #
# Sequence primitives
# --------------------------------------------------------------------------- #
def encode_bases(seq: str) -> np.ndarray:
    """Encode a nucleotide string as ``int8`` codes (``A C G T`` -> ``1..4``, else 5)."""
    if not seq:
        return np.zeros(0, dtype=np.int8)
    raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    return _ENCODE_LUT[raw]


def decode_bases(codes: np.ndarray) -> str:
    """Inverse of :func:`encode_bases` (unknown codes become ``N``)."""
    lut = np.array(list("$ACGTN"), dtype="<U1")
    return "".join(lut[np.clip(codes, 0, N_CODE)])


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    """Reverse-complement an encoded sequence."""
    return _COMPLEMENT[codes[::-1]]


def hash64(values: np.ndarray) -> np.ndarray:
    """Invertible 64-bit integer hash (minimap2's ``hash64``), vectorized.

    Used to order k-mers for minimizer selection, so the sketch does not
    correlate with lexicographic k-mer content.
    """
    x = values.astype(np.uint64, copy=True)
    u = np.uint64
    x = ~x + (x << u(21))
    x ^= x >> u(24)
    x = x + (x << u(3)) + (x << u(8))
    x ^= x >> u(14)
    x = x + (x << u(2)) + (x << u(4))
    x ^= x >> u(28)
    x = x + (x << u(31))
    return x


def pack_kmers(codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack every ``k``-mer of ``codes`` into a 2-bit integer.

    A Horner sweep over ``k`` offsets: ``O(k)`` passes and ``O(n)`` memory, so it
    scales to chromosome-sized inputs where a ``(n, k)`` sliding-window matrix
    would not.

    Returns ``(packed, valid)``, both length ``len(codes) - k + 1``. ``valid`` is
    False wherever the window contains a non-ACGT base.
    """
    if not 1 <= k <= _MAX_K:
        raise ValueError(f"k must be in [1, {_MAX_K}], got {k}")
    m = len(codes) - k + 1
    if m <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    digits = (codes.astype(np.int64) - 1)  # A C G T -> 0..3, N -> 4
    bad = digits > 3

    packed = np.zeros(m, dtype=np.int64)
    invalid = np.zeros(m, dtype=bool)
    for offset in range(k):
        packed = packed * 4 + digits[offset : offset + m]
        invalid |= bad[offset : offset + m]
    packed[invalid] = 0
    return packed, ~invalid


def pack_spaced_kmers(
    codes: np.ndarray, pattern: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pack spaced seeds: only the ``'1'`` positions of ``pattern`` are compared.

    A spaced seed of span ``len(pattern)`` and weight ``pattern.count('1')``
    tolerates mismatches at the ``'0'`` (don't-care) positions, which recovers
    seeds in the noisy long reads where every contiguous k-mer is broken.
    """
    kept = [i for i, ch in enumerate(pattern) if ch == "1"]
    if not kept:
        raise ValueError("spaced pattern must contain at least one '1'")
    if len(kept) > _MAX_K:
        raise ValueError(f"spaced-seed weight must be <= {_MAX_K}")

    span = len(pattern)
    m = len(codes) - span + 1
    if m <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    digits = codes.astype(np.int64) - 1
    bad = digits > 3

    packed = np.zeros(m, dtype=np.int64)
    invalid = np.zeros(m, dtype=bool)
    for offset in kept:
        packed = packed * 4 + digits[offset : offset + m]
        invalid |= bad[offset : offset + m]
    packed[invalid] = 0
    return packed, ~invalid


def minimizer_mask(hashes: np.ndarray, window: int) -> np.ndarray:
    """Boolean mask of the ``(window, k)`` minimizers among ``hashes``.

    A k-mer is a minimizer when it attains the minimum hash of at least one
    window of ``window`` consecutive k-mers. Computed as ``window`` vectorized
    comparisons — ``O(window)`` passes, ``O(n)`` memory — and ties are all kept,
    matching the standard sketch.
    """
    m = len(hashes)
    if m == 0:
        return np.zeros(0, dtype=bool)
    w = max(1, min(window, m))
    n_windows = m - w + 1

    window_min = hashes[:n_windows].copy()
    for offset in range(1, w):
        np.minimum(window_min, hashes[offset : offset + n_windows], out=window_min)

    selected = np.zeros(m, dtype=bool)
    for offset in range(w):
        selected[offset : offset + n_windows] |= (
            hashes[offset : offset + n_windows] == window_min
        )
    return selected


# --------------------------------------------------------------------------- #
# Shared k-mer table (backs the minimizer / DBG / fuzzy / multi-k indices)
# --------------------------------------------------------------------------- #
@dataclass
class KmerTable:
    """Sorted k-mer table with a flat position list — a CSR-style hash join.

    ``keys`` is sorted and unique, and ``positions[offsets[i]:offsets[i + 1]]``
    holds every reference offset of ``keys[i]``. Lookup is a ``searchsorted``,
    so querying a whole read is one vectorized call rather than a dict loop.
    """

    keys: np.ndarray  # (U,) int64, sorted ascending
    offsets: np.ndarray  # (U + 1,) int64
    positions: np.ndarray  # (P,) int64 reference offsets
    k: int
    span: int  # bases covered by one entry (== k, or the pattern span for spaced seeds)

    @classmethod
    def build(
        cls,
        packed: np.ndarray,
        valid: np.ndarray,
        k: int,
        span: Optional[int] = None,
        max_occ: int = 0,
    ) -> "KmerTable":
        """Build from packed k-mers; ``max_occ > 0`` drops over-represented keys.

        Discarding high-occurrence k-mers is what keeps repetitive regions from
        producing an anchor blow-up — the same role ``max_occ`` plays in BWA-MEM.
        """
        pos = np.flatnonzero(valid).astype(np.int64)
        codes = packed[pos]

        order = np.argsort(codes, kind="stable")
        codes, pos = codes[order], pos[order]

        keys, counts = np.unique(codes, return_counts=True)
        offsets = np.zeros(len(keys) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        if max_occ > 0 and len(keys) and counts.max() > max_occ:
            keep = counts <= max_occ
            # Rebuild the flat position list without the dropped keys.
            starts, ends = offsets[:-1][keep], offsets[1:][keep]
            pos = np.concatenate(
                [pos[s:e] for s, e in zip(starts, ends)]
                or [np.zeros(0, dtype=np.int64)]
            )
            keys, counts = keys[keep], counts[keep]
            offsets = np.zeros(len(keys) + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])

        return cls(keys=keys, offsets=offsets, positions=pos, k=k, span=span or k)

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized lookup: returns ``(slice_start, count)`` per query key."""
        if len(self.keys) == 0 or len(queries) == 0:
            z = np.zeros(len(queries), dtype=np.int64)
            return z, z.copy()
        idx = np.searchsorted(self.keys, queries)
        clipped = np.minimum(idx, len(self.keys) - 1)
        hit = self.keys[clipped] == queries
        start = np.where(hit, self.offsets[clipped], 0)
        count = np.where(hit, self.offsets[clipped + 1] - self.offsets[clipped], 0)
        return start, count

    def join(
        self, query_codes: np.ndarray, query_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cross-join query k-mers against the table.

        Returns ``(read_pos, ref_pos)``: every ``(query, reference)`` co-occurrence,
        expanded without a Python loop via ``repeat`` + an arange offset trick.
        """
        start, count = self.lookup(query_codes)
        total = int(count.sum())
        if total == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy()

        read_pos = np.repeat(query_pos, count)
        # Position within each query's group: arange minus the group's own base.
        group_base = np.repeat(np.concatenate([[0], np.cumsum(count)[:-1]]), count)
        within = np.arange(total, dtype=np.int64) - group_base
        ref_pos = self.positions[np.repeat(start, count) + within]
        return read_pos, ref_pos


# --------------------------------------------------------------------------- #
# FM-index (BWT + sampled suffix array)
# --------------------------------------------------------------------------- #
def suffix_array(codes: np.ndarray) -> np.ndarray:
    """Suffix array of ``codes`` by prefix doubling.

    ``O(n log n)`` NumPy sorts. ``codes`` must end with the unique
    :data:`SENTINEL` so every suffix gets a distinct rank.
    """
    n = len(codes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    _, rank = np.unique(codes, return_inverse=True)
    rank = rank.astype(np.int64).ravel()

    shift = 1
    order = np.argsort(rank, kind="stable")
    while shift < n:
        second = np.full(n, -1, dtype=np.int64)
        second[: n - shift] = rank[shift:]
        order = np.lexsort((second, rank))

        first_sorted, second_sorted = rank[order], second[order]
        new_rank = np.zeros(n, dtype=np.int64)
        np.cumsum(
            (first_sorted[1:] != first_sorted[:-1])
            | (second_sorted[1:] != second_sorted[:-1]),
            out=new_rank[1:],
        )
        rank[order] = new_rank
        if new_rank[-1] == n - 1:  # all ranks distinct -> `order` is the SA
            return order
        shift <<= 1
    return order


class FMIndex:
    """BWT-based FM-index over a reference, with a sampled suffix array.

    Supports batched backward extension: :meth:`backward_extend` advances many
    independent SA intervals in one call, each by its own character. That is what
    lets :class:`SMEMIndex` walk every read position in parallel.

    Memory is ``~2n`` bytes for the BWT plus ``8n / sa_sample`` for the sampled
    suffix array, instead of the ``8n`` a full SA would need.
    """

    def __init__(
        self,
        codes: np.ndarray,
        sa_sample: int = 8,
        occ_sample: int = 64,
    ):
        if len(codes) and codes.min() <= SENTINEL:
            raise ValueError("reference codes must not contain the sentinel value 0")
        self.text = np.concatenate([codes.astype(np.int8), [SENTINEL]])
        self.n = len(self.text)
        self.sa_sample = max(1, sa_sample)
        self.occ_sample = max(1, occ_sample)

        sa = suffix_array(self.text)
        self.bwt = self.text[sa - 1]  # sa == 0 wraps to the sentinel, as intended

        # C[c] = number of symbols in the text strictly less than c.
        counts = np.bincount(self.text, minlength=ALPHABET_SIZE).astype(np.int64)
        self.C = np.zeros(ALPHABET_SIZE + 1, dtype=np.int64)
        np.cumsum(counts, out=self.C[1:])

        # Rank checkpoints: occ[j, c] = count of c in bwt[: j * occ_sample].
        n_checkpoints = self.n // self.occ_sample + 1
        self.occ = np.zeros((n_checkpoints + 1, ALPHABET_SIZE), dtype=np.int64)
        for c in range(ALPHABET_SIZE):
            cumulative = np.concatenate([[0], np.cumsum(self.bwt == c)])
            idx = np.minimum(
                np.arange(n_checkpoints + 1) * self.occ_sample, self.n
            )
            self.occ[:, c] = cumulative[idx]

        # Sampled suffix array: keep sa[i] only where it is a multiple of the
        # sampling rate; everything else is recovered by walking LF.
        self.sa_mask = (sa % self.sa_sample) == 0
        self.sa_values = sa[self.sa_mask]
        self.sa_rank = np.cumsum(self.sa_mask) - 1  # index into sa_values

    # ---- rank / LF --------------------------------------------------------- #
    def rank(self, indices: np.ndarray, chars: np.ndarray) -> np.ndarray:
        """Batched ``Occ(c, i)``: occurrences of ``chars[t]`` in ``bwt[:indices[t]]``.

        Checkpoint lookup plus a bounded gather over the remaining
        ``< occ_sample`` positions, so cost is independent of ``indices``.
        """
        indices = np.asarray(indices, dtype=np.int64)
        chars = np.asarray(chars, dtype=np.int64)
        checkpoint = indices // self.occ_sample
        base = self.occ[checkpoint, chars]

        span = np.arange(self.occ_sample, dtype=np.int64)
        probe = checkpoint[:, None] * self.occ_sample + span[None, :]
        in_range = probe < indices[:, None]
        values = self.bwt[np.minimum(probe, self.n - 1)]
        return base + ((values == chars[:, None]) & in_range).sum(axis=1)

    def backward_extend(
        self, lo: np.ndarray, hi: np.ndarray, chars: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepend ``chars`` to each pattern, advancing SA intervals ``[lo, hi)``."""
        base = self.C[chars]
        return base + self.rank(lo, chars), base + self.rank(hi, chars)

    def lf(self, indices: np.ndarray) -> np.ndarray:
        """Last-to-first mapping: the row of the suffix one character earlier."""
        chars = self.bwt[indices].astype(np.int64)
        return self.C[chars] + self.rank(indices, chars)

    def locate(self, indices: np.ndarray) -> np.ndarray:
        """Text offsets for SA rows ``indices``, walking LF to the next sample."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return np.zeros(0, dtype=np.int64)

        steps = np.zeros(len(indices), dtype=np.int64)
        current = indices.copy()
        pending = ~self.sa_mask[current]
        # At most sa_sample iterations: each LF step moves one character back and
        # a sample occurs every sa_sample text positions.
        for _ in range(self.sa_sample + 1):
            if not pending.any():
                break
            active = np.flatnonzero(pending)
            current[active] = self.lf(current[active])
            steps[active] += 1
            pending = ~self.sa_mask[current]

        return self.sa_values[self.sa_rank[current]] + steps

    def count(self, pattern: np.ndarray) -> tuple[int, int]:
        """SA interval ``[lo, hi)`` of one exact pattern (backward search)."""
        lo, hi = np.array([0]), np.array([self.n])
        for char in pattern[::-1]:
            if char == N_CODE:
                return 0, 0
            lo, hi = self.backward_extend(lo, hi, np.array([char], dtype=np.int64))
            if hi[0] <= lo[0]:
                return 0, 0
        return int(lo[0]), int(hi[0])

    def occurrences(self, pattern: np.ndarray, max_occ: int = 0) -> np.ndarray:
        """Text offsets of every exact occurrence of ``pattern``."""
        lo, hi = self.count(pattern)
        if hi <= lo or (max_occ and hi - lo > max_occ):
            return np.zeros(0, dtype=np.int64)
        return self.locate(np.arange(lo, hi, dtype=np.int64))

    # ---- SMEMs -------------------------------------------------------------- #
    def smems(
        self, read_codes: np.ndarray, min_len: int = 13, max_occ: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Super-maximal exact matches of ``read_codes`` against the reference.

        For every read end position ``j`` the search finds the smallest ``i``
        such that ``read[i..j]`` still occurs in the reference. Those matches are
        right-maximal by construction (they end at ``j``) and left-maximal by
        minimality of ``i``; because ``i_min`` is non-decreasing in ``j``, an
        interval is contained in its successor exactly when
        ``i_min(j + 1) == i_min(j)``, so dropping those leaves the SMEMs.

        Every ``j`` is extended simultaneously, one batched rank query per
        extension step, which is what makes this affordable on long reads.

        Returns ``(read_start, length, sa_row)`` per SMEM; ``sa_row`` is the low
        end of the SA interval, and its size is recovered by the caller.
        """
        n = len(read_codes)
        if n == 0:
            return (np.zeros(0, dtype=np.int64),) * 3

        ends = np.arange(n, dtype=np.int64)
        lo = np.zeros(n, dtype=np.int64)
        hi = np.full(n, self.n, dtype=np.int64)
        cursor = ends.copy()  # next character to prepend for each end position
        # `N` in the read terminates extension, so ambiguous bases never seed.
        active = read_codes[cursor] != N_CODE
        interval_lo = lo.copy()
        interval_hi = hi.copy()
        left = ends.copy() + 1  # i_min(j); one past the end means "no match yet"

        while active.any():
            idx = np.flatnonzero(active)
            chars = read_codes[cursor[idx]].astype(np.int64)
            new_lo, new_hi = self.backward_extend(lo[idx], hi[idx], chars)

            grew = new_hi > new_lo
            kept = idx[grew]
            lo[kept], hi[kept] = new_lo[grew], new_hi[grew]
            interval_lo[kept], interval_hi[kept] = new_lo[grew], new_hi[grew]
            left[kept] = cursor[kept]
            cursor[kept] -= 1

            active[idx[~grew]] = False
            # Stop at the read start or at an ambiguous base.
            exhausted = kept[cursor[kept] < 0]
            active[exhausted] = False
            still = kept[cursor[kept] >= 0]
            active[still] = read_codes[cursor[still]] != N_CODE

        lengths = ends - left + 1
        occ = interval_hi - interval_lo

        # Super-maximality: drop j whose interval is contained in j + 1's.
        super_maximal = np.ones(n, dtype=bool)
        super_maximal[:-1] = left[1:] > left[:-1]

        keep = super_maximal & (lengths >= min_len) & (occ > 0)
        if max_occ > 0:
            keep &= occ <= max_occ
        sel = np.flatnonzero(keep)
        return left[sel], lengths[sel], interval_lo[sel]


# --------------------------------------------------------------------------- #
# Index implementations
# --------------------------------------------------------------------------- #
class MinimizerIndex:
    """``(w, k)`` minimizer sketch of the reference — the default fast path.

    Sketching keeps roughly ``2 / (w + 1)`` of the k-mers, so the index is small
    and the anchor set stays sparse while still guaranteeing that two sequences
    sharing a long exact stretch share a minimizer.
    """

    name = "minimizer"

    def __init__(self, ref_codes: np.ndarray, k: int = 15, window: int = 10, max_occ: int = 0):
        self.k, self.window = k, window
        packed, valid = pack_kmers(ref_codes, k)
        selected = valid & minimizer_mask(hash64(packed), window)
        self.table = KmerTable.build(packed, selected, k, max_occ=max_occ)

    def query(self, read_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        packed, valid = pack_kmers(read_codes, self.k)
        selected = valid & minimizer_mask(hash64(packed), self.window)
        pos = np.flatnonzero(selected)
        read_pos, ref_pos = self.table.join(packed[pos], pos)
        return read_pos, ref_pos, np.full(len(read_pos), self.k, dtype=np.int64)


class DeBruijnIndex:
    """De Bruijn k-mer index (``k=21``), aware of pangenome graph nodes.

    Every k-mer of every node sequence is indexed, so a k-mer that only exists on
    an alternate allele is still seedable — which a linear-reference index cannot
    do. Positions are stored in the concatenated node coordinate space; the
    per-anchor node id comes back from :meth:`node_of`.
    """

    name = "dbg"

    def __init__(
        self,
        node_codes: Sequence[np.ndarray],
        k: int = 21,
        max_occ: int = 0,
    ):
        self.k = k
        self.node_lengths = np.array([len(c) for c in node_codes], dtype=np.int64)
        self.node_starts = np.zeros(len(node_codes) + 1, dtype=np.int64)
        np.cumsum(self.node_lengths, out=self.node_starts[1:])

        packed_parts, valid_parts = [], []
        for codes in node_codes:
            packed, valid = pack_kmers(codes, k)
            # Pad each node to its full length so offsets stay aligned with the
            # concatenated coordinate space (the last k-1 starts are invalid).
            pad = len(codes) - len(packed)
            packed_parts.append(np.concatenate([packed, np.zeros(max(pad, 0), dtype=np.int64)]))
            valid_parts.append(np.concatenate([valid, np.zeros(max(pad, 0), dtype=bool)]))

        concat_packed = (
            np.concatenate(packed_parts) if packed_parts else np.zeros(0, dtype=np.int64)
        )
        concat_valid = (
            np.concatenate(valid_parts) if valid_parts else np.zeros(0, dtype=bool)
        )
        self.table = KmerTable.build(concat_packed, concat_valid, k, max_occ=max_occ)

    def node_of(self, offsets: np.ndarray) -> np.ndarray:
        """Map concatenated-space offsets back to node ids."""
        if len(offsets) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.clip(
            np.searchsorted(self.node_starts, offsets, side="right") - 1,
            0,
            len(self.node_lengths) - 1,
        )

    def local_offset(self, offsets: np.ndarray) -> np.ndarray:
        """Offset within the containing node."""
        return offsets - self.node_starts[self.node_of(offsets)]

    def query(
        self, read_codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        packed, valid = pack_kmers(read_codes, self.k)
        pos = np.flatnonzero(valid)
        read_pos, ref_pos = self.table.join(packed[pos], pos)
        return (
            read_pos,
            ref_pos,
            np.full(len(read_pos), self.k, dtype=np.int64),
            self.node_of(ref_pos),
        )


class FuzzySeedIndex:
    """Spaced-seed index — error-tolerant seeding for divergent / noisy reads.

    At a 15% error rate a contiguous 15-mer survives with probability
    ``0.85^15 ~ 0.09``, whereas a weight-11 spaced seed of span 18 tolerates
    errors at its seven don't-care positions, so seeds still land inside
    high-error stretches.
    """

    name = "fuzzy"

    def __init__(self, ref_codes: np.ndarray, pattern: str, max_occ: int = 0):
        self.pattern = pattern
        self.span = len(pattern)
        self.weight = pattern.count("1")
        packed, valid = pack_spaced_kmers(ref_codes, pattern)
        self.table = KmerTable.build(packed, valid, self.weight, span=self.span, max_occ=max_occ)

    def query(self, read_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        packed, valid = pack_spaced_kmers(read_codes, self.pattern)
        pos = np.flatnonzero(valid)
        read_pos, ref_pos = self.table.join(packed[pos], pos)
        # The anchor covers the seed's full span, not just its compared bases.
        return read_pos, ref_pos, np.full(len(read_pos), self.span, dtype=np.int64)


class MultiplexDBG:
    """Multi-``k`` de Bruijn index (``k = 15, 21, 31``) with adaptive selection.

    Short ``k`` seeds noisy or divergent regions; long ``k`` stays specific
    inside repeats. Queries start at the largest ``k`` and fall back to shorter
    ones only where the long k-mers found nothing, so specificity is preserved
    wherever it is available.
    """

    name = "multiplex_dbg"

    def __init__(self, ref_codes: np.ndarray, kmers: Sequence[int] = (15, 21, 31), max_occ: int = 0):
        self.kmers = tuple(sorted(kmers, reverse=True))
        self.tables: dict[int, KmerTable] = {}
        for k in self.kmers:
            packed, valid = pack_kmers(ref_codes, k)
            self.tables[k] = KmerTable.build(packed, valid, k, max_occ=max_occ)

    def query(
        self, read_codes: np.ndarray, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        read_pos_all, ref_pos_all, length_all = [], [], []
        covered = np.zeros(len(read_codes), dtype=bool)

        for k in self.kmers:  # largest k first
            packed, valid = pack_kmers(read_codes, k)
            candidate = valid & ~covered[: len(valid)]
            if stride > 1:
                keep = np.zeros_like(candidate)
                keep[::stride] = True
                candidate &= keep

            pos = np.flatnonzero(candidate)
            if pos.size == 0:
                continue
            read_pos, ref_pos = self.tables[k].join(packed[pos], pos)
            if read_pos.size == 0:
                continue

            read_pos_all.append(read_pos)
            ref_pos_all.append(ref_pos)
            length_all.append(np.full(len(read_pos), k, dtype=np.int64))
            # Mark the matched spans so shorter k don't re-seed them.
            for start in np.unique(read_pos):
                covered[start : start + k] = True

        if not read_pos_all:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()
        return (
            np.concatenate(read_pos_all),
            np.concatenate(ref_pos_all),
            np.concatenate(length_all),
        )


class GPUKmerIndex:
    """GPU-resident k-mer table with batched lookup across many reads.

    The table lives in device memory as sorted tensors, so seeding a batch is a
    single ``torch.searchsorted`` (or the CuPy binary-search kernel on CUDA)
    instead of one host-side hash join per read. This is the throughput path used
    by :class:`~graphmambaformer.alignment.pipeline.FastAlignmentPipeline`.
    """

    name = "gpu_kmer"

    def __init__(
        self,
        ref_codes: np.ndarray,
        k: int = 15,
        window: int = 10,
        max_occ: int = 0,
        device: torch.device | str | None = None,
    ):
        self.k, self.window = k, window
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        packed, valid = pack_kmers(ref_codes, k)
        selected = valid & minimizer_mask(hash64(packed), window)
        table = KmerTable.build(packed, selected, k, max_occ=max_occ)

        self.keys = torch.as_tensor(table.keys, device=self.device)
        self.offsets = torch.as_tensor(table.offsets, device=self.device, dtype=torch.int32)
        self.positions = torch.as_tensor(table.positions, device=self.device)
        self._table = table

    def lookup(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched ``(start, count)`` lookup for a flat tensor of k-mer codes."""
        queries = queries.to(self.device)
        if self.keys.numel() == 0:
            zeros = torch.zeros_like(queries, dtype=torch.int32)
            return zeros, zeros.clone()

        if self.device.type == "cuda":
            from ..accel.cuda_kernels import kmer_lookup as cuda_lookup

            try:
                return cuda_lookup(self.keys, self.offsets, queries)
            except RuntimeError:
                pass  # kernel unavailable; use the portable torch path

        idx = torch.searchsorted(self.keys, queries)
        clipped = idx.clamp(max=self.keys.numel() - 1)
        hit = self.keys[clipped] == queries
        start = torch.where(hit, self.offsets[clipped], torch.zeros_like(idx, dtype=torch.int32))
        count = torch.where(
            hit,
            self.offsets[clipped + 1] - self.offsets[clipped],
            torch.zeros_like(idx, dtype=torch.int32),
        )
        return start, count

    def query(self, read_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        packed, valid = pack_kmers(read_codes, self.k)
        selected = valid & minimizer_mask(hash64(packed), self.window)
        pos = np.flatnonzero(selected)
        if pos.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()

        start, count = self.lookup(torch.as_tensor(packed[pos]))
        start_np = start.cpu().numpy().astype(np.int64)
        count_np = count.cpu().numpy().astype(np.int64)

        total = int(count_np.sum())
        if total == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()

        read_pos = np.repeat(pos, count_np)
        group_base = np.repeat(np.concatenate([[0], np.cumsum(count_np)[:-1]]), count_np)
        within = np.arange(total, dtype=np.int64) - group_base
        ref_pos = self._table.positions[np.repeat(start_np, count_np) + within]
        return read_pos, ref_pos, np.full(total, self.k, dtype=np.int64)


class SMEMIndex:
    """Super-maximal exact match seeding over an :class:`FMIndex`.

    SMEMs are the most informative exact seeds available: each is as long as the
    reference allows, so a single SMEM often spans hundreds of bases where a
    fixed-``k`` index would emit dozens of redundant anchors.
    """

    name = "smem"

    def __init__(
        self,
        ref_codes: np.ndarray,
        min_seed_len: int = 13,
        max_occ: int = 200,
        sa_sample: int = 8,
        occ_sample: int = 64,
        fm: Optional[FMIndex] = None,
    ):
        self.fm = fm or FMIndex(ref_codes, sa_sample=sa_sample, occ_sample=occ_sample)
        self.min_seed_len = min_seed_len
        self.max_occ = max_occ

    def query(self, read_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts, lengths, sa_lo = self.fm.smems(
            read_codes, min_len=self.min_seed_len, max_occ=self.max_occ
        )
        if starts.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()

        # Re-derive each SMEM's interval width, then locate its occurrences.
        read_pos_all, ref_pos_all, length_all = [], [], []
        for start, length, lo in zip(starts, lengths, sa_lo):
            pattern = read_codes[start : start + length]
            interval_lo, interval_hi = self.fm.count(pattern)
            if interval_hi <= interval_lo:
                continue
            if self.max_occ and interval_hi - interval_lo > self.max_occ:
                continue
            positions = self.fm.locate(np.arange(interval_lo, interval_hi, dtype=np.int64))
            read_pos_all.append(np.full(len(positions), start, dtype=np.int64))
            ref_pos_all.append(positions)
            length_all.append(np.full(len(positions), length, dtype=np.int64))

        if not read_pos_all:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()
        return (
            np.concatenate(read_pos_all),
            np.concatenate(ref_pos_all),
            np.concatenate(length_all),
        )


class ExactKmerIndex:
    """Exact fixed-``k`` seeding through the FM-index (BWA-MEM style ``fmindex`` mode).

    Every ``k``-mer of the read is looked up by backward search. Slower than the
    hash table for the same ``k``, but it shares the FM-index with SMEM seeding
    and reports occurrence counts exactly.
    """

    name = "fmindex"

    def __init__(
        self,
        ref_codes: np.ndarray,
        k: int = 15,
        stride: int = 5,
        max_occ: int = 200,
        fm: Optional[FMIndex] = None,
        sa_sample: int = 8,
        occ_sample: int = 64,
    ):
        self.fm = fm or FMIndex(ref_codes, sa_sample=sa_sample, occ_sample=occ_sample)
        self.k, self.stride, self.max_occ = k, stride, max_occ

    def query(self, read_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        read_pos_all, ref_pos_all = [], []
        for start in range(0, max(len(read_codes) - self.k + 1, 0), self.stride):
            positions = self.fm.occurrences(
                read_codes[start : start + self.k], max_occ=self.max_occ
            )
            if positions.size:
                read_pos_all.append(np.full(len(positions), start, dtype=np.int64))
                ref_pos_all.append(positions)

        if not read_pos_all:
            z = np.zeros(0, dtype=np.int64)
            return z, z.copy(), z.copy()
        read_pos = np.concatenate(read_pos_all)
        return read_pos, np.concatenate(ref_pos_all), np.full(len(read_pos), self.k, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Reference bundle + stage entry point
# --------------------------------------------------------------------------- #
@dataclass
class SeedIndexBundle:
    """Every index built over one reference, keyed by seeding mode.

    Built once per reference and reused across reads — index construction
    dominates seeding cost, so the pipeline caches these.
    """

    ref_id: int
    ref_codes: np.ndarray
    indices: dict[str, object]
    node_starts: Optional[np.ndarray] = None  # graph node -> reference offset
    node_ids: Optional[np.ndarray] = None
    backbone: Optional[np.ndarray] = None  # bool per node: on the reference path

    @property
    def ref_len(self) -> int:
        return len(self.ref_codes)

    def node_of(self, ref_pos: np.ndarray) -> np.ndarray:
        """Map forward-reference offsets to graph node ids (``-1`` when unknown)."""
        if self.node_starts is None or len(self.node_starts) == 0:
            return np.full(len(ref_pos), -1, dtype=np.int64)
        slot = np.clip(
            np.searchsorted(self.node_starts, ref_pos, side="right") - 1,
            0,
            len(self.node_starts) - 1,
        )
        ids = self.node_ids if self.node_ids is not None else slot
        return np.asarray(ids, dtype=np.int64)[slot]


class SeedingEngine:
    """Stage 1 entry point: build the configured indices, then seed reads.

    ``SeedingConfig.modes`` may list several indices; their anchors are merged and
    then collapsed on shared diagonals, because different indices routinely
    rediscover the same match and un-merged duplicates inflate the chaining DP
    for no gain.
    """

    def __init__(self, cfg: SeedingConfig | None = None, device: torch.device | str | None = None):
        self.cfg = cfg or SeedingConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    # ---- index construction ------------------------------------------------ #
    def build_indices(
        self,
        ref_seq: str,
        ref_id: int = 0,
        node_seqs: Optional[Sequence[str]] = None,
        node_ref_start: Optional[Sequence[int]] = None,
        backbone_path: Optional[Sequence[int]] = None,
    ) -> SeedIndexBundle:
        """Build every index named in ``cfg.modes`` over one reference."""
        cfg = self.cfg
        ref_codes = encode_bases(ref_seq)
        indices: dict[str, object] = {}

        # SMEM and exact-k-mer seeding share one FM-index.
        shared_fm: Optional[FMIndex] = None
        if {"smem", "fmindex"} & set(cfg.modes):
            shared_fm = FMIndex(
                ref_codes, sa_sample=cfg.fm_sa_sample, occ_sample=cfg.fm_occ_sample
            )

        for mode in cfg.modes:
            if mode == "minimizer":
                indices[mode] = MinimizerIndex(
                    ref_codes, k=cfg.kmer, window=cfg.window, max_occ=cfg.max_occ
                )
            elif mode == "smem":
                indices[mode] = SMEMIndex(
                    ref_codes,
                    min_seed_len=cfg.min_seed_len,
                    max_occ=cfg.max_occ,
                    fm=shared_fm,
                )
            elif mode == "fmindex":
                indices[mode] = ExactKmerIndex(
                    ref_codes, k=cfg.kmer, max_occ=cfg.max_occ, fm=shared_fm
                )
            elif mode == "dbg":
                seqs = node_seqs if node_seqs is not None else [ref_seq]
                indices[mode] = DeBruijnIndex(
                    [encode_bases(s) for s in seqs], k=cfg.dbg_kmer, max_occ=cfg.max_occ
                )
            elif mode == "fuzzy":
                indices[mode] = FuzzySeedIndex(
                    ref_codes, cfg.spaced_pattern, max_occ=cfg.max_occ
                )
            elif mode == "multiplex_dbg":
                indices[mode] = MultiplexDBG(
                    ref_codes, kmers=cfg.multiplex_kmers, max_occ=cfg.max_occ
                )
            elif mode == "gpu_kmer":
                indices[mode] = GPUKmerIndex(
                    ref_codes,
                    k=cfg.kmer,
                    window=cfg.window,
                    max_occ=cfg.max_occ,
                    device=self.device,
                )
            else:  # pragma: no cover - guarded by SeedingConfig validation
                raise ValueError(f"Unknown seeding mode {mode!r}")

        node_starts = (
            np.asarray(
                [s for s in node_ref_start if s >= 0] if node_ref_start is not None else [],
                dtype=np.int64,
            )
            if node_ref_start is not None
            else None
        )
        node_ids = None
        if node_ref_start is not None:
            kept = [(i, s) for i, s in enumerate(node_ref_start) if s >= 0]
            kept.sort(key=lambda pair: pair[1])
            node_ids = np.asarray([i for i, _ in kept], dtype=np.int64)
            node_starts = np.asarray([s for _, s in kept], dtype=np.int64)

        backbone = None
        if backbone_path is not None and node_seqs is not None:
            backbone = np.zeros(len(node_seqs), dtype=bool)
            backbone[np.asarray(backbone_path, dtype=np.int64)] = True

        return SeedIndexBundle(
            ref_id=ref_id,
            ref_codes=ref_codes,
            indices=indices,
            node_starts=node_starts,
            node_ids=node_ids,
            backbone=backbone,
        )

    # ---- seeding ----------------------------------------------------------- #
    def seed_read(self, read_seq: str, bundle: SeedIndexBundle) -> AnchorSet:
        """Seed one read against a prebuilt reference bundle.

        The read is seeded in both orientations when ``cfg.both_strands`` is set.
        Reverse-strand anchor ``read_pos`` values are offsets into the
        *reverse-complemented* read, which keeps the reference coordinate
        increasing along the anchor chain (the same convention SAM uses when it
        stores reverse-strand reads reoriented).
        """
        forward = encode_bases(read_seq)
        orientations = [(1, forward)]
        if self.cfg.both_strands:
            orientations.append((-1, reverse_complement_codes(forward)))

        anchors = AnchorSet.empty(read_len=len(forward), ref_len=bundle.ref_len)
        for strand, codes in orientations:
            anchors = anchors.concat(self._seed_one_orientation(codes, strand, bundle))

        anchors = self._merge_diagonals(anchors)
        anchors = self._cap(anchors)
        anchors.node_id = bundle.node_of(anchors.ref_pos)
        return anchors

    def _seed_one_orientation(
        self, codes: np.ndarray, strand: int, bundle: SeedIndexBundle
    ) -> AnchorSet:
        parts = AnchorSet.empty(read_len=len(codes), ref_len=bundle.ref_len)
        for mode, index in bundle.indices.items():
            result = index.query(codes)
            if mode == "dbg":
                read_pos, ref_pos, length, node_id = result
            else:
                read_pos, ref_pos, length = result
                node_id = None
            if len(read_pos) == 0:
                continue
            parts = parts.concat(
                AnchorSet.from_lists(
                    read_pos=read_pos,
                    ref_pos=ref_pos,
                    length=length,
                    strand=np.full(len(read_pos), strand, dtype=np.int8),
                    node_id=node_id,
                    source=np.full(len(read_pos), source_id(mode), dtype=np.int8),
                    read_len=len(codes),
                    ref_len=bundle.ref_len,
                )
            )
        return parts

    def _merge_diagonals(self, anchors: AnchorSet) -> AnchorSet:
        """Collapse anchors that describe the same match.

        Two anchors on the same strand and diagonal whose read intervals touch or
        overlap are one exact match seen through two indices (or two adjacent
        k-mers of one longer match), so they are merged into the longer anchor.
        """
        if len(anchors) <= 1:
            return anchors

        slack = self.cfg.merge_diagonal_slack
        diagonal = anchors.diagonal
        order = np.lexsort((anchors.read_pos, diagonal, anchors.strand))
        ordered = anchors.take(order)

        same_group = np.zeros(len(ordered), dtype=bool)
        same_group[1:] = (ordered.strand[1:] == ordered.strand[:-1]) & (
            ordered.diagonal[1:] == ordered.diagonal[:-1]
        )
        gap = np.zeros(len(ordered), dtype=np.int64)
        gap[1:] = ordered.read_pos[1:] - ordered.read_end[:-1]
        mergeable = same_group & (gap <= slack)

        keep: list[int] = []
        read_pos = ordered.read_pos.copy()
        length = ordered.length.copy()
        for i in range(len(ordered)):
            if i > 0 and mergeable[i] and keep:
                last = keep[-1]
                end = max(read_pos[last] + length[last], ordered.read_end[i])
                length[last] = end - read_pos[last]
            else:
                keep.append(i)

        merged = ordered.take(np.asarray(keep, dtype=np.int64))
        merged.length = length[np.asarray(keep, dtype=np.int64)]
        return merged

    def _cap(self, anchors: AnchorSet) -> AnchorSet:
        """Keep only the ``max_anchors`` longest anchors.

        Length is the best label-free proxy for anchor reliability, so trimming
        by length keeps the informative seeds when a repeat floods the set.
        """
        limit = self.cfg.max_anchors
        if limit <= 0 or len(anchors) <= limit:
            return anchors
        keep = np.argsort(-anchors.length, kind="stable")[:limit]
        return anchors.take(np.sort(keep))
