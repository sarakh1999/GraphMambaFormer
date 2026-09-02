"""Data types carried between the alignment stages.

The stages communicate through arrays, not objects: Stage 1 emits an
:class:`AnchorSet` of parallel NumPy arrays, Stage 2 turns those into
:class:`Chain` objects, and Stage 3 attaches an :class:`ExtensionResult`. Keeping
anchors columnar is what makes the batched DP kernels possible — a per-anchor
Python object would dominate the runtime of every stage that touches them.

Coordinates are always **0-based**, half-open, and on the *forward* reference.
An anchor from a reverse-strand seed stores its read position in read
orientation and carries ``strand == -1``; :meth:`AnchorSet.for_strand` splits the
set so each strand is chained independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

# CIGAR operators used throughout (the extended set, as in the truth SAM).
CIGAR_OPS: tuple[str, ...] = ("=", "X", "I", "D", "S", "H", "M", "N")


@dataclass
class AnchorSet:
    """Columnar set of seed anchors for one read.

    Every array has length ``n``. An anchor is the pair "read offset
    ``read_pos``, forward-reference offset ``ref_pos``, matching for ``length``
    bases", so its *diagonal* is ``ref_pos - read_pos``.

    Attributes:
        read_pos: start offset in the read (read orientation).
        ref_pos: start offset on the forward reference.
        length: seed span in bases (exact-match length for contiguous indices;
            full spaced-pattern span for fuzzy seeds, which may include
            don't-care mismatches).
        strand: ``+1`` or ``-1`` per anchor.
        node_id: pangenome graph node containing ``ref_pos`` (``-1`` if unknown).
        source: index into :data:`SEEDING_SOURCES` recording which index emitted
            the anchor, so a mixed-mode run stays auditable.
        score: neural seed score in ``[0, 1]``, filled in by Stage 4
            (``NaN`` until then).
    """

    read_pos: np.ndarray
    ref_pos: np.ndarray
    length: np.ndarray
    strand: np.ndarray
    node_id: np.ndarray
    source: np.ndarray
    score: np.ndarray

    read_len: int = 0
    ref_len: int = 0

    def __post_init__(self) -> None:
        n = len(self.read_pos)
        for name in ("ref_pos", "length", "strand", "node_id", "source", "score"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"AnchorSet field {name!r} has inconsistent length")

    def __len__(self) -> int:
        return len(self.read_pos)

    @property
    def diagonal(self) -> np.ndarray:
        """``ref_pos - read_pos`` — anchors on one alignment share a diagonal."""
        return self.ref_pos.astype(np.int64) - self.read_pos.astype(np.int64)

    @property
    def read_end(self) -> np.ndarray:
        return self.read_pos + self.length

    @property
    def ref_end(self) -> np.ndarray:
        return self.ref_pos + self.length

    @classmethod
    def empty(cls, read_len: int = 0, ref_len: int = 0) -> "AnchorSet":
        z_int = np.zeros(0, dtype=np.int64)
        return cls(
            read_pos=z_int.copy(),
            ref_pos=z_int.copy(),
            length=z_int.copy(),
            strand=np.zeros(0, dtype=np.int8),
            node_id=z_int.copy(),
            source=np.zeros(0, dtype=np.int8),
            score=np.zeros(0, dtype=np.float32),
            read_len=read_len,
            ref_len=ref_len,
        )

    @classmethod
    def from_lists(
        cls,
        read_pos: Sequence[int],
        ref_pos: Sequence[int],
        length: Sequence[int],
        strand: Sequence[int],
        node_id: Optional[Sequence[int]] = None,
        source: Optional[Sequence[int]] = None,
        read_len: int = 0,
        ref_len: int = 0,
    ) -> "AnchorSet":
        n = len(read_pos)
        return cls(
            read_pos=np.asarray(read_pos, dtype=np.int64),
            ref_pos=np.asarray(ref_pos, dtype=np.int64),
            length=np.asarray(length, dtype=np.int64),
            strand=np.asarray(strand, dtype=np.int8),
            node_id=(
                np.asarray(node_id, dtype=np.int64)
                if node_id is not None
                else np.full(n, -1, dtype=np.int64)
            ),
            source=(
                np.asarray(source, dtype=np.int8)
                if source is not None
                else np.zeros(n, dtype=np.int8)
            ),
            score=np.full(n, np.nan, dtype=np.float32),
            read_len=read_len,
            ref_len=ref_len,
        )

    def take(self, idx: np.ndarray) -> "AnchorSet":
        """Reorder / subset the anchors by index."""
        idx = np.asarray(idx, dtype=np.int64)
        return AnchorSet(
            read_pos=self.read_pos[idx],
            ref_pos=self.ref_pos[idx],
            length=self.length[idx],
            strand=self.strand[idx],
            node_id=self.node_id[idx],
            source=self.source[idx],
            score=self.score[idx],
            read_len=self.read_len,
            ref_len=self.ref_len,
        )

    def for_strand(self, strand: int) -> tuple["AnchorSet", np.ndarray]:
        """Anchors on one strand, plus their indices in the original set."""
        idx = np.flatnonzero(self.strand == strand)
        return self.take(idx), idx

    def sorted_by_ref(self) -> tuple["AnchorSet", np.ndarray]:
        """Anchors ordered by reference end, then read end (chaining DP order)."""
        order = np.lexsort((self.read_end, self.ref_end))
        return self.take(order), order

    def concat(self, other: "AnchorSet") -> "AnchorSet":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        return AnchorSet(
            read_pos=np.concatenate([self.read_pos, other.read_pos]),
            ref_pos=np.concatenate([self.ref_pos, other.ref_pos]),
            length=np.concatenate([self.length, other.length]),
            strand=np.concatenate([self.strand, other.strand]),
            node_id=np.concatenate([self.node_id, other.node_id]),
            source=np.concatenate([self.source, other.source]),
            score=np.concatenate([self.score, other.score]),
            read_len=self.read_len or other.read_len,
            ref_len=self.ref_len or other.ref_len,
        )

    def to_seed_features(self) -> np.ndarray:
        """A ``(n, 12)`` geometric feature matrix for the neural seed scorer.

        Deliberately layout-compatible with ``Seed.features`` from the synthetic
        dataset, so a scorer trained on ground-truth seeds accepts anchors
        produced by any of the Stage 1 indices. The three columns the dataset
        fills from the reference (local GC, repeat overlap, base quality) are
        left at zero here and populated by the caller when it has that context.
        """
        n = len(self)
        if n == 0:
            return np.zeros((0, 12), dtype=np.float32)

        read_len = max(self.read_len, 1)
        ref_len = max(self.ref_len, 1)
        order = np.argsort(self.read_pos, kind="stable")
        gap_prev = np.zeros(n, dtype=np.float64)
        gap_next = np.zeros(n, dtype=np.float64)
        sorted_pos = self.read_pos[order].astype(np.float64)
        if n > 1:
            gap_prev[order[1:]] = sorted_pos[1:] - sorted_pos[:-1]
            gap_next[order[:-1]] = sorted_pos[1:] - sorted_pos[:-1]

        feats = np.zeros((n, 12), dtype=np.float32)
        feats[:, 0] = self.read_pos / read_len
        feats[:, 1] = self.ref_pos / ref_len
        feats[:, 2] = self.diagonal / ref_len
        feats[:, 3] = self.strand
        feats[:, 4] = self.length / read_len
        # 5 local GC, 6 repeat overlap: reference context, filled by the caller.
        feats[:, 7] = 1.0 / np.maximum(self.length, 1)  # uniqueness proxy
        # 8 mean base quality: filled by the caller.
        feats[:, 9] = (self.node_id >= 0).astype(np.float32)
        feats[:, 10] = gap_prev / read_len
        feats[:, 11] = gap_next / read_len
        return feats


#: Which Stage 1 index produced an anchor (values stored in ``AnchorSet.source``).
SEEDING_SOURCES: tuple[str, ...] = (
    "minimizer",
    "smem",
    "fmindex",
    "dbg",
    "fuzzy",
    "multiplex_dbg",
    "gpu_kmer",
)


def source_id(name: str) -> int:
    return SEEDING_SOURCES.index(name)


@dataclass
class Chain:
    """A collinear run of anchors: Stage 2's output, Stage 3's input.

    Attributes:
        anchor_idx: indices into the read's :class:`AnchorSet`, in read order.
        score: chaining DP score.
        strand: strand shared by every member anchor.
        read_start / read_end: read span covered by the chain (half-open).
        ref_start / ref_end: forward-reference span covered (half-open).
        neural_score: chain score from the Stage 4 re-ranker (``NaN`` until then).
        is_primary: set during primary/secondary selection.
    """

    anchor_idx: np.ndarray
    score: float
    strand: int
    read_start: int
    read_end: int
    ref_start: int
    ref_end: int
    neural_score: float = float("nan")
    is_primary: bool = False

    def __len__(self) -> int:
        return len(self.anchor_idx)

    @property
    def read_span(self) -> int:
        return self.read_end - self.read_start

    @property
    def ref_span(self) -> int:
        return self.ref_end - self.ref_start

    def coverage(self, read_len: int) -> float:
        """Fraction of the read spanned by the chain."""
        return self.read_span / max(read_len, 1)

    def anchor_bases(self, anchors: AnchorSet) -> int:
        """Total exactly-matched bases in the chain, without double counting.

        Anchors from overlapping seeds can share read positions, so the raw
        length sum overstates coverage; this merges the read intervals first.
        """
        if len(self.anchor_idx) == 0:
            return 0
        starts = anchors.read_pos[self.anchor_idx]
        ends = anchors.read_end[self.anchor_idx]
        order = np.argsort(starts, kind="stable")
        starts, ends = starts[order], ends[order]
        total = 0
        cur_start, cur_end = starts[0], ends[0]
        for s, e in zip(starts[1:], ends[1:]):
            if s > cur_end:
                total += cur_end - cur_start
                cur_start, cur_end = s, e
            else:
                cur_end = max(cur_end, e)
        return int(total + cur_end - cur_start)


@dataclass
class ExtensionResult:
    """Base-level alignment of one chain (Stage 3).

    Attributes:
        score: affine-gap alignment score.
        cigar: run-length ops over the read using ``=``/``X``/``I``/``D``/``S``.
        read_start / read_end: aligned read span, excluding soft clips.
        ref_start / ref_end: aligned forward-reference span.
        n_match / n_mismatch / n_insertion / n_deletion: op tallies.
    """

    score: float
    cigar: list[tuple[str, int]]
    read_start: int
    read_end: int
    ref_start: int
    ref_end: int
    n_match: int = 0
    n_mismatch: int = 0
    n_insertion: int = 0
    n_deletion: int = 0

    @property
    def cigar_string(self) -> str:
        return "".join(f"{n}{op}" for op, n in self.cigar)

    @property
    def identity(self) -> float:
        """Matches over aligned columns; 0.0 for an empty alignment."""
        total = self.n_match + self.n_mismatch + self.n_insertion + self.n_deletion
        return self.n_match / total if total else 0.0

    @property
    def edit_distance(self) -> int:
        return self.n_mismatch + self.n_insertion + self.n_deletion


@dataclass
class AlignmentRecord:
    """The pipeline's per-read output — one alignment, ready to become a SAM row.

    ``stages`` records which stages actually ran for this read (the router and
    the two-pass aligner both skip work), which is what makes a pipeline run
    auditable after the fact.
    """

    read_id: str
    read_len: int
    ref_id: int = -1
    ref_start: int = -1
    ref_end: int = -1
    strand: int = 1
    mapq: int = 0
    chain_score: float = 0.0
    alignment_score: float = 0.0
    cigar: list[tuple[str, int]] = field(default_factory=list)
    node_id: int = -1
    is_mapped: bool = False
    is_primary: bool = True
    is_supplementary: bool = False
    route: str = "full"
    pass_name: str = "hybrid"
    n_anchors: int = 0
    n_chains: int = 0
    stages: tuple[str, ...] = ()
    extension: Optional[ExtensionResult] = None

    @property
    def cigar_string(self) -> str:
        return "".join(f"{n}{op}" for op, n in self.cigar)

    @classmethod
    def unmapped(cls, read_id: str, read_len: int, **kwargs) -> "AlignmentRecord":
        return cls(read_id=read_id, read_len=read_len, is_mapped=False, **kwargs)


@dataclass
class ReadAlignments:
    """Every candidate for one read, with the primary called out.

    Stage 4 needs the full candidate list to re-rank and to derive a MAPQ from
    the primary/secondary score margin, so the pipeline keeps them together
    rather than collapsing to the best hit too early.
    """

    read_id: str
    read_len: int
    anchors: AnchorSet
    chains: list[Chain] = field(default_factory=list)
    records: list[AlignmentRecord] = field(default_factory=list)
    #: Per-read slices of enabled multi-task head outputs. Keys remain the
    #: configured head names; values are detached NumPy arrays.
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    def __iter__(self) -> Iterator[AlignmentRecord]:
        return iter(self.records)

    @property
    def primary(self) -> Optional[AlignmentRecord]:
        for record in self.records:
            if record.is_primary:
                return record
        return self.records[0] if self.records else None


# --------------------------------------------------------------------------- #
# CIGAR helpers
# --------------------------------------------------------------------------- #
def run_length_encode(ops: Sequence[str]) -> list[tuple[str, int]]:
    """Collapse a per-column op string into ``(op, count)`` runs."""
    if not ops:
        return []
    out: list[tuple[str, int]] = []
    current, count = ops[0], 1
    for op in ops[1:]:
        if op == current:
            count += 1
        else:
            out.append((current, count))
            current, count = op, 1
    out.append((current, count))
    return out


def cigar_read_length(cigar: Sequence[tuple[str, int]]) -> int:
    """Bases the CIGAR consumes from the read (``=``, ``X``, ``M``, ``I``, ``S``)."""
    return sum(n for op, n in cigar if op in ("=", "X", "M", "I", "S"))


def cigar_ref_length(cigar: Sequence[tuple[str, int]]) -> int:
    """Bases the CIGAR consumes from the reference (``=``, ``X``, ``M``, ``D``, ``N``)."""
    return sum(n for op, n in cigar if op in ("=", "X", "M", "D", "N"))


def merge_cigar(cigar: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """Merge adjacent runs of the same op and drop zero-length runs."""
    out: list[tuple[str, int]] = []
    for op, n in cigar:
        if n <= 0:
            continue
        if out and out[-1][0] == op:
            out[-1] = (op, out[-1][1] + n)
        else:
            out.append((op, n))
    return out
