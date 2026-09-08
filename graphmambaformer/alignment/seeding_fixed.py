"""Fixed Stage-1 seeding — NEW name, original ``seeding.py`` left untouched.

Diagnosis (why the seed/anchor classifier trained to validation AUC ~ 0.5, i.e.
random, and why chaining kept collapsing to a single degenerate candidate):

The classical pipeline's anchors are used BOTH as the model's Stage-1 input AND
as the supervision labels (an anchor is a *positive* when its implied diagonal
``ref_pos - read_pos`` matches the read's true ``ref_start``). So if the anchor
SET handed to training does not actually contain the true, on-diagonal anchors,
every label is negative and the seed head has nothing to learn -> AUC ~ 0.5, and
chaining has no true collinear run to assemble -> one junk candidate.

That is exactly what the stock ``SeedingEngine._cap`` produces on repeat-rich
reads:

    keep = np.argsort(-anchors.length, kind="stable")[:limit]   # seeding.py:1172
    return anchors.take(np.sort(keep))                          #          :1173

* Minimizer / exact-k-mer anchors are ALL length ``k``, so ``-length`` is a flat
  key; the ``stable`` sort therefore keeps whatever came first in array order.
* Array order out of ``_merge_diagonals`` is ``lexsort((read_pos, diagonal,
  strand))`` -- i.e. ascending diagonal. Repetitive off-diagonal k-mers pile up
  at LOW diagonals and crowd out the true (usually large) diagonal.
* Net effect on a repeat read (empirically): 2000+ anchors, ~20 of them the true
  collinear run, and ``_cap`` keeps 32 low-diagonal repeat anchors with ZERO true
  positives among them. Training then sees 32 negatives per read.

The fix here does not use any label / ground-truth information (it must not --
the same code runs at inference). It keeps anchors by *diagonal support*: a real
alignment deposits a long collinear run of anchors on ONE diagonal, whereas
repeats scatter one-off anchors across many diagonals. Ranking anchors by
``(support_of_its_diagonal, length)`` and keeping the top ``max_anchors`` retains
that true run, so the true anchors survive the cap and the labels become
informative. Anchors are then returned in read-position order so any downstream
"first N" slice (the trainer builds a fixed ``(B, max_anchors)`` tensor) is a
representative spatial sample rather than a diagonal-clustered one.

Also provided:

* :func:`minimizer_mask_one_per_window` -- a leftmost-tie, one-minimizer-per-window
  sketch matching the synthetic ground-truth generator (``data/synthetic.py``),
  for callers that want the pipeline's seed set to line up exactly with the
  labels the dataset was built from. NOT wired in by default (it changes the seed
  set), but importable for ablations.

Wire it in with ``scripts/train_fixed.py`` (which rebinds
``graphmambaformer.alignment.pipeline.SeedingEngine`` to
:class:`FixedSeedingEngine` before the pipeline is constructed). No existing file
is edited.
"""
from __future__ import annotations

import os

import numpy as np

from .seeding import SeedingEngine
from .types import AnchorSet


def minimizer_mask_one_per_window(hashes: np.ndarray, window: int) -> np.ndarray:
    """One minimizer per window, leftmost k-mer on a hash tie.

    Matches ``data/synthetic.py``'s sketch (``min(valid, key=(hash, pos))`` per
    window, de-duplicated by position). The stock ``seeding.minimizer_mask`` keeps
    *every* k-mer tying the window minimum, which can emit extra repeat-derived
    minimizers that are absent from the labels the synthetic data was generated
    with. Use this when you need the pipeline's seed set to match those labels.
    """
    m = len(hashes)
    if m == 0:
        return np.zeros(0, dtype=bool)
    w = max(1, min(window, m))
    n_windows = m - w + 1
    selected = np.zeros(m, dtype=bool)
    for start in range(n_windows):
        window_slice = hashes[start : start + w]
        # argmin returns the FIRST occurrence -> leftmost on ties.
        selected[start + int(np.argmin(window_slice))] = True
    return selected


class FixedSeedingEngine(SeedingEngine):
    """SeedingEngine whose anchor cap keeps the true collinear run.

    Overrides only :meth:`_cap`; every other stage (index build, per-orientation
    seeding, diagonal merge, node projection) is inherited unchanged.

    Env knobs (all optional):
        GMF_SEED_MAX_OCC   -- override ``cfg.max_occ`` for THIS engine only (via a
            private cfg copy, so the shared config object is not mutated). The
            stock code DROPS every reference k-mer whose occurrence count exceeds
            ``max_occ``, which on a periodic/repeat locus can delete the only key
            and yield an EMPTY anchor set. Raising or disabling (0) it keeps those
            keys; the support-aware cap below then trims the resulting blow-up
            safely. Default: leave ``cfg.max_occ`` untouched.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        override = os.environ.get("GMF_SEED_MAX_OCC")
        if override is not None:
            try:
                new_occ = int(override)
            except ValueError:
                new_occ = None
            if new_occ is not None and int(getattr(self.cfg, "max_occ", 0)) != new_occ:
                # Copy the config so we never mutate an object shared with other
                # pipeline components / references.
                try:
                    import dataclasses

                    self.cfg = dataclasses.replace(self.cfg, max_occ=new_occ)
                except Exception:
                    import copy

                    self.cfg = copy.copy(self.cfg)
                    try:
                        self.cfg.max_occ = new_occ
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    def _cap(self, anchors: AnchorSet) -> AnchorSet:
        """Keep ``max_anchors`` anchors, preferring well-supported diagonals.

        Ranking key per anchor: ``(support, length)`` where ``support`` is the
        number of anchors sharing its ``(strand, diagonal)``. A true alignment's
        collinear run has high support and therefore survives; scattered repeat
        anchors (support 1) are trimmed first. Length breaks ties among equally
        supported diagonals (favours MEM/SMEM seeds over bare k-mers). The kept
        anchors are returned in read-position order so a downstream fixed-size
        ``[:N]`` slice is a representative sample of the read.
        """
        limit = self.cfg.max_anchors
        if limit <= 0 or len(anchors) <= limit:
            return anchors

        strand = anchors.strand.astype(np.int64)
        diagonal = anchors.diagonal.astype(np.int64)

        # Per-anchor support = size of its (strand, diagonal) group.
        pairs = np.stack([strand, diagonal], axis=1)
        _, inverse, counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True
        )
        support = counts[inverse].astype(np.float64)
        length = anchors.length.astype(np.float64)

        # Composite descending key: support dominates, length breaks ties. Encode
        # as one float so a single stable argsort yields the ranking (support gets
        # a large multiplier that ``length`` can never overtake for realistic k).
        key = support * 1.0e9 + length
        keep = np.argsort(-key, kind="stable")[:limit]
        kept = anchors.take(np.sort(keep))

        # Representative order for any downstream first-N slice.
        order = np.argsort(kept.read_pos, kind="stable")
        return kept.take(order)
