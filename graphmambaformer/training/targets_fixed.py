"""Fixed TargetBuilder — NEW name, original ``targets.py`` left untouched.

Mirrors the two *data-side* fixes into a new file so the originals keep their
names:

* **Chaining** — guarantees at least one hard-negative decoy chain
  (``decoy_chains >= 1``) regardless of the base default, so the listwise
  chain-ranking loss is never the degenerate single-candidate case (softmax of
  one element == 1.0, zero gradient). The base class already implements the
  decoy machinery (``_make_decoys`` / ``_augment_with_decoys``); this subclass
  just pins the count on so a future default change can't silently regress it.

* **Position head** — replaces the *global* position target
  (``read.ref_start / len(whole reference)``, which is ill-posed to regress from
  a single read embedding, so the head degenerates to predicting the mean) with
  the *local* within-window offset from
  :func:`..training.run_fixes.local_position_target`. That target is expressed
  relative to the placement the classical chainer already found, which the fused
  embedding can actually see, so it is learnable.

Because the trainer rebuilds supervision from ``(reads, reference)`` every step
(``targets.py`` is only run at train time, not baked into the dataset cache),
wiring this in via ``scripts/train_fixed.py`` takes effect immediately on the
existing cache — no cache rebuild required.

The position fix is experimental (needs validation): a one-time telemetry line
warns if the local target has ~0 spread on a batch, which would mean the head
has nothing informative to learn and ``position_window`` should be revisited.
"""
from __future__ import annotations

import threading
from typing import Sequence

from .run_fixes import (
    local_position_target,
    position_target_is_degenerate,
    recommended_decoy_chains,
)
from .targets import Supervision, TargetBuilder


class FixedTargetBuilder(TargetBuilder):
    def __init__(self, *args, position_window: float = 512.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Pin the chain-ranking loss to a non-degenerate >=2-candidate task.
        self.decoy_chains = max(int(self.decoy_chains), recommended_decoy_chains())
        self.position_window = float(position_window)

        # ``build`` runs concurrently on the prefetch worker threads, all sharing
        # one ``pipeline``. Capturing the chainer's output through thread-local
        # state (rather than a temporary monkey-patch of ``pipeline.chain``)
        # keeps the local-frame lookup race-free: each worker sees only the
        # chains it computed on its own thread.
        self._tls = threading.local()
        self._warned = False
        self._warn_lock = threading.Lock()
        self._install_chain_probe()

    def _install_chain_probe(self) -> None:
        """Wrap ``pipeline.chain`` once so every call records its result in TLS."""
        pipeline = self.pipeline
        if getattr(pipeline, "_fixed_chain_probe", False):
            return
        orig_chain = pipeline.chain
        tls = self._tls

        def _probe(anchor_sets, reference, *args, **kwargs):
            res = orig_chain(anchor_sets, reference, *args, **kwargs)
            tls.last_chains = res
            return res

        pipeline.chain = _probe
        pipeline._fixed_chain_probe = True

    def build(self, reads: Sequence, reference) -> Supervision:
        # Clear any stale capture, then let the base builder run seed+chain once.
        self._tls.last_chains = None
        sup = super().build(reads, reference)

        chains_per_read = getattr(self._tls, "last_chains", None)
        if chains_per_read is not None:
            tgt, valid = local_position_target(
                reads, chains_per_read, window=self.position_window
            )
            sup.targets["position_target"] = tgt
            sup.targets["position_valid"] = valid

            if not self._warned and position_target_is_degenerate(sup.targets):
                with self._warn_lock:
                    if not self._warned:
                        self._warned = True
                        print(
                            "[targets_fixed] WARNING: local position_target has ~0 "
                            "spread on this batch. The position head then has no "
                            "informative signal; revisit position_window "
                            f"(={self.position_window}) or confirm chain.ref_start "
                            "differs from read.ref_start before trusting it."
                        )
        return sup
