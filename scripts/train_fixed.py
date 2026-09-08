#!/usr/bin/env python
"""Fixed-stack training entrypoint — NEW name, originals untouched.

This is the ONLY wiring needed to run with the fixes in
``graphmambaformer/losses/graph_mamba_loss_fixed.py`` and
``graphmambaformer/training/run_fixes.py``. It does not edit a single existing
file: it patches the fixed loss class into the trainer's module namespace, then
runs the real ``scripts/train.py`` main in-process, so every CLI flag, the
torchrun/DDP wiring, the dataset cache, AMP and resume behave exactly as before.

The trainer builds its criterion as ``GraphMambaLoss(...)`` looked up in its own
module globals (``graphmambaformer.training.trainer.GraphMambaLoss``). Rebinding
that name to :class:`FixedGraphMambaLoss` before the trainer is instantiated
gives the fixed router objective and eager Kendall weights with zero changes to
``trainer.py`` or ``train.py``.

Usage: identical to scripts/train.py, e.g.
    torchrun --standalone --nproc_per_node=2 scripts/train_fixed.py --data real ...
Or via the convenience launcher: scripts/transfer/run_hg005_fixed.sh
"""
from __future__ import annotations

import os
import runpy

import graphmambaformer.alignment.pipeline as _pipeline
import graphmambaformer.training.trainer as _trainer
from graphmambaformer.alignment.seeding_fixed import FixedSeedingEngine
from graphmambaformer.losses.graph_mamba_loss_fixed import FixedGraphMambaLoss
from graphmambaformer.training.targets_fixed import FixedTargetBuilder

# Swap the fixed loss AND target builder in before any Trainer is constructed.
# The trainer looks both names up in its own module globals
# (``GraphMambaLoss`` at criterion build, ``TargetBuilder`` at builder build),
# so rebinding them here applies every fix with zero edits to trainer.py:
#   * FixedGraphMambaLoss  -> router load-balance + eager Kendall weights
#   * FixedTargetBuilder   -> pinned chain decoys + local position target
_trainer.GraphMambaLoss = FixedGraphMambaLoss
_trainer.TargetBuilder = FixedTargetBuilder

# Swap the fixed Stage-1 seeder in before any AlignmentPipeline is built. The
# pipeline constructs ``self.seeder = SeedingEngine(...)`` from its own module
# global (pipeline.py:219), so rebinding that global here makes every pipeline
# -- the trainer's and the target builder's -- use the support-aware anchor cap
# that keeps the true collinear run (fixes seed labels being ~all-negative, i.e.
# anchor AUC ~ 0.5, and gives chaining a real run to assemble). Zero edits to
# pipeline.py or seeding.py.
_pipeline.SeedingEngine = FixedSeedingEngine
print("[train_fixed] fixes active: "
      "FixedGraphMambaLoss (router+Kendall) + FixedTargetBuilder (chain+position) "
      "+ FixedSeedingEngine (support-aware anchor cap)")

# --------------------------------------------------------------------------- #
# Efficiency fix: cap the per-epoch full validation.
# The end-of-epoch validate() runs over ALL ~6194 val batches/rank (~5h), while
# the intra-epoch check is already capped (intra_val_max_batches=12). Capping
# validate() to EPOCH_VAL_MAX_BATCHES (default 200) keeps a solid held-out
# estimate but reclaims hours per epoch. 12 < cap, so intra-epoch is unchanged.
# Set EPOCH_VAL_MAX_BATCHES=0 to restore the full pass.
# --------------------------------------------------------------------------- #
_EPOCH_VAL_CAP = int(os.environ.get("EPOCH_VAL_MAX_BATCHES", "200"))
if _EPOCH_VAL_CAP > 0:
    _orig_validate = _trainer.Trainer.validate

    def _capped_validate(self, batches):
        if batches is not None and len(batches) > _EPOCH_VAL_CAP:
            batches = list(batches)[:_EPOCH_VAL_CAP]
        return _orig_validate(self, batches)

    _trainer.Trainer.validate = _capped_validate
    print(f"[train_fixed] epoch-end validation capped to {_EPOCH_VAL_CAP} batches "
          "(set EPOCH_VAL_MAX_BATCHES=0 to run the full pass)")

# Run the real train.py main with the current argv/env, as if invoked directly.
_train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
runpy.run_path(_train_py, run_name="__main__")
