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

import graphmambaformer.training.trainer as _trainer
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
print("[train_fixed] fixes active: "
      "FixedGraphMambaLoss (router+Kendall) + FixedTargetBuilder (chain+position)")

# Run the real train.py main with the current argv/env, as if invoked directly.
_train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
runpy.run_path(_train_py, run_name="__main__")
