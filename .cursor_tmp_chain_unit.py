"""Lightweight unit check of the decoy-chain fix (no model/pipeline)."""
from __future__ import annotations

import numpy as np
import torch

from graphmambaformer.alignment.types import AnchorSet, Chain
from graphmambaformer.alignment.scoring import chain_features
from graphmambaformer.config import LossConfig
from graphmambaformer.losses import GraphMambaLoss
from graphmambaformer.training import TargetBuilder


class _Pipe:
    """Minimal stand-in so TargetBuilder.__init__ doesn't need a real pipeline."""
    model = None


def make_anchor_set(k: int) -> AnchorSet:
    # k collinear anchors on diagonal 100, read positions 0,10,20,...
    read_pos = np.arange(k, dtype=np.int64) * 10
    ref_pos = read_pos + 100
    length = np.full(k, 8, dtype=np.int64)
    strand = np.ones(k, dtype=np.int8)
    return AnchorSet.from_lists(read_pos, ref_pos, length, strand, read_len=200, ref_len=1000)


def make_chain(anchors: AnchorSet) -> Chain:
    idx = np.arange(len(anchors), dtype=np.int64)
    return Chain(
        anchor_idx=idx, score=float(anchors.length[idx].sum()), strand=1,
        read_start=int(anchors.read_pos.min()), read_end=int(anchors.read_end.max()),
        ref_start=int(anchors.ref_pos.min()), ref_end=int(anchors.ref_end.max()),
    )


tb = TargetBuilder(_Pipe(), model=None, decoy_chains=1)

# ---- 1. multi-anchor read: decoy is a strict sub-chain ----
for k in (1, 2, 5):
    anchors = make_anchor_set(k)
    primary = make_chain(anchors)
    aug = tb._augment_with_decoys([primary], anchors, read_len=200)
    assert len(aug) == 2, f"k={k}: expected 2 candidates, got {len(aug)}"
    decoy = aug[1]
    # features must be finite
    best = max(c.score for c in aug)
    for c in aug:
        f = chain_features(c, anchors, 200, best)
        assert np.isfinite(f).all(), f"k={k}: non-finite chain features"
    # truth = the primary's true span; label must remain the primary (index 0)
    lbl = tb._chain_label(aug, primary.ref_start, primary.ref_end)
    assert lbl == 0, f"k={k}: decoy stole the target (label={lbl})"
    print(f"k={k}: candidates={len(aug)} primary.score={primary.score:.1f} "
          f"decoy.score={decoy.score:.1f} decoy.ref=({decoy.ref_start},{decoy.ref_end}) label={lbl}")

# ---- 2. chain_loss is 0 for a singleton, > 0 for 2 candidates ----
torch.manual_seed(0)
crit = GraphMambaLoss(LossConfig()).alignment

# singleton candidate -> degenerate zero
logits1 = torch.randn(4, 1)
mask1 = torch.ones(4, 1, dtype=torch.bool)
tgt1 = torch.zeros(4, dtype=torch.long)
loss1 = float(crit.chain_loss(logits1, tgt1, mask1))

# two candidates -> real ranking signal
logits2 = torch.randn(4, 2)
mask2 = torch.ones(4, 2, dtype=torch.bool)
tgt2 = torch.zeros(4, dtype=torch.long)
loss2 = float(crit.chain_loss(logits2, tgt2, mask2))

print(f"chain_loss singleton={loss1:.6f}  two-candidate={loss2:.6f}")
assert loss1 == 0.0, "singleton should be exactly zero"
assert loss2 > 0.0, "two-candidate loss should be positive"
print("OK")
