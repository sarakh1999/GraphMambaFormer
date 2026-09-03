"""Lean verification that decoy injection de-degenerates the chain loss.

No model / pipeline: build a single-locus chain (what a normal HG005 read
produces), run TargetBuilder's decoy injection + labelling, then feed random
logits through the REAL AlignmentLoss.chain_loss to show the term is non-zero
with a gradient (vs identically 0 for the single-candidate case).
"""
from __future__ import annotations

import numpy as np
import torch

from graphmambaformer.alignment.types import AnchorSet, Chain
from graphmambaformer.training.targets import TargetBuilder
from graphmambaformer.losses import GraphMambaLoss
from graphmambaformer.config import LossConfig


def make_anchor_set(n=6, ref_start=1000):
    read_pos = (np.arange(n, dtype=np.int64) * 50).tolist()
    ref_pos = [ref_start + p for p in read_pos]
    length = [20] * n
    strand = [1] * n
    return AnchorSet.from_lists(
        read_pos=read_pos, ref_pos=ref_pos, length=length, strand=strand,
        read_len=int(read_pos[-1] + 20), ref_len=ref_start + int(read_pos[-1]) + 20,
    )


def one_primary_chain(anchors, ref_start=1000):
    idx = np.arange(len(anchors), dtype=np.int64)
    c = Chain(
        anchor_idx=idx, score=100.0, strand=1,
        read_start=int(anchors.read_pos.min()), read_end=int(anchors.read_end.max()),
        ref_start=int(anchors.ref_pos.min()), ref_end=int(anchors.ref_end.max()),
    )
    c.is_primary = True
    return c


class _FakeReadRef:
    pass


def run(decoys):
    anchors = make_anchor_set()
    primary = one_primary_chain(anchors)
    tb = TargetBuilder.__new__(TargetBuilder)  # skip __init__ (needs a pipeline)
    tb.decoy_chains = decoys
    tb.n_chain_features = 10

    chains = [primary]
    aug = tb._augment_with_decoys(chains, anchors, read_len=int(anchors.read_end.max()))
    n_cand = len(aug)
    ref_start = int(anchors.ref_pos.min())
    ref_end = int(anchors.ref_end.max())
    target = TargetBuilder._chain_label(aug, ref_start, ref_end)

    # Build the padded logits/mask/target exactly like the batch tensors.
    n_chain = max(1, n_cand)
    chain_mask = torch.zeros(1, n_chain, dtype=torch.bool)
    chain_mask[0, :n_cand] = True
    chain_target = torch.tensor([target], dtype=torch.long)

    torch.manual_seed(0)
    logits = torch.randn(1, n_chain, requires_grad=True)
    crit = GraphMambaLoss(LossConfig())
    loss = crit.alignment.chain_loss(logits, chain_target, chain_mask)
    loss.backward()
    grad_norm = float(logits.grad.norm())
    print(f"decoys={decoys}: candidates={n_cand} target={target} "
          f"chain_loss={float(loss):.6f} grad_norm={grad_norm:.6f}")


if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    print("OK")
