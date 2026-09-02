#!/usr/bin/env python3
"""Prove the chain-ranking loss is no longer identically zero.

Root cause (from the analysis): a normal single-locus read yields exactly one
dominant chain, and a listwise cross-entropy over a single candidate is
-log(softmax([x])) = 0 with zero gradient. TargetBuilder now injects a
hard-negative decoy chain when the chainer collapses to one candidate, so the
ranking term has >=2 candidates to order.

This script checks, with no GPU and no dataset, that:
  1. AlignmentLoss.chain_loss == 0 for a lone candidate (the degenerate case), and
  2. TargetBuilder injects a decoy so a single-chain read becomes >=2 candidates, and
  3. chain_loss > 0 with a non-zero gradient once the decoy is present.

Run:  PYTHONPATH=. .venv/bin/python scripts/transfer/verify_chain_loss_fix.py
"""
from __future__ import annotations

import numpy as np
import torch

from graphmambaformer.alignment.types import AnchorSet, Chain
from graphmambaformer.losses.alignment_loss import AlignmentLoss
from graphmambaformer.training.targets import TargetBuilder


def _anchor_set(n: int) -> AnchorSet:
    """A tiny collinear anchor set (read_pos == ref_pos, unit diagonal)."""
    read_pos = np.arange(n, dtype=np.int64) * 10
    return AnchorSet.from_lists(
        read_pos=read_pos,
        ref_pos=read_pos.copy(),
        length=np.full(n, 8, dtype=np.int64),
        strand=np.ones(n, dtype=np.int8),
        read_len=int(read_pos[-1]) + 8 if n else 0,
        ref_len=int(read_pos[-1]) + 8 if n else 0,
    )


def main() -> int:
    loss = AlignmentLoss()

    # 1) Degenerate: a single candidate -> loss is exactly zero (the bug).
    lone = torch.tensor([[2.5]], requires_grad=True)  # (B=1, n_chain=1)
    lone_mask = torch.tensor([[True]])
    lone_target = torch.tensor([0])
    lone_loss = loss.chain_loss(lone, lone_target, lone_mask)
    assert float(lone_loss) == 0.0, f"expected 0 for 1 candidate, got {float(lone_loss)}"
    print(f"[1] single-candidate chain_loss = {float(lone_loss):.6f}  (degenerate, as expected)")

    # 2) TargetBuilder injects a decoy for a single-chain read.
    builder = TargetBuilder(pipeline=None, decoy_chains=1)
    anchors = _anchor_set(4)
    true_chain = Chain(
        anchor_idx=np.arange(4, dtype=np.int64),
        score=40.0,
        strand=1,
        read_start=0, read_end=38,
        ref_start=0, ref_end=38,
    )
    augmented = builder._augment_with_decoys([true_chain], anchors, read_len=64)
    assert len(augmented) >= 2, f"decoy not injected: {len(augmented)} chain(s)"
    print(f"[2] chains after decoy injection = {len(augmented)}  "
          f"(scores: {[round(c.score, 2) for c in augmented]})")

    # The truth still overlaps the primary most, so the ranking target is index 0.
    target_idx = builder._chain_label(augmented, ref_start=0, ref_end=38)
    assert target_idx == 0, f"expected primary (0) as target, got {target_idx}"
    print(f"[3] chain_target index = {target_idx}  (true/primary chain)")

    # 3) With >=2 candidates the listwise loss is > 0 and has a real gradient.
    logits = torch.tensor([[1.0, 0.5]], requires_grad=True)  # primary, decoy
    mask = torch.tensor([[True, True]])
    tgt = torch.tensor([target_idx])
    l = loss.chain_loss(logits, tgt, mask)
    l.backward()
    assert float(l) > 0.0, f"chain_loss should be > 0, got {float(l)}"
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0, "no gradient"
    print(f"[4] two-candidate chain_loss = {float(l):.6f}  "
          f"grad|.|sum = {float(logits.grad.abs().sum()):.6f}  (non-zero signal)")

    # Even when the model is undecided (equal logits) the loss is -log(0.5) > 0,
    # so every single-locus read now contributes gradient instead of nothing.
    eq = torch.tensor([[0.0, 0.0]], requires_grad=True)
    l_eq = loss.chain_loss(eq, torch.tensor([0]), torch.tensor([[True, True]]))
    print(f"[5] undecided (equal logits) chain_loss = {float(l_eq):.6f}  "
          f"(= -log 0.5 = {np.log(2):.6f})")

    print("\nOK: chain-ranking loss is no longer identically zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
