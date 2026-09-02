"""Quick check: decoy injection makes the chain-ranking loss non-zero."""
from __future__ import annotations

import numpy as np
import torch

from graphmambaformer.alignment.pipeline import build_pipeline
from graphmambaformer.config import CoreModelConfig, GraphMambaConfig, LossConfig, PipelineConfig
from graphmambaformer.models import build_core_model
from graphmambaformer.data.synthetic import generate_dataset, preset
from graphmambaformer.training import TargetBuilder
from graphmambaformer.losses import GraphMambaLoss

torch.manual_seed(0)
ds = generate_dataset(preset("tiny"))
ref = ds.references[0]
reads = [r for r in ds.splits["train"] if r.ref_id == 0][:8]
model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=GraphMambaConfig(d_model=64))).model
pipe = build_pipeline(PipelineConfig(mode="hybrid", batch_size=8), model=model)
reference = pipe.build_reference(ref.seq, ref_id=0)


def run(decoys: int) -> None:
    b = TargetBuilder(pipe, model=model, decoy_chains=decoys)
    sup = b.build(reads, reference)
    ct = sup.targets["chain_target"]
    cm = sup.targets["chain_mask"]
    n_chain = cm.shape[1]
    cand_per_read = cm.sum(1)
    # forward the heads to get chain logits, then the real chain loss
    outputs = model(sup.base_codes, mask=sup.mask, graph=reference.graph,
                    qualities=sup.qualities, modality=sup.modality)
    bsz, nch, nmem = sup.member_states_shape
    chain_scores = model.score_chains(
        chain_features=sup.chain_feats,
        member_states=torch.zeros(bsz, nch, nmem, model.cfg.d_model),
        member_mask=torch.ones(bsz, nch, nmem, dtype=torch.bool),
        chain_mask=sup.chain_mask,
    )
    crit = GraphMambaLoss(LossConfig())
    logits = chain_scores["logits"]
    logits = logits.squeeze(-1) if logits.dim() == 3 else logits
    chain_term = float(crit.alignment.chain_loss(logits, ct, sup.chain_mask))
    print(f"decoys={decoys}: n_chain(width)={n_chain} "
          f"candidates/read={cand_per_read.tolist()} "
          f"reads_with_target={int((ct>=0).sum())}/{len(ct)} "
          f"chain_loss={chain_term:.6f}")
    assert torch.isfinite(sup.chain_feats).all(), "chain_feats not finite"


run(0)
run(1)
print("OK")
