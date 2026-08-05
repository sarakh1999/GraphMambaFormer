"""Loss tests: alignment terms, NaN safety, Kendall weighting, multi-task scopes.

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import torch

from graphmambaformer.config import (
    CoreModelConfig,
    GraphMambaConfig,
    LossConfig,
    MultiTaskConfig,
)
from graphmambaformer.losses import AlignmentLoss, GraphMambaLoss, MultiTaskLoss
from graphmambaformer.losses.alignment_loss import TASK_SPECS
from graphmambaformer.models import build_core_model
from graphmambaformer.models.graph_mamba import GraphBatch


def _batch(cfg, b=4, length=40, n_nodes=6):
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, 4, (b, length), generator=g)
    node_k = torch.randint(0, 4 ** cfg.graph_encoder.kmer_size, (n_nodes, 5), generator=g)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    graph = GraphBatch(
        node_kmer_ids=node_k,
        edge_index=edges,
        edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
    )
    return base, graph


def test_smooth_l1_beta_available():
    out = torch.nn.functional.smooth_l1_loss(
        torch.zeros(3), torch.ones(3), beta=0.1, reduction="none"
    )
    assert out.shape == (3,)
    print("F.smooth_l1_loss(beta=) available OK")


def test_alignment_terms_and_backward():
    cfg = GraphMambaConfig(d_model=64)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    base, graph = _batch(cfg)
    out = model(base, graph=graph)
    b = base.shape[0]
    n_anchor, n_chain = 7, 3

    seed_scores = model.score_seeds(
        out,
        seed_features=torch.randn(b, n_anchor, cfg.seed_scoring.num_seed_features),
        anchor_read_pos=torch.randint(0, base.shape[1], (b, n_anchor)),
        anchor_node=torch.randint(0, 6, (b, n_anchor)),
        anchor_mask=torch.ones(b, n_anchor, dtype=torch.bool),
    )
    chain_scores = model.score_chains(
        chain_features=torch.randn(b, n_chain, cfg.seed_scoring.num_chain_features),
        member_states=torch.randn(b, n_chain, 4, cfg.d_model),
        member_mask=torch.ones(b, n_chain, 4, dtype=torch.bool),
        chain_mask=torch.ones(b, n_chain, dtype=torch.bool),
    )

    targets = {
        "seed_labels": torch.randint(0, 2, (b, n_anchor)).float(),
        "anchor_mask": torch.ones(b, n_anchor, dtype=torch.bool),
        "chain_target": torch.tensor([0, 1, 2, -1]),
        "chain_mask": torch.ones(b, n_chain, dtype=torch.bool),
        "node_target": torch.tensor([0, 2, 4, -100]),
        "position_target": torch.rand(b),
        "mapq_target": torch.tensor([60.0, 30.0, 0.0, 12.0]),
        "best_score": torch.tensor([10.0, 9.0, 8.0, 7.0]),
        "decoy_score": torch.tensor([4.0, 8.9, 2.0, 7.5]),
    }

    loss_fn = GraphMambaLoss(LossConfig(), max_mapq=60)
    result = loss_fn(out, targets, seed_scores=seed_scores, chain_scores=chain_scores)
    expected = {"seed", "chain", "node", "position", "mapq", "router", "extension"}
    assert set(result.terms) == expected, set(result.terms)
    assert torch.isfinite(result.total), result.total
    result.total.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    assert grads > 0
    print(f"all 7 alignment terms present, finite, backward OK ({grads} params w/ grad)")
    print("   " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(result.terms.items())))


def test_partial_labels_skip_terms():
    cfg = GraphMambaConfig(d_model=64)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    base, graph = _batch(cfg)
    out = model(base, graph=graph)
    loss_fn = GraphMambaLoss(LossConfig())
    res = loss_fn(out, {"node_target": torch.tensor([0, 1, 2, 3])})
    assert set(res.terms) == {"node", "router"}, set(res.terms)
    print(f"partial labels -> only {sorted(res.terms)} scored")


def test_chain_loss_all_unmatched_is_safe():
    """A batch where no candidate is correct must not produce NaN."""
    al = AlignmentLoss(LossConfig())
    logits = torch.randn(3, 4, requires_grad=True)
    loss = al.chain_loss(logits, torch.tensor([-1, -1, -1]), torch.ones(3, 4, dtype=torch.bool))
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0, loss
    loss.backward()

    # Fully-masked rows would give an all -inf row: must also stay finite.
    logits2 = torch.randn(2, 4, requires_grad=True)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    loss2 = al.chain_loss(logits2, torch.tensor([0, 1]), mask)
    assert torch.isfinite(loss2), loss2
    print("chain loss safe for unmatched + fully-masked rows")


def test_chain_loss_prefers_correct_ordering():
    al = AlignmentLoss(LossConfig())
    good = torch.tensor([[5.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 0.0, 5.0]])
    t = torch.tensor([0])
    assert al.chain_loss(good, t, None) < al.chain_loss(bad, t, None)
    print("chain loss rewards ranking the true chain first")


def test_padded_chains_get_no_probability_mass():
    al = AlignmentLoss(LossConfig())
    logits = torch.tensor([[1.0, 9.0, 9.0]])  # padded slots have huge scores
    mask = torch.tensor([[True, False, False]])
    loss = al.chain_loss(logits, torch.tensor([0]), mask)
    assert float(loss) < 1e-5, loss  # only live candidate -> ~zero loss
    print("padded chain candidates excluded from the softmax")


def test_kendall_weights_are_learnable():
    cfg = LossConfig(learnable_weights=True)
    loss_fn = GraphMambaLoss(cfg)
    al = AlignmentLoss(cfg)
    losses = {"seed": torch.tensor(1.0, requires_grad=True), "node": torch.tensor(2.0, requires_grad=True)}
    total, applied = loss_fn.weighting.combine(losses)
    total.backward()
    params = list(loss_fn.weighting.log_vars.values())
    assert len(params) == 2 and all(p.grad is not None for p in params)

    static = GraphMambaLoss(LossConfig(learnable_weights=False))
    _, applied_static = static.weighting.combine(
        {"mapq": torch.tensor(1.0), "seed": torch.tensor(1.0)}
    )
    assert applied_static["mapq"] == cfg.w_mapq and applied_static["seed"] == cfg.w_seed
    print(f"Kendall log-vars learnable {applied}; static weights honored {applied_static}")


def test_multitask_losses():
    """Each head is scored at its own scope, with aux regressions picked up."""
    cfg = GraphMambaConfig(d_model=64)
    for name in ("variant_calling", "sv_genotyping", "copy_number", "haplotype",
                 "hla_typing", "bqsr", "methylation", "ancestry", "somatic", "pgx"):
        setattr(cfg.multi_task, name, True)
    cfg.multi_task.d_model = cfg.d_model
    model = build_core_model(CoreModelConfig(arch="multitask_graphmamba", graphmamba=cfg)).model
    base, graph = _batch(cfg)
    out = model(base, graph=graph)
    b, length = base.shape
    n_nodes = out.graph_nodes.shape[1]
    mt = cfg.multi_task

    targets = {
        # node scope
        "variant_calling": torch.randint(0, mt.num_genotypes, (b, n_nodes)),
        "variant_calling_aux": torch.rand(b, n_nodes),
        "sv_genotyping": torch.randint(0, mt.num_sv_types, (b, n_nodes)),
        "sv_genotyping_aux": torch.rand(b, n_nodes, 2),
        "copy_number": torch.randint(0, mt.num_cn_states, (b, n_nodes)),
        "ancestry_local": torch.randint(0, mt.num_populations, (b, n_nodes)),
        # read scope
        "haplotype": torch.randint(0, 2, (b,)),
        "hla_typing": torch.randint(0, mt.num_hla_alleles, (b,)),
        "ancestry": torch.randint(0, mt.num_populations, (b,)),
        "somatic": torch.randint(0, mt.num_somatic_classes, (b,)),
        "pgx": torch.randint(0, mt.num_pgx_alleles, (b,)),
        # base scope
        "bqsr": torch.randint(0, mt.num_quality_bins, (b, length)),
        "methylation": torch.randint(0, 2, (b, length)),
    }
    res = GraphMambaLoss(LossConfig(), mt)(out, targets)
    for name in TASK_SPECS:
        assert f"task/{name}" in res.terms, (name, sorted(res.terms))
    assert "task/variant_calling_aux" in res.terms
    assert "task/sv_genotyping_aux" in res.terms
    assert torch.isfinite(res.total), res.total
    res.total.backward()
    print(f"all {len(TASK_SPECS)} task heads scored at their own scope + 2 aux regressions")


def test_ignored_node_labels_are_skipped():
    """A fully unlabelled node-scope head must stay finite, not NaN."""
    mt = MultiTaskConfig(variant_calling=True)
    loss = MultiTaskLoss(mt, LossConfig())
    pred = {"variant_calling": torch.randn(2, 5, mt.num_genotypes + 1, requires_grad=True)}
    tgt = {"variant_calling": torch.full((2, 5), -100)}
    out = loss(pred, tgt)
    v = out["task/variant_calling"]
    assert torch.isfinite(v) and float(v.detach()) == 0.0, v
    v.backward()
    print("fully-ignored node labels give a finite zero")


def test_unlabelled_head_is_skipped():
    mt = MultiTaskConfig(ancestry=True, somatic=True)
    loss = MultiTaskLoss(mt, LossConfig())
    pred = {
        "ancestry": torch.randn(2, mt.num_populations),
        "somatic": torch.randn(2, mt.num_somatic_classes),
    }
    out = loss(pred, {"ancestry": torch.tensor([0, 1])})
    assert set(out) == {"task/ancestry"}, set(out)
    print("unlabelled head contributes no term")


def test_router_budget_is_one_sided():
    cfg = LossConfig(router_target_cost=0.6)
    al = AlignmentLoss(cfg)
    assert float(al.router_loss(torch.tensor([0.2, 0.3]))) == 0.0
    assert float(al.router_loss(torch.tensor([0.9, 0.9]))) > 0.0
    print("router budget penalizes overspend only")


if __name__ == "__main__":
    test_smooth_l1_beta_available()
    test_alignment_terms_and_backward()
    test_partial_labels_skip_terms()
    test_chain_loss_all_unmatched_is_safe()
    test_chain_loss_prefers_correct_ordering()
    test_padded_chains_get_no_probability_mass()
    test_kendall_weights_are_learnable()
    test_multitask_losses()
    test_ignored_node_labels_are_skipped()
    test_unlabelled_head_is_skipped()
    test_router_budget_is_one_sided()
    print("\nALL LOSS CHECKS PASSED")
