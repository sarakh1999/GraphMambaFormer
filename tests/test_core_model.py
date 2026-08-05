"""Core-model tests: SequenceEncoder, GraphMambaModel, heads, graph batching.

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import numpy as np
import torch

from graphmambaformer.config import (
    CoreModelConfig, GraphMambaConfig, MultiTaskConfig,
)
from graphmambaformer.encoders.sequence_encoder import SequenceEncoder
from graphmambaformer.models import build_core_model, GraphBatch
from graphmambaformer.tokenization import KmerTokenizer

torch.manual_seed(0)


def _graph_inputs(n_nodes, node_len=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    tok = KmerTokenizer(k=k)
    seqs = ["".join(rng.choice(list("ACGT"), node_len)) for _ in range(n_nodes)]
    enc = tok.batch_encode(seqs)
    src = np.arange(n_nodes - 1)
    edge_index = torch.as_tensor(np.stack([src, src + 1]), dtype=torch.long)
    edge_type = torch.as_tensor(rng.integers(0, 8, n_nodes - 1), dtype=torch.long)
    return {
        "node_kmer_ids": enc["token_ids"],
        "node_kmer_mask": enc["mask"],
        "edge_index": edge_index,
        "edge_type": edge_type,
        "node_lengths": [node_len] * n_nodes,
    }


def test_sequence_encoder_base_space():
    reads = ["ACGTACGTAA", "TTGCA", "ACGTNNACGT"]
    quals = [[30] * len(r) for r in reads]
    batch = SequenceEncoder.encode_batch(reads, quals)
    assert batch["base_codes"].shape == (3, 10)
    assert batch["mask"].sum(1).tolist() == [10, 5, 10]

    cfg = GraphMambaConfig(d_model=64)
    enc = SequenceEncoder(cfg.sequence_encoder)
    hidden, mask = enc(batch["base_codes"], batch["qualities"], batch["mask"])
    # one hidden state per read BASE (no k-mer shortening)
    assert hidden.shape == (3, 10, 64), hidden.shape
    # padded positions must be exactly zero
    assert hidden[1, 5:].abs().max().item() == 0.0
    print(f"SequenceEncoder base-space output OK {tuple(hidden.shape)}")


def test_graphmamba_forward_backward():
    cfg = GraphMambaConfig(d_model=64, n_mamba_layers=2, n_gat_layers=2, d_ff=128)

    spec = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg))
    model = spec.model
    assert spec.supports_alignment_heads and spec.base_space_input
    print(f"built {spec.summary()}")

    reads = ["".join(np.random.choice(list("ACGT"), 60)) for _ in range(3)]
    batch = SequenceEncoder.encode_batch(reads, [[35] * 60 for _ in reads])
    graph = GraphBatch.from_encoder_inputs(_graph_inputs(20), node_lengths=[12] * 20)

    out = model(
        batch["base_codes"], batch["qualities"], batch["mask"],
        graph=graph, modality="pacbio_hifi",
    )
    B, L, D = 3, 60, 64
    assert out.read_hidden.shape == (B, L, D), out.read_hidden.shape
    assert out.graph_nodes.shape == (B, 20, D), out.graph_nodes.shape
    assert out.fused.shape == (B, L + 20, D)
    assert out.pooled.shape == (B, D)
    assert out.mapping["node_logits"].shape == (B, 20)
    assert out.mapping["mapq"].shape == (B,)
    assert (out.mapping["mapq"] >= 0).all() and (out.mapping["mapq"] <= 60).all()
    assert out.router["route"].shape == (B,)
    assert set(model.router.route_names(out.router["route"])) <= {"fast", "medium", "full"}
    for name, tensor in [("pooled", out.pooled), ("mapq", out.mapping["mapq"]),
                         ("node_logits", out.mapping["node_logits"])]:
        assert torch.isfinite(tensor).all(), name

    loss = out.pooled.square().mean() + out.mapping["mapq"].mean()
    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 30, len(grads)
    print(f"forward + backward OK ({len(grads)} params received gradient)")


def test_scoring_heads():
    cfg = GraphMambaConfig(d_model=64, n_mamba_layers=1, n_gat_layers=1, d_ff=128)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model

    reads = ["".join(np.random.choice(list("ACGT"), 50)) for _ in range(2)]
    batch = SequenceEncoder.encode_batch(reads)
    graph = GraphBatch.from_encoder_inputs(_graph_inputs(10), node_lengths=[12] * 10)
    out = model(batch["base_codes"], batch["qualities"], batch["mask"], graph=graph)

    A = 7
    feats = torch.randn(2, A, 12)
    pos = torch.randint(0, 50, (2, A))
    node = torch.randint(-1, 10, (2, A))
    amask = torch.ones(2, A, dtype=torch.bool)
    amask[1, 5:] = False
    seed = model.score_seeds(out, feats, pos, node, amask)
    assert seed["score"].shape == (2, A)
    assert (seed["score"][1, 5:] == 0).all(), "padded anchors must score 0"
    assert ((seed["score"] >= 0) & (seed["score"] <= 1)).all()

    C, M = 3, 4
    chain_feats = torch.randn(2, C, 10)
    members = torch.randn(2, C, M, 64)
    mmask = torch.ones(2, C, M, dtype=torch.bool)
    mmask[:, 2, :] = False           # a chain with no members
    cmask = torch.ones(2, C, dtype=torch.bool)
    cmask[:, 2] = False
    chain = model.score_chains(chain_feats, members, mmask, cmask)
    assert chain["score"].shape == (2, C)
    assert torch.isfinite(chain["score"]).all()
    assert (chain["score"][:, 2] == 0).all()
    print("seed + chain scoring heads OK (masking respected, no NaN)")


def test_multitask_mode():
    tasks = MultiTaskConfig(variant_calling=True, haplotype=True, methylation=True,
                            ancestry=True, copy_number=True, bqsr=True)
    cfg = GraphMambaConfig(d_model=64, n_mamba_layers=1, n_gat_layers=1, d_ff=128,
                           multi_task=tasks)

    spec = build_core_model(CoreModelConfig(arch="multitask_graphmamba", graphmamba=cfg))
    model = spec.model
    reads = ["".join(np.random.choice(list("ACGT"), 40)) for _ in range(2)]
    batch = SequenceEncoder.encode_batch(reads)
    graph = GraphBatch.from_encoder_inputs(_graph_inputs(8), node_lengths=[12] * 8)
    out = model(batch["base_codes"], batch["qualities"], batch["mask"], graph=graph)

    expect_scope = {
        "variant_calling": (2, 8), "copy_number": (2, 8), "ancestry_local": (2, 8),
        "haplotype": (2,), "ancestry": (2,),
        "methylation": (2, 40), "bqsr": (2, 40),
    }
    for name, lead in expect_scope.items():
        assert name in out.multitask, (name, list(out.multitask))
        assert out.multitask[name].shape[:len(lead)] == lead, (name, out.multitask[name].shape)
    print(f"multitask mode OK ({len(out.multitask)} heads: {sorted(out.multitask)})")


def test_per_read_graph_batching():
    """Block-diagonal collate must equal running each graph on its own."""
    cfg = GraphMambaConfig(d_model=64, n_mamba_layers=1, n_gat_layers=2, d_ff=128)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    model.eval()

    graphs = [_graph_inputs(6, seed=1), _graph_inputs(9, seed=2)]
    batched = GraphBatch.collate(graphs)
    assert not batched.shared and batched.num_graphs == 2
    with torch.no_grad():
        nodes, mask, lengths = model.encode_graph(batched, batch_size=2)
    assert nodes.shape == (2, 9, 64), nodes.shape
    assert mask[0].sum() == 6 and mask[1].sum() == 9
    assert nodes[0, 6:].abs().max() == 0.0, "padding must stay zero"

    for i, g in enumerate(graphs):
        solo = GraphBatch.from_encoder_inputs(g, node_lengths=g["node_lengths"])
        with torch.no_grad():
            n_solo, _, _ = model.encode_graph(solo, batch_size=1)
        n = g["node_kmer_ids"].shape[0]
        assert torch.allclose(nodes[i, :n], n_solo[0], atol=1e-5), i
    print("per-read graph collate == per-graph encoding OK")


def test_encoder_baseline_modes():
    for arch in ("mambaformer", "hybrid"):
        cfg = CoreModelConfig(arch=arch)
        cfg.encoder.d_model = 64
        cfg.encoder.n_blocks = 2
        cfg.encoder.mambaformer.n_layer = 2
        cfg.encoder.__post_init__()
        spec = build_core_model(cfg)
        assert not spec.supports_alignment_heads
        assert spec.model.cfg.backbone == arch
        print(f"  baseline arch {arch}: {spec.summary()}")
    print("encoder baseline modes build OK")


def test_reference_param_count():
    """The reference config (d=256, 6 layers) should land near 14.2M params."""
    spec = build_core_model("graphmamba")
    n = spec.num_parameters
    assert 8e6 < n < 30e6, n
    print(f"reference GraphMambaModel: {n:,} params (d_model=256, 6 BiMamba2, 3 GATv2)")


if __name__ == "__main__":
    test_sequence_encoder_base_space()
    test_graphmamba_forward_backward()
    test_scoring_heads()
    test_multitask_mode()
    test_per_read_graph_batching()
    test_encoder_baseline_modes()
    test_reference_param_count()
    print("\nALL CORE MODEL CHECKS PASSED")
