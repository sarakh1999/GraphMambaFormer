"""Shape smoke test for the currently implemented GraphMambaFormer modules.

Run: ``.venv/bin/python scripts/smoke_test.py``

Exercises (with random / synthetic long-read data):
  1. Modality-Aware Read Encoder
  2. Reference Graph Encoder
  3. Bidirectional Mamba-2 mixer
  4. A hybrid block and the top-level encoder stack
"""

from __future__ import annotations

import random

import torch

from graphmambaformer import (
    AttentionConfig,
    BiMamba1,
    BiMamba2,
    BlockConfig,
    GraphMambaFormerBlock,
    GraphMambaFormerEncoder,
    Mamba1Config,
    Mamba1Mixer,
    Mamba2Config,
    MambaFormer,
    MambaFormerConfig,
    ModalityAwareReadEncoder,
    ModelConfig,
    MultiHeadSelfAttention,
    ReadEncoderConfig,
    ReferenceGraphEncoder,
)
from graphmambaformer.config import GraphEncoderConfig
from graphmambaformer.device import device_summary, get_device

torch.manual_seed(0)
random.seed(0)

BASES = "ACGT"


def random_read(n: int) -> str:
    return "".join(random.choice(BASES) for _ in range(n))


def main() -> None:
    device = get_device()
    print(f"[device] {device_summary()}")
    d_model = 128  # smaller than the 512 default to keep the smoke test snappy

    # ------------------------------------------------------------------ #
    # 1. Read encoder (long-read modalities: PacBio HiFi + ONT).
    # ------------------------------------------------------------------ #
    read_cfg = ReadEncoderConfig(d_model=d_model, kmer_size=3, kmer_stride=1)
    read_encoder = ModalityAwareReadEncoder(read_cfg).to(device)

    seqs = [random_read(200), random_read(150), random_read(180)]
    quals = [[random.randint(20, 40) for _ in s] for s in seqs]
    modalities = ["pacbio_hifi", "ont", "pacbio_hifi"]

    hidden, mask = read_encoder.encode_reads(seqs, modality=modalities, quals=quals, device=device)
    print(f"[read encoder]   hidden={tuple(hidden.shape)}  mask={tuple(mask.shape)}")
    assert hidden.shape[0] == 3 and hidden.shape[2] == d_model

    # ------------------------------------------------------------------ #
    # 2. Graph encoder (tiny synthetic pangenome subgraph).
    # ------------------------------------------------------------------ #
    graph_cfg = GraphEncoderConfig(d_model=d_model, kmer_size=3, lap_pe_dim=8)
    graph_encoder = ReferenceGraphEncoder(graph_cfg).to(device)

    node_seqs = [random_read(random.randint(20, 60)) for _ in range(6)]
    node_ids, node_mask = graph_encoder.encode_node_sequences(node_seqs, device=device)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4], [1, 2, 3, 3, 4, 5]], dtype=torch.long, device=device
    )
    edge_type = torch.tensor([0, 1, 0, 2, 0, 3], dtype=torch.long, device=device)

    genc = graph_encoder(node_ids, edge_index, edge_type, node_kmer_mask=node_mask)
    print(
        f"[graph encoder]  nodes={tuple(genc.node_embeddings.shape)}  "
        f"edges={tuple(genc.edge_type_embeddings.shape)}  lap_pe={tuple(genc.lap_pe.shape)}"
    )
    assert genc.node_embeddings.shape == (6, d_model)

    # ------------------------------------------------------------------ #
    # 3. Bidirectional Mamba-2 mixer.
    # ------------------------------------------------------------------ #
    mamba_cfg = Mamba2Config(d_model=d_model, d_inner=2 * d_model, headdim=32)
    bimamba = BiMamba2(mamba_cfg).to(device)
    x = torch.randn(2, 64, d_model, device=device)
    y = bimamba(x)
    print(f"[bi-mamba2]      in={tuple(x.shape)}  out={tuple(y.shape)}")
    assert y.shape == x.shape

    # ------------------------------------------------------------------ #
    # 3b. Reference Mamba-1 mixer (pure-PyTorch) + bidirectional wrapper.
    # ------------------------------------------------------------------ #
    mamba1_cfg = Mamba1Config(d_model=d_model, d_state=16, expand=2)
    m1 = Mamba1Mixer(mamba1_cfg).to(device)
    y1 = m1(x, mask=torch.ones(2, 64, dtype=torch.bool, device=device))
    print(f"[mamba1]         in={tuple(x.shape)}  out={tuple(y1.shape)}")
    assert y1.shape == x.shape

    bimamba1 = BiMamba1(mamba1_cfg).to(device)
    yb1 = bimamba1(x)
    print(f"[bi-mamba1]      in={tuple(x.shape)}  out={tuple(yb1.shape)}")
    assert yb1.shape == x.shape
    yb1.sum().backward()  # gradient sanity check through the reference scan

    # ------------------------------------------------------------------ #
    # 4. Multi-head self-attention (bidirectional, padding-mask aware).
    # ------------------------------------------------------------------ #
    attn_cfg = AttentionConfig(d_model=d_model, n_heads=4, d_head=32)
    attn = MultiHeadSelfAttention(attn_cfg).to(device)
    ya = attn(hidden, mask=mask)
    print(f"[mhsa]           in={tuple(hidden.shape)}  out={tuple(ya.shape)}")
    assert ya.shape == hidden.shape

    # ------------------------------------------------------------------ #
    # 5. MambaFormer backbone (reference MixerModel layout): leading Mamba
    #    then n_layer interleaved blocks (even=attn, odd=mamba).
    # ------------------------------------------------------------------ #
    # Default mamba_variant="mamba1" -> reference-port Mamba-1 mixer.
    mf_cfg = MambaFormerConfig(
        d_model=d_model, n_layer=4, mamba1=mamba1_cfg, attention=attn_cfg
    )
    mformer = MambaFormer(mf_cfg).to(device)
    ymf = mformer(hidden, mask=mask)
    print(f"[mambaformer:m1] layers={mformer.layer_types}  out={tuple(ymf.shape)}")
    assert ymf.shape == hidden.shape
    # leading Mamba then interleaved (i%2==0 -> attn)  ->  M A M A M
    assert mformer.layer_types == ["mamba", "attention", "mamba", "attention", "mamba"]

    # mamba_variant="mamba2" -> Mamba-2 / SSD mixer.
    mf_cfg_m2 = MambaFormerConfig(
        d_model=d_model, n_layer=4, mamba_variant="mamba2", mamba=mamba_cfg, attention=attn_cfg
    )
    mformer_m2 = MambaFormer(mf_cfg_m2).to(device)
    ymf_m2 = mformer_m2(hidden, mask=mask)
    print(f"[mambaformer:m2] layers={mformer_m2.layer_types}  out={tuple(ymf_m2.shape)}")
    assert ymf_m2.shape == hidden.shape

    # alternate parity: attention_first=False -> even indices become Mamba  ->  M M A M A
    mf_cfg_mf = MambaFormerConfig(
        d_model=d_model, n_layer=4, attention_first=False, mamba1=mamba1_cfg, attention=attn_cfg
    )
    mformer_mf = MambaFormer(mf_cfg_mf).to(device)
    ymf2 = mformer_mf(hidden, mask=mask)
    print(f"[mambaformer:mf] layers={mformer_mf.layer_types}  out={tuple(ymf2.shape)}")
    assert mformer_mf.layer_types == ["mamba", "mamba", "attention", "mamba", "attention"]

    # ------------------------------------------------------------------ #
    # 6. Hybrid block (extensible Figure 1B block).
    # ------------------------------------------------------------------ #
    block_cfg = BlockConfig(d_model=d_model, mamba=mamba_cfg, d_ff=4 * d_model)
    block = GraphMambaFormerBlock(block_cfg).to(device)
    yb = block(hidden, mask=mask)
    print(f"[hybrid block]   sublayers={block.sublayer_names}  out={tuple(yb.shape)}")

    # ------------------------------------------------------------------ #
    # 7. Top-level encoder with each backbone.
    # ------------------------------------------------------------------ #
    batch = read_encoder.tokenizer.batch_encode(seqs, quals=quals, device=device)

    for backbone in ("mambaformer", "hybrid"):
        model_cfg = ModelConfig(
            d_model=d_model,
            backbone=backbone,
            n_blocks=2,
            read_encoder=read_cfg,
            graph_encoder=graph_cfg,
            block=block_cfg,
            mambaformer=mf_cfg,
        )
        model = GraphMambaFormerEncoder(model_cfg).to(device)
        out, out_mask = model(
            token_ids=batch["token_ids"],
            modality=modalities,
            qualities=batch["qualities"],
            mask=batch["mask"],
            graph=genc,
        )
        print(
            f"[encoder:{backbone:<11}] out={tuple(out.shape)}  params={model.num_parameters():,}"
        )
        assert out.shape[-1] == d_model
        out.sum().backward()  # gradient sanity check
    print("[backward]       gradient flowed OK for both backbones")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
