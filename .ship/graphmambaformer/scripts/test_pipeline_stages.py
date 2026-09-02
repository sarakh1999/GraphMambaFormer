"""End-to-end CPU test of the GraphMambaFormer pipeline on the small samples.

Runs the reads + graphs produced by ``scripts/make_small_samples.py`` through
every implemented phase, printing the input/output of each and writing a
concrete output artifact per stage into ``data/stage_outputs/``:

    00_input_summary.json          what went in (reads + both graphs)
    01_read_encoder.pt             1A  read encoder      -> (B, L+1, d)
    02_graph_encoder_synthetic.pt  1A  graph encoder (synthetic, 8 edge types)
    02_graph_encoder_hprc.pt       1A  graph encoder (real CHM13 chr1 window)
    03_bimamba.pt                  1B L1 bidirectional Mamba (mamba1 + mamba2)
    04_attention.pt                1B L2 multi-head self-attention (+ windowed)
    05_residual_check.txt          residual wiring around every Mamba/attn layer
    06_backbone.pt                 MambaFormer backbone (M A M A M)
    07_full_model_<backbone>.pt    top-level encoder forward + backward

Also writes ``data/stage_outputs/PIPELINE_IO.md`` (the input/output table) and
``data/stage_outputs/STAGE_REPORT.txt`` (pass/fail summary).

Run: PYTHONPATH=. .venv/bin/python scripts/test_pipeline_stages.py
"""

from __future__ import annotations

import json
import os

import torch

from graphmambaformer import (
    AttentionConfig,
    BiMamba1,
    BiMamba2,
    GATv2Layer,
    GraphMambaFormerBlock,
    GraphMambaFormerEncoder,
    Mamba1Config,
    Mamba2Config,
    MambaFormer,
    MambaFormerConfig,
    ModalityAwareReadEncoder,
    ModelConfig,
    MultiHeadSelfAttention,
    ReadEncoderConfig,
    ReferenceGraphEncoder,
)
from graphmambaformer.config import BlockConfig, GATConfig, GraphEncoderConfig
from graphmambaformer.data import (
    AlignmentDataset,
    collate_reads,
    graph_to_encoder_inputs,
    load_dataset,
    read_gfa,
)
from graphmambaformer.device import device_summary, get_device
from graphmambaformer.layers.common import Residual
from graphmambaformer.tokenization import KmerTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO, "data", "small_samples")
OUT = os.path.join(REPO, "data", "stage_outputs")

DEVICE = str(get_device())
D_MODEL = 64          # tiny dims -> fast forward
MODEL_MAX_LEN = 128   # truncate reads (base space) for the Mamba/attention scans
KMER = 3
LAP_PE = 8

torch.manual_seed(0)
print(f"[device] {device_summary()}")

# rows for the PIPELINE_IO.md table: (stage, input, output, artifact)
IO_ROWS: list[tuple[str, str, str, str]] = []
PASSES: list[tuple[str, bool, str]] = []


def record(stage: str, inp: str, out: str, artifact: str) -> None:
    IO_ROWS.append((stage, inp, out, artifact))
    print(f"\n[{stage}]\n    in : {inp}\n    out: {out}\n    -> {artifact}")


def check(name: str, ok: bool, detail: str = "") -> None:
    PASSES.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"    [{tag}] {name}" + (f"  ({detail})" if detail else ""))


def _stats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": round(tf.mean().item(), 6),
        "std": round(tf.std().item(), 6),
        "finite": bool(torch.isfinite(tf).all()),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tok = KmerTokenizer(k=KMER, stride=1)

    # ---------------------------------------------------------------- #
    # Load small samples
    # ---------------------------------------------------------------- #
    ds = load_dataset(os.path.join(SAMPLES, "synthetic_small.pt"))
    datasets = {s: AlignmentDataset(recs, ds.references) for s, recs in ds.splits.items()}
    train = datasets["train"]
    batch = [train[i] for i in range(len(train))]
    inputs, targets = collate_reads(batch, tok, device=DEVICE, max_read_len=MODEL_MAX_LEN)

    synth_graph = read_gfa(os.path.join(SAMPLES, "synthetic_graph.gfa"))
    hprc_graph = read_gfa(os.path.join(SAMPLES, "hprc_chm13_chr1_window.gfa"))

    # ---------------------------------------------------------------- #
    # Stage 0 — input summary
    # ---------------------------------------------------------------- #
    print("=" * 72)
    print("Stage 0 — inputs")
    print("=" * 72)
    summary = {
        "reads": {
            "n_reads": len(batch),
            "modalities": [b["modality"] for b in batch],
            "token_ids_shape": list(inputs["token_ids"].shape),
            "max_read_len_basespace": MODEL_MAX_LEN,
            "example_read_ids": [b["record"].read_id for b in batch],
        },
        "synthetic_graph": {
            "nodes": len(synth_graph.node_seqs),
            "edges": len(synth_graph.edge_index),
            "edge_types_present": sorted(set(synth_graph.edge_type)),
        },
        "hprc_chm13_chr1_window": {
            "nodes": len(hprc_graph.node_seqs),
            "edges": len(hprc_graph.edge_index),
            "edge_types_present": sorted(set(hprc_graph.edge_type)),
            "note": "reference-backbone window (ref_link edges only)",
        },
    }
    with open(os.path.join(OUT, "00_input_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    record(
        "Stage 0 · inputs",
        f"{len(batch)} reads (modalities {summary['reads']['modalities']}); "
        f"synthetic graph {len(synth_graph.node_seqs)}N/{len(synth_graph.edge_index)}E; "
        f"HPRC chr1 window {len(hprc_graph.node_seqs)}N/{len(hprc_graph.edge_index)}E",
        "parsed reads + 2 graphs",
        "00_input_summary.json",
    )
    check("synthetic graph exposes multiple edge types",
          len(set(synth_graph.edge_type)) >= 3, str(sorted(set(synth_graph.edge_type))))
    check("HPRC real window parsed with nodes+edges",
          len(hprc_graph.node_seqs) > 1 and len(hprc_graph.edge_index) > 0)

    # ---------------------------------------------------------------- #
    # Stage 1A — Modality-Aware Read Encoder
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nStage 1A — Read encoder\n" + "=" * 72)
    read_enc = ModalityAwareReadEncoder(
        ReadEncoderConfig(d_model=D_MODEL, kmer_size=KMER, kmer_stride=1)
    ).to(DEVICE)
    hidden, mask = read_enc(
        token_ids=inputs["token_ids"], modality=inputs["modality"],
        qualities=inputs["qualities"], mask=inputs["mask"],
    )
    torch.save({"hidden": hidden.detach(), "mask": mask, "stats": _stats(hidden)},
               os.path.join(OUT, "01_read_encoder.pt"))
    B, Ltok = inputs["token_ids"].shape
    record("Stage 1A · read encoder",
           f"token_ids {list(inputs['token_ids'].shape)} + qualities + modality",
           f"hidden {list(hidden.shape)} (=(B, L+1, d)), mask {list(mask.shape)}",
           "01_read_encoder.pt")
    check("read hidden shape (B, L+1, d)", tuple(hidden.shape) == (B, Ltok + 1, D_MODEL),
          str(tuple(hidden.shape)))
    check("modality token prepended (mask[:,0] True)", bool(mask[:, 0].all()))
    check("read hidden finite", bool(torch.isfinite(hidden).all()))

    # ---------------------------------------------------------------- #
    # Stage 1A — Reference Graph Encoder (on BOTH graphs)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nStage 1A — Graph encoder\n" + "=" * 72)
    gcfg = GraphEncoderConfig(d_model=D_MODEL, kmer_size=KMER, lap_pe_dim=LAP_PE)
    graph_enc = ReferenceGraphEncoder(gcfg).to(DEVICE)
    graph_encodings: dict = {}
    for tag, graph in (("synthetic", synth_graph), ("hprc", hprc_graph)):
        gin = graph_to_encoder_inputs(graph, tok, device=DEVICE)
        genc = graph_enc(gin["node_kmer_ids"], gin["edge_index"], gin["edge_type"],
                         node_kmer_mask=gin["node_kmer_mask"])
        graph_encodings[tag] = genc
        torch.save(
            {"node_embeddings": genc.node_embeddings.detach(),
             "edge_index": genc.edge_index, "lap_pe": genc.lap_pe.detach(),
             "edge_type_embeddings": genc.edge_type_embeddings.detach(),
             "stats": _stats(genc.node_embeddings)},
            os.path.join(OUT, f"02_graph_encoder_{tag}.pt"),
        )
        N = len(graph.node_seqs)
        record(f"Stage 1A · graph encoder ({tag})",
               f"{N} nodes, {len(graph.edge_index)} edges, edge_type ids "
               f"{sorted(set(graph.edge_type))}",
               f"node_emb {list(genc.node_embeddings.shape)}, "
               f"lap_pe {list(genc.lap_pe.shape)}, "
               f"edge_emb {list(genc.edge_type_embeddings.shape)}",
               f"02_graph_encoder_{tag}.pt")
        check(f"{tag}: node embeddings (N, d)",
              tuple(genc.node_embeddings.shape) == (N, D_MODEL))
        check(f"{tag}: lap PE (N, {LAP_PE})", tuple(genc.lap_pe.shape) == (N, LAP_PE))
        check(f"{tag}: node embeddings finite",
              bool(torch.isfinite(genc.node_embeddings).all()))

    # short slice for the sequence layers (keep the CPU scan fast)
    x = hidden[:, : MODEL_MAX_LEN + 1].contiguous()
    m = mask[:, : MODEL_MAX_LEN + 1].contiguous()

    # ---------------------------------------------------------------- #
    # Stage 1B Layer 1 — Bidirectional Mamba (mamba1 + mamba2)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nStage 1B L1 — Bidirectional Mamba\n" + "=" * 72)
    bimamba1 = BiMamba1(Mamba1Config(d_model=D_MODEL, d_state=16, expand=2)).to(DEVICE)
    y1 = bimamba1(x, mask=m)
    bimamba2 = BiMamba2(Mamba2Config(d_model=D_MODEL, d_inner=2 * D_MODEL, headdim=32)).to(DEVICE)
    y2 = bimamba2(x, mask=m)
    torch.save({"mamba1_out": y1.detach(), "mamba2_out": y2.detach(),
                "in_stats": _stats(x), "mamba1_stats": _stats(y1), "mamba2_stats": _stats(y2)},
               os.path.join(OUT, "03_bimamba.pt"))
    record("Stage 1B L1 · Bi-Mamba",
           f"hidden slice {list(x.shape)} + mask",
           f"mamba1 {list(y1.shape)}, mamba2 {list(y2.shape)} (both == input)",
           "03_bimamba.pt")
    check("Bi-Mamba1 output shape == input", y1.shape == x.shape)
    check("Bi-Mamba2 output shape == input", y2.shape == x.shape)
    check("Bi-Mamba outputs finite", bool(torch.isfinite(y1).all() and torch.isfinite(y2).all()))

    # ---------------------------------------------------------------- #
    # Stage 1B Layer 2 — Multi-head self-attention (+ windowed)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nStage 1B L2 — Self-attention\n" + "=" * 72)
    attn = MultiHeadSelfAttention(AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16)).to(DEVICE)
    ya = attn(x, mask=m)
    winattn = MultiHeadSelfAttention(
        AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16, window=8)
    ).to(DEVICE)
    yw = winattn(x, mask=m)
    torch.save({"attn_out": ya.detach(), "windowed_attn_out": yw.detach(),
                "attn_stats": _stats(ya), "windowed_stats": _stats(yw)},
               os.path.join(OUT, "04_attention.pt"))
    record("Stage 1B L2 · attention",
           f"hidden slice {list(x.shape)} + mask",
           f"full-attn {list(ya.shape)}, windowed(w=8) {list(yw.shape)}",
           "04_attention.pt")
    check("attention output shape == input", ya.shape == x.shape)
    check("windowed attention output shape == input", yw.shape == x.shape)

    # ---------------------------------------------------------------- #
    # Stage 1B Layer 3 — GATv2 graph attention (edge-type-aware)
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nStage 1B L3 — GATv2 graph attention\n" + "=" * 72)
    sg = graph_encodings["synthetic"]
    gat = GATv2Layer(GATConfig(d_model=D_MODEL, n_heads=4, d_gat=32)).to(DEVICE)
    node_in = sg.node_embeddings.detach()
    gat_out = gat(node_in, sg.edge_index, sg.edge_type_embeddings)
    n_edges = sg.edge_index.shape[1]
    torch.save({"node_in": node_in, "gat_out": gat_out.detach(),
                "edge_index": sg.edge_index,
                "in_stats": _stats(node_in), "out_stats": _stats(gat_out)},
               os.path.join(OUT, "04b_gatv2.pt"))
    record("Stage 1B L3 · GATv2",
           f"nodes {list(node_in.shape)}, {n_edges} edges + self-loops, "
           f"edge_emb {list(sg.edge_type_embeddings.shape)}",
           f"refined nodes {list(gat_out.shape)} (same (N, d))",
           "04b_gatv2.pt")
    check("GATv2 output shape == (N, d)", gat_out.shape == node_in.shape)
    check("GATv2 output finite", bool(torch.isfinite(gat_out).all()))
    check("GATv2 refines node embeddings (output != input)",
          not torch.allclose(gat_out, node_in))

    # ---------------------------------------------------------------- #
    # Full hybrid block: Mamba -> windowed attention -> GATv2 -> FFN
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nFull hybrid block (mamba->attn->gat->ffn)\n" + "=" * 72)
    hb_cfg = BlockConfig(
        d_model=D_MODEL, use_attention=True, use_gat=True, d_ff=2 * D_MODEL, window=8,
        mamba=Mamba2Config(d_model=D_MODEL, d_inner=2 * D_MODEL, headdim=32),
        attention=AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16),
        gat=GATConfig(d_model=D_MODEL, n_heads=4, d_gat=32),
    )
    hb = GraphMambaFormerBlock(hb_cfg).to(DEVICE)
    ne_before = sg.node_embeddings.clone()
    hb_out = hb(x, mask=m, graph=sg)
    torch.save({"block_out": hb_out.detach(), "layer_names": hb.layer_names,
                "graph_refined": bool(not torch.allclose(sg.node_embeddings, ne_before)),
                "stats": _stats(hb_out)},
               os.path.join(OUT, "04c_hybrid_block.pt"))
    record("Full hybrid block",
           f"reads {list(x.shape)} + graph ({ne_before.shape[0]} nodes)",
           f"layers {hb.layer_names}; read out {list(hb_out.shape)}; graph refined",
           "04c_hybrid_block.pt")
    check("hybrid block layer order = mamba->attention->gat->ffn",
          hb.layer_names == ["mamba", "attention", "gat", "ffn"], str(hb.layer_names))
    check("hybrid block read output shape == input", hb_out.shape == x.shape)
    check("hybrid block refined the graph nodes",
          not torch.allclose(sg.node_embeddings, ne_before))

    # ---------------------------------------------------------------- #
    # Residual wiring check — around every Mamba AND attention sub-layer
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nResidual check — Mamba <-> attention\n" + "=" * 72)
    mf_cfg = MambaFormerConfig(
        d_model=D_MODEL, n_layer=4, mamba_variant="mamba1",
        mamba1=Mamba1Config(d_model=D_MODEL),
        attention=AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16),
    )
    mf = MambaFormer(mf_cfg).to(DEVICE).eval()
    lines: list[str] = []
    lines.append(f"layer_types = {mf.layer_types}")

    all_residual = all(isinstance(l, Residual) for l in mf.layers)
    lines.append(f"every layer is a pre-norm Residual wrapper: {all_residual}")
    check("every Mamba/attention layer wrapped in Residual", all_residual)

    expected = ["mamba", "attention", "mamba", "attention", "mamba"]
    check("layer order = leading Mamba + interleaved (M A M A M)",
          mf.layer_types == expected, str(mf.layer_types))

    # Numerically verify y == x + sublayer(norm(x)) for each layer, and that the
    # residual actually changes the output (skip-connection present, not a no-op).
    with torch.no_grad():
        xr = x
        residual_exact = True
        residual_nontrivial = True
        for i, layer in enumerate(mf.layers):
            manual = xr + layer.sublayer(layer.norm(xr), mask=m)
            y = layer(xr, mask=m)
            sub_only = layer.sublayer(layer.norm(xr), mask=m)
            exact = torch.allclose(y, manual, atol=1e-6)
            nontrivial = not torch.allclose(y, sub_only, atol=1e-6)  # residual added something
            residual_exact = residual_exact and exact
            residual_nontrivial = residual_nontrivial and nontrivial
            lines.append(
                f"  layer {i} [{mf.layer_types[i]:<9}] y==x+sublayer(norm(x)): {exact}; "
                f"differs from sublayer-only (residual present): {nontrivial}"
            )
            xr = y
    check("residual identity y == x + sublayer(norm(x)) for all layers", residual_exact)
    check("residual is non-trivial (skip connection changes output)", residual_nontrivial)

    with open(os.path.join(OUT, "05_residual_check.txt"), "w") as fh:
        fh.write("Residual wiring between Mamba and attention sub-layers\n")
        fh.write("=" * 60 + "\n")
        fh.write("Each backbone sub-layer is common.Residual: forward(x) = "
                 "x + sublayer(RMSNorm(x)).\n\n")
        fh.write("\n".join(lines) + "\n")
    record("Residual check",
           f"MambaFormer(n_layer=4) over hidden slice {list(x.shape)}",
           f"{'ALL PASS' if residual_exact and residual_nontrivial and all_residual else 'SEE FILE'}: "
           "residual present + exact around every Mamba/attention layer",
           "05_residual_check.txt")

    # ---------------------------------------------------------------- #
    # Backbone — MambaFormer
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nBackbone — MambaFormer\n" + "=" * 72)
    yb = mf(x, mask=m)
    torch.save({"backbone_out": yb.detach(), "layer_types": mf.layer_types,
                "stats": _stats(yb)}, os.path.join(OUT, "06_backbone.pt"))
    record("Backbone · MambaFormer",
           f"hidden slice {list(x.shape)} + mask",
           f"{list(yb.shape)} (== input); layers {mf.layer_types}",
           "06_backbone.pt")
    check("backbone output shape == input", yb.shape == x.shape)

    # ---------------------------------------------------------------- #
    # Full model — forward + backward (both backbones), graph threaded through
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72 + "\nFull encoder — forward + backward\n" + "=" * 72)
    gin = graph_to_encoder_inputs(synth_graph, tok, device=DEVICE)
    # (name, backbone, use_attention, use_gat): the last is the full Figure-1B block.
    variants = (
        ("mambaformer", "mambaformer", False, False),
        ("hybrid", "hybrid", False, False),
        ("hybrid+gat", "hybrid", True, True),
    )
    for name, backbone, use_attn, use_gat in variants:
        mcfg = ModelConfig(
            d_model=D_MODEL, backbone=backbone, n_blocks=2,
            read_encoder=ReadEncoderConfig(d_model=D_MODEL, kmer_size=KMER),
            graph_encoder=GraphEncoderConfig(d_model=D_MODEL, kmer_size=KMER, lap_pe_dim=LAP_PE),
            block=BlockConfig(d_model=D_MODEL, use_attention=use_attn, use_gat=use_gat,
                              window=8, mamba=Mamba2Config(d_model=D_MODEL,
                              d_inner=2 * D_MODEL, headdim=32), d_ff=2 * D_MODEL,
                              attention=AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16),
                              gat=GATConfig(d_model=D_MODEL, n_heads=4, d_gat=32)),
            mambaformer=MambaFormerConfig(d_model=D_MODEL, n_layer=2,
                              mamba1=Mamba1Config(d_model=D_MODEL),
                              attention=AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16)),
        )
        model = GraphMambaFormerEncoder(mcfg).to(DEVICE)
        genc = model.encode_graph(
            node_kmer_ids=gin["node_kmer_ids"], edge_index=gin["edge_index"],
            edge_type=gin["edge_type"], node_kmer_mask=gin["node_kmer_mask"],
        )
        ne_before = genc.node_embeddings.clone()
        out, out_mask = model(
            token_ids=inputs["token_ids"], modality=inputs["modality"],
            qualities=inputs["qualities"], mask=inputs["mask"], graph=genc,
        )
        refined = not torch.allclose(genc.node_embeddings, ne_before)
        loss = out.sum()
        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        finite_grad = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.save({"out": out.detach(), "out_mask": out_mask, "stats": _stats(out),
                    "graph_refined": refined, "n_params": model.num_parameters()},
                   os.path.join(OUT, f"07_full_model_{name}.pt"))
        record(f"Full model · {name}",
               f"reads (token_ids {list(inputs['token_ids'].shape)}) + graph "
               f"({len(synth_graph.node_seqs)} nodes) threaded through",
               f"hidden {list(out.shape)}, params {model.num_parameters():,}, "
               f"backward -> {n_grad} grad tensors"
               + (f"; graph refined by GATv2" if use_gat else ""),
               f"07_full_model_{name}.pt")
        check(f"{name}: forward output (B, *, d)",
              out.shape[0] == len(batch) and out.shape[-1] == D_MODEL, str(tuple(out.shape)))
        check(f"{name}: backward produced finite grads", finite_grad)
        if use_gat:
            check(f"{name}: GATv2 refined graph nodes across blocks", refined)

    # ---------------------------------------------------------------- #
    # Write PIPELINE_IO.md + STAGE_REPORT.txt
    # ---------------------------------------------------------------- #
    with open(os.path.join(OUT, "PIPELINE_IO.md"), "w") as fh:
        fh.write("# GraphMambaFormer pipeline — input / output per phase\n\n")
        fh.write(f"Run on CPU with `d_model={D_MODEL}`, k-mer `k={KMER}`, "
                 f"reads truncated to `{MODEL_MAX_LEN}` bp, Laplacian PE dim `{LAP_PE}`.\n\n")
        fh.write("| Phase | Input | Output | Artifact |\n")
        fh.write("| --- | --- | --- | --- |\n")
        for stage, inp, out_, art in IO_ROWS:
            fh.write(f"| {stage} | {inp} | {out_} | `{art}` |\n")

    passed = sum(1 for _, ok, _ in PASSES if ok)
    failed = len(PASSES) - passed
    with open(os.path.join(OUT, "STAGE_REPORT.txt"), "w") as fh:
        fh.write("GraphMambaFormer staged-pipeline test report\n")
        fh.write("=" * 60 + "\n")
        for name, ok, detail in PASSES:
            fh.write(f"[{'PASS' if ok else 'FAIL'}] {name}"
                     + (f"  ({detail})" if detail else "") + "\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"RESULT: {passed} passed, {failed} failed\n")

    print("\n" + "=" * 72)
    print(f"RESULT: {passed} passed, {failed} failed")
    print(f"artifacts + PIPELINE_IO.md + STAGE_REPORT.txt written to {OUT}")
    print("=" * 72)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
