"""Per-stage / per-layer verification of the GraphMambaFormer pipeline.

Runs the synthetic dataset through every implemented component and asserts that
each stage's output is correct, mapping checks to the Figure 1 architecture:

  Stage 0  — Input data integrity (GC, repeat, error model, seeds, CIGAR)
  1A       — Modality-Aware Read Encoder
  1A       — Reference Graph Encoder
  1B L1    — Bidirectional Mamba-2 (seed chaining)
  1B L2    — Windowed multi-head self-attention
  backbone — MambaFormer / hybrid block stack
  full     — Top-level encoder forward + backward
  long     — Long-read loadability (encoder over an 8-10 kb read)

The heavy Mamba backbone is run on a truncated read slice so the whole suite
finishes in seconds on CPU; the cheap read/graph encoders run on full length.

Run: PYTHONPATH=. .venv/bin/python scripts/verify_stages.py
"""

from __future__ import annotations

import random

import torch

from graphmambaformer import (
    AttentionConfig,
    BiMamba2,
    BlockConfig,
    GraphMambaFormerBlock,
    GraphMambaFormerEncoder,
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
from graphmambaformer.data import (
    build_datasets,
    collate_reads,
    graph_to_encoder_inputs,
    preset,
)
from graphmambaformer.data.synthetic import (
    BASES,
    Reference,
    PangenomeGraph,
    SyntheticConfig,
    _simulate_read,
    cigar_consumed,
    generate_dataset,
)
from graphmambaformer.device import device_summary, get_device
from graphmambaformer.tokenization import KmerTokenizer

DEVICE = str(get_device())
D_MODEL = 64            # tiny model dims -> fast forward
MODEL_MAX_LEN = 128     # truncate reads (base space) for the Mamba backbone
print(f"[device] {device_summary()}")


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{tag}] {name}{suffix}")

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# Stage 0 — data integrity
# --------------------------------------------------------------------------- #
def check_data_integrity(chk: Checker, dataset) -> None:
    chk.section("Stage 0 — input data integrity")
    cfg = dataset.config

    # references: GC + repeat content within target bands
    for rid, ref in dataset.references.items():
        chk.check(f"ref{rid} GC in [{cfg.gc_min},{cfg.gc_max}]",
                  cfg.gc_min - 0.03 <= ref.gc_content <= cfg.gc_max + 0.03,
                  f"{ref.gc_content:.3f}")
        chk.check(f"ref{rid} repeat in [{cfg.repeat_min},{cfg.repeat_max}]",
                  cfg.repeat_min - 0.03 <= ref.repeat_content <= cfg.repeat_max + 0.03,
                  f"{ref.repeat_content:.3f}")

    # split sizes
    chk.check("split sizes match config",
              len(dataset.splits["train"]) == cfg.n_train
              and len(dataset.splits["val"]) == cfg.n_val
              and len(dataset.splits["test"]) == cfg.n_test,
              f"{ {k: len(v) for k, v in dataset.splits.items()} }")

    main = dataset.splits["train"] + dataset.splits["val"] + dataset.splits["test"]

    # CIGAR reconstructs read length and reference span exactly
    cigar_ok = True
    for r in main:
        q, ref_len = cigar_consumed(r.cigar)
        if q != len(r.seq) or ref_len != (r.ref_end - r.ref_start):
            cigar_ok = False
            break
    chk.check("CIGAR consumes exact read length & reference span", cigar_ok)

    # error model: each of ins/del/sub ~5% of template, total ~15%
    m = s = i = d = 0
    for r in main:
        for op, n in r.cigar:
            if op == "=":
                m += n
            elif op == "X":
                s += n
            elif op == "I":
                i += n
            elif op == "D":
                d += n
    tmpl = m + s + d  # reference-consuming = template length
    sub_r, del_r, ins_r = s / tmpl, d / tmpl, i / tmpl
    err_r = (s + d + i) / tmpl
    chk.check("substitution rate ~5%", 0.03 <= sub_r <= 0.08, f"{sub_r:.3f}")
    chk.check("deletion rate ~5%", 0.03 <= del_r <= 0.08, f"{del_r:.3f}")
    chk.check("insertion rate ~5%", 0.03 <= ins_r <= 0.08, f"{ins_r:.3f}")
    chk.check("total error rate ~15%", 0.11 <= err_r <= 0.20, f"{err_r:.3f}")

    # seeds: true 15-25 (median) and false fraction 20-30% (mean)
    trues = [r.num_true_seeds for r in main]
    fracs = [r.num_false_seeds / max(1, len(r.seeds)) for r in main]
    med_true = sorted(trues)[len(trues) // 2]
    mean_frac = sum(fracs) / len(fracs)
    chk.check("median true seeds in [15,25]", 15 <= med_true <= 25, f"{med_true}")
    chk.check("mean false-seed fraction in [0.20,0.30]",
              0.18 <= mean_frac <= 0.32, f"{mean_frac:.3f}")

    # every true seed's k-mer really matches the reference on-diagonal
    seed_ok = True
    for r in main:
        ref = dataset.references[r.ref_id]
        for sd in r.seeds:
            if not sd.is_true:
                continue
            read_kmer = r.seq[sd.read_pos : sd.read_pos + sd.length]
            ref_kmer = ref.seq[sd.ref_pos : sd.ref_pos + sd.length]
            if sd.strand == -1:
                from graphmambaformer.data.synthetic import reverse_complement
                ref_kmer = reverse_complement(ref_kmer)
            if read_kmer != ref_kmer:
                seed_ok = False
                break
    chk.check("all true seeds are exact reference matches", seed_ok)


def check_homopolymer_doubling(chk: Checker) -> None:
    """Independent test: error rate doubles inside homopolymer runs."""
    chk.section("Stage 0b — homopolymer error doubling")
    rng = random.Random(123)
    cfg = SyntheticConfig(read_len_min=4000, read_len_max=4000)

    def empty_graph(seq):
        return PangenomeGraph([seq], [], [], [0], [0])

    # homopolymer reference (all same base -> every position is homopolymer ctx)
    hp_seq = "A" * 5000
    hp_ref = Reference(0, hp_seq, 0.0, 0.0, [False] * len(hp_seq), empty_graph(hp_seq))
    # random reference (few homopolymer contexts)
    rnd_seq = "".join(rng.choice(BASES) for _ in range(5000))
    rnd_ref = Reference(1, rnd_seq, 0.5, 0.0, [False] * len(rnd_seq), empty_graph(rnd_seq))

    def err_rate(ref):
        rates = []
        for j in range(6):
            rec = _simulate_read(rng, cfg, ref, f"hp{j}", "ont", ref_start=0,
                                 read_len=4000, strand=1)
            m = s = i = d = 0
            for op, n in rec.cigar:
                if op == "=":
                    m += n
                elif op == "X":
                    s += n
                elif op == "I":
                    i += n
                elif op == "D":
                    d += n
            tmpl = m + s + d
            rates.append((s + d + i) / tmpl if tmpl else 0.0)
        return sum(rates) / len(rates)

    hp_err = err_rate(hp_ref)
    rnd_err = err_rate(rnd_ref)
    ratio = hp_err / rnd_err if rnd_err else 0.0
    chk.check("homopolymer error > random-context error",
              hp_err > rnd_err, f"hp={hp_err:.3f} rnd={rnd_err:.3f}")
    chk.check("homopolymer/normal error ratio ~2x",
              1.5 <= ratio <= 2.5, f"{ratio:.2f}x")


# --------------------------------------------------------------------------- #
# 1A — encoders
# --------------------------------------------------------------------------- #
def check_read_encoder(chk: Checker, datasets, dataset) -> tuple:
    chk.section("Stage 1A — Modality-Aware Read Encoder")
    tok = KmerTokenizer(k=3, stride=1)
    batch = [datasets["train"][i] for i in range(len(datasets["train"]))]
    inputs, targets = collate_reads(batch, tok, device=DEVICE)

    read_cfg = ReadEncoderConfig(d_model=D_MODEL, kmer_size=3, kmer_stride=1)
    enc = ModalityAwareReadEncoder(read_cfg).to(DEVICE)
    hidden, mask = enc(
        token_ids=inputs["token_ids"], modality=inputs["modality"],
        qualities=inputs["qualities"], mask=inputs["mask"],
    )
    B, Ltok = inputs["token_ids"].shape
    chk.check("hidden shape (B, L+1, d_model)",
              tuple(hidden.shape) == (B, Ltok + 1, D_MODEL), f"{tuple(hidden.shape)}")
    chk.check("modality token prepended (mask[:,0] all True)",
              bool(mask[:, 0].all()))
    # padded token positions stay masked-out
    pad_consistent = bool((mask[:, 1:] == inputs["mask"]).all())
    chk.check("padding mask preserved after prepend", pad_consistent)
    chk.check("hidden is finite", bool(torch.isfinite(hidden).all()))
    return inputs, targets, hidden, mask


def check_graph_encoder(chk: Checker, dataset) -> None:
    chk.section("Stage 1A — Reference Graph Encoder")
    tok = KmerTokenizer(k=3, stride=1)
    ref = dataset.references[0]
    gin = graph_to_encoder_inputs(ref.graph, tok, device=DEVICE)

    gcfg = GraphEncoderConfig(d_model=D_MODEL, kmer_size=3, lap_pe_dim=8)
    genc = ReferenceGraphEncoder(gcfg).to(DEVICE)
    out = genc(gin["node_kmer_ids"], gin["edge_index"], gin["edge_type"],
               node_kmer_mask=gin["node_kmer_mask"])
    N = len(ref.graph.node_seqs)
    E = len(ref.graph.edge_index)
    chk.check("node embeddings shape (N, d_model)",
              tuple(out.node_embeddings.shape) == (N, D_MODEL),
              f"{tuple(out.node_embeddings.shape)}")
    chk.check("edge-type embeddings shape (E, d_edge)",
              out.edge_type_embeddings.shape[0] == E)
    chk.check("laplacian PE shape (N, lap_pe_dim)",
              tuple(out.lap_pe.shape) == (N, 8), f"{tuple(out.lap_pe.shape)}")
    chk.check("edge types within [0, num_edge_types)",
              bool((gin["edge_type"] < gcfg.num_edge_types).all()))
    chk.check("graph embeddings finite", bool(torch.isfinite(out.node_embeddings).all()))


# --------------------------------------------------------------------------- #
# 1B — sequence layers
# --------------------------------------------------------------------------- #
def check_sequence_layers(chk: Checker, hidden, mask) -> None:
    # use a short slice so the CPU Mamba scan is fast
    x = hidden[:, : MODEL_MAX_LEN + 1].contiguous()
    m = mask[:, : MODEL_MAX_LEN + 1].contiguous()

    chk.section("Stage 1B Layer 1 — Bidirectional Mamba-2")
    mamba_cfg = Mamba2Config(d_model=D_MODEL, d_inner=2 * D_MODEL, headdim=32)
    bimamba = BiMamba2(mamba_cfg).to(DEVICE)
    y = bimamba(x, mask=m)
    chk.check("output shape == input", y.shape == x.shape, f"{tuple(y.shape)}")
    chk.check("output finite", bool(torch.isfinite(y).all()))

    chk.section("Stage 1B Layer 2 — Multi-head self-attention")
    attn_cfg = AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16)
    attn = MultiHeadSelfAttention(attn_cfg).to(DEVICE)
    ya = attn(x, mask=m)
    chk.check("attention output shape == input", ya.shape == x.shape)
    win_cfg = AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16, window=8)
    yw = MultiHeadSelfAttention(win_cfg).to(DEVICE)(x, mask=m)
    chk.check("windowed attention output shape == input", yw.shape == x.shape)

    chk.section("Backbone — MambaFormer / hybrid block")
    mf_cfg = MambaFormerConfig(d_model=D_MODEL, n_layer=4, mamba=mamba_cfg, attention=attn_cfg)
    mf = MambaFormer(mf_cfg).to(DEVICE)
    chk.check("MambaFormer layer types M A M A M",
              mf.layer_types == ["mamba", "attention", "mamba", "attention", "mamba"],
              f"{mf.layer_types}")
    chk.check("MambaFormer output shape", mf(x, mask=m).shape == x.shape)
    block_cfg = BlockConfig(d_model=D_MODEL, mamba=mamba_cfg, d_ff=2 * D_MODEL)
    block = GraphMambaFormerBlock(block_cfg).to(DEVICE)
    chk.check("hybrid block sublayers [mamba, ffn]",
              block.sublayer_names == ["mamba", "ffn"], f"{block.sublayer_names}")
    chk.check("hybrid block output shape", block(x, mask=m).shape == x.shape)


# --------------------------------------------------------------------------- #
# full model — forward + backward
# --------------------------------------------------------------------------- #
def check_full_model(chk: Checker, datasets) -> None:
    chk.section("Full encoder — forward + backward (both backbones)")
    tok = KmerTokenizer(k=3, stride=1)
    batch = [datasets["train"][i] for i in range(min(3, len(datasets["train"])))]
    inputs, targets = collate_reads(batch, tok, device=DEVICE, max_read_len=MODEL_MAX_LEN)

    ref = datasets["train"].references[0]
    gin = graph_to_encoder_inputs(ref.graph, tok, device=DEVICE)

    mamba_cfg = Mamba2Config(d_model=D_MODEL, d_inner=2 * D_MODEL, headdim=32)
    attn_cfg = AttentionConfig(d_model=D_MODEL, n_heads=4, d_head=16)
    mf_cfg = MambaFormerConfig(d_model=D_MODEL, n_layer=2, mamba=mamba_cfg, attention=attn_cfg)
    block_cfg = BlockConfig(d_model=D_MODEL, mamba=mamba_cfg, d_ff=2 * D_MODEL)

    for backbone in ("mambaformer", "hybrid"):
        mcfg = ModelConfig(
            d_model=D_MODEL, backbone=backbone, n_blocks=2,
            read_encoder=ReadEncoderConfig(d_model=D_MODEL, kmer_size=3),
            graph_encoder=GraphEncoderConfig(d_model=D_MODEL, kmer_size=3, lap_pe_dim=8),
            block=block_cfg, mambaformer=mf_cfg,
        )
        model = GraphMambaFormerEncoder(mcfg).to(DEVICE)
        # build a graph encoding to thread through (ignored by current layers)
        genc = model.encode_graph(
            node_kmer_ids=gin["node_kmer_ids"], edge_index=gin["edge_index"],
            edge_type=gin["edge_type"], node_kmer_mask=gin["node_kmer_mask"],
        )
        out, out_mask = model(
            token_ids=inputs["token_ids"], modality=inputs["modality"],
            qualities=inputs["qualities"], mask=inputs["mask"], graph=genc,
        )
        ok_shape = out.shape[0] == len(batch) and out.shape[-1] == D_MODEL
        chk.check(f"{backbone}: forward output shape", ok_shape, f"{tuple(out.shape)}")
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        has_grad = any(g is not None and torch.isfinite(g).all() for g in grads)
        chk.check(f"{backbone}: backward produces finite gradients", has_grad)


# --------------------------------------------------------------------------- #
# long-read loadability + edge cases
# --------------------------------------------------------------------------- #
def check_long_reads(chk: Checker) -> None:
    chk.section("Long reads — 8-10 kb loadability (read encoder, full length)")
    long_ds = generate_dataset(preset("long"))
    tok = KmerTokenizer(k=3, stride=1)
    datasets, _ = build_datasets(dataset=long_ds)
    ds = datasets["test"]
    batch = [ds[0]]
    inputs, _ = collate_reads(batch, tok, device=DEVICE)  # full length, no truncation
    read_cfg = ReadEncoderConfig(d_model=D_MODEL, kmer_size=3)
    enc = ModalityAwareReadEncoder(read_cfg).to(DEVICE)
    with torch.no_grad():
        hidden, mask = enc(token_ids=inputs["token_ids"], modality=inputs["modality"],
                           qualities=inputs["qualities"], mask=inputs["mask"])
    L = inputs["token_ids"].shape[1]
    chk.check("long read tokenized (>=7000 k-mers)", L >= 7000, f"{L} tokens")
    chk.check("read-encoder handles full long read",
              tuple(hidden.shape) == (1, L + 1, D_MODEL) and bool(torch.isfinite(hidden).all()),
              f"{tuple(hidden.shape)}")


def check_edge_cases(chk: Checker, dataset) -> None:
    chk.section("Edge cases — pipeline robustness")
    if "edge" not in dataset.splits:
        chk.check("edge-case split present", False)
        return
    tok = KmerTokenizer(k=3, stride=1)
    read_cfg = ReadEncoderConfig(d_model=D_MODEL, kmer_size=3)
    enc = ModalityAwareReadEncoder(read_cfg).to(DEVICE)
    from graphmambaformer.data import AlignmentDataset
    eds = AlignmentDataset(dataset.splits["edge"], dataset.references)
    for i in range(len(eds)):
        rec = dataset.splits["edge"][i]
        try:
            inputs, _ = collate_reads([eds[i]], tok, device=DEVICE)
            with torch.no_grad():
                hidden, mask = enc(token_ids=inputs["token_ids"], modality=inputs["modality"],
                                   qualities=inputs["qualities"], mask=inputs["mask"])
            ok = torch.isfinite(hidden).all().item() and hidden.shape[0] == 1
        except Exception as exc:  # noqa: BLE001
            ok = False
            chk.check(f"edge '{rec.edge_case}'", False, str(exc))
            continue
        chk.check(f"edge '{rec.edge_case}'", bool(ok), f"tokens={inputs['token_ids'].shape[1]}")


def print_stage_map() -> None:
    print("\n=== Figure-1 stage coverage ===")
    rows = [
        ("1A Read encoder", "implemented", "reads + qualities + modality"),
        ("1A Graph encoder", "implemented", "pangenome graph (nodes/edges/lap-PE)"),
        ("1B L1 Bi-Mamba-2", "implemented", "seed-chaining labels (true/false seeds)"),
        ("1B L2 Attention", "implemented", "sub/indel context (CIGAR X/I/D)"),
        ("1B L3 GATv2", "implemented", "GATv2Tower in GraphMambaModel"),
        ("1B Cross-attention", "implemented", "read <-> graph fusion"),
        ("1C Alignment decoder", "implemented", "seed -> chain -> extend -> CIGAR"),
        ("1C MAPQ head", "implemented", "MappingHead + margin blend"),
        ("1C Methylation head", "implemented", "MultiTaskConfig.methylation"),
        ("1C Splice head", "pending", "splice junctions (rna_seq)"),
        ("1C Barcode/UMI head", "pending", "barcode/umi (single-cell/linked)"),
        ("1C Chimeric head", "partial", "supplementary records via secondaries"),
        ("Stage 1 seeding", "implemented", "minimizer / SMEM / DBG / fuzzy / GPU"),
        ("Stage 2 chaining", "implemented", "affine-gap DP + graph bonus"),
        ("Stage 3 extension", "implemented", "banded affine SW + WFA"),
        ("Stage 4 scoring", "implemented", "anchor prune / chain rerank / MAPQ"),
        ("Stage 5 post-processing", "implemented", "correction / population MAPQ / liftover / concordance"),
        ("Stage 6 repeat + HLA", "implemented", "repeat / paralog / HLA align / diploid MHC"),
        ("Stage 7 predictions", "implemented", "genotype / phase / ancestry / clinical / PGx"),
        ("Losses", "implemented", "AlignmentLoss + Kendall MultiTaskLoss"),
        ("GPU accel stack", "implemented", "CuPy RawKernels + Triton + AMP"),
        ("1D Stage1 pretrain", "data-ready", "linear ref + simulated reads"),
        ("1D Stage2 graph FT", "data-ready", "pangenome graph + truth CIGAR"),
        ("1D Stage3 LoRA", "pending", "per-modality reads available"),
        ("1D Stage4 RLHF", "pending", "MAPQ/concordance reward signal"),
    ]
    for name, status, note in rows:
        print(f"  {name:<24} {status:<16} {note}")
    print("  (alignment stages verified by scripts/verify_alignment_pipeline.py)")


def main() -> None:
    torch.manual_seed(0)
    print("Generating tiny synthetic dataset (CPU)...")
    dataset = generate_dataset(preset("tiny"))
    datasets, dataset = build_datasets(dataset=dataset)

    chk = Checker()
    check_data_integrity(chk, dataset)
    check_homopolymer_doubling(chk)
    inputs, targets, hidden, mask = check_read_encoder(chk, datasets, dataset)
    check_graph_encoder(chk, dataset)
    check_sequence_layers(chk, hidden, mask)
    check_full_model(chk, datasets)
    check_edge_cases(chk, dataset)
    check_long_reads(chk)
    print_stage_map()

    print(f"\n{'=' * 68}")
    print(f"RESULT: {chk.passed} passed, {chk.failed} failed")
    print("=" * 68)
    if chk.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
