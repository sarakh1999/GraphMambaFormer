#!/usr/bin/env python3
"""Build GraphMamba technical briefing as PowerPoint (.pptx)."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BG = RGBColor(0x0B, 0x10, 0x20)
PANEL = RGBColor(0x12, 0x1A, 0x2E)
TEXT = RGBColor(0xE8, 0xEE, 0xF7)
DIM = RGBColor(0x94, 0xA3, 0xB8)
ACCENT = RGBColor(0x38, 0xBD, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xFB, 0xBF, 0x24)
GREEN = RGBColor(0x34, 0xD3, 0x99)


def set_slide_bg(slide, color: RGBColor = BG) -> None:
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    spTree = slide.shapes._spTree
    sp = bg._element
    spTree.remove(sp)
    spTree.insert(2, sp)


def add_textbox(slide, left, top, width, height, text, *, size=18, bold=False, color=TEXT, align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def add_para(tf, text, *, size=16, bold=False, color=TEXT, space_before=4, space_after=2, font="Calibri", align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0]
    if p.text or p.runs:
        p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return p


def clear_first(tf):
    tf.word_wrap = True
    tf.paragraphs[0].clear()
    return tf


def header(slide, section: str, title: str) -> None:
    set_slide_bg(slide)
    add_textbox(slide, Inches(0.55), Inches(0.28), Inches(12), Inches(0.3), section.upper(), size=11, bold=True, color=ACCENT)
    add_textbox(slide, Inches(0.55), Inches(0.52), Inches(12.2), Inches(0.55), title, size=26, bold=True, color=WHITE)


def body_box(slide, top=Inches(1.15)):
    box = slide.shapes.add_textbox(Inches(0.55), top, Inches(12.2), Inches(5.9))
    return clear_first(box.text_frame)


def bullets(tf, items, *, size=15, color=TEXT):
    for item in items:
        add_para(tf, "•  " + item, size=size, color=color, space_before=5, space_after=2)


def numbered(tf, items, *, size=15):
    for i, item in enumerate(items, 1):
        add_para(tf, f"{i}.  {item}", size=size, space_before=5, space_after=2)


def code_block(tf, text, *, size=11):
    for line in text.splitlines() or [""]:
        add_para(tf, line if line.strip() else " ", size=size, color=DIM, font="Consolas", space_before=0, space_after=0)


def _fill_cell(cell, color: RGBColor):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for child in list(tcPr):
        if "solidFill" in child.tag:
            tcPr.remove(child)
    tcPr.append(
        parse_xml(
            f'<a:solidFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:srgbClr val="{color}"/>'
            f"</a:solidFill>"
        )
    )


def add_table(slide, headers, rows, *, left=Inches(0.5), top=Inches(1.2), width=Inches(12.3), col_widths=None, font_size=11):
    n_cols = len(headers)
    n_rows = len(rows) + 1
    row_h = min(0.42, 5.6 / max(n_rows, 1))
    height = Inches(row_h * n_rows)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size)
            p.font.bold = True
            p.font.color.rgb = ACCENT
            p.font.name = "Calibri"
        _fill_cell(cell, PANEL)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = TEXT
                p.font.name = "Calibri"
            _fill_cell(cell, BG if i % 2 == 0 else PANEL)
    return table


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def title_slide(prs, title, subtitle, chips=None):
    slide = new_slide(prs)
    set_slide_bg(slide)
    add_textbox(slide, Inches(0.8), Inches(1.8), Inches(11.7), Inches(0.4), "GRAPHMAMBAFORMER TECHNICAL BRIEFING", size=12, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.0), title, size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(1.5), Inches(3.5), Inches(10.3), Inches(1.2), subtitle, size=18, color=DIM, align=PP_ALIGN.CENTER)
    if chips:
        add_textbox(slide, Inches(1.0), Inches(5.1), Inches(11.3), Inches(0.6), "  ·  ".join(chips), size=13, color=DIM, align=PP_ALIGN.CENTER)


def content_slide(prs, section, title, items=None, code=None, note=None, numbered_items=None):
    slide = new_slide(prs)
    header(slide, section, title)
    tf = body_box(slide)
    if numbered_items:
        numbered(tf, numbered_items)
    if items:
        bullets(tf, items)
    if code:
        add_para(tf, " ", size=6, space_before=6)
        code_block(tf, code, size=11)
    if note:
        add_para(tf, note, size=12, color=AMBER, space_before=12)


def two_col_slide(prs, section, title, left_title, left_items, right_title, right_items, note=None):
    slide = new_slide(prs)
    header(slide, section, title)
    left = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(5.9), Inches(5.3))
    tf = clear_first(left.text_frame)
    add_para(tf, left_title, size=16, bold=True, color=ACCENT, space_before=0)
    bullets(tf, left_items, size=13)
    right = slide.shapes.add_textbox(Inches(6.7), Inches(1.2), Inches(5.9), Inches(5.3))
    tf2 = clear_first(right.text_frame)
    add_para(tf2, right_title, size=16, bold=True, color=GREEN, space_before=0)
    bullets(tf2, right_items, size=13)
    if note:
        add_textbox(slide, Inches(0.55), Inches(6.7), Inches(12.2), Inches(0.4), note, size=11, color=AMBER)


def code_slide(prs, section, title, code, note=None):
    slide = new_slide(prs)
    header(slide, section, title)
    tf = body_box(slide)
    code_block(tf, code, size=12)
    if note:
        add_para(tf, note, size=12, color=AMBER, space_before=14)


def table_content(prs, section, title, headers, rows, col_widths=None, note=None, font_size=11):
    slide = new_slide(prs)
    header(slide, section, title)
    add_table(slide, headers, rows, col_widths=col_widths, font_size=font_size)
    if note:
        add_textbox(slide, Inches(0.55), Inches(6.75), Inches(12.2), Inches(0.35), note, size=11, color=AMBER)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # 1 Title
    title_slide(
        prs,
        "GraphMamba",
        "Pangenome-Aware Neural Sequence Aligner\n& Predictive Genomics Engine",
        chips=["~14.9M params", "BiMamba-2 + GATv2", "seed → chain → extend → score", "HPRC / GIAB ready"],
    )

    # 2 Agenda
    content_slide(
        prs,
        "Overview",
        "Agenda",
        numbered_items=[
            "Landscape — linear & graph alignment algorithms",
            "Architecture — models, towers, heads, pseudocode",
            "Feature pipeline — seeding, chaining, extension, scoring",
            "Adaptive strategies — multiplex mining, traversal, two-pass, router",
            "Platform detect & hardware requirements",
            "Post-processing, coverage, data types",
            "Multitask / somatic / subclonal",
            "Training, validation, loss",
            "Citations & references",
        ],
    )

    # 3 Why
    content_slide(
        prs,
        "Landscape",
        "Why pangenome alignment?",
        items=[
            "Linear references (GRCh38) bias against non-reference haplotypes",
            "Variation graphs encode SNPs, indels, SVs as alternate paths",
            "Long reads (HiFi / ONT) span complex regions but need graph-aware mapping",
            "GraphMamba: classical graph alignment woven with a neural core (not bolted on)",
        ],
        note="Default focus: PacBio HiFi & ONT long reads · HPRC Minigraph-Cactus graphs",
    )

    # 4 Linear aligners
    table_content(
        prs,
        "Landscape",
        "Linear reference aligners",
        ["Tool", "Strategy", "Strength", "Citation"],
        [
            ["BWA-MEM", "FM-index SMEMs + chain + SW", "Short-read gold standard", "Li 2013"],
            ["minimap2", "Minimizers + affine chain + SW/WFA", "Long-read speed/accuracy", "Li 2018"],
            ["Bowtie2", "FM-index + multiseed", "Sensitive short reads", "Langmead 2012"],
            ["HISAT2", "Hierarchical graph FM", "RNA-seq splice", "Kim 2019"],
            ["Winnowmap2", "Weighted minimizers", "Highly repetitive long reads", "Jain 2022"],
            ["lra / LRA", "Sparse DP / anchor", "SV-aware long reads", "Ren 2021"],
            ["BLASR / pbmm2", "PacBio-tuned", "CCS / HiFi", "PacBio"],
        ],
        col_widths=[Inches(2.2), Inches(4.3), Inches(3.2), Inches(2.4)],
    )

    # 5 Graph aligners
    table_content(
        prs,
        "Landscape",
        "Graph-based alignment algorithms",
        ["Tool", "Graph model", "Core method", "Notes"],
        [
            ["vg map", "vg / XG / GCSA2", "MEM seeds + cluster + SW", "Garrison et al. 2018"],
            ["vg giraffe", "GBWT haplotypes", "Minimizer → hap cluster → extend", "Sirén et al. 2021; HPRC default"],
            ["GraphAligner", "GFA / VG", "Seed-and-extend; bitvector DP", "Rautiainen 2020; long reads"],
            ["minigraph", "rGFA", "minimap2-like on graph", "Li 2020; assembly graphs"],
            ["GraphMap2", "linear+graph", "Anchors + graph paths", "ONT-era"],
            ["HISAT2", "hierarchical FM", "Graph FM-index", "Splice + variants"],
            ["GraphMamba", "GFA + typed edges", "Multi-mode seed + graph-bonus chain + neural", "This project"],
        ],
        col_widths=[Inches(2.0), Inches(2.6), Inches(4.2), Inches(3.3)],
        font_size=10,
    )

    # 6 Giraffe vs ours
    two_col_slide(
        prs,
        "Landscape",
        "Giraffe vs GraphMamba (chr21 benchmark)",
        "vg giraffe (baseline)",
        [
            "HPRC MC GBZ + haplotype GBWT",
            "Minimizer index on haplotypes",
            "Distance index + zip codes",
            "DeepVariant / Sniffles downstream",
        ],
        "GraphMamba (ours)",
        [
            "GFA nodes + 8 edge types",
            "SMEM + minimizer (+ DBG/fuzzy/GPU)",
            "Graph-hop BFS bonus in chaining",
            "Optional neural re-rank / MAPQ / rescue",
            "Same BAM contract for DV / hap.py",
        ],
        note="scripts/chr21/: map_giraffe.sh · map_ours.sh · compare.sh · modes: fast | hybrid | two_pass",
    )

    # 7 What we add
    content_slide(
        prs,
        "Landscape",
        "What GraphMamba adds vs classical graph aligners",
        items=[
            "Base-space neural read tower — BiMamba-2 so Stage-4 heads index bases directly",
            "GATv2 graph tower — topology + edge-type message passing",
            "Cross-attention fusion — read↔graph bidirectional grounding",
            "Neural seed prune + chain re-rank woven into classical DP",
            "Complexity router — fast/medium/full compute labels (training hook)",
            "Multi-task heads — variant / SV / CN / HLA / somatic / PGx on same forward",
            "AccelContext — same config on H100, ROCm, MPS, CPU",
        ],
    )

    # 8 Algorithm family
    table_content(
        prs,
        "Landscape",
        "Algorithm family comparison",
        ["Family", "Seeding", "Chain", "Extend", "Graph?"],
        [
            ["BWA-MEM", "SMEM", "seed chain", "SW", "No"],
            ["minimap2", "minimizer", "affine DP", "SW / WFA", "No"],
            ["Giraffe", "minimizer+hap", "cluster", "gapless/gapped", "Yes (GBWT)"],
            ["GraphAligner", "seeds on nodes", "path", "bitvector DP", "Yes"],
            ["GraphMamba", "multi-mode", "affine+graph BFS", "SW / WFA", "Yes + neural"],
        ],
        col_widths=[Inches(2.2), Inches(2.5), Inches(2.5), Inches(2.5), Inches(2.4)],
    )

    # 9 Arch modes
    table_content(
        prs,
        "Architecture",
        "Core architecture modes",
        ["arch", "Model", "Alignment heads", "Role"],
        [
            ['"graphmamba"', "GraphMambaModel", "Yes", "Default · ~14.9M @ d=256"],
            ['"multitask_graphmamba"', "+ 10 task heads", "Yes", "~15.3M when all enabled"],
            ['"mambaformer"', "MambaFormer backbone", "No", "Ablation baseline"],
            ['"hybrid"', "Hybrid block stack", "No", "Ablation · Mamba→Attn→GAT→FFN"],
        ],
        col_widths=[Inches(3.2), Inches(3.2), Inches(2.4), Inches(3.3)],
        note='model = build_core_model().model  ·  pipeline = build_pipeline(..., model=model)',
    )

    # 10 Dual tower
    code_slide(
        prs,
        "Architecture",
        "GraphMambaModel — dual tower",
        """reads  → SequenceEncoder → BiMamba2 × 6 ─┐
                                         ├→ CrossAttentionFusion → pooled
graph  → GraphEncoder    → GATv2 × 3 ────┘
                                              ├→ ComplexityRouter
                                              ├→ MappingHead (node / offset / MAPQ)
                                              ├→ SeedScoringHead
                                              └→ ChainScoringHead

• Base space (not k-mer tokens): read_hidden[b,i] = base i
• Defaults: d_model=256, 6 Mamba layers, 3 GAT layers, d_ff=1024
• Block-diagonal graph batching via GraphBatch.collate()""",
    )

    # 11 Components
    table_content(
        prs,
        "Architecture",
        "What each model component has",
        ["Component", "Contents"],
        [
            ["SequenceEncoder", "Base(6→64) + Kmer(k=3→64) + Qual(42→32) + PE(96) → Linear→LN"],
            ["BiMamba2", "fwd+rev Mamba2; gate g·y_fwd+(1−g)·y_rev; d_state=64, expand=2"],
            ["GraphEncoder", "pooled k-mers + Laplacian PE (16 evecs) + 8 edge-type embeds"],
            ["GATv2", "4 heads × d_gat=128; edge-conditioned; Brody 2021"],
            ["CrossAttn", "8 heads × 32; read↔graph; FFN 4×; attention pool"],
            ["MappingHead", "node logits (dot vs nodes), offset, MAPQ∈[0,60]"],
            ["Seed / Chain heads", "12-D / 10-D geometry + hidden states"],
            ["Router", "MLP → {fast:0.35, med:0.65, full:1.0} costs"],
        ],
        col_widths=[Inches(2.8), Inches(9.3)],
        font_size=12,
    )

    # 12 Forward pseudocode
    code_slide(
        prs,
        "Architecture",
        "Pseudocode — GraphMamba forward",
        """def GraphMambaModel.forward(base_codes, qualities, mask, graph, modality):
    h, mask = SequenceEncoder(base_codes, qualities, mask, modality)  # (B,L,D)
    h = BiMambaTower(h, mask)                                         # ×6

    enc = GraphEncoder(graph.node_kmer_ids, graph.edge_index, graph.edge_type)
    nodes = GATv2Tower(enc.node_embeddings, ...)                       # ×3
    graph_nodes, graph_mask, node_lens = densify(nodes, graph, B)

    fused = CrossAttentionFusion(h, graph_nodes, read_mask, graph_mask)
    router = ComplexityRouter(fused.pooled)           # optional
    mapping = MappingHead(fused.pooled, graph_nodes, graph_mask, node_lens)
    return GraphMambaOutput(read_hidden, graph_nodes, fused, pooled, router, mapping)""",
    )

    # 13 Equations
    two_col_slide(
        prs,
        "Architecture",
        "BiMamba-2 & GATv2 — equations",
        "Bidirectional Mamba",
        [
            "y_fwd = Mamba2(x)",
            "y_rev = flip(Mamba2(flip(x)))",
            "g = σ(W [y_fwd ‖ y_rev])",
            "y = g ⊙ y_fwd + (1−g) ⊙ y_rev",
            "Mamba-2 SSD: selective Δ,B,C · d_conv=4",
            "CUDA mamba_ssm when available",
        ],
        "GATv2 (Brody 2021)",
        [
            "e_ij = aᵀ LeakyReLU(W_src h_j + W_dst h_i + W_e e_ij)",
            "α_ij = softmax_j(e_ij)",
            "h'_i = Σ_j α_ij W_src h_j",
            "8 edge types: ref_link, snp, insertion,",
            "deletion, sv, splice, cpg, barcode",
        ],
    )

    # 14 Cross attn
    code_slide(
        prs,
        "Architecture",
        "Cross-attention fusion — pseudocode",
        """# "Where could this base map?" / "Which reads support this node?"
read_ctx  = CrossAttn(Q=read,  K=V=graph, mask=graph_mask)
graph_ctx = CrossAttn(Q=graph, K=V=read,  mask=read_mask)

read_out  = read  + FFN([read  ‖ read_ctx])    # D→4D→D
graph_out = graph + FFN([graph ‖ graph_ctx])

fused  = concat(read_out, graph_out)           # (B, L+N, D)
pooled = AttentionPool(fused)                  # (B, D)""",
        note="Flash / mem-efficient SDP on CUDA · Triton fused LN+Linear+GELU for FFN",
    )

    # 15 Ablations
    two_col_slide(
        prs,
        "Architecture",
        "Ablation backbones",
        "MambaFormer (Park 2024)",
        [
            "arXiv:2402.04248 · krafton-ai/mambaformer-icl",
            "Inputs → Mamba → for i in layers:",
            "  Attention if i%2==0 else Mamba",
            "n_layer=12 → leading M + 6 Attn + 6 Mamba",
            "Ours: bidirectional Mamba + Attn",
        ],
        "Hybrid block ×N",
        [
            "Figure 1B stack",
            "x = BiMamba(x)        # O(n) state",
            "x = WindowAttn(x, w)  # O(n·w)",
            "nodes = GATv2(nodes)  # O(|E|)",
            "x = SwiGLU_FFN(x)",
            "Default n_blocks=12 · heads off",
        ],
    )

    # 16 Edge types
    two_col_slide(
        prs,
        "Architecture",
        "Edge types & Laplacian PE",
        "8 discrete edge types",
        ["ref_link", "snp", "insertion", "deletion", "sv", "splice", "cpg", "barcode"],
        "Laplacian PE",
        [
            "L = I − D^{−½} A D^{−½}",
            "smallest non-trivial evecs (16)",
            "Linear(16 → d_model) added to nodes",
            "Per connected component on batches",
            "MPS: eigh on CPU then copy back",
        ],
    )

    # 17 Mapping head
    code_slide(
        prs,
        "Architecture",
        "MappingHead — node / offset / MAPQ",
        """# Node: query from pooled · score = q · node_embᵀ  → softmax over N
# Position: MLP(pooled) → σ · length(chosen_node)   # within-node offset
# MAPQ: MLP(D→D/2→1) → σ · 60

mapping = MappingHead(pooled, graph_nodes, graph_mask, node_lengths)
# Fallback without graph: Linear(D, max_nodes=4096)""",
        note='Rescue path uses (node_id, offset) for unmapped reads at mapq_floor with stages=("seed","score","rescue")',
    )

    # 18 Router
    code_slide(
        prs,
        "Architecture",
        "ComplexityRouter — adaptive compute",
        """route_logits = MLP(pooled)           # D → 64 → 3
# train: Gumbel-softmax (τ=1.0) straight-through
# infer: argmax
costs = {fast: 0.35, medium: 0.65, full: 1.00}
L_router = (mean(cost) − 0.6)²       # one-sided budget, weight 0.05

• Today: written to AlignmentRecord.route
• Not yet: skip Stage 1–3 work by route (architecture claims 30–50% FLOP save)""",
    )

    # 19 Pipeline stages
    code_slide(
        prs,
        "Pipeline",
        "Feature engineering pipeline — 5+ stages",
        """Stage 1  Seeding     → anchors (minimizer / SMEM / DBG / fuzzy / GPU)
Stage 4a Neural      → score + prune anchors  (before DP)
Stage 2  Chaining    → collinear chains (affine DP + graph bonus)
Stage 4b Neural      → re-rank chains
Stage 3  Extension   → banded SW or WFA → CIGAR
Stage 4c MAPQ/rescue → mapping quality + position rescue
Stage 5  Post        → primary/secondary, soft clips, records
── roadmap ──
Stage 6  Repeat/HLA  → paralog / allele resolution
Stage 7  Predictions → sample-level VCF aggregation""",
    )

    # 20 Pipeline modes
    table_content(
        prs,
        "Pipeline",
        "Pipeline modes",
        ["Mode", "Class", "Behaviour"],
        [
            ["hybrid", "HybridAlignmentPipeline", "Full path + 1 neural forward (default)"],
            ["fast", "FastAlignmentPipeline", "Classical only; MAPQ from score margin"],
            ["two_pass", "TwoPassAligner", "Fast all → hybrid only on hard reads"],
        ],
        col_widths=[Inches(2.0), Inches(4.0), Inches(6.1)],
        note="Easy read: coverage(best) ≥ 0.80 AND (score1−score2)/score1 ≥ 0.25",
    )

    # 21 Hybrid pseudocode
    code_slide(
        prs,
        "Pipeline",
        "Pseudocode — hybrid pipeline",
        """INPUT: reads[], ReferenceIndex
anchors[] = SEED(reads, bundle)
out = MODEL(encode(reads), graph=index.graph)     # one batch forward
FOR each read r:
  SCORE_ANCHORS(out, anchors[r]); anchors[r] = PRUNE(...)
  chains[r] = CHAIN(anchors[r], oracle, backbone)
  SCORE_CHAINS(out, chains); chains[r] = RERANK(...)
  ext[r] = EXTEND(read[r], chains[r], anchors[r], ref_seq)
  mapq[r] = MAPQ(out, chains[r])
  records[r] = ASSEMBLE(...) OR RESCUE(...) OR UNMAPPED""",
    )

    # 22 Seeding catalogue
    table_content(
        prs,
        "Seeding",
        "Stage 1 — Seeding catalogue",
        ["Mode", "Defaults", "Idea"],
        [
            ["minimizer", "k=15, w=10, max_occ=200", "minimap2 hash64; dens.~2/(w+1)"],
            ["smem", "min_len=13, SA sample=8", "FM-index super-maximal exact matches"],
            ["fmindex", "k=15, stride=5", "Exact k-mer via BWT"],
            ["dbg", "k=21", "k-mers on all pangenome nodes"],
            ["fuzzy", "pattern 111010010100110111", "Spaced seed · weight 11 / span 18"],
            ["multiplex_dbg", "k∈{15,21,31}", "Long-k first, short-k fills gaps"],
            ["gpu_kmer", "same as minimizer", "CuPy binary-search lookup"],
        ],
        col_widths=[Inches(2.2), Inches(4.0), Inches(5.9)],
        note='Default modes: ("smem","minimizer") · both strands · merge slack=4 · max_anchors=5000',
        font_size=11,
    )

    # 23 Seed features
    table_content(
        prs,
        "Seeding",
        "Seed geometric features (12-D)",
        ["#", "Feature family", "Role"],
        [
            ["0–2", "read_pos, ref_pos, length", "Locus & seed strength"],
            ["3–4", "strand, diagonal (ref−read)", "Collinearity"],
            ["5–6", "occurrence / max_occ ratio", "Repeat risk"],
            ["7–8", "source mode one-hot / id", "SMEM vs minimizer vs DBG"],
            ["9–11", "node_id norm, backbone flag, gap", "Graph context"],
        ],
        col_widths=[Inches(1.2), Inches(5.0), Inches(5.9)],
        note="Exact packing in AnchorSet.to_seed_features() · fused with read_hidden[pos] + node emb",
    )

    # 24 Seeding merge
    code_slide(
        prs,
        "Seeding",
        "Pseudocode — seeding & merge",
        """BUILD SeedIndexBundle(ref, node_seqs, node_ref_start, backbone_path)
FOR each read:
  anchors = ∅
  FOR strand in {+1, −1}:
    codes = fwd or revcomp(read)
    FOR mode in cfg.modes:
      hits = indices[mode].query(codes)
      anchors += AnchorSet.from_hits(hits, strand, source=mode)
  anchors = MERGE_DIAGONALS(anchors, slack=4)
  anchors = CAP(anchors, max=5000, by=length)
  anchors.node_id = bundle.node_of(anchors.ref_pos)  # searchsorted""",
    )

    # 25 SMEM
    code_slide(
        prs,
        "Seeding",
        "Pseudocode — SMEM (batched FM)",
        """# For each end position j (batched rank queries):
FOR each j in parallel:
  Extend backward while read[j..cursor] matches (FM backward_extend)
  Track left_min(j), SA interval [lo, hi]
# Super-maximality:
Drop j if left_min(j+1) == left_min(j)
Filter: length ≥ min_seed_len AND occ ≤ max_occ
Locate all reference hits via sampled SA""",
        note="max_occ echoes BWA-MEM’s repeat filter · FM footprint ~2n + 8n/sa_sample",
    )

    # 26 Chaining
    code_slide(
        prs,
        "Chaining",
        "Stage 2 — Chaining (minimap2-style + graph)",
        """advance(j→i) = min(Δq, Δr, w_i)
gap = |Δr − Δq|
penalty = gap_open + gap_extend·gap + log_coeff·log₂(gap+1)   # if gap>0
f[i] = max( w_i ,  max_j  f[j] + advance − penalty + bonus[j→i] )
# Constraints: Δq>0, Δr>0, both ≤ max_gap; lookback ≤ 64

Defaults: gap_open=6, gap_extend=0.05, log_coeff=0.5
          graph_bonus=4.0, graph_max_hops=3, ref_path_bias=1.5
          min_chain_score=20, max_chains=8
          secondary_overlap=0.5, secondary_ratio=0.6""",
    )

    # 27 Node traversal
    two_col_slide(
        prs,
        "Chaining",
        "Node traversal & graph-distance bonus",
        "Node mapping",
        [
            "Build: sort node_ref_start",
            "slot = searchsorted(node_starts, ref_pos, 'right') − 1",
            "node_id = node_ids[slot]",
            "",
            "GraphDistanceOracle",
            "Undirected CSR; on-demand BFS",
            "depth ≤ graph_max_hops (3)",
            "bonus = graph_bonus · (1 − hops/max_hops)",
        ],
        "Backbone bias",
        [
            "w_i = length_i",
            "  + ref_path_bias · 𝟙[on backbone]",
            "    · (0.5 + neural_score_i)",
            "",
            "Prefer reference-path nodes",
            "Neural seed scores scale weights",
            "Strands chained independently then compete",
        ],
    )

    # 28 Chain select
    code_slide(
        prs,
        "Chaining",
        "Pseudocode — chaining + primary/secondary",
        """FOR strand in {+1, −1}:
  A = sort_by_ref_end(anchors on strand)
  w = anchor_weights(A, backbone, neural_scores)
  B = graph_bonus_matrix(A, oracle, lookback=64)
  f, parent = CHAIN_DP(A.read_end, A.ref_end, w, B, cfg)
  chains += BACKTRACK(f, parent, A)   # peel best; claim anchors
RETURN SELECT(chains):
  sort by score; drop if overlap > 0.5 of shorter
  stop if score < 0.6 × best; keep ≤ max_chains""",
    )

    # 29 Adaptive strategies
    table_content(
        prs,
        "Adaptive",
        "Adaptive mining / branch / traversal strategies",
        ["Strategy", "Where", "Behaviour"],
        [
            ["Multiplex k fallback", "MultiplexDBG", "k=31→21→15; short k on uncovered bases"],
            ["max_occ filter", "All seed indexes", "Drop repetitive k-mers / wide SA intervals"],
            ["Length-based cap", "SeedingEngine", "Keep longest ≤5000 anchors"],
            ["Neural prune", "Stage 4a", "Drop score<0.5 but keep ≥8 best"],
            ["BFS hop branch", "Chaining oracle", "Limited graph traversal for bonus"],
            ["Band widen", "Extension", "half_band += diagonal spread (≤512)"],
            ["Two-pass split", "TwoPassAligner", "Easy→fast; hard→hybrid neural"],
            ["ComplexityRouter", "Model", "fast/med/full labels; FLOP gate = hook"],
        ],
        col_widths=[Inches(2.8), Inches(2.6), Inches(6.7)],
        font_size=11,
    )

    # 30 Multiplex
    code_slide(
        prs,
        "Adaptive",
        "Pseudocode — multiplex adaptive mining",
        """covered = empty bitset over read positions
FOR k in sorted(multiplex_kmers, reverse=True):  # 31, 21, 15
  hits = dbg_index[k].query(read, only_where=¬covered)
  FOR hit in hits:
    mark covered[hit.read_start : hit.read_end]
    emit Anchor(hit, source=f"mux_k{k}")
# Long exact matches preferred; short k rescues divergent / gapped regions""",
    )

    # 31 Two-pass
    code_slide(
        prs,
        "Adaptive",
        "Two-pass adaptive branch — pseudocode",
        """pass1 = FastAlignmentPipeline.align(all_reads)
hard = [r for r,a in zip(reads, pass1)
        if not (a.best.coverage ≥ 0.80 and margin(a) ≥ 0.25)]
IF model has alignment heads AND hard:
  pass2 = HybridAlignmentPipeline.align(hard)
  merge: replace hard indices; pass_name = "two_pass"
ELSE:
  keep pass1   # degrades to classical""",
        note="Architecture HTML also describes a C fast path (7–20K r/s); current Python two_pass uses FastAlignmentPipeline",
    )

    # 32 Extension
    table_content(
        prs,
        "Extension",
        "Stage 3 — Extension (banded SW & WFA)",
        ["", "Banded affine SW", "WFA"],
        [
            ["Default", "Yes", "Optional"],
            ["Scores", "match=2, mism=4, open=6, ext=2", "Unit edit"],
            ["Band", "half=64 → +diag_spread ≤512", "—"],
            ["X-drop", "600", "max_distance=4096"],
            ["Window", "flank=100, max=32768", "Same window"],
            ["Fallback", "—", "→ banded SW on failure"],
        ],
        col_widths=[Inches(2.0), Inches(5.2), Inches(4.9)],
        note="Vectorized horizontal gaps via exclusive max-plus prefix scan (cummax)",
    )

    # 33 Ext pseudocode
    code_slide(
        prs,
        "Extension",
        "Pseudocode — extension window & SW row",
        """diagonal = ref_start − read_start
window = [max(0, diag−flank), min(ref_len, diag+read_len+flank)]
half_band = min(half_band + chain_diag_spread, max_half_band)

# Per query row i, band cells d:
M[d] = max(0, diag, vertical)
E = exclusive_max_plus_scan(M + d·extend) − open − (d−1)·extend
H[d] = max(M[d], E[d], 0)
# X-drop: stop if all batch members < best − x_drop
# Traceback → CIGAR; soft-clip unaligned flanks; lift coords to global""",
    )

    # 34 Band widen
    content_slide(
        prs,
        "Extension",
        "Why band widens with diagonal spread",
        items=[
            "A chain with a large indel has anchors off the main diagonal",
            "half_band += chain_diagonal_spread (capped at 512)",
            "Keeps the indel inside the SW band so extension doesn’t truncate",
            "Window centered on chain diagonal ± flank (100), capped at 32,768 bp",
        ],
        code="spread = max(anchor.ref_end - anchor.read_end) - min(...)\nhalf_band = min(cfg.half_band + spread, cfg.max_half_band)",
    )

    # 35 Neural scoring
    content_slide(
        prs,
        "Scoring",
        "Stage 4 — Neural scoring bridge",
        items=[
            "One forward per batch reused for anchors, chains, MAPQ",
            "SeedScoringHead — 12-D geometry + read hidden @ locus + node emb → BCE",
            "ChainScoringHead — 10-D features + attention over member anchors",
            "Blend: (0.5·norm_dp + 0.5·neural) / 1.0",
            "MAPQ: combine mapping-head confidence with primary/secondary margin",
            "Rescue: MappingHead (node, offset) for unmapped → mapq_floor, no CIGAR",
        ],
    )

    # 36 Chain features MAPQ
    code_slide(
        prs,
        "Scoring",
        "Chain features (10-D) & MAPQ",
        """0  read coverage = read_span / read_len
1  min(n_anchors/32, 1)
2  DP score / best DP score
3  merged exact bases / read_len
4–9  ref/read span, strand, diag spread, gaps, mean seed score, backbone frac

margin = clip((best − runner_up) / best, 0, 1)
combined = 0.5·min(mapq_head/60, margin) + 0.5·margin
MAPQ = round(combined · 60) ∈ [mapq_floor, 60]
# Fast mode: MAPQ = margin · 60 only""",
    )

    # 37 Post
    content_slide(
        prs,
        "Post",
        "Stage 5 — Post-processing",
        items=[
            "Primary — best re-ranked chain; only primary gets read MAPQ",
            "Supplementary / secondary — other chains; MAPQ=0; overlap filter",
            "Soft clips — S for unaligned read flanks in CIGAR",
            "Node ID — bundle.node_of(ref_start)",
            "Metadata — stages, route, pass_name, anchor/chain counts",
            "Unmapped — AlignmentRecord.unmapped() if no chain & rescue fails",
        ],
        note="Not yet in code: Stage 6 repeat/HLA resolver, Stage 7 sample VCF aggregation",
    )

    # 38 Coverage
    table_content(
        prs,
        "Coverage",
        "Coverage — definitions & uses",
        ["Quantity", "Definition", "Used for"],
        [
            ["Chain coverage", "(read_end−read_start)/read_len", "Two-pass easy test; chain feat[0]"],
            ["Anchor bases", "Merged exact-match bases in chain", "chain feat[3] (≠ span coverage)"],
            ["easy_coverage", "≥ 0.80 default", "Skip neural pass"],
            ["Primary overlap", "overlap / min(span) > 0.5", "Drop redundant secondaries"],
            ["SAM soft clips", "Full read length reported", "Downstream callers"],
            ["HPRC depth", "Sample coverage in manifests", "Training curriculum / selection"],
        ],
        col_widths=[Inches(2.5), Inches(4.5), Inches(5.1)],
        font_size=11,
    )

    # 39 Platform modality
    table_content(
        prs,
        "Platform",
        "Platform / modality detect",
        ["Canonical modality", "Aliases"],
        [
            ["illumina", "dnbseq, ultima, short_read"],
            ["pacbio_hifi", "hifi, pacbio, ccs, revio"],
            ["ont", "nanopore, ont_r10"],
            ["rna_seq", "—"],
            ["bisulfite", "—"],
            ["single_cell", "—"],
            ["linked_reads", "10x, chromium"],
        ],
        col_widths=[Inches(3.5), Inches(8.6)],
        note="validate_modality() resolves aliases; FASTQ mod= header overrides; uBAM auto-detected",
    )

    # 40 AccelContext
    table_content(
        prs,
        "Platform",
        "AccelContext — hardware platform detect",
        ["Tier", "Requirement", "Used for"],
        [
            ["cuda_rawkernel", "NVIDIA + CuPy", "k-mer lookup, chain DP, banded SW score"],
            ["triton", "CUDA + Triton", "Fused LN+Linear+GELU"],
            ["torch_cuda / xpu / mps", "Vendor GPU", "Batched torch"],
            ["torch_cpu", "Always", "Reference path"],
        ],
        col_widths=[Inches(3.2), Inches(3.2), Inches(5.7)],
        note="print(AccelContext().summary())  ·  ROCm as torch.cuda but vendor=amd — RawKernels NVIDIA-only",
    )

    # 41 Hardware
    table_content(
        prs,
        "Hardware",
        "Hardware requirements",
        ["Target", "Stack", "Notes"],
        [
            ["H100 / A100", "CuPy + Triton + mamba_ssm + Flash SDP + bf16", "Full speed path"],
            ["Consumer NVIDIA", "torch CUDA + AMP; CuPy if installed", "sm_70+ fp16; sm_80+ bf16/TF32"],
            ["Apple Silicon", "MPS + PyTorch; MLX optional", "eigh→CPU; no CuPy/mamba_ssm"],
            ["CPU / Docker", "Pure PyTorch SSD scan", "linux/amd64; Rosetta on Mac"],
            ["Memory", "~15M params; graph+indexes dominate", "chr21.d9.vg ~1GB; batch_size=16"],
        ],
        col_widths=[Inches(2.4), Inches(5.5), Inches(4.2)],
        font_size=11,
        note="Python ≥ 3.10 · scripts/check_gpu.py · scripts/smoke_test.py",
    )

    # 42 Backend override
    code_slide(
        prs,
        "Hardware",
        "Per-stage backend override",
        """AccelConfig.stage_backends = {
  "seeding": "auto",      # or "torch" | "cuda_rawkernel"
  "chaining": "auto",
  "extension": "auto",
}
# Global: tf32, flash_sdp, cudnn_benchmark, amp, cuda_graphs, compile
# Extension CIGAR traceback always on torch (CUDA kernel = score-only)

TF32: NVIDIA sm_80+
FP8: sm_89+ (gated)
torch.compile skipped on MPS""",
    )

    # 43 Data types
    table_content(
        prs,
        "Data",
        "Data types & I/O formats",
        ["Direction", "Formats"],
        [
            ["Input reads", "FASTQ (.gz), BAM, uBAM, SAM, CRAM"],
            ["Input graph", "GFA (→ GBZ via vg)"],
            ["Output", "BAM, CRAM, GFA, GBZ"],
            ["Labels / seeds", "labels.json · graph.gfa zt:Z tags"],
            ["Training bundle", ".pt + emit-dir: fasta/fastq/truth.sam/gfa/json"],
        ],
        col_widths=[Inches(3.0), Inches(9.1)],
        note="In-memory: ReadRecord, PangenomeGraph, GraphBatch, AnchorSet, Chain, AlignmentRecord",
    )

    # 44 Format contract
    code_slide(
        prs,
        "Data",
        "End-to-end format contract",
        """reads = read_reads("sample.fastq.gz", modality="ont")  # or bam/ubam/sam/cram
graph = read_gfa("pangenome.gfa")
results, stats = pipeline.align(reads, reference)
write_alignments(results, reads, "out.bam", references=refs)
write_gbz("out.gfa", "out.gbz")   # requires vg binary

• Unmapped written (not dropped) so counts match
• Aligned BAM skips unmapped by default; uBAM includes them
• Tests: test_formats.py, test_end_to_end_formats.py""",
    )

    # 45 Synthetic HPRC
    content_slide(
        prs,
        "Data",
        "Synthetic & HPRC data",
        items=[
            "Synthetic mirrors AGNES Table 1 (arXiv:2510.16013): GC 40–50%, repeats 10–15%, 15% errors (2× in homopolymers)",
            "Presets: tiny · long · table1 (640/160/200)",
            "HPRC: ~44 graph-training samples; GIAB eval default HG005 chr21",
            "Curriculum vision: GRCh37/38 → +T2T → HPRC R1 → R2",
        ],
    )

    # 46 Curriculum
    table_content(
        prs,
        "Training",
        "Curriculum & data splits",
        ["Stage", "Data"],
        [
            ["0 Smoke", "synthetic tiny · CPU minutes"],
            ["1 Linear", "GRCh37 / GRCh38 simulated"],
            ["2 T2T", "+ CHM13 / gapless regions"],
            ["3 HPRC R1", "~47 assemblies / graphs"],
            ["4 HPRC R2", "~159 + held-out GIAB"],
        ],
        col_widths=[Inches(2.5), Inches(9.6)],
        note="Synthetic table1: 640 train / 160 val / 200 test · chr21 pilot SAMPLE=HG005",
    )

    # 47 Multitask heads
    table_content(
        prs,
        "Multitask",
        "Multi-task heads (10 + ancestry_local)",
        ["Head", "Scope", "Output"],
        [
            ["variant_calling", "node", "3 genotypes + GQ"],
            ["sv_genotyping", "node", "5 SV types + 2 breakpoints"],
            ["copy_number", "node", "CN 0–5 + continuous"],
            ["haplotype", "read", "phase 0/1"],
            ["hla_typing", "read", "128 allele logits"],
            ["ancestry (+ local)", "read / node", "5 populations"],
            ["somatic", "read", "4-class: germline / somatic / artifact / absent"],
            ["pgx", "read", "32 star alleles"],
            ["bqsr", "base", "42 quality bins"],
            ["methylation", "base", "2 (unmeth / meth)"],
        ],
        col_widths=[Inches(2.8), Inches(2.2), Inches(7.1)],
        font_size=11,
        note="All opt-in via MultiTaskConfig · branching MLP d_model→128→out",
    )

    # 48 Somatic subclonal
    two_col_slide(
        prs,
        "Multitask",
        "Somatic & subclonal",
        "Implemented — Somatic head",
        [
            "num_somatic_classes = 4",
            "germline | somatic | artifact | absent",
            "logits = MLP(pooled)  # (B, 4)",
            "loss = CrossEntropy per-read",
            "Tumor vs normal style labels",
            "Shares backbone; one small MLP",
        ],
        "Roadmap — Subclonal",
        [
            "Clone assignment / phylogeny",
            "VAF regression per variant",
            "Per-node somatic (not only per-read)",
            "Synthetic labels for subclones",
            "Coverage-aware purity / multiplicity",
            "No SubclonalHead in code yet",
        ],
        note="Extend MultiTaskHeads when subclonal labels exist",
    )

    # 49 Multitask forward
    code_slide(
        prs,
        "Multitask",
        "Pseudocode — multitask forward",
        """def MultiTaskGraphMamba.forward(...):
    out = GraphMambaModel.forward(...)
    out.multitask = {}
    for name, head in task_heads.items():
        scope = SCOPES[name]   # read | node | base
        if scope == "node":
            out.multitask[name] = head(out.graph_nodes)   # (B,N,C)
        elif scope == "base":
            out.multitask[name] = head(out.read_hidden)   # (B,L,C)
        else:
            out.multitask[name] = head(out.pooled)        # (B,C)
    return out""",
    )

    # 50 Predictive genomics
    content_slide(
        prs,
        "Multitask",
        "Predictive genomics & clinical regions (roadmap)",
        items=[
            "Architecture catalogue: 86 clinical regions (HLA, PGx, cancer, cardiac, neuro, …)",
            "GenomePredictor vision: stream → aggregate → refine → VCF 4.3 / JSON report",
            "Heads already emit per-read / per-node / per-base logits during alignment",
            "Stage 7 aggregation + clinical DB wiring = future work",
        ],
        note="Zero marginal backbone cost for heads — only small MLPs on shared states",
    )

    # 51 Training
    table_content(
        prs,
        "Training",
        "Training pipeline",
        ["Setting", "Default"],
        [
            ["Optimizer", "AdamW · lr=3e-4 · wd=0.01"],
            ["Schedule", "10% linear warmup + cosine"],
            ["Grad clip", "1.0"],
            ["AMP", "AccelContext.autocast() bf16/fp16"],
            ["Early stop", "patience=4 on chain_accuracy (not loss)"],
            ["Forward includes", "model + score_seeds + score_chains"],
            ["Kendall s_i", "Optimized jointly with params"],
        ],
        col_widths=[Inches(3.0), Inches(9.1)],
        note="TargetBuilder: real seeding+chaining; anchor true if within 20 bp; chain = max overlap",
    )

    # 52 Validation
    table_content(
        prs,
        "Training",
        "Validation metrics",
        ["Metric", "Meaning"],
        [
            ["locus_accuracy", "Pipeline placement within 50 bp"],
            ["chain_accuracy", "Top chain correct (≥2 candidates)"],
            ["anchor_auc / precision / recall", "Seed head quality"],
            ["mapq_mae", "MAPQ regression error"],
            ["mapq_calibration", "Expected vs observed error"],
            ["mapped_fraction", "Fraction of reads mapped"],
        ],
        col_widths=[Inches(4.0), Inches(8.1)],
        note="Probes: activation stats, grad norms, router distribution · scripts/train.py",
    )

    # 53 Loss
    table_content(
        prs,
        "Loss",
        "Loss — AlignmentLoss + Kendall",
        ["Term", "Type", "w"],
        [
            ["Seed", "BCE logits", "1.0"],
            ["Chain", "Listwise CE (ordering)", "1.0"],
            ["Node", "CE + label smooth 0.05", "1.0"],
            ["Position / MAPQ", "Huber", "1.0 / 0.5"],
            ["Router", "one-sided (mean_cost−0.6)²", "0.05"],
            ["Extension margin", "Hinge best vs decoy", "0.5"],
        ],
        col_widths=[Inches(3.0), Inches(6.5), Inches(2.6)],
        note="Kendall arXiv:1705.07115: L = Σ_i [ exp(−s_i)·L_i + s_i ] · Missing labels SKIPPED",
    )

    # 54 Multitask loss
    code_slide(
        prs,
        "Loss",
        "MultiTaskLoss scopes",
        """# read scope:  CE(logits[B,C], y[B])
# node/base:   CE(flat, y_flat, ignore_index=-100)
# aux:         Huber on {name}_aux with optional mask

# GraphMambaLoss = AlignmentLoss ⊕ MultiTaskLoss
# multitask scaled by w_multitask then Kendall-balanced with alignment terms

# Partially labelled batches train only heads that have labels""",
    )

    # 55 Train step
    code_slide(
        prs,
        "Loss",
        "Pseudocode — training step",
        """for batch in loader:
  with accel.autocast():
    out = model(...)
    seed_logits = model.score_seeds(out, anchors)
    chain_logits = model.score_chains(out, chains)
    loss, parts = criterion(out, targets, seed_logits, chain_logits)
  scaler.scale(loss).backward()
  clip_grad_norm_(params, 1.0)
  scaler.step(opt); scaler.update(); scheduler.step()
# validate → early_stop on chain_accuracy""",
    )

    # 56 Benchmarks
    table_content(
        prs,
        "Coverage",
        "Benchmark platforms (architecture targets)",
        ["Platform", "Notes"],
        [
            ["Illumina 150bp", "High fast-path utilization"],
            ["ONT R10.4.1 / ultra-long", "Long-read focus"],
            ["PacBio HiFi", "More neural fallback expected"],
            ["Ultima / DNBSEQ / 10x", "Short / linked modalities"],
            ["Ion Torrent", "Homopolymer-heavy → neural-heavy"],
        ],
        col_widths=[Inches(4.0), Inches(8.1)],
        note="chr21 mentor: Giraffe vs ours BAM → DeepVariant → Sniffles → hap.py compare",
    )

    # 57 Citations main
    table_content(
        prs,
        "Citations",
        "Citations — algorithms & models",
        ["Topic", "Reference"],
        [
            ["minimap2 chaining / hash64", "Li, Bioinformatics 2018"],
            ["BWA-MEM / FM-index SMEMs", "Li & Durbin 2009; Li 2013"],
            ["vg / variation graphs", "Garrison et al. Nat Biotech 2018"],
            ["Giraffe", "Sirén et al. Science 2021"],
            ["GraphAligner", "Rautiainen & Marschall 2020"],
            ["WFA", "Marco-Sola et al. Bioinformatics 2021"],
            ["GATv2", "Brody et al. arXiv:2105.14491"],
            ["Mamba / Mamba-2", "Gu & Dao 2023; Dao & Gu 2024"],
            ["MambaFormer", "Park et al. arXiv:2402.04248"],
            ["Kendall multi-task", "Kendall et al. arXiv:1705.07115"],
            ["AGNES synthetic stats", "arXiv:2510.16013v3"],
            ["HPRC", "Liao et al. Nature 2023"],
        ],
        col_widths=[Inches(4.0), Inches(8.1)],
        font_size=11,
    )

    # 58 Related
    content_slide(
        prs,
        "Citations",
        "Citations — related aligners & callers",
        items=[
            "Langmead & Salzberg, Bowtie2 — Nat Methods 2012",
            "Kim et al., HISAT2 — Nat Biotech 2019",
            "Jain et al., Winnowmap2 — Bioinformatics 2022",
            "Li, minigraph — Genome Biol 2020",
            "Poplin et al., DeepVariant — Nat Biotech 2018",
            "Sedlazeck et al., Sniffles — Nat Methods 2018 / Sniffles2",
            "Human Pangenome Reference Consortium — humanpangenome.org",
        ],
    )

    # 59 Code-linked
    content_slide(
        prs,
        "Citations",
        "Code-linked references",
        items=[
            "layers/gat.py — Brody et al. arXiv:2105.14491",
            "blocks/mambaformer.py — Park et al. arXiv:2402.04248",
            "losses/alignment_loss.py — Kendall et al. arXiv:1705.07115",
            "alignment/chaining.py — minimap2-style affine + log gap",
            "data/synthetic.py — AGNES arXiv:2510.16013 · minimap2 hash64",
            "README.md — krafton-ai/mambaformer-icl Mamba-1 port notes",
        ],
    )

    # 60 Status
    table_content(
        prs,
        "Summary",
        "Implementation status snapshot",
        ["Area", "Status"],
        [
            ["Stages 1–5 (seed→post)", "✓ Implemented"],
            ["hybrid / fast / two_pass", "✓ Implemented"],
            ["GraphMamba + multitask heads", "✓ Implemented (heads opt-in)"],
            ["AccelContext tiers", "✓ Implemented"],
            ["Formats FASTQ/BAM/uBAM/GFA/CRAM", "✓ Implemented"],
            ["Router FLOP gating in pipeline", "△ Labels only today"],
            ["Stages 6–7, subclonal", "○ Roadmap"],
            ["chr21 Giraffe compare", "✓ Scripts ready"],
        ],
        col_widths=[Inches(5.0), Inches(7.1)],
    )

    # 61 Takeaways
    content_slide(
        prs,
        "Summary",
        "Key takeaways",
        numbered_items=[
            "Graph alignment landscape: Giraffe / GraphAligner / vg + linear giants",
            "GraphMamba = classical seed–chain–extend + BiMamba/GAT neural core",
            "Adaptive mining: multiplex-k, BFS hop bonus, two-pass, router",
            "Train with Kendall-balanced multi-task loss; validate on chain/locus accuracy",
            "Same AccelContext from laptop CPU → H100; modalities auto-resolved",
        ],
    )

    # 62 Thank you
    title_slide(
        prs,
        "Thank you",
        "GraphMambaFormer · Pangenome Neural Alignment\narchitecture/GraphMamba_Slides.pptx",
        chips=["Questions?", "See also: GraphMamba_Architecture.html"],
    )

    out = Path(__file__).resolve().parent / "GraphMamba_Slides.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
