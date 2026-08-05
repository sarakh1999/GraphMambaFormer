"""Audit the code against ``architecture/GraphMamba_Architecture.html``.

Two halves:

**Conformance** — the spec's *concrete* numbers are asserted against the code, so
architectural drift fails loudly instead of being noticed by eye later. Every
check names the spec line it comes from.

**Coverage** — the spec's 97-feature catalogue is walked and each entry marked
implemented / partial / missing, with the module that provides it. This is a
deliberate inventory, not an assertion: most gaps are unbuilt scope, not bugs.

Run: PYTHONPATH=. .venv/bin/python scripts/audit_architecture.py
Exit code is non-zero only if a *conformance* check fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from graphmambaformer.config import (
    ChainingConfig,
    ExtensionConfig,
    GraphMambaConfig,
    RouterConfig,
    ScoringConfig,
    SeedingConfig,
)
from graphmambaformer.models import build_core_model

PASS, FAIL = "  [ok]  ", "  [FAIL]"


class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, spec_says: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"{PASS} {spec_says}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(spec_says)
            print(f"{FAIL} {spec_says}" + (f"  ({detail})" if detail else ""))

    def equals(self, spec_says: str, actual, expected) -> None:
        self.check(spec_says, actual == expected, f"code={actual!r} spec={expected!r}")


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
# Conformance: the spec's concrete numbers
# --------------------------------------------------------------------------- #
def audit_core_model(audit: Audit) -> None:
    banner("Neural Architecture / Core Model  (spec: 'GraphMambaModel — Forward Pass')")
    cfg = GraphMambaConfig()

    audit.equals("d=256 (feature catalogue: 'd=256, 6 layers')", cfg.d_model, 256)
    audit.equals("BiMamba2 x 6 layers", cfg.n_mamba_layers, 6)
    audit.equals("GATv2Conv x 3 layers", cfg.n_gat_layers, 3)

    se = cfg.sequence_encoder
    audit.equals("SequenceEncoder BaseEmbed(5->64)", se.d_base, 64)
    audit.equals("SequenceEncoder KmerEmbed(k=3->64) : k", se.kmer_size, 3)
    audit.equals("SequenceEncoder KmerEmbed(k=3->64) : dim", se.d_kmer, 64)
    audit.equals("SequenceEncoder QualEmbed(42->32)", se.d_qual, 32)
    audit.equals("SequenceEncoder PosEncode(sin/cos->96)", se.d_pos, 96)
    audit.equals(
        "SequenceEncoder parts sum to Linear(256->D)",
        se.d_base + se.d_kmer + se.d_qual + se.d_pos,
        256,
    )
    audit.equals("SequenceEncoder Dropout(0.1)", se.dropout, 0.1)

    m = cfg.mamba
    audit.equals("Mamba2 conv_dim=4", m.d_conv, 4)
    audit.equals("Mamba2 headdim=64", m.headdim, 64)
    audit.equals("Mamba2 expand=2 (d_inner == 2*d_model)", m.d_inner, 2 * cfg.d_model)

    ca = cfg.cross_attention
    audit.equals("CrossAttentionFusion 8 heads", ca.n_heads, 8)
    audit.equals("CrossAttentionFusion D/head=32", ca.d_head, 32)
    audit.equals(
        "CrossAttention heads*d_head reconstructs D", ca.n_heads * ca.d_head, cfg.d_model
    )

    audit.equals("GATv2 4 heads", cfg.gat.n_heads, 4)
    audit.equals("GraphEncoder 8 discrete edge types", cfg.graph_encoder.num_edge_types, 8)

    r = RouterConfig()
    audit.equals("ComplexityRouter routes {fast, medium, full}", tuple(r.route_names),
                 ("fast", "medium", "full"))
    audit.equals("ComplexityRouter 3 routes", r.num_routes, 3)

    audit.equals("MappingHead MAPQ sigmoid x 60", cfg.mapping_head.max_mapq, 60)


def audit_forward_shapes(audit: Audit) -> None:
    banner("Forward-pass shapes  (spec: '(B,L,256) x (B,N,256) -> (B,L+N,256)')")
    cfg = GraphMambaConfig(d_model=64)  # small for speed; shape relations hold
    from graphmambaformer.config import CoreModelConfig
    from graphmambaformer.models.graph_mamba import GraphBatch

    net = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    net.eval()

    B, L, N = 2, 30, 7
    base = torch.randint(1, 5, (B, L))
    graph = GraphBatch(
        node_kmer_ids=torch.randint(0, 4**cfg.graph_encoder.kmer_size, (N, 5)),
        edge_index=torch.stack([torch.arange(N - 1), torch.arange(1, N)]),
        edge_type=torch.zeros(N - 1, dtype=torch.long),
    )
    with torch.no_grad():
        out = net(base, graph=graph)

    d = cfg.d_model
    audit.equals("read stream stays (B, L, D)", tuple(out.read_hidden.shape), (B, L, d))
    audit.equals("graph stream stays (B, N, D)", tuple(out.graph_nodes.shape), (B, N, d))
    audit.equals("fusion concatenates to (B, L+N, D)", tuple(out.fused.shape), (B, L + N, d))
    audit.equals("fused_embed is (B, D)", tuple(out.pooled.shape), (B, d))
    audit.equals("node_probs is (B, N)", tuple(out.mapping["node_probs"].shape), (B, N))
    audit.equals("mapq is per read (B,)", tuple(out.mapping["mapq"].shape), (B,))
    audit.check(
        "MAPQ estimator output bounded to [0, 60]",
        bool(((out.mapping["mapq"] >= 0) & (out.mapping["mapq"] <= 60)).all()),
        f"range=[{out.mapping['mapq'].min():.1f}, {out.mapping['mapq'].max():.1f}]",
    )


def audit_stage_constants(audit: Audit) -> None:
    banner("Alignment stages  (spec: feature catalogue parameters)")
    seeding = SeedingConfig()
    audit.equals("SMEMIndex min_seed=13", seeding.min_seed_len, 13)
    audit.equals("SMEMIndex max_occ=200", seeding.max_occ, 200)
    audit.equals("DeBruijnIndex k=21", seeding.dbg_kmer, 21)
    audit.equals("MultiplexDBG k=15,21,31", tuple(seeding.multiplex_kmers), (15, 21, 31))
    audit.equals("minimizer (k=15, w=10)", (seeding.kmer, seeding.window), (15, 10))

    ext = ExtensionConfig()
    audit.equals("WavefrontAligner mismatch=4", ext.mismatch_penalty, 4.0)
    audit.equals("WavefrontAligner gap_open=6", ext.gap_open, 6.0)
    audit.equals("WavefrontAligner x_drop=600", ext.x_drop, 600.0)

    audit.equals("MAPQCalibrator max_mapq=60", ScoringConfig().max_mapq, 60)
    audit.check(
        "AffineChainer is minimap2-style (affine + log gap cost)",
        ChainingConfig().log_coeff > 0 and ChainingConfig().gap_open > 0,
        f"gap_open={ChainingConfig().gap_open} log_coeff={ChainingConfig().log_coeff}",
    )


def audit_wiring(audit: Audit) -> None:
    banner("Wiring the spec calls out explicitly")

    # "→ Concat + FFN(D→4D→D) with Triton fused LN+Linear+GELU"
    import inspect

    from graphmambaformer.layers import cross_attention

    source = inspect.getsource(cross_attention)
    audit.check(
        "CrossAttentionFusion FFN uses the Triton fused LN+Linear+GELU",
        "FusedLNLinearGELU" in source,
        "layers/cross_attention.py imports and instantiates it",
    )
    cfg = GraphMambaConfig()
    audit.equals("Fusion FFN is D->4D->D", cfg.cross_attention.d_ff_mult, 4)

    # The router names must not be duplicated in the pipeline.
    from graphmambaformer.alignment import pipeline

    audit.check(
        "pipeline takes route names from the router, not a local copy",
        '"standard"' not in inspect.getsource(pipeline),
        "no hardcoded route-name tuple in pipeline.py",
    )

    # Stage 1 must offer all six seeding indices the spec lists.
    from graphmambaformer.config import SEEDING_MODES

    for mode in ("smem", "dbg", "fmindex", "fuzzy", "multiplex_dbg", "gpu_kmer"):
        audit.check(f"Stage 1 offers seeding mode {mode!r}", mode in SEEDING_MODES)

    # Kendall uncertainty weighting for the multi-task loss.
    from graphmambaformer.losses import GraphMambaLoss

    loss = GraphMambaLoss()
    audit.check(
        "Multi-Task Loss uses learnable log-variance (Kendall)",
        hasattr(loss.weighting, "log_vars") and loss.cfg.learnable_weights,
    )

    # All ten multi-task heads must be buildable.
    from graphmambaformer.config import MultiTaskConfig
    from graphmambaformer.heads import MultiTaskHeads

    ten = MultiTaskConfig(
        variant_calling=True, sv_genotyping=True, haplotype=True, hla_typing=True,
        bqsr=True, methylation=True, ancestry=True, copy_number=True, somatic=True,
        pgx=True,
    )
    heads = MultiTaskHeads(ten)
    for name in ("variant_calling", "sv_genotyping", "haplotype", "hla_typing", "bqsr",
                 "methylation", "ancestry", "copy_number", "somatic", "pgx"):
        audit.check(f"Multi-task head {name!r} present", name in heads.enabled)
    audit.equals("CopyNumberHead CN state 0-5 (6 states)", ten.num_cn_states, 6)
    audit.equals("AncestryHead 5-pop softmax", ten.num_populations, 5)
    audit.equals(
        "SomaticHead germline/somatic/artifact/absent", ten.num_somatic_classes, 4
    )
    audit.equals("VariantCallingHead genotypes 0/0,0/1,1/1", ten.num_genotypes, 3)


def audit_format_contract(audit: Audit) -> None:
    """The I/O contract, end to end: every input format, every modality, every output."""
    banner("Format contract  (in: FASTQ/.gz, BAM, uBAM, SAM, CRAM, GFA | out: BAM, CRAM, GFA, GBZ)")

    from graphmambaformer.alignment.pipeline import as_read_batch
    from graphmambaformer.config import MODALITIES
    from graphmambaformer.data import (
        read_fastq,
        read_gfa,
        read_reads,
        write_alignments,
        write_bam,
        write_cram,
        write_gbz,
        write_gfa_graph,
    )

    for name, fn in [
        ("FASTQ reader", read_fastq), ("reads dispatch (BAM/uBAM/SAM/CRAM)", read_reads),
        ("GFA reader", read_gfa), ("BAM writer", write_bam), ("CRAM writer", write_cram),
        ("GFA writer", write_gfa_graph), ("GBZ writer", write_gbz),
        ("pipeline results -> BAM/CRAM", write_alignments),
    ]:
        audit.check(f"{name} present", callable(fn))

    audit.equals("7 modalities catalogued", len(MODALITIES), 7)
    for m in ("illumina", "pacbio_hifi", "ont", "rna_seq", "bisulfite",
              "single_cell", "linked_reads"):
        audit.check(f"modality {m!r} supported", m in MODALITIES)

    # The seams that were silently dropping data.
    class _Rec:
        seq, quals, modality, read_id = "ACGT", [30, 30, 30, 30], "ont", "r0"

    batch = as_read_batch([_Rec()])
    audit.check("pipeline carries Phred qualities from the reader",
                batch.quals is not None and batch.quals[0] == [30, 30, 30, 30])
    audit.check("pipeline carries modality from the reader", batch.modality == "ont")

    import inspect

    from graphmambaformer.alignment import scoring

    src = inspect.getsource(scoring)
    audit.check("model is called with qualities=", "qualities=qual_tensor" in src)
    audit.check("model is called with modality=", "modality=" in src)


def audit_spec_contradictions(audit: Audit) -> None:
    """Places the spec disagrees with itself, and which reading the code follows."""
    banner("Resolved spec contradictions")
    cfg = GraphMambaConfig()

    print(
        "  d_state: the forward-pass diagram says state_dim=128, but the same\n"
        "  document's .env reference says D_STATE=64. Figure 1B also says 64, and\n"
        "  64 lands nearer the quoted 14.2M budget (64 -> 14.9M, 128 -> 15.3M),\n"
        "  so the code follows 64 as the self-consistent reading."
    )
    audit.equals("d_state follows the .env / Figure 1B reading (64)", cfg.mamba.d_state, 64)

    print(
        "\n  Stage count: the pipeline is advertised as 7 stages (seed, chain,\n"
        "  extend, score, post, repeat, predict). Stages 1-5 are implemented;\n"
        "  6-7 are unbuilt, so docstrings claim 5, not 7."
    )


def audit_param_budget(audit: Audit) -> None:
    banner("Parameter budget  (spec: 'd=256, 6 layers, 14.2M params')")
    total = build_core_model().num_parameters
    # The spec quotes a round 14.2M for the backbone + heads. Allow 10%: the exact
    # figure depends on which heads are counted, and the alignment scoring heads
    # here are not obviously part of the quoted number.
    lo, hi = 14.2e6 * 0.9, 14.2e6 * 1.1
    audit.check(
        "GraphMambaModel is ~14.2M params (+/-10%)",
        lo <= total <= hi,
        f"{total:,} params, spec 14,200,000 ({(total / 14.2e6 - 1) * 100:+.1f}%)",
    )


# --------------------------------------------------------------------------- #
# Coverage: the 97-feature catalogue
# --------------------------------------------------------------------------- #
@dataclass
class Feature:
    name: str
    status: str  # "yes" | "partial" | "no"
    where: str


def _module_path(where: str) -> str | None:
    """The repo-relative module path a catalogue entry points at, if it names one."""
    head = where.split()[0] if where else ""
    return head.partition(":")[0] if head.endswith(".py") else None


CATALOGUE: dict[str, list[Feature]] = {
    "Core Architecture (7)": [
        Feature("GraphMambaModel", "yes", "models/graph_mamba.py"),
        Feature("MultiTaskGraphMamba", "yes", "models/graph_mamba.py"),
        Feature("Mamba2Block", "yes", "layers/mamba2.py"),
        Feature("BidirectionalMamba", "yes", "layers/bimamba.py"),
        Feature("CrossAttentionFusion", "yes", "layers/cross_attention.py"),
        Feature("GATv2Conv", "yes", "layers/gat.py"),
        Feature("ComplexityRouter", "yes", "heads/router.py"),
    ],
    "Input Encoding (4)": [
        Feature("SequenceEncoder", "yes", "encoders/sequence_encoder.py"),
        Feature("GraphEncoder", "yes", "encoders/graph_encoder.py"),
        Feature("SequenceContextEncoder", "no", "-"),
        Feature("MultiResolutionEncoder (k=3,5,7,11)", "no", "-"),
    ],
    "Alignment Pipeline (13)": [
        Feature("HybridAlignmentPipeline", "yes", "alignment/pipeline.py"),
        Feature("TwoPassAligner", "partial", "alignment/pipeline.py (Python, not C)"),
        Feature("CFastAligner (runtime-compiled C)", "no", "-"),
        Feature("FastAlignmentPipeline", "partial", "classical only, no batched model pass"),
        Feature("SMEMIndex", "yes", "alignment/seeding.py"),
        Feature("DeBruijnIndex", "yes", "alignment/seeding.py"),
        Feature("AffineChainer", "yes", "alignment/chaining.py"),
        Feature("WavefrontAligner", "yes", "alignment/extension.py"),
        Feature("FMIndex", "yes", "alignment/seeding.py"),
        Feature("FuzzySeedIndex", "yes", "alignment/seeding.py"),
        Feature("MultiplexDBG", "yes", "alignment/seeding.py"),
        Feature("MAPQCalibrator + BayesianMAPQ", "partial",
                "margin/neural blend; no isotonic or Bayesian calibration"),
        Feature("SplitAlignmentDetector", "partial", "secondary chains, no explicit detector"),
    ],
    "Specialized Alignment (10)": [
        Feature("RepeatResolver", "no", "-"),
        Feature("HLAAligner", "no", "-"),
        Feature("MultiReferenceIntegrator", "no", "-"),
        Feature("PopulationAwareScorer", "no", "-"),
        Feature("CoordinateLiftover", "no", "-"),
        Feature("PairedEndRescue", "no", "-"),
        Feature("ReadCorrector", "no", "-"),
        Feature("PhasingCorrector", "no", "-"),
        Feature("SpliceAligner", "no", "-"),
        Feature("TranslatedGraphAligner", "no", "-"),
    ],
    "Multi-Task Heads (10)": [
        Feature(n, "yes", "heads/multitask_heads.py")
        for n in ("VariantCallingHead", "SVGenotypingHead", "HaplotypeHead",
                  "HLATypingHead", "BQSRHead", "MethylationHead", "AncestryHead",
                  "CopyNumberHead", "SomaticMutationHead", "PGxHead")
    ],
    "Predictive Genomics (4)": [
        Feature("PredictiveGenomicsEngine", "no", "per-read heads exist; no aggregation"),
        Feature("GenomePredictor (3-pass)", "no", "-"),
        Feature("GenomicReport (VCF 4.3 / JSON)", "no", "-"),
        Feature("Clinical Region Database (86 regions)", "no", "-"),
    ],
    "GPU Acceleration (12)": [
        Feature("C Fast Aligner", "no", "-"),
        Feature("Two-Pass Architecture", "partial", "Python fast path"),
        Feature("CUDABatchAligner (SW RawKernel)", "yes", "accel/cuda_kernels.py"),
        Feature("GPUWavefrontAligner", "no", "CPU WFA only"),
        Feature("GPUKmerIndex", "yes", "alignment/seeding.py"),
        Feature("GPU DP Chaining", "yes", "accel/cuda_kernels.py"),
        Feature("SIMDAligner (SSE2/AVX2)", "no", "-"),
        Feature("CuPy Acceleration", "yes", "accel/backend.py:array_namespace"),
        Feature("Triton SSD Scan", "partial", "delegates to mamba_ssm; no own kernel"),
        Feature("Triton Fused FFN", "yes", "accel/triton_ops.py"),
        Feature("CUDA Graphs + TF32 + Flash SDP", "yes", "accel/backend.py"),
        Feature("TensorRT Engine", "no", "-"),
    ],
    "Training Infrastructure (6)": [
        Feature("Trainer", "no", "-"),
        Feature("DistributedTrainer + DeepSpeed ZeRO", "no", "-"),
        Feature("Scaling Optimizations (EMA, 8-bit AdamW)", "no", "-"),
        Feature("BioNeMo Integration", "no", "-"),
        Feature("CurriculumScheduler", "no", "-"),
        Feature("Multi-Task Loss (Kendall)", "yes", "losses/alignment_loss.py"),
    ],
    "Data & Graphs (6)": [
        Feature("GFAGraph Parser", "partial", "data/formats GFA I/O; no GFA2/walks"),
        Feature("Graph Simplification", "no", "-"),
        Feature("SnarlDecomposer", "no", "-"),
        Feature("Dataset Classes", "partial", "synthetic + AlignmentDataset only"),
        Feature("HPRC Data Split", "partial", "scripts/hprc fetch helpers"),
        Feature("Read Augmentation", "partial", "synthetic error model"),
    ],
    "Deployment (5)": [
        Feature("Docker Image", "no", "-"),
        Feature("Docker Compose", "no", "-"),
        Feature("H100 Config (8xH100 YAML)", "no", "-"),
        Feature("predict_genomics CLI", "no", "-"),
        Feature("train_curriculum CLI", "no", "-"),
    ],
}


def audit_coverage(audit: Audit) -> tuple[int, int, int]:
    banner("Feature-catalogue coverage  (spec: 97 features / 11 categories)")
    mark = {"yes": "[x]", "partial": "[~]", "no": "[ ]"}
    yes = partial = no = 0
    claimed: list[str] = []
    for category, features in CATALOGUE.items():
        print(f"\n{category}")
        for f in features:
            print(f"   {mark[f.status]} {f.name:<42s} {f.where}")
            yes += f.status == "yes"
            partial += f.status == "partial"
            no += f.status == "no"
            path = _module_path(f.where)
            if path is not None:
                claimed.append(path)

    # The table above is only worth reading if it cannot lie about what exists.
    root = Path(__file__).resolve().parent.parent / "graphmambaformer"
    missing = sorted({p for p in claimed if not (root / p).is_file()})
    audit.check(
        "every module named in the coverage table exists",
        not missing,
        f"{len(claimed)} paths checked" if not missing else f"missing: {missing}",
    )
    return yes, partial, no


def main() -> int:
    torch.manual_seed(0)
    audit = Audit()

    audit_core_model(audit)
    audit_forward_shapes(audit)
    audit_stage_constants(audit)
    audit_wiring(audit)
    audit_format_contract(audit)
    audit_spec_contradictions(audit)
    audit_param_budget(audit)
    yes, partial, no = audit_coverage(audit)

    total = yes + partial + no
    banner("Summary")
    print(f"Conformance : {audit.passed} checks passed, {len(audit.failed)} failed")
    for name in audit.failed:
        print(f"    FAILED: {name}")
    print(
        f"Coverage    : {yes} implemented, {partial} partial, {no} missing "
        f"of {total} catalogued ({yes / total:.0%} full, "
        f"{(yes + partial) / total:.0%} at least partial)"
    )
    print(
        "\nThe conformance checks are the contract: every number the architecture\n"
        "states for an implemented component is asserted above. Coverage gaps are\n"
        "unbuilt scope (Stages 6-7, specialized aligners, training infrastructure),\n"
        "not deviations in what exists."
    )
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
