"""Dual-reference alignment: linear + pangenome in one pass, shared work.

Checks that :class:`DualReferenceAligner` aligns the same reads against several
references in a single sweep, that reusing the read encoding across references
does not change the result (it must equal aligning each reference separately),
that the integrated concordance output is produced, and that the one-FASTA-read
:func:`build_dual_reference_from_files` builds both arms in a shared frame.

Pytest-compatible and self-contained; also runs via ``tests/run_all.py``.
"""

import os
import tempfile

import numpy as np
import torch

from graphmambaformer.alignment import (
    DualAlignmentResult,
    DualReferenceAligner,
    build_pipeline,
)
from graphmambaformer.config import CoreModelConfig, GraphMambaConfig, PipelineConfig
from graphmambaformer.data import build_dual_reference_from_files
from graphmambaformer.models import build_core_model
from graphmambaformer.models.graph_mamba import GraphBatch

RNG = np.random.default_rng(11)
BASES = "ACGT"


def make_reference(length=3000):
    return "".join(RNG.choice(list(BASES), size=length))


def mutate(seq, rate=0.02):
    out = list(seq)
    for i in range(len(out)):
        if RNG.random() < rate:
            out[i] = RNG.choice(list(BASES))
    return "".join(out)


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def make_reads(ref, n=8, read_len=200, rate=0.02):
    reads = []
    for i in range(n):
        start = int(RNG.integers(0, len(ref) - read_len))
        seq = mutate(ref[start : start + read_len], rate)
        if i % 3 == 2:
            seq = revcomp(seq)
        reads.append(seq)
    return reads


def make_graph(cfg, n_nodes):
    node_k = torch.randint(0, 4 ** cfg.graph_encoder.kmer_size, (n_nodes, 6))
    src = torch.arange(n_nodes - 1)
    edges = torch.stack([src, src + 1])
    graph = GraphBatch(
        node_kmer_ids=node_k,
        edge_index=edges,
        edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
    )
    return graph, edges.numpy()


def build(mode="hybrid", with_model=True, **cfg_kwargs):
    cfg = PipelineConfig(mode=mode, **cfg_kwargs)
    cfg.seeding.modes = ("minimizer", "smem")
    model = None
    gm = GraphMambaConfig(d_model=64)
    if with_model:
        model = build_core_model(
            CoreModelConfig(arch="graphmamba", graphmamba=gm)
        ).model
        model.eval()
    return build_pipeline(cfg, model=model), gm


def pangenome_reference(pipeline, ref, gm, node_len=200):
    n_nodes = len(ref) // node_len
    node_seqs = [ref[i * node_len : (i + 1) * node_len] for i in range(n_nodes)]
    node_start = [i * node_len for i in range(n_nodes)]
    graph, edges = make_graph(gm, n_nodes)
    return pipeline.build_reference(
        ref,
        ref_id=0,
        node_seqs=node_seqs,
        node_ref_start=node_start,
        backbone_path=list(range(n_nodes)),
        edge_index=edges,
        graph=graph,
    )


def _primaries(alignments):
    out = []
    for row in alignments:
        rec = row.primary
        out.append(
            None if rec is None else (rec.is_mapped, rec.ref_start, rec.strand)
        )
    return out


def test_dual_aligns_both_references_in_one_pass():
    ref = make_reference()
    reads = make_reads(ref)
    pipe, gm = build("hybrid")
    references = {
        "linear": pipe.build_reference(ref),
        "pangenome": pangenome_reference(pipe, ref, gm),
    }
    result = DualReferenceAligner(pipe).align(reads, references)

    assert isinstance(result, DualAlignmentResult)
    assert set(result.per_reference) == {"linear", "pangenome"}
    for name in ("linear", "pangenome"):
        assert len(result.per_reference[name]) == len(reads)
    assert len(result.integrated) == len(reads)
    assert len(result.integrated_alignments()) == len(reads)
    print(f"dual pass over {result.names}: {len(reads)} reads each, integrated OK")


def test_shared_encoding_matches_separate_alignment():
    """Reusing one read encoding across references must not change the result."""
    ref = make_reference()
    reads = make_reads(ref)
    pipe, gm = build("hybrid")
    assert pipe.uses_neural_scoring
    linear = pipe.build_reference(ref)
    pangenome = pangenome_reference(pipe, ref, gm)

    aligner = DualReferenceAligner(pipe)
    assert aligner._can_share_encoding()  # the encoding-reuse path is active
    dual = aligner.align(reads, {"linear": linear, "pangenome": pangenome})

    sep_linear, _ = pipe.align(reads, linear)
    sep_pan, _ = pipe.align(reads, pangenome)

    assert _primaries(dual.per_reference["linear"]) == _primaries(sep_linear)
    assert _primaries(dual.per_reference["pangenome"]) == _primaries(sep_pan)
    print("shared-encoding dual result is identical to two separate alignments")


def test_integrated_folds_references_into_one_call():
    ref = make_reference()
    reads = make_reads(ref)
    pipe, gm = build("hybrid")
    references = {
        "linear": pipe.build_reference(ref),
        "pangenome": pangenome_reference(pipe, ref, gm),
    }
    result = DualReferenceAligner(pipe).align(reads, references)

    mapped = [c for c in result.integrated if c.primary is not None]
    assert mapped, "expected at least one integrated mapping"
    for concordance in mapped:
        assert concordance.primary.reference in references
        assert 0.0 <= concordance.confidence <= 1.0
    # The winner's own alignments are what the integrated BAM would carry.
    picked = result.integrated_alignments()
    for row, concordance in enumerate(result.integrated):
        if concordance.primary is not None:
            winner = concordance.primary.reference
            assert picked[row] is result.per_reference[winner][row]
    print(f"integrated: {len(mapped)}/{len(reads)} reads placed, winner alignments picked")


def test_dual_works_without_a_model():
    """No neural scoring: still one pass over both references, no encoding reuse."""
    ref = make_reference()
    reads = make_reads(ref)
    pipe, gm = build("hybrid", with_model=False)
    aligner = DualReferenceAligner(pipe)
    assert not aligner._can_share_encoding()
    result = aligner.align(
        reads,
        {"linear": pipe.build_reference(ref), "pangenome": pangenome_reference(pipe, ref, gm)},
    )
    assert set(result.per_reference) == {"linear", "pangenome"}
    assert len(result.integrated) == len(reads)
    print("dual pass runs classically when no model is loaded")


def test_integrate_false_skips_concordance():
    ref = make_reference()
    reads = make_reads(ref, n=4)
    pipe, gm = build("fast", with_model=False)
    result = DualReferenceAligner(pipe).align(
        reads,
        {"linear": pipe.build_reference(ref), "pangenome": pangenome_reference(pipe, ref, gm)},
        integrate=False,
    )
    assert result.integrated is None
    try:
        result.integrated_alignments()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("integrated_alignments should require integrate=True")
    print("integrate=False leaves concordance unset and guarded")


def test_empty_references_rejected():
    pipe, _ = build("fast", with_model=False)
    try:
        DualReferenceAligner(pipe).align(["ACGT"], {})
    except ValueError:
        print("empty reference mapping rejected")
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty references")


def _write_gfa(path, ref, node_len=200):
    """A minimal backbone GFA tiling ``ref`` into RS-tagged segments."""
    n_nodes = len(ref) // node_len
    with open(path, "w") as fh:
        fh.write("H\tVN:Z:1.0\n")
        for i in range(n_nodes):
            seq = ref[i * node_len : (i + 1) * node_len]
            fh.write(f"S\tseg{i}\t{seq}\tLN:i:{len(seq)}\tRS:i:{i * node_len}\n")
        for i in range(n_nodes - 1):
            fh.write(f"L\tseg{i}\t+\tseg{i + 1}\t+\t0M\tzt:Z:ref_link\n")


def test_build_dual_reference_from_files_reads_fasta_once():
    ref = make_reference(length=1200)
    pipe, _ = build("fast", with_model=False)
    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "ref.fa")
        gfa = os.path.join(tmp, "ref.gfa")
        with open(fasta, "w") as fh:
            fh.write(">chr1\n")
            for i in range(0, len(ref), 60):
                fh.write(ref[i : i + 60] + "\n")
        _write_gfa(gfa, ref)

        refs = build_dual_reference_from_files(pipe, fasta, gfa=gfa)

    assert set(refs) == {"linear", "pangenome"}
    linear, pangenome = refs["linear"], refs["pangenome"]
    # One FASTA read, shared frame: identical windowed sequence and offset.
    assert linear.ref_seq == pangenome.ref_seq == ref
    assert linear.offset == pangenome.offset
    assert linear.label == "linear" and not linear.with_graph
    assert pangenome.label == "pangenome" and pangenome.with_graph
    assert pangenome.n_nodes > 0
    # And the built indexes actually align a read through the dual aligner.
    reads = make_reads(ref, n=4)
    result = DualReferenceAligner(pipe).align(
        reads,
        {"linear": linear.reference, "pangenome": pangenome.reference},
    )
    assert len(result.integrated) == len(reads)
    print(f"build_dual_reference_from_files: shared ref_seq len={len(ref)}, "
          f"nodes={pangenome.n_nodes}")


if __name__ == "__main__":
    test_dual_aligns_both_references_in_one_pass()
    test_shared_encoding_matches_separate_alignment()
    test_integrated_folds_references_into_one_call()
    test_dual_works_without_a_model()
    test_integrate_false_skips_concordance()
    test_empty_references_rejected()
    test_build_dual_reference_from_files_reads_fasta_once()
    print("\nALL DUAL-REFERENCE CHECKS PASSED")
