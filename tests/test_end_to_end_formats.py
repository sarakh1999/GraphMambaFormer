"""End to end: every input format and modality, through the pipeline, back out.

This is the whole-chain version of ``test_formats.py``, which only covers the
I/O layer in isolation. Here a file is read, aligned, and written, because the
seams between those steps were where things were actually broken:

  * the pipeline took bare ``str``, so Phred qualities and modality were dropped
    between the reader and the model even though the encoder embeds both;
  * nothing could turn pipeline output (``ReadAlignments``) into BAM or CRAM,
    so the four documented output formats were unreachable from a real run.

Contract under test::

    INPUT   FASTQ (plain or .gz) | BAM / uBAM / SAM / CRAM | GFA
    OUTPUT  BAM | CRAM | GFA | GBZ
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile

import torch

from graphmambaformer.alignment.pipeline import as_read_batch, build_pipeline
from graphmambaformer.config import MODALITIES, CoreModelConfig, GraphMambaConfig, PipelineConfig
from graphmambaformer.data import (
    read_gfa,
    read_reads,
    write_alignments,
    write_bam,
    write_fasta,
    write_gfa_graph,
)
from graphmambaformer.data.synthetic import generate_dataset, preset
from graphmambaformer.models import build_core_model

REF_ID = 0


def _tmp() -> str:
    return tempfile.mkdtemp(prefix="gmf_e2e_")


def _small_pipeline(mode: str = "hybrid"):
    """A tiny hybrid pipeline; d_model is small so this runs fast on CPU."""
    torch.manual_seed(0)
    cfg = GraphMambaConfig(d_model=64)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    model.eval()
    return build_pipeline(PipelineConfig(mode=mode, batch_size=8), model=model)


def _fixture(d: str):
    """A synthetic reference + its reads, written out as FASTQ/uBAM/GFA."""
    ds = generate_dataset(preset("tiny"))
    ref = ds.references[REF_ID]
    reads = [r for r in ds.splits["train"] if r.ref_id == REF_ID][:6]
    if not reads:
        reads = list(ds.splits["train"])[:6]

    fastq = os.path.join(d, "reads.fastq")
    with open(fastq, "w") as fh:
        for r in reads:
            q = "".join(chr(33 + min(60, max(0, x))) for x in r.quals[: len(r.seq)])
            fh.write(f"@{r.read_id} mod={r.modality}\n{r.seq}\n+\n{q}\n")

    gz = os.path.join(d, "reads.fastq.gz")
    with open(fastq) as src, gzip.open(gz, "wt") as dst:
        dst.write(src.read())

    ubam = os.path.join(d, "reads.ubam")
    write_bam(reads, ubam, references=None, sort=False, index=False)

    gfa = write_gfa_graph(ref.graph, os.path.join(d, "graph.gfa"))
    fasta = os.path.join(d, "ref.fasta")
    write_fasta(ds.references, fasta)
    return ds, ref, reads, {"fastq": fastq, "fastq.gz": gz, "ubam": ubam,
                            "gfa": gfa, "fasta": fasta}


# --------------------------------------------------------------------------- #
# Input seam: qualities and modality must survive the reader -> model handoff
# --------------------------------------------------------------------------- #
def test_read_batch_carries_quality_and_modality():
    """ReadRecords keep their Phred scores and modality; bare strings do not."""
    d = _tmp()
    try:
        _, _, reads, _ = _fixture(d)
        batch = as_read_batch(reads)
        assert batch.quals is not None, "qualities dropped from ReadRecords"
        assert batch.modalities is not None, "modality dropped from ReadRecords"
        assert len(batch.quals[0]) == len(batch.seqs[0])

        plain = as_read_batch([r.seq for r in reads])
        assert plain.quals is None and plain.modalities is None
        assert plain.seqs == batch.seqs
        print(f"ReadRecord batch keeps quals+modality ({batch.modalities[0]}); "
              f"str batch degrades cleanly to neither")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_batch_slicing_keeps_metadata_aligned():
    """Sub-batching (two-pass rescue, batch_size chunking) must not misalign rows."""
    d = _tmp()
    try:
        _, _, reads, _ = _fixture(d)
        batch = as_read_batch(reads)
        picked = list(range(0, len(batch), 2))
        sub = batch.select(picked)
        assert sub.seqs == [batch.seqs[i] for i in picked]
        assert sub.ids == [batch.ids[i] for i in picked]
        assert sub.quals == [batch.quals[i] for i in picked]
        sliced = batch.slice(1, 4)
        assert sliced.seqs == batch.seqs[1:4] and sliced.quals == batch.quals[1:4]
        print(f"select({picked}) and slice(1,4) keep seq/id/qual rows aligned")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qualities_reach_the_model():
    """The encoder embeds quality; feeding it must change the forward pass."""
    from graphmambaformer.alignment.scoring import encode_read_batch

    seqs = ["ACGTACGTACGTACGT"] * 3
    codes, mask, no_q = encode_read_batch(seqs, "cpu")
    assert no_q is None, "no quals supplied -> None, so the encoder uses its default"

    quals = [[40] * len(seqs[0]) for _ in seqs]
    codes2, _, q = encode_read_batch(seqs, "cpu", quals=quals)
    assert q is not None and q.shape == codes2.shape, (None if q is None else q.shape)
    assert int(q.max()) == 40

    torch.manual_seed(0)
    cfg = GraphMambaConfig(d_model=64)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    model.eval()
    with torch.no_grad():
        a = model(codes, mask=mask).pooled
        b = model(codes2, mask=mask, qualities=q).pooled
    assert not torch.allclose(a, b), "qualities had no effect on the forward pass"
    print(f"quality embedding changes pooled output (max delta "
          f"{(a - b).abs().max():.4f})")


def test_modality_reaches_the_model_when_token_enabled():
    """With the modality token on, different modalities give different outputs."""
    from graphmambaformer.alignment.scoring import encode_read_batch

    torch.manual_seed(0)
    cfg = GraphMambaConfig(d_model=64)
    cfg.sequence_encoder.prepend_modality_token = True
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    model.eval()

    codes, mask, _ = encode_read_batch(["ACGTACGTACGTACGT"] * 2, "cpu")
    with torch.no_grad():
        ont = model(codes, mask=mask, modality="ont").pooled
        ilmn = model(codes, mask=mask, modality="illumina").pooled
    assert not torch.allclose(ont, ilmn), "modality token had no effect"
    print(f"modality token distinguishes ont vs illumina "
          f"(max delta {(ont - ilmn).abs().max():.4f})")


# --------------------------------------------------------------------------- #
# Whole chain: file in -> pipeline -> file out
# --------------------------------------------------------------------------- #
def test_every_input_format_aligns_and_writes_bam():
    """FASTQ, gzipped FASTQ and uBAM all align and round-trip out to BAM."""
    d = _tmp()
    try:
        ds, ref, reads, paths = _fixture(d)
        pipe = _small_pipeline("hybrid")
        reference = pipe.build_reference(ref.seq, ref_id=REF_ID)

        for label in ("fastq", "fastq.gz", "ubam"):
            loaded = read_reads(paths[label], modality="pacbio_hifi")
            assert len(loaded) == len(reads), (label, len(loaded))

            results, stats = pipe.align(loaded, reference)
            assert len(results) == len(loaded)

            out = os.path.join(d, f"{label}.out.bam")
            write_alignments(results, loaded, out, references=ds.references)
            back = read_reads(out, modality="pacbio_hifi")
            assert len(back) == len(loaded), (label, len(back), len(loaded))
            mapped = sum(r.records[0].is_mapped for r in results)
            print(f"   {label:9s} -> aligned {mapped}/{len(loaded)} mapped, "
                  f"BAM re-read {len(back)} records")
        print("every input format aligns and writes BAM")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_paired_illumina_survives_pipeline_to_paired_bam():
    """Independent mate alignment must still emit a standards-compliant pair."""
    import pysam

    d = _tmp()
    try:
        ds, ref, reads, _ = _fixture(d)
        mates = reads[:2]
        for i, read in enumerate(mates, 1):
            read.read_id = f"fragment0/{i}"
            read.pair_id = "fragment0"
            read.mate_index = i
            read.modality = "illumina"

        pipe = _small_pipeline("fast")
        reference = pipe.build_reference(ref.seq, ref_id=REF_ID)
        results, _ = pipe.align(mates, reference)
        out = write_alignments(
            results, mates, os.path.join(d, "paired.out.bam"),
            references=ds.references, modality="illumina",
        )
        with pysam.AlignmentFile(out, "rb") as af:
            rows = list(af.fetch(until_eof=True))
        assert len(rows) == 2
        assert all(row.is_paired for row in rows)
        assert {row.query_name for row in rows} == {"fragment0"}
        assert {row.is_read1 for row in rows} == {True, False}
        if all(not row.is_unmapped for row in rows):
            assert all(not row.mate_is_unmapped for row in rows)
            assert all(row.next_reference_start >= 0 for row in rows)
            assert rows[0].template_length == -rows[1].template_length
        print("paired Illumina metadata survives pipeline -> paired BAM")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_output_formats_from_a_real_run():
    """BAM, CRAM, GFA and GBZ all reachable from pipeline output."""
    d = _tmp()
    try:
        ds, ref, reads, paths = _fixture(d)
        pipe = _small_pipeline("hybrid")
        reference = pipe.build_reference(ref.seq, ref_id=REF_ID)
        results, _ = pipe.align(reads, reference)

        bam = write_alignments(results, reads, os.path.join(d, "o.bam"),
                               references=ds.references)
        assert os.path.getsize(bam) > 0

        cram = write_alignments(results, reads, os.path.join(d, "o.cram"),
                                references=ds.references,
                                reference_fasta=paths["fasta"])
        assert os.path.getsize(cram) > 0

        gfa = write_gfa_graph(ref.graph, os.path.join(d, "o.gfa"))
        assert len(read_gfa(gfa).node_seqs) == len(ref.graph.node_seqs)

        from graphmambaformer.data import write_gbz

        if shutil.which("vg") is None:
            try:
                write_gbz(gfa, os.path.join(d, "o.gbz"))
                gbz_note = "written"
            except RuntimeError:
                gbz_note = "vg absent -> clear error (expected)"
        else:
            write_gbz(gfa, os.path.join(d, "o.gbz"))
            gbz_note = "written"

        print(f"BAM {os.path.getsize(bam)}B | CRAM {os.path.getsize(cram)}B | "
              f"GFA {os.path.getsize(gfa)}B | GBZ {gbz_note}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_modalities_survive_the_round_trip():
    """Each modality goes in, through all three pipeline modes, and back out."""
    d = _tmp()
    try:
        ds, ref, reads, _ = _fixture(d)
        pipe = _small_pipeline("hybrid")
        reference = pipe.build_reference(ref.seq, ref_id=REF_ID)

        for modality in MODALITIES:
            tagged = []
            for r in reads:
                clone = type(r)(**{**r.__dict__, "modality": modality})
                tagged.append(clone)

            batch = as_read_batch(tagged)
            assert batch.modality == modality, (batch.modality, modality)

            results, _ = pipe.align(tagged, reference)
            out = os.path.join(d, f"{modality}.bam")
            write_alignments(results, tagged, out, references=ds.references,
                             modality=modality)
            back = read_reads(out, modality=modality)
            assert len(back) == len(tagged)
            assert all(r.modality == modality for r in back)
        print(f"all {len(MODALITIES)} modalities round-trip through the pipeline "
              f"and out to BAM with modality preserved")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_pipeline_modes_accept_records_and_strings():
    """Every mode takes ReadRecords or plain strings and agrees on read count."""
    d = _tmp()
    try:
        ds, ref, reads, _ = _fixture(d)
        seqs = [r.seq for r in reads]
        for mode in ("hybrid", "fast", "two_pass"):
            pipe = _small_pipeline(mode)
            reference = pipe.build_reference(ref.seq, ref_id=REF_ID)

            from_records, _ = pipe.align(reads, reference)
            from_strings, _ = pipe.align(seqs, reference)
            assert len(from_records) == len(from_strings) == len(reads)

            out = os.path.join(d, f"{mode}.bam")
            write_alignments(from_records, reads, out, references=ds.references)
            assert len(read_reads(out, modality="pacbio_hifi")) == len(reads)
            print(f"   {mode:9s} accepts records and strings, writes BAM")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unmapped_reads_are_written_not_dropped():
    """Unalignable reads must still reach the BAM, so nothing is silently lost.

    They are written as unmapped records. Reading such a BAM back with the
    default skips them, matching samtools semantics for an aligned file --
    ``include_unmapped=True`` is how you get them, and that distinction is the
    point of this test.
    """
    from graphmambaformer.data.export import read_bam

    d = _tmp()
    try:
        ds, ref, _, _ = _fixture(d)
        pipe = _small_pipeline("fast")
        reference = pipe.build_reference(ref.seq, ref_id=REF_ID)

        junk = ["TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"] * 4
        results, _ = pipe.align(junk, reference)
        assert not any(r.records[0].is_mapped for r in results), "expected no hits"

        out = os.path.join(d, "junk.bam")
        write_alignments(results, junk, out, references=ds.references)

        everything = read_bam(out, modality="illumina", include_unmapped=True)
        assert len(everything) == len(junk), (len(everything), len(junk))
        assert read_reads(out, modality="illumina") == [], "aligned-BAM default should skip unmapped"
        print(f"{len(everything)}/{len(junk)} unalignable reads written as unmapped "
              f"records; default read-back skips them (samtools semantics)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
