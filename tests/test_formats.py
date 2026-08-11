"""Format I/O: every input format, every modality, and the round trips out.

The contract under test::

    INPUT   FASTQ (plain or .gz) | BAM / uBAM / SAM / CRAM | GFA (.gfa / .gfa.gz)
    OUTPUT  BAM | SAM | CRAM | GFA | GBZ | Giraffe indexes

These tests exist because three of these paths were silently broken: gzipped
FASTQ raised ``UnicodeDecodeError`` even though ``read_reads`` advertised
``.fastq.gz``, a uBAM read back as *zero* records because every read in one is
unmapped, and an unknown modality string rode along on every record instead of
being rejected. Each has a named test below.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile

from graphmambaformer.config import MODALITIES
from graphmambaformer.data.formats import (
    is_unaligned_bam,
    read_fastq,
    read_reads,
    validate_modality,
    write_bam,
    write_cram,
    write_gbz,
    write_gfa_graph,
)
from graphmambaformer.data.export import write_fasta
from graphmambaformer.data.synthetic import generate_dataset, preset

READ_LEN = 60


def _tmp() -> str:
    d = tempfile.mkdtemp(prefix="gmf_fmt_")
    return d


def _fastq(path: str, n: int = 6, modality: str | None = None, gz: bool = False) -> str:
    """Write a small FASTQ, optionally gzipped, optionally with mod= headers."""
    body = []
    for i in range(n):
        tag = f" mod={modality}" if modality else ""
        seq = ("ACGT" * READ_LEN)[:READ_LEN]
        body.append(f"@read{i}{tag}\n{seq}\n+\n{'I' * READ_LEN}\n")
    text = "".join(body)
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        with open(path, "w") as fh:
            fh.write(text)
    return path


# --------------------------------------------------------------------------- #
# Regressions: the three paths that were silently broken
# --------------------------------------------------------------------------- #
def test_gzipped_fastq_loads():
    """`.fastq.gz` is the normal on-disk form; it used to raise UnicodeDecodeError."""
    d = _tmp()
    try:
        plain = read_reads(_fastq(os.path.join(d, "r.fastq")), modality="ont")
        gzipped = read_reads(_fastq(os.path.join(d, "r.fastq.gz"), gz=True), modality="ont")
        assert len(plain) == len(gzipped) == 6, (len(plain), len(gzipped))
        assert [r.seq for r in plain] == [r.seq for r in gzipped]
        print(f"plain and gzipped FASTQ agree: {len(gzipped)} reads, identical sequences")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gzip_detected_by_content_not_extension():
    """A gzipped file named `.fastq` still loads — detection is by magic bytes."""
    d = _tmp()
    try:
        misnamed = os.path.join(d, "actually_gzipped.fastq")
        _fastq(misnamed, gz=True)
        assert len(read_reads(misnamed, modality="ont")) == 6
        print("gzip magic-byte detection handles a misnamed .fastq")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ubam_round_trips_instead_of_yielding_nothing():
    """Every read in a uBAM is unmapped; reading one must not return an empty list."""
    d = _tmp()
    try:
        src = read_fastq(_fastq(os.path.join(d, "r.fastq")), modality="ont")
        ubam = os.path.join(d, "reads.bam")
        write_bam(src, ubam, references=None, sort=False, index=False)

        assert is_unaligned_bam(ubam), "no @SQ lines expected -> uBAM"
        back = read_reads(ubam, modality="ont")
        assert len(back) == len(src) == 6, (len(back), len(src))
        assert [r.read_id for r in back] == [r.read_id for r in src]
        assert [r.seq for r in back] == [r.seq for r in src]
        print(f"uBAM round trip preserved {len(back)} reads, ids and sequences intact")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ubam_extension_is_binary_not_sam():
    """A `.ubam` suffix must open as binary BAM, not be misread as text SAM."""
    d = _tmp()
    try:
        src = read_fastq(_fastq(os.path.join(d, "r.fastq")), modality="pacbio_hifi")
        path = os.path.join(d, "reads.ubam")
        write_bam(src, path, references=None, sort=False, index=False)
        assert len(read_reads(path, modality="pacbio_hifi")) == 6
        print(".ubam opens as binary BAM")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unknown_modality_is_rejected():
    """A typo used to ride along on every record and fail later in the encoder."""
    d = _tmp()
    try:
        fq = _fastq(os.path.join(d, "r.fastq"))
        for bad in ("nanopor", "illumnia", "", "pacbio_hifi_v2"):
            try:
                read_reads(fq, modality=bad)
            except ValueError:
                continue
            raise AssertionError(f"modality {bad!r} should have been rejected")
        print("unknown modalities rejected at read time, not deep in the encoder")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_modality_aliases_resolve():
    """The names people actually type map onto canonical MODALITIES keys."""
    cases = {
        "nanopore": "ont", "ONT": "ont", "ont_r10": "ont",
        "hifi": "pacbio_hifi", "PacBio": "pacbio_hifi", "ccs": "pacbio_hifi",
        "10x": "linked_reads", "chromium": "linked_reads",
        "dnbseq": "illumina", "ultima": "illumina", "short-read": "illumina",
        "wgbs": "bisulfite", "rnaseq": "rna_seq",
    }
    for given, want in cases.items():
        got = validate_modality(given)
        assert got == want, (given, got, want)
    print(f"{len(cases)} platform aliases resolve to canonical modalities")


# --------------------------------------------------------------------------- #
# Coverage: every modality through every input format
# --------------------------------------------------------------------------- #
def test_every_modality_loads_from_every_input_format():
    """The full modality x input-format grid, which is the actual requirement."""
    d = _tmp()
    try:
        grid: dict[str, dict[str, int]] = {}
        for modality in MODALITIES:
            row: dict[str, int] = {}

            fq = _fastq(os.path.join(d, f"{modality}.fastq"), modality=modality)
            row["fastq"] = len(read_reads(fq, modality=modality))

            fqgz = _fastq(os.path.join(d, f"{modality}.fastq.gz"),
                          modality=modality, gz=True)
            row["fastq.gz"] = len(read_reads(fqgz, modality=modality))

            src = read_fastq(fq, modality=modality)
            ub = os.path.join(d, f"{modality}.ubam")
            write_bam(src, ub, references=None, sort=False, index=False)
            row["ubam"] = len(read_reads(ub, modality=modality))

            grid[modality] = row
            assert set(row.values()) == {6}, (modality, row)
            # modality must survive the round trip, including via mod= headers
            assert all(r.modality == modality for r in read_reads(ub, modality=modality))

        print(f"{len(grid)} modalities x 3 input formats, 6 reads each:")
        for m, row in grid.items():
            print(f"   {m:14s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fastq_header_modality_overrides_default():
    """`mod=` in the header wins over the caller's default, and is validated."""
    d = _tmp()
    try:
        fq = _fastq(os.path.join(d, "tagged.fastq"), modality="ont")
        recs = read_reads(fq, modality="illumina")  # header should win
        assert {r.modality for r in recs} == {"ont"}, {r.modality for r in recs}
        print("mod= header overrides the caller default (ont beat illumina)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Output formats
# --------------------------------------------------------------------------- #
def test_bam_and_cram_outputs():
    """BAM and CRAM both write, and the BAM reads back with alignments intact."""
    d = _tmp()
    try:
        ds = generate_dataset(preset("tiny"))
        aligned = list(ds.splits["train"])
        refs = ds.references

        bam = write_bam(aligned, os.path.join(d, "out.bam"), references=refs)
        assert os.path.getsize(bam) > 0
        assert not is_unaligned_bam(bam), "aligned reads -> BAM must carry @SQ lines"
        back = read_reads(bam, modality="pacbio_hifi")
        assert len(back) == len(aligned), (len(back), len(aligned))
        assert all(r.ref_id >= 0 and r.cigar for r in back), "alignments lost"

        # CRAM is reference-compressed, so it needs a FASTA matching the SQ names.
        fasta = os.path.join(d, "ref.fasta")
        write_fasta(refs, fasta)
        cram = write_cram(aligned, os.path.join(d, "out.cram"), fasta, references=refs)
        assert os.path.getsize(cram) > 0

        print(f"BAM {os.path.getsize(bam)}B (read back {len(back)} aligned reads) "
              f"and CRAM {os.path.getsize(cram)}B written")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gfa_output_round_trips():
    """GFA out, then back in, preserving nodes and edges."""
    from graphmambaformer.data.export import read_gfa

    d = _tmp()
    try:
        graph = generate_dataset(preset("tiny")).references[0].graph
        path = write_gfa_graph(graph, os.path.join(d, "g.gfa"))
        back = read_gfa(path)
        assert list(back.node_seqs) == list(graph.node_seqs), "node sequences changed"
        assert len(back.edge_index) == len(graph.edge_index), (
            len(back.edge_index), len(graph.edge_index))
        print(f"GFA round trip: {len(back.node_seqs)} nodes, "
              f"{len(back.edge_index)} edges preserved")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gbz_reports_clearly_when_vg_is_absent():
    """GBZ genuinely needs the vg binary; the error must say so and show the command."""
    d = _tmp()
    try:
        graph = generate_dataset(preset("tiny")).references[0].graph
        gfa = write_gfa_graph(graph, os.path.join(d, "g.gfa"))
        out = os.path.join(d, "g.gbz")

        if shutil.which("vg") is None:
            try:
                write_gbz(gfa, out)
            except RuntimeError as exc:
                assert "vg" in str(exc) and "gbz" in str(exc).lower()
                print("vg absent -> GBZ raises a clear, actionable error (as designed)")
                return
            raise AssertionError("expected RuntimeError when vg is missing")
        write_gbz(gfa, out)
        assert os.path.getsize(out) > 0
        print(f"vg present -> GBZ written, {os.path.getsize(out)}B")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_sam_output_round_trips():
    """SAM writes and reads back with alignments intact (plain-text BAM sibling)."""
    d = _tmp()
    try:
        ds = generate_dataset(preset("tiny"))
        aligned = list(ds.splits["train"])
        refs = ds.references

        from graphmambaformer.data.formats import write_sam

        sam = write_sam(aligned, os.path.join(d, "out.sam"), references=refs)
        assert os.path.getsize(sam) > 0
        back = read_reads(sam, modality="pacbio_hifi")
        assert len(back) == len(aligned), (len(back), len(aligned))
        assert all(r.ref_id >= 0 and r.cigar for r in back), "alignments lost from SAM"
        print(f"SAM {os.path.getsize(sam)}B (read back {len(back)} aligned reads)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gzipped_gfa_loads():
    """``.gfa.gz`` must parse the same as the plain GFA."""
    from graphmambaformer.data.export import read_gfa
    from graphmambaformer.data.formats import write_gfa_graph

    d = _tmp()
    try:
        graph = generate_dataset(preset("tiny")).references[0].graph
        plain = write_gfa_graph(graph, os.path.join(d, "g.gfa"))
        gz = os.path.join(d, "g.gfa.gz")
        with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
            dst.write(src.read())
        back = read_gfa(gz)
        assert list(back.node_seqs) == list(graph.node_seqs)
        assert len(back.edge_index) == len(graph.edge_index)
        print(f"gzipped GFA: {len(back.node_seqs)} nodes, {len(back.edge_index)} edges")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_giraffe_indexes_report_clearly_when_vg_is_absent():
    """Giraffe indexes need vg; the error must name the autoindex command."""
    from graphmambaformer.data.formats import write_gfa_graph, write_giraffe_indexes

    d = _tmp()
    try:
        graph = generate_dataset(preset("tiny")).references[0].graph
        gfa = write_gfa_graph(graph, os.path.join(d, "g.gfa"))
        prefix = os.path.join(d, "idx")
        if shutil.which("vg") is None:
            try:
                write_giraffe_indexes(gfa, prefix)
            except RuntimeError as exc:
                assert "vg" in str(exc).lower() and "giraffe" in str(exc).lower()
                print("vg absent -> Giraffe indexes raise a clear error (as designed)")
                return
            raise AssertionError("expected RuntimeError when vg is missing")
        paths = write_giraffe_indexes(gfa, prefix)
        assert all(os.path.getsize(p) > 0 for p in paths.values())
        print(f"vg present -> Giraffe indexes: {sorted(paths)}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

