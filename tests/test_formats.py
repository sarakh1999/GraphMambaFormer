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
from types import SimpleNamespace

from graphmambaformer.config import MODALITIES
from graphmambaformer.data.formats import (
    is_unaligned_bam,
    read_fastq,
    read_paired_fastq,
    read_reads,
    validate_modality,
    write_bam,
    write_cram,
    write_gbz,
    write_gfa_graph,
)
from graphmambaformer.data.export import write_fasta
from graphmambaformer.data.real_data import load_real_reads
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


def test_ubam_preserves_mm_ml_methylation_tags():
    """ONT/PacBio uBAM MM/ML base-mod tags must survive into ``rec.methylation``.

    This is the whole reason a uBAM is preferred over FASTQ for long reads, and
    the reader used to silently drop it. We hand-build a uBAM with pysam so the
    MM/ML tags are real, then assert the calls come back in read-base frame with
    the 0.5 probability threshold applied (qual 200 -> 1, qual 10 -> 0).
    """
    import array

    import pysam

    d = _tmp()
    try:
        seq = "AACCGGTT"  # C bases at read positions 2 and 3
        header = pysam.AlignmentHeader.from_dict(
            {"HD": {"VN": "1.6", "SO": "unsorted"}}  # no @SQ -> uBAM
        )
        a = pysam.AlignedSegment(header)
        a.query_name = "modread0"
        a.flag = 0x4  # unmapped
        a.query_sequence = seq
        a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
        # 5mC on both C bases; ML gives per-call probability (256*p, 0-255).
        a.set_tag("MM", "C+m,0,0;", "Z")
        a.set_tag("ML", array.array("B", [200, 10]))

        path = os.path.join(d, "mods.ubam")
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            out.write(a)

        assert is_unaligned_bam(path), "no @SQ lines expected -> uBAM"
        (rec,) = read_reads(path, modality="ont")
        assert rec.seq == seq
        assert rec.methylation == [(2, 1), (3, 0)], rec.methylation
        print(f"uBAM MM/ML preserved: methylation calls {rec.methylation}")
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


def test_illumina_r1_r2_are_paired_and_long_reads_stay_single():
    """Two Illumina FASTQs pair automatically; one ONT FASTQ remains single."""
    d = _tmp()
    try:
        r1 = _fastq(os.path.join(d, "sample_R1.fastq"), n=3)
        r2 = _fastq(os.path.join(d, "sample_R2.fastq"), n=3)
        paired = read_paired_fastq(r1, r2, modality="illumina")
        assert len(paired) == 6
        for i in range(0, len(paired), 2):
            left, right = paired[i], paired[i + 1]
            assert left.pair_id == right.pair_id == f"read{i // 2}"
            assert (left.mate_index, right.mate_index) == (1, 2)
            assert left.read_id.endswith("/1") and right.read_id.endswith("/2")

        automatic, _ = load_real_reads(
            reads=[r1, r2], modality="illumina", layout="auto"
        )
        assert [(r.pair_id, r.mate_index) for r in automatic] == [
            (r.pair_id, r.mate_index) for r in paired
        ]

        long_reads, _ = load_real_reads(
            reads=[r1], modality="ont", layout="auto"
        )
        assert len(long_reads) == 3
        assert all(r.pair_id is None and r.mate_index == 0 for r in long_reads)
        print("Illumina R1/R2 paired; single-file ONT remains single-end")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_paired_bam_flags_mate_coordinates_and_tlen_round_trip():
    """Paired metadata survives BAM output/input while SE behavior is unchanged."""
    import pysam

    d = _tmp()
    try:
        r1_path = _fastq(os.path.join(d, "R1.fastq"), n=1)
        r2_path = _fastq(os.path.join(d, "R2.fastq"), n=1)
        r1, r2 = read_paired_fastq(r1_path, r2_path, modality="illumina")
        for rec, start, strand in ((r1, 100, 1), (r2, 220, -1)):
            rec.ref_id = 0
            rec.ref_start = start
            rec.ref_end = start + len(rec.seq)
            rec.strand = strand
            rec.cigar = [("M", len(rec.seq))]
            rec.mapq = 60
        span = r2.ref_end - r1.ref_start
        r1.mate_ref_id = r2.mate_ref_id = 0
        r1.mate_ref_start, r2.mate_ref_start = r2.ref_start, r1.ref_start
        r1.mate_strand, r2.mate_strand = r2.strand, r1.strand
        r1.template_length, r2.template_length = span, -span
        r1.proper_pair = r2.proper_pair = True

        path = os.path.join(d, "paired.bam")
        refs = {0: SimpleNamespace(seq="A" * 1000)}
        write_bam([r1, r2], path, references=refs)
        with pysam.AlignmentFile(path, "rb") as af:
            rows = list(af.fetch(until_eof=True))
        assert len(rows) == 2
        by_end = {1 if a.is_read1 else 2: a for a in rows}
        assert all(a.is_paired and a.is_proper_pair for a in rows)
        assert by_end[1].query_name == by_end[2].query_name == "read0"
        assert by_end[1].next_reference_start == 220
        assert by_end[2].next_reference_start == 100
        assert by_end[1].template_length == span
        assert by_end[2].template_length == -span

        back = read_reads(path, modality="illumina")
        assert {r.mate_index for r in back} == {1, 2}
        assert {r.pair_id for r in back} == {"read0"}
        assert {r.read_id for r in back} == {"read0/1", "read0/2"}
        print("paired BAM flags, mate positions and TLEN round-trip")
    finally:
        shutil.rmtree(d, ignore_errors=True)


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


def test_combined_short_and_long_bam():
    """One BAM can hold Illumina + HiFi with distinct @RG / XM tags."""
    import pysam
    from graphmambaformer.alignment.types import AlignmentRecord
    from graphmambaformer.data.alignment_io import write_alignments_split
    from graphmambaformer.data.synthetic import ReadRecord

    d = _tmp()
    try:
        refs = {0: SimpleNamespace(seq="ACGT" * 40)}
        short = ReadRecord(
            read_id="s1/1", ref_id=0, modality="illumina",
            seq="ACGTACGTAC", quals=[30] * 10,
            ref_start=0, ref_end=10, strand=1,
            cigar=[("=", 10)], ref_positions=list(range(10)), mapq=60,
            pair_id="s1", mate_index=1,
        )
        long = ReadRecord(
            read_id="l1", ref_id=0, modality="pacbio_hifi",
            seq="ACGT" * 20, quals=[40] * 80,
            ref_start=4, ref_end=84, strand=1,
            cigar=[("=", 80)], ref_positions=list(range(4, 84)), mapq=60,
        )
        combined = os.path.join(d, "both.sorted.bam")
        write_bam([short, long], combined, references=refs,
                  contig_names={0: "chr21"})

        with pysam.AlignmentFile(combined, "rb") as bam:
            rgs = {rg["ID"] for rg in bam.header.get("RG", [])}
            assert rgs == {"illumina", "pacbio_hifi"}, rgs
            tags = {(a.get_tag("RG"), a.get_tag("XM")) for a in bam.fetch(until_eof=True)}
            assert tags == {("illumina", "illumina"), ("pacbio_hifi", "pacbio_hifi")}, tags

        def _aln(rec: ReadRecord):
            return SimpleNamespace(
                read_id=rec.read_id,
                records=[AlignmentRecord(
                    read_id=rec.read_id, read_len=len(rec.seq),
                    ref_id=rec.ref_id, ref_start=rec.ref_start,
                    ref_end=rec.ref_end, strand=rec.strand, mapq=rec.mapq,
                    cigar=list(rec.cigar), is_mapped=True, is_primary=True,
                )],
            )

        paths = write_alignments_split(
            [_aln(short), _aln(long)], [short, long],
            os.path.join(d, "split.sorted.bam"),
            references=refs, contig_names={0: "chr21"},
        )
        assert set(paths) == {"illumina", "pacbio_hifi"}, paths
        assert all(os.path.getsize(p) > 0 for p in paths.values())
        print("combined short+long BAM has @RG per modality; separate BAMs also write")
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


def test_hprc_three_modality_inputs_combined_and_separate():
    """HiFi FASTQ.gz, Illumina CRAM, and ONT BAM all load; combined+separate BAM."""
    import pysam
    from graphmambaformer.data.formats import write_bam, write_cram
    from graphmambaformer.data.real_data import (
        discover_hprc_reads,
        expand_read_inputs,
        load_real_reads,
    )
    from graphmambaformer.data.synthetic import ReadRecord

    d = _tmp()
    try:
        # Mimic data/hprc/reads/<SAMPLE>/{hifi,illumina,ont}/
        sample = "HG00438"
        root = os.path.join(d, "reads")
        for sub in ("hifi", "illumina", "ont"):
            os.makedirs(os.path.join(root, sample, sub))

        hifi = _fastq(
            os.path.join(root, sample, "hifi", f"{sample}.run1.fastq.gz"),
            n=4, gz=True,
        )
        # Second HiFi run file (multi-file folders are normal).
        _fastq(
            os.path.join(root, sample, "hifi", f"{sample}.run2.fastq.gz"),
            n=2, gz=True,
        )

        refs = {0: SimpleNamespace(seq="ACGT" * 40)}
        fasta = os.path.join(d, "ref.fa")
        with open(fasta, "w") as fh:
            fh.write(f">chr21\n{refs[0].seq}\n")
        # CRAM writers need an fai next to the FASTA.
        import pysam as _pysam
        _pysam.faidx(fasta)

        illumina_recs = [
            ReadRecord(
                read_id=f"ill{i}/1", ref_id=0, modality="illumina",
                seq="ACGTACGTAC", quals=[30] * 10,
                ref_start=0, ref_end=10, strand=1,
                cigar=[("=", 10)], ref_positions=list(range(10)), mapq=60,
                pair_id=f"ill{i}", mate_index=1,
            )
            for i in range(3)
        ]
        cram = write_cram(
            illumina_recs,
            os.path.join(root, sample, "illumina", f"{sample}.final.cram"),
            fasta,
            references=refs,
            contig_names={0: "chr21"},
        )

        ont_recs = [
            ReadRecord(
                read_id=f"ont{i}", ref_id=0, modality="ont",
                seq="ACGT" * 20, quals=[20] * 80,
                ref_start=0, ref_end=80, strand=1,
                cigar=[("=", 80)], ref_positions=list(range(80)), mapq=40,
            )
            for i in range(2)
        ]
        ont_bam = write_bam(
            ont_recs,
            os.path.join(root, sample, "ont", f"{sample}.dorado.bam"),
            references=refs,
            contig_names={0: "chr21"},
        )

        # Discovery finds every modality's files.
        hifi_paths, hifi_mod = discover_hprc_reads(sample, "hifi", reads_root=root)
        ill_paths, ill_mod = discover_hprc_reads(sample, "illumina", reads_root=root)
        ont_paths, ont_mod = discover_hprc_reads(sample, "ont", reads_root=root)
        assert hifi_mod == "pacbio_hifi" and len(hifi_paths) == 2
        assert ill_mod == "illumina" and ill_paths == [cram]
        assert ont_mod == "ont" and ont_paths == [ont_bam]
        assert len(expand_read_inputs([os.path.join(root, sample, "hifi")])) == 2

        # Load each modality (CRAM/BAM as remappable sequences).
        hifi_reads, _ = load_real_reads(
            reads=hifi_paths, modality="pacbio_hifi", as_sequences=True
        )
        ill_reads, _ = load_real_reads(
            reads=ill_paths, modality="illumina", as_sequences=True,
            reference_fasta=fasta,
        )
        ont_reads, _ = load_real_reads(
            reads=ont_paths, modality="ont", as_sequences=True,
            reference_fasta=fasta,
        )
        assert len(hifi_reads) == 6
        assert all(r.modality == "pacbio_hifi" and r.cigar == [] for r in hifi_reads)
        assert len(ill_reads) == 3
        assert all(r.modality == "illumina" and r.ref_id < 0 and not r.cigar for r in ill_reads)
        assert len(ont_reads) == 2
        assert all(r.modality == "ont" and r.ref_id < 0 for r in ont_reads)

        # Combined BAM keeps all three @RG tags; separate writes three files.
        combined = os.path.join(d, "all.sorted.bam")
        mixed = hifi_reads + ill_reads + ont_reads
        write_bam(mixed, combined, references=refs, contig_names={0: "chr21"})
        with pysam.AlignmentFile(combined, "rb") as bam:
            rgs = {rg["ID"] for rg in bam.header.get("RG", [])}
        assert rgs == {"pacbio_hifi", "illumina", "ont"}, rgs

        from graphmambaformer.data.alignment_io import write_alignments_split
        from graphmambaformer.alignment.types import AlignmentRecord

        def _aln(rec: ReadRecord):
            return SimpleNamespace(
                read_id=rec.read_id,
                records=[AlignmentRecord(
                    read_id=rec.read_id, read_len=len(rec.seq),
                    ref_id=0, ref_start=0, ref_end=min(10, len(rec.seq)),
                    strand=1, mapq=20, cigar=[("=", min(10, len(rec.seq)))],
                    is_mapped=True, is_primary=True,
                )],
            )

        # Give temporary mapped fields so the writer emits aligned records.
        for rec in mixed:
            rec.ref_id = 0
            rec.ref_start = 0
            rec.ref_end = min(10, len(rec.seq))
            rec.cigar = [("=", min(10, len(rec.seq)))]
            rec.ref_positions = list(range(rec.ref_end))
            rec.mapq = 20

        paths = write_alignments_split(
            [_aln(r) for r in mixed], mixed,
            os.path.join(d, "split.sorted.bam"),
            references=refs, contig_names={0: "chr21"},
        )
        assert set(paths) == {"pacbio_hifi", "illumina", "ont"}, paths
        print(
            "HPRC modalities OK: "
            f"hifi FASTQ.gz={len(hifi_reads)}, "
            f"illumina CRAM sequences={len(ill_reads)}, "
            f"ont BAM sequences={len(ont_reads)}; "
            "combined @RG + separate BAMs"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)

