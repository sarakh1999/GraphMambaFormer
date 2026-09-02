"""Write what the aligner produced to the standard output formats.

The pipeline emits :class:`~graphmambaformer.alignment.types.ReadAlignments`,
while the writers in :mod:`.formats` speak :class:`ReadRecord`. This module is
the bridge, so a run can go from any input format to any output format::

    reads   = read_reads("sample.fastq.gz", modality="ont")
    results, _ = pipeline.align(reads, reference)
    write_alignments(results, reads, "out.bam", references=refs)

An ``AlignmentRecord`` deliberately carries no sequence — the bases live on the
input read — so both are needed to emit a record. Pairing is by ``read_id``
when available and by position otherwise, which keeps it correct even if the
pipeline reorders or drops reads.

Secondary and supplementary alignments are written alongside the primary when
``include_secondary`` is set, flagged per the SAM spec.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..progress import progress
from .formats import validate_modality, write_bam, write_cram, write_sam
from .synthetic import ReadRecord

__all__ = [
    "alignment_to_read_record",
    "alignments_to_records",
    "write_alignments",
    "write_alignments_split",
]


def _reference_positions(cigar: Sequence[tuple[str, int]], start: int,
                         read_len: int) -> list[int]:
    """Per-query-base reference coordinate, ``-1`` for insertions and clips."""
    positions = [-1] * read_len
    qpos, rpos = 0, start
    for op, n in cigar:
        if op in ("M", "=", "X"):
            for _ in range(n):
                if qpos < read_len:
                    positions[qpos] = rpos
                qpos += 1
                rpos += 1
        elif op in ("I", "S"):
            qpos += n
        elif op in ("D", "N"):
            rpos += n
        # H and P consume neither
    return positions


def alignment_to_read_record(
    record,
    seq: str,
    quals: Optional[Sequence[int]] = None,
    modality: str = "pacbio_hifi",
    source=None,
) -> ReadRecord:
    """Turn one :class:`AlignmentRecord` plus its read into a :class:`ReadRecord`.

    Unmapped records become valid unmapped entries rather than being dropped, so
    the read count out matches the read count in.
    """
    modality = validate_modality(modality)
    quals = list(quals) if quals else [0] * len(seq)
    if len(quals) < len(seq):
        quals = quals + [0] * (len(seq) - len(quals))

    pair_fields = {
        name: getattr(source, name, default)
        for name, default in (
            ("pair_id", None), ("mate_index", 0), ("mate_ref_id", -1),
            ("mate_ref_start", -1), ("mate_strand", 1),
            ("template_length", 0), ("proper_pair", False),
        )
    }

    if not record.is_mapped:
        return ReadRecord(
            read_id=record.read_id, ref_id=-1, modality=modality,
            seq=seq, quals=quals[: len(seq)], ref_start=0, ref_end=0, strand=1,
            cigar=[], ref_positions=[-1] * len(seq), mapq=0, edge_case="unmapped",
            **pair_fields,
        )

    cigar = list(record.cigar)
    return ReadRecord(
        read_id=record.read_id,
        ref_id=int(record.ref_id),
        modality=modality,
        seq=seq,
        quals=quals[: len(seq)],
        ref_start=int(record.ref_start),
        ref_end=int(record.ref_end),
        strand=int(record.strand),
        cigar=cigar,
        ref_positions=_reference_positions(cigar, int(record.ref_start), len(seq)),
        mapq=int(record.mapq),
        supplementary={"is_primary": bool(record.is_primary)}
        if not record.is_primary
        else None,
        **pair_fields,
    )


def _synchronize_pairs(records: Sequence[ReadRecord]) -> None:
    """Fill mate coordinates/TLEN from independently aligned R1/R2 records."""
    pairs: dict[str, dict[int, ReadRecord]] = {}
    for rec in records:
        if rec.pair_id and rec.mate_index in (1, 2) and rec.supplementary is None:
            pairs.setdefault(rec.pair_id, {})[rec.mate_index] = rec

    for mates in pairs.values():
        if 1 not in mates or 2 not in mates:
            continue
        r1, r2 = mates[1], mates[2]
        mapped1 = r1.ref_id >= 0 and bool(r1.cigar)
        mapped2 = r2.ref_id >= 0 and bool(r2.cigar)
        for rec, mate, mate_mapped in ((r1, r2, mapped2), (r2, r1, mapped1)):
            rec.mate_ref_id = mate.ref_id if mate_mapped else -1
            rec.mate_ref_start = mate.ref_start if mate_mapped else -1
            rec.mate_strand = mate.strand
        same_ref = mapped1 and mapped2 and r1.ref_id == r2.ref_id
        # A conservative proper-pair definition; no library insert-size model
        # is assumed, so orientation and same-contig mapping are required.
        proper = same_ref and r1.strand != r2.strand
        r1.proper_pair = r2.proper_pair = proper
        if same_ref:
            left = min(r1.ref_start, r2.ref_start)
            right = max(r1.ref_end, r2.ref_end)
            span = max(0, right - left)
            r1.template_length = span if r1.ref_start <= r2.ref_start else -span
            r2.template_length = -r1.template_length


def _read_index(reads: Sequence) -> tuple[dict[str, object], list[object]]:
    """Index the source reads by id where possible, keeping positional order."""
    by_id: dict[str, object] = {}
    for read in reads:
        rid = getattr(read, "read_id", None)
        if rid is not None and rid not in by_id:
            by_id[rid] = read
    return by_id, list(reads)


def alignments_to_records(
    results: Iterable,
    reads: Sequence,
    modality: str = "pacbio_hifi",
    include_secondary: bool = False,
) -> list[ReadRecord]:
    """Pair pipeline results with their source reads into writable records.

    ``reads`` may be the ``ReadRecord`` objects from a format reader (whose
    qualities and modality are then preserved) or plain sequence strings.
    """
    by_id, ordered = _read_index(reads)
    out: list[ReadRecord] = []

    for row, alignments in enumerate(
        progress(results, desc="alignments to records", unit="read", leave=False)
    ):
        source = by_id.get(alignments.read_id)
        if source is None:
            source = ordered[row] if row < len(ordered) else ""
        if isinstance(source, str):
            seq, quals, mod = source, None, modality
        else:
            seq = getattr(source, "seq", "")
            quals = getattr(source, "quals", None)
            mod = getattr(source, "modality", None) or modality

        records = alignments.records or []
        if not include_secondary:
            primary = next((r for r in records if r.is_primary), None)
            records = [primary] if primary is not None else records[:1]

        for record in records:
            if record is None:
                continue
            out.append(alignment_to_read_record(
                record, seq, quals, mod,
                source=None if isinstance(source, str) else source,
            ))
    _synchronize_pairs(out)
    return out


def write_alignments(
    results: Iterable,
    reads: Sequence,
    path: str,
    references: Optional[dict] = None,
    reference_fasta: Optional[str] = None,
    modality: str = "pacbio_hifi",
    include_secondary: bool = False,
    contig_names: Optional[dict] = None,
) -> str:
    """Write pipeline results to BAM, SAM, or CRAM by output extension.

    CRAM is reference-compressed and therefore needs ``reference_fasta``.

    ``contig_names`` maps ``ref_id`` to the ``@SQ`` name to write, so aligning
    against a real reference emits ``chr21`` (matching the caller's FASTA)
    instead of the synthetic default ``ref0``.

    Mixed short + long records are fine in one BAM: each modality becomes its
    own ``@RG`` (see :func:`write_bam`). Prefer :func:`write_alignments_split`
    only when a downstream caller needs homogeneous BAMs.
    """
    records = alignments_to_records(
        results, reads, modality=modality, include_secondary=include_secondary
    )
    return _write_records(
        records, path,
        references=references,
        reference_fasta=reference_fasta,
        contig_names=contig_names,
    )


def _write_records(
    records: Sequence,
    path: str,
    references: Optional[dict] = None,
    reference_fasta: Optional[str] = None,
    contig_names: Optional[dict] = None,
) -> str:
    low = path.lower()
    if low.endswith(".cram"):
        if not reference_fasta:
            raise ValueError("CRAM output requires reference_fasta=")
        return write_cram(
            records, path, reference_fasta, references=references,
            contig_names=contig_names,
        )
    if low.endswith(".sam"):
        return write_sam(
            records, path, references=references, contig_names=contig_names
        )
    if low.endswith((".bam", ".ubam")):
        return write_bam(records, path, references=references,
                         contig_names=contig_names)
    raise ValueError(
        f"unsupported alignment output format: {path}. "
        "Expected .bam / .ubam / .sam / .cram."
    )


def write_alignments_split(
    results: Iterable,
    reads: Sequence,
    path_stem: str,
    references: Optional[dict] = None,
    reference_fasta: Optional[str] = None,
    modality: str = "pacbio_hifi",
    include_secondary: bool = False,
    contig_names: Optional[dict] = None,
    ext: str = ".bam",
) -> dict[str, str]:
    """Write one BAM/SAM/CRAM per modality (fallback when a combined file is unwanted).

    ``path_stem`` may be ``out.sorted`` or ``out.sorted.bam``; the modality is
    inserted before the final extension, e.g. ``out.sorted.illumina.bam``.
    Returns ``{modality: path}``.
    """
    records = alignments_to_records(
        results, reads, modality=modality, include_secondary=include_secondary
    )
    if not ext.startswith("."):
        ext = "." + ext
    stem = path_stem
    low = stem.lower()
    for suffix in (".bam", ".ubam", ".sam", ".cram"):
        if low.endswith(suffix):
            stem = stem[: -len(suffix)]
            ext = suffix
            break

    by_mod: dict[str, list] = {}
    for rec in records:
        by_mod.setdefault(rec.modality or modality, []).append(rec)

    written: dict[str, str] = {}
    for mod, group in by_mod.items():
        out_path = f"{stem}.{mod}{ext}"
        written[mod] = _write_records(
            group, out_path,
            references=references,
            reference_fasta=reference_fasta,
            contig_names=contig_names,
        )
    return written
