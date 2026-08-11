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

from .formats import validate_modality, write_bam, write_cram, write_sam
from .synthetic import ReadRecord

__all__ = [
    "alignment_to_read_record",
    "alignments_to_records",
    "write_alignments",
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
) -> ReadRecord:
    """Turn one :class:`AlignmentRecord` plus its read into a :class:`ReadRecord`.

    Unmapped records become valid unmapped entries rather than being dropped, so
    the read count out matches the read count in.
    """
    modality = validate_modality(modality)
    quals = list(quals) if quals else [0] * len(seq)
    if len(quals) < len(seq):
        quals = quals + [0] * (len(seq) - len(quals))

    if not record.is_mapped:
        return ReadRecord(
            read_id=record.read_id, ref_id=-1, modality=modality,
            seq=seq, quals=quals[: len(seq)], ref_start=0, ref_end=0, strand=1,
            cigar=[], ref_positions=[-1] * len(seq), mapq=0, edge_case="unmapped",
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
    )


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

    for row, alignments in enumerate(results):
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
            out.append(alignment_to_read_record(record, seq, quals, mod))
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
    """
    records = alignments_to_records(
        results, reads, modality=modality, include_secondary=include_secondary
    )
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
