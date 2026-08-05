"""Synthetic alignment dataset for testing the GraphMambaFormer pipeline.

Generation (pure Python) lives in :mod:`synthetic`; the torch-facing
``Dataset`` / collate / (de)serialization live in :mod:`dataset`.
"""

from .alignment_io import (
    alignment_to_read_record,
    alignments_to_records,
    write_alignments,
)
from .dataset import (
    AlignmentDataset,
    build_datasets,
    collate_reads,
    graph_to_encoder_inputs,
    load_dataset,
    save_dataset,
)
from .export import (
    export_dataset,
    read_bam,
    read_gfa,
    sam_to_bam,
    write_fasta,
    write_fastq,
    write_gfa,
    write_labels_json,
    write_sam,
)
from .formats import (
    convert_graph,
    convert_reads,
    is_unaligned_bam,
    read_fastq,
    read_reads,
    validate_modality,
    write_bam,
    write_cram,
    write_gbz,
    write_gfa_graph,
)
from .synthetic import (
    EDGE_TYPES,
    PangenomeGraph,
    ReadRecord,
    Reference,
    Seed,
    SyntheticConfig,
    SyntheticDataset,
    cigar_consumed,
    generate_dataset,
    gc_content,
    minimizers,
    preset,
    reverse_complement,
    verify_read_alignment,
)

__all__ = [
    # generation
    "SyntheticConfig",
    "SyntheticDataset",
    "generate_dataset",
    "preset",
    "Reference",
    "ReadRecord",
    "Seed",
    "PangenomeGraph",
    "EDGE_TYPES",
    # helpers
    "minimizers",
    "reverse_complement",
    "gc_content",
    "cigar_consumed",
    "verify_read_alignment",
    # torch
    "AlignmentDataset",
    "collate_reads",
    "graph_to_encoder_inputs",
    "build_datasets",
    "save_dataset",
    "load_dataset",
    # standard-format export
    "export_dataset",
    "sam_to_bam",
    "read_bam",
    "read_gfa",
    # format I/O layer (FASTQ/BAM/uBAM/SAM/CRAM/GFA in; BAM/CRAM/GFA/GBZ out)
    "read_fastq",
    "read_reads",
    "is_unaligned_bam",
    "validate_modality",
    # pipeline results -> BAM / CRAM
    "write_alignments",
    "alignments_to_records",
    "alignment_to_read_record",
    "write_bam",
    "write_cram",
    "write_gfa_graph",
    "write_gbz",
    "convert_reads",
    "convert_graph",
    "write_fasta",
    "write_fastq",
    "write_sam",
    "write_gfa",
    "write_labels_json",
]
