"""Synthetic alignment dataset for testing the GraphMambaFormer pipeline.

Generation (pure Python) lives in :mod:`synthetic`; the torch-facing
``Dataset`` / collate / (de)serialization live in :mod:`dataset`.
"""

from .alignment_io import (
    alignment_to_read_record,
    alignments_to_records,
    write_alignments,
    write_alignments_split,
)
from .dataset import (
    AlignmentDataset,
    build_datasets,
    collate_reads,
    graph_to_encoder_inputs,
    load_dataset,
    save_dataset,
)
from .reference_build import (
    build_reference_from_synthetic,
    pangenome_graph_batch,
)
from .real_data import (
    RealReference,
    build_batches,
    build_reference_from_files,
    load_real_reads,
    parse_region,
    read_fasta_contig,
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
    write_sam as write_truth_sam,
)
from .formats import (
    convert_graph,
    convert_reads,
    is_unaligned_bam,
    read_fastq,
    read_paired_fastq,
    read_reads,
    validate_modality,
    write_bam,
    write_cram,
    write_gbz,
    write_gfa_graph,
    write_giraffe_indexes,
    write_sam,
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
    "build_reference_from_synthetic",
    "pangenome_graph_batch",
    # real-data ingestion (FASTA + optional GFA + reads/truth BAM)
    "RealReference",
    "build_reference_from_files",
    "load_real_reads",
    "build_batches",
    "parse_region",
    "read_fasta_contig",
    "build_datasets",
    "save_dataset",
    "load_dataset",
    # standard-format export
    "export_dataset",
    "sam_to_bam",
    "read_bam",
    "read_gfa",
    # format I/O layer (FASTQ/BAM/uBAM/SAM/CRAM/GFA in; BAM/SAM/CRAM/GFA/GBZ out)
    "read_fastq",
    "read_paired_fastq",
    "read_reads",
    "is_unaligned_bam",
    "validate_modality",
    # pipeline results -> BAM / SAM / CRAM
    "write_alignments",
    "write_alignments_split",
    "alignments_to_records",
    "alignment_to_read_record",
    "write_bam",
    "write_sam",
    "write_truth_sam",
    "write_cram",
    "write_gfa_graph",
    "write_gbz",
    "write_giraffe_indexes",
    "convert_reads",
    "convert_graph",
    "write_fasta",
    "write_fastq",
    "write_gfa",
    "write_labels_json",
]
