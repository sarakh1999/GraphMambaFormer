"""GPU acceleration stack: capability detection, CUDA RawKernels, Triton ops.

The alignment stages and the neural core are written once and then lifted onto
the fastest tier the host supports. :class:`AccelContext` is the entry point —
it applies the global torch switches (TF32, Flash SDP, cuDNN autotune), decides
the AMP dtype, and tells each stage which kernel tier to use.
"""

from .backend import (
    AccelCapabilities,
    AccelContext,
    CUDAGraphRunner,
    accel_summary,
    array_namespace,
    cupy_module,
    default_context,
    detect_capabilities,
    list_visible_gpus,
    mamba_ssm_available,
    nvidia_arch_label,
    to_numpy,
    triton_module,
)
from .cuda_kernels import kernels_available
from .cuda_graphs import GraphCapturedForward, wrap_model_forward
from .genomeworks_ops import (
    GlobalAlignment,
    MapperOverlap,
    UngappedExtension,
    genomeworks_available,
    genomeworks_backend,
    genomeworks_bindings_available,
    genomeworks_summary,
    global_align,
    global_align_batch,
    map_to_reference,
    poa_consensus,
    poa_msa,
    ungapped_extend,
    ungapped_extend_batch,
)
from .parallel import (
    ENV_WORKERS,
    Prefetcher,
    configure_torch_threads,
    default_worker_count,
    parallel_map,
)
from .simd_sw import simd_available, simd_isa
from .tensorrt_engine import (
    TensorRTInference,
    maybe_compile_tensorrt,
    tensorrt_available,
    tensorrt_summary,
    torch_tensorrt_available,
)
from .transformer_engine import (
    fp8_available_on_device,
    precision_context,
    te_summary,
    transformer_engine_available,
)
from .triton_ops import FusedLNLinearGELU, fused_ln_linear_gelu, triton_available

__all__ = [
    "AccelCapabilities",
    "AccelContext",
    "CUDAGraphRunner",
    "accel_summary",
    "array_namespace",
    "cupy_module",
    "default_context",
    "detect_capabilities",
    "kernels_available",
    "list_visible_gpus",
    "mamba_ssm_available",
    "simd_available",
    "simd_isa",
    "nvidia_arch_label",
    "to_numpy",
    "triton_module",
    "triton_available",
    "FusedLNLinearGELU",
    "fused_ln_linear_gelu",
    # CPU parallelism / GPU-feed overlap.
    "ENV_WORKERS",
    "Prefetcher",
    "configure_torch_threads",
    "default_worker_count",
    "parallel_map",
    # CUDA Graphs / TE FP8 / TensorRT.
    "GraphCapturedForward",
    "wrap_model_forward",
    "TensorRTInference",
    "maybe_compile_tensorrt",
    "tensorrt_available",
    "tensorrt_summary",
    "torch_tensorrt_available",
    "fp8_available_on_device",
    "precision_context",
    "te_summary",
    "transformer_engine_available",
    # GenomeWorks primitives (cudamapper / cudaaligner / cudaextender / cudapoa).
    "GlobalAlignment",
    "MapperOverlap",
    "UngappedExtension",
    "genomeworks_available",
    "genomeworks_backend",
    "genomeworks_bindings_available",
    "genomeworks_summary",
    "global_align",
    "global_align_batch",
    "map_to_reference",
    "poa_consensus",
    "poa_msa",
    "ungapped_extend",
    "ungapped_extend_batch",
]
