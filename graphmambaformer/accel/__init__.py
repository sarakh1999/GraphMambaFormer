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
    "nvidia_arch_label",
    "to_numpy",
    "triton_module",
    "triton_available",
    "FusedLNLinearGELU",
    "fused_ln_linear_gelu",
]
