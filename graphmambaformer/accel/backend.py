"""Capability detection and global switches for the GPU acceleration stack.

The alignment stages are written once against ``torch`` and then lifted onto the
fastest tier available on the host. :class:`AccelContext` is the single place
that answers "what can this machine do?", so no stage has to repeat the
``try: import cupy`` dance.

Tiers, fastest first (mirroring the architecture's 5-tier compute stack):

===========================  ============================================
Tier                         Requirement
===========================  ============================================
``cuda_rawkernel``           CUDA GPU + ``cupy``
``triton``                   CUDA GPU + ``triton``
``torch_cuda``               CUDA GPU
``torch_mps``                Apple Silicon / Metal
``torch_cpu``                always available
===========================  ============================================

Every kernel in :mod:`graphmambaformer.accel` exposes the same signature at
every tier, so correctness is verified on CPU and the CUDA tiers are pure
throughput wins.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from ..config import AccelConfig


# --------------------------------------------------------------------------- #
# Optional-dependency probes. Cached: importing cupy/triton is not cheap.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _try_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:  # ImportError, but also CUDA-init errors from cupy
        return None


def cupy_module() -> Any | None:
    """Return ``cupy`` if it is importable *and* backed by a working GPU."""
    cp = _try_import("cupy")
    if cp is None:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return None
    except Exception:
        return None
    return cp


def triton_module() -> Any | None:
    return _try_import("triton")


def mamba_ssm_available() -> bool:
    return _try_import("mamba_ssm") is not None


@dataclass(frozen=True)
class AccelCapabilities:
    """Immutable snapshot of what the host can accelerate.

    Written to span every GPU the code may land on rather than assuming a recent
    NVIDIA part. PyTorch reports ROCm devices through the same ``torch.cuda``
    API as CUDA, so ``has_cuda`` alone cannot tell the two apart -- ``vendor``
    is what the dtype and kernel gates key on.
    """

    device: torch.device
    has_cuda: bool
    has_mps: bool
    has_cupy: bool
    has_triton: bool
    has_mamba_ssm: bool
    compute_capability: tuple[int, int] | None
    device_name: str
    vendor: str = "cpu"
    has_xpu: bool = False
    #: ROCm/HIP arch string (e.g. ``gfx90a``); ``None`` off AMD.
    hip_arch: str | None = None

    @property
    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    @property
    def tier(self) -> str:
        # CuPy raw kernels are compiled with NVRTC, so they are NVIDIA-only.
        if self.is_nvidia and self.has_cupy:
            return "cuda_rawkernel"
        if self.has_cuda and self.has_triton:
            return "triton"
        if self.has_cuda:
            return "torch_cuda"
        if self.has_xpu:
            return "torch_xpu"
        if self.has_mps:
            return "torch_mps"
        return "torch_cpu"

    @property
    def supports_tf32(self) -> bool:
        """TF32 tensor cores land on Ampere (sm_80) and later. NVIDIA only."""
        return (
            self.is_nvidia
            and self.compute_capability is not None
            and self.compute_capability[0] >= 8
        )

    @property
    def supports_fp16(self) -> bool:
        """Half precision that is actually fast, not merely representable.

        Pre-Volta NVIDIA parts (sm_6x and older) have no fp16 tensor cores and
        run AMP slower than fp32, so they are excluded deliberately.
        """
        if self.is_nvidia:
            return self.compute_capability is not None and self.compute_capability >= (7, 0)
        if self.vendor == "amd":
            return True
        return self.vendor in ("intel", "apple")

    @property
    def supports_bf16(self) -> bool:
        if self.has_cuda:
            # True on NVIDIA sm_80+ and on AMD MI200+; torch answers for both.
            with contextlib.suppress(Exception):
                return bool(torch.cuda.is_bf16_supported())
            return False
        if self.has_xpu:
            return True
        # CPU/MPS bf16 autocast exists but is slower than fp32 for these shapes.
        return False

    @property
    def supports_fp8(self) -> bool:
        """FP8 arrives with Ada (sm_89), not just Hopper (sm_90)."""
        return (
            self.is_nvidia
            and self.compute_capability is not None
            and self.compute_capability >= (8, 9)
        )

    @property
    def arch_label(self) -> str:
        """Human-readable microarchitecture, for logs and the doctor script."""
        if self.vendor == "amd":
            return self.hip_arch or "rocm"
        if not self.is_nvidia or self.compute_capability is None:
            return self.vendor
        major, minor = self.compute_capability
        generations = {
            6: "Pascal", 7: "Volta" if minor == 0 else "Turing",
            8: "Ampere" if minor < 9 else "Ada", 9: "Hopper",
            10: "Blackwell", 12: "Blackwell",
        }
        return f"{generations.get(major, f'sm_{major}{minor}')} (sm_{major}{minor})"

    def summary(self) -> str:
        flags = [
            f"tier={self.tier}",
            f"vendor={self.vendor}",
            f"device={self.device}",
            f"name={self.device_name}",
            f"arch={self.arch_label}",
            f"cupy={self.has_cupy}",
            f"triton={self.has_triton}",
            f"mamba_ssm={self.has_mamba_ssm}",
            f"tf32={self.supports_tf32}",
            f"fp16={self.supports_fp16}",
            f"bf16={self.supports_bf16}",
        ]
        return " | ".join(flags)


def detect_capabilities(device: torch.device | str | None = None) -> AccelCapabilities:
    """Probe the host for every acceleration tier, on any vendor.

    Every probe is guarded: a torch build without ``torch.xpu``, a ROCm build
    that reports no arch, or a driver that refuses ``get_device_capability``
    must degrade to a lower tier rather than raise at import time.
    """
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    has_xpu = False
    with contextlib.suppress(AttributeError, RuntimeError):
        has_xpu = bool(torch.xpu.is_available())  # type: ignore[attr-defined]

    # A ROCm build exposes AMD GPUs through the torch.cuda namespace, so the
    # HIP version string is the only reliable discriminator.
    is_rocm = bool(getattr(torch.version, "hip", None))

    if device is None:
        resolved = torch.device(
            "cuda" if has_cuda else "xpu" if has_xpu else "mps" if has_mps else "cpu"
        )
    else:
        resolved = torch.device(device)

    capability: tuple[int, int] | None = None
    hip_arch: str | None = None
    name = "cpu"
    vendor = "cpu"

    if has_cuda:
        vendor = "amd" if is_rocm else "nvidia"
        with contextlib.suppress(Exception):
            name = torch.cuda.get_device_name(0)
        if is_rocm:
            with contextlib.suppress(Exception):
                hip_arch = torch.cuda.get_device_properties(0).gcnArchName
        else:
            with contextlib.suppress(Exception):
                capability = torch.cuda.get_device_capability(0)
    elif has_xpu:
        vendor, name = "intel", "intel-xpu"
        with contextlib.suppress(Exception):
            name = torch.xpu.get_device_name(0)  # type: ignore[attr-defined]
    elif has_mps:
        vendor, name = "apple", "apple-silicon-mps"

    # CuPy raw kernels need NVRTC, and mamba-ssm ships CUDA-only kernels, so
    # neither is claimed off NVIDIA. Triton does support ROCm.
    is_nvidia = vendor == "nvidia"
    return AccelCapabilities(
        device=resolved,
        has_cuda=has_cuda,
        has_mps=has_mps,
        has_cupy=is_nvidia and cupy_module() is not None,
        has_triton=has_cuda and triton_module() is not None,
        has_mamba_ssm=is_nvidia and mamba_ssm_available(),
        compute_capability=capability,
        device_name=name,
        vendor=vendor,
        has_xpu=has_xpu,
        hip_arch=hip_arch,
    )


class AccelContext:
    """Applies the acceleration switches and hands out per-stage backends.

    Construct once and pass it to the pipeline; the stages query it instead of
    probing hardware themselves.
    """

    def __init__(
        self,
        cfg: AccelConfig | None = None,
        device: torch.device | str | None = None,
    ):
        self.cfg = cfg or AccelConfig()
        self.caps = detect_capabilities(device or self.cfg.device)
        self._applied = False
        if self.cfg.apply_global_switches:
            self.apply_global_switches()

    # ---- global torch switches -------------------------------------------- #
    def apply_global_switches(self) -> None:
        """Enable TF32 matmul, Flash/MemEfficient SDP, and cuDNN autotuning."""
        if self._applied:
            return
        self._applied = True

        if self.cfg.tf32 and self.caps.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # torch>=2.x also exposes a string-valued precision knob.
            with contextlib.suppress(AttributeError, RuntimeError):
                torch.set_float32_matmul_precision("high")

        # Flash/mem-efficient SDP exist on ROCm too; the suppress() below covers
        # builds where a given setter is missing.
        if self.cfg.flash_sdp and self.caps.has_cuda:
            for enable, setter in (
                (True, "enable_flash_sdp"),
                (True, "enable_mem_efficient_sdp"),
                (self.cfg.allow_math_sdp, "enable_math_sdp"),
            ):
                with contextlib.suppress(AttributeError, RuntimeError):
                    getattr(torch.backends.cuda, setter)(enable)

        if self.cfg.cudnn_benchmark and self.caps.has_cuda:
            torch.backends.cudnn.benchmark = True

    # ---- mixed precision --------------------------------------------------- #
    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """The dtype to autocast to, or ``None`` to run in full precision."""
        if not self.cfg.amp:
            return None
        if self.cfg.amp_dtype == "bf16":
            return torch.bfloat16 if self.caps.supports_bf16 else None
        if self.cfg.amp_dtype == "fp16":
            return torch.float16 if self.caps.supports_fp16 else None
        # "auto": prefer bf16 (no GradScaler needed), else fp16, else off.
        if self.caps.supports_bf16:
            return torch.bfloat16
        if self.caps.supports_fp16:
            return torch.float16
        return None

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Mixed-precision context; a no-op when AMP is unavailable/disabled."""
        dtype = self.autocast_dtype
        if dtype is None:
            yield
            return
        with torch.autocast(device_type=self.caps.device.type, dtype=dtype):
            yield

    def grad_scaler(self) -> torch.amp.GradScaler | None:
        """A :class:`GradScaler` for fp16 AMP; ``None`` for bf16/off (not needed)."""
        if self.autocast_dtype is torch.float16:
            return torch.amp.GradScaler(device=self.caps.device.type)
        return None

    # ---- compilation ------------------------------------------------------- #
    def compile(self, module: torch.nn.Module) -> torch.nn.Module:
        """``torch.compile`` the module when it is safe and enabled.

        Skipped on MPS, where inductor support for these kernels is incomplete
        and eager is consistently faster.
        """
        if not self.cfg.compile or self.caps.device.type == "mps":
            return module
        with contextlib.suppress(Exception):
            return torch.compile(module, mode=self.cfg.compile_mode, dynamic=True)
        return module

    # ---- per-stage backend choice ------------------------------------------ #
    def kernel_backend(self, stage: str) -> str:
        """Which implementation tier ``stage`` should use.

        ``stage`` is one of ``"seeding"``, ``"chaining"``, ``"extension"``.
        Honours the per-stage override in :class:`AccelConfig` and otherwise
        falls back to the best tier the host supports.
        """
        override = self.cfg.stage_backends.get(stage)
        if override and override != "auto":
            return override
        if self.caps.is_nvidia and self.caps.has_cupy and self.cfg.cuda_rawkernels:
            return "cuda_rawkernel"
        return "torch"

    def summary(self) -> str:
        dtype = self.autocast_dtype
        return f"{self.caps.summary()} | amp={dtype if dtype else 'off'}"


# --------------------------------------------------------------------------- #
# Array-namespace lift (NumPy -> CuPy) used by the index-building code, which is
# array-programming rather than tensor-programming.
# --------------------------------------------------------------------------- #
def array_namespace(device: torch.device | str | None = None) -> Any:
    """Return ``cupy`` when a CUDA device is requested and available, else ``numpy``.

    Both expose the subset used by the seeding indices (``argsort``,
    ``searchsorted``, ``bincount``, bit ops), so callers stay backend-agnostic.
    """
    import numpy as np

    if device is not None and torch.device(device).type == "cuda":
        cp = cupy_module()
        if cp is not None:
            return cp
    return np


def to_numpy(array: Any) -> Any:
    """Bring a NumPy *or* CuPy array back to host memory as NumPy."""
    import numpy as np

    if isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    if callable(get):  # cupy.ndarray
        return get()
    return np.asarray(array)


# --------------------------------------------------------------------------- #
# CUDA Graphs: static capture removes per-launch overhead for fixed shapes.
# --------------------------------------------------------------------------- #
@dataclass
class CUDAGraphRunner:
    """Capture a fixed-shape callable into a CUDA graph and replay it.

    Falls back to plain eager execution whenever capture is unavailable (no
    CUDA, dynamic shapes, or a capture error), so call sites need no branching.
    """

    fn: Callable[..., Any]
    enabled: bool = True
    warmup_steps: int = 3
    _graph: Any = field(default=None, init=False, repr=False)
    _static_args: tuple = field(default=(), init=False, repr=False)
    _static_out: Any = field(default=None, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _failed: bool = field(default=False, init=False, repr=False)

    def __call__(self, *args: torch.Tensor) -> Any:
        if not self._capturable(args):
            return self.fn(*args)

        self._calls += 1
        if self._graph is None:
            if self._calls <= self.warmup_steps:
                return self.fn(*args)
            try:
                self._capture(args)
            except Exception:
                self._failed = True
                return self.fn(*args)

        for static, incoming in zip(self._static_args, args):
            static.copy_(incoming)
        self._graph.replay()
        return self._static_out

    def _capturable(self, args: tuple) -> bool:
        if not (self.enabled and torch.cuda.is_available()) or self._failed:
            return False
        if not all(isinstance(a, torch.Tensor) and a.is_cuda for a in args):
            return False
        if self._static_args:
            return all(
                s.shape == a.shape and s.dtype == a.dtype
                for s, a in zip(self._static_args, args)
            )
        return True

    def _capture(self, args: tuple) -> None:
        self._static_args = tuple(a.clone() for a in args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_out = self.fn(*self._static_args)
        self._graph = graph


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
_DEFAULT_CONTEXT: AccelContext | None = None


def default_context() -> AccelContext:
    """Process-wide :class:`AccelContext`, created on first use."""
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = AccelContext()
    return _DEFAULT_CONTEXT


def accel_summary() -> str:
    """One-line description of the active acceleration stack."""
    return default_context().summary()


# Keep MPS from hard-failing on ops without a Metal kernel (e.g. linalg.eigh).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
