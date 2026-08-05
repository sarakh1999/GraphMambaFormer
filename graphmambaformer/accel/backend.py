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
    """Immutable snapshot of what the host can accelerate."""

    device: torch.device
    has_cuda: bool
    has_mps: bool
    has_cupy: bool
    has_triton: bool
    has_mamba_ssm: bool
    compute_capability: tuple[int, int] | None
    device_name: str

    @property
    def tier(self) -> str:
        if self.has_cuda and self.has_cupy:
            return "cuda_rawkernel"
        if self.has_cuda and self.has_triton:
            return "triton"
        if self.has_cuda:
            return "torch_cuda"
        if self.has_mps:
            return "torch_mps"
        return "torch_cpu"

    @property
    def supports_tf32(self) -> bool:
        """TF32 tensor cores land on Ampere (sm_80) and later."""
        return self.compute_capability is not None and self.compute_capability[0] >= 8

    @property
    def supports_bf16(self) -> bool:
        if self.has_cuda:
            return torch.cuda.is_bf16_supported()
        # CPU bf16 autocast exists but is slower than fp32 for these shapes.
        return False

    @property
    def supports_fp8(self) -> bool:
        """TransformerEngine FP8 needs Hopper (sm_90) or newer."""
        return self.compute_capability is not None and self.compute_capability[0] >= 9

    def summary(self) -> str:
        flags = [
            f"tier={self.tier}",
            f"device={self.device}",
            f"name={self.device_name}",
            f"cupy={self.has_cupy}",
            f"triton={self.has_triton}",
            f"mamba_ssm={self.has_mamba_ssm}",
            f"tf32={self.supports_tf32}",
            f"bf16={self.supports_bf16}",
        ]
        return " | ".join(flags)


def detect_capabilities(device: torch.device | str | None = None) -> AccelCapabilities:
    """Probe the host for every acceleration tier."""
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if device is None:
        resolved = torch.device("cuda" if has_cuda else "mps" if has_mps else "cpu")
    else:
        resolved = torch.device(device)

    capability: tuple[int, int] | None = None
    name = "cpu"
    if has_cuda:
        capability = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    elif has_mps:
        name = "apple-silicon-mps"

    return AccelCapabilities(
        device=resolved,
        has_cuda=has_cuda,
        has_mps=has_mps,
        # cupy/triton only matter on CUDA; don't pay the import cost otherwise.
        has_cupy=has_cuda and cupy_module() is not None,
        has_triton=has_cuda and triton_module() is not None,
        has_mamba_ssm=has_cuda and mamba_ssm_available(),
        compute_capability=capability,
        device_name=name,
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
            return torch.float16 if self.caps.has_cuda else None
        # "auto": prefer bf16 (no GradScaler needed), else fp16, else off.
        if self.caps.supports_bf16:
            return torch.bfloat16
        if self.caps.has_cuda:
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
        if self.caps.has_cuda and self.caps.has_cupy and self.cfg.cuda_rawkernels:
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
