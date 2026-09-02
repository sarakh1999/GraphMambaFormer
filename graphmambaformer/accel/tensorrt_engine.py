"""TensorRT / torch-tensorrt inference engines with eager fallback.

Builds an optimized inference path for the core model when NVIDIA TensorRT (or
``torch_tensorrt``) is installed. Everything degrades to the plain ``nn.Module``
forward when the libraries, CUDA, or fixed example shapes are missing — so the
same call sites work on a laptop CPU and on an A6000.

Preferred order
---------------
1. ``torch_tensorrt.compile`` — stays in the PyTorch module tree
2. ONNX export + TensorRT runtime (``tensorrt`` + ``onnx``) when explicitly asked
3. Eager PyTorch (always available)
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "tensorrt_available",
    "torch_tensorrt_available",
    "tensorrt_summary",
    "TensorRTInference",
    "maybe_compile_tensorrt",
]


@functools.lru_cache(maxsize=None)
def torch_tensorrt_available() -> bool:
    try:
        import torch_tensorrt  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def tensorrt_available() -> bool:
    """True when either ``torch_tensorrt`` or the raw ``tensorrt`` package loads."""
    if torch_tensorrt_available():
        return True
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def tensorrt_summary() -> str:
    if torch_tensorrt_available():
        return "tensorrt=torch_tensorrt"
    if tensorrt_available():
        return "tensorrt=native"
    return "tensorrt=no"


@dataclass
class TensorRTInference:
    """Wrap a module for TensorRT-accelerated inference.

    ``enabled=False`` or missing libraries → every call is a plain ``model(*)``.
    Compilation is lazy on the first CUDA call with example inputs so CPU unit
    tests never pay the TRT cost.
    """

    model: nn.Module
    enabled: bool = True
    precision: str = "fp16"  # "fp16" | "fp32" | "bf16"
    workspace_gb: float = 2.0
    _compiled: Any = field(default=None, init=False, repr=False)
    _failed: bool = field(default=False, init=False, repr=False)
    _backend: str = field(default="eager", init=False, repr=False)

    @property
    def backend(self) -> str:
        return self._backend

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled or self._failed:
            return self.model(*args, **kwargs)
        if self._compiled is None:
            self._try_compile(args, kwargs)
        if self._compiled is None:
            return self.model(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception:
            # Shape / dtype mismatch after compile → fall back once, keep eager.
            self._failed = True
            self._backend = "eager"
            return self.model(*args, **kwargs)

    def _try_compile(self, args: tuple, kwargs: dict) -> None:
        if not torch.cuda.is_available():
            self._failed = True
            self._backend = "eager"
            return
        # Only compile when the leading input is a CUDA tensor — otherwise the
        # caller is still on CPU and TRT cannot help.
        lead = args[0] if args else kwargs.get("base_codes")
        if not isinstance(lead, torch.Tensor) or lead.device.type != "cuda":
            return  # defer until a CUDA batch arrives

        if torch_tensorrt_available():
            try:
                import torch_tensorrt

                enabled_precisions = {torch.float32}
                if self.precision == "fp16":
                    enabled_precisions.add(torch.float16)
                elif self.precision == "bf16":
                    enabled_precisions.add(torch.bfloat16)

                # torch_tensorrt.compile wants a module in eval mode.
                was_training = self.model.training
                self.model.eval()
                self._compiled = torch_tensorrt.compile(
                    self.model,
                    inputs=[lead] if args else [],
                    enabled_precisions=enabled_precisions,
                    workspace_size=int(self.workspace_gb * (1 << 30)),
                    truncate_long_and_double=True,
                )
                self.model.train(was_training)
                self._backend = "torch_tensorrt"
                return
            except Exception:
                self._compiled = None

        # Native TensorRT via ONNX is optional and heavier; mark failed so we
        # do not retry every step. Callers that need an .engine file can use
        # :func:`export_onnx` separately.
        self._failed = True
        self._backend = "eager"

    def export_onnx(
        self,
        path: str,
        example_base_codes: torch.Tensor,
        example_mask: Optional[torch.Tensor] = None,
        opset: int = 17,
    ) -> str:
        """Export a minimal ONNX graph for external ``trtexec`` builds.

        Only ``base_codes`` (+ optional ``mask``) are exported — graph-conditioned
        paths stay on the PyTorch engine. Returns the written path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.eval()
        args: list[torch.Tensor] = [example_base_codes.cpu()]
        input_names = ["base_codes"]
        dynamic_axes = {"base_codes": {0: "batch", 1: "seq"}}
        if example_mask is not None:
            args.append(example_mask.cpu())
            input_names.append("mask")
            dynamic_axes["mask"] = {0: "batch", 1: "seq"}

        def _forward(base_codes, mask=None):
            return self.model(base_codes, mask=mask, run_heads=True).pooled

        torch.onnx.export(
            _OnnxAdapter(self.model),
            tuple(args),
            path,
            input_names=input_names,
            output_names=["pooled"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
        return path


class _OnnxAdapter(nn.Module):
    """Thin wrapper so ONNX export sees a single Tensor → Tensor forward."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, base_codes: torch.Tensor, mask: torch.Tensor | None = None):
        out = self.model(base_codes, mask=mask, run_heads=True)
        return out.pooled


def maybe_compile_tensorrt(
    model: nn.Module,
    *,
    enabled: bool,
    precision: str = "fp16",
) -> TensorRTInference | nn.Module:
    """Return a :class:`TensorRTInference` wrapper when ``enabled``, else ``model``."""
    if not enabled:
        return model
    return TensorRTInference(model=model, enabled=True, precision=precision)
