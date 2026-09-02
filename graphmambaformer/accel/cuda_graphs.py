"""CUDA Graph capture for fixed-shape model forwards.

Wraps a callable (typically the core ``nn.Module``) so repeated inference with
identical shapes replays a captured CUDA graph and skips per-launch CPU
overhead. Dynamic shapes, CPU tensors, a live ``graph`` / ``modality`` argument,
missing CUDA, or a capture error all fall back to eager execution automatically.

Used by the hybrid alignment pipeline and :class:`NeuralScorer` when
``AccelConfig.cuda_graphs`` is on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from .backend import CUDAGraphRunner

__all__ = ["GraphCapturedForward", "wrap_model_forward"]


@dataclass
class GraphCapturedForward:
    """Capture ``fn(base_codes, mask=..., qualities=...)`` into a CUDA graph.

    ``graph`` / ``modality`` force eager mode (not plain fixed-shape CUDA
    tensors). After ``warmup`` eager calls with a stable shape, the next matching
    call is captured and subsequent ones replay.
    """

    fn: Callable[..., Any]
    enabled: bool = True
    warmup: int = 3
    _runners: dict[tuple, CUDAGraphRunner] = field(default_factory=dict, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _backend: str = field(default="eager", init=False, repr=False)

    @property
    def backend(self) -> str:
        return self._backend

    def __call__(
        self,
        base_codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        qualities: Optional[torch.Tensor] = None,
        graph: Any = None,
        modality: Any = None,
        **kwargs: Any,
    ) -> Any:
        use_eager = (
            not self.enabled
            or graph is not None
            or modality is not None
            or not torch.cuda.is_available()
            or not isinstance(base_codes, torch.Tensor)
            or base_codes.device.type != "cuda"
            or (mask is not None and mask.device.type != "cuda")
            or (qualities is not None and qualities.device.type != "cuda")
        )
        if use_eager:
            self._backend = "eager"
            return self.fn(
                base_codes,
                mask=mask,
                qualities=qualities,
                graph=graph,
                modality=modality,
                **kwargs,
            )

        key = (
            tuple(base_codes.shape),
            None if mask is None else tuple(mask.shape),
            None if qualities is None else tuple(qualities.shape),
            str(base_codes.dtype),
        )
        runner = self._runners.get(key)
        if runner is None:
            has_mask = mask is not None
            has_qual = qualities is not None
            # Freeze kwargs (e.g. run_heads) into the captured callable.
            frozen = dict(kwargs)

            def _fixed(*tensors: torch.Tensor) -> Any:
                bc = tensors[0]
                idx = 1
                m = None
                if has_mask:
                    m = tensors[idx]
                    idx += 1
                q = tensors[idx] if has_qual else None
                return self.fn(bc, mask=m, qualities=q, **frozen)

            runner = CUDAGraphRunner(_fixed, enabled=True, warmup_steps=self.warmup)
            self._runners[key] = runner

        pack: list[torch.Tensor] = [base_codes]
        if mask is not None:
            pack.append(mask)
        if qualities is not None:
            pack.append(qualities)

        self._calls += 1
        out = runner(*pack)
        self._backend = "cuda_graph" if runner._graph is not None else "eager"
        return out


def wrap_model_forward(
    model: torch.nn.Module, *, enabled: bool, warmup: int = 3
) -> GraphCapturedForward:
    """Return a :class:`GraphCapturedForward` around ``model`` (``__call__``)."""
    return GraphCapturedForward(fn=model, enabled=enabled, warmup=warmup)
