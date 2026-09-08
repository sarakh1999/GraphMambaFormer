"""Neutralize the ``mamba_ssm`` -> TileLang import path that hard-aborts here.

Why this exists
---------------
``mamba_ssm/__init__.py`` unconditionally imports :class:`Mamba3`, whose module
pulls in the TileLang MIMO kernels (``import tilelang``). TileLang ships a
*stub* ``libcudart_stub.so`` whose ``TryLoadLibCudart`` (see TileLang
``src/target/stubs/cudart.cc``) resolves the real CUDA runtime like this::

    sym = dlsym(RTLD_DEFAULT, "cudaGetErrorString");
    if (sym && sym != &cudaGetErrorString) return RTLD_DEFAULT;  // found a *real* cudart
    sym = dlsym(RTLD_NEXT, "cudaGetErrorString");                // ...or one loaded AFTER me
    if (sym) return RTLD_NEXT;
    abort();  // "libcudart symbols not found globally. ..."

That only succeeds when a real ``libcudart`` is loaded *after* the stub (so
``RTLD_NEXT`` finds it) or when the stub's own ``cudaGetErrorString`` is not
interposed. In every GraphMambaFormer entrypoint torch is imported *first*, so
torch's ``libcudart.so.13`` is already global and *earlier* than the stub: its
symbol interposes the stub's own (the ``!= &cudaGetErrorString`` guard fails)
and ``RTLD_NEXT`` finds nothing later — so the stub calls ``abort()`` and core
dumps mid-pipeline. There is no in-process way to reorder torch vs. the stub.

The fix: make ``import tilelang`` raise :class:`ImportError` by parking ``None``
in :data:`sys.modules`. ``mamba_ssm``'s Mamba-3 already wraps that import in
``try/except ImportError`` and transparently falls back to its Triton kernels,
so models keep working and Mamba-1/Mamba-2 paths are untouched. This changes
nothing about correctness — only that Mamba-3 MIMO uses Triton instead of the
(here-unusable) TileLang kernels.

Opt back in with ``GMF_ENABLE_TILELANG=1`` if you have a TileLang setup that
works in your environment (e.g. a ``-DTILELANG_USE_CUDA_STUBS=OFF`` build, or an
entrypoint that imports ``tilelang`` before ``torch``).
"""
from __future__ import annotations

import os
import sys


def disable_broken_tilelang() -> bool:
    """Block ``import tilelang`` (idempotent). Returns True if it was disabled."""
    if os.environ.get("GMF_ENABLE_TILELANG") == "1":
        return False
    existing = sys.modules.get("tilelang", "__absent__")
    if existing is not None and existing != "__absent__":
        # Already genuinely imported — respect it rather than yanking it out.
        return False
    # Parking ``None`` makes a subsequent ``import tilelang`` raise ImportError,
    # which mamba_ssm's Mamba-3 catches to select its Triton fallback.
    sys.modules["tilelang"] = None  # type: ignore[assignment]
    return True


__all__ = ["disable_broken_tilelang"]
