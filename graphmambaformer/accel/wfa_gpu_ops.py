"""WFA-GPU batched gap-affine alignment (optional GPU CIGAR backend).

Wraps the external `WFA-GPU <https://github.com/quim0/WFA-GPU>`_ library (MIT)
through a small C-ABI shim (``csrc/wfa_gpu_shim.c`` -> ``libgmf_wfa_gpu.so``,
built by ``scripts/build_wfa_gpu.sh``). WFA-GPU is the maintained,
CUDA-12-capable stand-in for the archived GenomeWorks ``cudaaligner`` module: it
computes batched gap-affine pairwise alignments *with the CIGAR on the GPU* —
the piece the in-tree CuPy ``wfa_distance`` kernel cannot do (it returns the
edit distance only, leaving the CPU :class:`~graphmambaformer.alignment.extension.WavefrontAligner`
to produce every CIGAR).

Everything here degrades gracefully. If the shim library has not been built (the
default), :func:`available` returns ``False`` and callers keep using the
existing CPU/CuPy WFA path, byte-for-byte unchanged. The library is located via
the ``GMF_WFA_GPU_LIB`` environment variable or the default build location under
``build/wfa_gpu/``.
"""
from __future__ import annotations

import ctypes
import functools
import os
from typing import Optional, Sequence

_ENV_LIB = "GMF_WFA_GPU_LIB"

# Set when a candidate library exists but fails to dlopen (e.g. a missing
# libcudart), so :func:`summary` / callers can distinguish "not built" from
# "built but unloadable" instead of silently reporting unavailable.
_LOAD_ERROR: "str | None" = None


def _candidate_paths() -> list[str]:
    paths: list[str] = []
    env = os.environ.get(_ENV_LIB)
    if env:
        paths.append(env)
    # Default location produced by scripts/build_wfa_gpu.sh (repo_root/build/wfa_gpu).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths.append(os.path.join(repo_root, "build", "wfa_gpu", "libgmf_wfa_gpu.so"))
    return paths


@functools.lru_cache(maxsize=None)
def _load_library() -> Optional[ctypes.CDLL]:
    """Load (once) the WFA-GPU shim, or return ``None`` if it is not present."""
    global _LOAD_ERROR
    for candidate in _candidate_paths():
        if not candidate or not os.path.exists(candidate):
            continue
        # RTLD_GLOBAL so libwfagpu.so's WFA2 references resolve against the
        # WFA2 code embedded in this shim (see scripts/build_wfa_gpu.sh).
        try:
            lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # present but unloadable — remember why
            _LOAD_ERROR = f"{candidate}: {exc}"
            continue
        _bind(lib)
        lib._gmf_path = candidate  # type: ignore[attr-defined]
        return lib
    return None


def load_error() -> Optional[str]:
    """Reason the shim failed to load, if a candidate existed but could not
    ``dlopen`` (e.g. a missing ``libcudart``); ``None`` otherwise."""
    _load_library()
    return _LOAD_ERROR


def _bind(lib: ctypes.CDLL) -> None:
    lib.gmf_wfagpu_create.restype = ctypes.c_void_p
    lib.gmf_wfagpu_create.argtypes = []
    lib.gmf_wfagpu_add.restype = ctypes.c_int
    lib.gmf_wfagpu_add.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.gmf_wfagpu_num_pairs.restype = ctypes.c_long
    lib.gmf_wfagpu_num_pairs.argtypes = [ctypes.c_void_p]
    lib.gmf_wfagpu_run.restype = ctypes.c_int
    lib.gmf_wfagpu_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # x, o, e
        ctypes.c_int,                              # compute_cigar
        ctypes.c_long,                             # batch_size
        ctypes.c_int, ctypes.c_int,                # max_error, band
    ]
    lib.gmf_wfagpu_error.restype = ctypes.c_uint
    lib.gmf_wfagpu_error.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.gmf_wfagpu_cigar.restype = ctypes.c_char_p
    lib.gmf_wfagpu_cigar.argtypes = [ctypes.c_void_p, ctypes.c_long]
    lib.gmf_wfagpu_destroy.restype = None
    lib.gmf_wfagpu_destroy.argtypes = [ctypes.c_void_p]


def library_path() -> Optional[str]:
    """Filesystem path of the loaded shim, or ``None``."""
    lib = _load_library()
    return getattr(lib, "_gmf_path", None) if lib is not None else None


def available() -> bool:
    """True when the WFA-GPU shim loads, so a GPU CIGAR path exists."""
    return _load_library() is not None


def summary() -> str:
    """One-line description of the WFA-GPU tier, for logs/doctor."""
    path = library_path()
    if path:
        return f"wfa_gpu: lib={path}"
    err = load_error()
    return f"wfa_gpu: unavailable (load error: {err})" if err else "wfa_gpu: unavailable"


def align_batch(
    pairs: Sequence[tuple[str, str]],
    *,
    x: int = 1,
    o: int = 0,
    e: int = 1,
    compute_cigar: bool = True,
    max_error: int = 0,
    band: int = 0,
    batch_size: int = 0,
) -> Optional[list[tuple[int, Optional[str]]]]:
    """Batched gap-affine alignment of ``(query, target)`` ASCII pairs.

    Returns one ``(error, cigar)`` per pair — ``cigar`` is ``None`` when
    ``compute_cigar`` is False or the buffer is empty — or ``None`` when the
    library is unavailable / a call fails, so the caller can fall back.

    Penalties are WFA-style (match is implicitly ``0``); the default
    ``x=1, o=0, e=1`` computes unit-cost edit distance, exactly matching the
    in-tree :class:`WavefrontAligner`. ``max_error``/``band``/``batch_size`` <= 0
    keep the library's automatic choice.
    """
    lib = _load_library()
    if lib is None:
        return None
    pairs = list(pairs)
    if not pairs:
        return []

    handle = lib.gmf_wfagpu_create()
    if not handle:
        return None
    try:
        for query, target in pairs:
            rc = lib.gmf_wfagpu_add(
                handle,
                query.encode("ascii", "replace"),
                target.encode("ascii", "replace"),
            )
            if rc != 0:
                return None
        rc = lib.gmf_wfagpu_run(
            handle,
            int(x), int(o), int(e),
            1 if compute_cigar else 0,
            int(batch_size),
            int(max_error), int(band),
        )
        if rc != 0:
            return None
        out: list[tuple[int, Optional[str]]] = []
        for i in range(len(pairs)):
            err = int(lib.gmf_wfagpu_error(handle, i))
            cig: Optional[str] = None
            if compute_cigar:
                raw = lib.gmf_wfagpu_cigar(handle, i)  # bytes copy (or None)
                if raw:
                    cig = raw.decode("ascii", "replace")
            out.append((err, cig))
        return out
    finally:
        lib.gmf_wfagpu_destroy(handle)


__all__ = ["available", "align_batch", "library_path", "load_error", "summary"]
