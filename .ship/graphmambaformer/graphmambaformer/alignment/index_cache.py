"""Disk cache for Stage-1 reference index bundles.

Building the FM-index / minimizer / fuzzy tables dominates startup cost. When
the same reference is aligned more than once — e.g. R1, then R2, then long
reads as separate invocations — caching the bundle to disk makes every run after
the first skip that work. A combined short+long run still builds once and reuses
the in-memory bundle for every read.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any

from ..config import SeedingConfig
from .seeding import SeedIndexBundle

__all__ = ["index_cache_key", "load_index_bundle", "save_index_bundle"]

_CACHE_VERSION = 1


def _stable_repr(obj: Any) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        return repr(sorted(asdict(obj).items()))
    return repr(obj)


def index_cache_key(
    ref_path: str,
    seeding: SeedingConfig,
    *,
    ref_id: int = 0,
) -> str:
    """Return a filesystem-safe key for ``(reference file, seeding config)``."""
    st = os.stat(ref_path)
    payload = "|".join(
        (
            f"v{_CACHE_VERSION}",
            os.path.abspath(ref_path),
            str(st.st_mtime_ns),
            str(st.st_size),
            str(ref_id),
            _stable_repr(seeding),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"seed_index_{key}.pkl")


def load_index_bundle(cache_dir: str | None, key: str) -> SeedIndexBundle | None:
    """Load a previously saved bundle, or ``None`` on miss / corruption."""
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            return None
        bundle = payload.get("bundle")
        return bundle if isinstance(bundle, SeedIndexBundle) else None
    except Exception:
        return None


def save_index_bundle(
    cache_dir: str | None, key: str, bundle: SeedIndexBundle
) -> str | None:
    """Persist ``bundle`` under ``cache_dir``. Returns the path written, or None."""
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, key)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(
            {"version": _CACHE_VERSION, "bundle": bundle},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)
    return path
