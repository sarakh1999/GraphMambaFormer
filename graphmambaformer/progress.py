"""Lightweight tqdm wrappers for long-running training and data paths.

Bars are on by default so Docker / batch logs still show ETA. Disable with
``GMF_DISABLE_TQDM=1`` or ``GMF_TQDM=0``. If tqdm is missing, iteration is
unchanged (no crash).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

__all__ = ["progress", "progress_disabled"]


def progress_disabled() -> bool:
    """True when the user opted out of progress bars via the environment."""
    flag = os.environ.get("GMF_DISABLE_TQDM", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    flag = os.environ.get("GMF_TQDM", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return True
    return False


def progress(
    iterable: Optional[Iterable[T]] = None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: str = "it",
    disable: bool = False,
    leave: bool = True,
    **kwargs,
):
    """Wrap an iterable (or open a manual bar) with tqdm when available.

    Pass ``iterable=None`` and ``total=N`` for a bar you ``update()`` yourself.
    """
    off = bool(disable) or progress_disabled()
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional at runtime
        if iterable is None:
            return _NullBar(total or 0)
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=off,
        leave=leave,
        dynamic_ncols=True,
        **kwargs,
    )


class _NullBar:
    """No-op stand-in when tqdm is not installed and a manual bar is requested."""

    def __init__(self, total: int = 0):
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
