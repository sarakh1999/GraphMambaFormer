"""CPU parallelism helpers for the alignment stages and the training feed.

The neural core runs on the accelerator, but Stages 1-2 (seeding, chaining) and
the supervision builder are array/Python code that runs on the host. When those
run single-threaded they *starve the GPU*: the accelerator sits idle waiting for
the next batch of anchors, so ``nvidia-smi`` reports ~0 % utilisation even though
CUDA is available and the model is on the device.

Two primitives fix that:

* :func:`parallel_map` fans a per-read function out across a thread pool. The
  seeding indices are NumPy-heavy and release the GIL inside their C loops, so
  threads give a real speed-up without the pickling cost of processes (the
  reference index and FM-index would be expensive to ship to workers).
* :class:`Prefetcher` builds the *next* training batch on background threads
  while the GPU is still busy with the current one, overlapping host work with
  device compute. This is the single biggest lever on GPU utilisation for this
  pipeline, because supervision building runs the classical stages every step.

Everything degrades to serial execution when only one worker is available or the
work item count is below a threshold, so correctness never depends on threading.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: Override the worker count without touching code (mirrors OMP_NUM_THREADS).
ENV_WORKERS = "GMF_NUM_WORKERS"

__all__ = [
    "ENV_WORKERS",
    "default_worker_count",
    "parallel_map",
    "configure_torch_threads",
    "Prefetcher",
]


def default_worker_count(requested: int | None = None) -> int:
    """Resolve the worker count: explicit request > ``$GMF_NUM_WORKERS`` > CPUs.

    Always returns at least 1. A non-positive request or env value means "auto".
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    env = os.environ.get(ENV_WORKERS, "").strip()
    if env.lstrip("+-").isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
    min_items: int = 2,
    pbar: str | None = None,
) -> list[R]:
    """Apply ``fn`` to every item, in parallel across a thread pool, in order.

    Falls back to a plain list comprehension when there is only one worker or
    fewer than ``min_items`` items, so the fast path carries no pool overhead.
    Results preserve input order regardless of completion order.

    When ``pbar`` is set, a tqdm bar with that description tracks completion
    (useful for long seed/extend passes over many reads).
    """
    from ..progress import progress

    items = list(items)
    n = len(items)
    w = default_worker_count(workers)
    show = pbar is not None and n > 0
    if w <= 1 or n < max(2, min_items):
        return [
            fn(x)
            for x in progress(
                items, desc=pbar, unit="read", disable=not show, leave=False
            )
        ]
    w = min(w, n)
    # A fresh pool per call keeps this reentrant (the trainer nests a prefetch
    # pool around these stage pools); pool creation is cheap next to the work.
    with ThreadPoolExecutor(max_workers=w, thread_name_prefix="gmf-stage") as ex:
        return list(
            progress(
                ex.map(fn, items),
                total=n,
                desc=pbar,
                unit="read",
                disable=not show,
                leave=False,
            )
        )


_THREADS_LOCK = threading.Lock()
_THREADS_CONFIGURED = False


def configure_torch_threads(
    device_type: str = "cpu",
    *,
    workers: int | None = None,
    force: bool = False,
) -> dict:
    """Size torch's intra-op and BLAS thread pools to the host's cores.

    On a CPU-only run this lets the model use every core; on a GPU run it keeps
    the host-side stages (encoding, chaining fallbacks) from being pinned to a
    single thread. Idempotent unless ``force`` is set. Returns a small dict of
    what was applied, for logging.
    """
    global _THREADS_CONFIGURED
    import torch

    with _THREADS_LOCK:
        if _THREADS_CONFIGURED and not force:
            return {"skipped": True, "num_threads": torch.get_num_threads()}
        cores = default_worker_count(workers)
        # Leave the OpenMP/MKL/OpenBLAS pools consistent with torch so numpy in
        # the stages and torch on the CPU don't oversubscribe each other.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, str(cores))
        applied = {"requested_cores": cores, "device": device_type}
        try:
            torch.set_num_threads(cores)
            applied["num_threads"] = torch.get_num_threads()
        except Exception:  # pragma: no cover - platform dependent
            applied["num_threads"] = None
        try:
            # Inter-op parallelism only helps when there are independent graph
            # branches; keep it modest to avoid contention with the stage pool.
            torch.set_num_interop_threads(max(1, min(4, cores // 2)))
            applied["interop_threads"] = torch.get_num_interop_threads()
        except Exception:  # pragma: no cover - can only be set once per process
            pass
        _THREADS_CONFIGURED = True
        return applied


class Prefetcher:
    """Overlap host-side batch preparation with device compute.

    ``build`` is called on background threads for upcoming items while the caller
    consumes the current one, so the GPU does not stall on CPU seeding/chaining.
    Iterating yields ``(prepared, item)`` in the original order. ``depth`` bounds
    how far ahead it runs (and therefore the memory held by pending batches).

    With ``depth <= 0`` or ``workers <= 1`` it degrades to a lazy serial map, so
    behaviour is identical to the un-prefetched loop apart from timing.
    """

    def __init__(
        self,
        build: Callable[[T], R],
        items: Sequence[T],
        *,
        depth: int = 2,
        workers: int | None = None,
    ):
        self.build = build
        self.items = list(items)
        self.depth = int(depth)
        self.workers = default_worker_count(workers)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[tuple[R, T]]:
        if self.depth <= 0 or self.workers <= 1 or len(self.items) <= 1:
            for item in self.items:
                yield self.build(item), item
            return

        # A bounded queue of in-flight futures gives look-ahead of `depth`
        # without materialising every batch at once.
        pending: "queue.Queue" = queue.Queue()
        max_inflight = min(self.depth + 1, len(self.items))
        with ThreadPoolExecutor(
            max_workers=min(self.workers, max_inflight),
            thread_name_prefix="gmf-prefetch",
        ) as ex:
            it = iter(self.items)
            submitted = 0
            for _ in range(max_inflight):
                item = next(it)
                pending.put((ex.submit(self.build, item), item))
                submitted += 1
            done = 0
            total = len(self.items)
            while done < total:
                future, item = pending.get()
                result = future.result()
                # Refill before yielding so the next build is already running
                # while the caller processes this one.
                if submitted < total:
                    nxt = next(it)
                    pending.put((ex.submit(self.build, nxt), nxt))
                    submitted += 1
                done += 1
                yield result, item
