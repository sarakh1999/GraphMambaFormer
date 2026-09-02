"""Distributed data-parallel helpers for multi-GPU training via ``torchrun``.

Why this instead of ``nn.DataParallel`` / ``nn.parallel.DistributedDataParallel``
wrapping:

* The core model's ``forward`` returns a custom :class:`GraphMambaOutput`
  dataclass (not a tensor), and the training step calls the scoring heads on the
  *unwrapped* module (``raw_model``) plus a loss module that owns its own
  learnable Kendall parameters. ``nn.DataParallel`` cannot gather the custom
  output, and a ``DistributedDataParallel`` wrapper keys its gradient reducer off
  the autograd graph reachable from the wrapped ``forward`` output -- so the
  scoring-head / criterion parameters used *outside* that forward would be
  mis-handled. Both would require a large, risky rewrite of the multi-stage step.

* The batches are a pre-built in-memory list of ``(reads, reference)`` tuples,
  not a ``Dataset`` fed through a ``DataLoader`` -- so the usual
  ``DistributedSampler`` sharding does not apply.

The lower-risk, provably-correct design used here: launch one process per GPU
with ``torchrun``; each rank runs the *unchanged* single-GPU training step over
its own strided shard of the batch list (``batches[rank::world_size]``); average
gradients across ranks once per optimizer step (matching DDP semantics); reduce
logged / early-stopping metrics across ranks so every rank agrees; and let only
rank 0 touch the disk. When not launched under ``torchrun`` (``world_size == 1``)
every method below is a no-op, so the single-GPU / CPU path is byte-for-byte the
behaviour it had before.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class DistContext:
    """Handle to the (optional) distributed process group.

    ``enabled`` is ``False`` for an ordinary single-process run; in that state
    every method is a no-op and ``rank``/``world_size`` are ``0``/``1``.
    """

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"

    @property
    def is_main(self) -> bool:
        """True on the one rank that owns disk writes / user-facing logs."""
        return self.rank == 0

    # ---- collectives (all no-ops when not enabled) ------------------------- #
    def barrier(self) -> None:
        if self.enabled:
            torch.distributed.barrier()

    def shard(self, items: Sequence) -> list:
        """Return this rank's slice of ``items``, with equal length on every rank.

        Rank ``i`` takes ``items[i::world_size]`` (a balanced round-robin), then
        every rank truncates to ``len(items) // world_size`` so the per-epoch
        optimizer-step count is identical across ranks. That lockstep is what
        keeps the per-step gradient/metric collectives from dead-locking; the
        dropped tail is at most ``world_size - 1`` batches (standard drop-last).
        """
        items = list(items)
        if not self.enabled:
            return items
        per_rank = len(items) // self.world_size
        shard = items[self.rank :: self.world_size]
        return shard[:per_rank]

    def average_gradients(self, params: Sequence[torch.nn.Parameter]) -> None:
        """All-reduce (mean) the ``.grad`` of ``params`` across ranks in place.

        Called once per optimizer step (after the grad-accum window, before
        clipping), so it matches DDP's "average the gradient over the global
        batch" semantics. Grads are bucketed by dtype and flattened so each
        dtype costs a single collective launch.
        """
        if not self.enabled:
            return
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        buckets: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
        for g in grads:
            buckets[g.dtype].append(g)
        for group in buckets.values():
            flat = torch._utils._flatten_dense_tensors(group)
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            flat.div_(self.world_size)
            for g, synced in zip(
                group, torch._utils._unflatten_dense_tensors(flat, group)
            ):
                g.copy_(synced)

    @property
    def _collective_device(self) -> torch.device:
        """Device that scalar collectives must live on for this backend.

        The NCCL backend only handles CUDA tensors -- all-reducing a CPU scalar
        on an NCCL group raises ``No backend type associated with device type
        cpu``. So place scalar reductions on this rank's GPU under NCCL, and on
        CPU under gloo.
        """
        if self.backend == "nccl":
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def reduce_mean(self, value: float) -> float:
        """Return the mean of a Python scalar across ranks (unchanged if off)."""
        if not self.enabled:
            return float(value)
        t = torch.tensor([float(value)], dtype=torch.float64, device=self._collective_device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return float(t.item() / self.world_size)

    def reduce_sum(self, value: float) -> float:
        """Return the sum of a Python scalar across ranks (unchanged if off)."""
        if not self.enabled:
            return float(value)
        t = torch.tensor([float(value)], dtype=torch.float64, device=self._collective_device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return float(t.item())


def maybe_init_distributed(requested_device: str | None = None) -> DistContext:
    """Initialise a process group iff launched under ``torchrun`` (WORLD_SIZE>1).

    Reads the standard ``torchrun`` environment (``RANK`` / ``WORLD_SIZE`` /
    ``LOCAL_RANK``). Picks ``nccl`` for CUDA and ``gloo`` otherwise. When run
    normally (no ``torchrun``, or ``WORLD_SIZE == 1``) returns a disabled
    :class:`DistContext` and does not touch ``torch.distributed`` at all, so the
    single-GPU path is untouched.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return DistContext(enabled=False)

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    want_cpu = bool(requested_device) and str(requested_device).lower().startswith("cpu")
    use_cuda = torch.cuda.is_available() and not want_cpu
    backend = "nccl" if use_cuda else "gloo"

    device_id = None
    if use_cuda:
        # Pin this rank to its own GPU *before* the NCCL group is created so all
        # subsequent allocations and the process group land on the right device.
        # local_rank is taken modulo the visible device count so a mismatch
        # between --nproc_per_node and the number of GPUs can't index past the
        # end (each GPU can host more than one rank if you oversubscribe).
        n_visible = max(1, torch.cuda.device_count())
        local_rank = local_rank % n_visible
        torch.cuda.set_device(local_rank)
        device_id = torch.device("cuda", local_rank)

    if not torch.distributed.is_initialized():
        # Passing device_id pins the rank->GPU mapping explicitly so NCCL does
        # not have to guess it from the global rank (which it warns about and
        # which can hang on heterogeneous mappings). Older torch builds lack the
        # kwarg, so fall back cleanly.
        try:
            torch.distributed.init_process_group(
                backend=backend, init_method="env://", device_id=device_id
            )
        except (TypeError, ValueError):
            torch.distributed.init_process_group(
                backend=backend, init_method="env://"
            )

    return DistContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def shutdown_distributed(ctx: DistContext) -> None:
    """Tear down the process group (best-effort); a no-op when not enabled."""
    if ctx.enabled and torch.distributed.is_initialized():
        with contextlib.suppress(Exception):
            torch.distributed.barrier()
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()


@contextlib.contextmanager
def main_process_first(ctx: DistContext) -> Iterator[None]:
    """Run the wrapped block on rank 0 first, then the other ranks.

    Used around the dataset build so rank 0 populates the on-disk dataset cache
    once while the other ranks wait, then the others run and hit the warm cache
    instead of all rebuilding it concurrently. A plain pass-through when not
    distributed.
    """
    if ctx.enabled and not ctx.is_main:
        ctx.barrier()
    try:
        yield
    finally:
        if ctx.enabled and ctx.is_main:
            ctx.barrier()
