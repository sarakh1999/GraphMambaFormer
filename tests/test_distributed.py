"""CPU (gloo) smoke test for the multi-GPU data-parallel plumbing.

The real training runs on GPUs, but the *correctness* of the distributed logic —
how the batch list is sharded across ranks, how gradients are averaged, and how
logged/early-stopping metrics are reduced — does not depend on CUDA. This test
spins up a real 2-process gloo group with ``torch.multiprocessing`` and exercises
:mod:`graphmambaformer.distributed` end to end, so the sharding + all-reduce math
is proven on a machine with no GPU before it is ever launched on a multi-GPU node.

Run: PYTHONPATH=. .venv/bin/python tests/run_all.py distributed
"""

from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp

from graphmambaformer.distributed import maybe_init_distributed, shutdown_distributed

WORLD_SIZE = 2


def _worker(rank: int, world_size: int, port: int, out: mp.Queue) -> None:
    """One rank of the gloo group; runs the assertions and reports results."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    result: dict = {"rank": rank}
    try:
        ctx = maybe_init_distributed(requested_device="cpu")
        assert ctx.enabled, "context should be enabled under WORLD_SIZE=2"
        assert ctx.world_size == world_size
        assert ctx.rank == rank
        assert ctx.backend == "gloo"
        assert ctx.is_main == (rank == 0)

        # 1) Sharding, even total: rank i takes items[i::world], equal lengths.
        even = ctx.shard(list(range(10)))
        assert len(even) == 5, even
        assert even == list(range(rank, 10, world_size)), even
        result["even_shard"] = even

        # 2) Sharding, odd total: both ranks truncate to floor(9/2)=4 (drop-last),
        #    so the per-epoch step counts stay identical and collectives can't hang.
        odd = ctx.shard(list(range(9)))
        assert len(odd) == 4, odd
        assert odd == list(range(rank, 9, world_size))[:4], odd
        result["odd_shard"] = odd

        # 3) Gradient averaging: each rank seeds a distinct gradient; after the
        #    all-reduce/mean both ranks must hold the average (here (1+2)/2 = 1.5).
        p_scalar = torch.nn.Parameter(torch.zeros(4))
        p_scalar.grad = torch.full((4,), float(rank + 1))
        p_mat = torch.nn.Parameter(torch.zeros(2, 3))
        p_mat.grad = torch.full((2, 3), float(rank + 1))
        ctx.average_gradients([p_scalar, p_mat])
        expected = (1.0 + 2.0) / 2.0
        assert torch.allclose(p_scalar.grad, torch.full((4,), expected)), p_scalar.grad
        assert torch.allclose(p_mat.grad, torch.full((2, 3), expected)), p_mat.grad
        result["grad_mean"] = float(p_scalar.grad[0])

        # 4) Scalar metric reductions used for logging + early stopping.
        result["reduce_mean_of_rank"] = ctx.reduce_mean(float(rank))  # (0+1)/2 = 0.5
        result["reduce_sum_of_rank"] = ctx.reduce_sum(float(rank))    # 0+1     = 1.0
        assert abs(result["reduce_mean_of_rank"] - 0.5) < 1e-9
        assert abs(result["reduce_sum_of_rank"] - 1.0) < 1e-9

        # 5) Barrier must not hang.
        ctx.barrier()
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - surface the failure to the parent
        import traceback

        result["ok"] = False
        result["error"] = f"{exc}\n{traceback.format_exc()}"
    finally:
        with_ctx = locals().get("ctx")
        if with_ctx is not None:
            shutdown_distributed(with_ctx)
        out.put(result)


def test_ddp_sharding_and_allreduce_gloo() -> None:
    """A real 2-process gloo group proves shard + grad/metric all-reduce."""
    # A per-invocation port keeps concurrent test runs from colliding.
    port = 20000 + (os.getpid() % 20000)
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()

    procs = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=_worker, args=(rank, WORLD_SIZE, port, queue))
        p.start()
        procs.append(p)

    results = [queue.get(timeout=120) for _ in range(WORLD_SIZE)]
    for p in procs:
        p.join(timeout=120)

    by_rank = {r["rank"]: r for r in results}
    for rank, res in sorted(by_rank.items()):
        assert res.get("ok"), f"rank {rank} failed:\n{res.get('error')}"

    # Cross-check the two ranks partitioned the list without overlap or loss.
    r0, r1 = by_rank[0], by_rank[1]
    assert sorted(r0["even_shard"] + r1["even_shard"]) == list(range(10))
    assert r0["grad_mean"] == r1["grad_mean"] == 1.5

    print(
        "gloo world_size=2: "
        f"even shards {r0['even_shard']} | {r1['even_shard']} (union=0..9), "
        f"odd shards {r0['odd_shard']} | {r1['odd_shard']} (drop-last), "
        f"grad mean={r0['grad_mean']}, "
        f"reduce_mean(rank)={r0['reduce_mean_of_rank']}, "
        f"reduce_sum(rank)={r0['reduce_sum_of_rank']}"
    )


if __name__ == "__main__":
    test_ddp_sharding_and_allreduce_gloo()
    print("OK")
