"""Acceleration-stack tests: capability detection and the portable fallbacks.

The whole point of :mod:`graphmambaformer.accel` is that the same code runs on an
H100 and on a laptop CPU, so the fallback paths are the ones that must be
correct: a fused Triton op has to produce the same numbers as the composed torch
ops, a CUDA-graph runner has to behave like a plain call, and the CuPy tier has
to decline cleanly rather than raise an import error.

The CUDA-only tiers are skipped (not failed) when the host lacks them, and the
test prints which tier it actually exercised so a CPU run is not mistaken for
coverage of the GPU path.

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from graphmambaformer.accel import (
    AccelCapabilities,
    AccelContext,
    CUDAGraphRunner,
    FusedLNLinearGELU,
    accel_summary,
    array_namespace,
    default_context,
    detect_capabilities,
    fused_ln_linear_gelu,
    kernels_available,
    to_numpy,
    triton_available,
)
from graphmambaformer.config import AccelConfig

HAS_CUDA = torch.cuda.is_available()


def test_capability_detection():
    caps = detect_capabilities()
    assert isinstance(caps, AccelCapabilities)
    valid_tiers = {"cuda_rawkernel", "triton", "torch_cuda", "torch_mps", "torch_cpu"}
    assert caps.tier in valid_tiers, caps.tier
    # Detection must be self-consistent: a CUDA tier requires a CUDA device.
    if caps.tier in ("cuda_rawkernel", "triton", "torch_cuda"):
        assert caps.has_cuda
    if not caps.has_cuda:
        assert caps.tier in ("torch_mps", "torch_cpu")
    print(f"detected tier={caps.tier} device={caps.device}")


def test_context_summary_and_default():
    ctx = AccelContext(AccelConfig())
    summary = ctx.summary()
    for field in ("tier=", "device=", "cupy=", "triton="):
        assert field in summary, summary
    assert isinstance(default_context(), AccelContext)
    assert default_context() is default_context(), "default context must be cached"
    assert "tier=" in accel_summary()
    print(f"summary: {summary}")


def test_stage_backends_and_overrides():
    ctx = AccelContext(AccelConfig())
    for stage in ("seeding", "chaining", "extension"):
        backend = ctx.kernel_backend(stage)
        assert backend in ("cuda_rawkernel", "torch"), (stage, backend)
        if not ctx.caps.has_cuda:
            assert backend == "torch", "must not claim a CUDA tier without CUDA"

    # An explicit per-stage override wins over auto-detection.
    forced = AccelContext(AccelConfig(stage_backends={"chaining": "torch"}))
    assert forced.kernel_backend("chaining") == "torch"
    print("stage backends resolve and honour explicit overrides")


def test_amp_dtype_is_honest_about_hardware():
    off = AccelContext(AccelConfig(amp=False))
    assert off.autocast_dtype is None

    ctx = AccelContext(AccelConfig(amp=True, amp_dtype="auto"))
    dtype = ctx.autocast_dtype
    if not ctx.caps.has_cuda and not ctx.caps.supports_bf16:
        assert dtype is None, dtype
    else:
        assert dtype in (torch.bfloat16, torch.float16), dtype

    # fp16 must not be claimed without CUDA.
    fp16 = AccelContext(AccelConfig(amp=True, amp_dtype="fp16"))
    if not torch.cuda.is_available():
        assert fp16.autocast_dtype is None

    # The autocast context must be usable either way.
    with ctx.autocast():
        result = torch.randn(4, 4) @ torch.randn(4, 4)
    assert torch.isfinite(result).all()
    print(f"AMP dtype={dtype}, grad_scaler={type(ctx.grad_scaler()).__name__}")


def test_global_switches_are_idempotent():
    ctx = AccelContext(AccelConfig(apply_global_switches=False))
    ctx.apply_global_switches()
    ctx.apply_global_switches()  # second call must be a no-op, not an error
    print("global switches apply idempotently")


def test_fused_ln_linear_gelu_matches_reference():
    """The fused op must equal LayerNorm -> Linear -> GELU computed separately."""
    torch.manual_seed(0)
    B, L, K, N = 2, 7, 16, 12
    x = torch.randn(B, L, K)
    weight = torch.randn(N, K) * 0.1
    bias = torch.randn(N) * 0.1
    gamma = torch.randn(K) * 0.1 + 1.0
    beta = torch.randn(K) * 0.1

    got = fused_ln_linear_gelu(x, weight, bias, gamma, beta)
    expected = F.gelu(
        F.linear(F.layer_norm(x, (K,), gamma, beta, 1e-5), weight, bias),
        approximate="tanh",
    )
    assert got.shape == (B, L, N), got.shape
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()

    # No bias, and a 2-D input (leading shape is arbitrary).
    got_2d = fused_ln_linear_gelu(torch.randn(5, K), weight, None, gamma, beta)
    assert got_2d.shape == (5, N)

    tier = "triton" if triton_available() else "torch fallback"
    print(f"fused LN+Linear+GELU matches reference via {tier}")


def test_fused_module_trains():
    torch.manual_seed(0)
    module = FusedLNLinearGELU(16, 8)
    x = torch.randn(3, 4, 16)
    out = module(x)
    assert out.shape == (3, 4, 8)
    out.square().mean().backward()
    assert all(p.grad is not None for p in module.parameters())

    # An empty batch must not blow up the kernel-selection path.
    assert module(torch.zeros(0, 16)).shape == (0, 8)
    print("FusedLNLinearGELU forward/backward OK, empty input safe")


def test_cupy_tier_declines_cleanly():
    """Without CuPy the tier must report unavailable, not raise on import."""
    available = kernels_available()
    assert isinstance(available, bool)
    if not available:
        print("CuPy RawKernel tier unavailable (expected off CUDA) - reported cleanly")
        return

    from graphmambaformer.accel.cuda_kernels import banded_sw, chain_dp, kmer_lookup

    assert all(callable(f) for f in (kmer_lookup, chain_dp, banded_sw))
    print("CuPy RawKernel tier compiled and callable")


def test_array_namespace_and_to_numpy():
    assert array_namespace(None) is np
    assert array_namespace("cpu") is np
    namespace = array_namespace("cuda")
    if not torch.cuda.is_available():
        assert namespace is np, "must not hand out cupy without CUDA"

    values = np.arange(5)
    assert to_numpy(values) is values
    assert np.array_equal(to_numpy(torch.arange(5).numpy()), values)
    print(f"array namespace -> {namespace.__name__}, to_numpy round-trips")


def test_cuda_graph_runner_falls_back_to_eager():
    """Off CUDA the runner must transparently behave like the plain function."""
    calls = {"n": 0}

    def double(t: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return t * 2

    runner = CUDAGraphRunner(double)
    x = torch.arange(4, dtype=torch.float32)
    for _ in range(5):
        assert torch.equal(runner(x), x * 2)
    assert calls["n"] == 5 if not HAS_CUDA else calls["n"] >= 1

    disabled = CUDAGraphRunner(double, enabled=False)
    assert torch.equal(disabled(x), x * 2)
    print("CUDAGraphRunner matches eager results (capture skipped off CUDA)")


def test_pipeline_uses_the_detected_backend():
    """The pipeline must adopt the context's device and chaining backend."""
    from graphmambaformer.alignment import build_pipeline
    from graphmambaformer.config import PipelineConfig

    ctx = AccelContext(AccelConfig())
    pipeline = build_pipeline(PipelineConfig(), accel=ctx)
    assert pipeline.device == ctx.caps.device
    assert pipeline.chainer.backend == ctx.kernel_backend("chaining")
    print(f"pipeline adopted device={pipeline.device} "
          f"chaining_backend={pipeline.chainer.backend}")
