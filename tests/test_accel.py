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


def _caps(vendor, cc=None, *, cupy=False, triton=False, mps=False, xpu=False,
          hip=None, name=None):
    """Synthesize a capability snapshot for a GPU this host may not have.

    The dtype and kernel gates are pure functions of vendor + compute
    capability, so they can be verified for every generation from one machine.
    Anything that needs a live driver (``is_bf16_supported``) is not asserted
    here -- see ``test_capability_detection`` for the real host.
    """
    return AccelCapabilities(
        device=torch.device("cpu"),
        has_cuda=vendor in ("nvidia", "amd"),
        has_mps=mps,
        has_cupy=cupy,
        has_triton=triton,
        has_mamba_ssm=False,
        compute_capability=cc,
        device_name=name or f"synthetic-{vendor}",
        vendor=vendor,
        has_xpu=xpu,
        hip_arch=hip,
        device_count=1 if vendor != "cpu" else 0,
    )


def test_dtype_gates_across_every_gpu_generation():
    """TF32/fp16/fp8 must follow the actual hardware, not merely 'is CUDA'."""
    # (label, caps, expect_tf32, expect_fp16, expect_fp8)
    # Datacentre SKUs called out explicitly — A6000/A100/H100/H200/… share one
    # binary; only the CC gates change.
    matrix = [
        ("Pascal sm_61", _caps("nvidia", (6, 1)), False, False, False),
        ("Volta V100 sm_70", _caps("nvidia", (7, 0), name="V100-SXM2"), False, True, False),
        ("Turing T4 sm_75", _caps("nvidia", (7, 5), name="Tesla T4"), False, True, False),
        ("A100 sm_80", _caps("nvidia", (8, 0), name="NVIDIA A100-SXM4-80GB"), True, True, False),
        ("A6000 sm_86", _caps("nvidia", (8, 6), name="NVIDIA RTX A6000"), True, True, False),
        ("L40 sm_89", _caps("nvidia", (8, 9), name="NVIDIA L40"), True, True, True),
        ("H100 sm_90", _caps("nvidia", (9, 0), name="NVIDIA H100"), True, True, True),
        ("H200 sm_90", _caps("nvidia", (9, 0), name="NVIDIA H200"), True, True, True),
        ("B200 sm_100", _caps("nvidia", (10, 0), name="NVIDIA B200"), True, True, True),
        ("AMD gfx90a", _caps("amd", hip="gfx90a"), False, True, False),
        ("Intel XPU", _caps("intel", xpu=True), False, True, False),
        ("Apple MPS", _caps("apple", mps=True), False, True, False),
        ("CPU", _caps("cpu"), False, False, False),
    ]
    for label, caps, tf32, fp16, fp8 in matrix:
        assert caps.supports_tf32 is tf32, f"{label}: tf32 {caps.supports_tf32} != {tf32}"
        assert caps.supports_fp16 is fp16, f"{label}: fp16 {caps.supports_fp16} != {fp16}"
        assert caps.supports_fp8 is fp8, f"{label}: fp8 {caps.supports_fp8} != {fp8}"
        # Ampere+ NVIDIA must claim bf16 via the CC gate (even offline).
        if caps.is_nvidia and caps.compute_capability and caps.compute_capability >= (8, 0):
            assert caps.supports_bf16, f"{label}: expected bf16 on Ampere+"
        print(f"   {label:22s} arch={caps.arch_label:32s} tier={caps.tier:16s} "
              f"tf32={tf32!s:5s} fp16={fp16!s:5s} fp8={fp8} bf16={caps.supports_bf16}")
    print(f"dtype gates correct for all {len(matrix)} device classes")


def test_sku_arch_labels():
    """Common datacentre names must surface in arch_label for logs/doctor."""
    from graphmambaformer.accel import nvidia_arch_label

    cases = [
        ((8, 0), "NVIDIA A100-SXM4-80GB", "Ampere/A100 (sm_80)"),
        ((8, 6), "NVIDIA RTX A6000", "Ampere/A6000 (sm_86)"),
        ((9, 0), "NVIDIA H100 80GB HBM3", "Hopper/H100 (sm_90)"),
        ((9, 0), "NVIDIA H200", "Hopper/H200 (sm_90)"),
        ((8, 9), "NVIDIA L40", "Ada/L40 (sm_89)"),
        ((10, 0), "NVIDIA B200", "Blackwell/B200 (sm_100)"),
    ]
    for cc, name, expect in cases:
        got = nvidia_arch_label(cc, name)
        assert got == expect, (cc, name, got, expect)
        print(f"   {name:28s} -> {got}")
    print(f"SKU labels correct for {len(cases)} datacentre cards")


def test_rocm_is_not_mistaken_for_cuda():
    """PyTorch reports AMD through torch.cuda, so vendor must do the gating."""
    amd = _caps("amd", hip="gfx942", cupy=True, triton=True)
    assert amd.has_cuda, "ROCm builds do surface as torch.cuda"
    assert not amd.is_nvidia
    # NVRTC-compiled raw kernels would not load on AMD, so that tier is refused
    # even though cupy imported; Triton does support ROCm, so it wins instead.
    assert amd.tier == "triton", amd.tier
    assert not amd.supports_tf32, "TF32 is an NVIDIA tensor-core feature"
    assert amd.arch_label == "gfx942"

    nvidia = _caps("nvidia", (9, 0), cupy=True, triton=True, name="NVIDIA H100")
    assert nvidia.tier == "cuda_rawkernel", nvidia.tier
    assert "H100" in nvidia.arch_label
    print("AMD declines NVRTC raw kernels and TF32, falls back to Triton; "
          "NVIDIA still takes the raw-kernel tier")


def test_amp_follows_the_fp16_gate_not_just_cuda():
    """A pre-Volta card must not be handed fp16 autocast."""
    ctx = AccelContext(AccelConfig(amp=True, amp_dtype="fp16"))
    object.__setattr__(ctx, "caps", _caps("nvidia", (6, 1)))
    assert ctx.autocast_dtype is None, "Pascal has no fp16 tensor cores"

    object.__setattr__(ctx, "caps", _caps("nvidia", (7, 5)))
    assert ctx.autocast_dtype is torch.float16
    print("fp16 AMP refused on Pascal sm_61, granted on Turing sm_75")


def test_capability_detection():
    caps = detect_capabilities()
    assert isinstance(caps, AccelCapabilities)
    valid_tiers = {
        "cuda_rawkernel", "triton", "torch_cuda", "torch_xpu", "torch_mps", "torch_cpu",
    }
    assert caps.tier in valid_tiers, caps.tier
    # Detection must be self-consistent: a CUDA tier requires a CUDA device.
    if caps.tier in ("cuda_rawkernel", "triton", "torch_cuda"):
        assert caps.has_cuda
        assert caps.device.type == "cuda"
    if not caps.has_cuda:
        assert caps.tier in ("torch_xpu", "torch_mps", "torch_cpu")
    # ``auto`` / empty must resolve the same as an unset device.
    assert detect_capabilities("auto").device.type == caps.device.type
    print(f"detected tier={caps.tier} device={caps.device} arch={caps.arch_label}")


def test_list_visible_gpus_is_safe_off_cuda():
    from graphmambaformer.accel import list_visible_gpus

    gpus = list_visible_gpus()
    assert isinstance(gpus, list)
    for g in gpus:
        assert "index" in g and "name" in g and "vendor" in g
    print(f"visible GPUs: {len(gpus)}"
          + (f" ({', '.join(g['name'] for g in gpus)})" if gpus else ""))


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

    from graphmambaformer.accel.cuda_kernels import (
        banded_sw,
        chain_dp,
        kmer_lookup,
        wfa_distance,
    )

    assert all(callable(f) for f in (kmer_lookup, chain_dp, banded_sw, wfa_distance))
    print("CuPy RawKernel tier compiled and callable")


def test_cuda_rawkernels_match_portable_references():
    """GPU CI cross-checks SW, WFA, and fractional-weight chaining semantics."""
    if not (torch.cuda.is_available() and kernels_available()):
        print("CUDA RawKernel equivalence skipped (CUDA/CuPy unavailable)")
        return

    from graphmambaformer.accel.cuda_kernels import (
        banded_sw,
        chain_dp,
        wfa_distance,
    )
    from graphmambaformer.alignment.chaining import chain_dp_numpy
    from graphmambaformer.alignment.extension import banded_affine_sw_batch
    from graphmambaformer.alignment.seeding import encode_bases
    from graphmambaformer.config import ChainingConfig, ExtensionConfig

    device = torch.device("cuda")
    extension_cfg = ExtensionConfig()
    query = torch.as_tensor(encode_bases("ACGT"), dtype=torch.int8, device=device)[None]
    target = torch.as_tensor(encode_bases("TTACGTAA"), dtype=torch.int8, device=device)[None]
    lengths_q = torch.tensor([4], device=device)
    lengths_t = torch.tensor([8], device=device)
    offset = torch.tensor([2], device=device)
    raw = banded_sw(
        query,
        target,
        lengths_q,
        lengths_t,
        half_band=2,
        match_score=extension_cfg.match_score,
        mismatch_penalty=extension_cfg.mismatch_penalty,
        gap_open=extension_cfg.gap_open,
        gap_extend=extension_cfg.gap_extend,
        band_offset=offset,
    )
    portable = banded_affine_sw_batch(
        query,
        target,
        lengths_q,
        lengths_t,
        extension_cfg,
        half_band=2,
        band_offset=offset,
    )
    assert torch.allclose(raw[0], portable.score, atol=1e-3)
    assert torch.equal(raw[1], portable.query_end)
    assert torch.equal(raw[2], portable.target_end)

    wfa_q = torch.as_tensor(encode_bases("ACGTAC"), dtype=torch.int8, device=device)[None]
    wfa_t = torch.as_tensor(encode_bases("ACGTTAC"), dtype=torch.int8, device=device)[None]
    got_distance = wfa_distance(
        wfa_q,
        wfa_t,
        torch.tensor([6], device=device),
        torch.tensor([7], device=device),
        max_distance=4,
    )
    assert int(got_distance[0]) == 1

    chain_cfg = ChainingConfig()
    read_end = np.array([10, 20, 30], dtype=np.int64)
    ref_end = np.array([10, 20, 31], dtype=np.int64)
    weights = np.array([10.5, 9.25, 8.75], dtype=np.float64)
    expected, expected_parent = chain_dp_numpy(
        read_end, ref_end, weights, chain_cfg
    )
    anchors = torch.tensor(
        [[[10, 10, 0], [20, 20, 0], [30, 31, 0]]],
        dtype=torch.int32,
        device=device,
    )
    got, got_parent = chain_dp(
        anchors,
        torch.tensor([3], dtype=torch.int32, device=device),
        lookback=chain_cfg.max_lookback,
        max_gap=chain_cfg.max_gap,
        gap_open=chain_cfg.gap_open,
        gap_extend=chain_cfg.gap_extend,
        log_coeff=chain_cfg.log_coeff,
        weights=torch.tensor(weights, dtype=torch.float32, device=device)[None],
    )
    assert np.allclose(got[0].cpu().numpy(), expected, atol=1e-4)
    assert np.array_equal(got_parent[0].cpu().numpy(), expected_parent)
    print("CUDA SW/WFA/chaining match portable references")


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
