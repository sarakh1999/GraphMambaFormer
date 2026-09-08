"""Windowed (block-local) read-tower self-attention — Figure 1B Layer 2.

Verifies the four properties the tower relies on:
  1. window >= L  ==  exact full bidirectional attention (short reads).
  2. O(n*w) memory: a 40k-base read runs without allocating an (L, L) matrix.
  3. padding / NaN safety: pad keys are ignored and pad rows never NaN.
  4. shifted windows genuinely move the block boundaries (long-read coverage).

Pytest-compatible but self-contained — run directly, or via
``PYTHONPATH=. .venv/bin/python tests/run_all.py``.
"""

import torch

from graphmambaformer.config import AttentionConfig
from graphmambaformer.layers.windowed_attention import WindowedSelfAttention

torch.manual_seed(0)


def _mod(d_model=32, n_heads=4, window=8):
    cfg = AttentionConfig(d_model=d_model, n_heads=n_heads,
                          d_head=d_model // n_heads, window=window, dropout=0.0)
    m = WindowedSelfAttention(cfg).eval()
    return m


def test_full_attention_equivalence():
    """window >= L (and shift=0) must equal a single dense-attention block."""
    m = _mod(window=4)
    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        windowed = m(x, mask=None, shift=0)      # w == L -> full-attention path
        m.window = None
        full = m(x, mask=None, shift=0)          # explicit full attention
    assert torch.allclose(windowed, full, atol=1e-6), (windowed - full).abs().max()
    print("full-attention equivalence (window>=L == full) OK")


def test_block_local_is_actually_local():
    """With window < L a position must NOT see keys outside its block."""
    m = _mod(window=4)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        base = m(x, shift=0)
        # Perturb a position in the *second* block (idx 5). Positions in the
        # first block (0..3) must be unchanged; the full-attention path would
        # have changed them.
        x2 = x.clone()
        x2[0, 5] += 10.0
        pert = m(x2, shift=0)
    first_block_delta = (pert[0, :4] - base[0, :4]).abs().max().item()
    second_block_delta = (pert[0, 4:] - base[0, 4:]).abs().max().item()
    assert first_block_delta < 1e-6, first_block_delta
    assert second_block_delta > 1e-3, second_block_delta
    print("block-locality OK (cross-block leakage = 0)")


def test_shift_moves_boundaries():
    """A half-window shift changes which positions share a block."""
    m = _mod(window=4)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        # Perturbing idx 3 (last of block-0 when shift=0) leaks into idx 4 only
        # when the window is shifted so 3 and 4 share a block.
        x2 = x.clone()
        x2[0, 4] += 10.0
        d0 = (m(x2, shift=0) - m(x, shift=0))[0, 3].abs().max().item()
        d2 = (m(x2, shift=2) - m(x, shift=2))[0, 3].abs().max().item()
    assert d0 < 1e-6, d0            # aligned: 3 and 4 are in different blocks
    assert d2 > 1e-3, d2            # shifted by 2: 3 and 4 now share a block
    print("shifted windows move block boundaries OK")


def test_padding_and_nan_safety():
    """Pad keys are ignored; fully-padded rows are finite and zeroed."""
    m = _mod(window=4)
    x = torch.randn(3, 10, 32)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[0, 6:] = False     # row 0 valid length 6 (a block straddles the edge)
    mask[1, 3:] = False     # row 1 valid length 3
    mask[2, :] = True
    with torch.no_grad():
        out = m(x, mask=mask, shift=2)
    assert torch.isfinite(out).all(), "windowed attention produced NaN/Inf"
    # padded query positions must be exactly zero
    assert out[0, 6:].abs().max() == 0.0
    assert out[1, 3:].abs().max() == 0.0
    # a valid position's output must not depend on padded keys: zero the padded
    # region of the input and confirm valid outputs are unchanged.
    x_clean = x.clone()
    x_clean[0, 6:] = 999.0
    x_clean[1, 3:] = -999.0
    with torch.no_grad():
        out2 = m(x_clean, mask=mask, shift=2)
    assert torch.allclose(out[0, :6], out2[0, :6], atol=1e-5)
    assert torch.allclose(out[1, :3], out2[1, :3], atol=1e-5)
    print("padding / NaN safety OK (pad keys ignored, pad rows zeroed & finite)")


def test_long_read_memory_and_grad():
    """A 40k-base read must run + backprop without an O(L^2) blowup."""
    m = _mod(d_model=32, n_heads=4, window=256).train()
    x = torch.randn(1, 40000, 32, requires_grad=True)
    out = m(x, mask=None, shift=128)
    assert out.shape == (1, 40000, 32)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("long-read (L=40000) forward+backward OK, O(n*w) memory")


if __name__ == "__main__":
    test_full_attention_equivalence()
    test_block_local_is_actually_local()
    test_shift_moves_boundaries()
    test_padding_and_nan_safety()
    test_long_read_memory_and_grad()
    print("\nALL WINDOWED-ATTENTION CHECKS PASSED")
