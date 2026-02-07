"""Test that HCSA with window=T matches dense attention.

When the window size equals the sequence length, every position can attend
to all previous positions (same as dense causal attention).  The only
remaining difference is the sparse gather path vs. the dense matmul path,
so outputs should match within floating point tolerance.
"""

from __future__ import annotations

import torch

from hcsa.attention_dense import DenseCausalSelfAttention
from hcsa.attention_hcsa import HCSASelfAttention


def test_hcsa_window_T_matches_dense_single_head() -> None:
    """With window=T, 1 head, HCSA output should approximate dense output."""
    torch.manual_seed(42)
    B, T, C, H = 1, 8, 16, 1

    x = torch.randn(B, T, C)

    dense = DenseCausalSelfAttention(C, H, dropout=0.0)
    hcsa = HCSASelfAttention(
        C, H,
        dropout=0.0,
        window=T,  # full window
        landmark_stride=None,  # no landmarks needed
        cycle="random",
        seed=42,
    )

    # Share weights
    hcsa.qkv.weight.data.copy_(dense.qkv.weight.data)
    hcsa.out.weight.data.copy_(dense.out.weight.data)

    dense.eval()
    hcsa.eval()

    with torch.no_grad():
        y_dense = dense(x)
        y_hcsa = hcsa(x)

    # They should be very close (differences from gather-based path)
    diff = (y_dense - y_hcsa).abs().max().item()
    assert diff < 1e-4, f"Dense vs HCSA (window=T) max diff: {diff}"


def test_hcsa_window_T_matches_dense_multi_head() -> None:
    """Multi-head version: window=T HCSA should still approximate dense."""
    torch.manual_seed(42)
    B, T, C, H = 2, 12, 32, 4

    x = torch.randn(B, T, C)

    dense = DenseCausalSelfAttention(C, H, dropout=0.0)
    hcsa = HCSASelfAttention(
        C, H,
        dropout=0.0,
        window=T,
        landmark_stride=None,
        cycle="random",
        seed=42,
    )

    hcsa.qkv.weight.data.copy_(dense.qkv.weight.data)
    hcsa.out.weight.data.copy_(dense.out.weight.data)

    dense.eval()
    hcsa.eval()

    with torch.no_grad():
        y_dense = dense(x)
        y_hcsa = hcsa(x)

    diff = (y_dense - y_hcsa).abs().max().item()
    assert diff < 1e-4, f"Dense vs HCSA (window=T, multi-head) max diff: {diff}"
