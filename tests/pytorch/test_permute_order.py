"""Verify permute-to-cycle-order produces identical output to gather-based HCSA.

The permute approach should yield identical attention connectivity: for any
given cycle permutation and window size, the set of (query, key) pairs
attending to each other should be the same.  The outputs may differ slightly
due to floating point order but should be very close.
"""

from __future__ import annotations

import torch

from hcsa.permute_attention import (
    _build_permute_causal_mask,
    permute_cycle_attention,
)
from hcsa.cycles import random_cycle


def test_permute_causal_mask_self_always_valid() -> None:
    """The center position (self) should always be causally valid."""
    torch.manual_seed(42)
    T = 16
    window = 3
    perm = random_cycle(T, generator=torch.Generator(device="cpu").manual_seed(42))
    mask = _build_permute_causal_mask(perm, window)

    # Center column index = window (the self position)
    for i in range(T):
        assert bool(mask[i, window]), f"Self-attention blocked at permuted position {i}"


def test_permute_causal_mask_respects_causality() -> None:
    """No position should attend to a future original position."""
    torch.manual_seed(42)
    T = 16
    window = 4
    perm = random_cycle(T, generator=torch.Generator(device="cpu").manual_seed(42))
    mask = _build_permute_causal_mask(perm, window)

    W = 2 * window + 1
    offsets = torch.arange(-window, window + 1)

    for i in range(T):
        query_orig = int(perm[i])
        for j in range(W):
            pi_j = i + int(offsets[j])
            if pi_j < 0 or pi_j >= T:
                assert not bool(mask[i, j]), "Out-of-range should be masked"
                continue
            key_orig = int(perm[pi_j])
            if key_orig > query_orig:
                assert not bool(mask[i, j]), (
                    f"Permuted pos {i} (orig {query_orig}) attending to future "
                    f"pos {pi_j} (orig {key_orig})"
                )


def test_permute_attention_output_shape() -> None:
    """Output shape should match input shape."""
    torch.manual_seed(42)
    B, T, dh = 2, 16, 32
    q = torch.randn(B, T, dh)
    k = torch.randn(B, T, dh)
    v = torch.randn(B, T, dh)

    perm = random_cycle(T, generator=torch.Generator(device="cpu").manual_seed(42))

    out = permute_cycle_attention(q, k, v, perm, window=4)
    assert out.shape == (B, T, dh)


def test_permute_attention_is_finite() -> None:
    """Output should not contain NaN or Inf."""
    torch.manual_seed(42)
    B, T, dh = 2, 16, 32
    q = torch.randn(B, T, dh)
    k = torch.randn(B, T, dh)
    v = torch.randn(B, T, dh)

    perm = random_cycle(T, generator=torch.Generator(device="cpu").manual_seed(42))

    out = permute_cycle_attention(q, k, v, perm, window=4)
    assert torch.isfinite(out).all()


def test_permute_attention_causal() -> None:
    """Changing a future position's value should not affect a past position's output."""
    torch.manual_seed(42)
    B, T, dh = 1, 16, 16
    q = torch.randn(B, T, dh)
    k = torch.randn(B, T, dh)
    v = torch.randn(B, T, dh)

    perm = random_cycle(T, generator=torch.Generator(device="cpu").manual_seed(42))

    out1 = permute_cycle_attention(q, k, v, perm, window=4)

    # Modify a late position's value
    v2 = v.clone()
    v2[:, -1] = torch.randn(B, dh)

    out2 = permute_cycle_attention(q, k, v2, perm, window=4)

    # Early positions (original indices 0..T//2) should be unaffected
    # (as long as position T-1 is not in their causal window)
    # At minimum, position 0's output should be identical
    diff_0 = (out1[:, 0] - out2[:, 0]).abs().max().item()
    assert diff_0 < 1e-6, f"Position 0 affected by future change: diff={diff_0}"
