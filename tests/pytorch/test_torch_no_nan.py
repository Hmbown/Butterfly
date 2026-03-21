from __future__ import annotations

import torch

from bna.torch.attention_wayfinder_sparse import sparse_row_attention


def test_all_masked_no_nan_and_zero_output(device: torch.device) -> None:
    b, h, t, dh = 2, 1, 4, 8
    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    neigh = torch.full((h, t, 3), -1, device=device, dtype=torch.long)
    edge = torch.zeros((h, t, 3), device=device, dtype=torch.uint8)

    y, w = sparse_row_attention(q, k, v, neigh_idx=neigh, edge_type=edge, return_weights=True)

    assert torch.isfinite(y).all()
    assert torch.isfinite(w).all()
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)
    assert torch.allclose(w, torch.zeros_like(w), atol=1e-7)


def test_zero_degree_no_nan(device: torch.device) -> None:
    b, h, t, dh = 1, 1, 3, 4
    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    neigh = torch.empty((h, t, 0), device=device, dtype=torch.long)
    edge = torch.empty((h, t, 0), device=device, dtype=torch.uint8)

    y, w = sparse_row_attention(q, k, v, neigh_idx=neigh, edge_type=edge, return_weights=True)

    assert torch.isfinite(y).all()
    assert torch.isfinite(w).all()
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)
