from __future__ import annotations

import torch

from hcsa.torch.attention_dense import dense_causal_attention
from hcsa.torch.attention_wayfinder_sparse import sparse_row_attention


def test_sparse_all_past_matches_dense(device: torch.device) -> None:
    b, h, t, dh = 2, 2, 8, 5

    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    neigh = torch.arange(t, device=device, dtype=torch.long).view(1, 1, 1, t)
    neigh = neigh.expand(1, h, t, t).clone()
    edge = torch.full((1, h, t, t), 2, device=device, dtype=torch.uint8)

    y_dense, _ = dense_causal_attention(q, k, v, return_weights=False)
    y_sparse, _ = sparse_row_attention(q, k, v, neigh_idx=neigh, edge_type=edge, return_weights=False)

    assert torch.allclose(y_sparse, y_dense, atol=2e-4, rtol=2e-4)
