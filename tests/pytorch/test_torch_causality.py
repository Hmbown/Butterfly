from __future__ import annotations

import torch

from hcsa.torch.attention_wayfinder_permute import wayfinder_permute_window_attention
from hcsa.torch.attention_wayfinder_sparse import sparse_row_attention


def test_torch_sparse_masks_future_neighbors(device: torch.device) -> None:
    b, h, t, dh = 1, 1, 6, 4
    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    neigh = torch.full((h, t, 2), -1, device=device, dtype=torch.long)
    edge = torch.zeros((h, t, 2), device=device, dtype=torch.uint8)
    for i in range(t):
        if i + 1 < t:
            neigh[0, i, 0] = i + 1  # future edge (must be masked)
            edge[0, i, 0] = 1
        neigh[0, i, 1] = i
        edge[0, i, 1] = 2

    v1 = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v2 = v1.clone()
    v2[:, :, 1:, :] = v2[:, :, 1:, :] + 1000.0

    y1, _ = sparse_row_attention(q, k, v1, neigh_idx=neigh, edge_type=edge)
    y2, _ = sparse_row_attention(q, k, v2, neigh_idx=neigh, edge_type=edge)

    # Token 0 may only attend to itself once future edge is masked.
    assert torch.allclose(y1[:, :, 0], y2[:, :, 0], atol=1e-5)


def test_torch_permute_enforces_original_causality(device: torch.device) -> None:
    b, h, t, dh = 1, 1, 8, 6

    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    perm = torch.randperm(t).tolist()

    neigh = torch.full((h, t, t), -1, device=device, dtype=torch.long)
    edge = torch.full((h, t, t), 2, device=device, dtype=torch.uint8)
    for i in range(t):
        neigh[0, i] = torch.arange(t, device=device)

    y1, _w1, _pm1, _am1 = wayfinder_permute_window_attention(
        q,
        k,
        v,
        window=t - 1,
        neigh_idx=neigh,
        edge_type=edge,
        graph_meta={"cycle_perms": [perm]},
        return_weights=False,
    )

    # Perturb future original positions only; output at position 0 must stay unchanged.
    v2 = v.clone()
    v2[:, :, 1:, :] += 1000.0
    y2, _w2, _pm2, _am2 = wayfinder_permute_window_attention(
        q,
        k,
        v2,
        window=t - 1,
        neigh_idx=neigh,
        edge_type=edge,
        graph_meta={"cycle_perms": [perm]},
        return_weights=False,
    )

    assert torch.allclose(y1[:, :, 0], y2[:, :, 0], atol=1e-5)
