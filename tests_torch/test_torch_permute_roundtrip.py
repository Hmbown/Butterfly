from __future__ import annotations

import torch

from hcsa.torch.attention_dense import dense_causal_attention
from hcsa.torch.attention_hha_permute import hha_permute_window_attention


def test_permute_inverse_roundtrip(device: torch.device) -> None:
    b, t, dh = 2, 17, 5
    x = torch.randn(b, t, dh, device=device, dtype=torch.float32)
    perm = torch.randperm(t, device=device)
    inv = torch.argsort(perm)

    x_pi = x[:, perm]
    x_back = x_pi[:, inv]

    assert torch.equal(x_back, x)


def test_full_window_permute_matches_dense(device: torch.device) -> None:
    b, h, t, dh = 2, 2, 12, 6

    q = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dh, device=device, dtype=torch.float32)

    perms = [torch.randperm(t, device=device).tolist() for _ in range(h)]

    neigh = torch.arange(t, device=device, dtype=torch.long).view(1, 1, 1, t).expand(h, -1, -1, -1)
    neigh = neigh[:, 0].contiguous()  # [H,T,T]
    edge = torch.full((h, t, t), 2, device=device, dtype=torch.uint8)

    y_perm, _w, _pms, _ams = hha_permute_window_attention(
        q,
        k,
        v,
        window=t - 1,
        neigh_idx=neigh,
        edge_type=edge,
        graph_meta={"cycle_perms": perms},
        return_weights=False,
    )
    y_dense, _ = dense_causal_attention(q, k, v, return_weights=False)

    assert torch.allclose(y_perm, y_dense, atol=2e-4, rtol=2e-4)
