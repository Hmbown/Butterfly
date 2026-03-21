from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bna.torch.attention_wayfinder_sparse import sparse_row_attention


# MLX is optional for this test file. We always verify against a shared numpy reference.
try:
    import mlx.core as mx  # type: ignore

    from bna.mlx.attention import sparse_gather_attention
    from bna.mlx.graph_abi import MLXGraphABI

    HAS_MLX = True
except Exception:  # pragma: no cover - environment dependent
    HAS_MLX = False


def _reference_sparse(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    neigh: np.ndarray,
) -> np.ndarray:
    b, h, t, dh = q.shape
    d = neigh.shape[-1]
    out = np.zeros((b, h, t, dh), dtype=np.float32)

    for bi in range(b):
        for hi in range(h):
            for i in range(t):
                valid: list[int] = []
                scores: list[float] = []
                for di in range(d):
                    j = int(neigh[hi, i, di])
                    if j < 0 or j > i:
                        continue
                    valid.append(j)
                    scores.append(float(np.dot(q[bi, hi, i], k[bi, hi, j]) / math.sqrt(float(dh))))

                if not valid:
                    continue

                s = np.asarray(scores, dtype=np.float32)
                s = s - np.max(s)
                w = np.exp(s)
                w = w / np.maximum(np.sum(w), 1e-9)

                acc = np.zeros((dh,), dtype=np.float32)
                for ww, j in zip(w.tolist(), valid):
                    acc += float(ww) * v[bi, hi, j]
                out[bi, hi, i] = acc

    return out


def test_torch_sparse_matches_shared_reference_small(device: torch.device) -> None:
    rng = np.random.default_rng(123)
    b, h, t, d, dh = 2, 2, 6, 4, 5

    q_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)
    k_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)
    v_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)

    neigh = np.full((h, t, d), -1, dtype=np.int32)
    for hi in range(h):
        for i in range(t):
            cand = [i, max(0, i - 1), min(t - 1, i + 1), 0]
            neigh[hi, i] = np.asarray(cand[:d], dtype=np.int32)

    edge = np.where(neigh >= 0, 2, 0).astype(np.uint8)

    q = torch.tensor(q_np, device=device)
    k = torch.tensor(k_np, device=device)
    v = torch.tensor(v_np, device=device)

    y_torch, _ = sparse_row_attention(
        q,
        k,
        v,
        neigh_idx=torch.tensor(neigh, device=device),
        edge_type=torch.tensor(edge, device=device),
        return_weights=False,
    )

    ref = _reference_sparse(q_np, k_np, v_np, neigh)
    assert np.allclose(y_torch.detach().cpu().numpy(), ref, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not HAS_MLX, reason="MLX runtime is not available")
def test_torch_sparse_matches_mlx_small_cpu() -> None:
    rng = np.random.default_rng(7)
    b, h, t, d, dh = 1, 2, 7, 5, 4

    q_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)
    k_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)
    v_np = rng.standard_normal((b, h, t, dh), dtype=np.float32)

    neigh = np.full((h, t, d), -1, dtype=np.int32)
    for hi in range(h):
        for i in range(t):
            vals = [i, max(0, i - 1), (i + 1) % t, 0, 2]
            neigh[hi, i] = np.asarray(vals[:d], dtype=np.int32)
    edge = np.where(neigh >= 0, 2, 0).astype(np.uint8)

    q_t = torch.tensor(q_np)
    k_t = torch.tensor(k_np)
    v_t = torch.tensor(v_np)

    y_torch, _ = sparse_row_attention(
        q_t,
        k_t,
        v_t,
        neigh_idx=torch.tensor(neigh),
        edge_type=torch.tensor(edge),
        return_weights=False,
    )

    graph = MLXGraphABI(neigh_idx=mx.array(neigh), edge_type=mx.array(edge), meta={})
    y_mlx, _ = sparse_gather_attention(mx.array(q_np), mx.array(k_np), mx.array(v_np), graph)
    mx.eval(y_mlx)

    assert np.allclose(y_torch.detach().cpu().numpy(), np.asarray(y_mlx), atol=2e-4, rtol=2e-4)
