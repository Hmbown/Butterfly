from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hcsa.mlx.attention import sparse_gather_attention
from hcsa.mlx.graph_abi import MLXGraphABI


def test_mlx_sparse_masks_future_neighbors() -> None:
    B, H, T, dh = 1, 1, 6, 4

    q = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    k = mx.random.normal((B, H, T, dh), dtype=mx.float32)

    # Each token attends to [future, self]. Future must be masked by causality.
    neigh = np.full((H, T, 2), -1, dtype=np.int32)
    et = np.zeros((H, T, 2), dtype=np.uint8)
    for i in range(T):
        if i + 1 < T:
            neigh[0, i, 0] = i + 1
            et[0, i, 0] = 1
        neigh[0, i, 1] = i
        et[0, i, 1] = 2

    graph = MLXGraphABI(neigh_idx=mx.array(neigh), edge_type=mx.array(et), meta={})

    v1 = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    v2 = mx.array(np.asarray(v1))
    v2_np = np.asarray(v2)
    v2_np[:, :, 1:, :] += 1000.0
    v2 = mx.array(v2_np, dtype=mx.float32)

    y1, _ = sparse_gather_attention(q, k, v1, graph, return_weights=False)
    y2, _ = sparse_gather_attention(q, k, v2, graph, return_weights=False)
    mx.eval(y1, y2)

    # Position 0 can only attend to self after masking future index 1.
    assert np.allclose(np.asarray(y1)[:, :, 0], np.asarray(y2)[:, :, 0], atol=1e-5)
