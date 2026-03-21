from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import sparse_gather_attention
from bna.mlx.graph_abi import MLXGraphABI


def test_all_masked_no_nan_and_zero_output() -> None:
    B, H, T, dh = 2, 1, 4, 8
    q = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    k = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    v = mx.random.normal((B, H, T, dh), dtype=mx.float32)

    neigh = mx.array(np.full((H, T, 3), -1, dtype=np.int32))
    et = mx.array(np.zeros((H, T, 3), dtype=np.uint8))
    graph = MLXGraphABI(neigh_idx=neigh, edge_type=et, meta={})

    y, w = sparse_gather_attention(q, k, v, graph, return_weights=True)
    mx.eval(y, w)

    y_np = np.asarray(y)
    w_np = np.asarray(w)
    assert np.isfinite(y_np).all()
    assert np.isfinite(w_np).all()
    assert np.allclose(y_np, 0.0, atol=1e-7)
    assert np.allclose(w_np, 0.0, atol=1e-7)


def test_zero_degree_no_nan() -> None:
    B, H, T, dh = 1, 1, 3, 4
    q = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    k = mx.random.normal((B, H, T, dh), dtype=mx.float32)
    v = mx.random.normal((B, H, T, dh), dtype=mx.float32)

    neigh = mx.array(np.empty((H, T, 0), dtype=np.int32))
    et = mx.array(np.empty((H, T, 0), dtype=np.uint8))
    graph = MLXGraphABI(neigh_idx=neigh, edge_type=et, meta={})

    y, w = sparse_gather_attention(q, k, v, graph, return_weights=True)
    mx.eval(y, w)
    assert np.isfinite(np.asarray(y)).all()
    assert np.isfinite(np.asarray(w)).all()
    assert np.allclose(np.asarray(y), 0.0, atol=1e-7)
