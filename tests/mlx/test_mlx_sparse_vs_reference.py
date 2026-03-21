from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import sparse_gather_attention
from bna.mlx.graph_abi import MLXGraphABI


def _reference_sparse(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    neigh: np.ndarray,
) -> np.ndarray:
    B, H, T, dh = q.shape
    D = neigh.shape[-1]
    out = np.zeros((B, H, T, dh), dtype=np.float32)

    for b in range(B):
        for h in range(H):
            for i in range(T):
                valid: list[int] = []
                scores: list[float] = []
                for d in range(D):
                    j = int(neigh[h, i, d])
                    if j < 0 or j > i:
                        continue
                    valid.append(j)
                    scores.append(float(np.dot(q[b, h, i], k[b, h, j]) / math.sqrt(dh)))

                if not valid:
                    continue

                s = np.asarray(scores, dtype=np.float32)
                s = s - np.max(s)
                w = np.exp(s)
                w = w / np.maximum(np.sum(w), 1e-9)

                acc = np.zeros((dh,), dtype=np.float32)
                for ww, j in zip(w.tolist(), valid):
                    acc += float(ww) * v[b, h, j]
                out[b, h, i] = acc

    return out


def test_sparse_matches_reference_small() -> None:
    rng = np.random.default_rng(123)
    B, H, T, D, dh = 2, 2, 6, 4, 5

    q_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)

    neigh = np.full((H, T, D), -1, dtype=np.int32)
    for h in range(H):
        for i in range(T):
            cand = [i, max(0, i - 1), min(T - 1, i + 1), 0]
            neigh[h, i] = np.asarray(cand[:D], dtype=np.int32)

    et = np.where(neigh >= 0, 2, 0).astype(np.uint8)

    graph = MLXGraphABI(neigh_idx=mx.array(neigh), edge_type=mx.array(et), meta={})
    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    y, _ = sparse_gather_attention(q, k, v, graph, return_weights=False)
    mx.eval(y)

    ref = _reference_sparse(q_np, k_np, v_np, neigh)
    y_np = np.asarray(y)

    assert np.allclose(y_np, ref, atol=2e-4, rtol=2e-4)
