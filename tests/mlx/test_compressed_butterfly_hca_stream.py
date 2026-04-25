from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import _compressed_stream_attention  # noqa: E402


def _ref_hca(q: np.ndarray, k_summary: np.ndarray, v_summary: np.ndarray, block_size: int) -> np.ndarray:
    """HCA-style reference: each query at position t in block b attends to all
    block summaries strictly earlier than block b. Queries inside block 0 attend
    to nothing (output remains zeros)."""
    B, H, T, dh = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / math.sqrt(dh)
    for t in range(T):
        q_block = t // block_size
        if q_block <= 0:
            continue
        kk = k_summary[:, :, :q_block, :]
        vv = v_summary[:, :, :q_block, :]
        scores = (q[:, :, t : t + 1, :] @ kk.transpose(0, 1, 3, 2)) * scale
        w = np.exp(scores - scores.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        out[:, :, t : t + 1, :] = w @ vv
    return out


def test_hca_stream_dense_over_compressed_blocks() -> None:
    rng = np.random.default_rng(13)
    B, H, T, dh = 1, 2, 32, 8
    block_size = 8
    num_blocks = T // block_size
    q = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    k_summary = rng.standard_normal((B, H, num_blocks, dh)).astype(np.float32)
    v_summary = rng.standard_normal((B, H, num_blocks, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    o, l, m = _compressed_stream_attention(
        mx.array(q), mx.array(k_summary), mx.array(v_summary),
        block_size=block_size, scale=scale, routed_indices=None,
    )
    mx.eval(o, l, m)

    ref = _ref_hca(q, k_summary, v_summary, block_size=block_size)
    # For tokens in block 0 (no causal-prior compressed blocks), the stream's
    # contribution must be zeroed out via m=-inf, l=0 sentinels.
    valid_query = (np.arange(T) // block_size) > 0
    o_np = np.asarray(o, dtype=np.float32)
    l_np = np.asarray(l, dtype=np.float32)
    m_np = np.asarray(m, dtype=np.float32)
    # Block-0 tokens: l should be 0 (no contribution), m should be very negative.
    assert np.all(l_np[:, :, ~valid_query, 0] == 0.0)
    assert np.all(m_np[:, :, ~valid_query, 0] < -1e20)
    # Valid tokens: output matches reference.
    assert np.allclose(o_np[:, :, valid_query, :], ref[:, :, valid_query, :], atol=3e-4, rtol=3e-4)


def test_hca_stream_with_routing_via_indices() -> None:
    """When routed_indices is provided, only routed neighbor blocks are attended."""
    rng = np.random.default_rng(17)
    B, H, T, dh = 1, 2, 16, 8
    block_size = 4
    num_blocks = T // block_size  # 4

    q = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    k_summary = rng.standard_normal((B, H, num_blocks, dh)).astype(np.float32)
    v_summary = rng.standard_normal((B, H, num_blocks, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    # routed_indices[h, q_block, slot] -> compressed-block index, or -1 = invalid.
    # For this test, query block b > 0 attends only to block (b - 1).
    routed = np.full((H, num_blocks, 1), -1, dtype=np.int32)
    for b in range(1, num_blocks):
        routed[:, b, 0] = b - 1

    o, l, m = _compressed_stream_attention(
        mx.array(q), mx.array(k_summary), mx.array(v_summary),
        block_size=block_size, scale=scale,
        routed_indices=mx.array(routed),
    )
    mx.eval(o, l, m)

    # Reference: each query at t in block b attends to summary at index b-1 (single key).
    ref = np.zeros_like(q)
    for t in range(T):
        b = t // block_size
        if b == 0:
            continue
        kk = k_summary[:, :, b - 1 : b, :]
        vv = v_summary[:, :, b - 1 : b, :]
        scores = (q[:, :, t : t + 1, :] @ kk.transpose(0, 1, 3, 2)) * scale
        w = np.exp(scores - scores.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        ref[:, :, t : t + 1, :] = w @ vv

    valid_query = (np.arange(T) // block_size) > 0
    o_np = np.asarray(o, dtype=np.float32)
    assert np.allclose(o_np[:, :, valid_query, :], ref[:, :, valid_query, :], atol=3e-4, rtol=3e-4)
