from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import (
    block_sparse_butterfly_attention,
    block_sparse_butterfly_attention_active,
    build_block_butterfly_layout,
)


def _valid_block_neighbors(layout, block_idx: int) -> list[int]:
    row = np.asarray(layout.block_neighbors[0, block_idx], dtype=np.int32)
    return row[row >= 0].tolist()


def _reference_block_sparse(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    layout,
    *,
    query_positions: np.ndarray | None = None,
) -> np.ndarray:
    B, H, Tq, dh = q.shape
    Tk = int(k.shape[2])
    block_size = int(layout.block_size)
    out = np.zeros((B, H, Tq, dh), dtype=np.float32)

    positions = (
        np.arange(Tq, dtype=np.int32)
        if query_positions is None
        else np.asarray(query_positions, dtype=np.int32)
    )
    for b in range(B):
        for h in range(H):
            for qi, pos in enumerate(positions.tolist()):
                block_idx = int(pos) // block_size
                token_neighbors: list[int] = []
                for neighbor_block in _valid_block_neighbors(layout, block_idx):
                    start = neighbor_block * block_size
                    end = min(Tk, start + block_size)
                    for tok in range(start, end):
                        if tok <= int(pos):
                            token_neighbors.append(tok)

                if not token_neighbors:
                    continue

                scores = np.asarray(
                    [
                        float(np.dot(q[b, h, qi], k[b, h, tok]) / math.sqrt(dh))
                        for tok in token_neighbors
                    ],
                    dtype=np.float32,
                )
                scores = scores - np.max(scores)
                weights = np.exp(scores)
                weights = weights / np.maximum(np.sum(weights), 1e-9)
                acc = np.zeros((dh,), dtype=np.float32)
                for weight, tok in zip(weights.tolist(), token_neighbors):
                    acc += float(weight) * v[b, h, tok]
                out[b, h, qi] = acc
    return out


def test_block_sparse_prefill_matches_reference_small() -> None:
    rng = np.random.default_rng(7)
    B, H, T, dh = 1, 2, 16, 8
    layout = build_block_butterfly_layout(
        seq_len=T,
        block_size=4,
        num_key_value_heads=H,
        num_key_value_groups=1,
        layer_idx=1,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule="xor",
    )

    q_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)

    y, _ = block_sparse_butterfly_attention(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        layout=layout,
    )
    mx.eval(y)

    ref = _reference_block_sparse(q_np, k_np, v_np, layout)
    assert np.allclose(np.asarray(y, dtype=np.float32), ref, atol=2e-4, rtol=2e-4)


def test_block_sparse_active_decode_matches_reference_small() -> None:
    rng = np.random.default_rng(11)
    B, H, Tk, Tq, dh = 1, 2, 16, 3, 8
    query_positions = np.asarray([10, 12, 15], dtype=np.int32)
    layout = build_block_butterfly_layout(
        seq_len=Tk,
        block_size=4,
        num_key_value_heads=H,
        num_key_value_groups=1,
        layer_idx=2,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule="benes",
    )

    q_np = rng.standard_normal((B, H, Tq, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)

    y, _ = block_sparse_butterfly_attention_active(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        layout=layout,
        query_positions=mx.array(query_positions, dtype=mx.int32),
    )
    mx.eval(y)

    ref = _reference_block_sparse(
        q_np,
        k_np,
        v_np,
        layout,
        query_positions=query_positions,
    )
    assert np.allclose(np.asarray(y, dtype=np.float32), ref, atol=2e-4, rtol=2e-4)
