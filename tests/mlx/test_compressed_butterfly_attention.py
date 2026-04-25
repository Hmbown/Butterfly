from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import (  # noqa: E402
    build_block_butterfly_layout,
    compressed_butterfly_attention,
    compressed_butterfly_attention_active,
)


def _valid_block_neighbors(layout, block_idx: int) -> list[int]:
    row = np.asarray(layout.block_neighbors[0, block_idx], dtype=np.int32)
    return row[row >= 0].tolist()


def _block_summaries(x: np.ndarray, *, block_size: int) -> np.ndarray:
    B, H, T, dh = x.shape
    num_blocks = math.ceil(T / block_size)
    out = np.zeros((B, H, num_blocks, dh), dtype=np.float32)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(T, start + block_size)
        out[:, :, block_idx, :] = x[:, :, start:end, :].mean(axis=2)
    return out


def _reference_compressed(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    layout,
    *,
    local_window_tokens: int,
    query_positions: np.ndarray | None = None,
) -> np.ndarray:
    B, H, Tq, dh = q.shape
    Tk = int(k.shape[2])
    block_size = int(layout.block_size)
    k_summary = _block_summaries(k, block_size=block_size)
    v_summary = _block_summaries(v, block_size=block_size)
    out = np.zeros((B, H, Tq, dh), dtype=np.float32)
    positions = (
        np.arange(Tq, dtype=np.int32)
        if query_positions is None
        else np.asarray(query_positions, dtype=np.int32)
    )

    for b in range(B):
        for h in range(H):
            for qi, pos_t in enumerate(positions.tolist()):
                pos = int(pos_t)
                block_idx = pos // block_size
                block_start = block_idx * block_size
                summary_cutoff = (
                    max(0, block_start - local_window_tokens + 1)
                    if query_positions is None
                    else max(0, pos - local_window_tokens + 1)
                )
                local_start = max(0, pos - local_window_tokens + 1)
                raw_tokens = list(range(local_start, min(Tk, pos + 1)))
                keys = [k[b, h, tok] for tok in raw_tokens]
                vals = [v[b, h, tok] for tok in raw_tokens]

                for neighbor in _valid_block_neighbors(layout, block_idx):
                    if neighbor >= block_idx:
                        continue
                    if (neighbor + 1) * block_size > summary_cutoff:
                        continue
                    keys.append(k_summary[b, h, neighbor])
                    vals.append(v_summary[b, h, neighbor])

                if not keys:
                    continue
                keys_np = np.stack(keys, axis=0)
                vals_np = np.stack(vals, axis=0)
                scores = (keys_np @ q[b, h, qi]) / math.sqrt(dh)
                scores = scores - np.max(scores)
                weights = np.exp(scores)
                weights = weights / np.maximum(weights.sum(), 1e-9)
                out[b, h, qi] = weights @ vals_np
    return out


def test_compressed_butterfly_prefill_matches_reference_small() -> None:
    rng = np.random.default_rng(17)
    B, H, T, dh = 1, 2, 18, 8
    layout = build_block_butterfly_layout(
        seq_len=T,
        block_size=4,
        num_key_value_heads=H,
        num_key_value_groups=1,
        layer_idx=2,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule="causal_shift",
    )
    q_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)

    y, _ = compressed_butterfly_attention(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        layout=layout,
        local_window_tokens=4,
    )
    mx.eval(y)

    ref = _reference_compressed(
        q_np,
        k_np,
        v_np,
        layout,
        local_window_tokens=4,
    )
    assert np.allclose(np.asarray(y, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)


def test_compressed_butterfly_active_matches_reference_small() -> None:
    rng = np.random.default_rng(19)
    B, H, Tk, Tq, dh = 1, 2, 18, 3, 8
    query_positions = np.asarray([9, 13, 17], dtype=np.int32)
    layout = build_block_butterfly_layout(
        seq_len=Tk,
        block_size=4,
        num_key_value_heads=H,
        num_key_value_groups=1,
        layer_idx=3,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule="causal_shift",
    )
    q_np = rng.standard_normal((B, H, Tq, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)

    y, _ = compressed_butterfly_attention_active(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        layout=layout,
        query_positions=mx.array(query_positions, dtype=mx.int32),
        local_window_tokens=4,
    )
    mx.eval(y)

    ref = _reference_compressed(
        q_np,
        k_np,
        v_np,
        layout,
        local_window_tokens=4,
        query_positions=query_positions,
    )
    assert np.allclose(np.asarray(y, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)
