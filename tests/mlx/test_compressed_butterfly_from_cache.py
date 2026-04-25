from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import (  # noqa: E402
    build_block_butterfly_layout,
    compressed_butterfly_attention_active,
    compressed_butterfly_attention_from_cache,
)
from bna.mlx.compressed_cache import CompressedKVCache  # noqa: E402


def test_compressed_butterfly_from_cache_matches_active_reference() -> None:
    rng = np.random.default_rng(202)
    B, H, Tk, Tq, dh = 1, 2, 24, 3, 8
    block_size = 4
    local_window = 4
    q_np = rng.standard_normal((B, H, Tq, dh), dtype=np.float32)
    k_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
    v_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
    query_positions = np.asarray([21, 22, 23], dtype=np.int32)
    layout = build_block_butterfly_layout(
        seq_len=Tk,
        block_size=block_size,
        num_key_value_heads=H,
        num_key_value_groups=1,
        layer_idx=3,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule="causal_shift",
    )

    ref, _ = compressed_butterfly_attention_active(
        mx.array(q_np),
        mx.array(k_np),
        mx.array(v_np),
        layout=layout,
        query_positions=mx.array(query_positions, dtype=mx.int32),
        local_window_tokens=local_window,
        return_weights=False,
    )

    cache = CompressedKVCache(block_size=block_size, local_window_tokens=local_window)
    for start in range(0, Tk, 3):
        cache.update_and_fetch(mx.array(k_np[:, :, start : start + 3, :]), mx.array(v_np[:, :, start : start + 3, :]))
    tail_k, tail_v, k_summary, v_summary, tail_start, offset = cache.get_compressed_state()
    got, _ = compressed_butterfly_attention_from_cache(
        mx.array(q_np),
        tail_k,
        tail_v,
        k_summary,
        v_summary,
        layout=layout,
        query_positions=mx.array(query_positions, dtype=mx.int32),
        local_window_tokens=local_window,
        tail_start=tail_start,
        kv_len=offset,
        return_weights=False,
    )
    mx.eval(ref, got)
    assert np.allclose(np.asarray(got, dtype=np.float32), np.asarray(ref, dtype=np.float32), atol=3e-4, rtol=3e-4)

    got_chunked, _ = compressed_butterfly_attention_from_cache(
        mx.array(q_np),
        tail_k,
        tail_v,
        k_summary,
        v_summary,
        layout=layout,
        query_positions=mx.array(query_positions, dtype=mx.int32),
        local_window_tokens=local_window,
        tail_start=tail_start,
        kv_len=offset,
        return_weights=False,
        query_chunk_size=1,
    )
    mx.eval(got_chunked)
    assert np.allclose(np.asarray(got_chunked, dtype=np.float32), np.asarray(ref, dtype=np.float32), atol=3e-4, rtol=3e-4)
