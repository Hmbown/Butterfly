from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import _block_mean_summaries  # noqa: E402
from bna.mlx.compressed_cache import CompressedKVCache  # noqa: E402


def test_compressed_kv_cache_summary_numerics() -> None:
    rng = np.random.default_rng(101)
    block_size = 4
    W = 4
    xk = rng.standard_normal((1, 2, 24, 8), dtype=np.float32)
    xv = rng.standard_normal((1, 2, 24, 8), dtype=np.float32)
    cache = CompressedKVCache(block_size=block_size, local_window_tokens=W)

    for start in range(0, 24, 3):
        cache.update_and_fetch(mx.array(xk[:, :, start : start + 3, :]), mx.array(xv[:, :, start : start + 3, :]))

    tail_k, _tail_v, k_summary, v_summary, tail_start, offset = cache.get_compressed_state()
    mx.eval(tail_k, k_summary, v_summary)
    assert offset == 24
    assert tail_start % block_size == 0
    assert int(tail_k.shape[2]) >= W
    assert int(tail_k.shape[2]) < W + 3 + block_size

    n_summary = int(k_summary.shape[2])
    summary_len = n_summary * block_size
    ref_k = _block_mean_summaries(mx.array(xk[:, :, :summary_len, :]), seq_len=summary_len, block_size=block_size, num_blocks=n_summary)
    ref_v = _block_mean_summaries(mx.array(xv[:, :, :summary_len, :]), seq_len=summary_len, block_size=block_size, num_blocks=n_summary)
    mx.eval(ref_k, ref_v)
    assert np.allclose(np.asarray(k_summary), np.asarray(ref_k), atol=1e-6, rtol=1e-6)
    assert np.allclose(np.asarray(v_summary), np.asarray(ref_v), atol=1e-6, rtol=1e-6)


def test_compressed_kv_cache_state_round_trip() -> None:
    cache = CompressedKVCache(block_size=4, local_window_tokens=4)
    k = mx.random.normal((1, 2, 12, 8))
    v = mx.random.normal((1, 2, 12, 8))
    cache.update_and_fetch(k, v)
    state = cache.state
    meta = cache.meta_state

    restored = CompressedKVCache(block_size=1, local_window_tokens=1)
    restored.state = state
    restored.meta_state = meta
    assert restored.offset == cache.offset
    assert restored.summary_offset == cache.summary_offset
    assert restored.block_size == cache.block_size
    assert restored.local_window_tokens == cache.local_window_tokens
