from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import _swa_stream_attention  # noqa: E402


def _ref_swa(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_win: int) -> np.ndarray:
    B, H, T, dh = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / math.sqrt(dh)
    for t in range(T):
        start = max(0, t - n_win + 1)
        kk = k[:, :, start : t + 1, :]
        vv = v[:, :, start : t + 1, :]
        scores = (q[:, :, t : t + 1, :] @ kk.transpose(0, 1, 3, 2)) * scale
        w = np.exp(scores - scores.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        out[:, :, t : t + 1, :] = w @ vv
    return out


def test_swa_stream_matches_dense_over_window_for_each_query() -> None:
    rng = np.random.default_rng(11)
    B, H, T, dh = 1, 2, 32, 8
    n_win = 8
    q = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    k = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    v = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    o, l, m = _swa_stream_attention(
        mx.array(q), mx.array(k), mx.array(v),
        n_win=n_win, scale=scale,
    )
    mx.eval(o, l, m)

    ref = _ref_swa(q, k, v, n_win=n_win)
    assert np.allclose(np.asarray(o, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)


def test_swa_stream_returns_consistent_l_m_for_merging() -> None:
    """o, l, m must be consistent so that _online_softmax_merge with an empty
    stream (m=-inf, l=0) reproduces o exactly."""
    from bna.mlx.attention import _online_softmax_merge

    rng = np.random.default_rng(13)
    B, H, T, dh = 1, 2, 16, 8
    n_win = 4
    q = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    k = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    v = rng.standard_normal((B, H, T, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    o, l, m = _swa_stream_attention(mx.array(q), mx.array(k), mx.array(v), n_win=n_win, scale=scale)

    empty_o = mx.zeros((B, H, T, dh), dtype=mx.float32)
    empty_l = mx.zeros((B, H, T, 1), dtype=mx.float32)
    empty_m = mx.full((B, H, T, 1), -1e30, dtype=mx.float32)
    out_merged = _online_softmax_merge((o, l, m), (empty_o, empty_l, empty_m))
    mx.eval(out_merged, o)

    assert np.allclose(np.asarray(out_merged, dtype=np.float32), np.asarray(o, dtype=np.float32), atol=3e-4, rtol=3e-4)
