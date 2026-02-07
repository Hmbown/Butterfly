from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hcsa.mlx.attention import WayfinderAttentionMLX


def _make_attn(
    *,
    window_drop: float = 0.0,
    path: str = "sparse",
    n_embd: int = 32,
    n_heads: int = 2,
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
) -> WayfinderAttentionMLX:
    return WayfinderAttentionMLX(
        n_embd,
        n_heads,
        window=window,
        landmark_stride=landmark_stride,
        strategy="random",
        path=path,
        seed=seed,
        dropout=0.0,
        window_drop=window_drop,
    )


def test_zero_drop_matches_baseline() -> None:
    """Window drop=0 in training should produce same output as eval mode."""
    mx.random.seed(123)
    x = mx.random.normal((2, 16, 32), dtype=mx.float16)

    attn = _make_attn(window_drop=0.0, seed=42)

    # Eval mode
    attn.eval()
    y_eval = attn(x)
    mx.eval(y_eval)

    # Training mode with drop=0 should be the same
    attn.train()
    y_train = attn(x)
    mx.eval(y_train)

    np.testing.assert_allclose(
        np.asarray(y_eval, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        atol=1e-5,
    )


def test_full_drop_preserves_cycle_and_self() -> None:
    """With drop=1.0, output should still be non-zero (cycle + self edges preserved)."""
    mx.random.seed(456)
    attn = _make_attn(window_drop=1.0, seed=42)
    attn.train()

    T = 16
    x = mx.random.normal((1, T, 32), dtype=mx.float16)
    y = attn(x)
    mx.eval(y)

    y_np = np.asarray(y, dtype=np.float32)
    # Should not be all zeros — cycle and self edges remain
    assert np.any(np.abs(y_np) > 1e-6), "Output is all zeros even with cycle/self preserved"


def test_drop_maintains_causality() -> None:
    """No future information should leak even with window drop."""
    mx.random.seed(789)
    attn = _make_attn(window_drop=0.5, seed=42)
    attn.train()

    T = 16
    x = mx.random.normal((1, T, 32), dtype=mx.float16)
    y, dbg = attn(x, return_debug=True)
    mx.eval(y)

    w = dbg["attn_weights"]
    neigh_idx = dbg["neigh_idx"]
    mx.eval(w, neigh_idx)

    w_np = np.asarray(w, dtype=np.float32)
    ni_np = np.asarray(neigh_idx, dtype=np.int32)

    H = w_np.shape[1]
    for h in range(H):
        for i in range(T):
            for d in range(ni_np.shape[-1]):
                j = ni_np[h, i, d]
                if j > i:
                    assert w_np[0, h, i, d] == 0.0, (
                        f"Causal violation with window drop: h={h} i={i} j={j}"
                    )


def test_drop_disabled_at_eval() -> None:
    """Eval mode should have no randomness from window drop."""
    mx.random.seed(111)
    attn = _make_attn(window_drop=0.5, seed=42)
    attn.eval()

    x = mx.random.normal((2, 16, 32), dtype=mx.float16)

    y1 = attn(x)
    mx.eval(y1)
    y2 = attn(x)
    mx.eval(y2)

    # In eval mode, both passes should be identical (deterministic)
    np.testing.assert_allclose(
        np.asarray(y1, dtype=np.float32),
        np.asarray(y2, dtype=np.float32),
        atol=1e-5,
    )
