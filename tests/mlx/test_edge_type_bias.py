from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import ButterflyAttentionMLX
from bna.mlx.metrics import edge_utilization_by_type


def _make_attn(
    *,
    edge_bias: bool = False,
    path: str = "sparse",
    n_embd: int = 32,
    n_heads: int = 2,
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
) -> ButterflyAttentionMLX:
    return ButterflyAttentionMLX(
        n_embd,
        n_heads,
        window=window,
        landmark_stride=landmark_stride,
        strategy="random",
        path=path,
        seed=seed,
        dropout=0.0,
        edge_bias=edge_bias,
    )


def test_zero_bias_matches_no_bias() -> None:
    """Zero-initialized bias should produce same output as no bias."""
    mx.random.seed(123)
    x = mx.random.normal((2, 16, 32), dtype=mx.float16)

    # Use a single module: run once with bias=None, then switch to zeros
    # (don't invalidate cache so graph stays the same)
    attn = _make_attn(edge_bias=False, seed=42)

    y_no = attn(x)
    mx.eval(y_no)

    # Enable zero bias without cache invalidation — same graph used
    attn.edge_type_bias = mx.zeros((4,))

    y_yes = attn(x)
    mx.eval(y_yes)

    np.testing.assert_allclose(
        np.asarray(y_no, dtype=np.float32),
        np.asarray(y_yes, dtype=np.float32),
        atol=1e-4,
    )


def test_bias_preserves_causality() -> None:
    """Even with large edge-type bias, future tokens should not be attended."""
    attn = _make_attn(edge_bias=True)
    # Set large CYCLE bias
    attn.edge_type_bias = mx.array([5.0, 0.0, 0.0, 0.0])  # CYCLE, WINDOW, LANDMARK, REWIRE

    T = 16
    x = mx.random.normal((1, T, 32), dtype=mx.float16)
    y, dbg = attn(x, return_debug=True)
    mx.eval(y)

    w = dbg["attn_weights"]
    neigh_idx = dbg["neigh_idx"]
    mx.eval(w, neigh_idx)

    w_np = np.asarray(w, dtype=np.float32)
    ni_np = np.asarray(neigh_idx, dtype=np.int32)

    # Check: for each token i, any neighbor j > i should have zero weight
    H = w_np.shape[1]
    for h in range(H):
        for i in range(T):
            for d in range(ni_np.shape[-1]):
                j = ni_np[h, i, d]
                if j > i:
                    assert w_np[0, h, i, d] == 0.0, (
                        f"Causal violation: h={h} i={i} j={j} w={w_np[0, h, i, d]}"
                    )


def test_bias_shifts_attention_mass() -> None:
    """Large CYCLE bias should increase cycle edge utilization."""
    T = 32
    x = mx.random.normal((2, T, 32), dtype=mx.float16)

    # Baseline (no bias)
    attn_base = _make_attn(edge_bias=True, seed=42)
    _, dbg_base = attn_base(x, return_debug=True)
    mx.eval(dbg_base["attn_weights"], dbg_base["edge_type"])
    w_base = np.asarray(dbg_base["attn_weights"], dtype=np.float32)
    et_base = np.asarray(dbg_base["edge_type"], dtype=np.uint8)
    util_base = edge_utilization_by_type(w_base, et_base)

    # Biased (large CYCLE bias)
    attn_biased = _make_attn(edge_bias=True, seed=42)
    attn_biased.edge_type_bias = mx.array([5.0, 0.0, 0.0, 0.0])
    _, dbg_biased = attn_biased(x, return_debug=True)
    mx.eval(dbg_biased["attn_weights"], dbg_biased["edge_type"])
    w_biased = np.asarray(dbg_biased["attn_weights"], dtype=np.float32)
    et_biased = np.asarray(dbg_biased["edge_type"], dtype=np.uint8)
    util_biased = edge_utilization_by_type(w_biased, et_biased)

    assert util_biased["cycle"] > util_base["cycle"], (
        f"Cycle util did not increase: base={util_base['cycle']:.4f} biased={util_biased['cycle']:.4f}"
    )
