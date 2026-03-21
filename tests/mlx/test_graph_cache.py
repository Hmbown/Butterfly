from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import WayfinderAttentionMLX, _GRAPH_CACHE_STORE
from bna.mlx.graph_abi import causal_neighbor_mask, safe_neighbor_idx


def _make_attn(
    *,
    path: str = "sparse",
    strategy: str = "random",
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
        strategy=strategy,
        path=path,
        seed=seed,
        dropout=0.0,
    )


def _get_cache(attn):
    return _GRAPH_CACHE_STORE.get(id(attn))


def test_cache_hit_returns_same_output() -> None:
    """Second pass with same T should use cached graph (graph_build_ms ~ 0)."""
    attn = _make_attn()
    x = mx.random.normal((1, 16, 32), dtype=mx.float16)

    # First pass — builds cache
    y1, d1 = attn(x, return_debug=True)
    mx.eval(y1)
    gb1 = float(d1["profile"]["graph_build_ms"])

    # Second pass — cache hit
    y2, d2 = attn(x, return_debug=True)
    mx.eval(y2)
    gb2 = d2["profile"]["graph_build_ms"]

    # Cache hit should be much faster
    assert gb2 < gb1 * 0.5 or gb2 < 1.0, (
        f"Cache miss? gb1={gb1:.1f} gb2={gb2:.1f}"
    )

    # Outputs should be identical (same graph, same input)
    np.testing.assert_allclose(
        np.asarray(y1, dtype=np.float32),
        np.asarray(y2, dtype=np.float32),
        atol=1e-5,
    )


def test_cache_invalidation_on_T_change() -> None:
    """Changing T should trigger rebuild."""
    attn = _make_attn()

    x16 = mx.random.normal((1, 16, 32), dtype=mx.float16)
    y1 = attn(x16)
    mx.eval(y1)
    cache1 = _get_cache(attn)
    assert cache1 is not None
    key1 = cache1.cache_key

    x24 = mx.random.normal((1, 24, 32), dtype=mx.float16)
    y2 = attn(x24)
    mx.eval(y2)
    cache2 = _get_cache(attn)
    assert cache2 is not None
    key2 = cache2.cache_key

    assert key1 != key2


def test_cache_output_matches_uncached() -> None:
    """Cached output should equal uncached output (atol=1e-5)."""
    attn = _make_attn()
    x = mx.random.normal((2, 16, 32), dtype=mx.float16)

    # Force build (first pass)
    y1 = attn(x)
    mx.eval(y1)

    # Second pass uses cache
    y2 = attn(x)
    mx.eval(y2)

    np.testing.assert_allclose(
        np.asarray(y1, dtype=np.float32),
        np.asarray(y2, dtype=np.float32),
        atol=1e-5,
    )


def test_dynamic_mode_always_rebuilds() -> None:
    """Greedy strategy should rebuild every pass (no cache stored)."""
    # Clear any stale entries from prior tests
    _GRAPH_CACHE_STORE.clear()

    attn = _make_attn(strategy="greedy")

    x = mx.random.normal((1, 16, 32), dtype=mx.float16)
    y1, d1 = attn(x, return_debug=True)
    mx.eval(y1)
    assert float(d1["profile"]["graph_build_ms"]) >= 0.0

    # Dynamic mode should NOT persist cache
    assert _get_cache(attn) is None

    # Second pass should also rebuild (non-trivial graph_build_ms)
    y2, d2 = attn(x, return_debug=True)
    mx.eval(y2)
    gb2 = d2["profile"]["graph_build_ms"]
    assert gb2 > 0.1, f"Dynamic mode should rebuild, got graph_build_ms={gb2}"


def test_sparse_precomputed_masks_match() -> None:
    """Precomputed safe_idx / causal_mask should match fresh computation."""
    attn = _make_attn()
    T = 16
    x = mx.random.normal((1, T, 32), dtype=mx.float16)

    y = attn(x)
    mx.eval(y)

    cache = _get_cache(attn)
    assert cache is not None
    graph = cache.mlx_graph

    fresh_safe = safe_neighbor_idx(graph.neigh_idx, T)
    fresh_mask = causal_neighbor_mask(graph.neigh_idx, T)

    np.testing.assert_array_equal(
        np.asarray(cache.safe_idx), np.asarray(fresh_safe)
    )
    np.testing.assert_array_equal(
        np.asarray(cache.causal_mask), np.asarray(fresh_mask)
    )


def test_permute_precomputed_perms_match() -> None:
    """Precomputed permute artifacts should match fresh computation."""
    attn = _make_attn(path="permute")
    T = 16
    x = mx.random.normal((1, T, 32), dtype=mx.float16)

    y = attn(x)
    mx.eval(y)

    cache = _get_cache(attn)
    assert cache is not None

    cycle_perms = cache.mlx_graph.meta.get("cycle_perms", [])
    for h in range(attn.n_heads):
        perm = cycle_perms[h]
        if perm is None:
            continue
        perm_arr = np.asarray(perm, dtype=np.int32)
        perm_mx = mx.array(perm_arr, dtype=mx.int32)

        np.testing.assert_array_equal(
            np.asarray(cache.perm_mx[h]), np.asarray(perm_mx)
        )

        inv_perm = mx.argsort(perm_mx)
        np.testing.assert_array_equal(
            np.asarray(cache.inv_perm[h]), np.asarray(inv_perm)
        )
