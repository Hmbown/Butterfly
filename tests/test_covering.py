from __future__ import annotations

import numpy as np
import pytest

from hcsa.cycles import covering_cycles
from hcsa.graph.analysis import compute_edge_coverage

try:
    import mlx.core as mx

    from hcsa.mlx.attention import (
        dense_causal_attention,
        wayfinder_covering_attention,
        wayfinder_permute_window_attention_batched,
    )

    _HAS_MLX = True
except Exception:  # pragma: no cover
    _HAS_MLX = False


def test_covering_cycles_reaches_target_at_t64() -> None:
    cycles, frac = covering_cycles(
        64,
        max_cycles=50,
        coverage_target=0.95,
        generator=np.random.default_rng(42),
    )
    assert len(cycles) <= 50
    assert frac >= 0.95


def test_coverage_monotonic_with_more_cycles() -> None:
    t = 64
    cycles, _ = covering_cycles(
        t,
        max_cycles=16,
        coverage_target=1.0,
        generator=np.random.default_rng(7),
    )
    prev = 0.0
    for i in range(1, len(cycles) + 1):
        cov = compute_edge_coverage(cycles[:i], t, causal_only=True)["coverage_fraction"]
        assert float(cov) >= prev
        prev = float(cov)


@pytest.mark.skipif(not _HAS_MLX, reason="mlx is required for covering attention tests")
def test_covering_attention_one_cycle_matches_single_cycle_path() -> None:
    rng = np.random.default_rng(3)
    b, h, t, dh = 1, 2, 48, 8
    q = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    k = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    v = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)

    perms_np = np.stack(
        [rng.permutation(t).astype(np.int32) for _ in range(h)],
        axis=0,
    )
    inv_np = np.argsort(perms_np, axis=-1).astype(np.int32)

    y_ref, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=mx.array(perms_np, dtype=mx.int32),
        all_inv_perms=mx.array(inv_np, dtype=mx.int32),
        window=6,
        query_chunk_size=24,
    )
    y_cov, _ = wayfinder_covering_attention(
        q,
        k,
        v,
        all_perms=mx.array(perms_np[:, None, :], dtype=mx.int32),
        all_inv_perms=mx.array(inv_np[:, None, :], dtype=mx.int32),
        window=6,
        query_chunk_size=24,
    )
    mx.eval(y_ref, y_cov)
    np.testing.assert_allclose(
        np.asarray(y_ref, dtype=np.float32),
        np.asarray(y_cov, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not _HAS_MLX, reason="mlx is required for covering attention tests")
def test_covering_attention_moves_toward_dense_with_more_cycles() -> None:
    rng = np.random.default_rng(9)
    b, h, t, dh = 1, 2, 64, 8
    q = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    k = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    v = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)

    y_dense, _ = dense_causal_attention(q, k, v)

    cyc1, _ = covering_cycles(
        t,
        max_cycles=1,
        coverage_target=1.0,
        generator=np.random.default_rng(42),
    )
    cyc8, _ = covering_cycles(
        t,
        max_cycles=8,
        coverage_target=1.0,
        generator=np.random.default_rng(42),
    )

    perms1 = np.stack([np.stack(cyc1, axis=0) for _ in range(h)], axis=0).astype(np.int32)
    inv1 = np.argsort(perms1, axis=-1).astype(np.int32)
    perms8 = np.stack([np.stack(cyc8, axis=0) for _ in range(h)], axis=0).astype(np.int32)
    inv8 = np.argsort(perms8, axis=-1).astype(np.int32)

    y1, _ = wayfinder_covering_attention(
        q,
        k,
        v,
        all_perms=mx.array(perms1, dtype=mx.int32),
        all_inv_perms=mx.array(inv1, dtype=mx.int32),
        window=4,
        query_chunk_size=32,
    )
    y8, _ = wayfinder_covering_attention(
        q,
        k,
        v,
        all_perms=mx.array(perms8, dtype=mx.int32),
        all_inv_perms=mx.array(inv8, dtype=mx.int32),
        window=4,
        query_chunk_size=32,
    )
    mx.eval(y_dense, y1, y8)
    d1 = np.linalg.norm(np.asarray(y1, dtype=np.float32) - np.asarray(y_dense, dtype=np.float32))
    d8 = np.linalg.norm(np.asarray(y8, dtype=np.float32) - np.asarray(y_dense, dtype=np.float32))
    assert float(d8) < float(d1)
