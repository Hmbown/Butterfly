from __future__ import annotations

import numpy as np
import pytest

from hcsa.cycles import edge_disjoint_random_cycles, verify_edge_disjoint

try:
    import mlx.core as mx

    from hcsa.mlx.attention import wayfinder_permute_window_attention_batched

    _HAS_MLX = True
except Exception:  # pragma: no cover - optional dependency in CI variants
    _HAS_MLX = False


def test_edge_disjoint_random_cycles_two_cycles() -> None:
    cycles = edge_disjoint_random_cycles(
        128,
        2,
        generator=np.random.default_rng(42),
    )
    assert len(cycles) == 2
    is_disjoint, shared_edges = verify_edge_disjoint(cycles)
    assert is_disjoint is True
    assert shared_edges == 0


def test_edge_disjoint_random_cycles_three_cycles() -> None:
    cycles = edge_disjoint_random_cycles(
        64,
        3,
        generator=np.random.default_rng(7),
    )
    assert len(cycles) == 3
    is_disjoint, shared_edges = verify_edge_disjoint(cycles)
    assert is_disjoint is True
    assert shared_edges == 0


def test_verify_edge_disjoint_detects_overlap() -> None:
    perm = np.arange(32, dtype=np.int64)
    is_disjoint, shared_edges = verify_edge_disjoint([perm, perm.copy()])
    assert is_disjoint is False
    assert shared_edges > 0


@pytest.mark.skipif(not _HAS_MLX, reason="mlx is required for permute kernel tests")
def test_multi_cycle_permute_path_shape_and_finite() -> None:
    rng = np.random.default_rng(123)
    b, h, t, dh = 1, 2, 64, 16
    q = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    k = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    v = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)

    perms = []
    invs = []
    for head in range(h):
        cycles_h = edge_disjoint_random_cycles(
            t,
            2,
            generator=np.random.default_rng(1000 + head),
        )
        perms_h = np.stack(cycles_h, axis=0).astype(np.int32)
        invs_h = np.argsort(perms_h, axis=-1).astype(np.int32)
        perms.append(perms_h)
        invs.append(invs_h)

    all_perms = mx.array(np.stack(perms, axis=0), dtype=mx.int32)  # [H,d,T]
    all_inv = mx.array(np.stack(invs, axis=0), dtype=mx.int32)  # [H,d,T]

    y, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=all_perms,
        all_inv_perms=all_inv,
        window=8,
        query_chunk_size=32,
    )
    mx.eval(y)
    y_np = np.asarray(y, dtype=np.float32)
    assert y_np.shape == (b, h, t, dh)
    assert np.isfinite(y_np).all()


@pytest.mark.skipif(not _HAS_MLX, reason="mlx is required for permute kernel tests")
def test_single_cycle_3d_matches_2d_path() -> None:
    rng = np.random.default_rng(999)
    b, h, t, dh = 1, 2, 48, 8
    q = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    k = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)
    v = mx.array(rng.standard_normal((b, h, t, dh), dtype=np.float32), dtype=mx.float32)

    perms_np = np.stack(
        [rng.permutation(t).astype(np.int32) for _ in range(h)],
        axis=0,
    )
    inv_np = np.argsort(perms_np, axis=-1).astype(np.int32)

    y_2d, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=mx.array(perms_np, dtype=mx.int32),
        all_inv_perms=mx.array(inv_np, dtype=mx.int32),
        window=6,
        query_chunk_size=24,
    )
    y_3d, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=mx.array(perms_np[:, None, :], dtype=mx.int32),
        all_inv_perms=mx.array(inv_np[:, None, :], dtype=mx.int32),
        window=6,
        query_chunk_size=24,
    )
    mx.eval(y_2d, y_3d)
    np.testing.assert_allclose(
        np.asarray(y_2d, dtype=np.float32),
        np.asarray(y_3d, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
