from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hcsa.mlx.attention import wayfinder_permute_window_attention_batched


def _make_inputs(*, seed: int = 7, B: int = 1, Hq: int = 2, Hkv: int = 1, T: int = 17, dh: int = 8):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, Hq, T, dh), dtype=np.float32))
    k = mx.array(rng.standard_normal((B, Hkv, T, dh), dtype=np.float32))
    v = mx.array(rng.standard_normal((B, Hkv, T, dh), dtype=np.float32))

    perms = []
    invs = []
    for h in range(Hq):
        perm = rng.permutation(T).astype(np.int32)
        inv = np.argsort(perm).astype(np.int32)
        perms.append(perm)
        invs.append(inv)

    all_perms = mx.array(np.stack(perms, axis=0), dtype=mx.int32)
    all_inv = mx.array(np.stack(invs, axis=0), dtype=mx.int32)
    return q, k, v, all_perms, all_inv


def test_retro_disabled_matches_baseline() -> None:
    """With retro disabled, output must match baseline exactly."""
    q, k, v, perms, inv = _make_inputs(seed=101)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=False,
        head_chunk_size=2,
        query_chunk_size=8,
    )
    y_off, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=False,
        head_chunk_size=2,
        query_chunk_size=8,
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.2,
        retro_backfill_training_only=True,
    )
    mx.eval(y_base, y_off)

    assert np.allclose(np.asarray(y_base), np.asarray(y_off), atol=1e-6, rtol=1e-6)


def test_retro_training_only_guard() -> None:
    """When training=False and training_only=True, retro must not activate."""
    q, k, v, perms, inv = _make_inputs(seed=202)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=False,
        head_chunk_size=2,
        query_chunk_size=8,
    )
    y_guarded, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=False,
        head_chunk_size=2,
        query_chunk_size=8,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.25,
        retro_backfill_training_only=True,
    )
    mx.eval(y_base, y_guarded)

    assert np.allclose(np.asarray(y_base), np.asarray(y_guarded), atol=1e-6, rtol=1e-6)


def test_simple_retro_active_changes_output_and_is_finite() -> None:
    """Simplified retro with training=True must change output and be numerically stable."""
    q, k, v, perms, inv = _make_inputs(seed=303)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=8,
    )
    y_retro, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=8,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.2,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=False,
    )
    mx.eval(y_base, y_retro)

    y_base_np = np.asarray(y_base)
    y_retro_np = np.asarray(y_retro)
    assert np.isfinite(y_retro_np).all()
    assert not np.allclose(y_base_np, y_retro_np, atol=1e-6, rtol=1e-6)


def test_simple_retro_alpha_zero_is_noop() -> None:
    """With alpha=0, retro should be a no-op even when enabled."""
    q, k, v, perms, inv = _make_inputs(seed=404)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=8,
    )
    y_retro, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=8,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.0,  # zero alpha
        retro_backfill_training_only=True,
        retro_backfill_causal_only=False,
    )
    mx.eval(y_base, y_retro)

    assert np.allclose(np.asarray(y_base), np.asarray(y_retro), atol=1e-6, rtol=1e-6)


def test_simple_retro_causal_only_blocks_identity_future() -> None:
    """With identity permutation, causal-only retro must be a no-op."""
    q, k, v, _perms, _inv = _make_inputs(seed=505, T=32, dh=16)
    T = int(q.shape[2])
    Hq = int(q.shape[1])
    perm_np = np.arange(T, dtype=np.int32)
    perms = mx.array(np.stack([perm_np for _ in range(Hq)], axis=0), dtype=mx.int32)
    inv = perms

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=16,
    )
    y_retro, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=16,
        retro_backfill_enabled=True,
        retro_backfill_alpha=1.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
    )
    mx.eval(y_base, y_retro)
    assert np.allclose(np.asarray(y_base), np.asarray(y_retro), atol=1e-6, rtol=1e-6)


def test_simple_retro_causal_only_changes_for_reverse_perm() -> None:
    """Reverse permutation has causal successor edges; causal-only retro should activate."""
    q, k, v, _perms, _inv = _make_inputs(seed=506, T=32, dh=16)
    T = int(q.shape[2])
    Hq = int(q.shape[1])
    perm_np = np.arange(T - 1, -1, -1, dtype=np.int32)
    inv_np = np.argsort(perm_np).astype(np.int32)
    perms = mx.array(np.stack([perm_np for _ in range(Hq)], axis=0), dtype=mx.int32)
    inv = mx.array(np.stack([inv_np for _ in range(Hq)], axis=0), dtype=mx.int32)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=16,
    )
    y_retro, _ = wayfinder_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=16,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.5,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
    )
    mx.eval(y_base, y_retro)
    assert np.isfinite(np.asarray(y_retro)).all()
    assert not np.allclose(np.asarray(y_base), np.asarray(y_retro), atol=1e-6, rtol=1e-6)


def test_simple_retro_different_perms_per_head() -> None:
    """Retro should work correctly with different permutations per head."""
    rng = np.random.default_rng(606)
    B, Hq, Hkv, T, dh = 1, 4, 2, 24, 8

    q = mx.array(rng.standard_normal((B, Hq, T, dh), dtype=np.float32))
    k = mx.array(rng.standard_normal((B, Hkv, T, dh), dtype=np.float32))
    v = mx.array(rng.standard_normal((B, Hkv, T, dh), dtype=np.float32))

    perms = mx.array(np.stack([rng.permutation(T).astype(np.int32) for _ in range(Hq)]), dtype=mx.int32)
    inv = mx.array(np.stack([np.argsort(perms[h]) for h in range(Hq)]), dtype=mx.int32)

    y_base, _ = wayfinder_permute_window_attention_batched(
        q, k, v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=12,
    )
    y_retro, _ = wayfinder_permute_window_attention_batched(
        q, k, v,
        all_perms=perms,
        all_inv_perms=inv,
        window=4,
        training=True,
        head_chunk_size=2,
        query_chunk_size=12,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.3,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=False,
    )
    mx.eval(y_base, y_retro)

    y_base_np = np.asarray(y_base)
    y_retro_np = np.asarray(y_retro)
    assert np.isfinite(y_retro_np).all()
    assert not np.allclose(y_base_np, y_retro_np, atol=1e-6, rtol=1e-6)
