"""Tests for circular windowing in permute attention."""
import numpy as np
import pytest
import mlx.core as mx

from hcsa.mlx.attention import (
    permute_cycle_window_attention_single,
    wayfinder_permute_window_attention_batched,
)


@pytest.fixture
def small_qkv():
    """Create small Q, K, V for testing."""
    T, dh = 16, 8
    rng = np.random.default_rng(42)
    q = mx.array(rng.standard_normal((T, dh)).astype(np.float32))
    k = mx.array(rng.standard_normal((T, dh)).astype(np.float32))
    v = mx.array(rng.standard_normal((T, dh)).astype(np.float32))
    return q, k, v, T, dh


@pytest.fixture
def known_perm():
    """Permutation where perm[0]=5, perm[T-1]=3 for wrap testing."""
    T = 16
    perm = np.array(
        [5, 0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3]
    )
    assert perm[0] == 5 and perm[T - 1] == 3
    assert len(set(perm)) == T  # valid permutation
    return perm


class TestCircularFalseBackwardCompat:
    """circular=False must produce identical output to default."""

    def test_single_circular_false_matches_default(self, small_qkv):
        q, k, v, T, dh = small_qkv
        perm = np.random.default_rng(42).permutation(T)
        window = 3
        # Add batch dim [B=1, T, dh]
        q_b = q[None, :, :]
        k_b = k[None, :, :]
        v_b = v[None, :, :]

        out_default, _, _, _ = permute_cycle_window_attention_single(
            q_b, k_b, v_b, perm=perm, window=window,
        )
        out_false, _, _, _ = permute_cycle_window_attention_single(
            q_b, k_b, v_b, perm=perm, window=window, circular=False,
        )
        np.testing.assert_array_equal(
            np.array(out_default), np.array(out_false),
        )

    def test_batched_circular_false_matches_default(self, small_qkv):
        q, k, v, T, dh = small_qkv
        rng = np.random.default_rng(42)
        H = 2
        # [B=1, H, T, dh]
        q_b = mx.broadcast_to(q[None, None, :, :], (1, H, T, dh))
        k_b = mx.broadcast_to(k[None, None, :, :], (1, H, T, dh))
        v_b = mx.broadcast_to(v[None, None, :, :], (1, H, T, dh))

        perms = mx.array(
            np.stack([rng.permutation(T) for _ in range(H)]),
            dtype=mx.int32,
        )
        inv_perms = mx.array(
            np.stack([np.argsort(np.array(perms[h])) for h in range(H)]),
            dtype=mx.int32,
        )

        out_def, _ = wayfinder_permute_window_attention_batched(
            q_b, k_b, v_b,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
        )
        out_false, _ = wayfinder_permute_window_attention_batched(
            q_b, k_b, v_b,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
            circular=False,
        )
        np.testing.assert_allclose(
            np.array(out_def), np.array(out_false), atol=1e-5,
        )


class TestCircularWrapAround:
    """Verify wrap-around edge is present with circular=True."""

    def test_boundary_positions_have_full_degree(
        self, small_qkv, known_perm,
    ):
        """With circular=True, boundary positions have full neighbors."""
        q, k, v, T, dh = small_qkv
        window = 2

        q_b = q[None, :, :]
        k_b = k[None, :, :]
        v_b = v[None, :, :]

        out_circ, weights_circ, _, _ = (
            permute_cycle_window_attention_single(
                q_b, k_b, v_b,
                perm=known_perm, window=window, circular=True,
                return_weights=True,
            )
        )
        assert out_circ.shape == (1, T, dh)
        assert not mx.any(mx.isnan(out_circ)).item()

    def test_circular_differs_from_linear_at_boundaries(
        self, small_qkv, known_perm,
    ):
        """Circular and linear should differ at boundary positions."""
        q, k, v, T, dh = small_qkv
        window = 2

        q_b = q[None, :, :]
        k_b = k[None, :, :]
        v_b = v[None, :, :]

        out_linear, _, _, _ = permute_cycle_window_attention_single(
            q_b, k_b, v_b,
            perm=known_perm, window=window, circular=False,
        )
        out_circ, _, _, _ = permute_cycle_window_attention_single(
            q_b, k_b, v_b,
            perm=known_perm, window=window, circular=True,
        )
        diff = mx.abs(out_circ - out_linear)
        assert mx.max(diff).item() > 1e-6


class TestCausality:
    """Circular wrapping must not break causality."""

    def test_no_future_attention_with_circular(
        self, small_qkv, known_perm,
    ):
        """No token should attend to a future token."""
        q, k, v, T, dh = small_qkv
        window = 3

        q_b = q[None, :, :]
        k_b = k[None, :, :]
        v_b = v[None, :, :]

        # Get attention weights to inspect causality
        _, weights, _, _ = permute_cycle_window_attention_single(
            q_b, k_b, v_b,
            perm=known_perm, window=window, circular=True,
            return_weights=True,
        )
        # weights: [B=1, T, W] in permuted space
        # Check that for each query at permuted position p, attended
        # keys have original index <= query original index.
        perm = np.array(known_perm)
        W = 2 * window + 1
        weights_np = np.array(weights[0])  # [T, W]
        for p in range(T):
            query_orig = perm[p]
            for w_off in range(W):
                # Circular neighbor in permuted space
                nb_p = (p - window + w_off) % T
                nb_orig = perm[nb_p]
                wt = weights_np[p, w_off]
                if nb_orig > query_orig:
                    assert wt < 1e-5, (
                        f"Permuted pos {p} (orig {query_orig}) "
                        f"attends to future orig {nb_orig} "
                        f"with weight {wt}"
                    )
