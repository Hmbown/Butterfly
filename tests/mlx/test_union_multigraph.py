"""Tests for union multigraph multi-cycle attention."""
import numpy as np
import mlx.core as mx

from hcsa.mlx.attention import (
    build_union_multigraph_index,
    wayfinder_permute_window_attention_batched,
)


def _make_multicycle_inputs(T=16, H=2, d=2, dh=8, seed=42):
    """Create test inputs for multi-cycle attention."""
    rng = np.random.default_rng(seed)
    B = 1
    q = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))

    # Build d random permutations per head
    perms = np.zeros((H, d, T), dtype=np.int32)
    inv_perms = np.zeros((H, d, T), dtype=np.int32)
    for h in range(H):
        for c in range(d):
            p = rng.permutation(T).astype(np.int32)
            perms[h, c] = p
            inv_p = np.empty(T, dtype=np.int32)
            inv_p[p] = np.arange(T, dtype=np.int32)
            inv_perms[h, c] = inv_p

    all_perms = mx.array(perms)
    all_inv_perms = mx.array(inv_perms)
    return q, k, v, all_perms, all_inv_perms


class TestBuildUnionMultigraphIndex:
    """Tests for the union multigraph index builder."""

    def test_union_graph_shape(self):
        """Union graph should have correct shape [H, T, D_union]."""
        T, H, d = 16, 2, 2
        rng = np.random.default_rng(42)
        perms = np.zeros((H, d, T), dtype=np.int32)
        inv_perms = np.zeros((H, d, T), dtype=np.int32)
        for h in range(H):
            for c in range(d):
                p = rng.permutation(T).astype(np.int32)
                perms[h, c] = p
                inv_p = np.empty(T, dtype=np.int32)
                inv_p[p] = np.arange(T, dtype=np.int32)
                inv_perms[h, c] = inv_p

        idx, mult, valid = build_union_multigraph_index(
            mx.array(perms), mx.array(inv_perms), window=2,
        )
        assert idx.shape[0] == H
        assert idx.shape[1] == T
        assert mult.shape == idx.shape
        assert valid.shape == idx.shape

    def test_union_has_more_neighbors_than_single(self):
        """Union of 2 cycles should have more neighbors than a single cycle."""
        T, H, d = 16, 1, 2
        window = 2
        rng = np.random.default_rng(42)
        perms = np.zeros((H, d, T), dtype=np.int32)
        inv_perms = np.zeros((H, d, T), dtype=np.int32)
        for h in range(H):
            for c in range(d):
                p = rng.permutation(T).astype(np.int32)
                perms[h, c] = p
                inv_p = np.empty(T, dtype=np.int32)
                inv_p[p] = np.arange(T, dtype=np.int32)
                inv_perms[h, c] = inv_p

        # Union of 2 cycles
        idx_union, _, valid_union = build_union_multigraph_index(
            mx.array(perms), mx.array(inv_perms), window=window,
        )
        # Single cycle
        idx_single, _, valid_single = build_union_multigraph_index(
            mx.array(perms[:, :1, :]), mx.array(inv_perms[:, :1, :]), window=window,
        )
        # Union degree should be >= single degree for most positions
        union_deg = np.array(valid_union[0]).sum(axis=1)  # [T]
        single_deg = np.array(valid_single[0]).sum(axis=1)  # [T]
        assert union_deg.sum() >= single_deg.sum()

    def test_multiplicity_for_shared_edges(self):
        """Shared edges between cycles should have multiplicity > 1."""
        T = 8
        # Use two identical permutations => all edges shared => multiplicity = 2
        p = np.arange(T, dtype=np.int32)
        perms = np.stack([[p, p]], axis=0)  # [1, 2, T]
        inv_p = np.arange(T, dtype=np.int32)
        inv_perms = np.stack([[inv_p, inv_p]], axis=0)  # [1, 2, T]

        _, mult, valid = build_union_multigraph_index(
            mx.array(perms), mx.array(inv_perms), window=2,
        )
        mult_np = np.array(mult[0])  # [T, D]
        valid_np = np.array(valid[0])
        # For identical perms, all valid neighbors should have multiplicity 2
        for i in range(T):
            for d_idx in range(mult_np.shape[1]):
                if valid_np[i, d_idx]:
                    assert mult_np[i, d_idx] == 2, (
                        f"Position {i}, neighbor {d_idx}: "
                        f"expected mult=2, got {mult_np[i, d_idx]}"
                    )

    def test_causality_enforced(self):
        """Union graph should only contain causal neighbors (j <= i)."""
        T, H, d = 16, 1, 2
        rng = np.random.default_rng(42)
        perms = np.zeros((H, d, T), dtype=np.int32)
        inv_perms = np.zeros((H, d, T), dtype=np.int32)
        for h in range(H):
            for c in range(d):
                p = rng.permutation(T).astype(np.int32)
                perms[h, c] = p
                inv_p = np.empty(T, dtype=np.int32)
                inv_p[p] = np.arange(T, dtype=np.int32)
                inv_perms[h, c] = inv_p

        idx, _, valid = build_union_multigraph_index(
            mx.array(perms), mx.array(inv_perms), window=3,
        )
        idx_np = np.array(idx[0])
        valid_np = np.array(valid[0])
        for i in range(T):
            for d_idx in range(idx_np.shape[1]):
                if valid_np[i, d_idx]:
                    j = idx_np[i, d_idx]
                    assert j <= i, f"Acausal: token {i} attends to future token {j}"


class TestUnionMulticycleMode:
    """Tests for multi_cycle_mode='union' in batched attention."""

    def test_average_mode_unchanged(self):
        """multi_cycle_mode='average' should produce same output as default."""
        q, k, v, perms, inv_perms = _make_multicycle_inputs()
        y_default, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
        )
        y_avg, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
            multi_cycle_mode="average",
        )
        np.testing.assert_array_equal(np.array(y_default), np.array(y_avg))

    def test_union_mode_valid_output(self):
        """Union mode should produce valid (non-NaN, correct shape) output."""
        q, k, v, perms, inv_perms = _make_multicycle_inputs()
        y_union, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
            multi_cycle_mode="union",
        )
        assert y_union.shape == q.shape
        assert not mx.any(mx.isnan(y_union)).item()

    def test_union_mode_differs_from_average(self):
        """Union and average modes should generally produce different outputs."""
        q, k, v, perms, inv_perms = _make_multicycle_inputs()
        y_avg, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
            multi_cycle_mode="average",
        )
        y_union, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms, window=3,
            multi_cycle_mode="union",
        )
        diff = mx.abs(y_avg - y_union)
        assert mx.max(diff).item() > 1e-6

    def test_union_attention_weights_sum_to_one(self):
        """In union mode, attention weights should effectively sum to 1."""
        T, H, d, dh = 8, 1, 2, 4
        rng = np.random.default_rng(42)
        B = 1
        q = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
        k = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
        # Use identity-like V to check weight summation
        v = mx.ones((B, H, T, dh), dtype=mx.float32)

        perms = np.zeros((H, d, T), dtype=np.int32)
        inv_perms = np.zeros((H, d, T), dtype=np.int32)
        for h in range(H):
            for c in range(d):
                p = rng.permutation(T).astype(np.int32)
                perms[h, c] = p
                inv_p = np.empty(T, dtype=np.int32)
                inv_p[p] = np.arange(T, dtype=np.int32)
                inv_perms[h, c] = inv_p

        y, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=mx.array(perms), all_inv_perms=mx.array(inv_perms),
            window=3, multi_cycle_mode="union",
        )
        # With all-ones V, output should be all-ones if weights sum to 1
        # (for positions with at least one valid neighbor)
        y_np = np.array(y[0, 0])  # [T, dh]
        for i in range(1, T):  # skip position 0 (may have no causal neighbors)
            np.testing.assert_allclose(y_np[i], 1.0, atol=1e-5)
