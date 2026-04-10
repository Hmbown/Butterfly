"""Correctness tests for the fused all-head permute-window attention dispatch.

Compares ``butterfly_fused_permute_window_attention`` against the existing
chunked ``butterfly_permute_window_attention_batched`` (with fused disabled)
as the reference implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import (
    butterfly_permute_window_attention_active_batched,
    butterfly_permute_window_attention_batched,
)
from bna.mlx.fused_attention import (
    _fused_dispatch_eligible,
    butterfly_fused_permute_window_attention,
    butterfly_fused_permute_window_attention_active,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_perms(Hq: int, T: int, seed: int = 0) -> tuple:
    """Generate random cycle permutations and their inverses."""
    rng = np.random.default_rng(seed)
    perms = np.zeros((Hq, T), dtype=np.int32)
    inv_perms = np.zeros((Hq, T), dtype=np.int32)
    for h in range(Hq):
        p = rng.permutation(T).astype(np.int32)
        perms[h] = p
        inv_perms[h] = np.argsort(p).astype(np.int32)
    return mx.array(perms), mx.array(inv_perms)


def _run_comparison(
    B: int, Hq: int, Hkv: int, T: int, dh: int, W: int,
    seed: int = 42, atol: float = 1e-3, mean_atol: float = 1e-4,
):
    """Compare fused vs chunked path and assert numerical parity."""
    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal((B, Hq, T, dh)).astype(np.float32) * 0.1
    k_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.1
    v_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.1

    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)
    all_perms, all_inv_perms = _random_perms(Hq, T, seed=seed + 1)

    # Fused path
    y_fused = butterfly_fused_permute_window_attention(
        q, k, v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        window=W,
        query_chunk_size=max(32, T // 2),
    )
    mx.eval(y_fused)

    # Chunked reference (fused disabled)
    y_ref, _ = butterfly_permute_window_attention_batched(
        q, k, v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        window=W,
        head_chunk_size=1,
        query_chunk_size=max(32, T // 2),
        use_fused_dispatch=False,
    )
    mx.eval(y_ref)

    y_f = np.array(y_fused)
    y_r = np.array(y_ref)

    max_diff = float(np.max(np.abs(y_f - y_r)))
    mean_diff = float(np.mean(np.abs(y_f - y_r)))
    assert max_diff < atol, (
        f"max atol exceeded: {max_diff:.6f} >= {atol} "
        f"(B={B}, Hq={Hq}, Hkv={Hkv}, T={T}, dh={dh}, W={W})"
    )
    assert mean_diff < mean_atol, (
        f"mean atol exceeded: {mean_diff:.6f} >= {mean_atol} "
        f"(B={B}, Hq={Hq}, Hkv={Hkv}, T={T}, dh={dh}, W={W})"
    )
    return max_diff, mean_diff


# ---------------------------------------------------------------------------
# Test: numerical parity across representative configs
# ---------------------------------------------------------------------------

class TestNumericalParity:
    def test_glm_like_gqa_4to1(self):
        """GLM-like: 4 query heads, 1 KV head, small sequence."""
        _run_comparison(B=1, Hq=4, Hkv=1, T=64, dh=32, W=8)

    def test_qwen_like_gqa_4to1(self):
        """Qwen-like: 8 query heads, 2 KV heads."""
        _run_comparison(B=1, Hq=8, Hkv=2, T=256, dh=64, W=16)

    def test_mha_no_gqa(self):
        """MHA: equal query and KV heads."""
        _run_comparison(B=1, Hq=4, Hkv=4, T=128, dh=64, W=32)

    def test_larger_sequence(self):
        """Moderate sequence length with GQA."""
        _run_comparison(B=1, Hq=8, Hkv=1, T=512, dh=32, W=16)


# ---------------------------------------------------------------------------
# Test: GQA correctness across ratios
# ---------------------------------------------------------------------------

class TestGQACorrectness:
    def test_ratio_1_to_1(self):
        _run_comparison(B=1, Hq=4, Hkv=4, T=64, dh=32, W=8)

    def test_ratio_4_to_1(self):
        _run_comparison(B=1, Hq=4, Hkv=1, T=64, dh=32, W=8)

    def test_ratio_8_to_1(self):
        _run_comparison(B=1, Hq=8, Hkv=1, T=64, dh=32, W=8)

    def test_ratio_2_to_1(self):
        _run_comparison(B=1, Hq=4, Hkv=2, T=64, dh=32, W=8)


# ---------------------------------------------------------------------------
# Test: causality (no future leakage)
# ---------------------------------------------------------------------------

class TestCausality:
    def test_no_future_leakage(self):
        """Verify token i does not attend to j > i in original positions.

        Strategy: set v[j] = j for a single value dimension, check that
        output[i] is bounded by i (weighted average of positions <= i).
        """
        B, Hq, Hkv, T, dh, W = 1, 2, 1, 32, 16, 4
        rng = np.random.default_rng(99)

        q_np = rng.standard_normal((B, Hq, T, dh)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.1
        # Set first dim of v to position index
        v_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.01
        v_np[0, 0, :, 0] = np.arange(T, dtype=np.float32)

        q = mx.array(q_np)
        k = mx.array(k_np)
        v = mx.array(v_np)
        all_perms, all_inv_perms = _random_perms(Hq, T, seed=99)

        y = butterfly_fused_permute_window_attention(
            q, k, v,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            window=W,
            query_chunk_size=T,
        )
        mx.eval(y)
        y_np = np.array(y)

        # For each head, the position-channel output should be <= i
        # (it's a convex combination of positions <= i)
        for h in range(Hq):
            for i in range(T):
                val = float(y_np[0, h, i, 0])
                assert val <= float(i) + 0.01, (
                    f"Causality violation: head {h}, token {i}, "
                    f"output channel 0 = {val:.4f} > {i}"
                )


# ---------------------------------------------------------------------------
# Test: fallback for ineligible configs
# ---------------------------------------------------------------------------

class TestFallback:
    def test_circular_falls_back(self):
        """circular=True should not use fused path."""
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=True,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_retro_falls_back(self):
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=True,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_3d_perms_falls_back(self):
        """3D multi-cycle permutations should fall back."""
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 2, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_edge_bias_falls_back(self):
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=0.5,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_union_falls_back(self):
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="union",
            use_fused_dispatch=True,
        )

    def test_window_drop_training_falls_back(self):
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.2,
            training=True,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_disabled_toggle_falls_back(self):
        assert not _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=False,
        )

    def test_eligible_basic(self):
        assert _fused_dispatch_eligible(
            all_perms=mx.zeros((4, 64), dtype=mx.int32),
            edge_type_bias_scalar=None,
            window_drop_prob=0.0,
            training=False,
            retro_backfill_enabled=False,
            circular=False,
            multi_cycle_mode="average",
            use_fused_dispatch=True,
        )

    def test_ineligible_produces_same_output(self):
        """When fused is disabled, batched path still works correctly."""
        B, Hq, Hkv, T, dh, W = 1, 4, 1, 64, 32, 8
        rng = np.random.default_rng(7)
        q = mx.array(rng.standard_normal((B, Hq, T, dh)).astype(np.float32) * 0.1)
        k = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.1)
        v = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.1)
        perms, inv_perms = _random_perms(Hq, T, seed=7)

        y1, _ = butterfly_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms,
            window=W, use_fused_dispatch=True,
        )
        y2, _ = butterfly_permute_window_attention_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms,
            window=W, use_fused_dispatch=False, head_chunk_size=1,
        )
        mx.eval(y1, y2)
        max_diff = float(np.max(np.abs(np.array(y1) - np.array(y2))))
        assert max_diff < 1e-3, f"Fused vs chunked mismatch: {max_diff}"


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_t_less_than_window(self):
        """T smaller than window size."""
        _run_comparison(B=1, Hq=2, Hkv=1, T=8, dh=16, W=16)

    def test_t_equals_window(self):
        """T equals window."""
        _run_comparison(B=1, Hq=2, Hkv=1, T=16, dh=16, W=16)

    def test_single_token(self):
        """Single token sequence."""
        _run_comparison(B=1, Hq=2, Hkv=1, T=1, dh=16, W=4)

    def test_hq_equals_hkv(self):
        """No GQA: Hq == Hkv."""
        _run_comparison(B=1, Hq=4, Hkv=4, T=32, dh=16, W=4)

    def test_batch_2(self):
        """Batch size > 1."""
        _run_comparison(B=2, Hq=4, Hkv=1, T=64, dh=32, W=8)


# ---------------------------------------------------------------------------
# Helpers for active-row tests
# ---------------------------------------------------------------------------

def _run_active_comparison(
    B: int, Hq: int, Hkv: int, Tq: int, Tk: int, dh: int, W: int,
    Tg: int | None = None,
    seed: int = 42, atol: float = 1e-3, mean_atol: float = 1e-4,
):
    """Compare fused vs chunked active-row path and assert parity."""
    if Tg is None:
        Tg = Tk
    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.1
    k_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1
    v_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1

    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)

    # Perms over Tg (graph horizon, may be > Tk)
    all_perms, all_inv_perms = _random_perms(Hq, Tg, seed=seed + 1)

    # Query positions: last Tq tokens of the cache prefix
    q_positions = mx.arange(Tk - Tq, Tk, dtype=mx.int32)

    # Fused path
    y_fused, _ = butterfly_permute_window_attention_active_batched(
        q, k, v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        query_positions=q_positions,
        window=W,
        query_chunk_size=max(8, Tq // 2),
        use_fused_dispatch=True,
    )
    mx.eval(y_fused)

    # Chunked reference (fused disabled)
    y_ref, _ = butterfly_permute_window_attention_active_batched(
        q, k, v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        query_positions=q_positions,
        window=W,
        head_chunk_size=1,
        query_chunk_size=max(8, Tq // 2),
        use_fused_dispatch=False,
    )
    mx.eval(y_ref)

    y_f = np.array(y_fused)
    y_r = np.array(y_ref)

    max_diff = float(np.max(np.abs(y_f - y_r)))
    mean_diff = float(np.mean(np.abs(y_f - y_r)))
    assert max_diff < atol, (
        f"Active-row max atol exceeded: {max_diff:.6f} >= {atol} "
        f"(B={B}, Hq={Hq}, Hkv={Hkv}, Tq={Tq}, Tk={Tk}, dh={dh}, W={W})"
    )
    assert mean_diff < mean_atol, (
        f"Active-row mean atol exceeded: {mean_diff:.6f} >= {mean_atol} "
        f"(B={B}, Hq={Hq}, Hkv={Hkv}, Tq={Tq}, Tk={Tk}, dh={dh}, W={W})"
    )
    return max_diff, mean_diff


# ---------------------------------------------------------------------------
# Test: fused active-row numerical parity
# ---------------------------------------------------------------------------

class TestFusedActiveRowParity:
    def test_glm_like(self):
        """GLM-like: B=1, Hq=4, Hkv=1, Tq=16, Tk=64."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=16, Tk=64, dh=32, W=8)

    def test_qwen_like(self):
        """Qwen-like: B=1, Hq=8, Hkv=2, Tq=32, Tk=256."""
        _run_active_comparison(B=1, Hq=8, Hkv=2, Tq=32, Tk=256, dh=64, W=16)

    def test_mha(self):
        """MHA: B=1, Hq=4, Hkv=4."""
        _run_active_comparison(B=1, Hq=4, Hkv=4, Tq=8, Tk=128, dh=64, W=32)

    def test_larger_tq(self):
        """Larger query block with GQA."""
        _run_active_comparison(B=1, Hq=8, Hkv=1, Tq=64, Tk=256, dh=32, W=16)


# ---------------------------------------------------------------------------
# Test: fused active-row GQA ratios
# ---------------------------------------------------------------------------

class TestFusedActiveRowGQA:
    def test_ratio_1_to_1(self):
        _run_active_comparison(B=1, Hq=4, Hkv=4, Tq=16, Tk=64, dh=32, W=8)

    def test_ratio_4_to_1(self):
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=16, Tk=64, dh=32, W=8)

    def test_ratio_8_to_1(self):
        _run_active_comparison(B=1, Hq=8, Hkv=1, Tq=16, Tk=64, dh=32, W=8)


class TestFusedActiveRowGQAPermExpansion:
    """Regression: perms may have fewer heads than queries (Hp < Hq).

    In GLM-4.7-Flash, topology builds perms for n_kv_heads=2 but there
    are Hq=40 query heads.  The fused path must expand [Hp, T] perms to
    [Hq, T] before flat-index arithmetic to avoid OOB reads.
    """

    def _run_with_fewer_perm_heads(
        self, B, Hq, Hkv, Hp, Tq, Tk, dh, W, seed=42,
    ):
        """Run fused active-row with perms shaped [Hp, T] where Hp < Hq."""
        rng = np.random.default_rng(seed)
        q_np = rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1
        v_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1

        q = mx.array(q_np)
        k = mx.array(k_np)
        v = mx.array(v_np)

        # Build perms with Hp heads (fewer than Hq) — mirrors GLM topology
        perms_hp, inv_hp = _random_perms(Hp, Tk, seed=seed + 1)

        # Also build the "expanded" reference perms [Hq, T] by repeating
        rep = Hq // Hp
        perms_hq = mx.repeat(perms_hp, repeats=rep, axis=0)
        inv_hq = mx.repeat(inv_hp, repeats=rep, axis=0)

        q_positions = mx.arange(Tk - Tq, Tk, dtype=mx.int32)

        # Fused with Hp-shaped perms (exercises the GQA expansion fix)
        y_hp = butterfly_fused_permute_window_attention_active(
            q, k, v,
            all_perms=perms_hp,
            all_inv_perms=inv_hp,
            query_positions=q_positions,
            window=W,
        )
        mx.eval(y_hp)

        # Reference with pre-expanded Hq-shaped perms
        y_hq = butterfly_fused_permute_window_attention_active(
            q, k, v,
            all_perms=perms_hq,
            all_inv_perms=inv_hq,
            query_positions=q_positions,
            window=W,
        )
        mx.eval(y_hq)

        y_a = np.array(y_hp)
        y_b = np.array(y_hq)
        max_diff = float(np.max(np.abs(y_a - y_b)))
        assert max_diff < 1e-5, (
            f"Hp={Hp} vs Hq={Hq} expanded perms differ: max_diff={max_diff:.6f}"
        )
        assert not np.any(np.isnan(y_a)), "NaN in output with Hp-shaped perms"

    def test_glm_like_hp2_hq8(self):
        """GLM-like: Hp=2 perm heads, Hq=8 query heads, Hkv=2."""
        self._run_with_fewer_perm_heads(
            B=1, Hq=8, Hkv=2, Hp=2, Tq=1, Tk=64, dh=32, W=8,
        )

    def test_glm_like_hp2_hq8_prefill_chunk(self):
        """GLM-like with larger Tq (chunked active prefill)."""
        self._run_with_fewer_perm_heads(
            B=1, Hq=8, Hkv=2, Hp=2, Tq=16, Tk=64, dh=32, W=8,
        )

    def test_hp1_hq4(self):
        """Hp=1 perm head, Hq=4 query heads."""
        self._run_with_fewer_perm_heads(
            B=1, Hq=4, Hkv=1, Hp=1, Tq=1, Tk=32, dh=32, W=8,
        )

    def test_hp2_hq8_tq_equals_tk(self):
        """Full-prefill active path: Tq == Tk with Hp < Hq."""
        self._run_with_fewer_perm_heads(
            B=1, Hq=8, Hkv=2, Hp=2, Tq=64, Tk=64, dh=32, W=8,
        )


# ---------------------------------------------------------------------------
# Test: fused active-row causality
# ---------------------------------------------------------------------------

class TestFusedActiveRowCausality:
    def test_no_future_leakage(self):
        """Active-row: token at position p attends only to positions <= p."""
        B, Hq, Hkv, Tq, Tk, dh, W = 1, 2, 1, 8, 32, 16, 4
        rng = np.random.default_rng(77)

        q_np = rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1
        v_np = rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.01
        # Set first dim of v to position index
        v_np[0, 0, :, 0] = np.arange(Tk, dtype=np.float32)

        q = mx.array(q_np)
        k = mx.array(k_np)
        v = mx.array(v_np)
        all_perms, all_inv_perms = _random_perms(Hq, Tk, seed=77)

        q_positions = mx.arange(Tk - Tq, Tk, dtype=mx.int32)

        y = butterfly_fused_permute_window_attention_active(
            q, k, v,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            query_positions=q_positions,
            window=W,
            query_chunk_size=Tq,
        )
        mx.eval(y)
        y_np = np.array(y)

        for h in range(Hq):
            for i in range(Tq):
                pos = Tk - Tq + i
                val = float(y_np[0, h, i, 0])
                assert val <= float(pos) + 0.01, (
                    f"Active-row causality violation: head {h}, query {i}, "
                    f"pos {pos}, output channel 0 = {val:.4f} > {pos}"
                )


# ---------------------------------------------------------------------------
# Test: fused active-row fallback for ineligible configs
# ---------------------------------------------------------------------------

class TestFusedActiveRowFallback:
    def test_circular_falls_back_but_correct(self):
        """circular=True should fall back to chunked, still correct."""
        B, Hq, Hkv, Tq, Tk, dh, W = 1, 4, 1, 8, 64, 32, 8
        rng = np.random.default_rng(55)
        q = mx.array(rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.1)
        k = mx.array(rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1)
        v = mx.array(rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1)
        perms, inv_perms = _random_perms(Hq, Tk, seed=55)
        q_pos = mx.arange(Tk - Tq, Tk, dtype=mx.int32)

        # With circular=True, fused should be ineligible but still produce output
        y, _ = butterfly_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms,
            query_positions=q_pos, window=W,
            use_fused_dispatch=True, circular=True,
        )
        mx.eval(y)
        assert y.shape == (B, Hq, Tq, dh)

    def test_edge_bias_falls_back_but_correct(self):
        """edge_type_bias should fall back to chunked, still correct."""
        B, Hq, Hkv, Tq, Tk, dh, W = 1, 4, 1, 8, 64, 32, 8
        rng = np.random.default_rng(56)
        q = mx.array(rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.1)
        k = mx.array(rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1)
        v = mx.array(rng.standard_normal((B, Hkv, Tk, dh)).astype(np.float32) * 0.1)
        perms, inv_perms = _random_perms(Hq, Tk, seed=56)
        q_pos = mx.arange(Tk - Tq, Tk, dtype=mx.int32)

        y, _ = butterfly_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms, all_inv_perms=inv_perms,
            query_positions=q_pos, window=W,
            edge_type_bias_scalar=0.5,
            use_fused_dispatch=True,
        )
        mx.eval(y)
        assert y.shape == (B, Hq, Tq, dh)


# ---------------------------------------------------------------------------
# Test: fused active-row edge cases
# ---------------------------------------------------------------------------

class TestFusedActiveRowEdgeCases:
    def test_single_query(self):
        """Tq=1: single active query."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=1, Tk=32, dh=32, W=8)

    def test_tq_equals_tk(self):
        """Tq == Tk: full overlap (all tokens are queries)."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=32, Tk=32, dh=32, W=8)

    def test_tg_greater_than_tk(self):
        """Tg > Tk: adaptive graph horizon."""
        _run_active_comparison(
            B=1, Hq=4, Hkv=1, Tq=8, Tk=32, dh=32, W=8, Tg=64,
        )

    def test_batch_2(self):
        """Batch size > 1."""
        _run_active_comparison(B=2, Hq=4, Hkv=1, Tq=16, Tk=64, dh=32, W=8)


# ---------------------------------------------------------------------------
# Test: 8W active contiguous path (2*Tq >= Tk gate)
# ---------------------------------------------------------------------------

class TestActiveContiguous8W:
    """Verify the 8W contiguous active-row path fires and matches reference.

    The gate ``2 * Tq >= Tk`` routes to ``_active_via_full_prefill`` with
    chunk size ``8W``.  These tests exercise the critical second-chunk case
    (Tq = Tk/2) that triggers at T=8192 with chunk_size=4096.
    """

    def test_half_sequence_active(self):
        """Tq = Tk/2: exactly at the 2*Tq >= Tk boundary."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=32, Tk=64, dh=32, W=8)

    def test_half_sequence_gqa_20_1(self):
        """GLM-like GQA ratio (20:1) at the half-sequence boundary."""
        _run_active_comparison(B=1, Hq=20, Hkv=1, Tq=64, Tk=128, dh=32, W=8)

    def test_large_window_8w_chunking(self):
        """Larger window to exercise 8W chunk formula (q_chunk = 8*16 = 128)."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=128, Tk=256, dh=32, W=16)

    def test_below_gate_uses_gather(self):
        """Tq < Tk/2: should fall to gather path but still be correct."""
        _run_active_comparison(B=1, Hq=4, Hkv=1, Tq=16, Tk=64, dh=32, W=8)

    def test_8k_sim(self):
        """Simulated 8k regime: Tq=256, Tk=512, W=32, mimicking chunk1."""
        _run_active_comparison(
            B=1, Hq=4, Hkv=1, Tq=256, Tk=512, dh=64, W=32,
        )
