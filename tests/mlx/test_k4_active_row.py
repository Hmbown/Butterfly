"""Correctness tests for butterfly_permute_window_attention_active_batched (K4 active-row path).

Compares the permute-window active-row function against a naive dense reference
that computes, for each query position, full attention over the cycle-window
neighborhood with causal masking.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import butterfly_permute_window_attention_active_batched


# ---------------------------------------------------------------------------
# Naive reference implementation
# ---------------------------------------------------------------------------

def _reference_active_row_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    perm: np.ndarray,
    inv_perm: np.ndarray,
    query_positions: np.ndarray,
    window: int,
) -> np.ndarray:
    """Pure-NumPy reference for a single head, single batch element.

    q: [Tq, dh]
    k: [Tk, dh]
    v: [Tk, dh]
    perm: [Tg]          cycle permutation (rank -> original position)
    inv_perm: [Tg]      inverse permutation (original position -> rank)
    query_positions: [Tq]  original token positions for each active query
    window: int          half-window size

    Returns: [Tq, dh]
    """
    Tq, dh = q.shape
    Tk = k.shape[0]
    Tg = perm.shape[0]
    scale = 1.0 / math.sqrt(dh)
    out = np.zeros((Tq, dh), dtype=np.float64)

    for qi in range(Tq):
        pos = int(query_positions[qi])
        rank = int(inv_perm[pos])

        # Gather candidate key indices from the cycle window.
        neighbors = []
        for offset in range(-window, window + 1):
            r = rank + offset
            if r < 0 or r >= Tg:
                continue
            orig = int(perm[r])
            # Must be within the available cache and causally valid.
            if orig >= Tk:
                continue
            if orig > pos:
                continue
            neighbors.append(orig)

        if not neighbors:
            # All masked -- output stays zero.
            continue

        nb = np.array(neighbors, dtype=np.int32)
        k_nb = k[nb].astype(np.float64)  # [N, dh]
        v_nb = v[nb].astype(np.float64)  # [N, dh]
        q_f = q[qi].astype(np.float64)   # [dh]

        scores = (k_nb @ q_f) * scale  # [N]
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum() + 1e-12
        out[qi] = w @ v_nb

    return out


def _run_comparison(
    B: int,
    Hq: int,
    Hkv: int,
    dh: int,
    Tq: int,
    Tk: int,
    window: int,
    *,
    max_atol: float = 1e-3,
    mean_atol: float = 1e-4,
    seed: int = 42,
) -> None:
    """Run the active-row function and compare against the naive reference."""
    rng = np.random.RandomState(seed)
    Tg = Tk  # graph horizon == cache length

    # Random inputs in float32.
    q_np = rng.randn(B, Hq, Tq, dh).astype(np.float32) * 0.5
    k_np = rng.randn(B, Hkv, Tk, dh).astype(np.float32) * 0.5
    v_np = rng.randn(B, Hkv, Tk, dh).astype(np.float32) * 0.5

    # Per-head permutations.
    perms = np.zeros((Hq, Tg), dtype=np.int32)
    inv_perms = np.zeros((Hq, Tg), dtype=np.int32)
    for h in range(Hq):
        p = rng.permutation(Tg).astype(np.int32)
        perms[h] = p
        inv_perms[h] = np.argsort(p).astype(np.int32)

    # Query positions: pick Tq distinct positions from [0, Tk).
    if Tq <= Tk:
        qp = np.sort(rng.choice(Tk, size=Tq, replace=False)).astype(np.int32)
    else:
        qp = np.sort(rng.choice(Tk, size=Tq, replace=True)).astype(np.int32)

    # --- Run the function under test ---
    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)
    perms_mx = mx.array(perms)
    inv_perms_mx = mx.array(inv_perms)
    qp_mx = mx.array(qp)

    y_mx, _ = butterfly_permute_window_attention_active_batched(
        q_mx,
        k_mx,
        v_mx,
        all_perms=perms_mx,
        all_inv_perms=inv_perms_mx,
        query_positions=qp_mx,
        window=window,
    )
    mx.eval(y_mx)
    y_actual = np.asarray(y_mx)

    # --- Compute reference for each batch / head ---
    kv_repeat = Hq // Hkv
    y_ref = np.zeros_like(y_actual, dtype=np.float64)
    for b in range(B):
        for h in range(Hq):
            kv_h = h // kv_repeat
            y_ref[b, h] = _reference_active_row_attention(
                q_np[b, h],
                k_np[b, kv_h],
                v_np[b, kv_h],
                perms[h],
                inv_perms[h],
                qp,
                window,
            )

    # --- Compare ---
    diff = np.abs(y_actual.astype(np.float64) - y_ref)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    assert max_err < max_atol, (
        f"Max absolute error {max_err:.6e} exceeds threshold {max_atol:.1e} "
        f"(Tq={Tq}, Tk={Tk}, window={window})"
    )
    assert mean_err < mean_atol, (
        f"Mean absolute error {mean_err:.6e} exceeds threshold {mean_atol:.1e} "
        f"(Tq={Tq}, Tk={Tk}, window={window})"
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestK4ActiveRowSmall:
    """Small synthetic test: B=1, Hq=2, Hkv=2, dh=32, Tq=4, Tk=64, W=8."""

    def test_basic_correctness(self) -> None:
        _run_comparison(
            B=1, Hq=2, Hkv=2, dh=32, Tq=4, Tk=64, window=8,
        )


class TestK4ActiveRowScaling:
    """Scaling tests from the execution prompt."""

    def test_tq1_tk1024(self) -> None:
        _run_comparison(
            B=1, Hq=2, Hkv=2, dh=32, Tq=1, Tk=1024, window=8,
        )

    def test_tq32_tk4096(self) -> None:
        _run_comparison(
            B=1, Hq=2, Hkv=2, dh=32, Tq=32, Tk=4096, window=8,
        )

    def test_tq256_tk16384(self) -> None:
        _run_comparison(
            B=1, Hq=2, Hkv=2, dh=32, Tq=256, Tk=16384, window=8,
            # Slightly relaxed for large-scale numerical accumulation.
            max_atol=1e-3,
            mean_atol=1e-4,
        )


class TestK4ActiveRowEdgeCases:
    """Edge cases: single query at position 0, GQA head ratio."""

    def test_query_at_position_zero(self) -> None:
        """Query at position 0 should attend only to itself."""
        rng = np.random.RandomState(99)
        B, Hq, Hkv, dh, Tk, window = 1, 1, 1, 16, 32, 4
        Tg = Tk

        q_np = rng.randn(B, Hq, 1, dh).astype(np.float32)
        k_np = rng.randn(B, Hkv, Tk, dh).astype(np.float32)
        v_np = rng.randn(B, Hkv, Tk, dh).astype(np.float32)

        perm = rng.permutation(Tg).astype(np.int32)
        inv_perm = np.argsort(perm).astype(np.int32)
        perms = perm.reshape(1, Tg)
        inv_perms = inv_perm.reshape(1, Tg)

        qp = np.array([0], dtype=np.int32)

        y_mx, _ = butterfly_permute_window_attention_active_batched(
            mx.array(q_np),
            mx.array(k_np),
            mx.array(v_np),
            all_perms=mx.array(perms),
            all_inv_perms=mx.array(inv_perms),
            query_positions=mx.array(qp),
            window=window,
        )
        mx.eval(y_mx)
        y_actual = np.asarray(y_mx).astype(np.float64)

        # Position 0 can only attend to itself (causal: j <= 0 means j == 0).
        # So the output must equal v[0] regardless of permutation.
        v0 = v_np[0, 0, 0].astype(np.float64)
        diff = np.abs(y_actual[0, 0, 0] - v0)
        assert diff.max() < 1e-5, f"Position-0 output should be v[0], max diff={diff.max():.6e}"

    def test_gqa_hq4_hkv2(self) -> None:
        """GQA with Hq=4, Hkv=2 (repeat factor 2)."""
        _run_comparison(
            B=1, Hq=4, Hkv=2, dh=32, Tq=8, Tk=128, window=8,
        )

    def test_larger_window(self) -> None:
        """Window larger than Tk -- all positions visible to each query."""
        _run_comparison(
            B=1, Hq=2, Hkv=2, dh=32, Tq=4, Tk=16, window=32,
            max_atol=1e-4,
            mean_atol=1e-5,
        )
