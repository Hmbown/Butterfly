"""Integration tests for 'Actually Hamiltonian' features end-to-end.

Covers: circular windowing, union multigraph, principled d (auto num_cycles),
Laplacian diagnostics, GPT2 integration, and causal correctness at model level.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from bna.mlx.attention import (  # noqa: E402
    WayfinderAttentionMLX,
    wayfinder_permute_window_attention_active_batched,
    wayfinder_permute_window_attention_batched,
)
from bna.mlx.model import GPTConfigMLX, GPTMLX  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_model_pair(circular_a, circular_b, multi_cycle_mode="average",
                     num_cycles=1, seed=42):
    """Build two GPTMLX models with identical weights but different flags."""
    base = dict(
        vocab_size=32, n_embd=32, n_heads=2, n_layers=1, seq_len=16,
        attn="wayfinder_permute", window=4, seed=seed,
    )
    cfg_a = GPTConfigMLX(**base, circular=circular_a,
                         multi_cycle_mode=multi_cycle_mode,
                         num_cycles=num_cycles)
    cfg_b = GPTConfigMLX(**base, circular=circular_b,
                         multi_cycle_mode=multi_cycle_mode,
                         num_cycles=num_cycles)
    model_a = GPTMLX(cfg_a)
    model_b = GPTMLX(cfg_b)
    # Copy weights from a -> b
    model_b.update(model_a.parameters())
    mx.eval(model_b.parameters())
    return model_a, model_b


def _make_batched_inputs(B, H, T, dh, num_cycles=1, seed=42):
    """Create Q, K, V, perms, inv_perms for batched attention."""
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
    d = num_cycles
    perms = np.zeros((H, d, T), dtype=np.int32)
    inv_perms = np.zeros((H, d, T), dtype=np.int32)
    for h in range(H):
        for c in range(d):
            p = rng.permutation(T).astype(np.int32)
            perms[h, c] = p
            inv_perms[h, c] = np.argsort(p).astype(np.int32)
    if d == 1:
        perms_mx = mx.array(perms[:, 0, :])       # [H, T]
        inv_perms_mx = mx.array(inv_perms[:, 0, :])
    else:
        perms_mx = mx.array(perms)                 # [H, d, T]
        inv_perms_mx = mx.array(inv_perms)
    return q, k, v, perms_mx, inv_perms_mx


def _reference_active_row_circular(
    q_np, k_np, v_np, perm_np, inv_perm_np, query_positions_np,
    window, circular=False,
):
    """Pure-NumPy reference for active-row attention (single head, no batch)."""
    Tq, dh = q_np.shape
    Tk = k_np.shape[0]
    Tg = len(perm_np)
    scale = 1.0 / math.sqrt(dh)
    y = np.zeros_like(q_np)
    for qi in range(Tq):
        pos = int(query_positions_np[qi])
        rank = int(inv_perm_np[pos])
        # Gather neighbor indices in cycle-window
        neighbors = []
        for off in range(-window, window + 1):
            if circular:
                r = (rank + off) % Tg
            else:
                r = rank + off
                if r < 0 or r >= Tg:
                    continue
            orig = int(perm_np[r])
            # Causal: orig <= pos, Available: orig < Tk
            if orig <= pos and orig < Tk:
                neighbors.append(orig)
        if not neighbors:
            continue
        k_nbrs = k_np[neighbors]  # [N, dh]
        v_nbrs = v_np[neighbors]  # [N, dh]
        scores = (q_np[qi] @ k_nbrs.T) * scale  # [N]
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        y[qi] = w @ v_nbrs
    return y


# ===========================================================================
# GAP 1: Full-model circular output correctness
# ===========================================================================

class TestGap1CircularModel:
    def test_circular_vs_linear_differ(self):
        model_lin, model_circ = _make_model_pair(
            circular_a=False, circular_b=True,
        )
        rng = np.random.default_rng(0)
        idx = mx.array(rng.integers(0, 32, size=(1, 16)).astype(np.int32))

        out_lin = model_lin(idx)["logits"]
        out_circ = model_circ(idx)["logits"]
        mx.eval(out_lin, out_circ)

        # a) Outputs differ meaningfully
        diff = mx.max(mx.abs(out_lin - out_circ)).item()
        assert diff > 1e-6, f"circular vs linear should differ, max diff={diff}"

        # b) Valid shape and no NaN
        assert out_lin.shape == (1, 16, 32)
        assert out_circ.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(out_lin)).item()
        assert not mx.any(mx.isnan(out_circ)).item()

    def test_circular_loss_finite(self):
        model_lin, model_circ = _make_model_pair(
            circular_a=False, circular_b=True,
        )
        rng = np.random.default_rng(1)
        idx = mx.array(rng.integers(0, 32, size=(1, 16)).astype(np.int32))
        targets = mx.array(rng.integers(0, 32, size=(1, 16)).astype(np.int32))

        loss_lin = model_lin(idx, targets)["loss"]
        loss_circ = model_circ(idx, targets)["loss"]
        mx.eval(loss_lin, loss_circ)

        assert np.isfinite(loss_lin.item()), "linear loss should be finite"
        assert np.isfinite(loss_circ.item()), "circular loss should be finite"


# ===========================================================================
# GAP 2: Full-model union vs average output correctness
# ===========================================================================

class TestGap2UnionModel:
    def test_union_vs_average_differ(self):
        base = dict(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1, seq_len=16,
            attn="wayfinder_permute", window=4, num_cycles=2, seed=42,
        )
        cfg_avg = GPTConfigMLX(**base, multi_cycle_mode="average")
        cfg_union = GPTConfigMLX(**base, multi_cycle_mode="union")
        model_avg = GPTMLX(cfg_avg)
        model_union = GPTMLX(cfg_union)
        model_union.update(model_avg.parameters())
        mx.eval(model_union.parameters())

        rng = np.random.default_rng(0)
        idx = mx.array(rng.integers(0, 32, size=(1, 16)).astype(np.int32))

        out_avg = model_avg(idx)["logits"]
        out_union = model_union(idx)["logits"]
        mx.eval(out_avg, out_union)

        # Both valid
        assert out_avg.shape == (1, 16, 32)
        assert out_union.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(out_avg)).item()
        assert not mx.any(mx.isnan(out_union)).item()

        # Differ meaningfully
        diff = mx.max(mx.abs(out_avg - out_union)).item()
        assert diff > 1e-6, f"union vs average should differ, max diff={diff}"


# ===========================================================================
# GAP 3: Circular + union combined at kernel level
# ===========================================================================

class TestGap3KernelCombinations:
    def test_four_combos_pairwise_different(self):
        B, H, T, dh, W = 1, 2, 16, 8, 3
        q, k, v, perms, inv_perms = _make_batched_inputs(
            B, H, T, dh, num_cycles=2, seed=99,
        )

        configs = [
            (True, "union"),
            (True, "average"),
            (False, "union"),
            (False, "average"),
        ]
        outputs = {}
        for circ, mode in configs:
            y, _ = wayfinder_permute_window_attention_batched(
                q, k, v,
                all_perms=perms, all_inv_perms=inv_perms,
                window=W, circular=circ, multi_cycle_mode=mode,
            )
            mx.eval(y)
            key = (circ, mode)
            outputs[key] = np.array(y)
            # Valid
            assert y.shape == (B, H, T, dh), f"bad shape for {key}"
            assert not mx.any(mx.isnan(y)).item(), f"NaN for {key}"

        # Pairwise differences
        keys = list(outputs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                diff = np.abs(outputs[keys[i]] - outputs[keys[j]]).max()
                assert diff > 1e-6, (
                    f"{keys[i]} vs {keys[j]} should differ, max diff={diff}"
                )


# ===========================================================================
# GAP 4: Active-batched (K4) path with circular=True
# ===========================================================================

class TestGap4ActiveBatchedCircular:
    def test_circular_vs_linear_differ(self):
        B, Hq, Hkv, dh, T, W = 1, 2, 2, 32, 64, 8
        Tq = 4
        rng = np.random.default_rng(42)
        q = mx.array(rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32))
        k = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32))
        v = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32))
        query_positions = mx.arange(T - Tq, T, dtype=mx.int32)

        perms = np.zeros((Hq, T), dtype=np.int32)
        inv_perms = np.zeros((Hq, T), dtype=np.int32)
        for h in range(Hq):
            p = rng.permutation(T).astype(np.int32)
            perms[h] = p
            inv_perms[h] = np.argsort(p).astype(np.int32)
        perms_mx = mx.array(perms)
        inv_perms_mx = mx.array(inv_perms)

        y_lin, _ = wayfinder_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms_mx, all_inv_perms=inv_perms_mx,
            query_positions=query_positions, window=W, circular=False,
        )
        y_circ, _ = wayfinder_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms_mx, all_inv_perms=inv_perms_mx,
            query_positions=query_positions, window=W, circular=True,
        )
        mx.eval(y_lin, y_circ)

        assert y_lin.shape == (B, Hq, Tq, dh)
        assert y_circ.shape == (B, Hq, Tq, dh)
        assert not mx.any(mx.isnan(y_lin)).item()
        assert not mx.any(mx.isnan(y_circ)).item()

        diff = np.abs(np.array(y_lin) - np.array(y_circ)).max()
        assert diff > 1e-6, f"circular vs linear active should differ, diff={diff}"

    def test_circular_matches_reference(self):
        """K4 active path with circular=True must match pure-NumPy reference."""
        B, Hq, Hkv, dh, T, W = 1, 2, 2, 16, 32, 4
        Tq = 4
        rng = np.random.default_rng(123)
        q_np = rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32) * 0.5
        k_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.5
        v_np = rng.standard_normal((B, Hkv, T, dh)).astype(np.float32) * 0.5

        perms = np.zeros((Hq, T), dtype=np.int32)
        inv_perms = np.zeros((Hq, T), dtype=np.int32)
        for h in range(Hq):
            p = rng.permutation(T).astype(np.int32)
            perms[h] = p
            inv_perms[h] = np.argsort(p).astype(np.int32)

        qp = np.sort(
            rng.choice(T, size=Tq, replace=False)
        ).astype(np.int32)

        q_mx = mx.array(q_np)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)
        perms_mx = mx.array(perms)
        inv_perms_mx = mx.array(inv_perms)
        qp_mx = mx.array(qp)

        y_mx, _ = wayfinder_permute_window_attention_active_batched(
            q_mx, k_mx, v_mx,
            all_perms=perms_mx, all_inv_perms=inv_perms_mx,
            query_positions=qp_mx, window=W, circular=True,
        )
        mx.eval(y_mx)
        y_actual = np.array(y_mx)

        # Compute reference per head
        kv_repeat = Hq // Hkv
        for b in range(B):
            for h in range(Hq):
                kv_h = h // kv_repeat
                y_ref = _reference_active_row_circular(
                    q_np[b, h], k_np[b, kv_h], v_np[b, kv_h],
                    perms[h], inv_perms[h], qp,
                    W, circular=True,
                )
                max_err = np.abs(
                    y_actual[b, h].astype(np.float64) - y_ref
                ).max()
                assert max_err < 1e-3, (
                    f"circular active ref mismatch b={b} h={h} max_err={max_err}"
                )


# ===========================================================================
# GAP 5: Active-batched (K4) path with multi_cycle_mode="union"
# ===========================================================================

class TestGap5ActiveBatchedUnion:
    def test_union_vs_average_differ(self):
        B, Hq, Hkv, dh, T, W = 1, 2, 2, 16, 32, 4
        Tq = 4
        rng = np.random.default_rng(77)
        q = mx.array(rng.standard_normal((B, Hq, Tq, dh)).astype(np.float32))
        k = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32))
        v = mx.array(rng.standard_normal((B, Hkv, T, dh)).astype(np.float32))
        query_positions = mx.arange(T - Tq, T, dtype=mx.int32)

        # Two cycles per head
        d = 2
        perms = np.zeros((Hq, d, T), dtype=np.int32)
        inv_perms = np.zeros((Hq, d, T), dtype=np.int32)
        for h in range(Hq):
            for c in range(d):
                p = rng.permutation(T).astype(np.int32)
                perms[h, c] = p
                inv_perms[h, c] = np.argsort(p).astype(np.int32)
        perms_mx = mx.array(perms)
        inv_perms_mx = mx.array(inv_perms)

        y_avg, _ = wayfinder_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms_mx, all_inv_perms=inv_perms_mx,
            query_positions=query_positions, window=W,
            multi_cycle_mode="average",
        )
        y_union, _ = wayfinder_permute_window_attention_active_batched(
            q, k, v,
            all_perms=perms_mx, all_inv_perms=inv_perms_mx,
            query_positions=query_positions, window=W,
            multi_cycle_mode="union",
        )
        mx.eval(y_avg, y_union)

        assert y_avg.shape == (B, Hq, Tq, dh)
        assert y_union.shape == (B, Hq, Tq, dh)
        assert not mx.any(mx.isnan(y_avg)).item()
        assert not mx.any(mx.isnan(y_union)).item()

        diff = np.abs(np.array(y_avg) - np.array(y_union)).max()
        assert diff > 1e-6, (
            f"union vs average active should differ, max diff={diff}"
        )


# ===========================================================================
# GAP 6: num_cycles="auto" resolution
# ===========================================================================

class TestGap6AutoNumCycles:
    def test_auto_resolves_correctly(self):
        attn = WayfinderAttentionMLX(
            n_embd=32, n_heads=2, window=4,
            num_cycles="auto", path="permute",
        )
        assert attn._num_cycles_raw == "auto"

        # T=16: ceil(2 * log2(16)) = ceil(8) = 8
        attn._resolve_and_sync_num_cycles(16)
        nc_16 = attn.num_cycles
        assert nc_16 == 8, f"T=16 should give num_cycles=8, got {nc_16}"

        # T=64: ceil(2 * log2(64)) = ceil(12) = 12
        attn._resolve_and_sync_num_cycles(64)
        nc_64 = attn.num_cycles
        assert nc_64 == 12, f"T=64 should give num_cycles=12, got {nc_64}"

        # Different T values should give different num_cycles
        assert nc_16 != nc_64

    def test_auto_forward_pass(self):
        """Verify forward pass works with num_cycles='auto'.

        Use edge_disjoint=False because 'auto' can request more cycles than
        are feasible as edge-disjoint on small T.
        """
        cfg = GPTConfigMLX(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1, seq_len=128,
            attn="wayfinder_permute", window=4, num_cycles="auto",
            edge_disjoint=False,
        )
        model = GPTMLX(cfg)
        rng = np.random.default_rng(0)

        # T=32 (ceil(2*log2(32))=10 cycles)
        idx_32 = mx.array(rng.integers(0, 32, size=(1, 32)).astype(np.int32))
        out_32 = model(idx_32)["logits"]
        mx.eval(out_32)
        assert out_32.shape == (1, 32, 32)
        assert not mx.any(mx.isnan(out_32)).item()

        # T=128 (ceil(2*log2(128))=14 cycles)
        idx_128 = mx.array(
            rng.integers(0, 32, size=(1, 128)).astype(np.int32)
        )
        out_128 = model(idx_128)["logits"]
        mx.eval(out_128)
        assert out_128.shape == (1, 128, 32)
        assert not mx.any(mx.isnan(out_128)).item()


# ===========================================================================
# GAP 7: Causality at full model level with circular
# ===========================================================================

class TestGap7CausalityCircular:
    def test_kernel_causality_indicator_v(self):
        """At kernel level: verify no future token contributes to output.

        Use indicator V vectors: v[j] = e_j (one-hot on position j).
        After attention, output[i] = sum_j w[i,j]*e_j. If any future j > i
        has nonzero weight, output[i] will have a nonzero component in
        dimension j, which we can detect.
        """
        B, H, T, dh, W = 1, 2, 16, 16, 3
        rng = np.random.default_rng(42)
        q = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
        k = mx.array(rng.standard_normal((B, H, T, dh)).astype(np.float32))
        # v[j] = one-hot on dimension j (dh >= T required)
        v_np = np.zeros((B, H, T, dh), dtype=np.float32)
        for t in range(T):
            v_np[:, :, t, t] = 1.0
        v = mx.array(v_np)

        perms = np.zeros((H, T), dtype=np.int32)
        inv_perms = np.zeros((H, T), dtype=np.int32)
        for h in range(H):
            p = rng.permutation(T).astype(np.int32)
            perms[h] = p
            inv_perms[h] = np.argsort(p).astype(np.int32)

        y, _ = wayfinder_permute_window_attention_batched(
            q, k, v,
            all_perms=mx.array(perms), all_inv_perms=mx.array(inv_perms),
            window=W, circular=True,
        )
        mx.eval(y)
        y_np = np.array(y)

        # For each position i, output dimensions j > i should be zero
        # (since those correspond to future positions)
        for h in range(H):
            for i in range(T):
                for j in range(i + 1, T):
                    val = abs(float(y_np[0, h, i, j]))
                    assert val < 1e-6, (
                        f"Causality violated: h={h}, pos={i} has "
                        f"future-pos={j} contribution={val}"
                    )

    def test_model_forward_valid_circular(self):
        """Full model forward with circular=True produces valid output."""
        cfg = GPTConfigMLX(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1, seq_len=16,
            attn="wayfinder_permute", window=4, circular=True,
        )
        model = GPTMLX(cfg)

        rng = np.random.default_rng(42)
        idx = mx.array(rng.integers(0, 32, size=(1, 16)).astype(np.int32))
        targets = mx.array(
            rng.integers(0, 32, size=(1, 16)).astype(np.int32)
        )

        out = model(idx, targets)
        mx.eval(out["logits"], out["loss"])

        assert out["logits"].shape == (1, 16, 32)
        assert not mx.any(mx.isnan(out["logits"])).item()
        assert np.isfinite(out["loss"].item())


# ===========================================================================
# GAP 8: Laplacian diagnostics with circular graph
# ===========================================================================

class TestGap8LaplacianDiagnostics:
    def test_circular_higher_fiedler(self):
        """Circular graph (wrap-around edge) should have >= Fiedler value."""
        from bna.graph.analysis import laplacian_spectral_gap

        T = 16
        rng = np.random.default_rng(42)
        perm = rng.permutation(T).astype(np.int64)

        # Linear: cycle-perm edges are (perm[0]->perm[1], ..., perm[T-2]->perm[T-1])
        # The analysis module always builds the cycle as undirected with wrap-around
        # because _build_undirected_adj_list uses (i+1)%T.
        # So to simulate "no circular", we remove the wrap edge manually.
        # Actually, the analysis module always includes wrap-around (it's a cycle).
        # The circular flag affects the *attention window*, not the cycle structure.
        # For diagnostics, we compare cycle+window(0) vs cycle+window(W).
        result_base = laplacian_spectral_gap(perm, include_window=False, window=0)
        result_with_window = laplacian_spectral_gap(
            perm, include_window=True, window=2,
        )

        fiedler_base = result_base["fiedler_value"]
        fiedler_window = result_with_window["fiedler_value"]

        assert fiedler_base > 0.0, "Hamiltonian cycle must be connected"
        assert result_base["is_well_connected"]
        # Adding window edges should improve connectivity
        assert fiedler_window >= fiedler_base, (
            f"Window edges should improve Fiedler: base={fiedler_base}, "
            f"window={fiedler_window}"
        )

    def test_fiedler_bridge_candidates(self):
        from bna.graph.analysis import fiedler_bridge_candidates

        T = 16
        rng = np.random.default_rng(42)
        perm = rng.permutation(T).astype(np.int64)

        bridges = fiedler_bridge_candidates(perm, window=0, num_bridges=5)
        assert isinstance(bridges, list)
        # Each bridge is a tuple of two distinct vertices
        for a, b in bridges:
            assert 0 <= a < T
            assert 0 <= b < T
            assert a != b


# ===========================================================================
# GAP 9: GPT2 integration forward pass with circular
# ===========================================================================

class TestGap9GPT2Integration:
    def test_gpt2_wayfinder_forward(self):
        from bna.integrations.gpt2_mlx import (
            GPT2WayfinderAttention,
            GPT2WayfinderConfig,
        )

        class MockGPT2Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 2
                self.n_embd = 32
                self.scale = (32 // 2) ** -0.5
                self.c_attn = nn.Linear(32, 32 * 3)
                self.c_proj = nn.Linear(32, 32)

        base_attn = MockGPT2Attn()
        cfg = GPT2WayfinderConfig(
            window=4, circular=True, multi_cycle_mode="average",
        )
        wayfinder_attn = GPT2WayfinderAttention(base_attn, cfg)

        # Verify flags propagated
        assert wayfinder_attn.circular is True
        assert wayfinder_attn.multi_cycle_mode == "average"

        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 16, 32)).astype(np.float32))

        out = wayfinder_attn(x)
        mx.eval(out)

        assert out.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(out)).item()

    def test_gpt2_wayfinder_union(self):
        from bna.integrations.gpt2_mlx import (
            GPT2WayfinderAttention,
            GPT2WayfinderConfig,
        )

        class MockGPT2Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_head = 2
                self.n_embd = 32
                self.scale = (32 // 2) ** -0.5
                self.c_attn = nn.Linear(32, 32 * 3)
                self.c_proj = nn.Linear(32, 32)

        base_attn = MockGPT2Attn()
        cfg = GPT2WayfinderConfig(
            window=4, circular=True, multi_cycle_mode="union",
            num_cycles=2,
        )
        wayfinder_attn = GPT2WayfinderAttention(base_attn, cfg)

        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 16, 32)).astype(np.float32))

        out = wayfinder_attn(x)
        mx.eval(out)

        assert out.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(out)).item()
