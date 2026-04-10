"""Lightweight E2E tests for GLM integration with Hamiltonian features.

Tests the actual GLMButterflyAttention code path with synthetic inputs
that match GLM-4.7-Flash MLA dimensions — no model loading required.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from bna.integrations.glm_mlx import (  # noqa: E402
    GLMButterflyAttention,
    GLMButterflyConfig,
)


# ---------------------------------------------------------------------------
# Mock GLM attention module matching GLM-4.7-Flash MLA dimensions
# ---------------------------------------------------------------------------

class _MockRoPE(nn.Module):
    """No-op RoPE that preserves shape."""
    def __call__(self, x, offset=0):
        return x


class _MockKVCache:
    """Minimal KV cache for testing active-row (Q_len < K_len) path."""
    def __init__(self):
        self.offset = 0
        self._keys = None
        self.max_size = None

    def update_and_fetch(self, keys, values):
        if self._keys is None:
            self._keys = keys
        else:
            self._keys = mx.concatenate([self._keys, keys], axis=2)
        self.offset = int(self._keys.shape[2])
        return self._keys, None


class MockGLMAttn(nn.Module):
    """Mock GLM-4.7-Flash MLA attention with realistic dimensions.

    Real GLM dims: num_heads=32, q_head_dim=192, qk_nope_head_dim=128,
    qk_rope_head_dim=64, kv_lora_rank=512, hidden_size=4096.
    We use scaled-down versions for speed.
    """
    def __init__(self, *, num_heads=4, hidden_size=256,
                 qk_nope_head_dim=32, qk_rope_head_dim=16,
                 kv_lora_rank=64):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 48
        self.kv_lora_rank = kv_lora_rank
        self.scale = self.q_head_dim ** -0.5

        # Q projection (no LoRA for simplicity)
        self.q_lora_rank = None
        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.q_head_dim, bias=False,
        )

        # KV compressed projection: output = kv_lora_rank + qk_rope_head_dim
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank)

        # q nope -> latent embed (qk_nope_head_dim -> kv_lora_rank per head)
        self.embed_q = nn.Linear(
            qk_nope_head_dim, kv_lora_rank, bias=False,
        )

        # Value unembed (kv_lora_rank -> v_head_dim per head)
        v_head_dim = hidden_size // num_heads
        self.unembed_out = nn.Linear(kv_lora_rank, v_head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = _MockRoPE()


def _make_glm_attn_and_cfg(**cfg_overrides):
    """Create a mock GLM attention and GLMButterflyConfig."""
    base_attn = MockGLMAttn()
    cfg_defaults = dict(
        window=8, permute_head_chunk_size=2, query_chunk_size=64,
        active_dense_threshold=0,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        edge_bias=False,
    )
    cfg_defaults.update(cfg_overrides)
    cfg = GLMButterflyConfig(**cfg_defaults)
    return base_attn, cfg


# ===========================================================================
# Tests
# ===========================================================================

class TestGLMPrefill:
    """Full-sequence prefill (Q_len == K_len) path."""

    def test_baseline_forward(self):
        base_attn, cfg = _make_glm_attn_and_cfg(
            circular=False, multi_cycle_mode="average",
        )
        attn = GLMButterflyAttention(base_attn, cfg)
        x = mx.random.normal((1, 64, 256))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 64, 256)
        assert not mx.any(mx.isnan(out)).item()

    def test_circular_forward(self):
        base_attn, cfg = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="average",
        )
        attn = GLMButterflyAttention(base_attn, cfg)
        x = mx.random.normal((1, 64, 256))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 64, 256)
        assert not mx.any(mx.isnan(out)).item()

    def test_circular_differs_from_linear(self):
        """Same weights, circular vs linear should produce different output."""
        base_attn_a, cfg_a = _make_glm_attn_and_cfg(
            circular=False, multi_cycle_mode="average",
        )
        attn_a = GLMButterflyAttention(base_attn_a, cfg_a)

        base_attn_b, cfg_b = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="average",
        )
        attn_b = GLMButterflyAttention(base_attn_b, cfg_b)
        # Copy weights
        attn_b.update(attn_a.parameters())
        mx.eval(attn_b.parameters())

        x = mx.random.normal((1, 64, 256))
        out_a = attn_a(x)
        out_b = attn_b(x)
        mx.eval(out_a, out_b)

        diff = mx.max(mx.abs(out_a - out_b)).item()
        assert diff > 1e-6, f"circular should differ from linear, diff={diff}"

    def test_union_forward(self):
        base_attn, cfg = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="union", num_cycles=2,
        )
        attn = GLMButterflyAttention(base_attn, cfg)
        x = mx.random.normal((1, 64, 256))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 64, 256)
        assert not mx.any(mx.isnan(out)).item()

    def test_union_differs_from_average(self):
        """Union vs average multi-cycle mode should differ."""
        base_attn_a, cfg_a = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="average", num_cycles=2,
        )
        attn_a = GLMButterflyAttention(base_attn_a, cfg_a)

        base_attn_b, cfg_b = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="union", num_cycles=2,
        )
        attn_b = GLMButterflyAttention(base_attn_b, cfg_b)
        attn_b.update(attn_a.parameters())
        mx.eval(attn_b.parameters())

        x = mx.random.normal((1, 64, 256))
        out_a = attn_a(x)
        out_b = attn_b(x)
        mx.eval(out_a, out_b)

        diff = mx.max(mx.abs(out_a - out_b)).item()
        assert diff > 1e-6, f"union should differ from average, diff={diff}"


class TestGLMChunkedPrefillDecode:
    """Chunked prefill + decode (active-row K4 path)."""

    def test_chunked_prefill_then_decode(self):
        """Simulate chunked prefill + decode without loading real model."""
        base_attn, cfg = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="average",
        )
        attn = GLMButterflyAttention(base_attn, cfg)

        T_total = 128
        chunk_size = 32

        # Chunked prefill
        cache = _MockKVCache()
        for start in range(0, T_total, chunk_size):
            end = min(start + chunk_size, T_total)
            x_chunk = mx.random.normal((1, end - start, 256))
            out = attn(x_chunk, cache=cache)
            mx.eval(out)
            assert out.shape == (1, end - start, 256), (
                f"chunk [{start}:{end}] shape mismatch"
            )
            assert not mx.any(mx.isnan(out)).item(), (
                f"NaN in chunk [{start}:{end}]"
            )

        # Verify cache offset matches total
        assert cache.offset == T_total

        # Decode steps (Q_len=1, K_len>1 → active-row path)
        for step in range(4):
            x_decode = mx.random.normal((1, 1, 256))
            out = attn(x_decode, cache=cache)
            mx.eval(out)
            assert out.shape == (1, 1, 256), f"decode step {step} shape"
            assert not mx.any(mx.isnan(out)).item(), (
                f"NaN at decode step {step}"
            )

        # Check profile shows the right path
        prof = attn.last_profile
        # Decode uses either active permute or dense fallback
        assert "permute" in prof.path or "dense" in prof.path

    def test_chunked_prefill_circular_with_union(self):
        """Chunked prefill + decode with circular + union d=2."""
        base_attn, cfg = _make_glm_attn_and_cfg(
            circular=True, multi_cycle_mode="union", num_cycles=2,
        )
        attn = GLMButterflyAttention(base_attn, cfg)

        cache = _MockKVCache()
        # Prefill in 2 chunks
        for _ in range(2):
            x = mx.random.normal((1, 32, 256))
            out = attn(x, cache=cache)
            mx.eval(out)
            assert not mx.any(mx.isnan(out)).item()

        # 2 decode steps
        for _ in range(2):
            x = mx.random.normal((1, 1, 256))
            out = attn(x, cache=cache)
            mx.eval(out)
            assert out.shape == (1, 1, 256)
            assert not mx.any(mx.isnan(out)).item()

    def test_sparse_chunked_prefill_decode_stays_sparse(self):
        """Sparse mode should not fall back to dense when q_len < k_len."""
        base_attn, cfg = _make_glm_attn_and_cfg(
            path="sparse",
            circular=False,
            multi_cycle_mode="average",
        )
        attn = GLMButterflyAttention(base_attn, cfg)
        cache = _MockKVCache()

        # First prefill chunk: q_len == k_len (regular sparse path)
        x0 = mx.random.normal((1, 32, 256))
        y0 = attn(x0, cache=cache)
        mx.eval(y0)
        assert y0.shape == (1, 32, 256)
        assert attn.last_profile.path == "sparse"
        assert not bool(attn.last_profile.notes.get("sparse_active_mode", False))

        # Second prefill chunk: q_len < k_len (active sparse path)
        x1 = mx.random.normal((1, 32, 256))
        y1 = attn(x1, cache=cache)
        mx.eval(y1)
        assert y1.shape == (1, 32, 256)
        assert attn.last_profile.path == "sparse"
        assert bool(attn.last_profile.notes.get("sparse_active_mode", False))
        assert bool(attn.last_profile.notes.get("active_query_mode", False))

        # Decode: q_len=1, k_len>1 (active sparse path must remain sparse)
        x_dec = mx.random.normal((1, 1, 256))
        y_dec = attn(x_dec, cache=cache)
        mx.eval(y_dec)
        assert y_dec.shape == (1, 1, 256)
        assert attn.last_profile.path == "sparse"
        assert bool(attn.last_profile.notes.get("sparse_active_mode", False))
        assert "dense_fallback" not in attn.last_profile.path


class TestGLMConfigPropagation:
    """Verify config flags propagate correctly."""

    def test_circular_flag(self):
        base_attn, cfg = _make_glm_attn_and_cfg(circular=True)
        attn = GLMButterflyAttention(base_attn, cfg)
        assert attn.circular is True

    def test_multi_cycle_mode_flag(self):
        base_attn, cfg = _make_glm_attn_and_cfg(multi_cycle_mode="union")
        attn = GLMButterflyAttention(base_attn, cfg)
        assert attn.multi_cycle_mode == "union"

    def test_defaults(self):
        base_attn, cfg = _make_glm_attn_and_cfg()
        attn = GLMButterflyAttention(base_attn, cfg)
        assert attn.circular is False
        assert attn.multi_cycle_mode == "average"
