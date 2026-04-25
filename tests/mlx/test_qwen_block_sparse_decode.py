"""Qwen MLX block-sparse Butterfly decode checks with lightweight mock attention."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from bna.integrations.qwen_mlx import QwenButterflyAttention, QwenButterflyConfig  # noqa: E402


class _MockIdentity(nn.Module):
    def __call__(self, x):
        return x


class _MockRoPE(nn.Module):
    def __call__(self, x, offset=0):
        return x


class _MockKVCache:
    def __init__(self):
        self.offset = 0
        self._keys = None
        self._values = None
        self.max_size = None

    def update_and_fetch(self, keys, values):
        if self._keys is None:
            self._keys = keys
            self._values = values
        else:
            self._keys = mx.concatenate([self._keys, keys], axis=2)
            self._values = mx.concatenate([self._values, values], axis=2)
        self.offset = int(self._keys.shape[2])
        return self._keys, self._values


class _MockQwenAttn(nn.Module):
    def __init__(self, *, n_heads: int = 4, n_kv_heads: int = 2, head_dim: int = 16):
        super().__init__()
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        hidden_size = int(n_heads * head_dim)
        self.scale = float(head_dim ** -0.5)

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = _MockIdentity()
        self.k_norm = _MockIdentity()
        self.rope = _MockRoPE()


def _make_qwen_attn_and_cfg(**cfg_overrides):
    base_attn = _MockQwenAttn()
    cfg_defaults = dict(
        path="block_sparse",
        strategy="random",
        window=8,
        landmark_stride=8,
        num_cycles=1,
        edge_disjoint=True,
        enforce_hamiltonian=True,
        seed=42,
        edge_bias=False,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        block_size=4,
        block_local_window_blocks=1,
        block_partner_count=1,
        block_sink_blocks=1,
        block_partner_rule="xor",
    )
    cfg_defaults.update(cfg_overrides)
    return base_attn, QwenButterflyConfig(**cfg_defaults)


def test_qwen_block_sparse_prefill_decode_stays_on_block_sparse_path() -> None:
    base_attn, cfg = _make_qwen_attn_and_cfg()
    attn = QwenButterflyAttention(base_attn, cfg, layer_idx=5)
    cache = _MockKVCache()

    x0 = mx.random.normal((1, 8, 64))
    y0 = attn(x0, cache=cache)
    mx.eval(y0)
    assert y0.shape == (1, 8, 64)
    assert attn.last_profile.path == "block_sparse"
    assert not bool(attn.last_profile.notes.get("block_sparse_active_mode", False))
    assert attn.last_profile.notes.get("block_sparse_stage") is not None
    assert attn.last_profile.notes.get("dense_fallback_reason") in (None, "")

    x1 = mx.random.normal((1, 4, 64))
    y1 = attn(x1, cache=cache)
    mx.eval(y1)
    assert y1.shape == (1, 4, 64)
    assert attn.last_profile.path == "block_sparse"
    assert bool(attn.last_profile.notes.get("block_sparse_active_mode", False))
    assert bool(attn.last_profile.notes.get("active_query_mode", False))
    assert attn.last_profile.notes.get("cache_source") != "dense_fallback"
    assert attn.last_profile.notes.get("block_partner_rule") == "xor"

    x_dec = mx.random.normal((1, 1, 64))
    y_dec = attn(x_dec, cache=cache)
    mx.eval(y_dec)
    assert y_dec.shape == (1, 1, 64)
    assert attn.last_profile.path == "block_sparse"
    assert bool(attn.last_profile.notes.get("block_sparse_active_mode", False))
    assert attn.last_profile.notes.get("dense_fallback_reason") in (None, "")


def test_qwen_block_sparse_stock_decode_uses_dense_fallback() -> None:
    base_attn, cfg = _make_qwen_attn_and_cfg(wayfinder_decode_backend="dense")
    attn = QwenButterflyAttention(base_attn, cfg, layer_idx=5)
    cache = _MockKVCache()

    x0 = mx.random.normal((1, 8, 64))
    y0 = attn(x0, cache=cache)
    mx.eval(y0)
    assert y0.shape == (1, 8, 64)
    assert attn.last_profile.path == "block_sparse"

    x_dec = mx.random.normal((1, 1, 64))
    y_dec = attn(x_dec, cache=cache)
    mx.eval(y_dec)
    assert y_dec.shape == (1, 1, 64)
    assert attn.last_profile.path == "block_sparse_dense_fallback"
    assert attn.last_profile.notes.get("dense_fallback_reason") == "butterfly_decode_stock"
    assert bool(attn.last_profile.notes.get("butterfly_decode_stock_triggered"))
