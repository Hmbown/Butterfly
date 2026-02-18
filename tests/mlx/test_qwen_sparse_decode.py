"""Qwen sparse decode integration checks with lightweight mock attention."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from hcsa.integrations.qwen_mlx import QwenWayfinderAttention, QwenWayfinderConfig  # noqa: E402


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
        path="sparse",
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
    )
    cfg_defaults.update(cfg_overrides)
    cfg = QwenWayfinderConfig(**cfg_defaults)
    return base_attn, cfg


def test_qwen_sparse_chunked_prefill_decode_stays_sparse():
    base_attn, cfg = _make_qwen_attn_and_cfg(path="sparse")
    attn = QwenWayfinderAttention(base_attn, cfg)
    cache = _MockKVCache()

    x0 = mx.random.normal((1, 16, 64))
    y0 = attn(x0, cache=cache)
    mx.eval(y0)
    assert y0.shape == (1, 16, 64)
    assert attn.last_profile.path == "sparse"
    assert not bool(attn.last_profile.notes.get("sparse_active_mode", False))
    assert attn.last_profile.notes.get("cache_source") != "dense_fallback"
    assert "graph_seq_len" in attn.last_profile.notes
    assert "adaptive_graph_reuse" in attn.last_profile.notes
    assert attn.last_profile.notes.get("dense_fallback_reason") in (None, "")

    x1 = mx.random.normal((1, 16, 64))
    y1 = attn(x1, cache=cache)
    mx.eval(y1)
    assert y1.shape == (1, 16, 64)
    assert attn.last_profile.path == "sparse"
    assert bool(attn.last_profile.notes.get("sparse_active_mode", False))
    assert bool(attn.last_profile.notes.get("active_query_mode", False))
    assert attn.last_profile.notes.get("cache_source") != "dense_fallback"
    assert "graph_seq_len" in attn.last_profile.notes
    assert "adaptive_graph_reuse" in attn.last_profile.notes
    assert attn.last_profile.notes.get("dense_fallback_reason") in (None, "")

    x_dec = mx.random.normal((1, 1, 64))
    y_dec = attn(x_dec, cache=cache)
    mx.eval(y_dec)
    assert y_dec.shape == (1, 1, 64)
    assert attn.last_profile.path == "sparse"
    assert bool(attn.last_profile.notes.get("sparse_active_mode", False))
    assert bool(attn.last_profile.notes.get("active_query_mode", False))
    assert attn.last_profile.notes.get("cache_source") != "dense_fallback"
    assert "graph_seq_len" in attn.last_profile.notes
    assert "adaptive_graph_reuse" in attn.last_profile.notes
    assert attn.last_profile.notes.get("dense_fallback_reason") in (None, "")
    assert "dense_fallback" not in attn.last_profile.path
