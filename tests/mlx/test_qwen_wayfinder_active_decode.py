"""Qwen wayfinder active decode integration checks with lightweight mock attention."""
from __future__ import annotations

import numpy as np
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


def test_qwen_wayfinder_active_decode_avoids_dense_fallback():
    base_attn = _MockQwenAttn()
    cfg = QwenWayfinderConfig(
        path="permute",
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
    attn = QwenWayfinderAttention(base_attn, cfg)
    cache = _MockKVCache()

    x_prefill = mx.random.normal((1, 16, 64))
    y_prefill = attn(x_prefill, cache=cache)
    mx.eval(y_prefill)
    assert y_prefill.shape == (1, 16, 64)

    x_decode = mx.random.normal((1, 1, 64))
    y_decode = attn(x_decode, cache=cache)
    mx.eval(y_decode)
    assert y_decode.shape == (1, 1, 64)

    notes = attn.last_profile.notes
    assert attn.last_profile.path == "permute"
    assert "dense_fallback" not in attn.last_profile.path
    assert bool(notes.get("active_query_mode", False))
    assert int(notes.get("q_len", 0)) < int(notes.get("seq_len", 0))
    assert "graph_seq_len" in notes
    assert int(notes.get("graph_seq_len", 0)) >= int(notes.get("seq_len", 0))
    assert notes.get("cache_source") != "dense_fallback"
    assert notes.get("dense_fallback_reason") in (None, "")

    y_np = np.asarray(y_decode)
    assert np.isfinite(y_np).all()


def test_qwen_wayfinder_active_decode_dense_backend_forces_fallback():
    base_attn = _MockQwenAttn()
    cfg = QwenWayfinderConfig(
        path="permute",
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
        wayfinder_decode_backend="dense",
    )
    attn = QwenWayfinderAttention(base_attn, cfg)
    cache = _MockKVCache()

    x_prefill = mx.random.normal((1, 16, 64))
    y_prefill = attn(x_prefill, cache=cache)
    mx.eval(y_prefill)
    assert y_prefill.shape == (1, 16, 64)

    x_decode = mx.random.normal((1, 1, 64))
    y_decode = attn(x_decode, cache=cache)
    mx.eval(y_decode)
    assert y_decode.shape == (1, 1, 64)

    notes = attn.last_profile.notes
    assert "dense_fallback" in attn.last_profile.path
    assert notes.get("dense_fallback_reason") == "wayfinder_decode_dense"
    assert bool(notes.get("wayfinder_decode_dense_triggered", False))
