"""KV Cache tests: verify cache operations and consistency."""

from __future__ import annotations

import torch

from bna.kv_cache import KVCache, LayerCaches


def test_kv_cache_update() -> None:
    """Cache should grow as new K, V are appended."""
    cache = KVCache()
    assert cache.seq_len == 0

    B, H, dh = 2, 4, 8
    k1 = torch.randn(B, H, 5, dh)
    v1 = torch.randn(B, H, 5, dh)
    cache.update(k1, v1)
    assert cache.seq_len == 5

    k2 = torch.randn(B, H, 3, dh)
    v2 = torch.randn(B, H, 3, dh)
    cache.update(k2, v2)
    assert cache.seq_len == 8

    # Values should match concatenation
    assert cache.k is not None
    assert torch.allclose(cache.k[:, :, :5], k1)
    assert torch.allclose(cache.k[:, :, 5:], k2)


def test_kv_cache_gather_sparse() -> None:
    """Sparse gather should retrieve correct cached values."""
    B, H, T, dh = 1, 2, 8, 4
    cache = KVCache()
    k = torch.randn(B, H, T, dh)
    v = torch.randn(B, H, T, dh)
    cache.update(k, v)

    # Gather for head 0 with specific neighbor indices
    neigh_idx = torch.tensor([0, 3, 5, -1])  # -1 = invalid
    k_n, v_n = cache.gather_for_sparse(neigh_idx, head_idx=0)

    assert k_n.shape == (B, 4, dh)
    # Valid indices should match
    assert torch.allclose(k_n[:, 0], k[:, 0, 0])
    assert torch.allclose(k_n[:, 1], k[:, 0, 3])
    assert torch.allclose(k_n[:, 2], k[:, 0, 5])
    # Invalid index (-1 clamped to 0) should give position 0
    assert torch.allclose(k_n[:, 3], k[:, 0, 0])


def test_layer_caches_create() -> None:
    """LayerCaches should create the right number of caches."""
    lc = LayerCaches.create(6)
    assert len(lc) == 6
    assert all(c.seq_len == 0 for c in lc.caches)


def test_layer_caches_reset() -> None:
    """Reset should clear all caches."""
    lc = LayerCaches.create(3)
    for i in range(3):
        lc[i].update(torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4))
    assert all(c.seq_len == 5 for c in lc.caches)

    lc.reset()
    assert all(c.seq_len == 0 for c in lc.caches)
