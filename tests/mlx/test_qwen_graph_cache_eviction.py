"""Regression test for the chunked-prefill graph-cache leak.

Prior to commit b89992f, `_QWEN_GRAPH_CACHE_BY_KEY` accumulated one entry
per `(T, layer_idx, ...)` seen during chunked prefill and was never evicted.
At 32k context with chunk_size=384, this stored ~516 distinct
`BlockSparseButterflyLayout` instances each holding `block_mask` /
`block_causal_mask` tensors that scale with T -- ~5 GB of layout tensors
nothing freed. Stock mode does not go through this dispatch.

The fix: `_qwen_graph_cache_drop_other_keys(current_T)` evicts every entry
whose T does not equal `current_T`, called at the start of each prefill
chunk. Within-chunk sharing across the 6 swapped layers is preserved.

This test directly exercises the eviction function so the leak cannot
return silently.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from bna.integrations.qwen_mlx import (  # noqa: E402
    _QWEN_GRAPH_CACHE_BY_KEY,
    _qwen_graph_cache_drop_other_keys,
)


def _make_block_sparse_key(T: int, layer_idx: int) -> tuple:
    # Mirrors the tuple shape produced by `_QwenGraphRuntime.cache_key` for
    # path="block_sparse"; only the index of T (=2) and the path field (=3)
    # matter for eviction.
    return (
        8,                 # n_heads
        2,                 # n_kv_heads
        int(T),            # T (index 2 -- this is what the evictor reads)
        "block_sparse",    # path (index 3)
        128,               # block_size
        1,                 # local_window_blocks
        1,                 # partner_count
        1,                 # sink_blocks
        "causal_shift",    # partner_rule
        "mean",            # block_compression
        64,                # local_window_tokens
        int(layer_idx),    # layer_idx
        0,                 # stage_idx
        1,                 # stage_count
    )


def _make_other_path_key(T: int) -> tuple:
    # For non-block_sparse paths T is at index 1 and path at index 10.
    return (
        8,                 # n_heads
        int(T),            # T (index 1)
        "default",         # strategy
        1,                 # num_cycles
        True,              # edge_disjoint
        True,              # enforce_hamiltonian
        0,                 # regular_num_clusters
        64,                # window
        None,              # landmark_stride
        0,                 # seed
        "permute",         # path (index 10)
        None,              # compiled_graph_dir
    )


def test_evicts_prior_T_keeps_current_T() -> None:
    _QWEN_GRAPH_CACHE_BY_KEY.clear()
    sentinel = object()  # using a sentinel; eviction does not inspect values
    # Populate with 6 layers' worth of entries at T=384 (prior chunk).
    for layer_idx in (3, 7, 11, 15, 19, 23):
        _QWEN_GRAPH_CACHE_BY_KEY[_make_block_sparse_key(384, layer_idx)] = sentinel
    # And 6 layers' worth at T=768 (current chunk).
    for layer_idx in (3, 7, 11, 15, 19, 23):
        _QWEN_GRAPH_CACHE_BY_KEY[_make_block_sparse_key(768, layer_idx)] = sentinel
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == 12

    dropped = _qwen_graph_cache_drop_other_keys(768)
    assert dropped == 6
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == 6
    # Remaining keys must all be at T=768.
    for key in _QWEN_GRAPH_CACHE_BY_KEY:
        assert int(key[2]) == 768

    _QWEN_GRAPH_CACHE_BY_KEY.clear()


def test_evicts_other_T_for_mixed_paths() -> None:
    _QWEN_GRAPH_CACHE_BY_KEY.clear()
    sentinel = object()
    _QWEN_GRAPH_CACHE_BY_KEY[_make_block_sparse_key(1024, 3)] = sentinel
    _QWEN_GRAPH_CACHE_BY_KEY[_make_block_sparse_key(2048, 3)] = sentinel
    _QWEN_GRAPH_CACHE_BY_KEY[_make_other_path_key(1024)] = sentinel
    _QWEN_GRAPH_CACHE_BY_KEY[_make_other_path_key(2048)] = sentinel
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == 4

    dropped = _qwen_graph_cache_drop_other_keys(2048)
    assert dropped == 2
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == 2

    _QWEN_GRAPH_CACHE_BY_KEY.clear()


def test_no_op_when_only_current_T_present() -> None:
    _QWEN_GRAPH_CACHE_BY_KEY.clear()
    sentinel = object()
    for layer_idx in (3, 7, 11, 15, 19, 23):
        _QWEN_GRAPH_CACHE_BY_KEY[_make_block_sparse_key(2048, layer_idx)] = sentinel
    before = len(_QWEN_GRAPH_CACHE_BY_KEY)

    dropped = _qwen_graph_cache_drop_other_keys(2048)
    assert dropped == 0
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == before

    _QWEN_GRAPH_CACHE_BY_KEY.clear()


def test_handles_empty_dict() -> None:
    _QWEN_GRAPH_CACHE_BY_KEY.clear()
    assert _qwen_graph_cache_drop_other_keys(123) == 0
    assert len(_QWEN_GRAPH_CACHE_BY_KEY) == 0
