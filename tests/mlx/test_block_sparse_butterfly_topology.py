from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import build_block_butterfly_layout  # noqa: E402
from bna.topology.butterfly import butterfly_width  # noqa: E402

PARTNER_RULES = ["xor", "bit_reversal", "benes", "causal_shift"]


def _valid_neighbors(layout, head: int, block_idx: int) -> list[int]:
    row = np.asarray(layout.block_neighbors[head, block_idx], dtype=np.int32)
    return row[row >= 0].tolist()


def _build(
    num_blocks: int,
    *,
    layer_idx: int = 0,
    block_size: int = 64,
    local_window_blocks: int = 1,
    sink_count: int = 1,
    partner_count: int = 1,
    partner_rule: str = "xor",
    num_kv_heads: int = 1,
    num_kv_groups: int = 1,
):
    return build_block_butterfly_layout(
        seq_len=num_blocks * block_size,
        block_size=block_size,
        num_key_value_heads=num_kv_heads,
        num_key_value_groups=num_kv_groups,
        layer_idx=layer_idx,
        local_window_blocks=local_window_blocks,
        sink_count=sink_count,
        partner_count=partner_count,
        partner_rule=partner_rule,
    )


def _reachability_after_layers(
    num_blocks: int,
    num_layers: int,
    *,
    partner_rule: str = "xor",
    partner_count: int = 1,
    local_window_blocks: int = 1,
    sink_count: int = 1,
) -> dict[int, set[int]]:
    reachable: dict[int, set[int]] = {i: {i} for i in range(num_blocks)}
    for layer_idx in range(num_layers):
        layout = _build(
            num_blocks,
            layer_idx=layer_idx,
            partner_rule=partner_rule,
            partner_count=partner_count,
            local_window_blocks=local_window_blocks,
            sink_count=sink_count,
        )
        new_reachable = {i: set(reachable[i]) for i in range(num_blocks)}
        for block_idx in range(num_blocks):
            for neighbor in _valid_neighbors(layout, 0, block_idx):
                new_reachable[block_idx] |= reachable[neighbor]
        reachable = new_reachable
    return reachable


@pytest.mark.parametrize("partner_rule", PARTNER_RULES)
@pytest.mark.parametrize("num_blocks", [8, 16, 17, 33])
def test_block_sparse_neighbors_are_causal(partner_rule: str, num_blocks: int) -> None:
    for layer_idx in range(12):
        layout = _build(num_blocks, layer_idx=layer_idx, partner_rule=partner_rule)
        for block_idx in range(num_blocks):
            neighbors = _valid_neighbors(layout, 0, block_idx)
            assert block_idx in neighbors
            for neighbor in neighbors:
                assert 0 <= neighbor < num_blocks
                assert neighbor <= block_idx


@pytest.mark.parametrize("partner_rule", PARTNER_RULES)
def test_block_sparse_degree_is_bounded(partner_rule: str) -> None:
    local_w = 2
    partner_c = 2
    sink_c = 1
    layout = _build(
        32,
        local_window_blocks=local_w,
        partner_count=partner_c,
        sink_count=sink_c,
        partner_rule=partner_rule,
    )
    max_expected = 1 + local_w + partner_c + sink_c
    for block_idx in range(layout.num_blocks):
        degree = len(_valid_neighbors(layout, 0, block_idx))
        assert 1 <= degree <= max_expected


@pytest.mark.parametrize("partner_rule", PARTNER_RULES)
def test_block_sparse_reaches_full_prefix_in_two_log_layers(partner_rule: str) -> None:
    num_blocks = 32
    num_layers = 2 * butterfly_width(num_blocks)
    reachable = _reachability_after_layers(
        num_blocks,
        num_layers,
        partner_rule=partner_rule,
        partner_count=1,
        local_window_blocks=1,
        sink_count=1,
    )
    for block_idx in range(num_blocks):
        assert reachable[block_idx] >= set(range(block_idx + 1))


@pytest.mark.parametrize("num_blocks", [16, 32, 33, 64, 65, 100, 129])
def test_block_sparse_causal_shift_reaches_full_prefix_in_log_layers(num_blocks: int) -> None:
    num_layers = butterfly_width(num_blocks)
    reachable = _reachability_after_layers(
        num_blocks,
        num_layers,
        partner_rule="causal_shift",
        partner_count=1,
        local_window_blocks=1,
        sink_count=1,
    )
    for block_idx in range(num_blocks):
        assert reachable[block_idx] >= set(range(block_idx + 1))


def test_block_sparse_gqa_head_groups_share_layout() -> None:
    layout = _build(16, num_kv_heads=4, num_kv_groups=2)
    assert tuple(layout.block_neighbors.shape[:2]) == (8, 16)
    assert np.array_equal(np.asarray(layout.block_neighbors[0]), np.asarray(layout.block_neighbors[1]))
    assert np.array_equal(np.asarray(layout.block_mask[0]), np.asarray(layout.block_mask[3]))
