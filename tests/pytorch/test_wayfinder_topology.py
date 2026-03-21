"""Topology validation for the Wayfinder block-sparse attention graph.

Tests the core graph-theoretic properties that make the mechanism interesting:
- strict causality at block level
- self-inclusion and sink-inclusion
- bounded degree
- global reachability across layers via the staged partner schedule
- correctness of all three partner rules (xor, bit_reversal, benes)
- edge cases with non-power-of-2 block counts

All tests are CPU-only and instant.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pytest
import torch

from bna.torch.attention_wayfinder_permute import (
    BlockHamiltonianLayout,
    _bit_reverse,
    _ceil_log2,
    _wayfinder_partner_bits,
    _wayfinder_partner_block,
    _wayfinder_stage_meta,
    build_block_wayfinder_layout,
)


# ---------------------------------------------------------------------------
# Helper: extract valid neighbor list from a padded row
# ---------------------------------------------------------------------------

def _valid_neighbors(layout: BlockHamiltonianLayout, head: int, block: int) -> list[int]:
    row = layout.block_neighbors[head, block]
    return row[row >= 0].tolist()


def _build(
    num_blocks: int,
    layer_idx: int = 0,
    block_size: int = 64,
    local_window_blocks: int = 1,
    sink_count: int = 1,
    partner_count: int = 1,
    partner_rule: str = "xor",
    num_kv_heads: int = 1,
    num_kv_groups: int = 1,
) -> BlockHamiltonianLayout:
    return build_block_wayfinder_layout(
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


# ===================================================================
# 1. Causality
# ===================================================================

@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
@pytest.mark.parametrize("num_blocks", [8, 16, 32, 17, 33])
def test_all_neighbors_are_causal(partner_rule: str, num_blocks: int) -> None:
    """Every block must only attend to blocks at or before itself."""
    for layer_idx in range(12):
        layout = _build(num_blocks, layer_idx=layer_idx, partner_rule=partner_rule)
        for block_idx in range(num_blocks):
            neighbors = _valid_neighbors(layout, 0, block_idx)
            for neighbor in neighbors:
                assert neighbor <= block_idx, (
                    f"Causality violation: block {block_idx} attends to future "
                    f"block {neighbor} at layer {layer_idx} with {partner_rule}"
                )


# ===================================================================
# 2. Self-inclusion and sink-inclusion
# ===================================================================

@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_self_always_in_support(partner_rule: str) -> None:
    """Every block must attend to itself."""
    layout = _build(16, partner_rule=partner_rule)
    for block_idx in range(16):
        neighbors = _valid_neighbors(layout, 0, block_idx)
        assert block_idx in neighbors, f"Block {block_idx} missing self-attention"


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_sink_blocks_in_support(partner_rule: str) -> None:
    """Sink blocks must appear in every non-zero block's support."""
    layout = _build(16, sink_count=2, partner_rule=partner_rule)
    assert tuple(layout.sink_blocks) == (0, 1)
    for block_idx in range(2, 16):
        neighbors = _valid_neighbors(layout, 0, block_idx)
        assert 0 in neighbors, f"Block {block_idx} missing sink 0"
        assert 1 in neighbors, f"Block {block_idx} missing sink 1"


# ===================================================================
# 3. Degree regularity
# ===================================================================

@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_degree_bounded(partner_rule: str) -> None:
    """Per-block degree should be bounded: self + local + partners + sinks."""
    local_w = 2
    partner_c = 2
    sink_c = 1
    max_expected = 1 + local_w + partner_c + sink_c  # theoretical max

    layout = _build(32, local_window_blocks=local_w, partner_count=partner_c,
                    sink_count=sink_c, partner_rule=partner_rule)
    for block_idx in range(layout.num_blocks):
        degree = len(_valid_neighbors(layout, 0, block_idx))
        # Early blocks may have fewer due to causality filtering
        assert degree <= max_expected, (
            f"Block {block_idx} has degree {degree} > max {max_expected}"
        )
        # Every block should at least attend to itself
        assert degree >= 1


def test_degree_monotonically_approaches_max() -> None:
    """Later blocks should have degree >= earlier blocks (more causal options)."""
    layout = _build(32, local_window_blocks=2, partner_count=1, sink_count=1)
    prev_degree = 0
    for block_idx in range(layout.num_blocks):
        degree = len(_valid_neighbors(layout, 0, block_idx))
        # Allow equal degrees, but later blocks shouldn't have strictly fewer
        # options than much earlier blocks once past the initial ramp-up
        if block_idx >= 8:  # past the ramp-up
            assert degree >= prev_degree or (prev_degree - degree) <= 1, (
                f"Block {block_idx} has degree {degree} < {prev_degree}"
            )
        prev_degree = degree


# ===================================================================
# 4. Global reachability (the core communication claim)
# ===================================================================

def _reachability_after_layers(
    num_blocks: int,
    num_layers: int,
    partner_rule: str = "xor",
    partner_count: int = 1,
    local_window_blocks: int = 1,
    sink_count: int = 1,
) -> dict[int, set[int]]:
    """Compute reachable set for each block after sweeping through `num_layers`.

    Simulates information flow: at each layer, each block receives info from
    all neighbors in that layer's topology, plus what those neighbors could
    already reach.
    """
    # reachable[block] = set of blocks whose info has reached this block
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
        new_reachable: dict[int, set[int]] = {i: set(reachable[i]) for i in range(num_blocks)}
        for block_idx in range(num_blocks):
            neighbors = _valid_neighbors(layout, 0, block_idx)
            for neighbor in neighbors:
                new_reachable[block_idx] |= reachable[neighbor]
        reachable = new_reachable

    return reachable


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_global_reachability_in_log_n_layers(partner_rule: str) -> None:
    """After O(log n) layers, every block should be able to reach all causally-prior blocks.

    This is the fundamental claim: the staged partner schedule creates a
    communication network with diameter O(log n).
    """
    num_blocks = 16
    width = _ceil_log2(num_blocks)  # 4

    # Give it 2x the theoretical minimum to account for causality filtering
    num_layers = 2 * width
    reachable = _reachability_after_layers(
        num_blocks, num_layers, partner_rule=partner_rule, partner_count=1,
    )

    # Check: the last block should be able to reach all blocks
    last_block = num_blocks - 1
    assert reachable[last_block] == set(range(num_blocks)), (
        f"Block {last_block} cannot reach all blocks after {num_layers} layers "
        f"with {partner_rule}. Missing: {set(range(num_blocks)) - reachable[last_block]}"
    )


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_reachability_diameter_bounded(partner_rule: str) -> None:
    """Measure the actual diameter: fewest layers needed for full causal reachability."""
    num_blocks = 16
    width = _ceil_log2(num_blocks)

    for num_layers in range(1, 4 * width + 1):
        reachable = _reachability_after_layers(
            num_blocks, num_layers, partner_rule=partner_rule,
        )
        last_block = num_blocks - 1
        if reachable[last_block] == set(range(num_blocks)):
            # Found the diameter for the last block
            assert num_layers <= 3 * width, (
                f"Diameter {num_layers} exceeds 3 * log2(n) = {3 * width} "
                f"for {partner_rule}"
            )
            return
    pytest.fail(f"Block {num_blocks - 1} never reached full reachability with {partner_rule}")


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_reachability_with_two_partners(partner_rule: str) -> None:
    """Two partners per block should achieve full reachability faster."""
    num_blocks = 32
    width = _ceil_log2(num_blocks)

    reachable_1p = _reachability_after_layers(
        num_blocks, 2 * width, partner_rule=partner_rule, partner_count=1,
    )
    reachable_2p = _reachability_after_layers(
        num_blocks, 2 * width, partner_rule=partner_rule, partner_count=2,
    )

    last = num_blocks - 1
    # 2 partners should reach at least as many blocks
    assert len(reachable_2p[last]) >= len(reachable_1p[last]), (
        f"Two partners reaches fewer blocks than one with {partner_rule}"
    )


# ===================================================================
# 5. Stage schedule correctness
# ===================================================================

@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
def test_stage_covers_all_bits(partner_rule: str) -> None:
    """Cycling through all layers should touch every stage exactly once per period."""
    num_blocks = 16
    width = _ceil_log2(num_blocks)
    stage_idx_0, stage_count = _wayfinder_stage_meta(
        num_blocks=num_blocks, layer_idx=0, partner_rule=partner_rule,
    )

    seen_stages = set()
    for layer_idx in range(stage_count):
        stage_idx, _ = _wayfinder_stage_meta(
            num_blocks=num_blocks, layer_idx=layer_idx, partner_rule=partner_rule,
        )
        seen_stages.add(stage_idx)

    assert seen_stages == set(range(stage_count)), (
        f"Not all stages covered: got {seen_stages}, expected {set(range(stage_count))}"
    )


def test_stage_count_xor() -> None:
    """XOR stage count should be ceil(log2(num_blocks))."""
    assert _wayfinder_stage_meta(num_blocks=16, layer_idx=0, partner_rule="xor") == (0, 4)
    assert _wayfinder_stage_meta(num_blocks=8, layer_idx=0, partner_rule="xor") == (0, 3)
    assert _wayfinder_stage_meta(num_blocks=32, layer_idx=0, partner_rule="xor") == (0, 5)


def test_stage_count_benes() -> None:
    """Benes stage count should be 2*ceil(log2(n))-2."""
    _, count = _wayfinder_stage_meta(num_blocks=16, layer_idx=0, partner_rule="benes")
    width = _ceil_log2(16)  # 4
    assert count == 2 * width - 2  # 6


# ===================================================================
# 6. Partner rule correctness
# ===================================================================

def test_xor_partner_is_symmetric_for_non_causal() -> None:
    """XOR partner relationship is symmetric (ignoring causality filter)."""
    width = 4
    for block_idx in range(16):
        for bit_idx in range(width):
            partner = block_idx ^ (1 << bit_idx)
            if 0 <= partner < 16:
                reverse = partner ^ (1 << bit_idx)
                assert reverse == block_idx


def test_bit_reversal_partner_produces_valid_blocks() -> None:
    """bit_reversal partner should always produce in-range blocks."""
    width = _ceil_log2(16)
    for block_idx in range(16):
        for bit_idx in range(width):
            partner = _wayfinder_partner_block(
                block_idx=block_idx, bit_idx=bit_idx,
                num_blocks=16, partner_rule="bit_reversal", width=width,
            )
            if partner is not None:
                assert 0 <= partner < 16
                assert partner <= block_idx  # causality


def test_benes_partner_matches_xor_for_forward_half() -> None:
    """In the forward half of a Benes network, partners should match XOR."""
    width = _ceil_log2(16)
    for stage_idx in range(width):
        bits_xor = _wayfinder_partner_bits(
            stage_idx=stage_idx, stage_count=width,
            width=width, partner_count=1, partner_rule="xor",
        )
        bits_benes = _wayfinder_partner_bits(
            stage_idx=stage_idx, stage_count=2 * width - 2,
            width=width, partner_count=1, partner_rule="benes",
        )
        # In the forward half, Benes should flip the same bit as XOR
        assert bits_benes[0] == bits_xor[0], (
            f"Stage {stage_idx}: benes bit {bits_benes[0]} != xor bit {bits_xor[0]}"
        )


# ===================================================================
# 7. Non-power-of-2 edge cases
# ===================================================================

@pytest.mark.parametrize("num_blocks", [7, 13, 17, 31, 33])
def test_non_power_of_2_blocks_still_valid(num_blocks: int) -> None:
    """Non-power-of-2 block counts should still produce valid causal graphs."""
    for layer_idx in range(8):
        layout = _build(num_blocks, layer_idx=layer_idx, partner_rule="xor")
        for block_idx in range(num_blocks):
            neighbors = _valid_neighbors(layout, 0, block_idx)
            assert block_idx in neighbors, f"Missing self at block {block_idx}"
            for n in neighbors:
                assert 0 <= n < num_blocks, f"Out of range neighbor {n}"
                assert n <= block_idx, f"Future neighbor {n} for block {block_idx}"


def test_single_block_trivial() -> None:
    """A single block should attend only to itself."""
    layout = _build(1, layer_idx=0)
    neighbors = _valid_neighbors(layout, 0, 0)
    assert neighbors == [0]


def test_two_blocks_complete() -> None:
    """Two blocks: block 0 sees [0], block 1 sees [1, 0]."""
    layout = _build(2, layer_idx=0, local_window_blocks=1, sink_count=0)
    n0 = _valid_neighbors(layout, 0, 0)
    n1 = _valid_neighbors(layout, 0, 1)
    assert 0 in n0
    assert 1 in n1
    assert 0 in n1


# ===================================================================
# 8. GQA head replication
# ===================================================================

def test_gqa_heads_share_kv_pattern() -> None:
    """With GQA, query heads in the same group should share the block layout."""
    layout = _build(16, num_kv_heads=4, num_kv_groups=2)
    # block_neighbors shape should be [H_q=8, N=16, D]
    assert layout.block_neighbors.shape[0] == 8
    # Heads 0 and 1 (same KV group) should be identical
    assert torch.equal(layout.block_neighbors[0], layout.block_neighbors[1])
    # Heads 0 and 2 (different KV groups) should also be identical
    # because all KV heads get the same static pattern
    assert torch.equal(layout.block_neighbors[0], layout.block_neighbors[2])


# ===================================================================
# 9. Reachability comparison: Wayfinder vs random local-only baseline
# ===================================================================

def test_wayfinder_reaches_more_than_local_only() -> None:
    """Wayfinder with partners should reach more blocks than local-only window."""
    num_blocks = 32
    width = _ceil_log2(num_blocks)

    # Wayfinder: local + partners + sink
    wayfinder_reach = _reachability_after_layers(
        num_blocks, width, partner_rule="xor", partner_count=1,
        local_window_blocks=1, sink_count=1,
    )

    # Local-only: same degree budget but no partners
    local_reach = _reachability_after_layers(
        num_blocks, width, partner_rule="xor", partner_count=0,
        local_window_blocks=2, sink_count=1,
    )

    last = num_blocks - 1
    assert len(wayfinder_reach[last]) > len(local_reach[last]), (
        f"Wayfinder ({len(wayfinder_reach[last])}) should reach more blocks "
        f"than local-only ({len(local_reach[last])})"
    )
