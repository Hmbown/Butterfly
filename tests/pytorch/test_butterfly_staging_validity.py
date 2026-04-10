"""Regression checks for the staged-vs-frozen Butterfly validity experiment."""

from __future__ import annotations

import numpy as np
import pytest

from bna.topology.butterfly import butterfly_width
from bna.topology.validation import (
    NeighborSpec,
    build_butterfly_neighbor_role_rows,
    build_frozen_long_range_neighbor_role_rows,
    build_local_only_neighbor_role_rows,
    build_support_operator,
    butterfly_stage_count,
    observed_butterfly_degree_budget,
)


def _last_row_support_curve(
    row_sequence: list[list[list[NeighborSpec]]],
    *,
    num_blocks: int,
) -> list[float]:
    support_state = np.eye(int(num_blocks), dtype=bool)
    curve: list[float] = []
    for role_rows in row_sequence:
        layer_support = build_support_operator(role_rows, int(num_blocks))
        support_state = (layer_support.astype(np.uint8) @ support_state.astype(np.uint8)) > 0
        curve.append(float(np.count_nonzero(support_state[-1, :])) / float(num_blocks))
    return curve


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
@pytest.mark.parametrize("num_blocks", [32, 64])
def test_staged_schedule_beats_local_and_best_frozen_on_support_auc(
    partner_rule: str,
    num_blocks: int,
) -> None:
    width = butterfly_width(num_blocks)
    depth_max = 2 * width
    stage_count = butterfly_stage_count(num_blocks, partner_rule)
    degree_budget = observed_butterfly_degree_budget(
        num_blocks=num_blocks,
        block_size=64,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule=partner_rule,
    )

    staged_rows = [
        build_butterfly_neighbor_role_rows(
            num_blocks=num_blocks,
            layer_idx=layer_idx,
            block_size=64,
            local_window_blocks=1,
            sink_count=1,
            partner_count=1,
            partner_rule=partner_rule,
        )
        for layer_idx in range(depth_max)
    ]
    local_rows_one = build_local_only_neighbor_role_rows(
        num_blocks=num_blocks,
        degree_budget=degree_budget,
        sink_count=1,
    )
    local_rows = [local_rows_one for _ in range(depth_max)]

    staged_curve = _last_row_support_curve(staged_rows, num_blocks=num_blocks)
    local_curve = _last_row_support_curve(local_rows, num_blocks=num_blocks)

    frozen_curves: list[list[float]] = []
    for frozen_stage_idx in range(stage_count):
        frozen_rows_one = build_frozen_long_range_neighbor_role_rows(
            num_blocks=num_blocks,
            frozen_stage_idx=frozen_stage_idx,
            block_size=64,
            local_window_blocks=1,
            sink_count=1,
            partner_count=1,
            partner_rule=partner_rule,
        )
        frozen_rows = [frozen_rows_one for _ in range(depth_max)]
        frozen_curves.append(_last_row_support_curve(frozen_rows, num_blocks=num_blocks))

    best_frozen_curve = max(
        frozen_curves,
        key=lambda curve: (float(np.mean(curve)), float(curve[-1])),
    )

    assert float(np.mean(staged_curve)) > float(np.mean(local_curve))
    assert float(np.mean(staged_curve)) > float(np.mean(best_frozen_curve))
    assert staged_curve[width - 1] == pytest.approx(1.0)
    assert local_curve[width - 1] < 1.0
    assert best_frozen_curve[width - 1] < 1.0
