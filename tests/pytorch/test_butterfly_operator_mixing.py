"""Regression checks for the Butterfly causal mixing operator experiment."""

from __future__ import annotations

import pytest

from bna.topology.butterfly import butterfly_width
from bna.topology.validation import (
    build_butterfly_neighbor_rows,
    build_local_only_neighbor_rows,
    compose_causal_operator,
    measure_operator,
    observed_butterfly_degree_budget,
)


@pytest.mark.parametrize("partner_rule", ["xor", "bit_reversal", "benes"])
@pytest.mark.parametrize("num_blocks", [32, 64])
def test_log_depth_operator_spreads_more_evenly_than_local_only(
    partner_rule: str,
    num_blocks: int,
) -> None:
    """Secondary surrogate diagnostic at log depth: support + TV vs local-only.

    This is intentionally secondary to the support/reachability proof surface.
    It checks one deterministic surrogate weighting model and should not be read
    as a general learned-mixing guarantee.
    """
    width = butterfly_width(num_blocks)
    degree_budget = observed_butterfly_degree_budget(
        num_blocks=num_blocks,
        block_size=64,
        local_window_blocks=1,
        sink_count=1,
        partner_count=1,
        partner_rule=partner_rule,
    )

    butterfly_operator = compose_causal_operator(
        num_blocks=num_blocks,
        num_layers=width,
        row_builder=lambda layer_idx: build_butterfly_neighbor_rows(
            num_blocks=num_blocks,
            layer_idx=layer_idx,
            block_size=64,
            local_window_blocks=1,
            sink_count=1,
            partner_count=1,
            partner_rule=partner_rule,
        ),
    )
    local_only_operator = compose_causal_operator(
        num_blocks=num_blocks,
        num_layers=width,
        row_builder=lambda _layer_idx: build_local_only_neighbor_rows(
            num_blocks=num_blocks,
            degree_budget=degree_budget,
            sink_count=1,
        ),
    )

    butterfly_metrics = measure_operator(butterfly_operator)
    local_only_metrics = measure_operator(local_only_operator)

    assert butterfly_metrics.support_coverage_last_row == pytest.approx(1.0)
    assert (
        butterfly_metrics.support_coverage_last_row
        > local_only_metrics.support_coverage_last_row
    )
    assert (
        butterfly_metrics.last_row_tv_to_uniform
        < local_only_metrics.last_row_tv_to_uniform
    )
