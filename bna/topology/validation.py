from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Sequence

import numpy as np

from bna.topology.butterfly import (
    NeighborSpec,
    RoleName,
    ROLE_ORDER,
    butterfly_layout_metadata,
    butterfly_partner_bits,
    butterfly_stage_count as butterfly_stage_count_public,
    butterfly_width,
    build_butterfly_role_map,
    ordered_roles,
)
from bna.torch.attention_wayfinder_permute import build_block_butterfly_layout


WeightingModel = Literal[
    "uniform",
    "local_biased",
    "partner_biased",
    "sink_biased",
    "dirichlet_random",
]

ROLE_WEIGHT_PRESETS: dict[str, dict[RoleName, float]] = {
    "uniform": {
        "self": 1.0,
        "local": 1.0,
        "partner": 1.0,
        "sink": 1.0,
    },
    "local_biased": {
        "self": 4.0,
        "local": 3.0,
        "partner": 1.0,
        "sink": 1.5,
    },
    "partner_biased": {
        "self": 1.5,
        "local": 1.25,
        "partner": 4.0,
        "sink": 1.0,
    },
    "sink_biased": {
        "self": 1.5,
        "local": 1.0,
        "partner": 1.0,
        "sink": 4.0,
    },
}


@dataclass(frozen=True)
class OperatorMetrics:
    support_coverage_last_row: float
    support_coverage_mean: float
    support_coverage_min: float
    normalized_effective_support_last_row: float
    normalized_effective_support_mean: float
    normalized_effective_support_min: float
    normalized_entropy_last_row: float
    normalized_entropy_mean: float
    normalized_entropy_min: float
    last_row_tv_to_uniform: float
    mean_tv_to_uniform: float
    last_row_max_mass: float
    mean_max_mass: float
    effective_rank: float
    effective_rank_ratio: float
    stable_rank: float
    stable_rank_ratio: float
    log10_condition_number: float
    normalized_singular_values: tuple[float, ...]


NeighborRowBuilder = Callable[[int], list[list[int]]]
NeighborRoleRowBuilder = Callable[[int], list[list[NeighborSpec]]]


_ceil_log2 = butterfly_width


def butterfly_stage_count(num_blocks: int, partner_rule: str) -> int:
    return int(
        butterfly_stage_count_public(
            int(num_blocks),
            str(partner_rule),
        )
    )


def build_butterfly_neighbor_role_rows(
    *,
    num_blocks: int,
    layer_idx: int,
    block_size: int,
    local_window_blocks: int,
    sink_count: int,
    partner_count: int,
    partner_rule: str,
) -> list[list[NeighborSpec]]:
    layout = build_block_butterfly_layout(
        seq_len=int(num_blocks) * int(block_size),
        block_size=int(block_size),
        num_key_value_heads=1,
        num_key_value_groups=1,
        layer_idx=int(layer_idx),
        local_window_blocks=int(local_window_blocks),
        sink_count=int(sink_count),
        partner_count=int(partner_count),
        partner_rule=str(partner_rule),
    )
    topology_meta = butterfly_layout_metadata(
        num_blocks=int(num_blocks),
        layer_idx=int(layer_idx),
        partner_rule=str(partner_rule),
        partner_count=int(partner_count),
        sink_count=int(sink_count),
        local_window_blocks=int(local_window_blocks),
    )
    stage_idx = int(layout.stage_idx or topology_meta.stage_idx)
    stage_count = int(layout.stage_count or topology_meta.stage_count)
    partner_bits = butterfly_partner_bits(
        stage_idx=int(stage_idx),
        stage_count=int(stage_count),
        width=int(topology_meta.width),
        partner_count=int(partner_count),
        partner_rule=str(partner_rule),
    )
    sink_blocks = (
        tuple(int(x) for x in layout.sink_blocks)
        if layout.sink_blocks
        else topology_meta.sink_blocks
    )

    rows: list[list[NeighborSpec]] = []
    for block_idx in range(int(num_blocks)):
        row = layout.block_neighbors[0, block_idx]
        valid_neighbors = [int(x) for x in row[row >= 0].tolist()]
        role_map = build_butterfly_role_map(
            block_idx=int(block_idx),
            num_blocks=int(num_blocks),
            local_window_blocks=int(local_window_blocks),
            sink_blocks=sink_blocks,
            partner_bits=partner_bits,
            partner_rule=str(partner_rule),
            width=int(topology_meta.width),
        )
        annotated_row: list[NeighborSpec] = []
        for neighbor in valid_neighbors:
            roles = role_map.get(int(neighbor))
            if roles is None:
                raise ValueError(
                    "Butterfly role annotation diverged from the production layout: "
                    f"block={block_idx} neighbor={neighbor}"
                )
            annotated_row.append(
                NeighborSpec(
                    neighbor=int(neighbor),
                    roles=ordered_roles(roles),
                )
            )
        rows.append(annotated_row)
    return rows


def build_frozen_long_range_neighbor_role_rows(
    *,
    num_blocks: int,
    frozen_stage_idx: int,
    block_size: int,
    local_window_blocks: int,
    sink_count: int,
    partner_count: int,
    partner_rule: str,
) -> list[list[NeighborSpec]]:
    stage_count = butterfly_stage_count(int(num_blocks), str(partner_rule))
    effective_layer_idx = int(frozen_stage_idx) % max(1, int(stage_count))
    return build_butterfly_neighbor_role_rows(
        num_blocks=int(num_blocks),
        layer_idx=int(effective_layer_idx),
        block_size=int(block_size),
        local_window_blocks=int(local_window_blocks),
        sink_count=int(sink_count),
        partner_count=int(partner_count),
        partner_rule=str(partner_rule),
    )


def build_butterfly_neighbor_rows(
    *,
    num_blocks: int,
    layer_idx: int,
    block_size: int,
    local_window_blocks: int,
    sink_count: int,
    partner_count: int,
    partner_rule: str,
) -> list[list[int]]:
    return [
        [int(spec.neighbor) for spec in row]
        for row in build_butterfly_neighbor_role_rows(
            num_blocks=int(num_blocks),
            layer_idx=int(layer_idx),
            block_size=int(block_size),
            local_window_blocks=int(local_window_blocks),
            sink_count=int(sink_count),
            partner_count=int(partner_count),
            partner_rule=str(partner_rule),
        )
    ]


def observed_butterfly_degree_budget(
    *,
    num_blocks: int,
    block_size: int,
    local_window_blocks: int,
    sink_count: int,
    partner_count: int,
    partner_rule: str,
) -> int:
    stage_count = butterfly_stage_count(int(num_blocks), str(partner_rule))
    max_degree = 1
    for layer_idx in range(int(stage_count)):
        rows = build_butterfly_neighbor_rows(
            num_blocks=int(num_blocks),
            layer_idx=int(layer_idx),
            block_size=int(block_size),
            local_window_blocks=int(local_window_blocks),
            sink_count=int(sink_count),
            partner_count=int(partner_count),
            partner_rule=str(partner_rule),
        )
        max_degree = max(max_degree, max(len(row) for row in rows))
    return int(max_degree)


def build_local_only_neighbor_role_rows(
    *,
    num_blocks: int,
    degree_budget: int,
    sink_count: int,
) -> list[list[NeighborSpec]]:
    local_window_blocks = max(0, int(degree_budget) - 1 - int(sink_count))
    sink_blocks = tuple(range(min(int(sink_count), int(num_blocks))))
    rows: list[list[NeighborSpec]] = []
    for block_idx in range(int(num_blocks)):
        entries: list[tuple[int, RoleName]] = [(int(block_idx), "self")]
        for offset in range(1, int(local_window_blocks) + 1):
            local_block = int(block_idx) - int(offset)
            if local_block < 0:
                break
            entries.append((int(local_block), "local"))
        for sink_block in sink_blocks:
            entries.append((int(sink_block), "sink"))
        rows.append(_ordered_unique_role_rows(entries, upper_bound=int(num_blocks)))
    return rows


def build_local_only_neighbor_rows(
    *,
    num_blocks: int,
    degree_budget: int,
    sink_count: int,
) -> list[list[int]]:
    return [
        [int(spec.neighbor) for spec in row]
        for row in build_local_only_neighbor_role_rows(
            num_blocks=int(num_blocks),
            degree_budget=int(degree_budget),
            sink_count=int(sink_count),
        )
    ]


def build_random_predecessor_neighbor_rows(
    *,
    num_blocks: int,
    layer_idx: int,
    degree_budget: int,
    local_window_blocks: int,
    sink_count: int,
    seed: int,
) -> list[list[int]]:
    sink_blocks = list(range(min(int(sink_count), int(num_blocks))))
    extra_random = max(
        0,
        int(degree_budget) - 1 - int(local_window_blocks) - int(sink_count),
    )
    rng = np.random.default_rng(
        int(seed)
        + (1009 * int(num_blocks))
        + (9173 * int(layer_idx))
        + (53 * int(degree_budget))
        + (97 * int(local_window_blocks))
        + (193 * int(sink_count))
    )

    rows: list[list[int]] = []
    for block_idx in range(int(num_blocks)):
        row: list[int] = [int(block_idx)]
        for offset in range(1, int(local_window_blocks) + 1):
            local_block = int(block_idx) - int(offset)
            if local_block < 0:
                break
            row.append(local_block)
        row.extend(sink_blocks)

        candidates = [candidate for candidate in range(int(block_idx)) if candidate not in row]
        if candidates and extra_random > 0:
            pick_count = min(int(extra_random), len(candidates))
            picks = rng.choice(candidates, size=pick_count, replace=False)
            row.extend(sorted(int(x) for x in np.asarray(picks).tolist()))

        rows.append(_ordered_unique_valid(row, upper_bound=int(num_blocks)))
    return rows


def build_row_stochastic_operator(neighbor_rows: list[list[int]], num_blocks: int) -> np.ndarray:
    operator = np.zeros((int(num_blocks), int(num_blocks)), dtype=np.float64)
    for row_idx, neighbors in enumerate(neighbor_rows):
        if not neighbors:
            raise ValueError(f"row {row_idx} has no neighbors")
        weight = 1.0 / float(len(neighbors))
        operator[row_idx, np.asarray(neighbors, dtype=np.int64)] = weight
    return operator


def build_role_weighted_operator(
    neighbor_role_rows: list[list[NeighborSpec]],
    num_blocks: int,
    *,
    weighting_model: WeightingModel = "uniform",
    layer_idx: int = 0,
    random_seed: int = 0,
) -> np.ndarray:
    operator = np.zeros((int(num_blocks), int(num_blocks)), dtype=np.float64)
    for row_idx, row in enumerate(neighbor_role_rows):
        if not row:
            raise ValueError(f"row {row_idx} has no neighbors")

        if str(weighting_model) == "dirichlet_random":
            weights = _dirichlet_row_weights(
                role_row=row,
                num_blocks=int(num_blocks),
                layer_idx=int(layer_idx),
                row_idx=int(row_idx),
                random_seed=int(random_seed),
            )
        else:
            score_vector = np.asarray(
                [_role_weight_score(spec.roles, str(weighting_model)) for spec in row],
                dtype=np.float64,
            )
            weights = score_vector / max(float(score_vector.sum()), np.finfo(np.float64).tiny)

        operator[row_idx, np.asarray([spec.neighbor for spec in row], dtype=np.int64)] = weights
    return operator


def build_support_operator(
    neighbor_role_rows: list[list[NeighborSpec]],
    num_blocks: int,
) -> np.ndarray:
    support = np.zeros((int(num_blocks), int(num_blocks)), dtype=bool)
    for row_idx, row in enumerate(neighbor_role_rows):
        if not row:
            raise ValueError(f"row {row_idx} has no neighbors")
        support[row_idx, np.asarray([spec.neighbor for spec in row], dtype=np.int64)] = True
    return support


def compose_causal_operator(
    *,
    num_blocks: int,
    num_layers: int,
    row_builder: NeighborRowBuilder,
) -> np.ndarray:
    composed = np.eye(int(num_blocks), dtype=np.float64)
    for layer_idx in range(int(num_layers)):
        layer_operator = build_row_stochastic_operator(row_builder(int(layer_idx)), int(num_blocks))
        composed = layer_operator @ composed
    return composed


def compose_role_weighted_operator(
    *,
    num_blocks: int,
    num_layers: int,
    row_builder: NeighborRoleRowBuilder,
    weighting_model: WeightingModel = "uniform",
    random_seed: int = 0,
) -> np.ndarray:
    composed = np.eye(int(num_blocks), dtype=np.float64)
    for layer_idx in range(int(num_layers)):
        layer_operator = build_role_weighted_operator(
            row_builder(int(layer_idx)),
            int(num_blocks),
            weighting_model=str(weighting_model),
            layer_idx=int(layer_idx),
            random_seed=int(random_seed),
        )
        composed = layer_operator @ composed
    return composed


def compose_support_operator(
    *,
    num_blocks: int,
    num_layers: int,
    row_builder: NeighborRoleRowBuilder,
) -> np.ndarray:
    composed = np.eye(int(num_blocks), dtype=bool)
    for layer_idx in range(int(num_layers)):
        layer_support = build_support_operator(row_builder(int(layer_idx)), int(num_blocks))
        composed = _boolean_compose(layer_support, composed)
    return composed


def measure_operator(
    composed: np.ndarray,
    *,
    support_matrix: np.ndarray | None = None,
    support_tol: float = 1e-12,
    singular_values_top_k: int = 8,
) -> OperatorMetrics:
    matrix = np.asarray(composed, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("composed must be a square matrix")

    num_blocks = int(matrix.shape[0])
    if support_matrix is None:
        support_view = matrix > float(support_tol)
    else:
        support_view = np.asarray(support_matrix, dtype=bool)
        if support_view.shape != matrix.shape:
            raise ValueError("support_matrix must match composed shape")

    coverage: list[float] = []
    effective_support: list[float] = []
    normalized_entropy: list[float] = []
    tv_to_uniform: list[float] = []
    max_mass: list[float] = []

    for row_idx in range(num_blocks):
        prefix = matrix[row_idx, : row_idx + 1].astype(np.float64, copy=True)
        prefix_mass = float(prefix.sum())
        if prefix_mass <= 0.0:
            raise ValueError(f"row {row_idx} has zero causal-prefix mass")
        prefix /= prefix_mass

        support = (
            float(np.count_nonzero(support_view[row_idx, : row_idx + 1]))
            / float(row_idx + 1)
        )
        coverage.append(support)

        inverse_participation = float(np.square(prefix).sum())
        eff_support = 1.0 / max(inverse_participation, np.finfo(np.float64).tiny)
        effective_support.append(float(eff_support / float(row_idx + 1)))

        if row_idx == 0:
            normalized_entropy.append(1.0)
        else:
            entropy = float(-(prefix * np.log(prefix + 1e-300)).sum())
            normalized_entropy.append(float(entropy / math.log(float(row_idx + 1))))

        uniform = np.full((row_idx + 1,), 1.0 / float(row_idx + 1), dtype=np.float64)
        tv_to_uniform.append(float(0.5 * np.abs(prefix - uniform).sum()))
        max_mass.append(float(prefix.max()))

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    singular_sum = float(singular_values.sum())
    normalized_spectrum = singular_values / max(singular_values[0], np.finfo(np.float64).tiny)
    spectrum_prob = singular_values / max(singular_sum, np.finfo(np.float64).tiny)
    effective_rank = float(np.exp(-np.sum(spectrum_prob * np.log(spectrum_prob + 1e-300))))
    stable_rank = float(
        np.square(singular_values).sum()
        / max(float(singular_values[0] ** 2), np.finfo(np.float64).tiny)
    )
    sigma_min = max(float(singular_values[-1]), np.finfo(np.float64).tiny)
    condition_number = float(singular_values[0]) / sigma_min

    return OperatorMetrics(
        support_coverage_last_row=float(coverage[-1]),
        support_coverage_mean=float(np.mean(coverage)),
        support_coverage_min=float(np.min(coverage)),
        normalized_effective_support_last_row=float(effective_support[-1]),
        normalized_effective_support_mean=float(np.mean(effective_support)),
        normalized_effective_support_min=float(np.min(effective_support)),
        normalized_entropy_last_row=float(normalized_entropy[-1]),
        normalized_entropy_mean=float(np.mean(normalized_entropy)),
        normalized_entropy_min=float(np.min(normalized_entropy)),
        last_row_tv_to_uniform=float(tv_to_uniform[-1]),
        mean_tv_to_uniform=float(np.mean(tv_to_uniform)),
        last_row_max_mass=float(max_mass[-1]),
        mean_max_mass=float(np.mean(max_mass)),
        effective_rank=float(effective_rank),
        effective_rank_ratio=float(effective_rank / float(num_blocks)),
        stable_rank=float(stable_rank),
        stable_rank_ratio=float(stable_rank / float(num_blocks)),
        log10_condition_number=float(math.log10(max(condition_number, 1.0))),
        normalized_singular_values=tuple(
            float(x) for x in normalized_spectrum[: max(1, int(singular_values_top_k))].tolist()
        ),
    )

def _ordered_unique_role_rows(
    entries: Sequence[tuple[int, RoleName]],
    *,
    upper_bound: int,
) -> list[NeighborSpec]:
    seen: dict[int, set[RoleName]] = {}
    order: list[int] = []
    for neighbor, role in entries:
        ineighbor = int(neighbor)
        if ineighbor < 0 or ineighbor >= int(upper_bound):
            continue
        if ineighbor not in seen:
            seen[ineighbor] = set()
            order.append(ineighbor)
        seen[ineighbor].add(role)
    return [
        NeighborSpec(neighbor=int(neighbor), roles=ordered_roles(seen[neighbor]))
        for neighbor in order
    ]


def _role_weight_score(roles: Sequence[RoleName], weighting_model: str) -> float:
    if str(weighting_model) not in ROLE_WEIGHT_PRESETS:
        raise ValueError(f"Unsupported weighting model: {weighting_model!r}")
    preset = ROLE_WEIGHT_PRESETS[str(weighting_model)]
    return max(float(preset[role]) for role in roles)


def _dirichlet_row_weights(
    *,
    role_row: list[NeighborSpec],
    num_blocks: int,
    layer_idx: int,
    row_idx: int,
    random_seed: int,
) -> np.ndarray:
    row_signature = 0
    for position, spec in enumerate(role_row):
        role_signature = sum((ROLE_ORDER.index(role) + 1) for role in spec.roles)
        row_signature += (position + 1) * (int(spec.neighbor) + 1) * (role_signature + 11)
    seed = (
        int(random_seed)
        + (1009 * int(num_blocks))
        + (9173 * int(layer_idx))
        + (97 * int(row_idx))
        + (13 * int(len(role_row)))
        + (31 * int(row_signature))
    )
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones((len(role_row),), dtype=np.float64))


def _boolean_compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left.astype(np.uint8) @ right.astype(np.uint8)) > 0


def _ordered_unique_valid(values: list[int], *, upper_bound: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue < 0 or ivalue >= int(upper_bound) or ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out
