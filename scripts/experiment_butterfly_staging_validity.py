#!/usr/bin/env python3
"""Validate whether staging itself matters in Butterfly's causal topology.

This experiment keeps the claim intentionally narrow and paper-safe.
It studies the exact block topology induced by `build_block_butterfly_layout(...)`
and asks two separate questions:

1. Support expansion: does the staged schedule reach causal-prefix support more
   effectively than a degree-matched local baseline and a frozen long-range
   control that reuses one fixed partner pattern every layer?
2. Mixing / concentration: if we keep the same support graph but change the
   admissible row weights within a small deterministic surrogate family, do
   stronger spread claims remain robust?

Support is measured from exact boolean reachability, independent of the weight
model. Spread / concentration is measured from a positive row-stochastic
surrogate operator under several deterministic weighting families.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.topology.butterfly import butterfly_width  # noqa: E402
from bna.topology.validation import (  # noqa: E402
    NeighborSpec,
    OperatorMetrics,
    WeightingModel,
    build_butterfly_neighbor_role_rows,
    build_frozen_long_range_neighbor_role_rows,
    build_local_only_neighbor_role_rows,
    build_role_weighted_operator,
    build_support_operator,
    butterfly_stage_count,
    measure_operator,
    observed_butterfly_degree_budget,
)  # noqa: E402


WEIGHT_MODELS: tuple[WeightingModel, ...] = (
    "uniform",
    "local_biased",
    "partner_biased",
    "sink_biased",
    "dirichlet_random",
)

WEIGHT_MODEL_DESCRIPTIONS: dict[str, str] = {
    "uniform": "Equal mass on every admissible predecessor.",
    "local_biased": "Prefer self/local edges while keeping every admissible edge positive.",
    "partner_biased": "Prefer staged partner edges while keeping self/local/sink edges positive.",
    "sink_biased": "Prefer sink edges while keeping every admissible edge positive.",
    "dirichlet_random": (
        "Deterministic random Dirichlet row weights with positive mass on every "
        "admissible edge."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    num_blocks: int
    partner_rule: str
    block_size: int
    local_window_blocks: int
    sink_count: int
    partner_count: int
    random_seed: int


@dataclass(frozen=True)
class SupportCurvePoint:
    depth: int
    support_coverage_last_row: float
    support_coverage_mean: float
    support_coverage_min: float


@dataclass(frozen=True)
class SupportCurveSummary:
    support_coverage_last_row_auc: float
    support_coverage_mean_auc: float
    support_coverage_min_auc: float
    first_depth_last_row_support_ge_095: int | None
    first_depth_last_row_support_full: int | None


@dataclass(frozen=True)
class FrozenSupportCandidate:
    frozen_stage_idx: int
    curve: list[SupportCurvePoint]
    summary: SupportCurveSummary


@dataclass(frozen=True)
class CurvePoint:
    depth: int
    metrics: OperatorMetrics


@dataclass(frozen=True)
class CurveSummary:
    support_coverage_last_row_auc: float
    support_coverage_mean_auc: float
    normalized_effective_support_last_row_auc: float
    normalized_entropy_last_row_auc: float
    last_row_tv_to_uniform_auc: float
    last_row_max_mass_auc: float
    effective_rank_ratio_auc: float
    first_depth_last_row_support_ge_095: int | None
    first_depth_last_row_support_full: int | None


@dataclass(frozen=True)
class TopologyCurveResult:
    label: str
    weighting_model: str
    frozen_stage_idx: int | None
    curve: list[CurvePoint]
    summary: CurveSummary


@dataclass(frozen=True)
class WeightModelResult:
    weighting_model: str
    butterfly: TopologyCurveResult
    local_only: TopologyCurveResult
    frozen_reference: TopologyCurveResult


@dataclass(frozen=True)
class SupportInvariantResult:
    butterfly: FrozenSupportCandidate
    local_only: FrozenSupportCandidate
    frozen_reference_stage_idx: int
    frozen_candidates: list[FrozenSupportCandidate]


@dataclass(frozen=True)
class ExperimentCaseResult:
    num_blocks: int
    partner_rule: str
    width: int
    depth_max: int
    stage_count: int
    degree_budget: int
    support_invariant: SupportInvariantResult
    weight_models: list[WeightModelResult]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _first_depth_at_least(values: Sequence[float], threshold: float) -> int | None:
    for depth, value in enumerate(values, start=1):
        if float(value) >= float(threshold):
            return int(depth)
    return None


def _auc(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _measure_support_state(support_state: np.ndarray) -> SupportCurvePoint:
    support = np.asarray(support_state, dtype=bool)
    if support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise ValueError("support_state must be a square matrix")

    num_blocks = int(support.shape[0])
    coverage: list[float] = []
    for row_idx in range(num_blocks):
        prefix = support[row_idx, : row_idx + 1]
        coverage.append(float(np.count_nonzero(prefix)) / float(row_idx + 1))
    return SupportCurvePoint(
        depth=0,
        support_coverage_last_row=float(coverage[-1]),
        support_coverage_mean=float(np.mean(coverage)),
        support_coverage_min=float(np.min(coverage)),
    )


def _summarize_support_curve(curve: Sequence[SupportCurvePoint]) -> SupportCurveSummary:
    last_row_values = [point.support_coverage_last_row for point in curve]
    mean_values = [point.support_coverage_mean for point in curve]
    min_values = [point.support_coverage_min for point in curve]
    return SupportCurveSummary(
        support_coverage_last_row_auc=_auc(last_row_values),
        support_coverage_mean_auc=_auc(mean_values),
        support_coverage_min_auc=_auc(min_values),
        first_depth_last_row_support_ge_095=_first_depth_at_least(last_row_values, 0.95),
        first_depth_last_row_support_full=_first_depth_at_least(last_row_values, 1.0),
    )


def _summarize_curve(curve: Sequence[CurvePoint]) -> CurveSummary:
    metrics = [point.metrics for point in curve]
    return CurveSummary(
        support_coverage_last_row_auc=_auc([m.support_coverage_last_row for m in metrics]),
        support_coverage_mean_auc=_auc([m.support_coverage_mean for m in metrics]),
        normalized_effective_support_last_row_auc=_auc(
            [m.normalized_effective_support_last_row for m in metrics]
        ),
        normalized_entropy_last_row_auc=_auc([m.normalized_entropy_last_row for m in metrics]),
        last_row_tv_to_uniform_auc=_auc([m.last_row_tv_to_uniform for m in metrics]),
        last_row_max_mass_auc=_auc([m.last_row_max_mass for m in metrics]),
        effective_rank_ratio_auc=_auc([m.effective_rank_ratio for m in metrics]),
        first_depth_last_row_support_ge_095=_first_depth_at_least(
            [m.support_coverage_last_row for m in metrics],
            0.95,
        ),
        first_depth_last_row_support_full=_first_depth_at_least(
            [m.support_coverage_last_row for m in metrics],
            1.0,
        ),
    )


def _compute_support_states(
    *,
    row_sequence: Sequence[list[list[NeighborSpec]]],
    num_blocks: int,
) -> tuple[list[np.ndarray], list[SupportCurvePoint], SupportCurveSummary]:
    support_state = np.eye(int(num_blocks), dtype=bool)
    support_states: list[np.ndarray] = []
    curve: list[SupportCurvePoint] = []
    for depth, role_rows in enumerate(row_sequence, start=1):
        layer_support = build_support_operator(role_rows, int(num_blocks))
        support_state = (layer_support.astype(np.uint8) @ support_state.astype(np.uint8)) > 0
        support_states.append(np.asarray(support_state, dtype=bool).copy())
        measured = _measure_support_state(support_state)
        curve.append(
            SupportCurvePoint(
                depth=int(depth),
                support_coverage_last_row=float(measured.support_coverage_last_row),
                support_coverage_mean=float(measured.support_coverage_mean),
                support_coverage_min=float(measured.support_coverage_min),
            )
        )
    return support_states, curve, _summarize_support_curve(curve)


def _compute_weighted_curve(
    *,
    label: str,
    weighting_model: WeightingModel,
    row_sequence: Sequence[list[list[NeighborSpec]]],
    support_states: Sequence[np.ndarray],
    num_blocks: int,
    random_seed: int,
    frozen_stage_idx: int | None = None,
) -> TopologyCurveResult:
    weighted_state = np.eye(int(num_blocks), dtype=np.float64)
    curve: list[CurvePoint] = []
    for depth, (role_rows, support_state) in enumerate(
        zip(row_sequence, support_states, strict=True),
        start=1,
    ):
        layer_operator = build_role_weighted_operator(
            role_rows,
            int(num_blocks),
            weighting_model=str(weighting_model),
            layer_idx=int(depth - 1),
            random_seed=int(random_seed),
        )
        weighted_state = layer_operator @ weighted_state
        curve.append(
            CurvePoint(
                depth=int(depth),
                metrics=measure_operator(
                    weighted_state,
                    support_matrix=support_state,
                ),
            )
        )
    return TopologyCurveResult(
        label=str(label),
        weighting_model=str(weighting_model),
        frozen_stage_idx=frozen_stage_idx,
        curve=curve,
        summary=_summarize_curve(curve),
    )


def _select_frozen_reference_stage(
    candidates: Sequence[FrozenSupportCandidate],
) -> FrozenSupportCandidate:
    def score(candidate: FrozenSupportCandidate) -> tuple[float, float, float, int]:
        full_depth = candidate.summary.first_depth_last_row_support_full
        full_depth_score = -float(full_depth if full_depth is not None else 10**9)
        return (
            float(candidate.summary.support_coverage_last_row_auc),
            float(candidate.summary.support_coverage_mean_auc),
            float(full_depth_score),
            -int(candidate.frozen_stage_idx),
        )

    return max(candidates, key=score)


def run_one(cfg: ExperimentConfig) -> ExperimentCaseResult:
    width = int(butterfly_width(int(cfg.num_blocks)))
    depth_max = int(2 * width)
    stage_count = int(butterfly_stage_count(int(cfg.num_blocks), str(cfg.partner_rule)))
    degree_budget = int(
        observed_butterfly_degree_budget(
            num_blocks=int(cfg.num_blocks),
            block_size=int(cfg.block_size),
            local_window_blocks=int(cfg.local_window_blocks),
            sink_count=int(cfg.sink_count),
            partner_count=int(cfg.partner_count),
            partner_rule=str(cfg.partner_rule),
        )
    )

    butterfly_rows = [
        build_butterfly_neighbor_role_rows(
            num_blocks=int(cfg.num_blocks),
            layer_idx=int(layer_idx),
            block_size=int(cfg.block_size),
            local_window_blocks=int(cfg.local_window_blocks),
            sink_count=int(cfg.sink_count),
            partner_count=int(cfg.partner_count),
            partner_rule=str(cfg.partner_rule),
        )
        for layer_idx in range(depth_max)
    ]
    local_rows_one = build_local_only_neighbor_role_rows(
        num_blocks=int(cfg.num_blocks),
        degree_budget=int(degree_budget),
        sink_count=int(cfg.sink_count),
    )
    local_rows = [local_rows_one for _ in range(depth_max)]

    butterfly_support_states, butterfly_support_curve, butterfly_support_summary = (
        _compute_support_states(
            row_sequence=butterfly_rows,
            num_blocks=int(cfg.num_blocks),
        )
    )
    local_support_states, local_support_curve, local_support_summary = _compute_support_states(
        row_sequence=local_rows,
        num_blocks=int(cfg.num_blocks),
    )

    frozen_candidates: list[FrozenSupportCandidate] = []
    frozen_rows_by_stage: dict[int, list[list[list[NeighborSpec]]]] = {}
    frozen_support_states_by_stage: dict[int, list[np.ndarray]] = {}

    for frozen_stage_idx in range(int(stage_count)):
        frozen_rows_one = build_frozen_long_range_neighbor_role_rows(
            num_blocks=int(cfg.num_blocks),
            frozen_stage_idx=int(frozen_stage_idx),
            block_size=int(cfg.block_size),
            local_window_blocks=int(cfg.local_window_blocks),
            sink_count=int(cfg.sink_count),
            partner_count=int(cfg.partner_count),
            partner_rule=str(cfg.partner_rule),
        )
        frozen_rows = [frozen_rows_one for _ in range(depth_max)]
        support_states, support_curve, support_summary = _compute_support_states(
            row_sequence=frozen_rows,
            num_blocks=int(cfg.num_blocks),
        )
        frozen_rows_by_stage[int(frozen_stage_idx)] = frozen_rows
        frozen_support_states_by_stage[int(frozen_stage_idx)] = support_states
        frozen_candidates.append(
            FrozenSupportCandidate(
                frozen_stage_idx=int(frozen_stage_idx),
                curve=support_curve,
                summary=support_summary,
            )
        )

    frozen_reference = _select_frozen_reference_stage(frozen_candidates)
    frozen_reference_rows = frozen_rows_by_stage[int(frozen_reference.frozen_stage_idx)]
    frozen_reference_support_states = frozen_support_states_by_stage[int(frozen_reference.frozen_stage_idx)]

    weight_model_results: list[WeightModelResult] = []
    for weighting_model in WEIGHT_MODELS:
        butterfly_weighted = _compute_weighted_curve(
            label="butterfly",
            weighting_model=weighting_model,
            row_sequence=butterfly_rows,
            support_states=butterfly_support_states,
            num_blocks=int(cfg.num_blocks),
            random_seed=int(cfg.random_seed),
        )
        local_weighted = _compute_weighted_curve(
            label="local_only",
            weighting_model=weighting_model,
            row_sequence=local_rows,
            support_states=local_support_states,
            num_blocks=int(cfg.num_blocks),
            random_seed=int(cfg.random_seed),
        )
        frozen_weighted = _compute_weighted_curve(
            label="frozen_reference",
            weighting_model=weighting_model,
            row_sequence=frozen_reference_rows,
            support_states=frozen_reference_support_states,
            num_blocks=int(cfg.num_blocks),
            random_seed=int(cfg.random_seed),
            frozen_stage_idx=int(frozen_reference.frozen_stage_idx),
        )
        weight_model_results.append(
            WeightModelResult(
                weighting_model=str(weighting_model),
                butterfly=butterfly_weighted,
                local_only=local_weighted,
                frozen_reference=frozen_weighted,
            )
        )

    return ExperimentCaseResult(
        num_blocks=int(cfg.num_blocks),
        partner_rule=str(cfg.partner_rule),
        width=int(width),
        depth_max=int(depth_max),
        stage_count=int(stage_count),
        degree_budget=int(degree_budget),
        support_invariant=SupportInvariantResult(
            butterfly=FrozenSupportCandidate(
                frozen_stage_idx=-1,
                curve=butterfly_support_curve,
                summary=butterfly_support_summary,
            ),
            local_only=FrozenSupportCandidate(
                frozen_stage_idx=-1,
                curve=local_support_curve,
                summary=local_support_summary,
            ),
            frozen_reference_stage_idx=int(frozen_reference.frozen_stage_idx),
            frozen_candidates=frozen_candidates,
        ),
        weight_models=weight_model_results,
    )


def _support_rows(results: Sequence[ExperimentCaseResult]) -> list[str]:
    lines = [
        "## Support Over Depth",
        "",
        "| rule | blocks | ref frozen stage | butterfly AUC | local AUC | frozen AUC | butterfly full-support depth | local full-support depth | frozen full-support depth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        butterfly = case.support_invariant.butterfly.summary
        local_only = case.support_invariant.local_only.summary
        frozen_reference = next(
            candidate
            for candidate in case.support_invariant.frozen_candidates
            if candidate.frozen_stage_idx == case.support_invariant.frozen_reference_stage_idx
        ).summary
        lines.append(
            f"| {case.partner_rule} | {case.num_blocks} | {case.support_invariant.frozen_reference_stage_idx} | "
            f"{_fmt(butterfly.support_coverage_last_row_auc)} | "
            f"{_fmt(local_only.support_coverage_last_row_auc)} | "
            f"{_fmt(frozen_reference.support_coverage_last_row_auc)} | "
            f"{butterfly.first_depth_last_row_support_full or 'n/a'} | "
            f"{local_only.first_depth_last_row_support_full or 'n/a'} | "
            f"{frozen_reference.first_depth_last_row_support_full or 'n/a'} |"
        )
    return lines


def _weight_model_win_rows(results: Sequence[ExperimentCaseResult]) -> list[str]:
    lines = [
        "## Secondary Weighted Surrogate Diagnostics",
        "",
        "| weighting model | entropy wins | effective-support wins | TV wins | max-mass wins | e-rank wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_cases = len(results)
    for weighting_model in WEIGHT_MODELS:
        entropy_wins = 0
        effective_support_wins = 0
        tv_wins = 0
        max_mass_wins = 0
        spectral_wins = 0
        for case in results:
            row = next(item for item in case.weight_models if item.weighting_model == weighting_model)
            butterfly = row.butterfly.summary
            local_only = row.local_only.summary
            frozen_reference = row.frozen_reference.summary
            if (
                butterfly.normalized_entropy_last_row_auc
                > local_only.normalized_entropy_last_row_auc
                and butterfly.normalized_entropy_last_row_auc
                > frozen_reference.normalized_entropy_last_row_auc
            ):
                entropy_wins += 1
            if (
                butterfly.normalized_effective_support_last_row_auc
                > local_only.normalized_effective_support_last_row_auc
                and butterfly.normalized_effective_support_last_row_auc
                > frozen_reference.normalized_effective_support_last_row_auc
            ):
                effective_support_wins += 1
            if (
                butterfly.last_row_tv_to_uniform_auc
                < local_only.last_row_tv_to_uniform_auc
                and butterfly.last_row_tv_to_uniform_auc
                < frozen_reference.last_row_tv_to_uniform_auc
            ):
                tv_wins += 1
            if (
                butterfly.last_row_max_mass_auc
                < local_only.last_row_max_mass_auc
                and butterfly.last_row_max_mass_auc
                < frozen_reference.last_row_max_mass_auc
            ):
                max_mass_wins += 1
            if (
                butterfly.effective_rank_ratio_auc
                > local_only.effective_rank_ratio_auc
                and butterfly.effective_rank_ratio_auc
                > frozen_reference.effective_rank_ratio_auc
            ):
                spectral_wins += 1
        lines.append(
            f"| {weighting_model} | {entropy_wins}/{total_cases} | "
            f"{effective_support_wins}/{total_cases} | {tv_wins}/{total_cases} | "
            f"{max_mass_wins}/{total_cases} | {spectral_wins}/{total_cases} |"
        )
    return lines


def _overall_counts(results: Sequence[ExperimentCaseResult]) -> dict[str, int]:
    support_auc_wins = 0
    support_full_by_width = 0
    support_full_by_width_local = 0
    support_full_by_width_frozen = 0
    total_weighted_cases = len(results) * len(WEIGHT_MODELS)
    entropy_wins = 0
    effective_support_wins = 0
    tv_wins = 0
    max_mass_wins = 0
    spectral_wins = 0

    for case in results:
        butterfly = case.support_invariant.butterfly.summary
        local_only = case.support_invariant.local_only.summary
        frozen_reference = next(
            candidate
            for candidate in case.support_invariant.frozen_candidates
            if candidate.frozen_stage_idx == case.support_invariant.frozen_reference_stage_idx
        ).summary
        if (
            butterfly.support_coverage_last_row_auc > local_only.support_coverage_last_row_auc
            and butterfly.support_coverage_last_row_auc > frozen_reference.support_coverage_last_row_auc
        ):
            support_auc_wins += 1
        if (
            butterfly.first_depth_last_row_support_full is not None
            and butterfly.first_depth_last_row_support_full <= case.width
        ):
            support_full_by_width += 1
        if (
            local_only.first_depth_last_row_support_full is not None
            and local_only.first_depth_last_row_support_full <= case.width
        ):
            support_full_by_width_local += 1
        if (
            frozen_reference.first_depth_last_row_support_full is not None
            and frozen_reference.first_depth_last_row_support_full <= case.width
        ):
            support_full_by_width_frozen += 1

        for row in case.weight_models:
            butterfly_weighted = row.butterfly.summary
            local_weighted = row.local_only.summary
            frozen_weighted = row.frozen_reference.summary
            if (
                butterfly_weighted.normalized_entropy_last_row_auc
                > local_weighted.normalized_entropy_last_row_auc
                and butterfly_weighted.normalized_entropy_last_row_auc
                > frozen_weighted.normalized_entropy_last_row_auc
            ):
                entropy_wins += 1
            if (
                butterfly_weighted.normalized_effective_support_last_row_auc
                > local_weighted.normalized_effective_support_last_row_auc
                and butterfly_weighted.normalized_effective_support_last_row_auc
                > frozen_weighted.normalized_effective_support_last_row_auc
            ):
                effective_support_wins += 1
            if (
                butterfly_weighted.last_row_tv_to_uniform_auc
                < local_weighted.last_row_tv_to_uniform_auc
                and butterfly_weighted.last_row_tv_to_uniform_auc
                < frozen_weighted.last_row_tv_to_uniform_auc
            ):
                tv_wins += 1
            if (
                butterfly_weighted.last_row_max_mass_auc
                < local_weighted.last_row_max_mass_auc
                and butterfly_weighted.last_row_max_mass_auc
                < frozen_weighted.last_row_max_mass_auc
            ):
                max_mass_wins += 1
            if (
                butterfly_weighted.effective_rank_ratio_auc
                > local_weighted.effective_rank_ratio_auc
                and butterfly_weighted.effective_rank_ratio_auc
                > frozen_weighted.effective_rank_ratio_auc
            ):
                spectral_wins += 1

    return {
        "support_auc_wins": support_auc_wins,
        "support_full_by_width": support_full_by_width,
        "support_full_by_width_local": support_full_by_width_local,
        "support_full_by_width_frozen": support_full_by_width_frozen,
        "entropy_wins": entropy_wins,
        "effective_support_wins": effective_support_wins,
        "tv_wins": tv_wins,
        "max_mass_wins": max_mass_wins,
        "spectral_wins": spectral_wins,
        "total_support_cases": len(results),
        "total_weighted_cases": total_weighted_cases,
    }


def _markdown_summary(results: Sequence[ExperimentCaseResult]) -> str:
    overall = _overall_counts(results)
    lines = [
        "# Butterfly Staging Validity Experiment",
        "",
        "This artifact isolates the staged schedule from the mere presence of long-range edges.",
        "For each rule and block count, it compares:",
        "- staged Butterfly from the real `build_block_butterfly_layout(...)` topology",
        "- a degree-matched local-only baseline",
        "- the strongest frozen long-range control on the primary support metric: one fixed production partner stage reused every layer",
        "",
        "Support is measured from exact boolean reachability, so it is independent of the positive row-weight model.",
        "Spread / concentration is then measured under five deterministic admissible weighting models as secondary diagnostics.",
        "",
    ]
    lines.extend(_support_rows(results))
    lines.extend([""])
    lines.extend(_weight_model_win_rows(results))
    lines.extend(
        [
            "",
            "## What This Now Supports",
            "",
            (
                f"- Staging itself matters for support expansion. Butterfly beat both local-only "
                f"and the best frozen long-range control on last-row support AUC in "
                f"{overall['support_auc_wins']}/{overall['total_support_cases']} tested "
                f"(rule, block-count) cases."
            ),
            (
                f"- Full last-row causal-prefix support by `L = ceil(log2 N)` held for staged "
                f"Butterfly in {overall['support_full_by_width']}/{overall['total_support_cases']} "
                f"cases, versus {overall['support_full_by_width_local']}/"
                f"{overall['total_support_cases']} for local-only and "
                f"{overall['support_full_by_width_frozen']}/"
                f"{overall['total_support_cases']} for the reference frozen control."
            ),
            (
                "- Because support curves are computed from boolean reachability, this support "
                "advantage is a topology claim about the staged schedule rather than an artifact "
                "of any single row-normalization rule."
            ),
            "",
            "## What This Still Does Not Prove",
            "",
            (
                f"- Stronger mixing claims are not robust. Across "
                f"{overall['total_weighted_cases']} weighted comparisons, Butterfly beat both "
                f"controls on last-row entropy AUC only {overall['entropy_wins']} times, on "
                f"effective-support AUC {overall['effective_support_wins']} times, on TV-to-uniform "
                f"AUC {overall['tv_wins']} times, on max-mass AUC {overall['max_mass_wins']} times, "
                f"and on effective-rank-ratio AUC {overall['spectral_wins']} times."
            ),
            (
                "- Even on support, this is not an optimality theorem over all possible "
                "long-range schedules. It only compares the tested staged production topology "
                "against local-only and the best frozen reuse of one production stage."
            ),
            (
                "- That means the paper-safe claim is narrow: the staged schedule robustly expands "
                "causal support faster than local-only or frozen-long-range controls, but the current "
                "topology-only surrogate does not establish a general advantage in conditioning, "
                "entropy, or near-uniform mixing."
            ),
            (
                "- These remain CPU-only surrogate operators over block neighborhoods, not learned "
                "attention weights and not downstream task evidence."
            ),
        ]
    )
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _config_slug(*, num_blocks: Sequence[int], partner_rules: Sequence[str]) -> str:
    block_slug = "-".join(str(int(value)) for value in num_blocks)
    rule_slug = "-".join(str(value) for value in partner_rules)
    return f"blocks-{block_slug}_rules-{rule_slug}"


def _write_outputs(
    *,
    out_dir: Path,
    payload: dict[str, object],
    markdown: str,
    run_label: str,
) -> tuple[Path, Path, Path, Path]:
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_json_path = runs_dir / f"{run_label}.json"
    snapshot_md_path = runs_dir / f"{run_label}.md"
    snapshot_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    snapshot_md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path, snapshot_json_path, snapshot_md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-blocks",
        type=int,
        nargs="+",
        default=[16, 32, 33, 64, 128],
        help="Block counts to evaluate.",
    )
    parser.add_argument(
        "--partner-rules",
        type=str,
        nargs="+",
        default=["xor", "bit_reversal", "benes", "causal_shift"],
        choices=["xor", "bit_reversal", "benes", "causal_shift"],
        help="Butterfly partner schedules to evaluate.",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--local-window-blocks", type=int, default=1)
    parser.add_argument("--sink-count", type=int, default=1)
    parser.add_argument("--partner-count", type=int, default=1)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Deterministic seed for the Dirichlet weight model.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/proof/butterfly_staging_validity"),
        help="Directory for JSON and Markdown outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_label = (
        f"{generated_at_utc}_"
        f"{_config_slug(num_blocks=args.num_blocks, partner_rules=args.partner_rules)}"
    )

    configs = [
        ExperimentConfig(
            num_blocks=int(num_blocks),
            partner_rule=str(partner_rule),
            block_size=int(args.block_size),
            local_window_blocks=int(args.local_window_blocks),
            sink_count=int(args.sink_count),
            partner_count=int(args.partner_count),
            random_seed=int(args.random_seed),
        )
        for partner_rule in args.partner_rules
        for num_blocks in args.num_blocks
    ]
    results = [run_one(cfg) for cfg in configs]

    payload = {
        "experiment": "butterfly_staging_validity",
        "claim_checked": (
            "Using the real staged Butterfly block topology, determine whether the "
            "staged schedule itself expands causal-prefix support more effectively "
            "than local-only and frozen-long-range controls, and test whether any "
            "stronger spread / conditioning claim survives a small deterministic "
            "family of admissible row-weight models."
        ),
        "support_definition": (
            "Support curves are computed from exact boolean reachability, independent "
            "of the weight model."
        ),
        "weight_models": WEIGHT_MODEL_DESCRIPTIONS,
        "generated_at_utc": generated_at_utc,
        "git_commit": _git_commit(),
        "config": {
            "num_blocks": [int(x) for x in args.num_blocks],
            "partner_rules": [str(x) for x in args.partner_rules],
            "block_size": int(args.block_size),
            "local_window_blocks": int(args.local_window_blocks),
            "sink_count": int(args.sink_count),
            "partner_count": int(args.partner_count),
            "random_seed": int(args.random_seed),
            "depth_range": "1 .. 2 * ceil(log2(num_blocks))",
        },
        "results": [asdict(row) for row in results],
        "overall": _overall_counts(results),
    }

    markdown = _markdown_summary(results) + "\n"
    json_path, md_path, snapshot_json_path, snapshot_md_path = _write_outputs(
        out_dir=out_dir,
        payload=payload,
        markdown=markdown,
        run_label=run_label,
    )

    print(markdown, end="")
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved Snapshot JSON: {snapshot_json_path}")
    print(f"Saved Snapshot Markdown: {snapshot_md_path}")


if __name__ == "__main__":
    main()
