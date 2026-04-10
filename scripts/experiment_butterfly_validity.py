#!/usr/bin/env python3
"""Run the Butterfly topology validity experiment with support-first reporting.

This experiment stays intentionally narrow and paper-safe. It does not claim
that Butterfly attention solves downstream tasks. It only studies the causal
communication operator induced by the real staged block topology returned by
`build_block_butterfly_layout(...)`.

For each tested block count and Butterfly partner rule, the script:

1. Builds the exact per-layer block neighborhood from the production topology.
2. Converts each layer into a row-stochastic causal operator by uniformly
   averaging over each block's admissible predecessors.
3. Composes that operator for `L = ceil(log2(num_blocks))` layers.
4. Compares Butterfly against a matched-degree local-only baseline and a
   deterministic random-predecessor control.

The saved artifact reports support/reachability as the primary result and keeps
surrogate spread/conditioning diagnostics explicitly secondary.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.topology.butterfly import butterfly_width
from bna.topology.validation import (
    OperatorMetrics,
    build_butterfly_neighbor_rows,
    build_local_only_neighbor_rows,
    build_random_predecessor_neighbor_rows,
    butterfly_stage_count,
    compose_causal_operator,
    measure_operator,
    observed_butterfly_degree_budget,
)


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
class ExperimentResult:
    num_blocks: int
    partner_rule: str
    width: int
    analysis_layers: int
    stage_count: int
    degree_budget: int
    butterfly: OperatorMetrics
    local_only: OperatorMetrics
    random_predecessor: OperatorMetrics


def run_one(cfg: ExperimentConfig) -> ExperimentResult:
    width = int(butterfly_width(int(cfg.num_blocks)))
    analysis_layers = int(width)
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

    butterfly_operator = compose_causal_operator(
        num_blocks=int(cfg.num_blocks),
        num_layers=int(analysis_layers),
        row_builder=lambda layer_idx: build_butterfly_neighbor_rows(
            num_blocks=int(cfg.num_blocks),
            layer_idx=int(layer_idx),
            block_size=int(cfg.block_size),
            local_window_blocks=int(cfg.local_window_blocks),
            sink_count=int(cfg.sink_count),
            partner_count=int(cfg.partner_count),
            partner_rule=str(cfg.partner_rule),
        ),
    )
    local_only_operator = compose_causal_operator(
        num_blocks=int(cfg.num_blocks),
        num_layers=int(analysis_layers),
        row_builder=lambda _layer_idx: build_local_only_neighbor_rows(
            num_blocks=int(cfg.num_blocks),
            degree_budget=int(degree_budget),
            sink_count=int(cfg.sink_count),
        ),
    )
    random_predecessor_operator = compose_causal_operator(
        num_blocks=int(cfg.num_blocks),
        num_layers=int(analysis_layers),
        row_builder=lambda layer_idx: build_random_predecessor_neighbor_rows(
            num_blocks=int(cfg.num_blocks),
            layer_idx=int(layer_idx),
            degree_budget=int(degree_budget),
            local_window_blocks=int(cfg.local_window_blocks),
            sink_count=int(cfg.sink_count),
            seed=int(cfg.random_seed),
        ),
    )

    return ExperimentResult(
        num_blocks=int(cfg.num_blocks),
        partner_rule=str(cfg.partner_rule),
        width=int(width),
        analysis_layers=int(analysis_layers),
        stage_count=int(stage_count),
        degree_budget=int(degree_budget),
        butterfly=measure_operator(butterfly_operator),
        local_only=measure_operator(local_only_operator),
        random_predecessor=measure_operator(random_predecessor_operator),
    )


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _table_primary_support(results: Sequence[ExperimentResult]) -> list[str]:
    lines = [
        "## Primary Support Evidence At `L = ceil(log2 N)`",
        "",
        "| rule | blocks | degree | butterfly support | local-only support | random support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.partner_rule} | {row.num_blocks} | {row.degree_budget} | "
            f"{_fmt(row.butterfly.support_coverage_last_row)} | "
            f"{_fmt(row.local_only.support_coverage_last_row)} | "
            f"{_fmt(row.random_predecessor.support_coverage_last_row)} |"
        )
    return lines


def _table_secondary_spread(results: Sequence[ExperimentResult]) -> list[str]:
    lines = [
        "## Secondary Surrogate Spread Diagnostics",
        "",
        "| rule | blocks | butterfly eff-support/N | local eff-support/N | random eff-support/N | butterfly tv | local tv | random tv |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.partner_rule} | {row.num_blocks} | "
            f"{_fmt(row.butterfly.normalized_effective_support_last_row)} | "
            f"{_fmt(row.local_only.normalized_effective_support_last_row)} | "
            f"{_fmt(row.random_predecessor.normalized_effective_support_last_row)} | "
            f"{_fmt(row.butterfly.last_row_tv_to_uniform)} | "
            f"{_fmt(row.local_only.last_row_tv_to_uniform)} | "
            f"{_fmt(row.random_predecessor.last_row_tv_to_uniform)} |"
        )
    return lines


def _table_secondary_spectrum(results: Sequence[ExperimentResult]) -> list[str]:
    lines = [
        "## Secondary Surrogate Spectrum Diagnostics",
        "",
        "| rule | blocks | butterfly e-rank/N | local e-rank/N | random e-rank/N | butterfly stable/N | local stable/N | random stable/N | butterfly log10(cond) | local log10(cond) | random log10(cond) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.partner_rule} | {row.num_blocks} | "
            f"{_fmt(row.butterfly.effective_rank_ratio)} | "
            f"{_fmt(row.local_only.effective_rank_ratio)} | "
            f"{_fmt(row.random_predecessor.effective_rank_ratio)} | "
            f"{_fmt(row.butterfly.stable_rank_ratio)} | "
            f"{_fmt(row.local_only.stable_rank_ratio)} | "
            f"{_fmt(row.random_predecessor.stable_rank_ratio)} | "
            f"{_fmt(row.butterfly.log10_condition_number)} | "
            f"{_fmt(row.local_only.log10_condition_number)} | "
            f"{_fmt(row.random_predecessor.log10_condition_number)} |"
        )
    return lines


def _overall_counts(results: Sequence[ExperimentResult]) -> dict[str, int]:
    total_cases = len(results)
    butterfly_full_support = sum(
        1 for row in results if float(row.butterfly.support_coverage_last_row) >= 1.0
    )
    local_full_support = sum(
        1 for row in results if float(row.local_only.support_coverage_last_row) >= 1.0
    )
    random_full_support = sum(
        1 for row in results if float(row.random_predecessor.support_coverage_last_row) >= 1.0
    )
    return {
        "total_cases": total_cases,
        "butterfly_full_support": butterfly_full_support,
        "local_full_support": local_full_support,
        "random_full_support": random_full_support,
    }


def _markdown_summary(results: Sequence[ExperimentResult]) -> str:
    overall = _overall_counts(results)
    lines = [
        "# Butterfly Causal Support Experiment",
        "",
        "This artifact upgrades the earlier reachability-only check into a topology-only operator study with support-first reporting.",
        "Each layer is converted into a causal row-stochastic operator by uniformly averaging over the exact block neighborhood returned by `build_block_butterfly_layout(...)`.",
        "The comparison horizon is `L = ceil(log2(num_blocks))`, using a matched max-degree budget for all controls.",
        "",
    ]
    lines.extend(_table_primary_support(results))
    lines.extend([""])
    lines.extend(_table_secondary_spread(results))
    lines.extend([""])
    lines.extend(_table_secondary_spectrum(results))
    lines.extend(
        [
            "",
            "## Primary Claim Supported",
            "",
            (
                f"- Staged Butterfly reached full last-row causal-prefix support by "
                f"`L = ceil(log2 N)` in {overall['butterfly_full_support']}/"
                f"{overall['total_cases']} tested (rule, block-count) cases."
            ),
            (
                f"- At the same max-degree budget, the local-only baseline reached full "
                f"last-row support in {overall['local_full_support']}/"
                f"{overall['total_cases']} cases."
            ),
            (
                f"- The deterministic random-predecessor control reached full last-row "
                f"support in {overall['random_full_support']}/"
                f"{overall['total_cases']} cases, so this experiment supports support "
                f"sufficiency for the real Butterfly schedule, not uniqueness against "
                f"every matched-degree long-range control."
            ),
            "",
            "## Secondary Diagnostics (Non-Durable Claims)",
            "",
            "- Effective-support, TV-to-uniform, stable/effective-rank, and condition-number readouts are retained as surrogate diagnostics only.",
            "- These diagnostics are not the canonical proof claim and should not be treated as general mixing guarantees.",
            "",
            "## What This Still Does Not Prove",
            "",
            "- The primary support metric is last-row causal-prefix reachability at a fixed log-depth horizon. It is not a proof that every row, every horizon, or every admissible weighting rule yields comparable spread.",
            "- These are not learned attention weights. The operator uses uniform row normalization, so it only tests what the topology makes possible, not what a trained model will choose.",
            "- Singular values and condition numbers show that all of these operators are smoothing maps and become increasingly ill-conditioned with depth. That means this experiment does not establish end-to-end task quality, optimization behavior, or dense-attention equivalence.",
            "- This remains a block-level communication argument, not a theorem about token-level semantics or model accuracy.",
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
        default=["xor", "bit_reversal", "benes"],
        choices=["xor", "bit_reversal", "benes"],
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
        help="Deterministic seed for the random-predecessor control.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/proof/butterfly_validity"),
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
        "experiment": "butterfly_causal_support",
        "claim_checked": (
            "Using a topology-only row-stochastic operator built from the real staged "
            "Butterfly layout, test whether Butterfly reaches causal-prefix support "
            "more effectively than matched-degree controls at L = ceil(log2(num_blocks))."
        ),
        "secondary_diagnostics": (
            "Uniform-weight surrogate spread and conditioning metrics are reported as "
            "secondary diagnostics only."
        ),
        "operator_definition": (
            "Each block row uniformly averages over its exact admissible predecessors. "
            "This is a topology-only surrogate, not learned attention."
        ),
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
            "analysis_layers": "ceil(log2(num_blocks))",
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
