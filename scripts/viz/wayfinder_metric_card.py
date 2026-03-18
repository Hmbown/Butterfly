#!/usr/bin/env python3
"""Generate a compact metric card from a Wayfinder stable profile summary JSON.

Example:
  python3 scripts/viz/wayfinder_metric_card.py \
    --stable-summary-json benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/stable_profile_summary.json \
    --out docs/assets/wayfinder_metric_card.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def _sec(value: float) -> str:
    return f"{value:.2f}s"


def _gb(value_bytes: float) -> str:
    return f"{value_bytes / (1024 ** 3):.2f} GB"


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _card(ax, x: float, y: float, w: float, h: float, title: str, value: str, detail: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#334155",
        facecolor="#111827",
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.07, title, color="#93c5fd", fontsize=11, fontweight="bold")
    ax.text(x + 0.02, y + h - 0.16, value, color="#f8fafc", fontsize=21, fontweight="bold")
    ax.text(x + 0.02, y + 0.05, detail, color="#cbd5e1", fontsize=10)


def render_metric_card(summary: dict, out_path: Path, title: str) -> None:
    dense = summary["dense"]
    wayfinder = summary["wayfinder"]
    delta_pct = summary["delta_wayfinder_vs_dense_pct"]

    prefill_pct = float(delta_pct["prefill_sec"])
    e2e_pct = float(delta_pct["e2e_sec"])
    memory_reduction = float(summary.get("memory_reduction_pct_convention", -delta_pct["peak_memory_bytes"]))

    fig = plt.figure(figsize=(11, 5.8), facecolor="#020617")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    run_id = summary.get("id", "unknown")
    model_path = summary.get("model_path", "unknown")
    seq_len = summary.get("seq_len", "?")
    decode_len = summary.get("decode_len", "?")

    ax.text(0.03, 0.92, title, color="#e2e8f0", fontsize=22, fontweight="bold")
    ax.text(
        0.03,
        0.86,
        f"Validated run {run_id} | model={model_path} | T={seq_len} decode_len={decode_len}",
        color="#94a3b8",
        fontsize=10,
    )

    _card(
        ax,
        x=0.03,
        y=0.40,
        w=0.29,
        h=0.38,
        title="Prefill Delta",
        value=_pct(prefill_pct),
        detail=f"dense {_sec(float(dense['prefill_sec']))} -> wayfinder {_sec(float(wayfinder['prefill_sec']))}",
    )
    _card(
        ax,
        x=0.355,
        y=0.40,
        w=0.29,
        h=0.38,
        title="End-to-End Delta",
        value=_pct(e2e_pct),
        detail=f"dense {_sec(float(dense['e2e_sec']))} -> wayfinder {_sec(float(wayfinder['e2e_sec']))}",
    )
    _card(
        ax,
        x=0.68,
        y=0.40,
        w=0.29,
        h=0.38,
        title="Peak Memory Reduction",
        value=_pct(memory_reduction),
        detail=f"dense {_gb(float(dense['peak_memory_bytes']))} -> wayfinder {_gb(float(wayfinder['peak_memory_bytes']))}",
    )

    ax.text(
        0.03,
        0.27,
        "Decode policy: dense-first for q_len <= 2 in validated default posture.",
        color="#cbd5e1",
        fontsize=10,
    )
    ax.text(
        0.03,
        0.22,
        "Source: docs/FIRST_RELEASE.md and stable_profile_summary.json",
        color="#94a3b8",
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wayfinder validated metric card")
    parser.add_argument("--stable-summary-json", required=True, help="Path to stable_profile_summary.json")
    parser.add_argument("--out", default="docs/assets/wayfinder_metric_card.png", help="Output image path")
    parser.add_argument(
        "--title",
        default="Wayfinder Validated Stable Profile",
        help="Figure title",
    )
    args = parser.parse_args()

    summary = _load_summary(Path(args.stable_summary_json))
    render_metric_card(summary=summary, out_path=Path(args.out), title=args.title)


if __name__ == "__main__":
    main()
