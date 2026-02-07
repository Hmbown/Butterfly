#!/usr/bin/env python3
"""Training dashboard: multi-run loss/perplexity/throughput overlays.

Reads experiment results JSON and plots comparison metrics.

Usage:
    python -m viz.dashboard experiments/dense_vs_hcsa/results.json --out dashboard.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")


def load_results(path: str | Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def plot_dashboard(
    results: List[Dict[str, Any]],
    save_path: str | Path | None = None,
    title: str = "Training Dashboard",
) -> None:
    """Plot comparison metrics across runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r.get("run_name", f"run_{i}") for i, r in enumerate(results)]
    val_ppls = [r.get("val_ppl", float("nan")) for r in results]
    val_losses = [r.get("val_loss", float("nan")) for r in results]
    throughputs = [r.get("tokens_per_sec", 0) for r in results]
    n_params = [r.get("n_params", 0) for r in results]

    # Color by attention type
    colors = []
    for r in results:
        cfg = r.get("config", {})
        if cfg.get("attn") == "hcsa":
            colors.append("#3B82F6")  # blue
        else:
            colors.append("#EF4444")  # red

    # 1. Val Perplexity bar chart
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), val_ppls, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Val Perplexity")
    ax.set_title("Validation Perplexity")
    ax.grid(axis="y", alpha=0.3)

    # 2. Throughput bar chart
    ax = axes[0, 1]
    ax.bar(range(len(names)), [t / 1000 for t in throughputs], color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Throughput (k tok/s)")
    ax.set_title("Training Throughput")
    ax.grid(axis="y", alpha=0.3)

    # 3. PPL vs throughput scatter
    ax = axes[1, 0]
    for i, (name, ppl, tput) in enumerate(zip(names, val_ppls, throughputs)):
        ax.scatter(tput / 1000, ppl, c=colors[i], s=80, zorder=5)
        ax.annotate(name, (tput / 1000, ppl), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Throughput (k tok/s)")
    ax.set_ylabel("Val Perplexity")
    ax.set_title("Perplexity vs Throughput")
    ax.grid(alpha=0.3)

    # 4. Parameters vs PPL
    ax = axes[1, 1]
    for i, (name, ppl, np_) in enumerate(zip(names, val_ppls, n_params)):
        ax.scatter(np_ / 1e6, ppl, c=colors[i], s=80, zorder=5)
        ax.annotate(name, (np_ / 1e6, ppl), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Val Perplexity")
    ax.set_title("Perplexity vs Model Size")
    ax.grid(alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EF4444", label="Dense"),
        Patch(facecolor="#3B82F6", label="HCSA"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=11)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=str, help="Path to results.json")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--title", type=str, default="Training Dashboard")
    args = p.parse_args()

    results = load_results(args.results)
    plot_dashboard(results, save_path=args.out, title=args.title)


if __name__ == "__main__":
    main()
