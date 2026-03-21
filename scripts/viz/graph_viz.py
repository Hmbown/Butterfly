#!/usr/bin/env python3
"""Cycle graph visualization with circle layout.

Draws token positions on a circle, with colored edges for:
- Cycle edges (blue)
- Window edges (green)
- Landmark edges (red)
- Causal masking overlay (gray dashed for acausal edges)

Usage:
    python -m viz.graph_viz --seq-len 32 --out cycle_graph.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
except ImportError:
    raise ImportError("matplotlib and networkx required: pip install matplotlib networkx")

from bna.cycles import cycle_prev_next_from_perm, random_cycle, greedy_cycle


def build_cycle_graph(
    T: int,
    cycle: str = "random",
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
) -> tuple[nx.DiGraph, dict[tuple[int, int], str]]:
    """Build a networkx directed graph with edge types.

    Returns (graph, edge_types) where edge_types maps (i, j) to one of
    'cycle', 'window', 'landmark'.
    """
    torch.manual_seed(seed)

    # Build cycle
    if cycle == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = random_cycle(T, generator=g, device=torch.device("cpu"))
    elif cycle == "greedy":
        r = torch.randn(T, 16)
        perm = greedy_cycle(r, start=0)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = random_cycle(T, generator=g, device=torch.device("cpu"))

    prev, nxt = cycle_prev_next_from_perm(perm)

    G = nx.DiGraph()
    G.add_nodes_from(range(T))
    edge_types: dict[tuple[int, int], str] = {}

    # Add cycle edges (both directions since cycle is undirected)
    for i in range(T):
        p = int(prev[i])
        n = int(nxt[i])
        for j in [p, n]:
            if (i, j) not in edge_types:
                edge_types[(i, j)] = "cycle"
                G.add_edge(i, j)

    # Add window edges
    for i in range(T):
        for j in range(max(0, i - window), i):
            if (i, j) not in edge_types:
                edge_types[(i, j)] = "window"
                G.add_edge(i, j)

    # Add landmark edges
    if landmark_stride and landmark_stride > 0:
        for i in range(T):
            for j in range(0, i, landmark_stride):
                if (i, j) not in edge_types:
                    edge_types[(i, j)] = "landmark"
                    G.add_edge(i, j)

    return G, edge_types


def plot_cycle_graph(
    T: int = 32,
    cycle: str = "random",
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
    show_causal: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot the graph on a circle layout with colored edges."""
    G, edge_types = build_cycle_graph(T, cycle, window, landmark_stride, seed)

    # Circle layout
    import numpy as np

    angles = np.linspace(0, 2 * np.pi, T, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Color mapping
    color_map = {"cycle": "#3B82F6", "window": "#22C55E", "landmark": "#DC2626"}
    alpha_map = {"cycle": 0.8, "window": 0.5, "landmark": 0.6}

    # Draw edges by type (landmarks first, then window, then cycle on top)
    for etype in ["landmark", "window", "cycle"]:
        edges = [(u, v) for (u, v), t in edge_types.items() if t == etype]
        # Filter for causal edges (j <= i) if show_causal
        if show_causal:
            edges = [(u, v) for u, v in edges if v <= u]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=color_map[etype], alpha=alpha_map[etype],
            arrows=False, width=1.5 if etype == "cycle" else 0.8,
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=max(300 - T * 5, 30),
        node_color="#1E293B", edgecolors="white", linewidths=1.0,
    )

    # Labels
    if T <= 48:
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=max(8 - T // 10, 5),
            font_color="white", font_weight="bold",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map["cycle"], label=f"Cycle edges"),
        mpatches.Patch(color=color_map["window"], label=f"Window (w={window})"),
        mpatches.Patch(color=color_map["landmark"], label=f"Landmark (s={landmark_stride})"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

    ax.set_title(
        f"HCSA Graph (T={T}, cycle={cycle}, w={window}, lm={landmark_stride})",
        fontsize=13,
    )
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--cycle", type=str, default="random")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--landmark-stride", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    plot_cycle_graph(
        T=args.seq_len,
        cycle=args.cycle,
        window=args.window,
        landmark_stride=args.landmark_stride,
        seed=args.seed,
        show_causal=not args.no_causal,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
