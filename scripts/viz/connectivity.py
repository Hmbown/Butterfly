#!/usr/bin/env python3
"""Graph connectivity analysis.

Computes and visualizes:
- Diameter
- Degree distribution
- Spectral gap (algebraic connectivity)
- Clustering coefficient

Usage:
    python -m viz.connectivity --seq-len 64 --cycle random --out connectivity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("matplotlib and numpy required")

from hcsa.cycles import cycle_prev_next_from_perm, random_cycle, greedy_cycle
from hcsa.attention_hcsa import _build_neighbors_index


def _build_adjacency_matrix(
    T: int,
    adj: List[List[int]],
) -> np.ndarray:
    """Build a T x T adjacency matrix from an adjacency list."""
    A = np.zeros((T, T), dtype=np.float64)
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            A[i, j] = 1
            A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def _build_adj_list(
    T: int,
    cycle: str = "random",
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
) -> List[List[int]]:
    """Build adjacency list from HCSA parameters (causal + acausal)."""
    torch.manual_seed(seed)

    if cycle == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = random_cycle(T, generator=g)
    elif cycle == "greedy":
        r = torch.randn(T, 16)
        perm = greedy_cycle(r, start=0)
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = random_cycle(T, generator=g)

    prev, nxt = cycle_prev_next_from_perm(perm)
    adj: List[set[int]] = [set() for _ in range(T)]
    for i in range(T):
        adj[i].add(int(prev[i]))
        adj[i].add(int(nxt[i]))
        for j in range(max(0, i - window), i):
            adj[i].add(j)
            adj[j].add(i)
        if landmark_stride > 0:
            for j in range(0, T, landmark_stride):
                if j != i:
                    adj[i].add(j)
                    adj[j].add(i)

    return [sorted(s) for s in adj]


def analyze_connectivity(
    T: int,
    cycle: str = "random",
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute connectivity metrics."""
    adj = _build_adj_list(T, cycle, window, landmark_stride, seed)
    A = _build_adjacency_matrix(T, adj)

    # Degree distribution
    degrees = A.sum(axis=1)
    avg_degree = float(degrees.mean())
    max_degree = float(degrees.max())
    min_degree = float(degrees.min())

    # Laplacian spectrum for spectral gap
    D = np.diag(degrees)
    L = D - A
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = float(eigenvalues[1]) if T > 1 else 0.0

    # Clustering coefficient
    clustering = 0.0
    for i in range(T):
        neighbors = [j for j in range(T) if A[i, j] > 0]
        k = len(neighbors)
        if k < 2:
            continue
        triangles = sum(
            1 for a in range(len(neighbors))
            for b in range(a + 1, len(neighbors))
            if A[neighbors[a], neighbors[b]] > 0
        )
        clustering += 2 * triangles / (k * (k - 1))
    clustering /= max(T, 1)

    # Diameter via BFS
    diameter = 0
    for start in range(min(T, 50)):  # sample for large T
        dist = [-1] * T
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in range(T):
                if A[u, v] > 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        max_dist = max(d for d in dist if d >= 0)
        diameter = max(diameter, max_dist)

    return {
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "spectral_gap": spectral_gap,
        "clustering_coefficient": clustering,
        "diameter": diameter,
    }


def plot_connectivity(
    T: int = 64,
    cycle: str = "random",
    window: int = 4,
    landmark_stride: int = 8,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> None:
    """Plot connectivity analysis."""
    adj = _build_adj_list(T, cycle, window, landmark_stride, seed)
    A = _build_adjacency_matrix(T, adj)
    metrics = analyze_connectivity(T, cycle, window, landmark_stride, seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Degree distribution
    degrees = A.sum(axis=1)
    axes[0, 0].hist(degrees, bins=max(10, int(degrees.max()) - int(degrees.min()) + 1),
                     color="#3B82F6", alpha=0.8, edgecolor="white")
    axes[0, 0].set_title("Degree Distribution")
    axes[0, 0].set_xlabel("Degree")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(metrics["avg_degree"], color="red", linestyle="--",
                        label=f"Mean={metrics['avg_degree']:.1f}")
    axes[0, 0].legend()

    # 2. Adjacency matrix
    axes[0, 1].imshow(A, cmap="Blues", interpolation="nearest")
    axes[0, 1].set_title("Adjacency Matrix")
    axes[0, 1].set_xlabel("Node")
    axes[0, 1].set_ylabel("Node")

    # 3. Laplacian eigenvalues
    D = np.diag(degrees)
    L = D - A
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    axes[1, 0].plot(eigenvalues, "o-", markersize=3, color="#3B82F6")
    axes[1, 0].set_title(f"Laplacian Spectrum (gap={metrics['spectral_gap']:.3f})")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Eigenvalue")
    axes[1, 0].grid(alpha=0.3)

    # 4. Metrics summary
    axes[1, 1].axis("off")
    text = "\n".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                     for k, v in metrics.items())
    axes[1, 1].text(0.1, 0.5, text, fontsize=12, family="monospace",
                    va="center", transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1, 1].set_title("Connectivity Metrics")

    fig.suptitle(
        f"Graph Connectivity (T={T}, cycle={cycle}, w={window}, lm={landmark_stride})",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--cycle", type=str, default="random")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--landmark-stride", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    metrics = analyze_connectivity(
        args.seq_len, args.cycle, args.window, args.landmark_stride, args.seed
    )
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_connectivity(
        T=args.seq_len, cycle=args.cycle, window=args.window,
        landmark_stride=args.landmark_stride, seed=args.seed,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
