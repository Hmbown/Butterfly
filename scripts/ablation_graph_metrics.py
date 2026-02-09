#!/usr/bin/env python3
"""Graph metrics ablation: systematically explore how graph topology
affects spectral gap, diameter, and degree across configurations.

Produces NDJSON output (one JSON object per configuration) and a formatted
markdown summary table to stdout.

Usage:
    python3 scripts/ablation_graph_metrics.py \
        --seq-lens 256 512 1024 \
        --num-cycles 1 2 4 8 \
        --window-sizes 4 8 16 \
        --output results/graph_metrics_ablation.ndjson

    python3 scripts/ablation_graph_metrics.py --quick  # fast sanity check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Graph construction helpers
#
# We build adjacency structures directly from numpy permutations rather than
# going through torch-dependent GraphStrategy objects.  This keeps the script
# lightweight (no torch/mlx import on the critical path) and lets us run on
# any machine with numpy + scipy.
# ---------------------------------------------------------------------------


def _random_perm(T: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hamiltonian cycle as a permutation of [0..T-1]."""
    return rng.permutation(T).astype(np.int64)


def _edge_set(perm: np.ndarray) -> set[tuple[int, int]]:
    """Undirected edge set induced by a cycle permutation."""
    T = len(perm)
    edges: set[tuple[int, int]] = set()
    for i in range(T):
        u, v = int(perm[i]), int(perm[(i + 1) % T])
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return edges


def _edge_disjoint_random_cycles(
    T: int,
    k: int,
    rng: np.random.Generator,
    max_retries: int = 200,
) -> list[np.ndarray]:
    """Generate k edge-disjoint random Hamiltonian cycles."""
    cycles: list[np.ndarray] = []
    used: set[tuple[int, int]] = set()
    for _ in range(k):
        for _retry in range(max_retries):
            perm = _random_perm(T, rng)
            edges = _edge_set(perm)
            if used.isdisjoint(edges):
                cycles.append(perm)
                used |= edges
                break
        else:
            # Fall back to non-disjoint if we exhaust retries
            cycles.append(_random_perm(T, rng))
    return cycles


def _greedy_cycle(T: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy nearest-neighbor cycle using random routing vectors.

    We synthesize random routing embeddings (d=32) and run the greedy
    TSP heuristic: start at node 0, repeatedly pick the unvisited node
    with highest cosine similarity.  This is the same logic as
    hcsa/cycles.py:greedy_cycle but in pure numpy.
    """
    d = 32
    r = rng.standard_normal((T, d)).astype(np.float64)
    # Normalize for cosine similarity
    norms = np.linalg.norm(r, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    r = r / norms

    sim = r @ r.T  # [T, T] cosine similarity

    visited = np.zeros(T, dtype=bool)
    perm = np.empty(T, dtype=np.int64)
    cur = 0
    for t in range(T):
        perm[t] = cur
        visited[cur] = True
        if t < T - 1:
            scores = sim[cur].copy()
            scores[visited] = -1e9
            cur = int(np.argmax(scores))
    return perm


def build_adj_matrix(
    T: int,
    cycle_perms: list[np.ndarray],
    window: int,
    landmark_stride: int | None,
) -> np.ndarray:
    """Build undirected adjacency matrix from cycles + window + landmarks.

    This is the union graph that the attention mechanism actually uses
    (before causal masking).  We need the full undirected graph to compute
    spectral properties.
    """
    A = np.zeros((T, T), dtype=np.float64)

    # Cycle edges (undirected)
    for perm in cycle_perms:
        for i in range(T):
            u, v = int(perm[i]), int(perm[(i + 1) % T])
            if u != v:
                A[u, v] = 1.0
                A[v, u] = 1.0

    # Window edges (undirected version of causal window)
    for i in range(T):
        lo = max(0, i - window)
        for j in range(lo, i):
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Landmark edges (undirected)
    if landmark_stride is not None and landmark_stride > 0:
        for i in range(T):
            for j in range(0, T, landmark_stride):
                if j != i:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

    # Zero out diagonal (no self-loops for graph metrics)
    np.fill_diagonal(A, 0.0)
    return A


def build_adj_list(A: np.ndarray) -> list[list[int]]:
    """Convert adjacency matrix to adjacency list."""
    T = A.shape[0]
    adj: list[list[int]] = []
    for i in range(T):
        adj.append(np.nonzero(A[i])[0].tolist())
    return adj


# ---------------------------------------------------------------------------
# Graph metric computations
# ---------------------------------------------------------------------------


def compute_degree_stats(A: np.ndarray) -> Dict[str, float]:
    """Degree statistics from adjacency matrix."""
    deg = A.sum(axis=1)
    return {
        "degree_min": float(deg.min()),
        "degree_max": float(deg.max()),
        "degree_mean": float(deg.mean()),
        "degree_std": float(deg.std()),
    }


def compute_edge_count(A: np.ndarray) -> Dict[str, Any]:
    """Edge count and density."""
    T = A.shape[0]
    # A is symmetric; count upper triangle
    edges = int(np.triu(A, k=1).sum())
    max_edges = T * (T - 1) // 2
    density = edges / max(max_edges, 1)
    return {
        "edge_count": edges,
        "max_possible_edges": max_edges,
        "density": float(density),
    }


def compute_spectral_gap(A: np.ndarray) -> Dict[str, float]:
    """Adjacency spectral gap: lambda_1 - |lambda_2|.

    For regular or near-regular graphs, a large gap indicates good
    expansion (Alon-Boppana style).
    """
    T = A.shape[0]
    if T <= 1:
        return {
            "lambda_1": 0.0,
            "lambda_2_abs": 0.0,
            "adj_spectral_gap": 0.0,
            "expansion_ratio": 0.0,
        }

    if T <= 4096:
        eigvals = np.linalg.eigvalsh(A)
    else:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            sparse_A = sp.csr_matrix(A)
            eigvals, _ = spla.eigsh(sparse_A, k=min(6, T - 1), which="LA")
        except Exception:
            eigvals = np.linalg.eigvalsh(A)

    eigvals = np.sort(eigvals)[::-1]  # descending
    lambda_1 = float(eigvals[0])
    lambda_2_abs = float(np.abs(eigvals[1])) if len(eigvals) > 1 else 0.0

    deg_mean = float(A.sum(axis=1).mean())
    gap = lambda_1 - lambda_2_abs

    return {
        "lambda_1": lambda_1,
        "lambda_2_abs": lambda_2_abs,
        "adj_spectral_gap": float(gap),
        "expansion_ratio": float(deg_mean / max(lambda_2_abs, 1e-12)),
    }


def compute_laplacian_gap(A: np.ndarray) -> Dict[str, float]:
    """Laplacian spectral gap (Fiedler value / algebraic connectivity).

    lambda_2(L) bounds the Cheeger constant:
        lambda_2/2 <= h(G) <= sqrt(2 * lambda_2 * d_max)

    Higher = better expansion / faster mixing.
    """
    T = A.shape[0]
    if T <= 1:
        return {
            "fiedler_value": 0.0,
            "cheeger_lower": 0.0,
            "cheeger_upper": 0.0,
            "normalized_fiedler": 0.0,
        }

    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    if T <= 4096:
        eigvals = np.linalg.eigvalsh(L)
    else:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            sparse_L = sp.csr_matrix(L)
            eigvals, _ = spla.eigsh(sparse_L, k=min(6, T - 1), which="SM")
        except Exception:
            eigvals = np.linalg.eigvalsh(L)

    eigvals = np.sort(eigvals)
    # lambda_2 is the second-smallest; first should be ~0
    fiedler = float(max(eigvals[1], 0.0)) if len(eigvals) > 1 else 0.0
    d_max = float(deg.max())

    cheeger_lower = fiedler / 2.0
    cheeger_upper = math.sqrt(2.0 * fiedler * max(d_max, 1e-12))

    # Normalized: fiedler / degree_mean gives a scale-free measure
    deg_mean = float(deg.mean())
    normalized = fiedler / max(deg_mean, 1e-12)

    return {
        "fiedler_value": fiedler,
        "cheeger_lower": cheeger_lower,
        "cheeger_upper": cheeger_upper,
        "normalized_fiedler": normalized,
    }


def compute_diameter_and_paths(
    adj: list[list[int]],
    T: int,
    *,
    sample_fraction: float = 1.0,
    max_sources: int = 512,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """BFS-based diameter and average shortest path.

    For large graphs, we sample a subset of source nodes to keep
    runtime manageable.  The diameter is estimated as the max over
    sampled BFS trees.
    """
    if T <= 1:
        return {
            "diameter": 0,
            "avg_shortest_path": 0.0,
            "num_components": 1 if T == 1 else 0,
        }

    # Determine which source nodes to BFS from
    num_sources = min(T, max(1, int(T * sample_fraction)))
    num_sources = min(num_sources, max_sources)

    if num_sources < T:
        if rng is None:
            rng = np.random.default_rng(42)
        sources = rng.choice(T, size=num_sources, replace=False).tolist()
    else:
        sources = list(range(T))

    max_dist = 0
    total_dist = 0
    total_pairs = 0

    # Count connected components via full BFS from node 0
    component_count = 0
    seen_global = np.zeros(T, dtype=bool)

    for start in range(T):
        if seen_global[start]:
            continue
        component_count += 1
        q: deque[int] = deque([start])
        seen_global[start] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen_global[v]:
                    seen_global[v] = True
                    q.append(v)

    # BFS from sampled sources for distance metrics
    for src in sources:
        dist = np.full(T, -1, dtype=np.int32)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            d_u = dist[u]
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = d_u + 1
                    q.append(v)

        reachable = dist[dist >= 0]
        if len(reachable) > 1:
            max_dist = max(max_dist, int(reachable.max()))
            # Exclude self-distance
            total_dist += int(reachable.sum())
            total_pairs += len(reachable) - 1

    avg_path = float(total_dist) / max(total_pairs, 1)

    return {
        "diameter": int(max_dist),
        "avg_shortest_path": avg_path,
        "num_components": component_count,
    }


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    """A single graph configuration to evaluate."""

    name: str
    strategy: str  # window_only, random_cycles, greedy_cycles, landmarks,
    #                full_hybrid
    seq_len: int
    window: int
    num_cycles: int = 0
    landmark_stride: int | None = None
    cycle_type: str = "random"  # random or greedy
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        parts = [self.strategy, f"T={self.seq_len}", f"w={self.window}"]
        if self.num_cycles > 0:
            parts.append(f"k={self.num_cycles}")
        if self.landmark_stride is not None:
            parts.append(f"lm={self.landmark_stride}")
        return " ".join(parts)


def build_config_grid(
    seq_lens: list[int],
    num_cycles_list: list[int],
    window_sizes: list[int],
    landmark_strides: list[int],
    include_greedy: bool = False,
) -> list[GraphConfig]:
    """Build the full ablation grid."""
    configs: list[GraphConfig] = []

    for T in seq_lens:
        for w in window_sizes:
            if w >= T:
                continue  # window >= T is just dense

            # 1. Window-only baseline
            configs.append(
                GraphConfig(
                    name=f"window_only_T{T}_w{w}",
                    strategy="window_only",
                    seq_len=T,
                    window=w,
                )
            )

            # 2. Random cycles + window
            for k in num_cycles_list:
                configs.append(
                    GraphConfig(
                        name=f"random_cycles_T{T}_k{k}_w{w}",
                        strategy="random_cycles",
                        seq_len=T,
                        window=w,
                        num_cycles=k,
                        cycle_type="random",
                    )
                )

            # 3. Greedy cycles + window (optional, expensive for large T)
            if include_greedy:
                for k in num_cycles_list:
                    if T > 2048:
                        continue  # greedy is O(T^2) per cycle
                    configs.append(
                        GraphConfig(
                            name=f"greedy_cycles_T{T}_k{k}_w{w}",
                            strategy="greedy_cycles",
                            seq_len=T,
                            window=w,
                            num_cycles=k,
                            cycle_type="greedy",
                        )
                    )

            # 4. Landmarks + window (no cycles)
            for stride in landmark_strides:
                if stride >= T:
                    continue
                configs.append(
                    GraphConfig(
                        name=f"landmarks_T{T}_s{stride}_w{w}",
                        strategy="landmarks",
                        seq_len=T,
                        window=w,
                        landmark_stride=stride,
                    )
                )

            # 5. Full hybrid: cycles + window + landmarks
            for k in num_cycles_list:
                for stride in landmark_strides:
                    if stride >= T:
                        continue
                    configs.append(
                        GraphConfig(
                            name=f"hybrid_T{T}_k{k}_w{w}_s{stride}",
                            strategy="full_hybrid",
                            seq_len=T,
                            window=w,
                            num_cycles=k,
                            landmark_stride=stride,
                            cycle_type="random",
                        )
                    )

    return configs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_config(
    cfg: GraphConfig,
    rng: np.random.Generator,
    *,
    bfs_sample_fraction: float = 1.0,
    bfs_max_sources: int = 512,
) -> Dict[str, Any]:
    """Build graph for config and compute all metrics."""
    T = cfg.seq_len
    t0 = time.perf_counter()

    # Generate cycle permutations
    cycle_perms: list[np.ndarray] = []
    if cfg.num_cycles > 0:
        if cfg.cycle_type == "greedy":
            for _ in range(cfg.num_cycles):
                cycle_perms.append(_greedy_cycle(T, rng))
        else:
            cycle_perms = _edge_disjoint_random_cycles(T, cfg.num_cycles, rng)

    # Build adjacency matrix
    A = build_adj_matrix(
        T,
        cycle_perms,
        cfg.window,
        cfg.landmark_stride,
    )
    adj = build_adj_list(A)

    build_time = time.perf_counter() - t0

    # Compute metrics
    t1 = time.perf_counter()

    degree = compute_degree_stats(A)
    edges = compute_edge_count(A)
    spectral = compute_spectral_gap(A)
    laplacian = compute_laplacian_gap(A)
    paths = compute_diameter_and_paths(
        adj,
        T,
        sample_fraction=bfs_sample_fraction,
        max_sources=bfs_max_sources,
        rng=rng,
    )

    metrics_time = time.perf_counter() - t1

    # Efficiency score: spectral_gap / degree (bang for buck)
    efficiency = spectral["adj_spectral_gap"] / max(degree["degree_mean"], 1e-12)
    # Normalized efficiency: fiedler / degree
    norm_efficiency = laplacian["normalized_fiedler"]

    result: Dict[str, Any] = {
        # Config
        "name": cfg.name,
        "strategy": cfg.strategy,
        "seq_len": T,
        "window": cfg.window,
        "num_cycles": cfg.num_cycles,
        "landmark_stride": cfg.landmark_stride,
        "cycle_type": cfg.cycle_type,
        # Degree
        **degree,
        # Edges
        **edges,
        # Spectral (adjacency)
        **spectral,
        # Laplacian
        **laplacian,
        # Paths
        **paths,
        # Derived
        "efficiency_adj": efficiency,
        "efficiency_lap": norm_efficiency,
        # Timing
        "graph_build_s": build_time,
        "metrics_compute_s": metrics_time,
    }
    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def rank_configs_by_efficiency(
    results: list[Dict[str, Any]],
    metric: str = "efficiency_lap",
) -> list[Dict[str, Any]]:
    """Sort configs by spectral efficiency (spectral_gap / degree).

    Higher is better: more expansion per unit of degree budget.
    """
    return sorted(results, key=lambda r: -r.get(metric, 0.0))


def find_pareto_frontier(
    results: list[Dict[str, Any]],
    x: str = "degree_mean",
    y: str = "fiedler_value",
) -> list[Dict[str, Any]]:
    """Find Pareto-optimal configs: not dominated on (x, y).

    We want to MINIMIZE x (degree = compute cost) and MAXIMIZE y
    (spectral gap = quality).  A point is dominated if another point
    has both lower-or-equal x AND higher-or-equal y.
    """
    frontier: list[Dict[str, Any]] = []
    for r in results:
        rx, ry = r.get(x, 0.0), r.get(y, 0.0)
        dominated = False
        for other in results:
            if other is r:
                continue
            ox, oy = other.get(x, 0.0), other.get(y, 0.0)
            if ox <= rx and oy >= ry and (ox < rx or oy > ry):
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    # Sort by x for readability
    return sorted(frontier, key=lambda r: r.get(x, 0.0))


def plot_spectral_gap_vs_degree(
    results: list[Dict[str, Any]],
    output_path: str | Path | None = None,
) -> None:
    """Scatter plot of Fiedler value vs degree, colored by strategy.

    Only runs if matplotlib is available; silently skips otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[plot] matplotlib not available, skipping plot.",
            file=sys.stderr,
        )
        return

    strategies = sorted(set(r["strategy"] for r in results))
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(strategies), 1)))
    strategy_color = dict(zip(strategies, colors))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Fiedler value vs degree
    ax = axes[0]
    for r in results:
        c = strategy_color[r["strategy"]]
        ax.scatter(
            r["degree_mean"],
            r["fiedler_value"],
            color=c,
            alpha=0.7,
            s=40,
            label=r["strategy"],
        )
    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("Mean Degree")
    ax.set_ylabel("Fiedler Value (algebraic connectivity)")
    ax.set_title("Fiedler Value vs Degree")
    ax.grid(True, alpha=0.3)

    # Right: Efficiency (fiedler/degree) vs seq_len
    ax = axes[1]
    for r in results:
        c = strategy_color[r["strategy"]]
        ax.scatter(
            r["seq_len"],
            r["efficiency_lap"],
            color=c,
            alpha=0.7,
            s=40,
        )
    ax.set_xlabel("Sequence Length T")
    ax.set_ylabel("Efficiency (Fiedler / Degree)")
    ax.set_title("Spectral Efficiency vs Scale")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"[plot] Saved: {path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(v: float, width: int = 8) -> str:
    """Format a float for table display."""
    if abs(v) >= 1000:
        return f"{v:>{width},.1f}"
    if abs(v) >= 1:
        return f"{v:>{width}.3f}"
    if abs(v) >= 0.001:
        return f"{v:>{width}.4f}"
    return f"{v:>{width}.2e}"


def print_results_table(
    results: list[Dict[str, Any]],
    *,
    file: Any = None,
) -> None:
    """Print a formatted markdown table of results."""
    if file is None:
        file = sys.stdout

    # Group by seq_len for readability
    by_T: Dict[int, list[Dict[str, Any]]] = {}
    for r in results:
        T = r["seq_len"]
        by_T.setdefault(T, []).append(r)

    header = (
        "| strategy | T | w | k | lm | deg_mean | edges |"
        " fiedler | adj_gap | diam | avg_path | comps |"
        " eff_lap |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    print(header, file=file)
    print(sep, file=file)

    for T in sorted(by_T.keys()):
        rows = by_T[T]
        # Sort by efficiency within each T group
        rows_sorted = sorted(
            rows,
            key=lambda r: -r.get("efficiency_lap", 0.0),
        )
        for r in rows_sorted:
            lm = r["landmark_stride"]
            lm_str = str(lm) if lm is not None else "-"
            k_str = str(r["num_cycles"]) if r["num_cycles"] > 0 else "-"
            line = (
                f"| {r['strategy']:<15s}"
                f" | {r['seq_len']:>5d}"
                f" | {r['window']:>3d}"
                f" | {k_str:>3s}"
                f" | {lm_str:>4s}"
                f" | {_fmt(r['degree_mean'])}"
                f" | {r['edge_count']:>7d}"
                f" | {_fmt(r['fiedler_value'])}"
                f" | {_fmt(r['adj_spectral_gap'])}"
                f" | {r['diameter']:>4d}"
                f" | {_fmt(r['avg_shortest_path'])}"
                f" | {r['num_components']:>5d}"
                f" | {_fmt(r['efficiency_lap'])}"
                f" |"
            )
            print(line, file=file)


def print_pareto_summary(
    pareto: list[Dict[str, Any]],
    *,
    file: Any = None,
) -> None:
    """Print a summary of Pareto-optimal configurations."""
    if file is None:
        file = sys.stdout

    print("\n## Pareto Frontier (min degree, max fiedler)", file=file)
    print(
        "| name | deg_mean | fiedler | eff_lap |",
        file=file,
    )
    print("|---|---:|---:|---:|", file=file)
    for r in pareto:
        print(
            f"| {r['name']}"
            f" | {_fmt(r['degree_mean'])}"
            f" | {_fmt(r['fiedler_value'])}"
            f" | {_fmt(r['efficiency_lap'])}"
            f" |",
            file=file,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Graph metrics ablation: spectral gap, diameter, and degree "
            "across sparse attention configurations."
        ),
    )
    p.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to evaluate (default: 256 512 1024 2048)",
    )
    p.add_argument(
        "--num-cycles",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of Hamiltonian cycles to union (default: 1 2 4 8)",
    )
    p.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Local causal window sizes (default: 4 8 16)",
    )
    p.add_argument(
        "--landmark-strides",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Landmark strides to evaluate (default: 64 128)",
    )
    p.add_argument(
        "--include-greedy",
        action="store_true",
        help=("Include greedy cycle strategy (slow for T>1024, uses synthetic routing vectors)"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/graph_metrics_ablation.ndjson"),
        help="Output NDJSON file path",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save scatter plot to this path (e.g. results/ablation.png)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--bfs-max-sources",
        type=int,
        default=512,
        help="Max BFS sources for diameter estimation (default: 512)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=("Quick mode: small grid for sanity checking (T=128,256, k=1,2, w=4,8)"),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (still prints final table)",
    )

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    # Quick mode overrides
    if args.quick:
        args.seq_lens = [128, 256]
        args.num_cycles = [1, 2]
        args.window_sizes = [4, 8]
        args.landmark_strides = [32]
        args.include_greedy = False

    configs = build_config_grid(
        seq_lens=args.seq_lens,
        num_cycles_list=args.num_cycles,
        window_sizes=args.window_sizes,
        landmark_strides=args.landmark_strides,
        include_greedy=args.include_greedy,
    )

    if not args.quiet:
        print(
            f"Evaluating {len(configs)} configurations "
            f"(T in {args.seq_lens}, k in {args.num_cycles}, "
            f"w in {args.window_sizes})",
            file=sys.stderr,
        )

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results: list[Dict[str, Any]] = []
    timestamp = datetime.now(UTC).isoformat()

    for i, cfg in enumerate(configs):
        if not args.quiet:
            print(
                f"  [{i + 1}/{len(configs)}] {cfg.label} ...",
                end="",
                flush=True,
                file=sys.stderr,
            )

        t0 = time.perf_counter()

        # For large T, sample BFS sources to keep runtime bounded
        bfs_frac = 1.0
        if cfg.seq_len > 1024:
            bfs_frac = min(1.0, 256.0 / cfg.seq_len)

        try:
            result = evaluate_config(
                cfg,
                rng,
                bfs_sample_fraction=bfs_frac,
                bfs_max_sources=args.bfs_max_sources,
            )
            result["timestamp"] = timestamp
            result["status"] = "ok"
        except Exception as exc:
            result = {
                "name": cfg.name,
                "strategy": cfg.strategy,
                "seq_len": cfg.seq_len,
                "window": cfg.window,
                "num_cycles": cfg.num_cycles,
                "landmark_stride": cfg.landmark_stride,
                "cycle_type": cfg.cycle_type,
                "status": "error",
                "error": str(exc),
                "timestamp": timestamp,
            }

        elapsed = time.perf_counter() - t0
        result["wall_time_s"] = elapsed
        results.append(result)

        if not args.quiet:
            if result["status"] == "ok":
                print(
                    f" fiedler={result['fiedler_value']:.4f}"
                    f" deg={result['degree_mean']:.1f}"
                    f" diam={result['diameter']}"
                    f" ({elapsed:.2f}s)",
                    file=sys.stderr,
                )
            else:
                print(
                    f" ERROR: {result.get('error', '?')}",
                    file=sys.stderr,
                )

    # Write NDJSON (append mode so repeated runs accumulate)
    with open(args.output, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    if not args.quiet:
        print(
            f"\nWrote {len(results)} records to {args.output}",
            file=sys.stderr,
        )

    # Filter successful results for analysis
    ok_results = [r for r in results if r["status"] == "ok"]

    if ok_results:
        # Print markdown table
        print("\n# Graph Metrics Ablation Results\n")
        print(f"Generated: {timestamp}")
        print(f"Configs evaluated: {len(ok_results)}")
        print(f"Sequence lengths: {sorted(set(r['seq_len'] for r in ok_results))}")
        print()
        print_results_table(ok_results)

        # Pareto frontier
        pareto = find_pareto_frontier(ok_results)
        if pareto:
            print_pareto_summary(pareto)

        # Top-5 by efficiency
        ranked = rank_configs_by_efficiency(ok_results)
        print("\n## Top 5 by Spectral Efficiency (fiedler/degree)")
        for i, r in enumerate(ranked[:5]):
            print(
                f"  {i + 1}. {r['name']}: "
                f"eff={r['efficiency_lap']:.4f} "
                f"(fiedler={r['fiedler_value']:.4f}, "
                f"deg={r['degree_mean']:.1f})"
            )

        # Plot if requested
        if args.plot is not None:
            plot_spectral_gap_vs_degree(ok_results, output_path=args.plot)

    total_time = sum(r.get("wall_time_s", 0) for r in results)
    print(f"\nTotal wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
