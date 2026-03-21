"""Expander quality analysis for Wayfinder attention graphs.

This module provides graph-theoretic diagnostics that operate directly
on the ``neigh_idx [T, D]`` format produced by ``WayfinderGraphABI``.

**Why expander quality matters for attention.**

In sparse attention, each token sees a fixed-size neighborhood instead
of all previous tokens.  Information must propagate across *multiple
layers* to reach distant positions.  The number of layers needed is
governed by the *mixing time* of random walks on the attention graph,
which is itself controlled by the *spectral gap*.

A graph is a good expander when:
  - The spectral gap (1 - |lambda_2|) of its normalized adjacency is
    bounded away from zero.  Larger gap => faster mixing.
  - Every small vertex subset S has a neighborhood |N(S)| >> |S|.
  - The effective diameter is O(log n), so no token is more than a
    few hops from any other.

These properties directly map to transformer behavior:
  - Spectral gap  =>  layers needed for global information flow
  - Expansion     =>  robustness to token dropout / masking
  - Diameter      =>  worst-case depth for cross-document reasoning

All functions use NumPy for computation (CPU-side, during graph
construction or ablation sweeps).

Notation
--------
- T : sequence length (number of token positions / graph nodes)
- D : max neighbor degree in the padded neigh_idx tensor
- neigh_idx [T, D] : neighbor index tensor, -1 = padding
- adj [T, T] : dense adjacency matrix (undirected, unweighted)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


__all__ = [
    "spectral_gap",
    "expansion_ratio",
    "mixing_time_estimate",
    "effective_diameter",
    "degree_stats",
    "graph_quality_report",
    "is_good_expander",
    "neigh_idx_to_adj",
    "causal_reachability",
    "causal_diameter",
    "causal_mixing_comparison",
    "graph_quality_report_causal",
]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def neigh_idx_to_adj(neigh_idx: np.ndarray) -> np.ndarray:
    """Convert a padded neighbor-index tensor to a dense adjacency matrix.

    Parameters
    ----------
    neigh_idx : ndarray of shape [T, D]
        Padded neighbor indices.  Entries == -1 are padding.
        If shape is [H, T, D] (multi-head), the union across
        all heads is taken.

    Returns
    -------
    adj : ndarray of shape [T, T], dtype float64
        Symmetric binary adjacency matrix (undirected, unweighted).
        Self-loops are preserved if present in neigh_idx.
    """
    ni = np.asarray(neigh_idx)
    if ni.ndim == 3:
        # Multi-head: union of all heads
        T = int(ni.shape[1])
        adj = np.zeros((T, T), dtype=np.float64)
        for h in range(ni.shape[0]):
            adj = np.maximum(adj, _single_head_to_adj(ni[h]))
        return adj
    if ni.ndim != 2:
        raise ValueError(f"neigh_idx must be [T, D] or [H, T, D], got {ni.shape}")
    return _single_head_to_adj(ni)


def _single_head_to_adj(ni: np.ndarray) -> np.ndarray:
    """Build symmetric adjacency from single-head [T, D] neigh_idx."""
    T = int(ni.shape[0])
    adj = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        row = ni[i]
        neighbors = row[row >= 0]
        if neighbors.size > 0:
            adj[i, neighbors] = 1.0
            adj[neighbors, i] = 1.0  # symmetrize
    return adj


def _neigh_idx_to_adj_list(
    neigh_idx: np.ndarray,
) -> list[list[int]]:
    """Convert neigh_idx [T, D] to adjacency list (undirected)."""
    ni = np.asarray(neigh_idx)
    if ni.ndim == 3:
        # Multi-head: union
        T = int(ni.shape[1])
        adj: list[set[int]] = [set() for _ in range(T)]
        for h in range(ni.shape[0]):
            for i in range(T):
                row = ni[h, i]
                for j in row[row >= 0].tolist():
                    j = int(j)
                    adj[i].add(j)
                    adj[j].add(i)
        return [sorted(s) for s in adj]
    if ni.ndim != 2:
        raise ValueError(f"neigh_idx must be [T, D] or [H, T, D], got {ni.shape}")
    T = int(ni.shape[0])
    adj_sets: list[set[int]] = [set() for _ in range(T)]
    for i in range(T):
        row = ni[i]
        for j in row[row >= 0].tolist():
            j = int(j)
            adj_sets[i].add(j)
            adj_sets[j].add(i)
    return [sorted(s) for s in adj_sets]


# ---------------------------------------------------------------------------
# 1. Spectral gap
# ---------------------------------------------------------------------------


def spectral_gap(
    adj_or_neigh_idx: np.ndarray,
    *,
    normalized: bool = True,
    method: str = "auto",
) -> dict[str, Any]:
    """Compute the spectral gap of the graph.

    The spectral gap measures how well-connected the graph is.
    For a d-regular expander, the Alon-Boppana bound gives
    |lambda_2| >= 2*sqrt(d-1) - o(1), so the gap
    1 - |lambda_2/lambda_1| is at most 1 - 2*sqrt(d-1)/d.

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Either a dense adjacency matrix [T, T] or a padded
        neighbor-index tensor [T, D] / [H, T, D].
    normalized : bool
        If True, compute gap of the *normalized* adjacency
        D^{-1/2} A D^{-1/2} (eigenvalues in [-1, 1]).
        If False, compute gap of the raw adjacency matrix.
    method : str
        ``"auto"`` picks numpy for T <= 4096, scipy.sparse
        otherwise.  ``"dense"`` forces numpy.  ``"sparse"``
        forces scipy.

    Returns
    -------
    dict with keys:
        - nodes: int
        - spectral_gap: float  (1 - |lambda_2 / lambda_1|)
        - lambda_1: float
        - lambda_2: float  (second eigenvalue by magnitude)
        - normalized: bool
        - method: str  (eigen solver used)
    """
    adj = _ensure_adj(adj_or_neigh_idx)
    T = int(adj.shape[0])

    if T <= 1:
        return {
            "nodes": T,
            "spectral_gap": 0.0,
            "lambda_1": 0.0,
            "lambda_2": 0.0,
            "normalized": normalized,
            "method": "degenerate",
        }

    mat = adj
    if normalized:
        mat = _normalized_adjacency(adj)

    lambda_1, lambda_2, eig_method = _top_two_eigenvalues(mat, method=method)

    if abs(lambda_1) < 1e-12:
        gap = 0.0
    else:
        gap = 1.0 - abs(lambda_2) / abs(lambda_1)
        gap = max(gap, 0.0)

    return {
        "nodes": T,
        "spectral_gap": float(gap),
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "normalized": normalized,
        "method": eig_method,
    }


def _ensure_adj(adj_or_neigh_idx: np.ndarray) -> np.ndarray:
    """Accept either an adjacency matrix or neigh_idx and return adj."""
    arr = np.asarray(adj_or_neigh_idx)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        # Looks like an adjacency matrix already
        return arr.astype(np.float64, copy=False)
    # Treat as neigh_idx
    return neigh_idx_to_adj(arr)


def _normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    """Compute D^{-1/2} A D^{-1/2} (normalized adjacency)."""
    deg = adj.sum(axis=1)
    # Avoid division by zero for isolated nodes
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    # D^{-1/2} A D^{-1/2}
    D = np.diag(deg_inv_sqrt)
    return D @ adj @ D


def _top_two_eigenvalues(
    mat: np.ndarray,
    *,
    method: str = "auto",
) -> tuple[float, float, str]:
    """Return (lambda_1, lambda_2, method_name) for a symmetric matrix.

    lambda_1 is the largest eigenvalue by magnitude.
    lambda_2 is the second largest by magnitude.
    """
    T = int(mat.shape[0])
    use_sparse = method == "sparse" or (method == "auto" and T > 4096)

    if use_sparse:
        return _top_two_sparse(mat)
    return _top_two_dense(mat)


def _top_two_dense(mat: np.ndarray) -> tuple[float, float, str]:
    """Dense eigendecomposition via numpy."""
    eigvals = np.linalg.eigvalsh(mat)
    # Sort by absolute value descending
    order = np.argsort(-np.abs(eigvals))
    lambda_1 = float(eigvals[order[0]])
    lambda_2 = float(eigvals[order[1]]) if len(order) > 1 else 0.0
    return lambda_1, lambda_2, "numpy.linalg.eigvalsh"


def _top_two_sparse(
    mat: np.ndarray,
) -> tuple[float, float, str]:
    """Sparse eigendecomposition via scipy, with dense fallback."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        sparse_mat = sp.csr_matrix(mat)
        # Request top 3 to ensure we get a reliable second eigenvalue
        k = min(3, mat.shape[0] - 1)
        if k < 2:
            return _top_two_dense(mat)
        vals, _ = spla.eigsh(sparse_mat, k=k, which="LM")
        vals = np.sort(np.abs(vals))[::-1]
        return (
            float(vals[0]),
            float(vals[1]) if len(vals) > 1 else 0.0,
            "scipy.sparse.linalg.eigsh",
        )
    except Exception:
        l1, l2, _ = _top_two_dense(mat)
        return l1, l2, "numpy.linalg.eigvalsh_fallback"


# ---------------------------------------------------------------------------
# 2. Vertex expansion ratio
# ---------------------------------------------------------------------------


def expansion_ratio(
    adj_or_neigh_idx: np.ndarray,
    *,
    sample_size: int = 200,
    subset_sizes: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate vertex expansion by sampling random subsets.

    For a subset S of vertices, the *vertex expansion* is
    |N(S) \\ S| / |S|, where N(S) is the set of all neighbors
    of vertices in S (excluding S itself).

    A good expander has expansion >= 1 for all |S| <= n/2.
    We estimate this by sampling random subsets of various sizes
    and reporting the minimum observed ratio.

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Adjacency matrix [T, T] or neigh_idx [T, D] / [H, T, D].
    sample_size : int
        Number of random subsets to sample per subset size.
    subset_sizes : list[int] or None
        Explicit subset sizes to test.  If None, uses a
        geometrically-spaced sequence from 1 to T/2.
    rng : Generator or None
        NumPy random generator for reproducibility.

    Returns
    -------
    dict with keys:
        - nodes: int
        - min_expansion: float  (worst observed |N(S)\\S|/|S|)
        - mean_expansion: float
        - expansion_by_size: dict[int, float]  (per subset size)
        - sample_size: int
    """
    adj_list = _ensure_adj_list(adj_or_neigh_idx)
    T = len(adj_list)
    rand = rng if rng is not None else np.random.default_rng()

    if T <= 1:
        return {
            "nodes": T,
            "min_expansion": 0.0,
            "mean_expansion": 0.0,
            "expansion_by_size": {},
            "sample_size": sample_size,
        }

    # Determine subset sizes to probe
    if subset_sizes is None:
        max_s = max(1, T // 2)
        # Geometric sequence: 1, 2, 4, ..., up to T/2
        sizes = []
        s = 1
        while s <= max_s:
            sizes.append(s)
            s *= 2
        if sizes[-1] != max_s and max_s > 1:
            sizes.append(max_s)
        subset_sizes = sizes

    # Clamp to valid range
    subset_sizes = [s for s in subset_sizes if 1 <= s <= T // 2]
    if not subset_sizes:
        subset_sizes = [1]

    all_ratios: list[float] = []
    expansion_by_size: dict[int, float] = {}

    vertices = np.arange(T, dtype=np.int64)

    for s_size in subset_sizes:
        ratios_for_size: list[float] = []
        for _ in range(sample_size):
            S = set(rand.choice(vertices, size=s_size, replace=False).tolist())
            # Compute N(S) \ S
            boundary: set[int] = set()
            for u in S:
                for v in adj_list[u]:
                    if v not in S:
                        boundary.add(v)
            ratio = len(boundary) / max(len(S), 1)
            ratios_for_size.append(ratio)

        mean_r = float(np.mean(ratios_for_size))
        expansion_by_size[s_size] = mean_r
        all_ratios.extend(ratios_for_size)

    return {
        "nodes": T,
        "min_expansion": float(min(all_ratios)) if all_ratios else 0.0,
        "mean_expansion": (float(np.mean(all_ratios)) if all_ratios else 0.0),
        "expansion_by_size": expansion_by_size,
        "sample_size": sample_size,
    }


def _ensure_adj_list(
    adj_or_neigh_idx: np.ndarray,
) -> list[list[int]]:
    """Accept adj matrix or neigh_idx, return adjacency list."""
    arr = np.asarray(adj_or_neigh_idx)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        # Adjacency matrix
        T = arr.shape[0]
        adj_list: list[list[int]] = []
        for i in range(T):
            adj_list.append(np.where(arr[i] > 0)[0].tolist())
        return adj_list
    # neigh_idx format
    return _neigh_idx_to_adj_list(arr)


# ---------------------------------------------------------------------------
# 3. Mixing time estimate
# ---------------------------------------------------------------------------


def mixing_time_estimate(
    adj_or_neigh_idx: np.ndarray,
    *,
    epsilon: float = 0.01,
    method: str = "spectral",
) -> dict[str, Any]:
    """Estimate random-walk mixing time.

    The mixing time t_mix(epsilon) is the number of steps until the
    distribution of a lazy random walk is within total variation
    distance epsilon of the stationary distribution.

    For a d-regular graph with spectral gap gamma:
      t_mix(eps) <= (1/gamma) * ln(n / eps)

    This maps directly to transformer depth: if mixing time is k,
    then k layers of sparse attention suffice for full information
    propagation.

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Adjacency matrix [T, T] or neigh_idx [T, D] / [H, T, D].
    epsilon : float
        Target total variation distance (default 0.01).
    method : str
        ``"spectral"`` uses the spectral gap bound (fast).
        ``"power"`` explicitly iterates the transition matrix
        (accurate but O(T^2) per step, only feasible for T <= 2048).

    Returns
    -------
    dict with keys:
        - nodes: int
        - mixing_time: int  (estimated steps to eps-close)
        - spectral_gap: float  (of normalized adjacency)
        - method: str
        - epsilon: float
        - interpretation: str  (human-readable)
    """
    adj = _ensure_adj(adj_or_neigh_idx)
    T = int(adj.shape[0])

    if T <= 1:
        return {
            "nodes": T,
            "mixing_time": 0,
            "spectral_gap": 0.0,
            "method": "degenerate",
            "epsilon": epsilon,
            "interpretation": "trivial (single node)",
        }

    if method == "spectral" or (method == "auto" and T > 2048):
        return _mixing_time_spectral(adj, T, epsilon)
    elif method == "power":
        return _mixing_time_power(adj, T, epsilon)
    else:
        return _mixing_time_spectral(adj, T, epsilon)


def _mixing_time_spectral(
    adj: np.ndarray,
    T: int,
    epsilon: float,
) -> dict[str, Any]:
    """Upper bound via spectral gap: t_mix <= (1/gap) * ln(T/eps)."""
    gap_info = spectral_gap(adj, normalized=True)
    gap = gap_info["spectral_gap"]

    if gap < 1e-12:
        t_mix = T * T  # pessimistic: graph is disconnected or near so
        interp = "disconnected or near-disconnected graph"
    else:
        # Standard bound: t_mix(eps) <= (1/gap) * ln(sqrt(T) / eps)
        # The sqrt(T) comes from the l2 -> TV conversion for a
        # uniform start distribution.
        t_mix = int(math.ceil((1.0 / gap) * math.log(math.sqrt(T) / max(epsilon, 1e-15))))
        layers_gloss = (
            "excellent"
            if t_mix <= 4
            else "good"
            if t_mix <= 8
            else "moderate"
            if t_mix <= 16
            else "poor"
        )
        interp = f"{t_mix} layers for global mixing ({layers_gloss})"

    return {
        "nodes": T,
        "mixing_time": t_mix,
        "spectral_gap": float(gap),
        "method": "spectral_bound",
        "epsilon": float(epsilon),
        "interpretation": interp,
    }


def _mixing_time_power(
    adj: np.ndarray,
    T: int,
    epsilon: float,
) -> dict[str, Any]:
    """Direct power iteration of the transition matrix.

    Lazy random walk: P = (I + D^{-1}A) / 2.
    We iterate until ||p_t - uniform||_TV < epsilon.
    """
    deg = adj.sum(axis=1)
    # Transition matrix: D^{-1} A
    P = np.zeros_like(adj)
    nonzero = deg > 0
    P[nonzero] = adj[nonzero] / deg[nonzero, None]
    # Lazy version: (I + P) / 2
    P = (np.eye(T, dtype=np.float64) + P) / 2.0

    uniform = np.full(T, 1.0 / T, dtype=np.float64)
    # Start from worst case: a single vertex
    p = np.zeros(T, dtype=np.float64)
    p[0] = 1.0

    max_steps = min(T * T, 10000)
    t_mix = max_steps

    for step in range(1, max_steps + 1):
        p = p @ P
        tv = 0.5 * np.sum(np.abs(p - uniform))
        if tv < epsilon:
            t_mix = step
            break

    gap_info = spectral_gap(adj, normalized=True)

    return {
        "nodes": T,
        "mixing_time": t_mix,
        "spectral_gap": float(gap_info["spectral_gap"]),
        "method": "power_iteration",
        "epsilon": float(epsilon),
        "interpretation": f"{t_mix} steps (direct measurement)",
    }


# ---------------------------------------------------------------------------
# 4. Effective diameter
# ---------------------------------------------------------------------------


def effective_diameter(
    adj_or_neigh_idx: np.ndarray,
    *,
    num_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate effective diameter via sampled BFS.

    The *diameter* of a graph is the maximum shortest-path distance
    between any two vertices.  Computing it exactly is O(T^2).
    We estimate it by running BFS from ``num_samples`` random
    source vertices and reporting statistics.

    For a good expander on T nodes with degree d, the diameter
    should be O(log T / log d).  A path graph has diameter T-1.

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Adjacency matrix [T, T] or neigh_idx [T, D] / [H, T, D].
    num_samples : int
        Number of BFS source nodes to sample.
    rng : Generator or None
        NumPy random generator for reproducibility.

    Returns
    -------
    dict with keys:
        - nodes: int
        - max_distance: int  (estimated diameter)
        - mean_distance: float
        - median_distance: float
        - p90_distance: float  (90th percentile)
        - num_samples: int
        - is_connected: bool
    """
    adj_list = _ensure_adj_list(adj_or_neigh_idx)
    T = len(adj_list)
    rand = rng if rng is not None else np.random.default_rng()

    if T <= 1:
        return {
            "nodes": T,
            "max_distance": 0,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p90_distance": 0.0,
            "num_samples": 0,
            "is_connected": True,
        }

    n_src = min(num_samples, T)
    sources = rand.choice(T, size=n_src, replace=False)

    all_eccentricities: list[int] = []
    all_distances: list[int] = []
    is_connected = True

    for src in sources:
        dist = _bfs_distances(adj_list, int(src), T)
        reachable = dist[dist < T]
        if len(reachable) < T:
            is_connected = False
        if len(reachable) > 1:
            ecc = int(reachable.max())
            all_eccentricities.append(ecc)
            # Collect all finite distances for distribution stats
            all_distances.extend(reachable[reachable > 0].tolist())

    if not all_eccentricities:
        return {
            "nodes": T,
            "max_distance": 0,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p90_distance": 0.0,
            "num_samples": n_src,
            "is_connected": is_connected,
        }

    dist_arr = np.asarray(all_distances, dtype=np.float64)
    return {
        "nodes": T,
        "max_distance": int(max(all_eccentricities)),
        "mean_distance": float(dist_arr.mean()),
        "median_distance": float(np.median(dist_arr)),
        "p90_distance": float(np.percentile(dist_arr, 90)),
        "num_samples": n_src,
        "is_connected": is_connected,
    }


def _bfs_distances(
    adj_list: list[list[int]],
    src: int,
    T: int,
) -> np.ndarray:
    """BFS from src, return distance array (T = unreachable)."""
    dist = np.full(T, T, dtype=np.int64)
    dist[src] = 0
    queue: deque[int] = deque([src])
    while queue:
        u = queue.popleft()
        d_next = int(dist[u]) + 1
        for v in adj_list[u]:
            if dist[v] > d_next:
                dist[v] = d_next
                queue.append(v)
    return dist


# ---------------------------------------------------------------------------
# 5. Degree statistics
# ---------------------------------------------------------------------------


def degree_stats(
    adj_or_neigh_idx: np.ndarray,
) -> dict[str, Any]:
    """Compute degree statistics for the (undirected) graph.

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Adjacency matrix [T, T] or neigh_idx [T, D] / [H, T, D].

    Returns
    -------
    dict with keys:
        - nodes: int
        - degree_min: int
        - degree_max: int
        - degree_mean: float
        - degree_std: float
        - is_regular: bool  (all degrees equal)
    """
    adj = _ensure_adj(adj_or_neigh_idx)
    T = int(adj.shape[0])
    if T == 0:
        return {
            "nodes": 0,
            "degree_min": 0,
            "degree_max": 0,
            "degree_mean": 0.0,
            "degree_std": 0.0,
            "is_regular": True,
        }

    # Exclude self-loops from degree count for graph-theoretic
    # consistency (self-loops don't contribute to expansion).
    np.fill_diagonal(adj, 0.0)
    deg = adj.sum(axis=1).astype(np.int64)

    return {
        "nodes": T,
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "degree_mean": float(deg.mean()),
        "degree_std": float(deg.std()),
        "is_regular": bool(int(deg.min()) == int(deg.max())),
    }


# ---------------------------------------------------------------------------
# 6. Graph quality report
# ---------------------------------------------------------------------------


def graph_quality_report(
    neigh_idx: np.ndarray,
    *,
    expansion_samples: int = 100,
    diameter_samples: int = 50,
    epsilon: float = 0.01,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """One-call comprehensive graph quality assessment.

    Runs all diagnostics and returns a unified report dict.
    Designed for use in ablation sweeps where you want a single
    function call to characterize each graph configuration.

    Parameters
    ----------
    neigh_idx : ndarray of shape [T, D] or [H, T, D]
        Padded neighbor-index tensor from WayfinderGraphABI.
    expansion_samples : int
        Samples per subset size for expansion_ratio.
    diameter_samples : int
        BFS sources for effective_diameter.
    epsilon : float
        Target TV distance for mixing_time_estimate.
    rng : Generator or None
        Seed for reproducibility.

    Returns
    -------
    dict with sections:
        - degree: {min, max, mean, std, is_regular}
        - spectral: {spectral_gap, lambda_1, lambda_2}
        - expansion: {min_expansion, mean_expansion}
        - mixing: {mixing_time, interpretation}
        - diameter: {max_distance, mean_distance, is_connected}
        - summary: {is_good_expander, quality_score}
    """
    rand = rng if rng is not None else np.random.default_rng()
    ni = np.asarray(neigh_idx)

    deg = degree_stats(ni)
    spec = spectral_gap(ni, normalized=True)
    exp = expansion_ratio(
        ni,
        sample_size=expansion_samples,
        rng=rand,
    )
    mix = mixing_time_estimate(ni, epsilon=epsilon)
    diam = effective_diameter(
        ni,
        num_samples=diameter_samples,
        rng=rand,
    )

    # Composite quality score in [0, 1]:
    #   - 40% spectral gap (clamped to [0, 0.5] range)
    #   - 30% expansion (min_expansion clamped to [0, 5])
    #   - 30% diameter quality (log(T)/diameter, clamped)
    T = deg["nodes"]
    gap_score = min(spec["spectral_gap"] / 0.5, 1.0)

    exp_score = min(exp["min_expansion"] / 5.0, 1.0)

    if T > 1 and diam["max_distance"] > 0:
        ideal_diam = math.log2(max(T, 2))
        diam_score = min(ideal_diam / diam["max_distance"], 1.0)
    else:
        diam_score = 1.0

    quality = 0.4 * gap_score + 0.3 * exp_score + 0.3 * diam_score
    good = spec["spectral_gap"] >= 0.1

    return {
        "nodes": T,
        "degree": {
            "min": deg["degree_min"],
            "max": deg["degree_max"],
            "mean": deg["degree_mean"],
            "std": deg["degree_std"],
            "is_regular": deg["is_regular"],
        },
        "spectral": {
            "spectral_gap": spec["spectral_gap"],
            "lambda_1": spec["lambda_1"],
            "lambda_2": spec["lambda_2"],
        },
        "expansion": {
            "min_expansion": exp["min_expansion"],
            "mean_expansion": exp["mean_expansion"],
            "expansion_by_size": exp["expansion_by_size"],
        },
        "mixing": {
            "mixing_time": mix["mixing_time"],
            "spectral_gap": mix["spectral_gap"],
            "interpretation": mix["interpretation"],
        },
        "diameter": {
            "max_distance": diam["max_distance"],
            "mean_distance": diam["mean_distance"],
            "median_distance": diam["median_distance"],
            "p90_distance": diam["p90_distance"],
            "is_connected": diam["is_connected"],
        },
        "summary": {
            "is_good_expander": good,
            "quality_score": float(quality),
        },
    }


# ---------------------------------------------------------------------------
# 7. Quick boolean check
# ---------------------------------------------------------------------------


def is_good_expander(
    adj_or_neigh_idx: np.ndarray,
    *,
    min_spectral_gap: float = 0.1,
) -> bool:
    """Quick boolean: does the graph meet minimum expander quality?

    Parameters
    ----------
    adj_or_neigh_idx : ndarray
        Adjacency matrix [T, T] or neigh_idx [T, D] / [H, T, D].
    min_spectral_gap : float
        Minimum spectral gap of the normalized adjacency to qualify.
        0.1 is a practical threshold; Ramanujan graphs achieve
        ~1 - 2*sqrt(d-1)/d which for d=6 is ~0.25.

    Returns
    -------
    bool
    """
    info = spectral_gap(adj_or_neigh_idx, normalized=True)
    return bool(info["spectral_gap"] >= min_spectral_gap)


# ---------------------------------------------------------------------------
# 8. Causal (directed) reachability analysis
# ---------------------------------------------------------------------------
#
# Undirected metrics (spectral gap, Fiedler, diameter) treat edges as
# bidirectional.  But causal attention is *directed*: token i can only
# attend to tokens j <= i.  So the relevant graph for information flow
# is the DAG induced by keeping only edges where j <= i.
#
# This section provides directed counterparts that answer the question
# "how many layers of causal sparse attention does token i need to
# aggregate information from all earlier positions?"


def _neigh_idx_to_causal_predecessors(
    neigh_idx: np.ndarray,
) -> list[list[int]]:
    """Build causal predecessor lists from neigh_idx.

    For each token i, returns the list of tokens j < i that can
    directly send information to i (i.e., j appears in neigh_idx[i]
    and j < i).  Self-loops (j == i) are excluded since they carry
    no new information from the past.

    Multi-head inputs [H, T, D] are unioned across heads.

    Returns
    -------
    preds : list[list[int]]
        preds[i] is a sorted list of token indices j < i that i
        directly attends to.  preds[0] is always [].
    """
    ni = np.asarray(neigh_idx)
    if ni.ndim == 3:
        # Multi-head: union across heads
        T = int(ni.shape[1])
        pred_sets: list[set[int]] = [set() for _ in range(T)]
        for h in range(ni.shape[0]):
            for i in range(T):
                row = ni[h, i]
                for j in row[row >= 0].tolist():
                    j = int(j)
                    if j < i:
                        pred_sets[i].add(j)
        return [sorted(s) for s in pred_sets]
    if ni.ndim != 2:
        raise ValueError(f"neigh_idx must be [T, D] or [H, T, D], got {ni.shape}")
    T = int(ni.shape[0])
    preds: list[list[int]] = [[] for _ in range(T)]
    for i in range(T):
        row = ni[i]
        valid = row[row >= 0]
        causal = valid[valid < i]
        if causal.size > 0:
            preds[i] = sorted(set(causal.tolist()))
    return preds


def causal_reachability(
    neigh_idx: np.ndarray,
    *,
    max_layers: int = 16,
    sample_stride: int | None = None,
) -> dict[str, Any]:
    """Compute backward reachability under causal masking.

    For each token position i, this measures how many of {0, ..., i-1}
    can propagate information to i within L layers of sparse causal
    attention.  This is the *transitive closure* of the causal DAG,
    layer by layer.

    Why this matters: undirected diameter might be 5, but in the
    causal direction the effective diameter can be much larger because
    "backward" edges (from later to earlier tokens) are forbidden.
    This function reveals the *actual* information propagation depth.

    Parameters
    ----------
    neigh_idx : ndarray of shape [T, D] or [H, T, D]
        Padded neighbor-index tensor.
    max_layers : int
        Simulate up to this many layers of attention.
    sample_stride : int or None
        If T > 1000, only compute coverage for every sample_stride-th
        position to stay tractable.  None = auto (every 10th for T>1000).

    Returns
    -------
    dict with keys:
        - coverage_by_layer : ndarray [max_layers, T_sampled]
            Fraction of {0..i-1} reachable in l+1 hops (l=0..max_layers-1).
        - mean_coverage : ndarray [max_layers]
            Average coverage across sampled positions at each layer.
        - p95_layers_to_full : int
            Layers until 95% of sampled positions have >= 95% coverage.
            Returns max_layers + 1 if never reached.
        - worst_position : ndarray [max_layers]
            Position index with worst coverage at each layer.
        - sampled_positions : ndarray
            Which positions were actually evaluated.
        - nodes : int
    """
    preds = _neigh_idx_to_causal_predecessors(neigh_idx)
    T = len(preds)

    if T <= 1:
        return {
            "coverage_by_layer": np.zeros((max_layers, T), dtype=np.float64),
            "mean_coverage": np.zeros(max_layers, dtype=np.float64),
            "p95_layers_to_full": 0,
            "worst_position": np.zeros(max_layers, dtype=np.int64),
            "sampled_positions": np.arange(T, dtype=np.int64),
            "nodes": T,
        }

    # Decide which positions to sample
    if sample_stride is None:
        sample_stride = max(1, T // 100) if T > 1000 else 1
    # Always include position 0 and sample the rest
    positions = np.arange(0, T, sample_stride, dtype=np.int64)
    # Always include the last position
    if positions[-1] != T - 1:
        positions = np.append(positions, T - 1)
    n_pos = len(positions)

    coverage = np.zeros((max_layers, n_pos), dtype=np.float64)

    for p_idx, i in enumerate(positions):
        i = int(i)
        if i == 0:
            # No predecessors possible
            continue

        # BFS-like layer expansion: reachable_l is the set of tokens
        # that can reach i within l layers.
        # Layer 0 (no attention applied): only i itself
        # Layer 1: direct causal neighbors of i
        # Layer l: union of causal neighbors of everything reachable at l-1

        # We track it backwards: "who can reach i?"
        # Equivalently: starting from i, follow predecessor edges.
        # After 1 layer: preds[i].
        # After 2 layers: preds[i] union preds[j] for j in preds[i].
        # This is BFS on the *reversed* causal DAG from i.

        reachable: set[int] = set()
        frontier: set[int] = {i}
        num_predecessors = i  # tokens 0..i-1

        for layer in range(max_layers):
            # Expand frontier by one layer of predecessor edges
            next_frontier: set[int] = set()
            for node in frontier:
                for pred in preds[node]:
                    if pred not in reachable:
                        reachable.add(pred)
                        next_frontier.add(pred)
            frontier = next_frontier
            coverage[layer, p_idx] = len(reachable) / num_predecessors
            if len(reachable) == num_predecessors:
                # Full coverage reached; fill remaining layers
                for ll in range(layer + 1, max_layers):
                    coverage[ll, p_idx] = 1.0
                break

    # Compute summary statistics
    # Exclude position 0 from mean (it has no predecessors, coverage is N/A)
    valid_mask = positions > 0
    if valid_mask.any():
        mean_cov = coverage[:, valid_mask].mean(axis=1)
    else:
        mean_cov = np.zeros(max_layers, dtype=np.float64)

    # p95_layers_to_full: first layer L where >= 95% of valid positions
    # have coverage >= 0.95
    p95_layers = max_layers + 1
    for layer in range(max_layers):
        if not valid_mask.any():
            p95_layers = 0
            break
        valid_cov = coverage[layer, valid_mask]
        frac_above_95 = (valid_cov >= 0.95).mean()
        if frac_above_95 >= 0.95:
            p95_layers = layer + 1  # 1-indexed layer count
            break

    # Worst position at each layer (among valid positions)
    worst_pos = np.zeros(max_layers, dtype=np.int64)
    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        for layer in range(max_layers):
            worst_idx = valid_indices[np.argmin(coverage[layer, valid_indices])]
            worst_pos[layer] = positions[worst_idx]

    return {
        "coverage_by_layer": coverage,
        "mean_coverage": mean_cov,
        "p95_layers_to_full": int(p95_layers),
        "worst_position": worst_pos,
        "sampled_positions": positions,
        "nodes": T,
    }


# ---------------------------------------------------------------------------
# 9. Causal diameter
# ---------------------------------------------------------------------------


def causal_diameter(
    neigh_idx: np.ndarray,
    *,
    num_samples: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate directed diameter of the causal attention graph.

    The causal diameter is max_{j < i} d(j -> i), where d(j -> i) is
    the shortest directed path from j to i through the causal DAG
    (only edges where source < target are traversed, simulating
    information flow through attention layers).

    Unlike undirected diameter, this can be much larger because
    "backward" edges are unavailable.  The causal diameter directly
    bounds the minimum model depth for full information propagation.

    Parameters
    ----------
    neigh_idx : ndarray [T, D] or [H, T, D]
        Padded neighbor-index tensor.
    num_samples : int
        Number of source nodes to BFS from.  The BFS follows
        *successor* edges (j -> i where j < i and j in neigh_idx[i]).
    rng : Generator or None
        For reproducible sampling.

    Returns
    -------
    dict with keys:
        - nodes : int
        - causal_diameter : int  (estimated max directed distance)
        - mean_distance : float  (average finite directed distance)
        - median_distance : float
        - p90_distance : float
        - unreachable_frac : float  (fraction of (j, i) pairs with
          j < i that are mutually unreachable in the causal DAG)
        - num_samples : int
    """
    preds = _neigh_idx_to_causal_predecessors(neigh_idx)
    T = len(preds)
    rand = rng if rng is not None else np.random.default_rng()

    if T <= 1:
        return {
            "nodes": T,
            "causal_diameter": 0,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p90_distance": 0.0,
            "unreachable_frac": 0.0,
            "num_samples": 0,
        }

    # Build successor lists for forward BFS:
    # succs[j] = list of tokens i > j such that j in preds[i]
    # (i.e., i attends to j, so information flows j -> i)
    succs: list[list[int]] = [[] for _ in range(T)]
    for i in range(T):
        for j in preds[i]:
            succs[j].append(i)

    n_src = min(num_samples, T)
    sources = rand.choice(T, size=n_src, replace=False)

    all_distances: list[int] = []
    max_dist = 0
    total_pairs = 0
    unreachable_count = 0

    for src in sources:
        src = int(src)
        # BFS forward from src along successor edges
        dist = np.full(T, -1, dtype=np.int64)
        dist[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            u = queue.popleft()
            d_next = int(dist[u]) + 1
            for v in succs[u]:
                if dist[v] < 0:
                    dist[v] = d_next
                    queue.append(v)

        # Only consider tokens i > src (causal direction)
        for i in range(src + 1, T):
            total_pairs += 1
            if dist[i] < 0:
                unreachable_count += 1
            else:
                d = int(dist[i])
                all_distances.append(d)
                if d > max_dist:
                    max_dist = d

    if not all_distances:
        return {
            "nodes": T,
            "causal_diameter": 0,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p90_distance": 0.0,
            "unreachable_frac": (unreachable_count / total_pairs if total_pairs > 0 else 0.0),
            "num_samples": n_src,
        }

    dist_arr = np.asarray(all_distances, dtype=np.float64)
    return {
        "nodes": T,
        "causal_diameter": int(max_dist),
        "mean_distance": float(dist_arr.mean()),
        "median_distance": float(np.median(dist_arr)),
        "p90_distance": float(np.percentile(dist_arr, 90)),
        "unreachable_frac": (unreachable_count / total_pairs if total_pairs > 0 else 0.0),
        "num_samples": n_src,
    }


# ---------------------------------------------------------------------------
# 10. Causal vs undirected comparison
# ---------------------------------------------------------------------------


def causal_mixing_comparison(
    neigh_idx: np.ndarray,
    *,
    max_layers: int = 16,
    diameter_samples: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compare directed (causal) vs undirected graph metrics.

    This is the key diagnostic for answering "how much does causality
    hurt?"  Undirected metrics are optimistic because they allow
    information to flow backward in time.  This function puts both
    views side by side.

    Parameters
    ----------
    neigh_idx : ndarray [T, D] or [H, T, D]
    max_layers : int
        Max layers for causal_reachability.
    diameter_samples : int
        BFS samples for both directed and undirected diameter.
    rng : Generator or None

    Returns
    -------
    dict with sections:
        - undirected : {spectral_gap, fiedler, diameter}
        - causal : {diameter, mean_coverage_curve, p95_layers_to_full}
        - gap : {diameter_ratio, layers_needed_vs_mixing_time}
    """
    rand = rng if rng is not None else np.random.default_rng()
    ni = np.asarray(neigh_idx)

    # --- Undirected metrics ---
    spec = spectral_gap(ni, normalized=True)
    diam_und = effective_diameter(ni, num_samples=diameter_samples, rng=rand)
    mix = mixing_time_estimate(ni, method="spectral")

    # --- Causal metrics ---
    reach = causal_reachability(ni, max_layers=max_layers)
    diam_causal = causal_diameter(ni, num_samples=diameter_samples, rng=rand)

    # --- Gap analysis ---
    und_diam = diam_und["max_distance"]
    c_diam = diam_causal["causal_diameter"]
    if und_diam > 0:
        diameter_ratio = c_diam / und_diam
    else:
        diameter_ratio = float("inf") if c_diam > 0 else 1.0

    mixing_t = mix["mixing_time"]
    p95 = reach["p95_layers_to_full"]
    if mixing_t > 0:
        layers_vs_mixing = p95 / mixing_t
    else:
        layers_vs_mixing = float("inf") if p95 > 0 else 1.0

    return {
        "undirected": {
            "spectral_gap": spec["spectral_gap"],
            "diameter": und_diam,
            "mixing_time": mixing_t,
            "is_connected": diam_und["is_connected"],
        },
        "causal": {
            "diameter": c_diam,
            "mean_coverage_curve": reach["mean_coverage"].tolist(),
            "p95_layers_to_full": p95,
            "unreachable_frac": diam_causal["unreachable_frac"],
        },
        "gap": {
            "diameter_ratio": float(diameter_ratio),
            "layers_needed_vs_mixing_time": float(layers_vs_mixing),
            "interpretation": _causality_interpretation(diameter_ratio, p95, max_layers),
        },
        "nodes": reach["nodes"],
    }


def _causality_interpretation(
    diameter_ratio: float,
    p95_layers: int,
    max_layers: int,
) -> str:
    """Human-readable interpretation of causality gap."""
    if p95_layers <= 4:
        speed = "excellent"
    elif p95_layers <= 8:
        speed = "good"
    elif p95_layers <= max_layers:
        speed = "moderate"
    else:
        speed = "poor (did not converge)"

    if diameter_ratio <= 2.0:
        penalty = "minimal causality penalty"
    elif diameter_ratio <= 5.0:
        penalty = "moderate causality penalty"
    else:
        penalty = "severe causality penalty"

    return f"causal mixing: {speed}; {penalty} ({diameter_ratio:.1f}x diameter)"


# ---------------------------------------------------------------------------
# 11. Causal-aware graph quality report
# ---------------------------------------------------------------------------


def graph_quality_report_causal(
    neigh_idx: np.ndarray,
    *,
    max_layers: int = 12,
    expansion_samples: int = 100,
    diameter_samples: int = 50,
    epsilon: float = 0.01,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Comprehensive graph quality report including causal analysis.

    Extends ``graph_quality_report()`` with a ``causal`` section that
    captures directed reachability and causal diameter.  This is the
    recommended "one call" function for evaluating sparse attention
    graphs under realistic (causal) constraints.

    Parameters
    ----------
    neigh_idx : ndarray [T, D] or [H, T, D]
    max_layers : int
        Max layers for causal_reachability.
    expansion_samples : int
        Samples per subset size for expansion_ratio.
    diameter_samples : int
        BFS sources for diameter estimates.
    epsilon : float
        Target TV distance for mixing_time_estimate.
    rng : Generator or None

    Returns
    -------
    dict with sections:
        - degree, spectral, expansion, mixing, diameter, summary
          (same as graph_quality_report)
        - causal : {reachability, diameter, comparison}
    """
    rand = rng if rng is not None else np.random.default_rng()
    ni = np.asarray(neigh_idx)

    # Base undirected report
    base = graph_quality_report(
        ni,
        expansion_samples=expansion_samples,
        diameter_samples=diameter_samples,
        epsilon=epsilon,
        rng=rand,
    )

    # Causal additions
    reach = causal_reachability(ni, max_layers=max_layers)
    c_diam = causal_diameter(ni, num_samples=diameter_samples, rng=rand)

    # Causality gap
    und_diam = base["diameter"]["max_distance"]
    cd = c_diam["causal_diameter"]
    if und_diam > 0:
        diameter_ratio = cd / und_diam
    else:
        diameter_ratio = float("inf") if cd > 0 else 1.0

    base["causal"] = {
        "reachability": {
            "mean_coverage_curve": reach["mean_coverage"].tolist(),
            "p95_layers_to_full": reach["p95_layers_to_full"],
        },
        "diameter": {
            "causal_diameter": cd,
            "mean_distance": c_diam["mean_distance"],
            "unreachable_frac": c_diam["unreachable_frac"],
        },
        "comparison": {
            "diameter_ratio": float(diameter_ratio),
            "interpretation": _causality_interpretation(
                diameter_ratio,
                reach["p95_layers_to_full"],
                max_layers,
            ),
        },
    }

    return base
