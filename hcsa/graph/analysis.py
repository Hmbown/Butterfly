from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng()


def _validate_perm(cycle_perm: np.ndarray) -> np.ndarray:
    perm = np.asarray(cycle_perm, dtype=np.int64).reshape(-1)
    t = int(perm.shape[0])
    if t <= 0:
        raise ValueError("cycle_perm must be non-empty")
    if int(perm.min()) < 0 or int(perm.max()) >= t:
        raise ValueError("cycle_perm values must be in [0, T)")
    if np.unique(perm).shape[0] != t:
        raise ValueError("cycle_perm must be a permutation")
    return perm


def _build_undirected_adj_list(cycle_perm: np.ndarray, *, window: int = 0) -> list[set[int]]:
    perm = _validate_perm(cycle_perm)
    t = int(perm.shape[0])
    adj: list[set[int]] = [set() for _ in range(t)]

    for i in range(t):
        u = int(perm[i])
        v = int(perm[(i + 1) % t])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    w = int(max(0, window))
    if w > 0:
        for i in range(t):
            lo = max(0, i - w)
            for j in range(lo, i):
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)

    return adj


def _build_undirected_adj_matrix(cycle_perm: np.ndarray, *, window: int = 0) -> np.ndarray:
    adj = _build_undirected_adj_list(cycle_perm, window=window)
    t = len(adj)
    a = np.zeros((t, t), dtype=np.float64)
    for u in range(t):
        if not adj[u]:
            continue
        idx = np.fromiter(adj[u], dtype=np.int64)
        a[u, idx] = 1.0
    return a


def _largest_component_size(adj: list[set[int]]) -> int:
    t = len(adj)
    if t == 0:
        return 0
    seen = [False] * t
    best = 0
    for s in range(t):
        if seen[s]:
            continue
        q: deque[int] = deque([s])
        seen[s] = True
        cnt = 0
        while q:
            u = q.popleft()
            cnt += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        best = max(best, cnt)
    return best


def spectral_gap(
    cycle_perm: np.ndarray,
    *,
    include_window: bool = False,
    window: int = 0,
    expander_threshold: float = 4.0,
) -> dict[str, Any]:
    """Compute spectral diagnostics for cycle(+window) graph.

    Returns degree/lambda statistics and d/lambda expansion ratio.
    """
    a = _build_undirected_adj_matrix(
        cycle_perm,
        window=(window if include_window else 0),
    )
    t = int(a.shape[0])
    deg = a.sum(axis=1)
    degree_mean = float(deg.mean()) if t > 0 else 0.0
    degree_min = float(deg.min()) if t > 0 else 0.0
    degree_max = float(deg.max()) if t > 0 else 0.0

    if t == 1:
        lambda_1 = 0.0
        lambda_2_abs = 0.0
        eig_method = "degenerate"
    elif t <= 4096:
        eigvals = np.linalg.eigvalsh(a)
        top_idx = int(np.argmax(eigvals))
        lambda_1 = float(eigvals[top_idx])
        rem = np.delete(eigvals, top_idx)
        lambda_2_abs = float(np.max(np.abs(rem))) if rem.size else 0.0
        eig_method = "numpy.linalg.eigvalsh"
    else:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            sparse_a = sp.csr_matrix(a)
            vals, _vecs = spla.eigsh(sparse_a, k=2, which="LA")
            vals = np.asarray(vals, dtype=np.float64)
            lambda_1 = float(np.max(vals))
            if vals.size == 1:
                lambda_2_abs = 0.0
            else:
                rem = vals[vals != lambda_1]
                if rem.size == 0:
                    rem = vals
                lambda_2_abs = float(np.max(np.abs(rem)))
            eig_method = "scipy.sparse.linalg.eigsh"
        except Exception:
            eigvals = np.linalg.eigvalsh(a)
            top_idx = int(np.argmax(eigvals))
            lambda_1 = float(eigvals[top_idx])
            rem = np.delete(eigvals, top_idx)
            lambda_2_abs = float(np.max(np.abs(rem))) if rem.size else 0.0
            eig_method = "numpy.linalg.eigvalsh_fallback"

    gap = float(lambda_1 - lambda_2_abs)
    ratio = float(degree_mean / max(lambda_2_abs, 1e-12))
    return {
        "nodes": t,
        "include_window": bool(include_window),
        "window": int(window if include_window else 0),
        "degree": degree_mean,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2_abs,
        "spectral_gap": gap,
        "expansion_ratio": ratio,
        "is_good_expander": bool(ratio >= float(expander_threshold)),
        "expander_threshold": float(expander_threshold),
        "eigen_method": eig_method,
    }


def expansion_proxy(
    cycle_perm: np.ndarray,
    *,
    window: int = 0,
    num_walks: int = 1000,
    walk_len: int = 20,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate expansion quality via random-walk mixing behavior."""
    perm = _validate_perm(cycle_perm)
    t = int(perm.shape[0])
    if t == 1:
        return {
            "nodes": 1,
            "window": int(max(0, window)),
            "num_walks": int(max(1, num_walks)),
            "walk_len": int(max(1, walk_len)),
            "mixing_time_estimate": 0,
            "endpoint_uniformity": 0.0,
            "is_fast_mixer": True,
            "mixing_threshold_steps": 0.0,
            "chi2_threshold": 0.0,
        }

    adj = _build_undirected_adj_list(perm, window=int(max(0, window)))
    rand = _as_rng(rng)
    nwalk = int(max(1, num_walks))
    wlen = int(max(1, walk_len))
    max_steps = max(wlen, int(math.ceil(4.0 * math.log2(max(2, t)))))

    # Start from a single source to probe convergence to stationarity.
    start_vertex = int(rand.integers(0, t, endpoint=False))
    endpoints = np.full((nwalk,), start_vertex, dtype=np.int64)
    expected = float(nwalk) / float(t)
    # Good expanders should approach the random-sampling chi^2 regime quickly.
    # For multinomial samples, E[chi^2] ~= (T-1), so 1.25x is a practical bound.
    chi2_threshold = float(1.25 * max(1, t - 1))

    mixing_time_estimate = max_steps + 1
    endpoint_uniformity = float("inf")

    for step in range(1, max_steps + 1):
        next_ep = endpoints.copy()
        for idx, u in enumerate(endpoints.tolist()):
            nbrs = adj[u]
            if not nbrs:
                continue
            arr = np.fromiter(nbrs, dtype=np.int64)
            choice = int(rand.integers(0, arr.shape[0], endpoint=False))
            next_ep[idx] = int(arr[choice])
        endpoints = next_ep

        counts = np.bincount(endpoints, minlength=t).astype(np.float64)
        chi2 = float(np.sum(((counts - expected) ** 2) / max(expected, 1e-12)))
        if step == wlen:
            endpoint_uniformity = chi2
        if chi2 <= chi2_threshold and mixing_time_estimate > max_steps:
            mixing_time_estimate = step

    if not np.isfinite(endpoint_uniformity):
        counts = np.bincount(endpoints, minlength=t).astype(np.float64)
        endpoint_uniformity = float(np.sum(((counts - expected) ** 2) / max(expected, 1e-12)))

    fast_thresh = float(2.0 * math.log2(max(2, t)))
    return {
        "nodes": int(t),
        "window": int(max(0, window)),
        "num_walks": nwalk,
        "walk_len": wlen,
        "mixing_time_estimate": int(mixing_time_estimate),
        "endpoint_uniformity": float(endpoint_uniformity),
        "is_fast_mixer": bool(mixing_time_estimate <= fast_thresh),
        "mixing_threshold_steps": fast_thresh,
        "chi2_threshold": chi2_threshold,
        "largest_component": int(_largest_component_size(adj)),
    }


def check_resilience(
    cycle_perm: np.ndarray,
    window: int,
    drop_rate: float,
    *,
    num_trials: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Empirically test resilience of cycle+window graph under random edge drop."""
    if not (0.0 <= float(drop_rate) <= 1.0):
        raise ValueError("drop_rate must be in [0, 1]")
    trials = int(max(1, num_trials))
    rand = _as_rng(rng)

    adj_full = _build_undirected_adj_list(cycle_perm, window=int(max(0, window)))
    t = len(adj_full)
    base_deg = np.asarray([len(nbrs) for nbrs in adj_full], dtype=np.float64)
    base_min = float(base_deg.min()) if base_deg.size else 0.0
    base_mean = float(base_deg.mean()) if base_deg.size else 0.0

    # Theorem-inspired bound: preserve at least ~half of baseline local degree.
    theoretical_threshold = float(0.5 * base_min)

    edges: list[tuple[int, int]] = []
    for u in range(t):
        for v in adj_full[u]:
            if u < v:
                edges.append((u, v))

    survived = 0
    connected_count = 0
    min_degrees: list[int] = []

    for _trial in range(trials):
        adj_trial: list[set[int]] = [set() for _ in range(t)]
        keep = rand.random(len(edges)) >= float(drop_rate)
        for idx, (u, v) in enumerate(edges):
            if not bool(keep[idx]):
                continue
            adj_trial[u].add(v)
            adj_trial[v].add(u)

        deg = [len(nbrs) for nbrs in adj_trial]
        min_deg = int(min(deg)) if deg else 0
        min_degrees.append(min_deg)

        is_connected = _largest_component_size(adj_trial) == t
        if is_connected:
            connected_count += 1

        strong_dirac = min_deg >= (t // 2)
        theorem_like = float(min_deg) >= theoretical_threshold
        if is_connected and (theorem_like or strong_dirac):
            survived += 1

    min_deg_arr = np.asarray(min_degrees, dtype=np.float64)
    return {
        "nodes": int(t),
        "window": int(max(0, window)),
        "drop_rate": float(drop_rate),
        "num_trials": trials,
        "survival_rate": float(survived / trials),
        "connected_rate": float(connected_count / trials),
        "min_degree_mean": float(min_deg_arr.mean()) if min_deg_arr.size else 0.0,
        "min_degree_min": int(min_deg_arr.min()) if min_deg_arr.size else 0,
        "base_degree_mean": base_mean,
        "base_degree_min": base_min,
        "theoretical_threshold": theoretical_threshold,
        "dirac_threshold": int(t // 2),
    }


def check_regularity(
    cycle_perm: np.ndarray,
    num_clusters: int = 8,
    *,
    epsilon: float = 0.25,
) -> dict[str, Any]:
    """Estimate epsilon-regularity of cycle-edge distribution across clusters."""
    perm = _validate_perm(cycle_perm)
    t = int(perm.shape[0])
    k = int(max(1, min(num_clusters, t)))
    clusters = [np.asarray(c, dtype=np.int64) for c in np.array_split(np.arange(t), k)]
    cluster_of = np.empty((t,), dtype=np.int64)
    for cid, nodes in enumerate(clusters):
        cluster_of[nodes] = int(cid)
    sizes = np.asarray([len(c) for c in clusters], dtype=np.int64)

    counts = np.zeros((k, k), dtype=np.float64)
    for i in range(t):
        u = int(perm[i])
        v = int(perm[(i + 1) % t])
        if u == v:
            continue
        cu = int(cluster_of[u])
        cv = int(cluster_of[v])
        if cu == cv:
            continue
        counts[cu, cv] += 1.0
        counts[cv, cu] += 1.0

    possible = np.zeros((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            if a != b:
                possible[a, b] = float(sizes[a] * sizes[b])

    density = np.divide(counts, np.maximum(possible, 1e-12))
    inter_mask = np.ones((k, k), dtype=bool)
    np.fill_diagonal(inter_mask, False)
    total_possible = float(possible[inter_mask].sum() / 2.0)
    total_edges = float(counts[inter_mask].sum() / 2.0)
    global_density = total_edges / max(total_possible, 1e-12)
    expected = possible * global_density

    dev = np.zeros_like(expected)
    valid = inter_mask & (expected > 0)
    dev[valid] = np.abs(counts[valid] - expected[valid]) / expected[valid]

    pair_devs: list[float] = []
    for a in range(k):
        for b in range(a + 1, k):
            if valid[a, b]:
                pair_devs.append(float(dev[a, b]))
    row_means: list[float] = []
    for a in range(k):
        vals = [float(dev[a, b]) for b in range(k) if b != a and valid[a, b]]
        row_means.append(float(np.mean(vals)) if vals else 0.0)

    max_dev = float(max(row_means)) if row_means else 0.0
    mean_dev = float(np.mean(pair_devs)) if pair_devs else 0.0

    return {
        "nodes": int(t),
        "num_clusters": int(k),
        "cluster_sizes": [int(x) for x in sizes.tolist()],
        "global_density": float(global_density),
        "max_deviation": max_dev,
        "mean_deviation": mean_dev,
        "is_epsilon_regular": bool(max_dev < float(epsilon)),
        "epsilon": float(epsilon),
        "cluster_pair_densities": density.tolist(),
    }


def laplacian_spectral_gap(
    cycle_perm: np.ndarray,
    *,
    include_window: bool = False,
    window: int = 0,
) -> dict[str, Any]:
    """Compute Laplacian spectral gap (algebraic connectivity / Fiedler value).

    The Laplacian L = D - A where D is the degree matrix.
    The second-smallest eigenvalue lambda_2(L) (Fiedler value) bounds the
    Cheeger constant: lambda_2/2 <= h(G) <= sqrt(2 * lambda_2).

    Returns dict with:
    - fiedler_value: lambda_2(L), the algebraic connectivity
    - cheeger_lower: lambda_2 / 2
    - cheeger_upper: sqrt(2 * lambda_2)
    - fiedler_vector: eigenvector for lambda_2
    - is_well_connected: fiedler_value > 0.01
    """
    a = _build_undirected_adj_matrix(
        cycle_perm,
        window=(window if include_window else 0),
    )
    t = int(a.shape[0])

    if t <= 1:
        return {
            "fiedler_value": 0.0,
            "cheeger_lower": 0.0,
            "cheeger_upper": 0.0,
            "fiedler_vector": np.zeros(t, dtype=np.float64),
            "is_well_connected": False,
        }

    deg = a.sum(axis=1)
    d_mat = np.diag(deg)
    lap = d_mat - a

    eigvals, eigvecs = np.linalg.eigh(lap)
    # eigh returns sorted ascending; lambda_2 is index 1
    lambda_2 = float(max(eigvals[1], 0.0))
    fiedler_vec = eigvecs[:, 1].copy()

    cheeger_lower = lambda_2 / 2.0
    cheeger_upper = math.sqrt(2.0 * lambda_2)

    return {
        "fiedler_value": lambda_2,
        "cheeger_lower": cheeger_lower,
        "cheeger_upper": cheeger_upper,
        "fiedler_vector": fiedler_vec,
        "is_well_connected": bool(lambda_2 > 0.01),
    }


def fiedler_bridge_candidates(
    cycle_perm: np.ndarray,
    *,
    window: int = 0,
    num_bridges: int = 10,
) -> list[tuple[int, int]]:
    """Identify best bridge edges to add for connectivity improvement.

    Uses the Fiedler vector to find the graph bottleneck, then proposes
    edges between the most positive and most negative Fiedler-vector
    vertices.
    """
    result = laplacian_spectral_gap(
        cycle_perm, include_window=(window > 0), window=window
    )
    fv = result["fiedler_vector"]
    t = len(fv)
    if t <= 2:
        return []

    order = np.argsort(fv)
    neg_verts = order[:num_bridges].tolist()
    pos_verts = order[-num_bridges:][::-1].tolist()

    existing = set()
    adj = _build_undirected_adj_list(cycle_perm, window=window)
    for u in range(t):
        for v in adj[u]:
            if u < v:
                existing.add((u, v))

    bridges: list[tuple[int, int]] = []
    for ni, pi in zip(neg_verts, pos_verts):
        a, b = (min(ni, pi), max(ni, pi))
        if a != b and (a, b) not in existing:
            bridges.append((a, b))
        if len(bridges) >= num_bridges:
            break

    return bridges


def compute_edge_coverage(
    cycles: list[np.ndarray],
    T: int,
    *,
    causal_only: bool = True,
) -> dict[str, Any]:
    """Compute union edge coverage induced by cycle list."""
    t = int(T)
    if t <= 1:
        return {
            "total_possible_edges": 0,
            "covered_edges": 0,
            "coverage_fraction": 1.0,
            "edges_per_cycle": 0,
            "theoretical_min_cycles": 0,
            "causal_only": bool(causal_only),
        }

    covered: set[tuple[int, int]] = set()
    for perm in cycles:
        p = _validate_perm(np.asarray(perm, dtype=np.int64))
        if int(p.shape[0]) != t:
            raise ValueError(f"cycle length {p.shape[0]} does not match T={t}")
        for i in range(t):
            u = int(p[i])
            v = int(p[(i + 1) % t])
            if u == v:
                continue
            a, b = (u, v) if u > v else (v, u)
            if causal_only and a <= b:
                continue
            covered.add((a, b))

    total_possible = int(t * (t - 1) // 2) if causal_only else int(t * (t - 1) // 2)
    covered_n = int(len(covered))
    frac = float(covered_n) / float(max(total_possible, 1))
    edges_per_cycle = int(t)
    theo_min = int(math.ceil(total_possible / max(edges_per_cycle, 1)))

    return {
        "total_possible_edges": total_possible,
        "covered_edges": covered_n,
        "coverage_fraction": frac,
        "edges_per_cycle": edges_per_cycle,
        "theoretical_min_cycles": theo_min,
        "causal_only": bool(causal_only),
    }
