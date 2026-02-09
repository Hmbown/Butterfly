"""Tests for hcsa.graph.expander — expander quality analysis."""

from __future__ import annotations

import numpy as np

from hcsa.graph.expander import (
    causal_diameter,
    causal_reachability,
    degree_stats,
    effective_diameter,
    expansion_ratio,
    graph_quality_report,
    graph_quality_report_causal,
    is_good_expander,
    mixing_time_estimate,
    neigh_idx_to_adj,
    spectral_gap,
)


def _complete_neigh_idx(T: int) -> np.ndarray:
    """neigh_idx for a complete graph (every node connects to all others)."""
    ni = np.full((T, T - 1), -1, dtype=np.int32)
    for i in range(T):
        others = [j for j in range(T) if j != i]
        ni[i, : len(others)] = others
    return ni


def _path_neigh_idx(T: int) -> np.ndarray:
    """neigh_idx for a path graph 0-1-2-...(T-1)."""
    ni = np.full((T, 2), -1, dtype=np.int32)
    for i in range(T):
        col = 0
        if i > 0:
            ni[i, col] = i - 1
            col += 1
        if i < T - 1:
            ni[i, col] = i + 1
    return ni


def _cycle_plus_window(T: int, window: int = 4) -> np.ndarray:
    """Random cycle + local window neigh_idx."""
    rng = np.random.default_rng(42)
    perm = rng.permutation(T)
    # build neighbor sets
    adj: list[set[int]] = [set() for _ in range(T)]
    for idx in range(T):
        u = int(perm[idx])
        v = int(perm[(idx + 1) % T])
        adj[u].add(v)
        adj[v].add(u)
    for i in range(T):
        for j in range(max(0, i - window), i):
            adj[i].add(j)
            adj[j].add(i)
    # convert to neigh_idx
    max_d = max(len(s) for s in adj)
    ni = np.full((T, max_d), -1, dtype=np.int32)
    for i, s in enumerate(adj):
        nbrs = sorted(s - {i})
        ni[i, : len(nbrs)] = nbrs
    return ni


# ------------------------------------------------------------------
# neigh_idx_to_adj
# ------------------------------------------------------------------

class TestNeighIdxToAdj:
    def test_complete_graph(self):
        ni = _complete_neigh_idx(5)
        A = neigh_idx_to_adj(ni)
        assert A.shape == (5, 5)
        assert np.diag(A).sum() == 0  # no self-loops
        assert A.sum() == 5 * 4  # each of 5 nodes has 4 edges

    def test_path_graph(self):
        A = neigh_idx_to_adj(_path_neigh_idx(6))
        assert A[0, 1] == 1
        assert A[5, 4] == 1
        assert A[0, 5] == 0

    def test_symmetric(self):
        ni = _cycle_plus_window(32)
        A = neigh_idx_to_adj(ni)
        np.testing.assert_array_equal(A, A.T)


# ------------------------------------------------------------------
# spectral_gap
# ------------------------------------------------------------------

class TestSpectralGap:
    def test_complete_graph_high(self):
        ni = _complete_neigh_idx(8)
        gap = spectral_gap(ni)["spectral_gap"]
        assert gap > 0.5  # complete graph has excellent gap

    def test_path_graph_low(self):
        gap = spectral_gap(_path_neigh_idx(32))["spectral_gap"]
        assert gap < 0.3  # path graph has poor expansion

    def test_single_node(self):
        ni = np.full((1, 1), -1, dtype=np.int32)
        assert spectral_gap(ni)["spectral_gap"] == 0.0


# ------------------------------------------------------------------
# expansion_ratio
# ------------------------------------------------------------------

class TestExpansionRatio:
    def test_complete_graph(self):
        ni = _complete_neigh_idx(8)
        er = expansion_ratio(ni, sample_size=50, rng=np.random.default_rng(0))
        assert er["min_expansion"] > 0

    def test_path_low(self):
        er = expansion_ratio(
            _path_neigh_idx(32), sample_size=50,
            rng=np.random.default_rng(0),
        )
        # path has low expansion for larger subsets
        assert er["mean_expansion"] < 5.0


# ------------------------------------------------------------------
# mixing_time_estimate
# ------------------------------------------------------------------

class TestMixingTime:
    def test_complete_fast(self):
        ni = _complete_neigh_idx(8)
        m = mixing_time_estimate(ni)
        assert m["mixing_time"] <= 8  # spectral bound is conservative

    def test_path_slow(self):
        m = mixing_time_estimate(_path_neigh_idx(64))
        assert m["mixing_time"] > 4


# ------------------------------------------------------------------
# effective_diameter
# ------------------------------------------------------------------

class TestDiameter:
    def test_complete_graph(self):
        ni = _complete_neigh_idx(8)
        d = effective_diameter(ni, rng=np.random.default_rng(0))
        assert d["max_distance"] == 1

    def test_path_graph(self):
        ni = _path_neigh_idx(10)
        d = effective_diameter(ni, num_samples=10, rng=np.random.default_rng(0))
        assert d["max_distance"] == 9


# ------------------------------------------------------------------
# degree_stats
# ------------------------------------------------------------------

class TestDegreeStats:
    def test_regular(self):
        # cycle graph is 2-regular
        T = 8
        ni = np.full((T, 2), -1, dtype=np.int32)
        for i in range(T):
            ni[i, 0] = (i - 1) % T
            ni[i, 1] = (i + 1) % T
        ds = degree_stats(ni)
        assert ds["is_regular"]
        assert ds["degree_min"] == ds["degree_max"] == 2


# ------------------------------------------------------------------
# graph_quality_report
# ------------------------------------------------------------------

class TestQualityReport:
    def test_has_all_sections(self):
        ni = _cycle_plus_window(32)
        r = graph_quality_report(ni, rng=np.random.default_rng(0))
        assert "degree" in r
        assert "spectral" in r
        assert "expansion" in r
        assert "mixing" in r
        assert "diameter" in r
        assert "summary" in r


# ------------------------------------------------------------------
# Causal reachability
# ------------------------------------------------------------------

class TestCausalReachability:
    def test_complete_graph_fast(self):
        ni = _complete_neigh_idx(8)
        cr = causal_reachability(ni, max_layers=4)
        # complete graph: 1 layer should give full coverage
        assert cr["mean_coverage"][0] > 0.9

    def test_window_only_slow(self):
        # window-only (path) should be slow
        ni = _path_neigh_idx(32)
        cr = causal_reachability(ni, max_layers=4)
        # at layer 1, coverage should be small
        assert cr["mean_coverage"][0] < 0.5

    def test_cycle_beats_window(self):
        cw = _cycle_plus_window(64, window=4)
        pw = _path_neigh_idx(64)
        cr_cw = causal_reachability(cw, max_layers=6)
        cr_pw = causal_reachability(pw, max_layers=6)
        # cycle + window should have better coverage at layer 5
        assert cr_cw["mean_coverage"][4] > cr_pw["mean_coverage"][4]

    def test_monotonic(self):
        ni = _cycle_plus_window(32)
        cr = causal_reachability(ni, max_layers=8)
        for i in range(len(cr["mean_coverage"]) - 1):
            assert cr["mean_coverage"][i + 1] >= cr["mean_coverage"][i] - 1e-9


# ------------------------------------------------------------------
# Causal diameter
# ------------------------------------------------------------------

class TestCausalDiameter:
    def test_complete_graph(self):
        ni = _complete_neigh_idx(8)
        cd = causal_diameter(ni, rng=np.random.default_rng(0))
        assert cd["causal_diameter"] == 1

    def test_path_scales(self):
        ni = _path_neigh_idx(16)
        cd = causal_diameter(ni, num_samples=16, rng=np.random.default_rng(0))
        assert cd["causal_diameter"] >= 10  # should be ~15 for path


# ------------------------------------------------------------------
# Combined reports
# ------------------------------------------------------------------

class TestCausalReport:
    def test_has_causal_section(self):
        ni = _cycle_plus_window(32)
        r = graph_quality_report_causal(
            ni, max_layers=4, rng=np.random.default_rng(0),
        )
        assert "causal" in r
        assert "reachability" in r["causal"]
        assert "diameter" in r["causal"]

    def test_is_good_expander(self):
        ni = _complete_neigh_idx(8)
        assert is_good_expander(ni)
        # single node is not
        ni1 = np.full((1, 1), -1, dtype=np.int32)
        assert not is_good_expander(ni1)
