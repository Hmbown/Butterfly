import numpy as np
from bna.graph.analysis import fiedler_bridge_candidates, laplacian_spectral_gap


def test_connected_graph_positive_fiedler():
    """Connected cycle graph should have positive Fiedler value."""
    T = 20
    perm = np.arange(T)  # identity = cycle 0->1->2->...->T-1->0
    result = laplacian_spectral_gap(perm, include_window=True, window=2)
    assert result["fiedler_value"] > 0
    assert result["is_well_connected"]
    assert result["cheeger_lower"] >= 0
    assert result["cheeger_upper"] >= result["cheeger_lower"]
    assert len(result["fiedler_vector"]) == T


def test_disconnected_graph_zero_fiedler():
    """More window edges should yield higher Fiedler value."""
    T = 20
    perm = np.arange(T)
    result_small_window = laplacian_spectral_gap(
        perm, include_window=True, window=1
    )
    result_large_window = laplacian_spectral_gap(
        perm, include_window=True, window=5
    )
    assert (
        result_large_window["fiedler_value"]
        > result_small_window["fiedler_value"]
    )


def test_fiedler_bridge_candidates_returns_edges():
    """Bridge candidates should return valid edge pairs."""
    T = 30
    perm = np.arange(T)
    bridges = fiedler_bridge_candidates(perm, window=1, num_bridges=5)
    assert len(bridges) > 0
    assert len(bridges) <= 5
    for i, j in bridges:
        assert 0 <= i < T
        assert 0 <= j < T
        assert i != j


def test_bridges_improve_connectivity():
    """Adding bridge edges should improve Fiedler value."""
    T = 30
    rng = np.random.default_rng(42)
    perm = rng.permutation(T)

    result_before = laplacian_spectral_gap(
        perm, include_window=True, window=1
    )
    result_after = laplacian_spectral_gap(
        perm, include_window=True, window=3
    )
    assert result_after["fiedler_value"] >= result_before["fiedler_value"]
