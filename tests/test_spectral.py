from __future__ import annotations

import numpy as np

from bna.graph.analysis import expansion_proxy, spectral_gap


def test_random_cycle_gap_beats_identity_with_window() -> None:
    t = 128
    rng = np.random.default_rng(42)
    perm_rand = rng.permutation(t)
    perm_identity = np.arange(t, dtype=np.int64)

    rand_info = spectral_gap(
        perm_rand,
        include_window=True,
        window=8,
        expander_threshold=1.1,
    )
    id_info = spectral_gap(
        perm_identity,
        include_window=True,
        window=8,
        expander_threshold=1.1,
    )

    assert float(rand_info["spectral_gap"]) > 2.0
    assert float(id_info["spectral_gap"]) < 1.0
    assert float(rand_info["expansion_ratio"]) > float(id_info["expansion_ratio"])
    assert bool(rand_info["is_good_expander"]) is True
    assert bool(id_info["is_good_expander"]) is False


def test_window_edges_increase_gap() -> None:
    t = 256
    rng = np.random.default_rng(7)
    perm = rng.permutation(t)
    no_window = spectral_gap(perm, include_window=False)
    with_window = spectral_gap(perm, include_window=True, window=16)

    assert float(with_window["spectral_gap"]) > float(no_window["spectral_gap"])
    assert float(with_window["degree"]) > float(no_window["degree"])


def test_expansion_proxy_directionally_matches_spectral_signal() -> None:
    t = 128
    rng = np.random.default_rng(11)
    perm_rand = rng.permutation(t)
    perm_identity = np.arange(t, dtype=np.int64)

    proxy_rand = expansion_proxy(
        perm_rand,
        window=8,
        num_walks=2000,
        walk_len=20,
        rng=np.random.default_rng(3),
    )
    proxy_identity = expansion_proxy(
        perm_identity,
        window=8,
        num_walks=2000,
        walk_len=20,
        rng=np.random.default_rng(3),
    )

    assert float(proxy_rand["endpoint_uniformity"]) < float(proxy_identity["endpoint_uniformity"])
    assert int(proxy_rand["mixing_time_estimate"]) <= int(proxy_identity["mixing_time_estimate"])
