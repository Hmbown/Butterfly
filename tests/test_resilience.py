from __future__ import annotations

import numpy as np

from hcsa.graph.analysis import check_resilience


def test_resilience_safe_drop_rate_survives() -> None:
    perm = np.random.default_rng(42).permutation(128)
    out = check_resilience(
        perm,
        window=32,
        drop_rate=0.3,
        num_trials=100,
        rng=np.random.default_rng(1),
    )
    assert float(out["survival_rate"]) > 0.95
    assert float(out["min_degree_mean"]) >= float(out["theoretical_threshold"])


def test_resilience_aggressive_drop_rate_fails() -> None:
    perm = np.random.default_rng(42).permutation(128)
    out = check_resilience(
        perm,
        window=32,
        drop_rate=0.8,
        num_trials=100,
        rng=np.random.default_rng(1),
    )
    assert float(out["survival_rate"]) < 0.5


def test_resilience_threshold_consistency() -> None:
    perm = np.random.default_rng(7).permutation(128)
    out = check_resilience(
        perm,
        window=32,
        drop_rate=0.3,
        num_trials=64,
        rng=np.random.default_rng(3),
    )
    assert float(out["theoretical_threshold"]) > 0.0
    assert int(out["base_degree_min"]) >= int(out["theoretical_threshold"])
