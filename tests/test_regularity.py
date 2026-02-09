from __future__ import annotations

import numpy as np

from hcsa.cycles import regular_partition_cycle
from hcsa.graph.analysis import check_regularity


def test_regular_partition_cycle_is_valid_permutation() -> None:
    t = 256
    perm = regular_partition_cycle(
        t,
        num_clusters=8,
        generator=np.random.default_rng(42),
    )
    assert perm.shape == (t,)
    assert perm.dtype == np.int64
    assert int(perm.min()) == 0
    assert int(perm.max()) == t - 1
    assert np.unique(perm).shape[0] == t


def test_regular_partition_cycle_visits_every_vertex_once() -> None:
    t = 128
    perm = regular_partition_cycle(
        t,
        num_clusters=16,
        generator=np.random.default_rng(7),
    )
    seen = set(int(x) for x in perm.tolist())
    assert len(seen) == t
    assert seen == set(range(t))


def test_check_regularity_beats_identity_baseline() -> None:
    t = 256
    regular = regular_partition_cycle(
        t,
        num_clusters=8,
        generator=np.random.default_rng(3),
    )
    reg_info = check_regularity(regular, num_clusters=8)
    id_info = check_regularity(np.arange(t, dtype=np.int64), num_clusters=8)

    assert float(reg_info["max_deviation"]) < float(id_info["max_deviation"])
    assert float(reg_info["mean_deviation"]) < float(id_info["mean_deviation"])


def test_regular_partition_max_deviation_bound() -> None:
    t = 256
    perm = regular_partition_cycle(
        t,
        num_clusters=8,
        generator=np.random.default_rng(11),
    )
    reg_info = check_regularity(perm, num_clusters=8)
    assert float(reg_info["max_deviation"]) < 0.5
