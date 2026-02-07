from __future__ import annotations

import numpy as np


def test_permute_inverse_roundtrip() -> None:
    rng = np.random.default_rng(7)
    B, T, dh = 2, 17, 5
    x = rng.standard_normal((B, T, dh), dtype=np.float32)
    perm = rng.permutation(T)
    inv = np.argsort(perm)

    x_pi = x[:, perm]
    x_back = x_pi[:, inv]

    assert np.array_equal(x_back, x)


def test_cycle_neighbors_become_local_in_permuted_space() -> None:
    rng = np.random.default_rng(13)
    T = 19
    perm = rng.permutation(T)

    inv = np.empty(T, dtype=np.int32)
    inv[perm] = np.arange(T, dtype=np.int32)

    for pos in range(T):
        node = int(perm[pos])
        prev_node = int(perm[(pos - 1) % T])
        next_node = int(perm[(pos + 1) % T])

        prev_pos = int(inv[prev_node])
        next_pos = int(inv[next_node])

        assert (pos - prev_pos) % T in (1, T - 1)
        assert (next_pos - pos) % T in (1, T - 1)
        assert 0 <= node < T
