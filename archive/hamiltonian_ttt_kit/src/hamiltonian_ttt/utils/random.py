from __future__ import annotations

import numpy as np


def rng_from_seed(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
