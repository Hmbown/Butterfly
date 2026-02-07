from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mse(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def energy_drift(energies: NDArray[np.floating]) -> float:
    energies = np.asarray(energies, dtype=float).reshape(-1)
    if energies.size == 0:
        return 0.0
    return float(energies[-1] - energies[0])


def energy_std(energies: NDArray[np.floating]) -> float:
    energies = np.asarray(energies, dtype=float).reshape(-1)
    return float(np.std(energies))
