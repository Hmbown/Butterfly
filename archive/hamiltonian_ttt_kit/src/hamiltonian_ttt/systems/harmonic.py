from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import SeparableHamiltonianSystem


@dataclass(frozen=True)
class HarmonicOscillator(SeparableHamiltonianSystem):
    """n-D harmonic oscillator with diagonal mass and spring constants.

    H(q,p) = sum_i [ p_i^2 / (2 m_i) + 1/2 k_i q_i^2 ]
    """

    dim: int = 1
    m: float | NDArray[np.floating] = 1.0
    k: float | NDArray[np.floating] = 1.0

    def __post_init__(self):
        object.__setattr__(self, "m", np.asarray(self.m, dtype=float))
        object.__setattr__(self, "k", np.asarray(self.k, dtype=float))

    def T(self, p: NDArray[np.floating]):
        return 0.5 * np.sum((p**2) / self.m, axis=-1)

    def V(self, q: NDArray[np.floating]):
        return 0.5 * np.sum(self.k * (q**2), axis=-1)

    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        return p / self.m

    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.k * q
