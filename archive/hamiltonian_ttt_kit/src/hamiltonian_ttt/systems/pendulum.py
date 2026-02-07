from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import SeparableHamiltonianSystem


@dataclass(frozen=True)
class SimplePendulum(SeparableHamiltonianSystem):
    """Simple pendulum in Hamiltonian form.

    Coordinate q is angle (radians), momentum p is conjugate angular momentum.

    H(q,p) = p^2/(2 m L^2) + m g L (1 - cos(q))
    """

    dim: int = 1
    m: float = 1.0
    L: float = 1.0
    g: float = 9.81

    def T(self, p: NDArray[np.floating]):
        return 0.5 * np.sum(p**2, axis=-1) / (self.m * self.L**2)

    def V(self, q: NDArray[np.floating]):
        # q may have shape (...,1)
        return (self.m * self.g * self.L) * np.sum(1.0 - np.cos(q), axis=-1)

    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        return p / (self.m * self.L**2)

    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        return (self.m * self.g * self.L) * np.sin(q)
