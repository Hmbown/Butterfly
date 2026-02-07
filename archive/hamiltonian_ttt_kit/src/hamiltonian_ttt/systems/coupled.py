from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import SeparableHamiltonianSystem


@dataclass(frozen=True)
class CoupledOscillators(SeparableHamiltonianSystem):
    """1D chain of n coupled oscillators with nearest-neighbor coupling.

    q, p are both shape (..., n)

    H = sum_i p_i^2/(2m)
        + sum_i 0.5*k*q_i^2
        + sum_{i=1..n-1} 0.5*k_c*(q_{i+1}-q_i)^2
    """

    dim: int
    m: float = 1.0
    k: float = 1.0
    k_c: float = 0.2

    def T(self, p: NDArray[np.floating]):
        return 0.5 * np.sum(p**2, axis=-1) / self.m

    def V(self, q: NDArray[np.floating]):
        # onsite spring
        V0 = 0.5 * self.k * np.sum(q**2, axis=-1)
        # coupling
        dq = q[..., 1:] - q[..., :-1]
        Vc = 0.5 * self.k_c * np.sum(dq**2, axis=-1)
        return V0 + Vc

    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        return p / self.m

    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        # gradient of onsite term: k*q
        g = self.k * q
        # coupling contributions
        # For i: derivative wrt q_i includes:
        #   from (q_i - q_{i-1})^2 term: +k_c*(q_i - q_{i-1})
        #   from (q_{i+1} - q_i)^2 term: -k_c*(q_{i+1} - q_i) = k_c*(q_i - q_{i+1})
        # Handle boundaries carefully
        if q.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {q.shape[-1]}")
        # left difference
        g[..., 1:] += self.k_c * (q[..., 1:] - q[..., :-1])
        # right difference
        g[..., :-1] += self.k_c * (q[..., :-1] - q[..., 1:])
        return g
