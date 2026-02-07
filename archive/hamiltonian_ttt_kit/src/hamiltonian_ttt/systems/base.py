from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SeparableHamiltonian(Protocol):
    """Protocol for separable Hamiltonians H(q,p)=T(p)+V(q)."""

    dim: int

    def T(self, p: NDArray[np.floating]) -> NDArray[np.floating] | float: ...
    def V(self, q: NDArray[np.floating]) -> NDArray[np.floating] | float: ...
    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]: ...


@dataclass(frozen=True)
class SeparableHamiltonianSystem:
    """Base class for separable Hamiltonian systems.

    State convention:
      - q: position / generalized coordinate, shape (..., dim)
      - p: momentum / generalized momentum, shape (..., dim)
      - x: concatenated state [q, p], shape (..., 2*dim)
    """

    dim: int

    def T(self, p: NDArray[np.floating]) -> NDArray[np.floating] | float:
        raise NotImplementedError

    def V(self, q: NDArray[np.floating]) -> NDArray[np.floating] | float:
        raise NotImplementedError

    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    # Convenience wrappers

    def split(self, x: ArrayLike) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != 2 * self.dim:
            raise ValueError(f"Expected last dim {2*self.dim}, got {x.shape[-1]}")
        q = x[..., : self.dim]
        p = x[..., self.dim :]
        return q, p

    def join(self, q: ArrayLike, p: ArrayLike) -> NDArray[np.floating]:
        q = np.asarray(q, dtype=float)
        p = np.asarray(p, dtype=float)
        return np.concatenate([q, p], axis=-1)

    def H(self, x: ArrayLike) -> NDArray[np.floating]:
        q, p = self.split(x)
        return np.asarray(self.T(p) + self.V(q), dtype=float)

    def grad_H(self, x: ArrayLike) -> NDArray[np.floating]:
        """Gradient of H wrt [q,p] as [dV/dq, dT/dp]."""
        q, p = self.split(x)
        dV = self.dV_dq(q)
        dT = self.dT_dp(p)
        return self.join(dV, dT)

    def vector_field(self, x: ArrayLike) -> NDArray[np.floating]:
        """Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq."""
        q, p = self.split(x)
        dq = self.dT_dp(p)
        dp = -self.dV_dq(q)
        return self.join(dq, dp)


def finite_difference_grad(f, x: NDArray[np.floating], eps: float = 1e-6) -> NDArray[np.floating]:
    """Simple finite-difference gradient for debugging / prototyping."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)
    for i in range(x.shape[-1]):
        e = np.zeros_like(x, dtype=float)
        e[..., i] = eps
        g[..., i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g
