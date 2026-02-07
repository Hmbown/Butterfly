"""Template for adding your own Hamiltonian system.

Most of this repo assumes *separable* Hamiltonians:

    H(q,p) = T(p) + V(q)

If your system is not separable, you can still prototype here, but the symplectic
integrators may not strictly apply.

Suggested workflow:
  1) Define your system here.
  2) Add it to systems/__init__.py
  3) Use datasets.make_supervised_dataset(...) to generate training pairs.
  4) Train an HNN, then use TTT to adapt on a test prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import SeparableHamiltonianSystem


@dataclass(frozen=True)
class YourSystem(SeparableHamiltonianSystem):
    dim: int = 2  # example: 2D

    # add parameters here, e.g.
    # m: float = 1.0
    # k: float = 1.0

    def T(self, p: NDArray[np.floating]):
        # TODO: implement kinetic energy
        # return ...
        raise NotImplementedError

    def V(self, q: NDArray[np.floating]):
        # TODO: implement potential energy
        # return ...
        raise NotImplementedError

    def dT_dp(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        # TODO: gradient of T wrt p
        raise NotImplementedError

    def dV_dq(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        # TODO: gradient of V wrt q
        raise NotImplementedError
