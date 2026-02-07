from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..systems.base import SeparableHamiltonianSystem


def symplectic_euler(
    system: SeparableHamiltonianSystem,
    x0: ArrayLike,
    dt: float,
    steps: int,
) -> NDArray[np.floating]:
    """Symplectic Euler (semi-implicit) for separable H(q,p)=T(p)+V(q).

    Update:
      p_{n+1} = p_n - dt * dV/dq(q_n)
      q_{n+1} = q_n + dt * dT/dp(p_{n+1})

    Returns: trajectory array of shape (steps+1, 2*dim)
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape[0] != 2 * system.dim:
        raise ValueError(f"Expected state dim {2*system.dim}, got {x0.shape[0]}")
    q, p = system.split(x0)
    traj = np.zeros((steps + 1, 2 * system.dim), dtype=float)
    traj[0] = system.join(q, p)

    q = q.copy()
    p = p.copy()
    for t in range(steps):
        p = p - dt * system.dV_dq(q)
        q = q + dt * system.dT_dp(p)
        traj[t + 1] = system.join(q, p)
    return traj


def velocity_verlet(
    system: SeparableHamiltonianSystem,
    x0: ArrayLike,
    dt: float,
    steps: int,
) -> NDArray[np.floating]:
    """Velocity Verlet / Stormer-Verlet (symplectic, good energy behavior)."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape[0] != 2 * system.dim:
        raise ValueError(f"Expected state dim {2*system.dim}, got {x0.shape[0]}")
    q, p = system.split(x0)
    traj = np.zeros((steps + 1, 2 * system.dim), dtype=float)
    traj[0] = system.join(q, p)

    q = q.copy()
    p = p.copy()
    for t in range(steps):
        p_half = p - 0.5 * dt * system.dV_dq(q)
        q = q + dt * system.dT_dp(p_half)
        p = p_half - 0.5 * dt * system.dV_dq(q)
        traj[t + 1] = system.join(q, p)
    return traj


def integrate(
    system: SeparableHamiltonianSystem,
    x0: ArrayLike,
    dt: float,
    steps: int,
    method: Literal["verlet", "symplectic_euler"] = "verlet",
) -> NDArray[np.floating]:
    if method == "verlet":
        return velocity_verlet(system, x0, dt, steps)
    if method == "symplectic_euler":
        return symplectic_euler(system, x0, dt, steps)
    raise ValueError(f"Unknown method: {method}")
