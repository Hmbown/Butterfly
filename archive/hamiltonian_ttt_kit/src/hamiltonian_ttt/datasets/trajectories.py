from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..integrators.symplectic import integrate
from ..systems.base import SeparableHamiltonianSystem
from ..utils.random import rng_from_seed


@dataclass(frozen=True)
class TrajectoryBatch:
    """A batch of simulated data suitable for supervised training."""
    x: NDArray[np.floating]  # shape (N, 2*dim)
    dxdt: NDArray[np.floating]  # shape (N, 2*dim)
    meta: dict


def simulate_trajectory(
    system: SeparableHamiltonianSystem,
    x0: ArrayLike,
    dt: float,
    steps: int,
    method: Literal["verlet", "symplectic_euler"] = "verlet",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return (ts, xs) for a single trajectory."""
    xs = integrate(system, x0=x0, dt=dt, steps=steps, method=method)
    ts = dt * np.arange(xs.shape[0], dtype=float)
    return ts, xs


def sample_initial_conditions(
    dim: int,
    n: int,
    q_scale: float = 1.0,
    p_scale: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.floating]:
    rng = rng_from_seed(seed)
    q0 = rng.normal(size=(n, dim)) * q_scale
    p0 = rng.normal(size=(n, dim)) * p_scale
    return np.concatenate([q0, p0], axis=-1).astype(float)


def make_supervised_dataset(
    system: SeparableHamiltonianSystem,
    n_traj: int,
    steps: int,
    dt: float,
    method: Literal["verlet", "symplectic_euler"] = "verlet",
    q_scale: float = 1.0,
    p_scale: float = 1.0,
    seed: int | None = 0,
) -> TrajectoryBatch:
    """Generate a supervised dataset of (x, dx/dt) pairs.

    Targets use the *analytic* vector field from the system (not finite differences).
    """
    x0s = sample_initial_conditions(
        dim=system.dim, n=n_traj, q_scale=q_scale, p_scale=p_scale, seed=seed
    )

    xs_all = []
    dxdt_all = []
    energies = []
    for i in range(n_traj):
        _, xs = simulate_trajectory(system, x0s[i], dt=dt, steps=steps, method=method)
        xs_flat = xs[:-1]  # drop last
        dxdt = system.vector_field(xs_flat)
        xs_all.append(xs_flat)
        dxdt_all.append(dxdt)
        energies.append(system.H(xs_flat))

    X = np.concatenate(xs_all, axis=0)
    dX = np.concatenate(dxdt_all, axis=0)
    E = np.concatenate(energies, axis=0)
    meta = {
        "n_traj": n_traj,
        "steps": steps,
        "dt": dt,
        "method": method,
        "energy_mean": float(np.mean(E)),
        "energy_std": float(np.std(E)),
    }
    return TrajectoryBatch(x=X, dxdt=dX, meta=meta)
