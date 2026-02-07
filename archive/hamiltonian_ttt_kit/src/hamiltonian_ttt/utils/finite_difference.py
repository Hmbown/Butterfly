from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def finite_difference_dxdt(
    xs: NDArray[np.floating],
    dt: float,
    method: str = "central",
) -> NDArray[np.floating]:
    """Estimate time derivatives from a trajectory xs.

    xs: shape (T, D)
    returns: shape (T, D)

    Notes:
    - 'central' uses central differences for interior points, forward/backward at ends.
    - This is used for self-supervised-ish TTT on observed sequences.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.ndim != 2:
        raise ValueError("xs must be 2D (T, D)")
    T, D = xs.shape
    dx = np.zeros_like(xs, dtype=float)

    if T < 2:
        return dx

    if method == "forward":
        dx[:-1] = (xs[1:] - xs[:-1]) / dt
        dx[-1] = dx[-2]
        return dx

    if method == "central":
        dx[1:-1] = (xs[2:] - xs[:-2]) / (2 * dt)
        dx[0] = (xs[1] - xs[0]) / dt
        dx[-1] = (xs[-1] - xs[-2]) / dt
        return dx

    raise ValueError(f"Unknown method: {method}")
