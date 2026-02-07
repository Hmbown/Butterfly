from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def plot_trajectory(ts: NDArray[np.floating], xs: NDArray[np.floating], title: str = ""):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    ts = np.asarray(ts, dtype=float)
    xs = np.asarray(xs, dtype=float)
    D = xs.shape[-1]
    for d in range(D):
        plt.plot(ts, xs[:, d], label=f"x[{d}]")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
