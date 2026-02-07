import numpy as np

from hamiltonian_ttt.integrators import velocity_verlet
from hamiltonian_ttt.systems import HarmonicOscillator


def test_verlet_energy_drift_small():
    sys = HarmonicOscillator(dim=1, m=1.0, k=1.0)
    x0 = np.array([1.0, 0.0])
    dt = 0.05
    steps = 2000
    xs = velocity_verlet(sys, x0=x0, dt=dt, steps=steps)
    E = sys.H(xs)
    drift = float(E[-1] - E[0])

    # Symplectic integrators should keep drift bounded (not strictly 0).
    assert abs(drift) < 1e-2
