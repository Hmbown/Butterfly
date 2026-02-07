"""Run an analytic Hamiltonian simulation (no torch required).

Example:
    python examples/simulate_only.py --system pendulum
"""

from __future__ import annotations

import argparse

import numpy as np

from hamiltonian_ttt.integrators import integrate
from hamiltonian_ttt.systems import CoupledOscillators, HarmonicOscillator, SimplePendulum


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["harmonic", "pendulum", "coupled"], default="harmonic")
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    if args.system == "harmonic":
        sys = HarmonicOscillator(dim=args.dim, m=1.0, k=1.0)
        x0 = np.concatenate([np.ones(args.dim), np.zeros(args.dim)])
    elif args.system == "pendulum":
        sys = SimplePendulum(dim=1, m=1.0, L=1.0, g=9.81)
        x0 = np.array([1.0, 0.0])
    else:
        sys = CoupledOscillators(dim=args.dim, m=1.0, k=1.0, k_c=0.2)
        x0 = np.concatenate([np.ones(args.dim), np.zeros(args.dim)])

    xs = integrate(sys, x0=x0, dt=args.dt, steps=args.steps, method="verlet")
    E = sys.H(xs)
    print("Energy drift:", float(E[-1] - E[0]))
    print("Energy std:", float(np.std(E)))

    try:
        import matplotlib.pyplot as plt

        t = args.dt * np.arange(xs.shape[0])
        plt.figure()
        plt.plot(t, E)
        plt.xlabel("t")
        plt.ylabel("H")
        plt.title(f"{args.system} energy vs time (verlet)")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
