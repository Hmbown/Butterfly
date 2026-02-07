# Hamiltonian + TTT kit

This is a small, hackable codebase that combines:

- **Hamiltonian mechanics** (separable Hamiltonians `H(q,p)=T(p)+V(q)`) with symplectic simulation.
- **Hamiltonian Neural Networks (HNNs)** that learn a scalar Hamiltonian `H_θ(q,p)` and produce dynamics via Hamilton’s equations.
- A simple **Test-Time Tuning / Test-Time Training (TTT)** idea: adapt a small “head” at inference time using only a short observed prefix of a trajectory.

The goal is *iteration speed*: you can swap in your own Hamiltonian, add constraints, or change the TTT objective.

## Quickstart

```bash
# 1) Create a venv
python -m venv .venv
source .venv/bin/activate

# 2) Install (base deps)
pip install -e .

# 3) If you want the neural pieces:
pip install -e ".[torch]"
```

## Run a demo

The demo:
1) Generates training data for a harmonic oscillator.
2) Trains an HNN on that parameter regime.
3) Evaluates on a shifted regime (distribution shift).
4) Applies TTT on the *first K observed steps* of the test trajectory.

```bash
python examples/run_demo.py
```

If you don’t have `torch`, you can still play with the analytic Hamiltonians + integrators and the dataset generator.

## What to edit first

- `src/hamiltonian_ttt/systems/`:
  add your system (full Hamiltonian / potential / kinetic).
- `src/hamiltonian_ttt/ttt/`:
  change the test-time objective (energy conservation, FD matching, constraints, etc.).
- `examples/run_demo.py`:
  your “main loop” for idea iteration.

## Project layout

- `systems/` – analytic Hamiltonians for a few toy systems
- `integrators/` – symplectic Euler + velocity Verlet
- `datasets/` – trajectory simulation & supervised derivative targets
- `models/` – HNN + small TTT head (PyTorch)
- `ttt/` – tuning loop (optimize head on observed prefix)
- `examples/` – runnable experiments

## Notes

- The included integrators assume *separable* Hamiltonians for best stability.
- The included TTT implementation is intentionally simple: it adapts only a small head, not the full model.

License: MIT
