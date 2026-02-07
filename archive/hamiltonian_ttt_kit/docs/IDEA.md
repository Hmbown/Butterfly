# Hamiltonian + TTT idea sketch

This repo is structured around the following principle:

> If your system is (approximately) Hamiltonian, you can often use structure as a **self-supervised signal**
> at test time, even when labels are missing.

## 1) Hamiltonian basics (canonical coordinates)

Let `q ∈ R^d` be generalized coordinates and `p ∈ R^d` conjugate momenta.
A Hamiltonian `H(q,p)` defines a vector field:

- `dq/dt =  ∂H/∂p`
- `dp/dt = -∂H/∂q`

For **separable** Hamiltonians `H(q,p) = T(p) + V(q)`, symplectic integrators like velocity Verlet
have good long-term energy behavior.

In code: `SeparableHamiltonianSystem.vector_field(x)`.

## 2) Hamiltonian Neural Networks (HNNs)

An HNN learns a scalar function `H_θ(q,p)` and uses automatic differentiation to compute the gradient.
This guarantees the learned dynamics follow the Hamiltonian form (up to model expressivity).

In code: `models/HamiltonianNN`.

## 3) The “TTT” hook: tuning a small correction at inference time

When your test distribution shifts (new masses, new stiffness, friction-free assumption violated, etc.),
the base HNN will drift.

Instead of fully fine-tuning the whole network at inference time, we tune a **small head**:

- Base model learns `H_θ`
- Head learns a correction `ΔH_φ`
- Combined Hamiltonian: `H(q,p) = H_θ(q,p) + ΔH_φ(q,p)`

And we tune only `φ` at test time.

In code: `models/DeltaHamiltonianHead` + `models/HamiltonianWithHead`.

## 4) Test-time objectives (pick your poison)

This repo includes two simple objectives for the observed prefix trajectory `x_0, …, x_K`:

### A) Finite-difference matching (“fd”)

Estimate `dx/dt` from the observed prefix using finite differences:

`dx/dt ≈ (x_{t+1} - x_{t-1}) / (2Δt)`

Then tune `φ` to minimize MSE:

`L_fd(φ) = || f_{θ,φ}(x_t) - ẋ_t ||^2`

This is “self-supervised-ish” in the sense that it doesn’t need *external* labels,
but it does need observed time series.

### B) Energy consistency (“energy”)

For conservative systems, energy should be constant along trajectories:

`H(x_t) ≈ const`

Tune `φ` to minimize the variance of predicted energies over the observed prefix:

`L_energy(φ) = Var_t[ H_{θ,φ}(x_t) ]`

This is weaker than derivative supervision, but can work when observations are sparse/noisy.

In code: `ttt/tune.py`.

## 5) Where to extend

- Add new Hamiltonians in `systems/`.
- Add constraints (symmetries, invariants) as losses in `ttt/tune.py`.
- Replace the head with LoRA-style adapters if you want distributed capacity.
- Replace Euler rollouts with a symplectic integrator driven by the learned Hamiltonian.

## 6) Caveats

- If the real system is not conservative (damping/forcing), pure Hamiltonian structure may be mismatched.
  You may need port-Hamiltonian or dissipative extensions.
- Finite differences can be noisy; consider smoothing or higher-order schemes.
