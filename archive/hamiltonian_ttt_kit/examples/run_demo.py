"""End-to-end demo: analytic Hamiltonian -> HNN -> test-time tuning head.

Run:
    python examples/run_demo.py

Notes:
- Requires torch. Install with: pip install -e ".[torch]"
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from hamiltonian_ttt.datasets import make_supervised_dataset, simulate_trajectory
from hamiltonian_ttt.systems import HarmonicOscillator
from hamiltonian_ttt.utils.metrics import mse

try:
    import torch
except Exception as e:
    raise SystemExit(
        "This demo needs PyTorch. Install with: pip install -e '.[torch]'"
    ) from e

from hamiltonian_ttt.models import (
    DeltaHamiltonianHead,
    HNNConfig,
    HamiltonianNN,
    HamiltonianWithHead,
    TTTHeadConfig,
)
from hamiltonian_ttt.ttt import TTTConfig, tune_head


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_hnn(
    model: HamiltonianNN,
    X: np.ndarray,
    dX: np.ndarray,
    *,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 1024,
    device: str = "cpu",
) -> list[float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    dX_t = torch.tensor(dX, dtype=torch.float32, device=device)

    losses = []
    N = X_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        count = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x = X_t[idx].clone().detach().requires_grad_(True)
            y = dX_t[idx]

            opt.zero_grad(set_to_none=True)
            pred = model.time_derivative(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item()) * x.shape[0]
            count += x.shape[0]
        losses.append(total / max(count, 1))
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {ep+1:03d}/{epochs} loss={losses[-1]:.6f}")
    return losses


def rollout_euler(model, x0: np.ndarray, dt: float, steps: int, device: str = "cpu") -> np.ndarray:
    x = torch.tensor(x0, dtype=torch.float32, device=device)
    xs = [x.detach().cpu().numpy()]
    for _ in range(steps):
        # Need grad for autograd-based dynamics
        xg = x.clone().detach().requires_grad_(True)
        dx = model.time_derivative(xg)
        x = x.detach() + dt * dx.detach()
        xs.append(x.detach().cpu().numpy())
    return np.stack(xs, axis=0)


def main():
    set_seed(0)

    # ----- Training system -----
    train_sys = HarmonicOscillator(dim=1, m=1.0, k=1.0)
    dt = 0.05
    steps = 80

    train = make_supervised_dataset(
        train_sys, n_traj=256, steps=steps, dt=dt, q_scale=1.0, p_scale=1.0, seed=0
    )
    print("train meta:", train.meta)

    # ----- Train HNN -----
    hnn = HamiltonianNN(HNNConfig(dim=train_sys.dim, hidden_dims=[128, 128, 128], activation="tanh"))
    train_hnn(hnn, train.x, train.dxdt, lr=1e-3, epochs=25, batch_size=2048)

    # ----- Test system: distribution shift -----
    test_sys = HarmonicOscillator(dim=1, m=1.0, k=2.25)  # higher stiffness => different frequency
    x0 = np.array([1.0, 0.0], dtype=float)
    T_total = 200
    ts, xs_true = simulate_trajectory(test_sys, x0=x0, dt=dt, steps=T_total, method="verlet")

    # ----- Baseline rollout from midpoint -----
    K_obs = 40
    horizon = 120
    x_start = xs_true[K_obs]

    xs_pred_base = rollout_euler(hnn, x_start, dt=dt, steps=horizon)

    xs_true_future = xs_true[K_obs : K_obs + horizon + 1]
    base_err = mse(xs_pred_base, xs_true_future)
    print(f"baseline future MSE over horizon={horizon}: {base_err:.6f}")

    # ----- TTT: add a small delta-H head and tune on observed prefix -----
    head = DeltaHamiltonianHead(TTTHeadConfig(dim=train_sys.dim, hidden_dims=[64, 64], activation="tanh"))
    model_ttt = HamiltonianWithHead(base=hnn, head=head)

    xs_obs = xs_true[: K_obs + 1]  # observed prefix (including x_start)
    cfg = TTTConfig(lr=5e-2, steps=300, loss="fd+energy", fd_weight=1.0, energy_weight=0.1, l2_head_weight=1e-6)
    print("TTT cfg:", asdict(cfg))
    out = tune_head(model_ttt, xs_obs=xs_obs, dt=dt, cfg=cfg)
    print(f"TTT final loss: {out['losses'][-1]:.6f}")

    xs_pred_ttt = rollout_euler(out["tuned_model"], x_start, dt=dt, steps=horizon)
    ttt_err = mse(xs_pred_ttt, xs_true_future)
    print(f"TTT future MSE over horizon={horizon}: {ttt_err:.6f}")

    # ----- Plot (q coordinate only) -----
    try:
        import matplotlib.pyplot as plt

        t_future = ts[K_obs : K_obs + horizon + 1] - ts[K_obs]
        plt.figure()
        plt.plot(t_future, xs_true_future[:, 0], label="true q")
        plt.plot(t_future, xs_pred_base[:, 0], label="baseline q")
        plt.plot(t_future, xs_pred_ttt[:, 0], label="TTT q")
        plt.xlabel("t (relative)")
        plt.ylabel("q")
        plt.title("Harmonic oscillator (shifted k) — baseline vs TTT")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped (matplotlib issue):", e)


if __name__ == "__main__":
    main()
