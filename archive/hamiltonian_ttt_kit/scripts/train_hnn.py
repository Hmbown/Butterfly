"""Train an HNN on an analytic Hamiltonian system.

Example:
    python scripts/train_hnn.py --system harmonic --k 1.0 --m 1.0 --epochs 50

Requires torch: pip install -e ".[torch]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hamiltonian_ttt.datasets import make_supervised_dataset
from hamiltonian_ttt.systems import CoupledOscillators, HarmonicOscillator, SimplePendulum

try:
    import torch
except Exception as e:
    raise SystemExit("This script needs PyTorch. Install with: pip install -e '.[torch]'") from e

from hamiltonian_ttt.models import HNNConfig, HamiltonianNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["harmonic", "pendulum", "coupled"], default="harmonic")
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--n_traj", type=int, default=256)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=str, default="128,128,128")
    p.add_argument("--activation", type=str, default="tanh")
    p.add_argument("--seed", type=int, default=0)

    # system-specific
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--k_c", type=float, default=0.2)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--g", type=float, default=9.81)

    p.add_argument("--outdir", type=str, default="runs/hnn")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.system == "harmonic":
        system = HarmonicOscillator(dim=args.dim, m=args.m, k=args.k)
    elif args.system == "pendulum":
        if args.dim != 1:
            raise SystemExit("pendulum demo assumes dim=1 (single angle)")
        system = SimplePendulum(dim=1, m=args.m, L=args.L, g=args.g)
    else:
        system = CoupledOscillators(dim=args.dim, m=args.m, k=args.k, k_c=args.k_c)

    train = make_supervised_dataset(
        system, n_traj=args.n_traj, steps=args.steps, dt=args.dt, seed=args.seed
    )

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = HamiltonianNN(HNNConfig(dim=system.dim, hidden_dims=hidden, activation=args.activation))

    device = "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    X = torch.tensor(train.x, dtype=torch.float32, device=device)
    Y = torch.tensor(train.dxdt, dtype=torch.float32, device=device)

    N = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        count = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = X[idx].clone().detach().requires_grad_(True)
            y = Y[idx]

            opt.zero_grad(set_to_none=True)
            pred = model.time_derivative(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item()) * x.shape[0]
            count += x.shape[0]
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {ep+1:03d}/{args.epochs} loss={total / max(count, 1):.6f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"hnn_{args.system}_dim{args.dim}_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "train_meta": train.meta}, ckpt_path)

    meta_path = outdir / f"hnn_{args.system}_dim{args.dim}_seed{args.seed}.json"
    meta_path.write_text(json.dumps({"args": vars(args), "train_meta": train.meta}, indent=2))
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()
