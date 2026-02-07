from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None
    _torch_import_error = e

from ..utils.finite_difference import finite_difference_dxdt
from ..models.ttt_head import HamiltonianWithHead


def require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for hamiltonian_ttt.ttt. Install with: pip install -e '.[torch]'"
        ) from _torch_import_error


@dataclass(frozen=True)
class TTTConfig:
    lr: float = 1e-2
    steps: int = 200
    weight_decay: float = 0.0
    loss: Literal["fd", "energy", "fd+energy"] = "fd"
    energy_weight: float = 1.0
    fd_weight: float = 1.0
    l2_head_weight: float = 1e-4
    device: str = "cpu"


def tune_head(
    model: HamiltonianWithHead,
    xs_obs: np.ndarray,
    dt: float,
    cfg: TTTConfig = TTTConfig(),
) -> dict:
    """Tune the model.head parameters on an observed prefix trajectory.

    xs_obs:
      numpy array shape (T, D)
      Usually the first K steps you have at test time.

    Returns dict with:
      - losses: list[float]
      - tuned_model: model (mutated in-place, also returned for convenience)
    """
    require_torch()
    model = model.to(cfg.device)

    # Freeze base model
    for p in model.base.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)

    xs_obs = np.asarray(xs_obs, dtype=float)
    x_t = torch.tensor(xs_obs, dtype=torch.float32, device=cfg.device)

    # Precompute finite-difference dxdt labels if needed
    if cfg.loss in ("fd", "fd+energy"):
        dxdt_fd = finite_difference_dxdt(xs_obs, dt=dt, method="central")
        dxdt_t = torch.tensor(dxdt_fd, dtype=torch.float32, device=cfg.device)
    else:
        dxdt_t = None

    opt = torch.optim.Adam(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    losses: list[float] = []
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        # Need grad wrt x for Hamiltonian time_derivative
        x = x_t.clone().detach().requires_grad_(True)
        dxdt_pred = model.time_derivative(x)

        loss = 0.0

        if cfg.loss in ("fd", "fd+energy"):
            fd = torch.mean((dxdt_pred - dxdt_t) ** 2)
            loss = loss + cfg.fd_weight * fd

        if cfg.loss in ("energy", "fd+energy"):
            H = model.hamiltonian(x)
            H_centered = H - torch.mean(H)
            energy_var = torch.mean(H_centered**2)
            loss = loss + cfg.energy_weight * energy_var

        # small regularization to avoid huge corrections
        l2 = 0.0
        for p in model.head.parameters():
            l2 = l2 + torch.sum(p**2)
        loss = loss + cfg.l2_head_weight * l2

        loss.backward()
        opt.step()

        losses.append(float(loss.detach().cpu().item()))

    return {"losses": losses, "tuned_model": model}
