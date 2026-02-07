from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    _torch_import_error = e

from .mlp import make_mlp, require_torch


@dataclass(frozen=True)
class HNNConfig:
    dim: int
    hidden_dims: list[int] = (128, 128, 128)
    activation: str = "tanh"


class HamiltonianNN(nn.Module):
    """Hamiltonian Neural Network.

    Learns a scalar H_θ(x). Dynamics come from Hamilton's equations:

        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q

    State x is concatenated [q, p], shape (..., 2*dim)
    """

    def __init__(self, cfg: HNNConfig):
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.net = make_mlp(
            in_dim=2 * cfg.dim,
            hidden_dims=list(cfg.hidden_dims),
            out_dim=1,
            activation=cfg.activation,
        )

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        # shape (..., 1) -> (...,)
        return self.net(x).squeeze(-1)

    def time_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dx/dt from H via autograd."""
        # Ensure x requires grad for autograd-based gradient
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        H = self.hamiltonian(x)
        # grad of sum gives per-sample grads efficiently
        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dim = self.cfg.dim
        dH_dq = gradH[..., :dim]
        dH_dp = gradH[..., dim:]
        dqdt = dH_dp
        dpdt = -dH_dq
        return torch.cat([dqdt, dpdt], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.time_derivative(x)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)
