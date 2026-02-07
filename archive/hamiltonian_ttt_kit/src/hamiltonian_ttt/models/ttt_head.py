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
from .hnn import HamiltonianNN


@dataclass(frozen=True)
class TTTHeadConfig:
    dim: int
    hidden_dims: list[int] = (64, 64)
    activation: str = "tanh"


class DeltaHamiltonianHead(nn.Module):
    """A small module that outputs a scalar correction ΔH(x).

    Intended to be *tuned at test time* on a short observed prefix.
    """

    def __init__(self, cfg: TTTHeadConfig):
        require_torch()
        super().__init__()
        self.cfg = cfg
        self.net = make_mlp(
            in_dim=2 * cfg.dim,
            hidden_dims=list(cfg.hidden_dims),
            out_dim=1,
            activation=cfg.activation,
        )
        # Optionally start near zero: small initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HamiltonianWithHead(nn.Module):
    """Wrap a base HamiltonianNN with a tunable delta-H head."""

    def __init__(self, base: HamiltonianNN, head: DeltaHamiltonianHead):
        require_torch()
        super().__init__()
        self.base = base
        self.head = head
        self.dim = base.cfg.dim

    def hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.hamiltonian(x) + self.head(x)

    def time_derivative(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        H = self.hamiltonian(x)
        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dH_dq = gradH[..., : self.dim]
        dH_dp = gradH[..., self.dim :]
        dqdt = dH_dp
        dpdt = -dH_dq
        return torch.cat([dqdt, dpdt], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.time_derivative(x)
