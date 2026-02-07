from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    _torch_import_error = e


def require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for hamiltonian_ttt.models. "
            "Install with: pip install -e '.[torch]'"
        ) from _torch_import_error


def make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: str = "tanh",
):
    require_torch()
    act_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation: {activation}")

    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(act_map[activation]())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
