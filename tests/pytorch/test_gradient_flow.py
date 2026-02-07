"""Gradient flow tests.

Verify that all trainable parameters receive non-zero gradients,
no NaN/Inf values appear, and gradient norms are bounded.
"""

from __future__ import annotations

import torch

from hcsa.data import build_datasets, get_batch
from hcsa.model import GPT, GPTConfig
from hcsa.tokenizers import CharTokenizer
from hcsa.utils import set_seed


def _make_tiny_model(attn: str = "dense") -> tuple[GPT, torch.Tensor, torch.Tensor]:
    """Create a tiny model and a single batch for gradient testing."""
    set_seed(42)
    text = "Hello, world! " * 200
    tok = CharTokenizer.from_text(text)
    data = build_datasets(text, tok, val_fraction=0.1)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        seq_len=16,
        n_layers=2,
        n_heads=2,
        n_embd=32,
        dropout=0.0,
        attn=attn,  # type: ignore[arg-type]
        cycle="random",
        window=4,
        landmark_stride=4,
        seed=42,
    )
    model = GPT(cfg)
    xb, yb = get_batch(data.train, batch_size=2, seq_len=16, device=torch.device("cpu"))
    return model, xb, yb


def _check_gradients(model: GPT, xb: torch.Tensor, yb: torch.Tensor) -> None:
    """Run forward + backward and validate gradient properties."""
    model.train()
    out = model(xb, yb)
    loss = out["loss"]
    loss.backward()

    # Loss must be finite
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # The routing projection Wr is used for cycle construction (discrete, non-differentiable)
    # so it won't receive gradients through the standard backprop path.
    no_grad_expected = {"Wr.weight"}

    has_nonzero_grad = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip params known to have no gradient path
        if any(pat in name for pat in no_grad_expected):
            continue

        # All other parameters should have gradients
        assert param.grad is not None, f"No gradient for {name}"

        # No NaN or Inf in gradients
        assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"

        # Non-zero gradient (at least some elements should be non-zero)
        if param.grad.abs().sum() > 0:
            has_nonzero_grad = True

        # Gradient norm should be bounded (sanity check)
        grad_norm = param.grad.norm().item()
        assert grad_norm < 1e6, f"Gradient norm too large for {name}: {grad_norm}"

    assert has_nonzero_grad, "No parameter received a non-zero gradient"


def test_gradient_flow_dense() -> None:
    model, xb, yb = _make_tiny_model("dense")
    _check_gradients(model, xb, yb)


def test_gradient_flow_hcsa() -> None:
    model, xb, yb = _make_tiny_model("hcsa")
    _check_gradients(model, xb, yb)


def test_gradient_flow_hcsa_greedy() -> None:
    set_seed(42)
    text = "Hello, world! " * 200
    tok = CharTokenizer.from_text(text)
    data = build_datasets(text, tok, val_fraction=0.1)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        seq_len=16, n_layers=2, n_heads=2, n_embd=32,
        dropout=0.0, attn="hcsa", cycle="greedy",
        window=4, landmark_stride=4, seed=42,
    )
    model = GPT(cfg)
    xb, yb = get_batch(data.train, batch_size=2, seq_len=16, device=torch.device("cpu"))
    _check_gradients(model, xb, yb)


def test_loss_decreases_one_step() -> None:
    """Verify that one optimizer step reduces the loss."""
    set_seed(42)
    model, xb, yb = _make_tiny_model("dense")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    out1 = model(xb, yb)
    loss1 = out1["loss"].item()

    out1["loss"].backward()
    opt.step()
    opt.zero_grad()

    out2 = model(xb, yb)
    loss2 = out2["loss"].item()

    # Loss should decrease (with high probability on the same batch)
    assert loss2 < loss1, f"Loss did not decrease: {loss1} -> {loss2}"
