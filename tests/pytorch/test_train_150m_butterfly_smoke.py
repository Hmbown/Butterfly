"""Smoke tests for ``GPTConfigTorch.tiny_150m`` + one training step."""

from __future__ import annotations

import pytest
import torch

from bna.torch.model import GPTConfigTorch, GPTTorch


def test_tiny_150m_config_estimates_params() -> None:
    """CPU-only: build the model at vocab_size=32000 and assert param count is in range."""
    cfg = GPTConfigTorch.tiny_150m(vocab_size=32_000)
    assert cfg.attn == "wayfinder_permute"
    assert cfg.n_embd == 768 and cfg.n_layers == 12 and cfg.n_heads == 12

    model = GPTTorch(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000_000 <= n_params <= 200_000_000, (
        f"tiny_150m produced {n_params/1e6:.1f}M params; expected 100M–200M"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_single_training_step_150m_butterfly() -> None:
    """CUDA: forward → backward → step on a tiny-vocab tiny_150m run."""
    cfg = GPTConfigTorch.tiny_150m(vocab_size=1024)
    model = GPTTorch(cfg).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, 1024, (2, 128), device="cuda")
    y = torch.randint(0, 1024, (2, 128), device="cuda")

    opt.zero_grad(set_to_none=True)
    out = model(x, y)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None and torch.isfinite(lm_head_grad).all()

    opt.step()
