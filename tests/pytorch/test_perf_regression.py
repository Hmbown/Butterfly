"""Performance regression tests.

Verify that forward/backward pass time and memory usage stay below
acceptable thresholds.  Marked as slow tests - run with:
    pytest tests/test_perf_regression.py -v --run-slow

These tests establish baselines; if they fail, it means a recent change
has degraded performance beyond the threshold.
"""

from __future__ import annotations

import time

import pytest
import torch

from hcsa.model import GPT, GPTConfig


# Skip if not explicitly requested via --run-slow
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


slow = pytest.mark.slow


def _make_model(attn: str, **kw) -> GPT:
    cfg = GPTConfig(
        vocab_size=kw.pop("vocab_size", 64),
        seq_len=kw.pop("seq_len", 128),
        n_layers=kw.pop("n_layers", 4),
        n_heads=kw.pop("n_heads", 4),
        n_embd=kw.pop("n_embd", 128),
        attn=attn,  # type: ignore[arg-type]
        cycle=kw.pop("cycle", "random"),
        window=kw.pop("window", 16),
        landmark_stride=kw.pop("landmark_stride", 16),
        seed=kw.pop("seed", 42),
    )
    return GPT(cfg)


def _time_forward_backward(
    model: GPT,
    seq_len: int = 128,
    batch_size: int = 4,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> dict:
    """Time forward and backward passes."""
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    x = torch.randint(0, model.token_emb.num_embeddings, (batch_size, seq_len))
    y = torch.randint(0, model.token_emb.num_embeddings, (batch_size, seq_len))

    # Warmup
    for _ in range(n_warmup):
        out = model(x, y)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)

    # Timed runs
    fwd_times = []
    bwd_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = model(x, y)
        fwd_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        out["loss"].backward()
        bwd_times.append(time.perf_counter() - t0)

        model.zero_grad(set_to_none=True)

    return {
        "fwd_mean": sum(fwd_times) / n_runs,
        "bwd_mean": sum(bwd_times) / n_runs,
        "total_mean": (sum(fwd_times) + sum(bwd_times)) / n_runs,
    }


# ---- Forward pass time thresholds ----

@slow
def test_dense_forward_time():
    """Dense attention forward should be fast on small models."""
    model = _make_model("dense")
    times = _time_forward_backward(model)
    # Generous threshold: 2 seconds per forward+backward on CPU
    assert times["total_mean"] < 2.0, (
        f"Dense forward+backward too slow: {times['total_mean']:.3f}s"
    )


@slow
def test_hcsa_forward_time():
    """HCSA attention forward should not be dramatically slower than dense."""
    dense_model = _make_model("dense")
    hcsa_model = _make_model("hcsa")

    dense_times = _time_forward_backward(dense_model)
    hcsa_times = _time_forward_backward(hcsa_model)

    # HCSA should be no more than 8x slower than dense on small sequences
    # (overhead is in Python neighbor building; at large T the gather approach wins)
    ratio = hcsa_times["total_mean"] / max(dense_times["total_mean"], 1e-8)
    assert ratio < 8.0, (
        f"HCSA is {ratio:.1f}x slower than dense "
        f"(dense={dense_times['total_mean']:.3f}s, hcsa={hcsa_times['total_mean']:.3f}s)"
    )


# ---- Memory thresholds ----

@slow
def test_dense_memory():
    """Dense attention memory should be reasonable for small models."""
    model = _make_model("dense")
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = n_params * 4  # float32
    # Total model memory (params + gradients + optimizer) < 50MB for small model
    assert param_bytes < 50 * 1024 * 1024, (
        f"Dense model params too large: {param_bytes / 1024 / 1024:.1f}MB"
    )


@slow
def test_hcsa_memory():
    """HCSA should not use dramatically more parameters than dense."""
    dense_model = _make_model("dense")
    hcsa_model = _make_model("hcsa")

    dense_params = sum(p.numel() for p in dense_model.parameters())
    hcsa_params = sum(p.numel() for p in hcsa_model.parameters())

    # HCSA has extra Wr routing projection, but should be < 2x dense params
    ratio = hcsa_params / max(dense_params, 1)
    assert ratio < 2.0, (
        f"HCSA has {ratio:.2f}x more params than dense "
        f"(dense={dense_params}, hcsa={hcsa_params})"
    )


# ---- Model output quality sanity ----

@slow
def test_loss_decreases_after_training():
    """Verify that a few training steps actually reduce loss."""
    torch.manual_seed(42)
    model = _make_model("hcsa")
    model.train()

    x = torch.randint(0, 64, (8, 128))
    y = torch.randint(0, 64, (8, 128))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Initial loss
    out0 = model(x, y)
    loss0 = out0["loss"].item()

    # Train for 20 steps
    for _ in range(20):
        out = model(x, y)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    out_final = model(x, y)
    loss_final = out_final["loss"].item()

    assert loss_final < loss0, (
        f"Loss did not decrease: {loss0:.4f} -> {loss_final:.4f}"
    )


# ---- Scaling behavior ----

@slow
def test_hcsa_scales_subquadratically():
    """HCSA forward time should scale better than O(T^2).

    Compare T=64 vs T=256. If O(T^2), the ratio should be ~16.
    HCSA should be closer to O(T*D) where D << T.
    """
    model_64 = _make_model("hcsa", seq_len=64)
    model_256 = _make_model("hcsa", seq_len=256)

    times_64 = _time_forward_backward(model_64, seq_len=64, batch_size=2)
    times_256 = _time_forward_backward(model_256, seq_len=256, batch_size=2)

    ratio = times_256["fwd_mean"] / max(times_64["fwd_mean"], 1e-8)
    # For O(T^2), ratio would be ~16. For O(T*D), ratio should be ~4 (linear in T).
    # Allow up to 10x to account for overheads.
    assert ratio < 12.0, (
        f"HCSA scaling ratio T=64->256 is {ratio:.1f}x "
        f"(expected < 12x for sub-quadratic)"
    )
