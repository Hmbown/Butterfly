#!/usr/bin/env python3
"""Scaling laws: measure how forward pass time scales with sequence length.

Dense should scale as O(T^2), HCSA should scale as O(T*D).
Produces log-log plots with power law fits.

Usage:
    python scripts/scaling_laws.py --seq-lens 128 256 512 1024 2048 --out scaling.png
"""

from __future__ import annotations

import argparse
import time
from statistics import median
from typing import List, Tuple

import torch
import numpy as np

from bna.model import GPT, GPTConfig
from bna.utils import auto_device

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def bench_forward(
    attn: str,
    seq_len: int,
    device: torch.device,
    batch: int = 4,
    n_embd: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    iters: int = 20,
    warmup: int = 5,
    **hcsa_kwargs: int,
) -> float:
    """Return median forward pass time in ms."""
    cfg = GPTConfig(
        vocab_size=256,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_embd,
        attn=attn,  # type: ignore[arg-type]
        cycle=hcsa_kwargs.get("cycle", "random"),
        window=hcsa_kwargs.get("window", 32),
        landmark_stride=hcsa_kwargs.get("landmark_stride", 32),
    )
    model = GPT(cfg).to(device).eval()
    idx = torch.randint(0, 256, (batch, seq_len), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(idx)
        sync(device)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(idx)
            sync(device)
            times.append(time.perf_counter() - t0)

    del model
    return 1000.0 * median(times)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a * x^b in log-log space. Returns (a, b)."""
    log_x = np.log(x)
    log_y = np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return float(np.exp(log_a)), float(b)


def run_scaling_experiment(
    seq_lens: List[int],
    device: torch.device,
    **kwargs: int,
) -> dict:
    """Run scaling experiment for dense and HCSA."""
    results = {"seq_lens": seq_lens, "dense_ms": [], "hcsa_ms": []}

    for T in seq_lens:
        print(f"  T={T:5d} ...", end=" ", flush=True)

        t_dense = bench_forward("dense", T, device, **kwargs)
        t_hcsa = bench_forward("hcsa", T, device, **kwargs)

        results["dense_ms"].append(t_dense)
        results["hcsa_ms"].append(t_hcsa)

        print(f"dense={t_dense:.1f}ms  hcsa={t_hcsa:.1f}ms  ratio={t_dense/max(t_hcsa,1e-6):.2f}x")

    # Power law fits
    x = np.array(seq_lens, dtype=np.float64)
    a_d, b_d = fit_power_law(x, np.array(results["dense_ms"]))
    a_h, b_h = fit_power_law(x, np.array(results["hcsa_ms"]))

    results["dense_power"] = b_d
    results["hcsa_power"] = b_h
    print(f"\nPower law fits:")
    print(f"  Dense: t ~ T^{b_d:.2f} (expected ~2.0)")
    print(f"  HCSA:  t ~ T^{b_h:.2f} (expected ~1.0)")

    return results


def plot_scaling(results: dict, save_path: str | None = None) -> None:
    if plt is None:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array(results["seq_lens"])
    y_dense = np.array(results["dense_ms"])
    y_hcsa = np.array(results["hcsa_ms"])

    ax.loglog(x, y_dense, "o-", label=f"Dense (T^{results['dense_power']:.2f})", color="#EF4444", linewidth=2)
    ax.loglog(x, y_hcsa, "s-", label=f"HCSA (T^{results['hcsa_power']:.2f})", color="#3B82F6", linewidth=2)

    # Reference lines
    x_ref = np.linspace(x[0], x[-1], 100)
    scale = y_dense[0] / x[0] ** 2
    ax.loglog(x_ref, scale * x_ref ** 2, "--", alpha=0.3, color="gray", label="O(T^2) ref")
    scale = y_hcsa[0] / x[0]
    ax.loglog(x_ref, scale * x_ref, ":", alpha=0.3, color="gray", label="O(T) ref")

    ax.set_xlabel("Sequence Length (T)", fontsize=12)
    ax.set_ylabel("Forward Pass Time (ms)", fontsize=12)
    ax.set_title("Scaling Laws: Dense vs HCSA", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--landmark-stride", type=int, default=32)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = auto_device(args.device)
    print(f"Device: {device}")

    results = run_scaling_experiment(
        args.seq_lens, device,
        batch=args.batch, n_embd=args.n_embd, n_layers=args.n_layers,
        n_heads=args.n_heads, iters=args.iters,
        window=args.window, landmark_stride=args.landmark_stride,
    )
    plot_scaling(results, save_path=args.out)


if __name__ == "__main__":
    main()
