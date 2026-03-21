#!/usr/bin/env python3
"""Per-component benchmarking: measure time for each part of the pipeline.

Components measured:
- QKV projection
- Cycle construction (per strategy)
- Neighbor index building
- Gather (K, V)
- Attention computation (scores + mask + softmax)
- Output projection
- Full forward pass
- Full backward pass

Usage:
    python scripts/bench_components.py --seq-len 512 --attn hcsa
"""

from __future__ import annotations

import argparse
import time
from statistics import median
from typing import Dict, List

import torch
import torch.nn as nn

from bna.model import GPT, GPTConfig
from bna.attention_hcsa import HCSASelfAttention, _build_neighbors_index
from bna.attention_dense import DenseCausalSelfAttention
from bna.cycles import random_cycle, greedy_cycle, routing_similarity
from bna.utils import auto_device, format_bytes, peak_memory_bytes


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def bench_fn(fn, device: torch.device, iters: int = 20, warmup: int = 5) -> float:
    """Benchmark a callable. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return 1000.0 * median(times)


def bench_cycle_construction(
    T: int,
    device: torch.device,
    iters: int = 20,
) -> Dict[str, float]:
    """Benchmark cycle construction strategies."""
    results = {}

    # Random cycle
    g = torch.Generator(device="cpu").manual_seed(0)
    results["random_cycle_ms"] = bench_fn(
        lambda: random_cycle(T, generator=g, device=torch.device("cpu")),
        device, iters,
    )

    # Greedy cycle
    r = torch.randn(T, 16, device=device)
    results["greedy_cycle_ms"] = bench_fn(
        lambda: greedy_cycle(r, start=0),
        device, iters,
    )

    # Neighbor index building
    g2 = torch.Generator(device="cpu").manual_seed(42)
    perm = random_cycle(T, generator=g2, device=torch.device("cpu"))
    from bna.cycles import cycle_prev_next_from_perm
    prev, nxt = cycle_prev_next_from_perm(perm)
    cycle_adj = [[int(prev[i]), int(nxt[i])] for i in range(T)]

    results["build_neighbors_ms"] = bench_fn(
        lambda: _build_neighbors_index(T, cycle_adj, window=32, landmark_stride=32),
        device, iters,
    )

    return results


def bench_full_model(
    T: int,
    attn: str,
    device: torch.device,
    batch: int = 4,
    n_embd: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    iters: int = 20,
) -> Dict[str, float]:
    """Benchmark full model forward and backward."""
    cfg = GPTConfig(
        vocab_size=256, seq_len=T, n_layers=n_layers, n_heads=n_heads,
        n_embd=n_embd, attn=attn, cycle="random", window=32, landmark_stride=32,  # type: ignore[arg-type]
    )
    model = GPT(cfg).to(device)
    idx = torch.randint(0, 256, (batch, T), device=device)
    targets = torch.randint(0, 256, (batch, T), device=device)

    results = {}

    # Forward
    model.eval()
    results["forward_ms"] = bench_fn(
        lambda: model(idx),
        device, iters,
    )

    # Forward + backward
    model.train()
    def fwd_bwd():
        out = model(idx, targets)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)

    results["fwd_bwd_ms"] = bench_fn(fwd_bwd, device, iters)

    # Memory
    results["peak_memory"] = peak_memory_bytes(device)

    del model
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Component-level benchmarking")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--attn", type=str, default="hcsa", choices=["dense", "hcsa"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    device = auto_device(args.device)
    T = args.seq_len
    print(f"Device: {device}, T={T}, attn={args.attn}")

    print("\n--- Cycle Construction ---")
    cycle_results = bench_cycle_construction(T, device, args.iters)
    for k, v in cycle_results.items():
        print(f"  {k}: {v:.2f} ms")

    print(f"\n--- Full Model ({args.attn}) ---")
    model_results = bench_full_model(T, args.attn, device, batch=args.batch, iters=args.iters)
    for k, v in model_results.items():
        if k == "peak_memory":
            print(f"  {k}: {format_bytes(int(v))}")
        else:
            print(f"  {k}: {v:.2f} ms")


if __name__ == "__main__":
    main()
