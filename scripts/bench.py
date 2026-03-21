#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from statistics import median

import torch

from bna.model import GPT, GPTConfig
from bna.utils import auto_device, format_bytes, peak_memory_bytes, reset_peak_memory_stats


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def bench_one(model: GPT, idx: torch.Tensor, iters: int, device: torch.device) -> dict:
    # Warmup
    for _ in range(5):
        _ = model(idx)["logits"]
    sync(device)

    times = []
    reset_peak_memory_stats(device)
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(idx)["logits"]
        sync(device)
        times.append(time.perf_counter() - t0)
    mem = peak_memory_bytes(device)
    return {
        "median_ms": 1000.0 * median(times),
        "tokens_per_s": float(idx.numel() / median(times)),
        "peak_mem": mem,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark dense vs HCSA forward pass")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--embd", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--cycle", type=str, default="random", choices=["random", "greedy", "online_insertion"])
    args = p.parse_args()

    device = auto_device(args.device)

    print(f"device={device}")
    print(f"embd={args.embd} layers={args.layers} heads={args.heads}")
    print()

    for T in args.seq_lens:
        idx = torch.randint(0, args.vocab, (args.batch, T), device=device)

        dense_cfg = GPTConfig(
            vocab_size=args.vocab,
            seq_len=T,
            n_layers=args.layers,
            n_heads=args.heads,
            n_embd=args.embd,
            attn="dense",
        )
        hcsa_cfg = GPTConfig(
            vocab_size=args.vocab,
            seq_len=T,
            n_layers=args.layers,
            n_heads=args.heads,
            n_embd=args.embd,
            attn="hcsa",
            cycle=args.cycle,
            window=args.window,
            landmark_stride=None if args.landmark_stride <= 0 else args.landmark_stride,
        )

        dense = GPT(dense_cfg).to(device).eval()
        hcsa = GPT(hcsa_cfg).to(device).eval()

        with torch.no_grad():
            d = bench_one(dense, idx, args.iters, device)
            h = bench_one(hcsa, idx, args.iters, device)

        print(f"T={T:4d} | dense {d['median_ms']:.2f} ms ({d['tokens_per_s']:.0f} tok/s) peak {format_bytes(d['peak_mem'])}")
        print(f"      | hcsa  {h['median_ms']:.2f} ms ({h['tokens_per_s']:.0f} tok/s) peak {format_bytes(h['peak_mem'])}")
        print()


if __name__ == "__main__":
    main()
