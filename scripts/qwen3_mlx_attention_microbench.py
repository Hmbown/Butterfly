#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

from hcsa.mlx.model import DenseCausalAttentionMLX
from hcsa.mlx.attention import WayfinderAttentionMLX


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, int]:
    for _ in range(max(1, warmup)):
        y = fn()
        mx.eval(y)

    _reset_peak_memory()
    times = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - t0)

    return float(statistics.median(times)), _peak_memory()


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-shaped MLX attention microbench")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--embd", type=int, default=4096)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dense = DenseCausalAttentionMLX(args.embd, args.heads, dropout=0.0)
    permute = WayfinderAttentionMLX(
        args.embd,
        args.heads,
        window=args.window,
        landmark_stride=None if args.landmark_stride <= 0 else args.landmark_stride,
        num_cycles=args.num_cycles,
        strategy="random",
        path="permute",
        dropout=0.0,
    )

    x = mx.random.normal((args.batch, args.seq_len, args.embd), dtype=mx.float16)

    dense_s, dense_mem = _bench(lambda: dense(x), warmup=args.warmup, iters=args.iters)
    permute_s, permute_mem = _bench(lambda: permute(x), warmup=args.warmup, iters=args.iters)

    result: Dict[str, Any] = {
        "config": {
            **vars(args),
            "out": None if args.out is None else str(args.out),
        },
        "dense": {
            "tok_s": float((args.batch * args.seq_len) / max(dense_s, 1e-12)),
            "latency_ms": dense_s * 1000.0,
            "peak_memory_bytes": dense_mem,
        },
        "wayfinder_permute": {
            "tok_s": float((args.batch * args.seq_len) / max(permute_s, 1e-12)),
            "latency_ms": permute_s * 1000.0,
            "peak_memory_bytes": permute_mem,
        },
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
