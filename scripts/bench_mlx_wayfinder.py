#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx

from bna.compiler import compile_graph_spec
from bna.mlx.attention import WayfinderAttentionMLX
from bna.mlx.metrics import largest_intermediate_bytes
from bna.mlx.model import DenseCausalAttentionMLX, GPTConfigMLX, GPTMLX


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return
    mx.metal.reset_peak_memory()


def _get_peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0


def _sync(*arrays: mx.array) -> None:
    if arrays:
        mx.eval(*arrays)


def _bench_attention(
    mode: str,
    *,
    batch: int,
    heads: int,
    seq_len: int,
    embd: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    warmup: int,
    iters: int,
    compiled_graph_dir: str | None,
) -> Dict[str, Any]:
    if mode == "dense":
        attn = DenseCausalAttentionMLX(embd, heads, dropout=0.0)
    elif mode == "wayfinder_sparse":
        attn = WayfinderAttentionMLX(
            embd,
            heads,
            window=window,
            landmark_stride=landmark_stride,
            num_cycles=num_cycles,
            strategy=strategy,
            path="sparse",
            dropout=0.0,
            compiled_graph_dir=compiled_graph_dir,
        )
    elif mode == "wayfinder_permute":
        attn = WayfinderAttentionMLX(
            embd,
            heads,
            window=window,
            landmark_stride=landmark_stride,
            num_cycles=num_cycles,
            strategy=strategy,
            path="permute",
            dropout=0.0,
            compiled_graph_dir=compiled_graph_dir,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x = mx.random.normal((batch, seq_len, embd), dtype=mx.float16)

    # Warmup (also builds cache for static strategies)
    graph_build_ms_first = 0.0
    for wi in range(warmup):
        if mode != "dense" and wi == 0:
            y, dbg = attn(x, return_debug=True)
            _sync(y)
            graph_build_ms_first = float(dbg["profile"].get("graph_build_ms", 0.0))
        else:
            y = attn(x)
            _sync(y)

    _reset_peak_memory()

    total_s: list[float] = []
    graph_ms: list[float] = []
    permute_ms: list[float] = []
    attn_ms: list[float] = []
    degree = 0

    for _ in range(iters):
        t0 = time.perf_counter()
        if mode == "dense":
            y = attn(x)
            _sync(y)
            elapsed = time.perf_counter() - t0
            total_s.append(elapsed)
            graph_ms.append(0.0)
            permute_ms.append(0.0)
            attn_ms.append(_to_ms(elapsed))
            degree = seq_len
        else:
            y, dbg = attn(x, return_debug=True)
            _sync(y)
            elapsed = time.perf_counter() - t0
            total_s.append(elapsed)
            prof = dbg["profile"]
            graph_ms.append(float(prof.get("graph_build_ms", 0.0)))
            permute_ms.append(float(prof.get("permute_ms", 0.0)))
            attn_ms.append(float(prof.get("attention_ms", _to_ms(elapsed))))
            degree = int(prof.get("max_degree", degree))

    median_s = _median(total_s)
    median_ms = _to_ms(median_s)
    tok_s = float((batch * seq_len) / max(median_s, 1e-12))

    mem_peak = _get_peak_memory()
    mem_proxy = largest_intermediate_bytes(
        B=batch,
        H=heads,
        T=seq_len,
        D=degree,
        dh=embd // heads,
        path="dense" if mode == "dense" else "sparse",
    )

    return {
        "mode": mode,
        "seq_len": seq_len,
        "batch": batch,
        "heads": heads,
        "embd": embd,
        "tokens_per_sec": tok_s,
        "latency_ms": median_ms,
        "graph_build_ms_first": graph_build_ms_first,
        "graph_build_ms_cached": _median(graph_ms),
        "graph_build_ms": _median(graph_ms),
        "permute_ms": _median(permute_ms),
        "attention_ms": _median(attn_ms),
        "total_ms": median_ms,
        "peak_memory_bytes": mem_peak,
        "largest_intermediate_bytes": int(mem_proxy.get("largest", 0)),
        "memory_proxy": mem_proxy,
    }


def _bench_block(
    mode: str,
    *,
    batch: int,
    seq_len: int,
    embd: int,
    heads: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    warmup: int,
    iters: int,
    compiled_graph_dir: str | None,
) -> Dict[str, Any]:
    cfg = GPTConfigMLX(
        vocab_size=256,
        seq_len=seq_len,
        n_layers=1,
        n_heads=heads,
        n_embd=embd,
        dropout=0.0,
        attn=mode,  # type: ignore[arg-type]
        strategy=strategy,
        window=window,
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        compiled_graph_dir=compiled_graph_dir,
    )
    model = GPTMLX(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, shape=(batch, seq_len), dtype=mx.int32)

    for _ in range(warmup):
        out = model(idx)
        _sync(out["logits"])

    _reset_peak_memory()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model(idx)
        _sync(out["logits"])
        times.append(time.perf_counter() - t0)

    med = _median(times)
    return {
        "mode": mode,
        "seq_len": seq_len,
        "block_latency_ms": _to_ms(med),
        "block_tokens_per_sec": float((batch * seq_len) / max(med, 1e-12)),
        "block_peak_memory_bytes": _get_peak_memory(),
    }


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{n}B"


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark MLX Wayfinder attention paths")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--embd", type=int, default=512)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument(
        "--strategy", type=str, default="random", choices=["random", "greedy", "online_insertion"]
    )
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-block", action="store_true", help="Skip end-to-end 1-block benchmark")
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/mlx/bench_results.json"))
    args = p.parse_args()

    modes = ["dense", "wayfinder_sparse", "wayfinder_permute"]
    lm_stride = None if args.landmark_stride <= 0 else int(args.landmark_stride)

    rows: list[Dict[str, Any]] = []
    block_rows: list[Dict[str, Any]] = []

    for T in args.seq_lens:
        compiled_graph_dir = None
        if args.graph_spec is not None:
            compiled = compile_graph_spec(
                args.graph_spec,
                T=T,
                H=args.heads,
                out_root=args.graph_cache_root,
            )
            compiled_graph_dir = str(compiled["artifact"]["artifact_dir"])
        for mode in modes:
            row = _bench_attention(
                mode,
                batch=args.batch,
                heads=args.heads,
                seq_len=T,
                embd=args.embd,
                window=args.window,
                landmark_stride=lm_stride,
                num_cycles=args.num_cycles,
                strategy=args.strategy,
                warmup=args.warmup,
                iters=args.iters,
                compiled_graph_dir=compiled_graph_dir,
            )
            rows.append(row)

            if not args.skip_block:
                b = _bench_block(
                    mode,
                    batch=args.batch,
                    seq_len=T,
                    embd=args.embd,
                    heads=args.heads,
                    window=args.window,
                    landmark_stride=lm_stride,
                    num_cycles=args.num_cycles,
                    strategy=args.strategy,
                    warmup=max(2, args.warmup // 2),
                    iters=max(4, args.iters // 2),
                    compiled_graph_dir=compiled_graph_dir,
                )
                block_rows.append(b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "seq_lens": args.seq_lens,
            "batch": args.batch,
            "heads": args.heads,
            "embd": args.embd,
            "window": args.window,
            "landmark_stride": lm_stride,
            "num_cycles": args.num_cycles,
            "strategy": args.strategy,
            "warmup": args.warmup,
            "iters": args.iters,
            "graph_spec": None if args.graph_spec is None else str(args.graph_spec),
        },
        "attention": rows,
        "block": block_rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("MLX Wayfinder attention benchmark")
    print(
        "mode                T      tok/s      total_ms  graph_1st  graph_cached  permute_ms"
        "  attention_ms  peak_mem"
    )
    for row in rows:
        print(
            f"{row['mode']:<18} {row['seq_len']:>5d} "
            f"{row['tokens_per_sec']:>10.1f} {row['total_ms']:>11.3f} "
            f"{row['graph_build_ms_first']:>10.3f} {row['graph_build_ms_cached']:>12.3f} "
            f"{row['permute_ms']:>11.3f} "
            f"{row['attention_ms']:>12.3f} {_format_bytes(row['peak_memory_bytes']):>10}"
        )

    if block_rows:
        print("\n1-block forward benchmark")
        print("mode                T      tok/s      block_ms   peak_mem")
        for row in block_rows:
            print(
                f"{row['mode']:<18} {row['seq_len']:>5d} "
                f"{row['block_tokens_per_sec']:>10.1f} {row['block_latency_ms']:>10.3f} "
                f"{_format_bytes(row['block_peak_memory_bytes']):>10}"
            )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
