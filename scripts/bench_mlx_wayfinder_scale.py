#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx

from bna.compiler import compile_graph_spec
from bna.mlx.metrics import largest_intermediate_bytes
from bna.mlx.model import DenseCausalAttentionMLX, GPTConfigMLX, GPTMLX
from bna.mlx.attention import WayfinderAttentionMLX


def _sync(*arrays: mx.array) -> None:
    if arrays:
        mx.eval(*arrays)


def _to_ms(seconds: float) -> float:
    return float(seconds * 1000.0)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _pstdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{n}B"


def _build_attn(
    mode: str,
    *,
    embd: int,
    heads: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    regular_num_clusters: int,
    compiled_graph_dir: str | None,
):
    if mode == "dense":
        return DenseCausalAttentionMLX(embd, heads, dropout=0.0)

    path = "sparse" if mode == "wayfinder_sparse" else "permute"
    return WayfinderAttentionMLX(
        embd,
        heads,
        window=window,
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        strategy=strategy,
        regular_num_clusters=regular_num_clusters,
        path=path,
        dropout=0.0,
        compiled_graph_dir=compiled_graph_dir,
    )


def _bench_attention_mode(
    mode: str,
    *,
    B: int,
    H: int,
    T: int,
    C: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    regular_num_clusters: int,
    warmup: int,
    iters: int,
    compiled_graph_dir: str | None,
) -> Dict[str, Any]:
    attn = _build_attn(
        mode,
        embd=C,
        heads=H,
        window=window,
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        strategy=strategy,
        regular_num_clusters=regular_num_clusters,
        compiled_graph_dir=compiled_graph_dir,
    )
    if hasattr(attn, "eval"):
        attn.eval()
    x = mx.random.normal((B, T, C), dtype=mx.float16)

    graph_first = 0.0
    for wi in range(max(1, warmup)):
        if mode == "dense":
            y = attn(x)
            _sync(y)
            continue
        y, dbg = attn(x, return_debug=True)
        _sync(y)
        if wi == 0:
            graph_first = float(dbg["profile"].get("graph_build_ms", 0.0))

    _reset_peak_memory()
    lat_s: list[float] = []
    attention_ms: list[float] = []
    permute_ms: list[float] = []
    graph_cached_ms: list[float] = []
    cache_hits = 0
    cache_total = 0
    degree = T

    for _ in range(iters):
        t0 = time.perf_counter()
        if mode == "dense":
            y = attn(x)
            _sync(y)
            dt = time.perf_counter() - t0
            lat_s.append(dt)
            attention_ms.append(_to_ms(dt))
            permute_ms.append(0.0)
            graph_cached_ms.append(0.0)
            degree = T
            continue

        y, dbg = attn(x, return_debug=True)
        _sync(y)
        dt = time.perf_counter() - t0
        lat_s.append(dt)
        prof = dbg["profile"]
        attention_ms.append(float(prof.get("attention_ms", _to_ms(dt))))
        permute_ms.append(float(prof.get("permute_ms", 0.0)))
        graph_cached_ms.append(float(prof.get("graph_build_ms", 0.0)))
        cache_hits += int(bool(prof.get("cache_hit", False)))
        cache_total += 1
        degree = int(prof.get("max_degree", degree))

    lat_med_s = _median(lat_s)
    tok_s_med = float((B * T) / max(lat_med_s, 1e-12))
    peak_mem = _peak_memory()

    cache_persistent = 0
    if isinstance(attn, WayfinderAttentionMLX):
        cache_persistent = int(attn.cache_persistent_bytes())

    mem_proxy = largest_intermediate_bytes(
        B=B,
        H=H,
        T=T,
        D=degree,
        dh=C // H,
        path="dense" if mode == "dense" else "sparse",
    )
    step_intermediate = max(0, peak_mem - cache_persistent)

    return {
        "mode": mode,
        "batch": B,
        "heads": H,
        "seq_len": T,
        "embd": C,
        "tokens_per_sec": tok_s_med,
        "tokens_per_sec_mean": float(_mean([float((B * T) / max(s, 1e-12)) for s in lat_s])),
        "tokens_per_sec_std": float(_pstdev([float((B * T) / max(s, 1e-12)) for s in lat_s])),
        "total_ms": _to_ms(lat_med_s),
        "attention_ms": _median(attention_ms),
        "permute_ms": _median(permute_ms),
        "graph_build_ms_first": graph_first,
        "graph_build_ms_cached": _median(graph_cached_ms),
        "cache_hit_rate": float(cache_hits / max(cache_total, 1)),
        "peak_memory_bytes": peak_mem,
        "persistent_cache_bytes": cache_persistent,
        "step_intermediate_bytes": step_intermediate,
        "largest_intermediate_bytes_proxy": int(mem_proxy.get("largest", 0)),
        "memory_proxy": mem_proxy,
    }


def _bench_block_mode(
    mode: str,
    *,
    B: int,
    H: int,
    T: int,
    C: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    regular_num_clusters: int,
    warmup: int,
    iters: int,
    compiled_graph_dir: str | None,
) -> Dict[str, Any]:
    attn_mode = (
        "dense"
        if mode == "dense"
        else ("wayfinder_sparse" if mode == "wayfinder_sparse" else "wayfinder_permute")
    )
    cfg = GPTConfigMLX(
        vocab_size=256,
        seq_len=T,
        n_layers=1,
        n_heads=H,
        n_embd=C,
        dropout=0.0,
        attn=attn_mode,
        strategy=strategy,
        window=window,
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        regular_num_clusters=regular_num_clusters,
        compiled_graph_dir=compiled_graph_dir,
    )
    model = GPTMLX(cfg)
    if hasattr(model, "eval"):
        model.eval()
    idx = mx.random.randint(0, cfg.vocab_size, shape=(B, T), dtype=mx.int32)

    for _ in range(max(1, warmup)):
        out = model(idx)
        _sync(out["logits"])

    _reset_peak_memory()
    lat_s: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model(idx)
        _sync(out["logits"])
        lat_s.append(time.perf_counter() - t0)

    med_s = _median(lat_s)
    peak_mem = _peak_memory()

    cache_persistent = 0
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if isinstance(attn, WayfinderAttentionMLX):
            cache_persistent += attn.cache_persistent_bytes()

    return {
        "mode": mode,
        "seq_len": T,
        "block_tokens_per_sec": float((B * T) / max(med_s, 1e-12)),
        "block_latency_ms": _to_ms(med_s),
        "block_peak_memory_bytes": peak_mem,
        "block_persistent_cache_bytes": int(cache_persistent),
        "block_step_intermediate_bytes": int(max(0, peak_mem - cache_persistent)),
    }


def _make_readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MLX Wayfinder Scaling Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- command: `{payload['command']}`")
    lines.append("")
    lines.append("## Attention Throughput & Memory")
    lines.append("")
    lines.append(
        "| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["attention"]:
        lines.append(
            "| {mode} | {T} | {tok:.1f} | {g0:.4f} | {g1:.4f} | {a:.4f} | {p:.4f} | {m} | {pc} | {si} |".format(
                mode=row["mode"],
                T=row["seq_len"],
                tok=row["tokens_per_sec"],
                g0=row["graph_build_ms_first"],
                g1=row["graph_build_ms_cached"],
                a=row["attention_ms"],
                p=row["permute_ms"],
                m=_fmt_bytes(int(row["peak_memory_bytes"])),
                pc=_fmt_bytes(int(row["persistent_cache_bytes"])),
                si=_fmt_bytes(int(row["step_intermediate_bytes"])),
            )
        )

    lines.append("")
    lines.append("## 1-Block End-to-End Memory")
    lines.append("")
    lines.append(
        "| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["block"]:
        lines.append(
            "| {mode} | {T} | {tok:.1f} | {lat:.4f} | {m} | {pc} | {si} |".format(
                mode=row["mode"],
                T=row["seq_len"],
                tok=row["block_tokens_per_sec"],
                lat=row["block_latency_ms"],
                m=_fmt_bytes(int(row["block_peak_memory_bytes"])),
                pc=_fmt_bytes(int(row["block_persistent_cache_bytes"])),
                si=_fmt_bytes(int(row["block_step_intermediate_bytes"])),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- `graph_build_ms_cached` should be near zero on cache hits.")
    lines.append(
        "- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Scale benchmark for MLX Wayfinder dense/sparse/permute")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "greedy", "online_insertion", "regular_partition"],
    )
    p.add_argument("--regular-num-clusters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-4096", action="store_true")
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    seq_lens = [t for t in args.seq_lens if (t != 4096 or not args.skip_4096)]
    modes = ["dense", "wayfinder_sparse", "wayfinder_permute"]

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("benchmarks/mlx") / f"scale_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lm_stride = None if args.landmark_stride <= 0 else int(args.landmark_stride)

    attention_rows: list[Dict[str, Any]] = []
    block_rows: list[Dict[str, Any]] = []

    for T in seq_lens:
        compiled_dir: str | None = None
        if args.graph_spec is not None:
            comp = compile_graph_spec(
                args.graph_spec,
                T=T,
                H=args.heads,
                out_root=args.graph_cache_root,
            )
            compiled_dir = str(comp["artifact"]["artifact_dir"])

        for mode in modes:
            attn_row = _bench_attention_mode(
                mode,
                B=args.batch,
                H=args.heads,
                T=T,
                C=args.embd,
                window=args.window,
                landmark_stride=lm_stride,
                num_cycles=args.num_cycles,
                strategy=args.strategy,
                regular_num_clusters=args.regular_num_clusters,
                warmup=args.warmup,
                iters=args.iters,
                compiled_graph_dir=compiled_dir,
            )
            attention_rows.append(attn_row)

            block_row = _bench_block_mode(
                mode,
                B=args.batch,
                H=args.heads,
                T=T,
                C=args.embd,
                window=args.window,
                landmark_stride=lm_stride,
                num_cycles=args.num_cycles,
                strategy=args.strategy,
                regular_num_clusters=args.regular_num_clusters,
                warmup=max(1, args.warmup // 2),
                iters=max(2, args.iters // 2),
                compiled_graph_dir=compiled_dir,
            )
            block_rows.append(block_row)

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "config": {
            "seq_lens": seq_lens,
            "batch": args.batch,
            "heads": args.heads,
            "embd": args.embd,
            "window": args.window,
            "landmark_stride": lm_stride,
            "num_cycles": args.num_cycles,
            "strategy": args.strategy,
            "regular_num_clusters": int(max(1, args.regular_num_clusters)),
            "warmup": args.warmup,
            "iters": args.iters,
            "graph_spec": None if args.graph_spec is None else str(args.graph_spec),
        },
        "attention": attention_rows,
        "block": block_rows,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_make_readme(payload), encoding="utf-8")

    print(f"Wrote: {out_dir / 'results.json'}")
    print(f"Wrote: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
