#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from hcsa.compiler import compile_graph_spec
from hcsa.torch.attention_dense import DenseCausalAttentionTorch
from hcsa.torch.attention_hha_sparse import WayfinderAttentionTorch
from hcsa.torch.bench_utils import largest_intermediate_bytes, sync_device


def _to_ms(seconds: float) -> float:
    return float(seconds * 1000.0)


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _pstdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{n}B"


def _parse_seq_lens(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--seq-lens produced an empty list")
    return vals


def _build_attn(
    mode: str,
    *,
    embd: int,
    heads: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
    compiled_graph_dir: str | None,
) -> torch.nn.Module:
    if mode == "dense":
        return DenseCausalAttentionTorch(embd, heads, dropout=0.0)

    path = "sparse" if mode == "hha_sparse" else "permute"
    return WayfinderAttentionTorch(
        embd,
        heads,
        window=window,
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        strategy=strategy,
        path=path,
        dropout=0.0,
        compiled_graph_dir=compiled_graph_dir,
    )


def _bench_attention_mode(
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    H: int,
    T: int,
    C: int,
    window: int,
    landmark_stride: int | None,
    num_cycles: int,
    strategy: str,
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
        compiled_graph_dir=compiled_graph_dir,
    ).to(device=device, dtype=dtype)
    attn.eval()

    x = torch.randn(B, T, C, device=device, dtype=dtype)

    graph_first = 0.0
    for wi in range(max(1, warmup)):
        t0 = time.perf_counter()
        if mode == "dense":
            y = attn(x)
            sync_device(device)
            _ = y
            continue

        y, dbg = attn(x, return_debug=True)
        sync_device(device)
        _ = y
        if wi == 0:
            graph_first = float(dbg["profile"].get("graph_build_ms", 0.0))
        _ = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
            sync_device(device)
            lat_s.append(time.perf_counter() - t0)
            attention_ms.append(_to_ms(lat_s[-1]))
            permute_ms.append(0.0)
            graph_cached_ms.append(0.0)
            degree = T
            _ = y
            continue

        y, dbg = attn(x, return_debug=True)
        sync_device(device)
        dt = time.perf_counter() - t0
        lat_s.append(dt)
        prof = dbg["profile"]
        attention_ms.append(float(prof.get("attention_ms", _to_ms(dt))))
        permute_ms.append(float(prof.get("permute_ms", 0.0)))
        graph_cached_ms.append(float(prof.get("graph_build_ms", 0.0)))
        cache_hits += int(bool(prof.get("cache_hit", False)))
        cache_total += 1
        degree = int(prof.get("max_degree", degree))
        _ = y

    lat_med_s = _median(lat_s)
    tok_s_med = float((B * T) / max(lat_med_s, 1e-12))

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))

    cache_persistent = 0
    if isinstance(attn, WayfinderAttentionTorch):
        cache_persistent = int(attn.cache_persistent_bytes())

    mem_proxy = largest_intermediate_bytes(
        B=B,
        H=H,
        T=T,
        D=degree,
        dh=C // H,
        path="dense" if mode == "dense" else "sparse",
        dtype_bytes=torch.finfo(dtype).bits // 8,
    )
    step_intermediate = max(0, peak_mem - cache_persistent)

    return {
        "mode": mode,
        "batch": B,
        "heads": H,
        "seq_len": T,
        "embd": C,
        "dtype": str(dtype).replace("torch.", ""),
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
        "compiled_graph_dir": compiled_graph_dir,
    }


def _make_readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Torch Wayfinder CUDA Scaling Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- command: `{payload['command']}`")
    lines.append(f"- device: `{payload['device']}`")
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
    lines.append("## Notes")
    lines.append("")
    lines.append("- `graph_build_ms_cached` should be near zero for static/random strategy on cache hits.")
    lines.append("- `hha_sparse` is the correctness/reference path; `hha_permute` is the fast path.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Torch CUDA scaling benchmark for Wayfinder/HHA attention")
    p.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--embd", type=int, default=2048)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--strategy", type=str, default="random", choices=["random", "greedy", "online_insertion"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--compile-out-root", type=Path, default=Path(".cache/wayfinder_torch"))
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    seq_lens = _parse_seq_lens(args.seq_lens)
    dtype = getattr(torch, args.dtype)

    if args.out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("benchmarks/cuda") / f"scale_{stamp}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled_dirs: dict[int, str | None] = {t: None for t in seq_lens}
    if args.graph_spec is not None:
        for t in seq_lens:
            result = compile_graph_spec(
                args.graph_spec,
                T=t,
                H=args.heads,
                out_root=args.compile_out_root,
            )
            compiled_dirs[t] = str(result["artifact"]["artifact_dir"])

    rows: list[Dict[str, Any]] = []
    modes = ["dense", "hha_sparse", "hha_permute"]
    for t in seq_lens:
        for mode in modes:
            row = _bench_attention_mode(
                mode,
                device=device,
                dtype=dtype,
                B=args.batch,
                H=args.heads,
                T=t,
                C=args.embd,
                window=args.window,
                landmark_stride=None if args.landmark_stride <= 0 else args.landmark_stride,
                num_cycles=args.num_cycles,
                strategy=args.strategy,
                warmup=args.warmup,
                iters=args.iters,
                compiled_graph_dir=compiled_dirs[t],
            )
            rows.append(row)

    payload: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "device": str(device),
        "config": {
            "seq_lens": seq_lens,
            "batch": args.batch,
            "heads": args.heads,
            "embd": args.embd,
            "window": args.window,
            "landmark_stride": args.landmark_stride,
            "num_cycles": args.num_cycles,
            "strategy": args.strategy,
            "warmup": args.warmup,
            "iters": args.iters,
            "dtype": args.dtype,
            "graph_spec": None if args.graph_spec is None else str(args.graph_spec),
            "compile_out_root": str(args.compile_out_root),
        },
        "attention": rows,
        "artifacts": {
            "out_dir": str(out_dir.resolve()),
            "compiled_graph_dirs": {str(k): v for k, v in compiled_dirs.items()},
        },
    }

    json_path = out_dir / "results.json"
    readme_path = out_dir / "README.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(_make_readme(payload) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
