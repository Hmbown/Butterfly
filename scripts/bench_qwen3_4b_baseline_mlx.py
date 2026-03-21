#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.qwen_mlx import extract_qkv_from_qwen_attention


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:  # pragma: no cover
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())  # pragma: no cover


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, int]:
    for _ in range(max(1, warmup)):
        y = fn()
        mx.eval(y)

    _reset_peak_memory()
    times: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return _median(times), _peak_memory()


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    warmup: int,
    iters: int,
    dtype_name: str,
) -> Dict[str, Any]:
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    attn = layer0.self_attn

    def baseline_attn_fn():
        q, k, v = extract_qkv_from_qwen_attention(attn, x, cache=None)
        y = scaled_dot_product_attention(q, k, v, cache=None, scale=attn.scale, mask="causal")
        return attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))

    attn_s, attn_mem = _bench(baseline_attn_fn, warmup=warmup, iters=iters)

    def baseline_block_fn():
        return layer0(x, mask="causal", cache=None)

    block_s, block_mem = _bench(baseline_block_fn, warmup=max(1, warmup // 2), iters=max(1, iters // 2))

    return {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "attention": {
            "tokens_per_sec": float((batch * seq_len) / max(attn_s, 1e-12)),
            "latency_ms": float(attn_s * 1000.0),
            "peak_memory_bytes": int(attn_mem),
        },
        "block": {
            "tokens_per_sec": float((batch * seq_len) / max(block_s, 1e-12)),
            "latency_ms": float(block_s * 1000.0),
            "peak_memory_bytes": int(block_mem),
        },
    }


def _readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3-4B Baseline MLX Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- model_path: `{payload['model_path']}`")
    lines.append("")
    lines.append("| T | attn tok/s | attn ms | attn peak mem | block tok/s | block ms | block peak mem |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["results"]:
        lines.append(
            "| {T} | {atok:.1f} | {ams:.2f} | {amem} | {btok:.1f} | {bms:.2f} | {bmem} |".format(
                T=row["seq_len"],
                atok=row["attention"]["tokens_per_sec"],
                ams=row["attention"]["latency_ms"],
                amem=row["attention"]["peak_memory_bytes"],
                btok=row["block"]["tokens_per_sec"],
                bms=row["block"]["latency_ms"],
                bmem=row["block"]["peak_memory_bytes"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline Qwen3-4B MLX benchmark")
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("benchmarks/mlx/qwen3_4b_baseline") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    for T in args.seq_lens:
        try:
            rows.append(
                _bench_one_seq(
                    model,
                    seq_len=int(T),
                    batch=int(args.batch),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    dtype_name=args.dtype,
                )
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            rows.append(
                {
                    "seq_len": int(T),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "model_path": args.model_path,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "model_config": {
            "model_type": config.get("model_type"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "hidden_size": config.get("hidden_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_scaling": config.get("rope_scaling"),
        },
        "results": rows,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(payload), encoding="utf-8")
    print(f"Wrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
