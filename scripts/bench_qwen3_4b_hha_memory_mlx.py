#!/usr/bin/env python3
"""Focused HHA memory benchmark for Qwen3-4B MLX.

Measures peak memory and single-pass latency at various T values.
Avoids expensive multi-iteration timing — optimized for large T.
"""
from __future__ import annotations

import argparse
import json
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

from hcsa.integrations.qwen_mlx import (
    QwenHHAConfig,
    QwenWayfinderAttention,
    extract_qkv_from_qwen_attention,
)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _bench_one_pass(fn) -> tuple[float, int]:
    """Single timed forward pass + peak memory."""
    _reset_peak_memory()
    t0 = time.perf_counter()
    y = fn()
    mx.eval(y)
    elapsed = time.perf_counter() - t0
    peak = _peak_memory()
    return elapsed, peak


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    dtype_name: str,
    hha_cfg: QwenHHAConfig,
) -> Dict[str, Any]:
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.self_attn

    # --- Dense baseline attention (single pass) ---
    def dense_fn():
        q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
        y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
        return base_attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))

    # Warmup dense
    yw = dense_fn()
    mx.eval(yw)
    del yw

    dense_s, dense_mem = _bench_one_pass(dense_fn)

    # --- HHA attention (single pass) ---
    hha_attn = QwenWayfinderAttention(base_attn, hha_cfg)

    # First call builds graph
    _reset_peak_memory()
    t_first = time.perf_counter()
    y_first = hha_attn(x, mask="causal", cache=None)
    mx.eval(y_first)
    first_elapsed = time.perf_counter() - t_first
    first_mem = _peak_memory()
    first_profile = hha_attn.last_profile.to_dict()

    # Second call should be cache hit
    hha_s, hha_mem = _bench_one_pass(lambda: hha_attn(x, mask="causal", cache=None))
    cached_profile = hha_attn.last_profile.to_dict()

    # MAE sanity check (compare dense vs HHA on same input)
    dense_out = dense_fn()
    hha_out = hha_attn(x, mask="causal", cache=None)
    mx.eval(dense_out, hha_out)
    mae = mx.mean(mx.abs(dense_out.astype(mx.float32) - hha_out.astype(mx.float32)))
    mx.eval(mae)

    return {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "dense_attention": {
            "latency_s": float(dense_s),
            "tokens_per_sec": float((batch * seq_len) / max(dense_s, 1e-12)),
            "peak_memory_bytes": int(dense_mem),
            "peak_memory_mb": round(float(dense_mem) / (1024 * 1024), 1),
        },
        "hha_attention_first": {
            "latency_s": float(first_elapsed),
            "tokens_per_sec": float((batch * seq_len) / max(first_elapsed, 1e-12)),
            "peak_memory_bytes": int(first_mem),
            "peak_memory_mb": round(float(first_mem) / (1024 * 1024), 1),
            "graph_build_ms": float(first_profile.get("graph_build_ms", 0.0)),
            "cache_hit": bool(first_profile.get("cache_hit", False)),
        },
        "hha_attention_cached": {
            "latency_s": float(hha_s),
            "tokens_per_sec": float((batch * seq_len) / max(hha_s, 1e-12)),
            "peak_memory_bytes": int(hha_mem),
            "peak_memory_mb": round(float(hha_mem) / (1024 * 1024), 1),
            "graph_build_ms": float(cached_profile.get("graph_build_ms", 0.0)),
            "cache_hit": bool(cached_profile.get("cache_hit", False)),
            "cache_source": cached_profile.get("cache_source"),
            "cache_persistent_bytes": int(hha_attn.cache_persistent_bytes()),
        },
        "sanity_mae": float(mae.item()),
        "memory_ratio": round(float(hha_mem) / max(float(dense_mem), 1), 3),
        "graph_metrics": hha_attn.last_graph_metrics,
        "edge_utilization": hha_attn.last_edge_utilization_proxy,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Focused HHA memory benchmark")
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--path", type=str, default="permute", choices=["sparse", "permute"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    hha_cfg = QwenHHAConfig(
        path=args.path,
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=int(args.num_cycles),
        seed=int(args.seed),
        edge_bias=True,
        window_drop=0.0,
        compiled_graph_dir=None,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("benchmarks/mlx/qwen3_4b_hha") / f"memory_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for T in args.seq_lens:
        print(f"--- T={T} ---", flush=True)
        try:
            row = _bench_one_seq(
                model,
                seq_len=int(T),
                batch=int(args.batch),
                dtype_name=args.dtype,
                hha_cfg=hha_cfg,
            )
            rows.append(row)
            print(
                f"  dense: {row['dense_attention']['peak_memory_mb']}MB "
                f"({row['dense_attention']['tokens_per_sec']:.0f} tok/s)"
            )
            print(
                f"  hha:   {row['hha_attention_cached']['peak_memory_mb']}MB "
                f"({row['hha_attention_cached']['tokens_per_sec']:.0f} tok/s)"
            )
            print(f"  ratio: {row['memory_ratio']}x  mae: {row['sanity_mae']:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({"seq_len": int(T), "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "model_config": {
            "model_type": config.get("model_type"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "hidden_size": config.get("hidden_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "hha_config": hha_cfg.__dict__,
        "results": rows,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
