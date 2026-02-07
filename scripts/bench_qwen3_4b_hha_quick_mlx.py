#!/usr/bin/env python3
"""Quick HHA benchmark: measures graph build cost and per-head attention memory.

Runs only 1-4 heads instead of all 32 to keep runtime practical.
Extrapolates total memory from per-head measurements.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.graph.abi import WayfinderGraphABI, graph_metrics, stack_head_abis, validate_graph_abi
from hcsa.graph_strategies import build_strategy
from hcsa.integrations.qwen_mlx import extract_qkv_from_qwen_attention
from hcsa.mlx.attention import (
    permute_cycle_window_attention_single,
    sparse_gather_attention,
    stable_masked_softmax,
)
from hcsa.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)

import math


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _mb(nbytes: int) -> float:
    return round(float(nbytes) / (1024 * 1024), 1)


def _build_graph(
    T: int,
    n_heads: int,
    window: int,
    landmark_stride: Optional[int],
    seed: int,
    num_cycles: int,
) -> tuple[MLXGraphABI, WayfinderGraphABI, float]:
    """Build graph ABI and return (mlx_graph, numpy_abi, build_time_ms)."""
    t0 = time.perf_counter()
    head_abis = []
    for h in range(n_heads):
        strat = build_strategy("random", num_cycles=num_cycles, seed=seed + 7919 * h)
        abi_h = strat.build(
            T=T, r=None, head_idx=h,
            window=window, landmark_stride=landmark_stride, include_self=True,
        )
        head_abis.append(abi_h)

    abi = stack_head_abis(head_abis)
    validate_graph_abi(abi, expect_heads=n_heads, expect_tokens=T, enforce_hamiltonian=True)
    mlx_graph = to_mlx_graph_abi(abi, heads=n_heads, validate=False)
    build_ms = (time.perf_counter() - t0) * 1000.0
    return mlx_graph, abi, build_ms


def _bench_dense_attn(
    base_attn,
    x: mx.array,
    batch: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Benchmark dense attention: latency + memory."""
    q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
    mx.eval(q, k, v)

    # Warmup
    y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
    out = base_attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))
    mx.eval(out)

    _reset_peak_memory()
    t0 = time.perf_counter()
    y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
    out = base_attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))
    mx.eval(out)
    elapsed = time.perf_counter() - t0
    peak = _peak_memory()

    return {
        "latency_s": float(elapsed),
        "tokens_per_sec": float((batch * seq_len) / max(elapsed, 1e-12)),
        "peak_memory_bytes": int(peak),
        "peak_memory_mb": _mb(peak),
    }


def _bench_permute_single_head(
    q_h: mx.array,
    k_h: mx.array,
    v_h: mx.array,
    perm: np.ndarray,
    window: int,
    causal_mask: mx.array,
    perm_mx: mx.array,
    inv_perm: mx.array,
    pi_idx_clamped: mx.array,
    valid_mask: mx.array,
) -> tuple[float, int]:
    """Benchmark one head of permute attention: latency + memory."""
    _reset_peak_memory()
    t0 = time.perf_counter()
    y_h, _, _, _ = permute_cycle_window_attention_single(
        q_h, k_h, v_h,
        perm=perm,
        window=window,
        return_weights=False,
        pre_perm_mx=perm_mx,
        pre_inv_perm=inv_perm,
        pre_pi_idx_clamped=pi_idx_clamped,
        pre_valid_mask=valid_mask,
        pre_causal_mask=causal_mask,
    )
    mx.eval(y_h)
    elapsed = time.perf_counter() - t0
    peak = _peak_memory()
    return elapsed, peak


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    dtype_name: str,
    window: int,
    landmark_stride: Optional[int],
    seed: int,
    num_cycles: int,
    sample_heads: int,
) -> Dict[str, Any]:
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.hidden_size)
    n_heads = int(model.args.num_attention_heads)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.self_attn

    # Dense baseline
    print(f"  Dense attention...", flush=True)
    dense_result = _bench_dense_attn(base_attn, x, batch, seq_len)
    print(f"    {dense_result['peak_memory_mb']}MB, {dense_result['tokens_per_sec']:.0f} tok/s")

    # Build graph
    print(f"  Building graph (T={seq_len}, H={n_heads})...", flush=True)
    mlx_graph, numpy_abi, graph_build_ms = _build_graph(
        seq_len, n_heads, window, landmark_stride, seed, num_cycles,
    )
    gm = graph_metrics(numpy_abi)
    print(f"    graph_build_ms={graph_build_ms:.1f}, degree={gm['degree_mean']:.1f}")

    # Measure graph cache memory
    s_idx = safe_neighbor_idx(mlx_graph.neigh_idx, seq_len)
    c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, seq_len)
    mx.eval(s_idx, c_mask)

    neigh_bytes = int(np.asarray(mlx_graph.neigh_idx).nbytes)
    edge_type_bytes = int(np.asarray(mlx_graph.edge_type).nbytes)
    safe_idx_bytes = int(np.asarray(s_idx).nbytes)
    causal_mask_bytes = int(np.asarray(c_mask).nbytes)
    graph_cache_bytes = neigh_bytes + edge_type_bytes + safe_idx_bytes + causal_mask_bytes

    # Build permute artifacts for sampled heads
    cycle_perms = mlx_graph.meta.get("cycle_perms", [])
    W = 2 * window + 1
    offsets = mx.arange(-window, window + 1, dtype=mx.int32)

    per_head_permute_bytes = 0
    perm_artifacts = []
    for h in range(min(sample_heads, n_heads)):
        perm = cycle_perms[h]
        perm_arr = np.asarray(perm, dtype=np.int32)
        p_mx = mx.array(perm_arr, dtype=mx.int32)
        ip = mx.argsort(p_mx)
        pi_idx = mx.arange(seq_len, dtype=mx.int32).reshape(seq_len, 1) + offsets.reshape(1, W)
        valid = (pi_idx >= 0) & (pi_idx < seq_len)
        pi_clamped = mx.clip(pi_idx, 0, seq_len - 1)
        orig_idx = p_mx
        neigh_orig = orig_idx[pi_clamped]
        query_orig = orig_idx.reshape(seq_len, 1)
        causal_h = neigh_orig <= query_orig
        mx.eval(p_mx, ip, pi_clamped, valid, causal_h)

        h_bytes = (
            int(np.asarray(p_mx).nbytes) + int(np.asarray(ip).nbytes) +
            int(np.asarray(pi_clamped).nbytes) + int(np.asarray(valid).nbytes) +
            int(np.asarray(causal_h).nbytes)
        )
        per_head_permute_bytes = h_bytes
        perm_artifacts.append((perm_arr, p_mx, ip, pi_clamped, valid, causal_h))

    total_cache_bytes = graph_cache_bytes + per_head_permute_bytes * n_heads

    # Per-head HHA attention benchmark (sample a few heads)
    print(f"  Per-head permute attention (sampling {sample_heads} heads)...", flush=True)
    q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
    mx.eval(q, k, v)

    # Expand KV for GQA
    n_kv = int(model.args.num_key_value_heads)
    if n_kv != n_heads:
        repeats = n_heads // n_kv
        k = mx.broadcast_to(k[:, :, None, :, :], (batch, n_kv, repeats, seq_len, k.shape[-1]))
        k = k.reshape(batch, n_heads, seq_len, -1)
        v = mx.broadcast_to(v[:, :, None, :, :], (batch, n_kv, repeats, seq_len, v.shape[-1]))
        v = v.reshape(batch, n_heads, seq_len, -1)
        mx.eval(k, v)

    head_latencies = []
    head_memories = []
    for i, (perm_arr, p_mx, ip, pi_clamped, valid, causal_h) in enumerate(perm_artifacts):
        elapsed, peak = _bench_permute_single_head(
            q[:, i], k[:, i], v[:, i],
            perm_arr, window,
            causal_h, p_mx, ip, pi_clamped, valid,
        )
        head_latencies.append(elapsed)
        head_memories.append(peak)
        print(f"    head {i}: {elapsed:.2f}s, {_mb(peak)}MB")

    avg_head_latency = float(np.mean(head_latencies))
    avg_head_memory = float(np.mean(head_memories))

    # Theoretical extrapolation for all heads
    # In Python-loop impl, peak memory is max of per-head peaks + graph cache
    # In a batched impl, it would be per-head * H
    # Reality with lazy eval: depends on eval granularity
    estimated_sequential_peak = int(avg_head_memory) + total_cache_bytes
    estimated_total_latency = avg_head_latency * n_heads

    # MAE sanity: compare dense vs single-head sparse
    dense_out = scaled_dot_product_attention(
        q[:, 0:1], k[:, 0:1], v[:, 0:1],
        cache=None, scale=base_attn.scale, mask="causal",
    )
    mx.eval(dense_out)
    hha_out, _, _, _ = permute_cycle_window_attention_single(
        q[:, 0], k[:, 0], v[:, 0],
        perm=perm_artifacts[0][0], window=window,
        return_weights=False,
        pre_perm_mx=perm_artifacts[0][1],
        pre_inv_perm=perm_artifacts[0][2],
        pre_pi_idx_clamped=perm_artifacts[0][3],
        pre_valid_mask=perm_artifacts[0][4],
        pre_causal_mask=perm_artifacts[0][5],
    )
    mx.eval(hha_out)
    mae = mx.mean(mx.abs(dense_out[:, 0].astype(mx.float32) - hha_out.astype(mx.float32)))
    mx.eval(mae)

    return {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "n_heads": n_heads,
        "sample_heads": sample_heads,
        "dense_attention": dense_result,
        "graph": {
            "build_ms": float(graph_build_ms),
            "cache_bytes": int(graph_cache_bytes),
            "cache_mb": _mb(graph_cache_bytes),
            "per_head_permute_bytes": int(per_head_permute_bytes),
            "total_cache_bytes": int(total_cache_bytes),
            "total_cache_mb": _mb(total_cache_bytes),
            "metrics": gm,
        },
        "hha_per_head": {
            "avg_latency_s": float(avg_head_latency),
            "avg_peak_memory_bytes": int(avg_head_memory),
            "avg_peak_memory_mb": _mb(int(avg_head_memory)),
            "head_latencies": [float(x) for x in head_latencies],
            "head_memories": [int(x) for x in head_memories],
        },
        "hha_estimated_full": {
            "estimated_latency_s": float(estimated_total_latency),
            "estimated_tokens_per_sec": float(
                (batch * seq_len) / max(estimated_total_latency, 1e-12)
            ),
            "estimated_sequential_peak_bytes": int(estimated_sequential_peak),
            "estimated_sequential_peak_mb": _mb(estimated_sequential_peak),
            "note": "Sequential per-head processing; peak = max(per_head_peak) + graph_cache",
        },
        "dense_attention_memory_theoretical": {
            "t_squared_bytes": int(seq_len * seq_len * 4),  # float32 attention matrix
            "t_squared_mb": _mb(seq_len * seq_len * 4),
            "per_head_bytes": int(seq_len * seq_len * 4 // n_heads),
        },
        "hha_attention_memory_theoretical": {
            "t_times_w_bytes": int(seq_len * (2 * window + 1) * 4),
            "t_times_w_mb": _mb(seq_len * (2 * window + 1) * 4),
            "per_head_bytes": int(seq_len * (2 * window + 1) * 4),
            "window": 2 * window + 1,
        },
        "memory_ratio_theoretical": round(
            float(seq_len * seq_len) / max(float(seq_len * (2 * window + 1)), 1.0), 1
        ),
        "sanity_mae_head0": float(mae.item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Quick HHA benchmark (per-head sampling)")
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-heads", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("benchmarks/mlx/qwen3_4b_hha") / f"quick_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for T in args.seq_lens:
        print(f"\n=== T={T} ===", flush=True)
        try:
            row = _bench_one_seq(
                model,
                seq_len=int(T),
                batch=int(args.batch),
                dtype_name=args.dtype,
                window=int(args.window),
                landmark_stride=(
                    None if int(args.landmark_stride) <= 0 else int(args.landmark_stride)
                ),
                seed=int(args.seed),
                num_cycles=int(args.num_cycles),
                sample_heads=int(args.sample_heads),
            )
            rows.append(row)
            print(f"  Dense: {row['dense_attention']['peak_memory_mb']}MB")
            print(f"  HHA graph cache: {row['graph']['total_cache_mb']}MB")
            print(f"  HHA per-head peak: {row['hha_per_head']['avg_peak_memory_mb']}MB")
            print(f"  Theoretical memory ratio (dense/sparse): {row['memory_ratio_theoretical']}x")
        except Exception as exc:
            import traceback
            traceback.print_exc()
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
        "hha_settings": {
            "path": "permute",
            "strategy": "random",
            "window": int(args.window),
            "landmark_stride": int(args.landmark_stride),
            "num_cycles": int(args.num_cycles),
            "seed": int(args.seed),
            "sample_heads": int(args.sample_heads),
        },
        "results": rows,
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
