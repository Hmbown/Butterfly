#!/usr/bin/env python3
"""Micro-benchmark: batched vs per-head-loop permute attention.

No model loading — pure attention kernel comparison with synthetic data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx
from hcsa.mlx.attention import (
    _GraphCache,
    wayfinder_permute_window_attention,
    wayfinder_permute_window_attention_batched,
)
from hcsa.mlx.graph_abi import MLXGraphABI


def build_synthetic_cache(H: int, T: int, W_half: int):
    """Build synthetic permutations and window artifacts."""
    W = 2 * W_half + 1
    offsets = mx.arange(-W_half, W_half + 1, dtype=mx.int32)

    perm_list, inv_list, pi_list, valid_list, causal_list = [], [], [], [], []
    cycle_perms_np = []

    for h in range(H):
        rng = np.random.RandomState(42 + h)
        perm_np = rng.permutation(T).astype(np.int32)
        cycle_perms_np.append(perm_np.tolist())
        p_mx = mx.array(perm_np, dtype=mx.int32)
        ip = mx.argsort(p_mx)
        pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
        valid = (pi_idx >= 0) & (pi_idx < T)
        pi_clamped = mx.clip(pi_idx, 0, T - 1)
        neigh_orig = p_mx[pi_clamped]
        query_orig = p_mx.reshape(T, 1)
        causal_h = neigh_orig <= query_orig

        perm_list.append(p_mx)
        inv_list.append(ip)
        pi_list.append(pi_clamped)
        valid_list.append(valid)
        causal_list.append(causal_h)

    # Stacked versions for batched path
    perm_stacked = mx.stack(perm_list, axis=0)
    inv_stacked = mx.stack(inv_list, axis=0)
    pi_stacked = mx.stack(pi_list, axis=0)
    valid_stacked = mx.stack(valid_list, axis=0)
    causal_stacked = mx.stack(causal_list, axis=0)

    # Dummy MLXGraphABI with cycle_perms in meta
    neigh_idx = mx.zeros((H, T, 1), dtype=mx.int32)
    edge_type = mx.zeros((H, T, 1), dtype=mx.uint8)
    mlx_graph = MLXGraphABI(
        neigh_idx=neigh_idx,
        edge_type=edge_type,
        meta={"cycle_perms": cycle_perms_np},
    )

    cache = _GraphCache(
        mlx_graph=mlx_graph,
        numpy_abi=None,  # type: ignore
        safe_idx=mx.zeros((H, T, 1), dtype=mx.int32),
        causal_mask=mx.zeros((H, T, 1), dtype=mx.bool_),
        perm_mx=perm_list,
        inv_perm=inv_list,
        pi_idx_clamped=pi_list,
        valid_mask=valid_list,
        causal_masks=causal_list,
        cache_key=(),
    )

    return cache, mlx_graph, perm_stacked, inv_stacked, pi_stacked, valid_stacked, causal_stacked


def bench(label: str, fn, warmup: int = 1, iters: int = 3):
    """Time a function, return median seconds."""
    for _ in range(warmup):
        y = fn()
        mx.eval(y)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    return med


def main():
    configs = [
        # (H, T, dh, W_half)
        (32, 512, 80, 64),
        (32, 1024, 80, 64),
        (32, 2048, 80, 64),
    ]

    B = 1
    print(f"{'Config':>30s}  {'Loop (s)':>10s}  {'Batched (s)':>12s}  {'Speedup':>8s}  {'MAE':>10s}")
    print("-" * 80)

    for H, T, dh, W_half in configs:
        label = f"H={H} T={T} dh={dh} W={2*W_half+1}"
        q = mx.random.normal((B, H, T, dh), dtype=mx.float16)
        k = mx.random.normal((B, H, T, dh), dtype=mx.float16)
        v = mx.random.normal((B, H, T, dh), dtype=mx.float16)
        mx.eval(q, k, v)

        cache, graph, p_s, ip_s, pi_s, v_s, c_s = build_synthetic_cache(H, T, W_half)
        mx.eval(p_s, ip_s, pi_s, v_s, c_s)

        def loop_fn():
            y, _, _, _ = wayfinder_permute_window_attention(
                q, k, v, graph, window=W_half, cache=cache,
            )
            return y

        def batched_fn():
            y, _ = wayfinder_permute_window_attention_batched(
                q, k, v,
                all_perms=p_s, all_inv_perms=ip_s,
                window=W_half,
                query_chunk_size=256,
            )
            return y

        # Correctness check
        y_loop = loop_fn()
        y_batch = batched_fn()
        mx.eval(y_loop, y_batch)
        mae = mx.mean(mx.abs(y_loop.astype(mx.float32) - y_batch.astype(mx.float32)))
        mx.eval(mae)

        t_loop = bench("loop", loop_fn, warmup=1, iters=3)
        t_batch = bench("batched", batched_fn, warmup=1, iters=3)
        speedup = t_loop / max(t_batch, 1e-9)

        print(f"{label:>30s}  {t_loop:10.4f}  {t_batch:12.4f}  {speedup:7.1f}x  {float(mae.item()):10.6f}")


if __name__ == "__main__":
    main()
