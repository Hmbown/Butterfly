#!/usr/bin/env python3
"""Micro-profiler for HCSA permute-window attention inner loop.

Measures per-stage costs to identify the dominant term in constant C
of the empirical law R(T) ≈ T/(W·C).

Outputs JSON with per-stage timing breakdown.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def profile_permute_batched(
    *,
    B: int,
    H: int,
    T: int,
    dh: int,
    window: int,
    head_chunk_size: int,
    query_chunk_size: int,
    dtype=mx.float16,
) -> Dict[str, Any]:
    """Profile the permute-window attention inner loop with fine-grained timing."""

    W = 2 * window + 1
    scale = 1.0 / math.sqrt(dh)

    # Create random Q, K, V and permutations
    q = mx.random.normal((B, H, T, dh), dtype=dtype)
    k = mx.random.normal((B, H, T, dh), dtype=dtype)
    v = mx.random.normal((B, H, T, dh), dtype=dtype)
    mx.eval(q, k, v)

    # Build random cycle permutations per head
    all_perms_np = np.zeros((H, T), dtype=np.int32)
    all_inv_perms_np = np.zeros((H, T), dtype=np.int32)
    rng = np.random.default_rng(42)
    for h in range(H):
        p = rng.permutation(T).astype(np.int32)
        all_perms_np[h] = p
        inv = np.empty(T, dtype=np.int32)
        inv[p] = np.arange(T, dtype=np.int32)
        all_inv_perms_np[h] = inv

    all_perms = mx.array(all_perms_np)
    all_inv_perms = mx.array(all_inv_perms_np)
    mx.eval(all_perms, all_inv_perms)

    h_chunk = min(head_chunk_size, H)
    q_chunk = min(query_chunk_size, T)
    num_h_chunks = (H + h_chunk - 1) // h_chunk
    num_q_chunks = (T + q_chunk - 1) // q_chunk

    results: Dict[str, Any] = {
        "B": B, "H": H, "T": T, "dh": dh,
        "window": window, "W": W,
        "head_chunk_size": h_chunk,
        "query_chunk_size": q_chunk,
        "num_head_chunks": num_h_chunks,
        "num_query_chunks": num_q_chunks,
        "total_eval_syncs_inference_fast_path": num_h_chunks * num_q_chunks + num_h_chunks,
    }

    # ===== PROFILE 1: Current code path (with per-chunk mx.eval) =====
    _reset_peak_memory()
    t_total_0 = _now_ms()
    t_permute_total = 0.0
    t_gather_total = 0.0
    t_mask_total = 0.0
    t_sdpa_total = 0.0
    t_unpermute_total = 0.0
    t_eval_total = 0.0
    eval_count = 0

    y_chunks_all: list[mx.array] = []
    for h0 in range(0, H, h_chunk):
        h1 = min(h0 + h_chunk, H)
        hc = h1 - h0
        q_c = q[:, h0:h1, :, :]
        k_c = k[:, h0:h1, :, :]
        v_c = v[:, h0:h1, :, :]
        perms_c = all_perms[h0:h1, :]
        inv_c = all_inv_perms[h0:h1, :]

        y_pi = mx.zeros((B, hc, T, dh), dtype=v.dtype)

        for s in range(0, T, q_chunk):
            e = min(T, s + q_chunk)
            ks = max(0, s - window)
            ke = min(T, e + window)

            # Gather Q/K/V blocks
            t_g0 = _now_ms()
            q_idx = perms_c[:, s:e]
            k_idx = perms_c[:, ks:ke]
            q_gidx = mx.broadcast_to(q_idx[None, :, :, None], (B, hc, e - s, 1))
            q_blk = mx.take_along_axis(q_c, q_gidx, axis=2)
            k_gidx = mx.broadcast_to(k_idx[None, :, :, None], (B, hc, ke - ks, 1))
            k_blk = mx.take_along_axis(k_c, k_gidx, axis=2)
            v_blk = mx.take_along_axis(v_c, k_gidx, axis=2)
            t_gather_total += _now_ms() - t_g0

            # Mask construction
            t_m0 = _now_ms()
            q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, e - s, 1)
            k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, ke - ks)
            rel = k_pos - q_pos
            in_window = (rel >= -window) & (rel <= window)
            causal = k_idx[:, None, :] <= q_idx[:, :, None]
            mask_eff = in_window & causal
            t_mask_total += _now_ms() - t_m0

            # SDPA
            t_s0 = _now_ms()
            y_blk = mx.fast.scaled_dot_product_attention(
                q_blk, k_blk, v_blk,
                scale=scale,
                mask=mask_eff[None, :, :, :],
            ).astype(v.dtype)
            t_sdpa_total += _now_ms() - t_s0

            # Accumulate + eval (current code path)
            y_pi = y_pi.at[:, :, s:e, :].add(y_blk)
            t_e0 = _now_ms()
            mx.eval(y_pi)
            t_eval_total += _now_ms() - t_e0
            eval_count += 1

        # Unpermute
        t_u0 = _now_ms()
        inv_idx = mx.broadcast_to(inv_c[None, :, :, None], (B, hc, T, 1))
        y_h = mx.take_along_axis(y_pi, inv_idx, axis=2).astype(v.dtype)
        mx.eval(y_h)
        t_unpermute_total += _now_ms() - t_u0
        eval_count += 1
        y_chunks_all.append(y_h)

    t_total_current = _now_ms() - t_total_0
    peak_mem_current = _peak_memory()

    results["current_path"] = {
        "total_ms": t_total_current,
        "gather_ms": t_gather_total,
        "mask_ms": t_mask_total,
        "sdpa_ms": t_sdpa_total,
        "eval_sync_ms": t_eval_total,
        "unpermute_ms": t_unpermute_total,
        "eval_count": eval_count,
        "peak_memory_bytes": peak_mem_current,
    }

    # ===== PROFILE 2: No per-chunk eval (lazy graph, single eval per head chunk) =====
    _reset_peak_memory()
    t_total_0 = _now_ms()
    t_gather_total2 = 0.0
    t_mask_total2 = 0.0
    t_sdpa_total2 = 0.0
    t_eval_total2 = 0.0
    t_unpermute_total2 = 0.0
    eval_count2 = 0

    y_chunks_all2: list[mx.array] = []
    for h0 in range(0, H, h_chunk):
        h1 = min(h0 + h_chunk, H)
        hc = h1 - h0
        q_c = q[:, h0:h1, :, :]
        k_c = k[:, h0:h1, :, :]
        v_c = v[:, h0:h1, :, :]
        perms_c = all_perms[h0:h1, :]
        inv_c = all_inv_perms[h0:h1, :]

        y_pi_chunks: list[mx.array] = []

        for s in range(0, T, q_chunk):
            e = min(T, s + q_chunk)
            ks = max(0, s - window)
            ke = min(T, e + window)

            # Gather Q/K/V blocks
            t_g0 = _now_ms()
            q_idx = perms_c[:, s:e]
            k_idx = perms_c[:, ks:ke]
            q_gidx = mx.broadcast_to(q_idx[None, :, :, None], (B, hc, e - s, 1))
            q_blk = mx.take_along_axis(q_c, q_gidx, axis=2)
            k_gidx = mx.broadcast_to(k_idx[None, :, :, None], (B, hc, ke - ks, 1))
            k_blk = mx.take_along_axis(k_c, k_gidx, axis=2)
            v_blk = mx.take_along_axis(v_c, k_gidx, axis=2)
            t_gather_total2 += _now_ms() - t_g0

            # Mask construction
            t_m0 = _now_ms()
            q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, e - s, 1)
            k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, ke - ks)
            rel = k_pos - q_pos
            in_window = (rel >= -window) & (rel <= window)
            causal = k_idx[:, None, :] <= q_idx[:, :, None]
            mask_eff = in_window & causal
            t_mask_total2 += _now_ms() - t_m0

            # SDPA
            t_s0 = _now_ms()
            y_blk = mx.fast.scaled_dot_product_attention(
                q_blk, k_blk, v_blk,
                scale=scale,
                mask=mask_eff[None, :, :, :],
            ).astype(v.dtype)
            t_sdpa_total2 += _now_ms() - t_s0

            # NO per-chunk eval — just append
            y_pi_chunks.append(y_blk)

        # Concatenate + unpermute + single eval
        t_u0 = _now_ms()
        if len(y_pi_chunks) == 1:
            y_pi = y_pi_chunks[0]
        else:
            y_pi = mx.concatenate(y_pi_chunks, axis=2)
        inv_idx = mx.broadcast_to(inv_c[None, :, :, None], (B, hc, T, 1))
        y_h = mx.take_along_axis(y_pi, inv_idx, axis=2).astype(v.dtype)
        t_unpermute_total2 += _now_ms() - t_u0

        t_e0 = _now_ms()
        mx.eval(y_h)
        t_eval_total2 += _now_ms() - t_e0
        eval_count2 += 1
        y_chunks_all2.append(y_h)

    t_total_lazy = _now_ms() - t_total_0
    peak_mem_lazy = _peak_memory()

    results["lazy_path"] = {
        "total_ms": t_total_lazy,
        "gather_ms": t_gather_total2,
        "mask_ms": t_mask_total2,
        "sdpa_ms": t_sdpa_total2,
        "eval_sync_ms": t_eval_total2,
        "unpermute_ms": t_unpermute_total2,
        "eval_count": eval_count2,
        "peak_memory_bytes": peak_mem_lazy,
    }

    # Verify numerics match
    y_current = mx.concatenate(y_chunks_all, axis=1) if len(y_chunks_all) > 1 else y_chunks_all[0]
    y_lazy = mx.concatenate(y_chunks_all2, axis=1) if len(y_chunks_all2) > 1 else y_chunks_all2[0]
    diff = mx.max(mx.abs(y_current.astype(mx.float32) - y_lazy.astype(mx.float32)))
    mx.eval(diff)
    results["max_abs_diff"] = float(diff.item())

    results["speedup_lazy_over_current"] = t_total_current / max(t_total_lazy, 1e-9)
    results["memory_ratio_lazy_over_current"] = peak_mem_lazy / max(peak_mem_current, 1)

    # ===== PROFILE 3: Dense baseline (mx.fast.scaled_dot_product_attention) =====
    _reset_peak_memory()
    t_d0 = _now_ms()
    y_dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(y_dense)
    t_dense = _now_ms() - t_d0
    peak_mem_dense = _peak_memory()

    results["dense_baseline"] = {
        "total_ms": t_dense,
        "peak_memory_bytes": peak_mem_dense,
    }
    results["ratio_current_vs_dense"] = t_dense / max(t_total_current, 1e-9)
    results["ratio_lazy_vs_dense"] = t_dense / max(t_total_lazy, 1e-9)

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Micro-profiler for HCSA permute-window inner loop")
    p.add_argument("--T", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=12)
    p.add_argument("--dh", type=int, default=64)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--head-chunk-size", type=int, default=8)
    p.add_argument("--query-chunk-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    all_results: list[Dict[str, Any]] = []
    for T in args.T:
        print(f"\n=== T={T} ===", flush=True)

        # Warmup
        for _ in range(args.warmup):
            profile_permute_batched(
                B=args.B, H=args.H, T=T, dh=args.dh,
                window=args.window,
                head_chunk_size=args.head_chunk_size,
                query_chunk_size=args.query_chunk_size,
            )

        r = profile_permute_batched(
            B=args.B, H=args.H, T=T, dh=args.dh,
            window=args.window,
            head_chunk_size=args.head_chunk_size,
            query_chunk_size=args.query_chunk_size,
        )
        all_results.append(r)

        cur = r["current_path"]
        lazy = r["lazy_path"]
        dense = r["dense_baseline"]
        print(f"  Current path: {cur['total_ms']:.1f} ms  (eval_sync={cur['eval_sync_ms']:.1f} ms, "
              f"gather={cur['gather_ms']:.1f} ms, mask={cur['mask_ms']:.1f} ms, "
              f"sdpa={cur['sdpa_ms']:.1f} ms, unpermute={cur['unpermute_ms']:.1f} ms, "
              f"eval_count={cur['eval_count']}, peak_mem={cur['peak_memory_bytes']/1e6:.1f} MB)")
        print(f"  Lazy path:    {lazy['total_ms']:.1f} ms  (eval_sync={lazy['eval_sync_ms']:.1f} ms, "
              f"gather={lazy['gather_ms']:.1f} ms, mask={lazy['mask_ms']:.1f} ms, "
              f"sdpa={lazy['sdpa_ms']:.1f} ms, unpermute={lazy['unpermute_ms']:.1f} ms, "
              f"eval_count={lazy['eval_count']}, peak_mem={lazy['peak_memory_bytes']/1e6:.1f} MB)")
        print(f"  Dense SDPA:   {dense['total_ms']:.1f} ms  (peak_mem={dense['peak_memory_bytes']/1e6:.1f} MB)")
        print(f"  Speedup lazy/current: {r['speedup_lazy_over_current']:.2f}x")
        print(f"  Throughput ratio current/dense: {r['ratio_current_vs_dense']:.3f}x")
        print(f"  Throughput ratio lazy/dense:    {r['ratio_lazy_vs_dense']:.3f}x")
        print(f"  Memory ratio lazy/current: {r['memory_ratio_lazy_over_current']:.3f}")
        print(f"  Max abs diff (numerics check): {r['max_abs_diff']:.6f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nWrote profile to {out_path}")


if __name__ == "__main__":
    main()
