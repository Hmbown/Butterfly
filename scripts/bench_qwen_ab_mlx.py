#!/usr/bin/env python3
"""Qwen3-1.7B A/B benchmark: Dense attention vs HCSA sparse attention.

Measures at THREE levels for each sequence length:
  1. Isolated attention — just the attention function (where HCSA wins)
  2. Full block — attention + MLP + norms (single transformer layer)
  3. Full model prefill — all layers, chunked (end-to-end)

This lets you see where the memory savings actually live.
"""
from __future__ import annotations

import argparse
import gc
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
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.integrations.qwen_mlx import (  # noqa: E402
    QwenWayfinderConfig,
    QwenWayfinderAttention,
    extract_qkv_from_qwen_attention,
    swap_qwen_attention_with_wayfinder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _clear() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    gc.collect()


def _mb(b: int) -> float:
    return b / (1024 * 1024)


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _bench_fn(fn: Any, *, warmup: int, iters: int) -> tuple[float, int]:
    """Run fn with warmup, then measure median time and peak memory."""
    for _ in range(max(1, warmup)):
        y = fn()
        mx.eval(y)

    _clear()
    _reset_peak_memory()
    times: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return _median(times), _peak_memory()


# ---------------------------------------------------------------------------
# Isolated attention benchmark (the "river" — where HCSA savings live)
# ---------------------------------------------------------------------------

def _bench_attention_isolated(
    model: Any,
    *,
    seq_len: int,
    batch: int,
    wf_cfg: QwenWayfinderConfig,
    warmup: int = 1,
    iters: int = 3,
) -> Dict[str, Any]:
    """Benchmark just the attention function in isolation."""
    dtype = mx.bfloat16
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.self_attn

    # Build HCSA attention module from layer0's base attention
    wf_attn = QwenWayfinderAttention(base_attn, wf_cfg)
    wf_attn.compute_edge_utilization_proxy = False
    wf_attn.compute_graph_metrics = False

    # --- Dense baseline: extract QKV + scaled_dot_product_attention ---
    def dense_fn():
        q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
        y = scaled_dot_product_attention(
            q, k, v, cache=None, scale=base_attn.scale, mask="causal",
        )
        return base_attn.o_proj(
            y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        )

    # --- HCSA: wayfinder attention ---
    def hcsa_fn():
        return wf_attn(x, mask="causal", cache=None)

    # Warm the graph cache
    mx.eval(hcsa_fn())

    dense_s, dense_mem = _bench_fn(dense_fn, warmup=warmup, iters=iters)
    hcsa_s, hcsa_mem = _bench_fn(hcsa_fn, warmup=warmup, iters=iters)

    # Sanity check: MAE between outputs
    d_out = dense_fn()
    h_out = hcsa_fn()
    mx.eval(d_out, h_out)
    mae = float(mx.mean(mx.abs(
        d_out.astype(mx.float32) - h_out.astype(mx.float32)
    )).item())

    tps_d = float((batch * seq_len) / max(dense_s, 1e-12))
    tps_h = float((batch * seq_len) / max(hcsa_s, 1e-12))
    mem_red = 100.0 * (1.0 - hcsa_mem / max(dense_mem, 1))

    return {
        "dense": {
            "latency_ms": round(dense_s * 1000, 1),
            "tok_s": round(tps_d, 0),
            "peak_memory_bytes": dense_mem,
            "peak_memory_mb": round(_mb(dense_mem), 1),
        },
        "hcsa": {
            "latency_ms": round(hcsa_s * 1000, 1),
            "tok_s": round(tps_h, 0),
            "peak_memory_bytes": hcsa_mem,
            "peak_memory_mb": round(_mb(hcsa_mem), 1),
        },
        "memory_reduction_pct": round(mem_red, 1),
        "throughput_speedup": round(tps_h / max(tps_d, 1e-12), 3),
        "mae": mae,
    }


# ---------------------------------------------------------------------------
# Block-level benchmark (attention + MLP + norms)
# ---------------------------------------------------------------------------

def _bench_block_isolated(
    model: Any,
    *,
    seq_len: int,
    batch: int,
    wf_cfg: QwenWayfinderConfig,
    warmup: int = 1,
    iters: int = 2,
) -> Dict[str, Any]:
    """Benchmark a single transformer block (layer0)."""
    dtype = mx.bfloat16
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    orig_attn = layer0.self_attn

    # --- Dense block ---
    def dense_block():
        return layer0(x, mask="causal", cache=None)

    dense_s, dense_mem = _bench_fn(dense_block, warmup=warmup, iters=iters)

    # --- HCSA block ---
    wf_attn = QwenWayfinderAttention(orig_attn, wf_cfg)
    wf_attn.compute_edge_utilization_proxy = False
    wf_attn.compute_graph_metrics = False
    layer0.self_attn = wf_attn
    # Warm graph cache
    mx.eval(layer0(x, mask="causal", cache=None))

    def hcsa_block():
        return layer0(x, mask="causal", cache=None)

    hcsa_s, hcsa_mem = _bench_fn(hcsa_block, warmup=warmup, iters=iters)

    # Restore
    layer0.self_attn = orig_attn

    tps_d = float((batch * seq_len) / max(dense_s, 1e-12))
    tps_h = float((batch * seq_len) / max(hcsa_s, 1e-12))
    mem_red = 100.0 * (1.0 - hcsa_mem / max(dense_mem, 1))

    return {
        "dense": {
            "latency_ms": round(dense_s * 1000, 1),
            "tok_s": round(tps_d, 0),
            "peak_memory_bytes": dense_mem,
            "peak_memory_mb": round(_mb(dense_mem), 1),
        },
        "hcsa": {
            "latency_ms": round(hcsa_s * 1000, 1),
            "tok_s": round(tps_h, 0),
            "peak_memory_bytes": hcsa_mem,
            "peak_memory_mb": round(_mb(hcsa_mem), 1),
        },
        "memory_reduction_pct": round(mem_red, 1),
        "throughput_speedup": round(tps_h / max(tps_d, 1e-12), 3),
    }


# ---------------------------------------------------------------------------
# Full model prefill benchmark (the "ocean")
# ---------------------------------------------------------------------------

def _bench_full_model(
    model: Any,
    *,
    seq_len: int,
    batch: int,
    chunk_size: int,
    vocab_size: int,
    wf_cfg: QwenWayfinderConfig,
    originals: List[Any],
) -> Dict[str, Any]:
    """Full-model chunked prefill A/B."""
    prompt = mx.random.randint(
        0, vocab_size, shape=(batch, seq_len), dtype=mx.int32,
    )

    def _run(use_hcsa: bool) -> Dict[str, Any]:
        _restore_original_attns(model, originals)
        if use_hcsa:
            replaced = swap_qwen_attention_with_wayfinder(
                model, cfg=wf_cfg,
            )
            for idx in replaced:
                a = model.layers[idx].self_attn
                if isinstance(a, QwenWayfinderAttention):
                    a.compute_edge_utilization_proxy = False
                    a.compute_graph_metrics = False
        _clear()
        cache = list(make_prompt_cache(model))
        _clear()
        _reset_peak_memory()

        t0 = time.perf_counter()
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            logits = model(prompt[:, start:end], cache=cache)
            mx.eval(logits)
        sec = time.perf_counter() - t0
        peak = _peak_memory()
        del cache
        _clear()
        _restore_original_attns(model, originals)
        return {
            "prefill_sec": round(sec, 3),
            "prefill_tok_s": round((batch * seq_len) / max(sec, 1e-12), 0),
            "peak_memory_bytes": peak,
            "peak_memory_mb": round(_mb(peak), 1),
        }

    dense = _run(False)
    hcsa = _run(True)

    d_mem = dense["peak_memory_bytes"]
    h_mem = hcsa["peak_memory_bytes"]
    mem_red = 100.0 * (1.0 - h_mem / max(d_mem, 1)) if d_mem > 0 else 0.0
    d_tok = dense["prefill_tok_s"]
    h_tok = hcsa["prefill_tok_s"]

    return {
        "dense": dense,
        "hcsa": hcsa,
        "memory_reduction_pct": round(mem_red, 1),
        "throughput_speedup": round(h_tok / max(d_tok, 1e-12), 3),
    }


def _restore_original_attns(model: Any, originals: List[Any]) -> None:
    for i, attn in enumerate(originals):
        model.layers[i].self_attn = attn


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

def _print_attention_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "",
        "## Isolated Attention (single layer, the 'river')",
        "",
        "| T | dense_mem_MB | hcsa_mem_MB | mem_reduction "
        "| dense_tok/s | hcsa_tok/s | speedup | MAE |",
        "|------:|------------:|------------:|--------------:"
        "|-----------:|----------:|--------:|----:|",
    ]
    for row in rows:
        T = row["seq_len"]
        a = row.get("attention", {})
        if not a:
            continue
        d = a["dense"]
        h = a["hcsa"]
        lines.append(
            f"| {T:>5} | {d['peak_memory_mb']:>11.0f} | "
            f"{h['peak_memory_mb']:>11.0f} | "
            f"{a['memory_reduction_pct']:>12.1f}% | "
            f"{d['tok_s']:>10.0f} | {h['tok_s']:>9.0f} | "
            f"{a['throughput_speedup']:>6.3f}x | "
            f"{a.get('mae', 0):.4f} |"
        )
    table = "\n".join(lines)
    print(table, flush=True)
    return table


def _print_block_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "",
        "## Full Block (attn + MLP + norms)",
        "",
        "| T | dense_mem_MB | hcsa_mem_MB | mem_reduction "
        "| dense_tok/s | hcsa_tok/s | speedup |",
        "|------:|------------:|------------:|--------------:"
        "|-----------:|----------:|--------:|",
    ]
    for row in rows:
        T = row["seq_len"]
        b = row.get("block", {})
        if not b:
            continue
        d = b["dense"]
        h = b["hcsa"]
        lines.append(
            f"| {T:>5} | {d['peak_memory_mb']:>11.0f} | "
            f"{h['peak_memory_mb']:>11.0f} | "
            f"{b['memory_reduction_pct']:>12.1f}% | "
            f"{d['tok_s']:>10.0f} | {h['tok_s']:>9.0f} | "
            f"{b['throughput_speedup']:>6.3f}x |"
        )
    table = "\n".join(lines)
    print(table, flush=True)
    return table


def _print_model_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "",
        "## Full Model Prefill (all layers, chunked)",
        "",
        "| T | dense_mem_MB | hcsa_mem_MB | mem_reduction "
        "| dense_tok/s | hcsa_tok/s | speedup |",
        "|------:|------------:|------------:|--------------:"
        "|-----------:|----------:|--------:|",
    ]
    for row in rows:
        T = row["seq_len"]
        m = row.get("model", {})
        if not m:
            continue
        d = m["dense"]
        h = m["hcsa"]
        lines.append(
            f"| {T:>5} | {d['peak_memory_mb']:>11.0f} | "
            f"{h['peak_memory_mb']:>11.0f} | "
            f"{m['memory_reduction_pct']:>12.1f}% | "
            f"{d['prefill_tok_s']:>10.0f} | {h['prefill_tok_s']:>9.0f} | "
            f"{m['throughput_speedup']:>6.3f}x |"
        )
    table = "\n".join(lines)
    print(table, flush=True)
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Qwen3-1.7B A/B: Dense vs HCSA at three isolation levels"
    )
    p.add_argument(
        "--model-path", type=str, default="mlx-community/Qwen3-1.7B-4bit",
    )
    p.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-model", action="store_true",
        help="Skip full-model prefill (faster, attention+block only)",
    )

    # HCSA config
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--head-chunk-size", type=int, default=2)
    p.add_argument("--query-chunk-size", type=int, default=384)
    p.add_argument("--circular", action="store_true", default=True)

    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    try:
        mx.random.seed(args.seed)
    except Exception:
        pass

    _log(f"Loading model {args.model_path} ...")
    model, _tok, config = load(  # type: ignore[misc]
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded.")

    vocab_size = int(
        config.get("vocab_size", getattr(model.args, "vocab_size", 0))
    )
    originals = [layer.self_attn for layer in model.layers]

    wf_cfg = QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=args.window,
        landmark_stride=(
            None if args.landmark_stride <= 0 else args.landmark_stride
        ),
        num_cycles=args.num_cycles,
        seed=args.seed,
        edge_bias=True,
        permute_head_chunk_size=args.head_chunk_size,
        query_chunk_size=args.query_chunk_size,
        circular=args.circular,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        permute_log_chunks=False,
    )

    model_tag = (
        args.model_path.rstrip("/").split("/")[-1].lower().replace("-", "_")
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(
        f"benchmarks/mlx/{model_tag}_wayfinder/ab_comparison"
    ) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    _log(f"Output dir: {out_dir}")

    rows: List[Dict[str, Any]] = []
    total = len(args.seq_lens)
    run_t0 = time.perf_counter()

    for i, T in enumerate(args.seq_lens, start=1):
        _log(f"\n{'='*60}")
        _log(f"  [{i}/{total}] seq_len = {T}")
        _log(f"{'='*60}")
        t0 = time.perf_counter()

        row: Dict[str, Any] = {"seq_len": T, "batch": args.batch}

        # Level 1: Isolated attention
        _log(f"  [attention] isolated A/B T={T} ...")
        try:
            attn_result = _bench_attention_isolated(
                model, seq_len=T, batch=args.batch, wf_cfg=wf_cfg,
                warmup=args.warmup, iters=args.iters,
            )
            row["attention"] = attn_result
            _log(
                f"  [attention] mem reduction: "
                f"{attn_result['memory_reduction_pct']:.1f}%  "
                f"speedup: {attn_result['throughput_speedup']:.3f}x"
            )
        except Exception as exc:
            row["attention"] = {"error": f"{type(exc).__name__}: {exc}"}
            _log(f"  [attention] ERROR: {exc}")
        _clear()

        # Level 2: Full block
        _log(f"  [block] isolated A/B T={T} ...")
        try:
            block_result = _bench_block_isolated(
                model, seq_len=T, batch=args.batch, wf_cfg=wf_cfg,
                warmup=args.warmup, iters=max(1, args.iters // 2),
            )
            row["block"] = block_result
            _log(
                f"  [block] mem reduction: "
                f"{block_result['memory_reduction_pct']:.1f}%  "
                f"speedup: {block_result['throughput_speedup']:.3f}x"
            )
        except Exception as exc:
            row["block"] = {"error": f"{type(exc).__name__}: {exc}"}
            _log(f"  [block] ERROR: {exc}")
        _clear()

        # Level 3: Full model prefill
        if not args.skip_model:
            _log(f"  [model] full prefill A/B T={T} ...")
            try:
                model_result = _bench_full_model(
                    model, seq_len=T, batch=args.batch,
                    chunk_size=args.chunk_size, vocab_size=vocab_size,
                    wf_cfg=wf_cfg, originals=originals,
                )
                row["model"] = model_result
                _log(
                    f"  [model] mem reduction: "
                    f"{model_result['memory_reduction_pct']:.1f}%  "
                    f"speedup: {model_result['throughput_speedup']:.3f}x"
                )
            except Exception as exc:
                row["model"] = {"error": f"{type(exc).__name__}: {exc}"}
                _log(f"  [model] ERROR: {exc}")
            _clear()

        row["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        rows.append(row)

        # Save incremental
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "command": " ".join(sys.argv),
            "model_path": args.model_path,
            "wayfinder_config": wf_cfg.__dict__,
            "chunk_size": args.chunk_size,
            "rows": rows,
        }
        results_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        _log(f"  [{i}/{total}] Done in {row['elapsed_sec']:.1f}s")

    total_sec = time.perf_counter() - run_t0
    _log(f"\nTotal benchmark time: {total_sec:.1f}s")

    # Print all three tables
    attn_table = _print_attention_table(rows)
    block_table = _print_block_table(rows)
    model_table = "" if args.skip_model else _print_model_table(rows)

    # Write README
    readme = out_dir / "README.md"
    readme.write_text(
        f"# Qwen3-1.7B A/B: Dense vs HCSA\n\n"
        f"- model: `{args.model_path}`\n"
        f"- chunk_size: {args.chunk_size}\n"
        f"- window: {args.window}, circular: {args.circular}\n"
        f"- created: {datetime.now(UTC).isoformat()}\n"
        f"\n{attn_table}\n"
        f"\n{block_table}\n"
        f"\n{model_table}\n",
        encoding="utf-8",
    )
    _log(f"\nWrote {results_path}")
    _log(f"Wrote {readme}")


if __name__ == "__main__":
    main()
