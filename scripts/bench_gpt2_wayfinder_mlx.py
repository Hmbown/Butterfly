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

from hcsa.compiler import compile_graph_spec
from hcsa.integrations.gpt2_mlx import (
    GPT2WayfinderAttention,
    GPT2WayfinderConfig,
    extract_qkv_from_gpt2_attention,
    swap_gpt2_attention_with_wayfinder,
)
from hcsa.integrations.qwen_mlx import _QWEN_GRAPH_CACHE_BY_KEY, _QWEN_GRAPH_CACHE_STORE


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


def _log(message: str) -> None:
    print(message, flush=True)


def _bench(fn, *, warmup: int, iters: int, label: str):
    warmup_n = max(1, warmup)
    iter_n = max(1, iters)
    for i in range(warmup_n):
        _log(f"    {label}: warmup {i + 1}/{warmup_n}")
        y = fn()
        mx.eval(y)

    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    _reset_peak_memory()
    times: list[float] = []
    for i in range(iter_n):
        _log(f"    {label}: iter {i + 1}/{iter_n} start")
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        dt = time.perf_counter() - t0
        times.append(dt)
        _log(f"    {label}: iter {i + 1}/{iter_n} done in {dt * 1000.0:.1f} ms")
    return _median(times), _peak_memory()


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    warmup: int,
    iters: int,
    dtype_name: str,
    wf_cfg: GPT2WayfinderConfig,
    graph_spec: Path | None,
    graph_cache_root: Path,
    full_swap: bool,
    permute_memory_budget_multiplier: float,
) -> Dict[str, Any]:
    _log(f"  building inputs for T={seq_len}")
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.n_embd)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.attn

    if graph_spec is not None:
        compiled = compile_graph_spec(
            graph_spec,
            T=seq_len,
            H=int(base_attn.n_head),
            out_root=graph_cache_root,
        )
        wf_cfg = GPT2WayfinderConfig(
            **{**wf_cfg.__dict__, "compiled_graph_dir": str(compiled["artifact"]["artifact_dir"])}
        )

    wf_attn = GPT2WayfinderAttention(base_attn, wf_cfg)
    wf_attn.eval()
    wf_attn.compute_edge_utilization_proxy = False
    wf_attn.compute_graph_metrics = False

    def baseline_attn_fn():
        q, k, v = extract_qkv_from_gpt2_attention(base_attn, x, cache=None)
        y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
        return base_attn.c_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))

    def wf_attn_fn():
        return wf_attn(x, mask="causal", cache=None)

    _log("  stage: level_a baseline attention")
    dense_s, dense_mem = _bench(
        baseline_attn_fn,
        warmup=warmup,
        iters=iters,
        label=f"T={seq_len} baseline_attn",
    )

    _log("  stage: level_a first Wayfinder call (build/cache)")
    mx.eval(wf_attn_fn())
    first_profile = wf_attn.last_profile.to_dict()
    _log("  stage: level_a cached Wayfinder attention")
    wf_mem_budget = int(max(0.0, permute_memory_budget_multiplier) * float(dense_mem))
    wf_attn.set_runtime_controls(memory_budget_bytes=wf_mem_budget)

    wf_s, wf_mem = _bench(
        wf_attn_fn,
        warmup=max(0, warmup - 1),
        iters=iters,
        label=f"T={seq_len} wf_attn",
    )

    wf_out = wf_attn_fn()
    mx.eval(wf_out)
    dense_out_ref = baseline_attn_fn()
    mx.eval(dense_out_ref)
    diff = mx.abs(dense_out_ref.astype(mx.float32) - wf_out.astype(mx.float32))
    mae = mx.mean(diff)
    mx.eval(mae)

    profile = wf_attn.last_profile.to_dict()
    row: Dict[str, Any] = {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "level_a_real_qkv": {
            "sanity_mae": float(mae.item()),
            "baseline_attention": {
                "tokens_per_sec": float((batch * seq_len) / max(dense_s, 1e-12)),
                "latency_ms": float(dense_s * 1000.0),
                "peak_memory_bytes": int(dense_mem),
            },
            "wayfinder_attention": {
                "tokens_per_sec": float((batch * seq_len) / max(wf_s, 1e-12)),
                "latency_ms": float(wf_s * 1000.0),
                "peak_memory_bytes": int(wf_mem),
                "graph_build_ms_first": float(first_profile.get("graph_build_ms", 0.0)),
                "graph_build_ms_cached": float(profile.get("graph_build_ms", 0.0)),
                "cache_hit_rate": float(1.0 if bool(profile.get("cache_hit", False)) else 0.0),
                "cache_source": profile.get("cache_source"),
            },
        },
    }

    def baseline_block_fn():
        return layer0(x, mask="causal", cache=None)

    _log("  stage: level_a baseline block")
    base_block_s, base_block_mem = _bench(
        baseline_block_fn,
        warmup=max(1, warmup // 2),
        iters=max(1, iters // 2),
        label=f"T={seq_len} baseline_block",
    )

    orig_attn = layer0.attn
    layer0.attn = wf_attn
    try:

        def wf_block_fn():
            return layer0(x, mask="causal", cache=None)

        _log("  stage: level_a Wayfinder block")
        wf_block_s, wf_block_mem = _bench(
            wf_block_fn,
            warmup=max(1, warmup // 2),
            iters=max(1, iters // 2),
            label=f"T={seq_len} wf_block",
        )
        row["block"] = {
            "baseline": {
                "tokens_per_sec": float((batch * seq_len) / max(base_block_s, 1e-12)),
                "latency_ms": float(base_block_s * 1000.0),
                "peak_memory_bytes": int(base_block_mem),
            },
            "wayfinder": {
                "tokens_per_sec": float((batch * seq_len) / max(wf_block_s, 1e-12)),
                "latency_ms": float(wf_block_s * 1000.0),
                "peak_memory_bytes": int(wf_block_mem),
                "cache_persistent_bytes": int(wf_attn.cache_persistent_bytes()),
                "edge_utilization_proxy": wf_attn.last_edge_utilization_proxy,
                "graph_metrics": wf_attn.last_graph_metrics,
            },
        }
    finally:
        layer0.attn = orig_attn

    if full_swap:
        _log("  stage: level_b full-model swap smoke")
        orig_attn_layers = [layer.attn for layer in model.layers]
        replaced = swap_gpt2_attention_with_wayfinder(model, cfg=wf_cfg, layer_indices=None)
        swapped_layers = [model.layers[idx].attn for idx in replaced]
        for idx in replaced:
            swapped = model.layers[idx].attn
            if isinstance(swapped, GPT2WayfinderAttention):
                swapped.compute_edge_utilization_proxy = False
                swapped.compute_graph_metrics = False
        try:
            z = mx.random.randint(
                0,
                int(model.args.vocab_size),
                shape=(1, min(256, seq_len)),
                dtype=mx.int32,
            )
            _reset_peak_memory()
            t0 = time.perf_counter()
            out = model(z)
            mx.eval(out)
            row["level_b_full_swap_smoke"] = {
                "replaced_layers": replaced,
                "tokens_per_sec": float(
                    (z.shape[0] * z.shape[1]) / max(time.perf_counter() - t0, 1e-12)
                ),
                "peak_memory_bytes": int(_peak_memory()),
                "seq_len": int(z.shape[1]),
            }
        finally:
            for idx, attn in enumerate(orig_attn_layers):
                model.layers[idx].attn = attn
            for attn in swapped_layers:
                _QWEN_GRAPH_CACHE_STORE.pop(id(attn), None)
            _QWEN_GRAPH_CACHE_BY_KEY.clear()

    return row


def _build_payload(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    config: Dict[str, Any],
    wf_cfg: GPT2WayfinderConfig,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "model_path": args.model_path,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "model_config": {
            "model_type": config.get("model_type"),
            "n_head": config.get("n_head"),
            "n_layer": config.get("n_layer"),
            "n_embd": config.get("n_embd"),
            "n_positions": config.get("n_positions"),
            "vocab_size": config.get("vocab_size"),
        },
        "wayfinder_config": wf_cfg.__dict__,
        "results": rows,
    }


def _write_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    results_path = out_dir / "results.json"
    readme_path = out_dir / "README.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme_path.write_text(_readme(payload), encoding="utf-8")
    _log(f"Wrote {results_path}")
    _log(f"Wrote {readme_path}")


def _readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    model_name = payload["model_path"].rstrip("/").split("/")[-1]
    lines.append(f"# {model_name} HCSA MLX Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- model_path: `{payload['model_path']}`")
    lines.append("")
    lines.append("| T | dense attn tok/s | wayfinder attn tok/s | dense attn mem | wayfinder attn mem |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in payload["results"]:
        if "error" in row:
            lines.append(f"| {row['seq_len']} | error | error | error | error |")
            continue
        a = row["level_a_real_qkv"]
        lines.append(
            "| {T} | {d:.1f} | {h:.1f} | {dm} | {hm} |".format(
                T=row["seq_len"],
                d=a["baseline_attention"]["tokens_per_sec"],
                h=a["wayfinder_attention"]["tokens_per_sec"],
                dm=a["baseline_attention"]["peak_memory_bytes"],
                hm=a["wayfinder_attention"]["peak_memory_bytes"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="GPT-2 HCSA MLX benchmark")
    p.add_argument("--model-path", type=str, default="openai-community/gpt2")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--path", type=str, default="permute", choices=["sparse", "permute"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-head-chunk-size", type=int, default=8)
    p.add_argument("--permute-query-chunk-size", type=int, default=256)
    p.add_argument(
        "--permute-prepermute-mode",
        type=str,
        choices=["auto", "off", "kv", "qkv", "on"],
        default="auto",
    )
    p.add_argument(
        "--permute-memory-budget-multiplier",
        type=float,
        default=1.0,
        help="Auto planner memory budget = multiplier * dense attention peak memory at the same seq len.",
    )
    p.add_argument("--edge-bias", action="store_true")
    p.add_argument("--window-drop", type=float, default=0.0)
    p.add_argument("--retro-backfill", action="store_true")
    p.add_argument("--retro-alpha", type=float, default=0.2)
    p.add_argument(
        "--retro-training-only",
        dest="retro_training_only",
        action="store_true",
        default=True,
        help="Apply retro backfill only during training calls (default).",
    )
    p.add_argument(
        "--retro-allow-inference",
        dest="retro_training_only",
        action="store_false",
        help="Allow retro backfill during inference/benchmark calls.",
    )
    p.add_argument(
        "--retro-causal-only",
        dest="retro_causal_only",
        action="store_true",
        default=True,
        help="Allow retro backfill only when successor is not in original future (default).",
    )
    p.add_argument(
        "--retro-allow-future",
        dest="retro_causal_only",
        action="store_false",
        help="Allow retro backfill from cycle-successor even if it is future in original order.",
    )
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    p.add_argument("--full-swap", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    _log(f"Loading model {args.model_path}")
    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded")

    wf_cfg = GPT2WayfinderConfig(
        path=args.path,  # type: ignore[arg-type]
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=int(args.num_cycles),
        seed=int(args.seed),
        permute_head_chunk_size=int(args.permute_head_chunk_size),
        query_chunk_size=int(args.permute_query_chunk_size),
        permute_prepermute_mode=str(args.permute_prepermute_mode),
        edge_bias=bool(args.edge_bias),
        window_drop=float(args.window_drop),
        compiled_graph_dir=None,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        permute_log_chunks=False,
        retro_backfill_enabled=bool(args.retro_backfill),
        retro_backfill_alpha=float(args.retro_alpha),
        retro_backfill_training_only=bool(args.retro_training_only),
        retro_backfill_causal_only=bool(args.retro_causal_only),
    )

    model_tag = args.model_path.rstrip("/").split("/")[-1].lower().replace("-", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"benchmarks/mlx/{model_tag}_wayfinder") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {out_dir}")

    rows: list[Dict[str, Any]] = []
    run_t0 = time.perf_counter()
    total = len(args.seq_lens)
    for i, T in enumerate(args.seq_lens, start=1):
        seq_t0 = time.perf_counter()
        _log(f"[{i}/{total}] seq_len={int(T)} start")
        try:
            row = _bench_one_seq(
                model,
                seq_len=int(T),
                batch=int(args.batch),
                warmup=int(args.warmup),
                iters=int(args.iters),
                dtype_name=args.dtype,
                wf_cfg=wf_cfg,
                graph_spec=args.graph_spec,
                graph_cache_root=args.graph_cache_root,
                full_swap=bool(args.full_swap),
                permute_memory_budget_multiplier=float(args.permute_memory_budget_multiplier),
            )
            rows.append(row)
            _log(f"[{i}/{total}] seq_len={int(T)} done in {time.perf_counter() - seq_t0:.2f}s")
        except Exception as exc:  # pragma: no cover - runtime dependent
            rows.append(
                {
                    "seq_len": int(T),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(f"[{i}/{total}] seq_len={int(T)} error: {type(exc).__name__}: {exc}")

        payload = _build_payload(
            args=args,
            tokenizer=tokenizer,
            config=config,
            wf_cfg=wf_cfg,
            rows=rows,
        )
        _write_outputs(out_dir, payload)
        progress = {
            "completed_seq_lens": int(i),
            "total_seq_lens": int(total),
            "last_seq_len": int(T),
            "elapsed_sec": float(time.perf_counter() - run_t0),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (out_dir / "progress.json").write_text(
            json.dumps(progress, indent=2),
            encoding="utf-8",
        )
        _log(f"Elapsed total: {time.perf_counter() - run_t0:.2f}s")


if __name__ == "__main__":
    main()
