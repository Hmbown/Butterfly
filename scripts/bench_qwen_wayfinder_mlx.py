#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.compiler import compile_graph_spec
from hcsa.integrations.qwen_mlx import (
    QwenWayfinderConfig,
    QwenWayfinderAttention,
    _QWEN_GRAPH_CACHE_BY_KEY,
    _QWEN_GRAPH_CACHE_STORE,
    extract_qkv_from_qwen_attention,
    swap_qwen_attention_with_wayfinder,
)


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


def _parse_num_cycles(raw: str) -> int | str:
    value = str(raw).strip().lower()
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--num-cycles must be >= 0 or 'auto'")
    return parsed


def _resolved_cycle_count_for_degree(num_cycles: int | str, seq_len: int) -> int:
    if num_cycles == "auto":
        from hcsa.cycles import recommended_num_cycles

        return int(recommended_num_cycles(int(seq_len)))
    return int(num_cycles)


def _landmark_stride_for_budget(
    *,
    seq_len: int,
    max_degree: int,
    window: int,
    num_cycles: int,
) -> Optional[int]:
    budget = int(max_degree) - int(window) - 1 - 2 * int(num_cycles)
    if budget <= 0:
        return None
    stride = math.ceil(float(seq_len - 1) / float(budget))
    return int(max(1, stride))


def _resolve_landmark_stride(
    *,
    seq_len: int,
    base_landmark_stride: Optional[int],
    use_max_degree_budget: bool,
    max_degree: Optional[int],
    window: int,
    num_cycles: int | str,
) -> Optional[int]:
    if not use_max_degree_budget:
        return base_landmark_stride
    if max_degree is None or int(max_degree) <= 0:
        raise ValueError("--max-degree must be a positive integer when --landmark-stride-from-max-degree is used")
    cycles_for_degree = _resolved_cycle_count_for_degree(num_cycles, seq_len)
    return _landmark_stride_for_budget(
        seq_len=seq_len,
        max_degree=int(max_degree),
        window=int(window),
        num_cycles=int(cycles_for_degree),
    )


def _make_wf_cfg(
    *,
    args: argparse.Namespace,
    seq_len: int,
    num_cycles: int | str,
    base_landmark_stride: Optional[int],
) -> QwenWayfinderConfig:
    landmark_stride = _resolve_landmark_stride(
        seq_len=int(seq_len),
        base_landmark_stride=base_landmark_stride,
        use_max_degree_budget=bool(args.landmark_stride_from_max_degree),
        max_degree=None if args.max_degree is None else int(args.max_degree),
        window=int(args.window),
        num_cycles=num_cycles,
    )
    return QwenWayfinderConfig(
        path=args.path,  # type: ignore[arg-type]
        strategy="random",
        window=int(args.window),
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        edge_disjoint=bool(args.edge_disjoint),
        enforce_hamiltonian=bool(args.enforce_hamiltonian),
        seed=int(args.seed),
        edge_bias=True,
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


def _clear_graph_caches() -> None:
    _QWEN_GRAPH_CACHE_STORE.clear()
    _QWEN_GRAPH_CACHE_BY_KEY.clear()


def _post_seq_cleanup() -> None:
    _clear_graph_caches()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    gc.collect()


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    warmup: int,
    iters: int,
    dtype_name: str,
    wf_cfg: QwenWayfinderConfig,
    graph_spec: Path | None,
    graph_cache_root: Path,
    full_swap: bool,
    run_block_bench: bool,
) -> Dict[str, Any]:
    _log(f"  building inputs for T={seq_len}")
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.self_attn

    if graph_spec is not None:
        compiled = compile_graph_spec(
            graph_spec,
            T=seq_len,
            H=int(base_attn.n_heads),
            out_root=graph_cache_root,
        )
        wf_cfg = QwenWayfinderConfig(
            **{**wf_cfg.__dict__, "compiled_graph_dir": str(compiled["artifact"]["artifact_dir"])}
        )

    wf_attn = QwenWayfinderAttention(base_attn, wf_cfg)
    wf_attn.compute_edge_utilization_proxy = False
    wf_attn.compute_graph_metrics = False

    def baseline_attn_fn():
        q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
        y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
        return base_attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))

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

    wf_s, wf_mem = _bench(
        wf_attn_fn,
        warmup=max(0, warmup - 1),
        iters=iters,
        label=f"T={seq_len} wf_attn",
    )

    # Recompute outputs for sanity MAE after timed runs so benchmark iterations
    # don't retain full result tensors between stages.
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
        "wayfinder_config": {
            "path": str(wf_cfg.path),
            "window": int(wf_cfg.window),
            "landmark_stride": None if wf_cfg.landmark_stride is None else int(wf_cfg.landmark_stride),
            "num_cycles": wf_cfg.num_cycles,
            "resolved_num_cycles": int(wf_attn.graph_runtime.num_cycles),
            "edge_disjoint": bool(wf_cfg.edge_disjoint),
            "enforce_hamiltonian": bool(wf_cfg.enforce_hamiltonian),
        },
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

    if run_block_bench:
        def baseline_block_fn():
            return layer0(x, mask="causal", cache=None)

        _log("  stage: level_a baseline block")
        base_block_s, base_block_mem = _bench(
            baseline_block_fn,
            warmup=max(1, warmup // 2),
            iters=max(1, iters // 2),
            label=f"T={seq_len} baseline_block",
        )

        orig_attn = layer0.self_attn
        layer0.self_attn = wf_attn
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
            layer0.self_attn = orig_attn
    else:
        row["block"] = {"skipped": True, "reason": "run_block_bench flag disabled"}
        _log("  stage: level_a block benchmark skipped by default for memory safety")

    if full_swap:
        _log("  stage: level_b full-model swap smoke")
        orig_attn_layers = [layer.self_attn for layer in model.layers]
        replaced = swap_qwen_attention_with_wayfinder(model, cfg=wf_cfg, layer_indices=None)
        swapped_layers = [model.layers[idx].self_attn for idx in replaced]
        for idx in replaced:
            swapped = model.layers[idx].self_attn
            if isinstance(swapped, QwenWayfinderAttention):
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
            # Restore baseline model for the next seq_len; avoid cross-row contamination.
            for idx, attn in enumerate(orig_attn_layers):
                model.layers[idx].self_attn = attn
            # Drop caches created by temporary swapped modules.
            for attn in swapped_layers:
                _QWEN_GRAPH_CACHE_STORE.pop(id(attn), None)
            _QWEN_GRAPH_CACHE_BY_KEY.clear()

    return row


def _build_payload(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    config: Dict[str, Any],
    wayfinder_config: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
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
        "wayfinder_config": wayfinder_config,
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
    p = argparse.ArgumentParser(description="Qwen3 HCSA MLX benchmark")
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-1.7B-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--path", type=str, default="permute", choices=["sparse", "permute"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument(
        "--landmark-stride-from-max-degree",
        action="store_true",
        help="Compute landmark stride per seq_len from --max-degree, --window, and --num-cycles.",
    )
    p.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="Target max degree used by --landmark-stride-from-max-degree.",
    )
    p.add_argument("--num-cycles", type=str, default="1")
    p.add_argument(
        "--edge-disjoint",
        dest="edge_disjoint",
        action="store_true",
        default=True,
        help="Require edge-disjoint cycles when num_cycles > 1 (default).",
    )
    p.add_argument(
        "--no-edge-disjoint",
        dest="edge_disjoint",
        action="store_false",
        help="Allow overlapping cycles when num_cycles > 1.",
    )
    p.add_argument(
        "--allow-non-hamiltonian",
        dest="enforce_hamiltonian",
        action="store_false",
        default=True,
        help="Allow graphs without Hamiltonian backbone (e.g., num_cycles=0).",
    )
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument(
        "--run-block-bench",
        action="store_true",
        help="Run block-level benchmark stage (disabled by default for memory safety).",
    )
    p.add_argument("--full-swap", action="store_true")
    p.add_argument(
        "--allow-multi-seq",
        action="store_true",
        help="Allow multiple --seq-lens in one process. Default is one seq_len per process.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    num_cycles = _parse_num_cycles(args.num_cycles)
    base_landmark_stride = None if int(args.landmark_stride) <= 0 else int(args.landmark_stride)
    if bool(args.landmark_stride_from_max_degree) and (args.max_degree is None or int(args.max_degree) <= 0):
        raise ValueError("--max-degree must be a positive integer with --landmark-stride-from-max-degree")
    if len(args.seq_lens) > 1 and not bool(args.allow_multi_seq):
        raise ValueError(
            "Memory-safety guard: run one seq_len per process. "
            "Pass --allow-multi-seq to override."
        )

    _log(f"Loading model {args.model_path}")
    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded")

    wayfinder_config: Dict[str, Any] = {
        "path": str(args.path),
        "strategy": "random",
        "window": int(args.window),
        "landmark_stride": base_landmark_stride,
        "landmark_stride_from_max_degree": bool(args.landmark_stride_from_max_degree),
        "max_degree": None if args.max_degree is None else int(args.max_degree),
        "num_cycles": num_cycles,
        "edge_disjoint": bool(args.edge_disjoint),
        "enforce_hamiltonian": bool(args.enforce_hamiltonian),
        "seed": int(args.seed),
        "window_drop": float(args.window_drop),
        "retro_backfill_enabled": bool(args.retro_backfill),
        "retro_backfill_alpha": float(args.retro_alpha),
        "retro_backfill_training_only": bool(args.retro_training_only),
        "retro_backfill_causal_only": bool(args.retro_causal_only),
    }

    # Derive a short model tag from the model path for output directory naming
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
            wf_cfg = _make_wf_cfg(
                args=args,
                seq_len=int(T),
                num_cycles=num_cycles,
                base_landmark_stride=base_landmark_stride,
            )
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
                run_block_bench=bool(args.run_block_bench),
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
        finally:
            _post_seq_cleanup()

        payload = _build_payload(
            args=args,
            tokenizer=tokenizer,
            config=config,
            wayfinder_config=wayfinder_config,
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
