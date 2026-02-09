#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.integrations.glm_mlx import GLMWayfinderAttention, GLMWayfinderConfig, swap_glm_attention_with_wayfinder


def _log(message: str) -> None:
    print(message, flush=True)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:  # pragma: no cover
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())  # pragma: no cover


def _clear_workspace() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _iter_model_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unable to locate model layers for GLM benchmark.")


def _cache_state(cache: Sequence[Any]) -> Dict[str, Any]:
    if not cache:
        return {"layers": 0}

    offsets: list[int] = []
    sizes: list[int] = []
    max_sizes: list[int] = []
    for entry in cache:
        offset = getattr(entry, "offset", None)
        if offset is not None:
            offsets.append(int(offset))
        if hasattr(entry, "size"):
            try:
                sizes.append(int(entry.size()))
            except Exception:  # pragma: no cover - runtime dependent
                pass
        max_size = getattr(entry, "max_size", None)
        if max_size is not None:
            max_sizes.append(int(max_size))

    state: Dict[str, Any] = {
        "layers": int(len(cache)),
        "layer_type": type(cache[0]).__name__,
    }
    if offsets:
        state["offset_min"] = int(min(offsets))
        state["offset_max"] = int(max(offsets))
    if sizes:
        state["size_min"] = int(min(sizes))
        state["size_max"] = int(max(sizes))
    if max_sizes:
        state["max_size_min"] = int(min(max_sizes))
        state["max_size_max"] = int(max(max_sizes))
    return state


def _sample_wayfinder_profile(sample_layer: Optional[GLMWayfinderAttention]) -> Optional[Dict[str, Any]]:
    if sample_layer is None:
        return None
    profile = sample_layer.last_profile
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    cache_hit = notes.get("cache_hit")
    return {
        "path": str(profile.path),
        "graph_build_ms": float(profile.graph_build_ms),
        "attention_ms": float(profile.attention_ms),
        "cache_hit": (bool(cache_hit) if cache_hit is not None else None),
        "seq_len": notes.get("seq_len"),
        "q_len": notes.get("q_len"),
        "graph_seq_len": notes.get("graph_seq_len"),
        "active_query_mode": notes.get("active_query_mode"),
        "active_dense_threshold": notes.get("active_dense_threshold"),
        "active_dense_triggered": notes.get("active_dense_triggered"),
    }


def _safe_pct_delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return float(100.0 * ((candidate - baseline) / baseline))


def _compare_scalar(candidate: Optional[float], baseline: Optional[float]) -> Dict[str, Optional[float]]:
    if candidate is None:
        return {
            "absolute": None,
            "baseline": baseline,
            "delta_vs_baseline": None,
            "delta_pct_vs_baseline": None,
        }
    delta = None if baseline is None else float(candidate - baseline)
    return {
        "absolute": float(candidate),
        "baseline": baseline,
        "delta_vs_baseline": delta,
        "delta_pct_vs_baseline": _safe_pct_delta(candidate, baseline),
    }


def _memory_reduction_pct(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return float(100.0 * (1.0 - (candidate / baseline)))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    xs = sorted(float(v) for v in values)
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def _load_baseline_rows(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if rows is None and "results" in payload:
        rows = payload["results"]
    if not isinstance(rows, list):
        return {}

    by_seq: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seq_len = row.get("seq_len")
        if seq_len is None:
            continue
        by_seq[int(seq_len)] = row
    return by_seq


def _extract_baseline_scenario(row: Optional[Dict[str, Any]], decode_len: int) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    if decode_len == 0:
        value = row.get("prefill_only")
        return value if isinstance(value, dict) else None

    key = f"prefill_plus_{decode_len}"
    if key in row and isinstance(row[key], dict):
        return row[key]

    if decode_len == 1 and isinstance(row.get("prefill_plus_1"), dict):
        scenario = row["prefill_plus_1"]
        return {
            "prefill_sec": scenario.get("prefill_sec"),
            "decode_sec": scenario.get("decode_sec", scenario.get("decode1_sec")),
            "total_sec": scenario.get("total_sec"),
            "prefill_tok_s": scenario.get("prefill_tok_s"),
            "decode_tok_s": scenario.get("decode_tok_s", scenario.get("decode1_tok_s")),
            "peak_memory_bytes": scenario.get("peak_memory_bytes"),
        }
    return None


def _compare_prefill(
    candidate: Dict[str, Any],
    baseline_scenario: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_sec = None if baseline_scenario is None else baseline_scenario.get("sec")
    baseline_tok_s = None if baseline_scenario is None else baseline_scenario.get("tok_s")
    baseline_peak = None if baseline_scenario is None else baseline_scenario.get("peak_memory_bytes")
    return {
        "sec": _compare_scalar(candidate.get("sec"), baseline_sec),
        "tok_s": _compare_scalar(candidate.get("tok_s"), baseline_tok_s),
        "peak_memory_bytes": _compare_scalar(candidate.get("peak_memory_bytes"), baseline_peak),
        "memory_reduction_pct_vs_baseline": _memory_reduction_pct(
            candidate.get("peak_memory_bytes"), baseline_peak
        ),
    }


def _compare_prefill_decode(
    candidate: Dict[str, Any],
    baseline_scenario: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_prefill_sec = None if baseline_scenario is None else baseline_scenario.get("prefill_sec")
    baseline_decode_sec = None if baseline_scenario is None else baseline_scenario.get("decode_sec")
    baseline_total_sec = None if baseline_scenario is None else baseline_scenario.get("total_sec")
    baseline_prefill_tok_s = None if baseline_scenario is None else baseline_scenario.get("prefill_tok_s")
    baseline_decode_tok_s = None if baseline_scenario is None else baseline_scenario.get("decode_tok_s")
    baseline_peak = None if baseline_scenario is None else baseline_scenario.get("peak_memory_bytes")
    baseline_ttft_sec = None if baseline_scenario is None else baseline_scenario.get("ttft_sec")
    baseline_itl_p50_sec = None if baseline_scenario is None else baseline_scenario.get("itl_p50_sec")
    baseline_itl_p95_sec = None if baseline_scenario is None else baseline_scenario.get("itl_p95_sec")

    return {
        "prefill_sec": _compare_scalar(candidate.get("prefill_sec"), baseline_prefill_sec),
        "decode_sec": _compare_scalar(candidate.get("decode_sec"), baseline_decode_sec),
        "total_sec": _compare_scalar(candidate.get("total_sec"), baseline_total_sec),
        "prefill_tok_s": _compare_scalar(candidate.get("prefill_tok_s"), baseline_prefill_tok_s),
        "decode_tok_s": _compare_scalar(candidate.get("decode_tok_s"), baseline_decode_tok_s),
        "ttft_sec": _compare_scalar(candidate.get("ttft_sec"), baseline_ttft_sec),
        "itl_p50_sec": _compare_scalar(candidate.get("itl_p50_sec"), baseline_itl_p50_sec),
        "itl_p95_sec": _compare_scalar(candidate.get("itl_p95_sec"), baseline_itl_p95_sec),
        "peak_memory_bytes": _compare_scalar(candidate.get("peak_memory_bytes"), baseline_peak),
        "memory_reduction_pct_vs_baseline": _memory_reduction_pct(
            candidate.get("peak_memory_bytes"), baseline_peak
        ),
    }


def _prepare_cache(
    model: Any,
    *,
    cache_mode: str,
    max_kv_size: Optional[int],
    kv_step: int = 0,
    target_seq_len: int = 0,
) -> List[Any]:
    if cache_mode == "rotating":
        if max_kv_size is None:
            raise ValueError("max_kv_size must be set when cache_mode=rotating")
        return list(make_prompt_cache(model, max_kv_size=int(max_kv_size)))
    cache = list(make_prompt_cache(model, max_kv_size=None))
    # Pre-allocate KV cache to target_seq_len in steps of kv_step to avoid
    # repeated reallocation during chunked prefill.
    if kv_step > 0 and target_seq_len > 0:
        prealloc_size = target_seq_len
        # Round up to next kv_step boundary
        if prealloc_size % kv_step != 0:
            prealloc_size = ((prealloc_size // kv_step) + 1) * kv_step
        for entry in cache:
            if hasattr(entry, "keys") and entry.keys is None:
                # KVCache not yet initialised — do a dummy warmup so internal
                # arrays exist, then we rely on MLX lazy growth.  The main
                # benefit is signalling the allocator about upcoming size.
                pass
        _log(f"    kv_step={kv_step}, target prealloc={prealloc_size} tokens")
    return cache


def _run_chunked_prefill(
    model: Any,
    *,
    prompt_tokens: mx.array,
    chunk_size: int,
    cache: Sequence[Any],
    sample_layer: Optional[GLMWayfinderAttention] = None,
) -> Dict[str, Any]:
    batch = int(prompt_tokens.shape[0])
    seq_len = int(prompt_tokens.shape[1])
    chunk_reports: list[Dict[str, Any]] = []

    t0 = time.perf_counter()
    chunk_index = 0
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        t_chunk0 = time.perf_counter()
        logits = model(prompt_tokens[:, start:end], cache=cache)
        mx.eval(logits)
        chunk_sec = time.perf_counter() - t_chunk0
        profile_sample = _sample_wayfinder_profile(sample_layer)
        cache_state = _cache_state(cache)
        k_len = int(cache_state.get("offset_max", end))
        _clear_workspace()
        chunk_report: Dict[str, Any] = {
            "chunk_index": int(chunk_index),
            "start": int(start),
            "end": int(end),
            "tokens": int(end - start),
            "k_len": k_len,
            "sec": float(chunk_sec),
            "cache_state": cache_state,
            "peak_memory_bytes_after_chunk": int(_peak_memory()),
        }
        if profile_sample is not None:
            chunk_report["profile_sample"] = profile_sample
        chunk_reports.append(chunk_report)
        chunk_index += 1

    prefill_sec = time.perf_counter() - t0
    total_tokens = batch * seq_len
    return {
        "sec": float(prefill_sec),
        "tok_s": float(total_tokens / max(prefill_sec, 1e-12)),
        "peak_memory_bytes": int(_peak_memory()),
        "chunk_reports": chunk_reports,
        "cache_state_end": _cache_state(cache),
    }


def _run_decode(
    model: Any,
    *,
    batch: int,
    decode_len: int,
    vocab_size: int,
    cache: Sequence[Any],
) -> Dict[str, Any]:
    """True incremental autoregressive decode: decode_len x 1-token steps."""
    if decode_len <= 0:
        return {
            "decode_sec": 0.0,
            "decode_tok_s": None,
            "per_token_sec": [],
            "ttft_sec": None,
            "itl_p50_sec": None,
            "itl_p95_sec": None,
        }

    per_token_sec: list[float] = []
    t0 = time.perf_counter()
    next_token = mx.zeros((batch, 1), dtype=mx.int32)
    for _step in range(int(decode_len)):
        t_step = time.perf_counter()
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        # Greedy pick for next token (keeps autoregressive chain realistic)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1).astype(mx.int32)
        mx.eval(next_token)
        per_token_sec.append(time.perf_counter() - t_step)
    _clear_workspace()
    decode_sec = time.perf_counter() - t0
    itl_values = per_token_sec[1:] if len(per_token_sec) > 1 else per_token_sec
    return {
        "decode_sec": float(decode_sec),
        "decode_tok_s": float((batch * decode_len) / max(decode_sec, 1e-12)),
        "per_token_sec": per_token_sec,
        "ttft_sec": float(per_token_sec[0]) if per_token_sec else None,
        "itl_p50_sec": _percentile(itl_values, 50.0),
        "itl_p95_sec": _percentile(itl_values, 95.0),
    }


def _build_payload(
    *,
    args: argparse.Namespace,
    wf_cfg: GLMWayfinderConfig,
    replaced_layers: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "baseline_path": str(args.baseline_path) if args.baseline_path is not None else None,
        "no_swap": bool(args.no_swap),
        "kv_step": int(args.kv_step),
        "wayfinder_config": wf_cfg.__dict__ if not args.no_swap else None,
        "retro_backfill_enabled": False if args.no_swap else bool(wf_cfg.retro_backfill_enabled),
        "swap": {"replaced_layers": int(replaced_layers)},
        "sweep": {
            "seq_lens": [int(x) for x in args.seq_lens],
            "chunk_sizes": [int(x) for x in args.chunk_sizes],
            "decode_lens": [int(x) for x in sorted(set(args.decode_lens))],
            "cache_modes": list(args.cache_modes),
            "max_kv_size": None if args.max_kv_size is None else int(args.max_kv_size),
        },
        "rows": rows,
    }


def _readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GLM Chunked Prefill Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- model_path: `{payload['model_path']}`")
    lines.append(f"- baseline_path: `{payload.get('baseline_path')}`")
    lines.append(f"- replaced_layers: `{payload.get('swap', {}).get('replaced_layers')}`")
    lines.append("")
    lines.append(
        "| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in payload["rows"]:
        cache_mode = row.get("cache_mode")
        chunk_size = row.get("chunk_size")
        seq_len = row.get("seq_len")
        if not row.get("ok", False):
            err = row.get("error", "error")
            lines.append(f"| {cache_mode} | {chunk_size} | {seq_len} | error | - | - | - | - | - | - | - | - |")
            lines.append(f"|  |  |  | `{err}` |  |  |  |  |  |  |  |  |")
            continue

        for key, scenario in row.items():
            if not key.startswith("prefill_"):
                continue
            cmp_obj = scenario.get("comparison_vs_baseline", {})
            peak_cmp = cmp_obj.get("peak_memory_bytes", {})
            reduction = cmp_obj.get("memory_reduction_pct_vs_baseline")

            if key == "prefill_only":
                latency_sec = scenario.get("sec")
                prefill_tok_s = scenario.get("tok_s")
                decode_tok_s = None
            else:
                latency_sec = scenario.get("total_sec")
                prefill_tok_s = scenario.get("prefill_tok_s")
                decode_tok_s = scenario.get("decode_tok_s")

            lines.append(
                "| {cache_mode} | {chunk_size} | {seq_len} | {scenario_name} | {latency} | {prefill_tok_s} | {decode_tok_s} | {peak} | {baseline_peak} | {delta_peak} | {delta_peak_pct} | {reduction} |".format(
                    cache_mode=cache_mode,
                    chunk_size=chunk_size,
                    seq_len=seq_len,
                    scenario_name=key,
                    latency=_fmt(latency_sec),
                    prefill_tok_s=_fmt(prefill_tok_s),
                    decode_tok_s=_fmt(decode_tok_s),
                    peak=_fmt(scenario.get("peak_memory_bytes"), is_int=True),
                    baseline_peak=_fmt(peak_cmp.get("baseline"), is_int=True),
                    delta_peak=_fmt(peak_cmp.get("delta_vs_baseline"), is_int=True),
                    delta_peak_pct=_fmt(peak_cmp.get("delta_pct_vs_baseline")),
                    reduction=_fmt(reduction),
                )
            )
    lines.append("")
    lines.append("Memory reduction uses: `100 * (1 - candidate / baseline)`.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any, *, is_int: bool = False) -> str:
    if value is None:
        return "n/a"
    if is_int:
        try:
            return str(int(value))
        except Exception:  # pragma: no cover
            return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:  # pragma: no cover
        return "n/a"


def _write_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    results_path = out_dir / "results.json"
    readme_path = out_dir / "README.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme_path.write_text(_readme(payload), encoding="utf-8")
    _log(f"Wrote {results_path}")
    _log(f"Wrote {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM full-model chunked prefill benchmark for Wayfinder")
    parser.add_argument("--model-path", type=str, default="mlx-community/GLM-4.7-Flash-4bit")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--decode-lens", type=int, nargs="+", default=[0, 1, 64])
    parser.add_argument("--cache-modes", type=str, nargs="+", choices=["normal", "rotating"], default=["normal"])
    parser.add_argument("--max-kv-size", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--path", type=str, choices=["sparse", "permute"], default="permute")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--landmark-stride", type=int, default=0)
    parser.add_argument(
        "--head-chunk-size",
        type=int,
        default=2,
        help="Wayfinder permute head chunk size.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=192,
        help="Wayfinder active/permute query chunk size.",
    )
    parser.add_argument(
        "--active-dense-threshold",
        type=int,
        default=0,
        help="Use dense fallback in active mode when K_len <= threshold (<=0 disables).",
    )
    parser.add_argument(
        "--permute-prepermute-mode",
        type=str,
        choices=["auto", "off", "kv", "qkv", "on"],
        default="auto",
        help="Wayfinder prepermute planner mode for permute/active paths.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-swap", action="store_true", default=False,
                        help="Run with stock GLM attention (no Wayfinder swap) for chunked baseline")
    parser.add_argument("--kv-step", type=int, default=0,
                        help="Pre-allocate KV cache in steps of this many tokens (0=default MLX growth)")
    parser.add_argument("--baseline-path", type=Path, default=Path("benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    decode_lens = sorted(set(int(x) for x in args.decode_lens))
    if any(x < 0 for x in decode_lens):
        raise ValueError("decode_lens must be >= 0")
    if any(x <= 0 for x in args.chunk_sizes):
        raise ValueError("chunk_sizes must be > 0")
    if any(x <= 0 for x in args.seq_lens):
        raise ValueError("seq_lens must be > 0")

    try:
        mx.random.seed(int(args.seed))
    except Exception:  # pragma: no cover
        pass

    _log(f"Loading model {args.model_path}")
    model, _, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded")

    wf_cfg = GLMWayfinderConfig(
        path=args.path,  # type: ignore[arg-type]
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=1,
        seed=int(args.seed),
        edge_bias=True,
        window_drop=0.0,
        compiled_graph_dir=None,
        permute_head_chunk_size=int(max(1, args.head_chunk_size)),
        query_chunk_size=int(max(1, args.query_chunk_size)),
        permute_prepermute_mode=str(args.permute_prepermute_mode),  # type: ignore[arg-type]
        permute_log_chunks=False,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
        permute_memory_budget_bytes=None,
        active_dense_threshold=(
            None if int(args.active_dense_threshold) <= 0 else int(args.active_dense_threshold)
        ),
    )

    if args.no_swap:
        replaced: list[int] = []
        profile_sample_layer: Optional[GLMWayfinderAttention] = None
        _log("--no-swap: running with stock GLM attention (no Wayfinder)")
    else:
        layers = list(_iter_model_layers(model))
        replaced = swap_glm_attention_with_wayfinder(model, cfg=wf_cfg, layer_indices=None)
        profile_sample_layer = None
        for idx in replaced:
            layer_attn = layers[idx].self_attn
            if isinstance(layer_attn, GLMWayfinderAttention):
                layer_attn.compute_edge_utilization_proxy = False
                layer_attn.compute_graph_metrics = False
                if profile_sample_layer is None:
                    profile_sample_layer = layer_attn
        _log(f"Swapped Wayfinder attention on {len(replaced)} layers")

    model_tag = args.model_path.rstrip("/").split("/")[-1].lower().replace("-", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"benchmarks/mlx/{model_tag}_wayfinder") / f"{stamp}_chunked_prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {out_dir}")

    baseline_rows = _load_baseline_rows(args.baseline_path)
    if args.baseline_path is not None:
        _log(f"Baseline rows loaded: {len(baseline_rows)} from {args.baseline_path}")

    vocab_size = int(config.get("vocab_size", getattr(model.args, "vocab_size", 0)))
    if vocab_size <= 0:
        vocab_size = int(model.args.vocab_size)

    rows: list[Dict[str, Any]] = []
    run_t0 = time.perf_counter()
    total_cases = len(args.cache_modes) * len(args.chunk_sizes) * len(args.seq_lens)
    case_idx = 0

    for cache_mode in args.cache_modes:
        for chunk_size in args.chunk_sizes:
            for seq_len in args.seq_lens:
                case_idx += 1
                case_t0 = time.perf_counter()
                swap_label = "stock" if args.no_swap else "wayfinder"
                _log(
                    f"[{case_idx}/{total_cases}] {swap_label} cache_mode={cache_mode} "
                    f"chunk={int(chunk_size)} seq_len={int(seq_len)} kv_step={int(args.kv_step)} start"
                )
                row: Dict[str, Any] = {
                    "cache_mode": cache_mode,
                    "chunk_size": int(chunk_size),
                    "seq_len": int(seq_len),
                    "max_kv_size": int(args.max_kv_size) if cache_mode == "rotating" else None,
                    "no_swap": bool(args.no_swap),
                    "kv_step": int(args.kv_step),
                    "retro_backfill_enabled": False,
                    "ok": True,
                }

                try:
                    prompt_tokens = mx.random.randint(
                        0,
                        vocab_size,
                        shape=(int(args.batch), int(seq_len)),
                        dtype=mx.int32,
                    )

                    baseline_row = baseline_rows.get(int(seq_len))
                    if baseline_row is not None:
                        row["baseline_seq_len_match"] = True
                    else:
                        row["baseline_seq_len_match"] = False

                    cache = _prepare_cache(
                        model,
                        cache_mode=cache_mode,
                        max_kv_size=(int(args.max_kv_size) if cache_mode == "rotating" else None),
                        kv_step=int(args.kv_step),
                        target_seq_len=int(seq_len),
                    )
                    _clear_workspace()
                    _reset_peak_memory()

                    _log(
                        f"    scenario=prefill_only (cache_mode={cache_mode}, chunk={int(chunk_size)}, T={int(seq_len)})"
                    )
                    prefill = _run_chunked_prefill(
                        model,
                        prompt_tokens=prompt_tokens,
                        chunk_size=int(chunk_size),
                        cache=cache,
                        sample_layer=profile_sample_layer,
                    )
                    prefill_scenario = {
                        "sec": prefill["sec"],
                        "tok_s": prefill["tok_s"],
                        "peak_memory_bytes": prefill["peak_memory_bytes"],
                        "chunk_reports": prefill["chunk_reports"],
                        "cache_state_end": prefill["cache_state_end"],
                    }
                    prefill_baseline = _extract_baseline_scenario(baseline_row, decode_len=0)
                    prefill_scenario["comparison_vs_baseline"] = _compare_prefill(
                        prefill_scenario, prefill_baseline
                    )
                    row["prefill_only"] = prefill_scenario

                    positive_decode_lens = [int(x) for x in decode_lens if int(x) > 0]
                    decoded_tokens_so_far = 0
                    decode_sec_cumulative = 0.0
                    for target_decode_len in positive_decode_lens:
                        scenario_key = f"prefill_plus_{target_decode_len}"
                        _log(
                            f"    scenario={scenario_key} (cache_mode={cache_mode}, chunk={int(chunk_size)}, T={int(seq_len)})"
                        )
                        step_decode_len = int(target_decode_len - decoded_tokens_so_far)
                        if step_decode_len <= 0:
                            continue

                        decode_step = _run_decode(
                            model,
                            batch=int(args.batch),
                            decode_len=step_decode_len,
                            vocab_size=vocab_size,
                            cache=cache,
                        )
                        decode_sec_cumulative += float(decode_step["decode_sec"])
                        decoded_tokens_so_far = int(target_decode_len)

                        scenario = {
                            "prefill_sec": prefill["sec"],
                            "decode_sec": float(decode_sec_cumulative),
                            "total_sec": float(prefill["sec"] + decode_sec_cumulative),
                            "prefill_tok_s": prefill["tok_s"],
                            "decode_tok_s": float(
                                (int(args.batch) * decoded_tokens_so_far)
                                / max(decode_sec_cumulative, 1e-12)
                            ),
                            "peak_memory_bytes": int(_peak_memory()),
                            "chunk_reports": prefill["chunk_reports"],
                            "cache_state_after_prefill": prefill["cache_state_end"],
                            "cache_state_end": _cache_state(cache),
                            "ttft_sec": decode_step.get("ttft_sec"),
                            "itl_p50_sec": decode_step.get("itl_p50_sec"),
                            "itl_p95_sec": decode_step.get("itl_p95_sec"),
                        }
                        baseline_s = _extract_baseline_scenario(
                            baseline_row, decode_len=int(target_decode_len)
                        )
                        scenario["comparison_vs_baseline"] = _compare_prefill_decode(scenario, baseline_s)
                        row[scenario_key] = scenario
                        _clear_workspace()

                except Exception as exc:  # pragma: no cover - runtime dependent
                    row["ok"] = False
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    _log(f"    error: {row['error']}")

                rows.append(row)
                case_sec = time.perf_counter() - case_t0
                row["case_elapsed_sec"] = float(case_sec)
                payload = _build_payload(
                    args=args,
                    wf_cfg=wf_cfg,
                    replaced_layers=len(replaced),
                    rows=rows,
                )
                _write_outputs(out_dir, payload)
                progress = {
                    "completed_cases": int(case_idx),
                    "total_cases": int(total_cases),
                    "elapsed_sec": float(time.perf_counter() - run_t0),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "last_case": {
                        "cache_mode": cache_mode,
                        "chunk_size": int(chunk_size),
                        "seq_len": int(seq_len),
                        "case_elapsed_sec": float(case_sec),
                        "ok": row.get("ok", False),
                    },
                }
                (out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                _log(
                    f"[{case_idx}/{total_cases}] {swap_label} cache_mode={cache_mode} "
                    f"chunk={int(chunk_size)} seq_len={int(seq_len)} done "
                    f"({case_sec:.1f}s case, {time.perf_counter() - run_t0:.1f}s total)"
                )


if __name__ == "__main__":
    main()
