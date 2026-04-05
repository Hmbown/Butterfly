#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import json
import os
import numpy as np
import re
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.qwen_mlx import (  # noqa: E402
    QwenWayfinderAttention,
    QwenWayfinderConfig,
    get_qwen_full_attention_layer_indices,
    swap_qwen_attention_with_wayfinder,
    validate_qwen35_full_attention_layers,
)
from bna.integrations.qwen_mlx_loader import (  # noqa: E402
    load_qwen_mlx_model,
    resolve_qwen_mlx_model_path,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _configure_hf_cache(
    *,
    hf_home: Optional[str],
    hf_hub_cache: Optional[str],
    hf_offline: bool,
) -> Dict[str, Optional[str]]:
    default_home = Path("/Volumes/VIXinSSD/hf_cache")
    resolved_home = str(default_home) if default_home.exists() else None
    if hf_home:
        resolved_home = str(Path(hf_home).expanduser())
    elif os.environ.get("HF_HOME"):
        resolved_home = str(Path(os.environ["HF_HOME"]).expanduser())

    resolved_hub = None
    if hf_hub_cache:
        resolved_hub = str(Path(hf_hub_cache).expanduser())
    elif os.environ.get("HF_HUB_CACHE"):
        resolved_hub = str(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    elif resolved_home:
        resolved_hub = str(Path(resolved_home) / "hub")

    if resolved_home:
        os.environ["HF_HOME"] = resolved_home
    if resolved_hub:
        os.environ["HF_HUB_CACHE"] = resolved_hub
    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    return {
        "hf_home": resolved_home,
        "hf_hub_cache": resolved_hub,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


@contextmanager
def _stage_timeout(
    timeout_sec: float,
    *,
    stage: str,
    seq_len: Optional[int] = None,
    repeat: Optional[int] = None,
) -> Any:
    timeout = float(max(0.0, timeout_sec))
    if timeout <= 0.0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        context = f"stage={stage}"
        if seq_len is not None:
            context += f" seq_len={int(seq_len)}"
        if repeat is not None:
            context += f" repeat={int(repeat)}"
        raise TimeoutError(f"timeout after {timeout:.1f}s ({context})")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _clear_workspace() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _iter_model_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unable to locate model layers for Qwen benchmark.")


def _resolve_position_limit(model_config: Any, tokenizer: Any = None) -> Optional[int]:
    candidates: List[Any] = []
    if isinstance(model_config, dict):
        text_cfg = model_config.get("text_config")
        if isinstance(text_cfg, dict):
            candidates.extend(
                [
                    text_cfg.get("max_position_embeddings"),
                    text_cfg.get("model_max_length"),
                ]
            )
        candidates.extend(
            [
                model_config.get("max_position_embeddings"),
                model_config.get("model_max_length"),
            ]
        )
    if tokenizer is not None:
        candidates.extend(
            [
                getattr(tokenizer, "model_max_length", None),
                getattr(getattr(tokenizer, "tokenizer", None), "model_max_length", None),
                getattr(getattr(tokenizer, "_tokenizer", None), "model_max_length", None),
            ]
        )

    for raw_value in candidates:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if 0 < value < 10_000_000:
            return value
    return None


def _collect_hsa_trace_snapshot(
    model: Any,
    *,
    layer_indices: Optional[Sequence[int]] = None,
    max_layers: int = 0,
) -> List[Dict[str, Any]]:
    layers = list(_iter_model_layers(model))
    if layer_indices is not None:
        selected = [int(idx) for idx in layer_indices if 0 <= int(idx) < len(layers)]
    else:
        selected = [
            i
            for i, layer in enumerate(layers)
            if isinstance(getattr(layer, "self_attn", None), QwenWayfinderAttention)
        ]
    if int(max_layers) > 0:
        selected = selected[: int(max_layers)]

    out: List[Dict[str, Any]] = []
    note_keys = (
        "seq_len",
        "graph_seq_len",
        "q_len",
        "cache_hit",
        "cache_source",
        "cache_mode",
        "cache_persistent_bytes",
        "active_query_mode",
        "dense_fallback_reason",
        "active_dense_triggered",
        "active_large_q_dense_triggered",
        "adaptive_graph_reuse",
        "decode_local_tail_tokens",
    )
    for idx in selected:
        attn = getattr(layers[idx], "self_attn", None)
        if not isinstance(attn, QwenWayfinderAttention):
            continue
        profile = attn.last_profile.to_dict() if hasattr(attn, "last_profile") else {}
        notes_obj = profile.get("notes")
        notes_source = notes_obj if isinstance(notes_obj, dict) and notes_obj else profile
        notes_subset = {k: notes_source.get(k) for k in note_keys if k in notes_source}
        out.append(
            {
                "layer_idx": int(idx),
                "path": profile.get("path"),
                "graph_build_ms": profile.get("graph_build_ms"),
                "permute_ms": profile.get("permute_ms"),
                "attention_ms": profile.get("attention_ms"),
                "total_ms": profile.get("total_ms"),
                "notes": notes_subset,
            }
        )
    return out


def _collect_hsa_layer_snapshot(
    model: Any,
    *,
    layer_indices: Optional[Sequence[int]] = None,
    max_layers: int = 0,
) -> List[Dict[str, Any]]:
    """Backward-compatible alias for existing call sites."""
    return _collect_hsa_trace_snapshot(
        model,
        layer_indices=layer_indices,
        max_layers=max_layers,
    )


def _public_path_label(raw_path: str) -> str:
    if raw_path == "permute":
        return "butterfly_attention"
    if raw_path == "permute_dense_fallback":
        return "butterfly_stock_fallback"
    if raw_path in {"dense", "native_hybrid"}:
        return "stock_attention"
    return raw_path


def _public_stock_fallback_reason(
    *,
    raw_reason: Any,
    notes: Dict[str, Any],
    raw_path: str,
) -> Optional[str]:
    reason = "" if raw_reason is None else str(raw_reason).strip()
    if reason and reason.lower() not in {"none", "null"}:
        mapping = {
            "wayfinder_decode_dense": "decode_backend_stock",
            "active_dense_threshold": "stock_threshold",
            "active_large_q": "stock_large_query",
            "q_len_mismatch": "sequence_shape_guard",
            "active_runtime_error": "butterfly_runtime_guard",
            "unspecified": "stock_fallback",
        }
        return mapping.get(reason, reason)
    if bool(notes.get("active_dense_triggered")):
        return "stock_threshold"
    if bool(notes.get("active_large_q_dense_triggered")):
        return "stock_large_query"
    if "dense_fallback" in raw_path:
        return "stock_fallback"
    return None


def _summarize_hsa_trace(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    path_counts: Counter[str] = Counter()
    cache_source_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    graph_seq_len_counts: Counter[str] = Counter()
    q_len_counts: Counter[str] = Counter()
    stock_reason_counts: Counter[str] = Counter()
    graph_build_ms_values: List[float] = []
    butterfly_ms_values: List[float] = []
    attention_ms_values: List[float] = []
    total_ms_values: List[float] = []
    decode_butterfly_ms_values: List[float] = []
    prefill_butterfly_ms_values: List[float] = []

    total_obs = 0
    fallback_layer_obs = 0
    decode_layer_obs = 0
    decode_fallback_layer_obs = 0
    prefill_layer_obs = 0
    prefill_fallback_layer_obs = 0
    decode_step_obs = 0
    decode_fallback_steps = 0
    prefill_step_obs = 0
    prefill_fallback_steps = 0
    active_query_obs = 0
    adaptive_reuse_obs = 0
    cache_hit_obs = 0

    for sample in samples:
        phase = str(sample.get("phase", "unknown"))
        phase_counts[phase] += 1
        if phase == "decode":
            decode_step_obs += 1
        elif phase == "prefill":
            prefill_step_obs += 1
        sample_has_fallback = False
        for layer_row in sample.get("layers", []):
            total_obs += 1
            raw_path = str(layer_row.get("path") or "unknown")
            public_path = _public_path_label(raw_path)
            path_counts[public_path] += 1
            notes = layer_row.get("notes") or {}
            graph_build_ms = layer_row.get("graph_build_ms")
            if graph_build_ms is not None:
                graph_build_ms_values.append(float(graph_build_ms))
            butterfly_ms = layer_row.get("permute_ms")
            if butterfly_ms is not None:
                butterfly_ms_values.append(float(butterfly_ms))
                if phase == "decode":
                    decode_butterfly_ms_values.append(float(butterfly_ms))
                elif phase == "prefill":
                    prefill_butterfly_ms_values.append(float(butterfly_ms))
            attention_ms = layer_row.get("attention_ms")
            if attention_ms is not None:
                attention_ms_values.append(float(attention_ms))
            total_ms = layer_row.get("total_ms")
            if total_ms is not None:
                total_ms_values.append(float(total_ms))

            cache_source = notes.get("cache_source")
            if cache_source is not None:
                cache_source_counts[str(cache_source)] += 1
            if bool(notes.get("active_query_mode")):
                active_query_obs += 1
            if bool(notes.get("adaptive_graph_reuse")):
                adaptive_reuse_obs += 1
            if bool(notes.get("cache_hit")):
                cache_hit_obs += 1

            graph_seq_len = notes.get("graph_seq_len")
            if graph_seq_len is not None:
                graph_seq_len_counts[str(int(graph_seq_len))] += 1
            q_len = notes.get("q_len")
            if q_len is not None:
                q_len_counts[str(int(q_len))] += 1

            stock_reason = _public_stock_fallback_reason(
                raw_reason=notes.get("dense_fallback_reason"),
                notes=notes,
                raw_path=raw_path,
            )
            is_stock_fallback = stock_reason is not None

            if phase == "decode":
                decode_layer_obs += 1
            elif phase == "prefill":
                prefill_layer_obs += 1
            if is_stock_fallback:
                stock_reason_counts[stock_reason] += 1
                fallback_layer_obs += 1
                sample_has_fallback = True
                if phase == "decode":
                    decode_fallback_layer_obs += 1
                elif phase == "prefill":
                    prefill_fallback_layer_obs += 1
        if sample_has_fallback:
            if phase == "decode":
                decode_fallback_steps += 1
            elif phase == "prefill":
                prefill_fallback_steps += 1

    stock_fallback_share_run = float(fallback_layer_obs / total_obs) if total_obs else None
    stock_fallback_share_decode_layers = (
        float(decode_fallback_layer_obs / decode_layer_obs) if decode_layer_obs else None
    )
    stock_fallback_share_prefill_layers = (
        float(prefill_fallback_layer_obs / prefill_layer_obs) if prefill_layer_obs else None
    )
    stock_fallback_share_decode_steps = (
        float(decode_fallback_steps / decode_step_obs) if decode_step_obs else None
    )
    stock_fallback_share_prefill_steps = (
        float(prefill_fallback_steps / prefill_step_obs) if prefill_step_obs else None
    )

    return {
        "sample_count": int(len(samples)),
        "layer_observations": int(total_obs),
        "phase_counts": dict(phase_counts),
        "path_counts": dict(path_counts),
        "cache_source_counts": dict(cache_source_counts),
        "graph_seq_len_counts": dict(graph_seq_len_counts),
        "q_len_counts": dict(q_len_counts),
        "stock_fallback_reason_counts": dict(stock_reason_counts),
        "stock_fallback_layer_observations": int(fallback_layer_obs),
        "decode_layer_observations": int(decode_layer_obs),
        "stock_fallback_decode_layer_observations": int(decode_fallback_layer_obs),
        "prefill_layer_observations": int(prefill_layer_obs),
        "stock_fallback_prefill_layer_observations": int(prefill_fallback_layer_obs),
        "decode_step_observations": int(decode_step_obs),
        "stock_fallback_decode_steps": int(decode_fallback_steps),
        "prefill_step_observations": int(prefill_step_obs),
        "stock_fallback_prefill_steps": int(prefill_fallback_steps),
        "stock_fallback_share_run": stock_fallback_share_run,
        "stock_fallback_share_decode_layers": stock_fallback_share_decode_layers,
        "stock_fallback_share_prefill_layers": stock_fallback_share_prefill_layers,
        "stock_fallback_share_decode_steps": stock_fallback_share_decode_steps,
        "stock_fallback_share_prefill_steps": stock_fallback_share_prefill_steps,
        "fallback_share_known": bool(total_obs > 0),
        "active_query_ratio": (float(active_query_obs / total_obs) if total_obs else 0.0),
        "adaptive_graph_reuse_ratio": (float(adaptive_reuse_obs / total_obs) if total_obs else 0.0),
        "cache_hit_ratio": (float(cache_hit_obs / total_obs) if total_obs else 0.0),
        "graph_build_ms_mean": (
            float(sum(graph_build_ms_values) / len(graph_build_ms_values))
            if graph_build_ms_values
            else None
        ),
        "graph_build_ms_p95": _percentile(graph_build_ms_values, 95.0),
        "butterfly_ms_mean": (
            float(sum(butterfly_ms_values) / len(butterfly_ms_values))
            if butterfly_ms_values
            else None
        ),
        "butterfly_ms_p95": _percentile(butterfly_ms_values, 95.0),
        "butterfly_ms_decode_mean": (
            float(sum(decode_butterfly_ms_values) / len(decode_butterfly_ms_values))
            if decode_butterfly_ms_values
            else None
        ),
        "butterfly_ms_prefill_mean": (
            float(sum(prefill_butterfly_ms_values) / len(prefill_butterfly_ms_values))
            if prefill_butterfly_ms_values
            else None
        ),
        "attention_ms_mean": (
            float(sum(attention_ms_values) / len(attention_ms_values))
            if attention_ms_values
            else None
        ),
        "attention_ms_p95": _percentile(attention_ms_values, 95.0),
        "total_ms_mean": (
            float(sum(total_ms_values) / len(total_ms_values))
            if total_ms_values
            else None
        ),
        "total_ms_p95": _percentile(total_ms_values, 95.0),
    }


def _expected_primary_path_for_mode(mode: str) -> str:
    mode_s = str(mode).strip().lower()
    if mode_s in {"wayfinder", "butterfly"}:
        return "butterfly_attention"
    if mode_s == "sparse":
        return "sparse"
    return "stock_attention"


def _parse_layer_indices(spec: str, *, total_layers: int) -> List[int]:
    out: Set[int] = set()
    raw = str(spec or "").strip()
    if not raw:
        return []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        idx = int(s)
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {total_layers - 1}]")
        out.add(idx)
    return sorted(out)


def _resolve_swap_layers(
    *,
    total_layers: int,
    swap_first_n_layers: int,
    swap_last_n_layers: int,
    swap_layer_indices: str,
) -> Optional[List[int]]:
    explicit = _parse_layer_indices(swap_layer_indices, total_layers=total_layers)
    if explicit:
        return explicit
    selected: Set[int] = set()
    if int(swap_first_n_layers) > 0:
        n = min(int(swap_first_n_layers), total_layers)
        selected.update(range(0, n))
    if int(swap_last_n_layers) > 0:
        n = min(int(swap_last_n_layers), total_layers)
        selected.update(range(total_layers - n, total_layers))
    return sorted(selected) if selected else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if q <= 0:
        return xs[0]
    if q >= 100:
        return xs[-1]
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return xs[idx]


def _prepare_cache(model: Any, *, kv_step: int, target_seq_len: int) -> List[Any]:
    prealloc_size: Optional[int] = int(target_seq_len) if int(target_seq_len) > 0 else None
    if kv_step > 0 and target_seq_len > 0:
        assert prealloc_size is not None
        if prealloc_size % kv_step != 0:
            prealloc_size = ((prealloc_size // kv_step) + 1) * kv_step
    if prealloc_size is not None:
        _log(f"    kv_step={kv_step}, target prealloc={prealloc_size} tokens")
    cache = list(make_prompt_cache(model, max_kv_size=prealloc_size))
    return cache


def _run_chunked_prefill(
    model: Any,
    *,
    prompt_tokens: mx.array,
    chunk_size: int,
    cache: Sequence[Any],
    heartbeat_sec: float = 0.0,
    heartbeat_prefix: str = "",
    hsa_trace_samples: Optional[List[Dict[str, Any]]] = None,
    hsa_trace_layer_indices: Optional[Sequence[int]] = None,
    hsa_trace_max_layers: int = 0,
    observability_default_path: Optional[str] = None,
) -> Dict[str, Any]:
    batch = int(prompt_tokens.shape[0])
    seq_len = int(prompt_tokens.shape[1])

    t0 = time.perf_counter()
    last_beat = t0
    chunk_idx = 0
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        chunk_idx += 1
        logits = model(prompt_tokens[:, start:end], cache=cache)
        mx.eval(logits)
        if hsa_trace_samples is not None:
            layer_rows = _collect_hsa_trace_snapshot(
                model,
                layer_indices=hsa_trace_layer_indices,
                max_layers=hsa_trace_max_layers,
            )
            if not layer_rows and observability_default_path:
                layer_rows = [
                    {
                        "layer_idx": -1,
                        "path": str(observability_default_path),
                        "graph_build_ms": None,
                        "attention_ms": None,
                        "total_ms": None,
                        "notes": {"dense_fallback_reason": "none"},
                    }
                ]
            if layer_rows:
                hsa_trace_samples.append(
                    {
                        "phase": "prefill",
                        "chunk": int(chunk_idx),
                        "chunk_start": int(start),
                        "chunk_end": int(end),
                        "layers": layer_rows,
                    }
                )
        now = time.perf_counter()
        if float(heartbeat_sec) > 0 and (now - last_beat) >= float(heartbeat_sec):
            prefix = str(heartbeat_prefix).strip()
            if prefix:
                prefix = prefix + " "
            _log(
                f"heartbeat {prefix}prefill: chunk={chunk_idx} tokens={end}/{seq_len} elapsed={now - t0:.1f}s"
            )
            last_beat = now
        _clear_workspace()

    prefill_sec = time.perf_counter() - t0
    total_tokens = batch * seq_len
    return {
        "prefill_sec": float(prefill_sec),
        "prefill_tok_s": float(total_tokens / max(prefill_sec, 1e-12)),
        "peak_memory_bytes": int(_peak_memory()),
    }


def _run_decode(
    model: Any,
    *,
    batch: int,
    decode_len: int,
    cache: Sequence[Any],
    heartbeat_sec: float = 0.0,
    heartbeat_prefix: str = "",
    trace_enabled: bool = False,
    trace_topk: int = 5,
    trace_max_steps: int = 0,
    hsa_trace_samples: Optional[List[Dict[str, Any]]] = None,
    hsa_trace_layer_indices: Optional[Sequence[int]] = None,
    hsa_trace_max_layers: int = 0,
    hsa_trace_max_steps: int = 0,
    observability_default_path: Optional[str] = None,
) -> Dict[str, Any]:
    if decode_len <= 0:
        return {
            "decode_sec": 0.0,
            "decode_tok_s": None,
            "ttft_sec": None,
            "itl_p50_sec": None,
            "itl_p95_sec": None,
            "token_ids": [],
            "trace": [],
        }

    per_token_sec: List[float] = []
    token_ids: List[int] = []
    trace_rows: List[Dict[str, Any]] = []
    next_token = mx.zeros((batch, 1), dtype=mx.int32)
    t0 = time.perf_counter()
    last_beat = t0
    topk = max(1, int(trace_topk))
    max_steps = int(trace_max_steps)
    hsa_decode_steps = 0
    for step_idx in range(int(decode_len)):
        t_step = time.perf_counter()
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        should_hsa_trace = hsa_trace_samples is not None and (
            int(hsa_trace_max_steps) <= 0 or hsa_decode_steps < int(hsa_trace_max_steps)
        )
        if should_hsa_trace:
            layer_rows = _collect_hsa_trace_snapshot(
                model,
                layer_indices=hsa_trace_layer_indices,
                max_layers=hsa_trace_max_layers,
            )
            if not layer_rows and observability_default_path:
                layer_rows = [
                    {
                        "layer_idx": -1,
                        "path": str(observability_default_path),
                        "graph_build_ms": None,
                        "attention_ms": None,
                        "total_ms": None,
                        "notes": {"dense_fallback_reason": "none"},
                    }
                ]
            if layer_rows:
                hsa_decode_steps += 1
                hsa_trace_samples.append(
                    {
                        "phase": "decode",
                        "step": int(step_idx + 1),
                        "layers": layer_rows,
                    }
                )
        should_trace = bool(trace_enabled) and (max_steps <= 0 or len(trace_rows) < max_steps)
        if should_trace:
            logits_last = logits[0, -1, :].astype(mx.float32)
            mx.eval(logits_last)
            logits_np = np.array(logits_last)
            k = min(topk, int(logits_np.shape[0]))
            top_idx = np.argpartition(logits_np, -k)[-k:]
            top_idx = top_idx[np.argsort(logits_np[top_idx])[::-1]]
            top_candidates = [
                {"token_id": int(i), "logit": float(logits_np[int(i)])} for i in top_idx.tolist()
            ]
        else:
            top_candidates = []
        next_token = mx.argmax(logits[:, -1:, :], axis=-1).astype(mx.int32)
        mx.eval(next_token)
        if should_trace:
            chosen_token_id = int(next_token[0, 0].item())
            trace_rows.append(
                {
                    "step": int(len(trace_rows) + 1),
                    "chosen_token_id": chosen_token_id,
                    "chosen_token_logit": float(logits_np[chosen_token_id]),
                    "topk": top_candidates,
                }
            )
        per_token_sec.append(time.perf_counter() - t_step)
        token_ids.append(int(next_token[0, 0].item()))
        now = time.perf_counter()
        if float(heartbeat_sec) > 0 and (now - last_beat) >= float(heartbeat_sec):
            prefix = str(heartbeat_prefix).strip()
            if prefix:
                prefix = prefix + " "
            _log(
                f"heartbeat {prefix}decode: step={step_idx + 1}/{decode_len} elapsed={now - t0:.1f}s"
            )
            last_beat = now
    _clear_workspace()
    decode_sec = time.perf_counter() - t0
    itl_values = per_token_sec[1:] if len(per_token_sec) > 1 else per_token_sec
    return {
        "decode_sec": float(decode_sec),
        "decode_tok_s": float((batch * decode_len) / max(decode_sec, 1e-12)),
        "ttft_sec": float(per_token_sec[0]) if per_token_sec else None,
        "itl_p50_sec": _percentile(itl_values, 50.0),
        "itl_p95_sec": _percentile(itl_values, 95.0),
        "token_ids": token_ids,
        "trace": trace_rows,
    }


def _encode_text(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            out = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(out, list):
                return [int(x) for x in out]
        except TypeError:
            out = tokenizer.encode(text)
            if isinstance(out, list):
                return [int(x) for x in out]
    raise RuntimeError("Tokenizer encode unavailable")


def _decode_tokens(tokenizer: Any, token_ids: List[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(token_ids))
        except Exception:
            return ""
    return ""


def _build_prompt_tokens(
    tokenizer: Any,
    target_len: int,
    *,
    seed_text: str,
    heartbeat_sec: float = 0.0,
    heartbeat_prefix: str = "",
) -> List[int]:
    # Build deterministic chat-like prompt text and trim to target token length.
    text = seed_text
    t0 = time.perf_counter()
    last_beat = t0
    loops = 0
    while True:
        ids = _encode_text(tokenizer, text)
        loops += 1
        now = time.perf_counter()
        if float(heartbeat_sec) > 0 and (now - last_beat) >= float(heartbeat_sec):
            prefix = str(heartbeat_prefix).strip()
            if prefix:
                prefix = prefix + " "
            _log(
                f"heartbeat {prefix}prompt-build: loops={loops} tokens={len(ids)}/{target_len} elapsed={now - t0:.1f}s"
            )
            last_beat = now
        if len(ids) >= target_len:
            return ids[:target_len]
        text += "\nUser: Summarize the policy and list three key constraints with exact numbers.\nAssistant:"


def _normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().lower())


def _run_single_turn(
    model: Any,
    tokenizer: Any,
    *,
    mode: str,
    seq_lens: List[int],
    decode_len: int,
    repeats: int,
    chunk_size: int,
    kv_step: int,
    cooldown_sec: float,
    stage_timeout_sec: float = 0.0,
    heartbeat_sec: float = 0.0,
    hsa_trace: bool = False,
    hsa_trace_layer_indices: Optional[Sequence[int]] = None,
    hsa_trace_max_layers: int = 0,
    hsa_trace_max_steps: int = 0,
    on_row: Any = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expected_primary_path = _expected_primary_path_for_mode(mode)
    for seq_len in seq_lens:
        # Unlogged warmup for measured set.
        with _stage_timeout(stage_timeout_sec, stage="warmup_prompt", seq_len=seq_len):
            warm_tokens = _build_prompt_tokens(
                tokenizer,
                max(256, min(2048, seq_len)),
                seed_text="User: Warmup chat prompt.\nAssistant:",
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=f"T={seq_len} warmup",
            )
        with _stage_timeout(stage_timeout_sec, stage="warmup_prefill_decode", seq_len=seq_len):
            cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=seq_len + decode_len)
            _reset_peak_memory()
            _run_chunked_prefill(
                model,
                prompt_tokens=mx.array([warm_tokens], dtype=mx.int32),
                chunk_size=chunk_size,
                cache=cache,
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=f"T={seq_len} warmup",
            )
            _run_decode(
                model,
                batch=1,
                decode_len=min(16, decode_len),
                cache=cache,
                heartbeat_sec=heartbeat_sec,
                heartbeat_prefix=f"T={seq_len} warmup",
            )
            _clear_workspace()

        for r in range(1, repeats + 1):
            prompt_start = time.perf_counter()
            with _stage_timeout(
                stage_timeout_sec,
                stage="prompt_build",
                seq_len=seq_len,
                repeat=r,
            ):
                prompt_tokens = _build_prompt_tokens(
                    tokenizer,
                    seq_len,
                    seed_text=(
                        "System: You are a helpful assistant in a long-running session.\n"
                        "User: Please read all previous notes and answer precisely with concise output.\nAssistant:"
                    ),
                    heartbeat_sec=heartbeat_sec,
                    heartbeat_prefix=f"T={seq_len} r={r}",
                )
            prompt_build_sec = float(time.perf_counter() - prompt_start)

            cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=seq_len + decode_len)
            _reset_peak_memory()
            hsa_trace_samples: List[Dict[str, Any]] = []
            with _stage_timeout(
                stage_timeout_sec,
                stage="prefill",
                seq_len=seq_len,
                repeat=r,
            ):
                pre = _run_chunked_prefill(
                    model,
                    prompt_tokens=mx.array([prompt_tokens], dtype=mx.int32),
                    chunk_size=chunk_size,
                    cache=cache,
                    heartbeat_sec=heartbeat_sec,
                    heartbeat_prefix=f"T={seq_len} r={r}",
                    hsa_trace_samples=hsa_trace_samples,
                    hsa_trace_layer_indices=hsa_trace_layer_indices,
                    hsa_trace_max_layers=hsa_trace_max_layers,
                    observability_default_path=expected_primary_path,
                )
            with _stage_timeout(
                stage_timeout_sec,
                stage="decode",
                seq_len=seq_len,
                repeat=r,
            ):
                dec = _run_decode(
                    model,
                    batch=1,
                    decode_len=decode_len,
                    cache=cache,
                    heartbeat_sec=heartbeat_sec,
                    heartbeat_prefix=f"T={seq_len} r={r}",
                    hsa_trace_samples=hsa_trace_samples,
                    hsa_trace_layer_indices=hsa_trace_layer_indices,
                    hsa_trace_max_layers=hsa_trace_max_layers,
                    hsa_trace_max_steps=hsa_trace_max_steps,
                    observability_default_path=expected_primary_path,
                )
            e2e = float(pre["prefill_sec"] + dec["decode_sec"])
            hsa_summary = _summarize_hsa_trace(hsa_trace_samples)
            row: Dict[str, Any] = {
                "seq_len": int(seq_len),
                "decode_len": int(decode_len),
                "repeat": int(r),
                "prefill_sec": pre["prefill_sec"],
                "ttft_sec": dec["ttft_sec"],
                "itl_p50_sec": dec["itl_p50_sec"],
                "itl_p95_sec": dec["itl_p95_sec"],
                "decode_sec": dec["decode_sec"],
                "e2e_sec": e2e,
                "decode_tok_s": dec["decode_tok_s"],
                "peak_memory_bytes": int(_peak_memory()),
                "sample_output_text": _decode_tokens(tokenizer, dec["token_ids"][:32]),
                "expected_primary_path": expected_primary_path,
                "hsa_trace_summary": hsa_summary,
                "path_counts": dict(hsa_summary.get("path_counts", {})),
                "stock_fallback_reason_counts": dict(
                    hsa_summary.get("stock_fallback_reason_counts", {})
                ),
                "stock_fallback_share_run": hsa_summary.get("stock_fallback_share_run"),
                "stock_fallback_share_decode_steps": hsa_summary.get(
                    "stock_fallback_share_decode_steps"
                ),
                "stock_fallback_share_prefill_steps": hsa_summary.get(
                    "stock_fallback_share_prefill_steps"
                ),
                "graph_build_ms_mean": hsa_summary.get("graph_build_ms_mean"),
                "butterfly_ms_mean": hsa_summary.get("butterfly_ms_mean"),
                "butterfly_ms_decode_mean": hsa_summary.get("butterfly_ms_decode_mean"),
                "butterfly_ms_prefill_mean": hsa_summary.get("butterfly_ms_prefill_mean"),
                "attention_ms_mean": hsa_summary.get("attention_ms_mean"),
                "observability_fallback_share_known": bool(hsa_summary.get("fallback_share_known")),
                "stage_timing_sec": {
                    "prompt_build_sec": prompt_build_sec,
                    "prompt_tokenize_sec": prompt_build_sec,
                    "prefill_sec": float(pre["prefill_sec"]),
                    "decode_sec": float(dec["decode_sec"]),
                },
            }
            if hsa_trace:
                row["hsa_trace_samples_head"] = hsa_trace_samples[:16]
            rows.append(row)
            if on_row is not None:
                on_row(rows[-1])
            _log(
                f"single_turn T={seq_len} r={r}/{repeats}: e2e={e2e:.3f}s ttft={dec['ttft_sec']:.4f}s itl_p95={dec['itl_p95_sec']:.4f}s peak={_peak_memory()}"
            )
            if r < repeats:
                time.sleep(max(0.0, cooldown_sec))
    return rows


def _run_multi_turn(
    model: Any,
    tokenizer: Any,
    *,
    turns: int,
    target_total_context: int,
    decode_len: int,
    chunk_size: int,
    kv_step: int,
    position_limit: Optional[int] = None,
    on_turn: Any = None,
) -> Dict[str, Any]:
    per_turn: List[Dict[str, Any]] = []
    history = "System: You are a concise assistant.\n"
    total_e2e = 0.0
    prompt_budget_limit = None
    if position_limit is not None:
        prompt_budget_limit = int(position_limit) - int(decode_len)
        if prompt_budget_limit <= 0:
            raise ValueError(
                f"decode_len={int(decode_len)} leaves no prompt budget within max_position_embeddings={int(position_limit)}"
            )
    for t in range(1, turns + 1):
        requested_target_len = max(1024, int(target_total_context * t / turns))
        target_len = requested_target_len
        if prompt_budget_limit is not None:
            target_len = min(target_len, prompt_budget_limit)
            if target_len != requested_target_len:
                _log(
                    f"multi_turn t={t}/{turns}: clamped prompt target from {requested_target_len} to {target_len} "
                    f"to reserve decode headroom within max_position_embeddings={int(position_limit)}"
                )
        turn_prompt = (
            history
            + f"User turn {t}: extract ID, date, and checksum from prior context and explain in one line.\nAssistant:"
        )
        prompt_tokens = _build_prompt_tokens(tokenizer, target_len, seed_text=turn_prompt)
        cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=target_len + decode_len)
        _reset_peak_memory()
        pre = _run_chunked_prefill(
            model,
            prompt_tokens=mx.array([prompt_tokens], dtype=mx.int32),
            chunk_size=chunk_size,
            cache=cache,
        )
        dec = _run_decode(model, batch=1, decode_len=decode_len, cache=cache)
        e2e = float(pre["prefill_sec"] + dec["decode_sec"])
        total_e2e += e2e
        out_text = _decode_tokens(tokenizer, dec["token_ids"])
        history += f"User turn {t}: context growth checkpoint.\nAssistant: {out_text}\n"
        turn_row = {
            "turn": int(t),
            "seq_len": int(target_len),
            "decode_len": int(decode_len),
            "ttft_sec": dec["ttft_sec"],
            "itl_p50_sec": dec["itl_p50_sec"],
            "itl_p95_sec": dec["itl_p95_sec"],
            "e2e_sec": e2e,
            "decode_tok_s": dec["decode_tok_s"],
            "peak_memory_bytes": int(_peak_memory()),
        }
        per_turn.append(turn_row)
        if on_turn is not None:
            on_turn(turn_row)
        _log(
            f"multi_turn t={t}/{turns}: T={target_len} e2e={e2e:.3f}s ttft={dec['ttft_sec']:.4f}s itl_p95={dec['itl_p95_sec']:.4f}s"
        )
    return {
        "turns": turns,
        "target_total_context": target_total_context,
        "decode_len_per_turn": decode_len,
        "max_position_embeddings": int(position_limit) if position_limit is not None else None,
        "prompt_budget_limit": int(prompt_budget_limit) if prompt_budget_limit is not None else None,
        "session_e2e_sec": float(total_e2e),
        "per_turn": per_turn,
    }


def _run_quality(
    model: Any,
    tokenizer: Any,
    *,
    dataset_path: Path,
    decode_len: int,
    chunk_size: int,
    kv_step: int,
    task_id_filter: Optional[Set[str]] = None,
    trace_task_id: Optional[str] = None,
    trace_topk: int = 5,
    trace_max_steps: int = 0,
    on_task: Any = None,
) -> Dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    tasks_raw = payload.get("tasks", [])
    tasks = []
    if task_id_filter:
        for task in tasks_raw:
            if str(task.get("id")) in task_id_filter:
                tasks.append(task)
    else:
        tasks = list(tasks_raw)
    rows: List[Dict[str, Any]] = []
    correct = 0
    for task in tasks:
        prompt = str(task["prompt"])
        expected = str(task["expected"])
        task_id = str(task.get("id"))
        prompt_ids = _encode_text(tokenizer, prompt)
        cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=len(prompt_ids) + decode_len)
        _reset_peak_memory()
        _run_chunked_prefill(
            model,
            prompt_tokens=mx.array([prompt_ids], dtype=mx.int32),
            chunk_size=chunk_size,
            cache=cache,
        )
        dec = _run_decode(
            model,
            batch=1,
            decode_len=decode_len,
            cache=cache,
            trace_enabled=bool(trace_task_id) and task_id == str(trace_task_id),
            trace_topk=int(trace_topk),
            trace_max_steps=int(trace_max_steps),
        )
        out_text = _decode_tokens(tokenizer, dec["token_ids"])
        ok = _normalize_text(expected) in _normalize_text(out_text)
        if ok:
            correct += 1
        row = {
            "id": task_id,
            "expected": expected,
            "output": out_text,
            "correct": bool(ok),
            "ttft_sec": dec["ttft_sec"],
            "itl_p95_sec": dec["itl_p95_sec"],
            "peak_memory_bytes": int(_peak_memory()),
        }
        if dec.get("trace"):
            row["decode_trace"] = dec["trace"]
        rows.append(row)
        if on_task is not None:
            on_task(rows[-1], int(correct), int(len(rows)))
    accuracy = float(correct / max(1, len(tasks)))
    return {
        "dataset_path": str(dataset_path),
        "num_tasks": int(len(tasks)),
        "correct": int(correct),
        "accuracy": accuracy,
        "rows": rows,
        "task_id_filter": sorted(task_id_filter) if task_id_filter else None,
        "trace_task_id": (str(trace_task_id) if trace_task_id else None),
        "trace_topk": int(max(1, trace_topk)),
        "trace_max_steps": int(max(0, trace_max_steps)),
    }


def _resolve_primary_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    mode_arg = None if args.mode is None else str(args.mode).strip().lower()
    path_arg = None if args.path is None else str(args.path).strip().lower()

    if path_arg is not None and path_arg not in {"permute", "sparse"}:
        parser.error(f"--path must be one of ['permute', 'sparse']; got {path_arg!r}")
    if bool(args.no_swap) and path_arg == "sparse":
        parser.error("--no-swap conflicts with --path sparse")

    legacy_mode = None
    if bool(args.no_swap):
        legacy_mode = "stock"
    elif path_arg == "sparse":
        legacy_mode = "sparse"
    elif path_arg == "permute":
        legacy_mode = "butterfly"

    if mode_arg is None:
        return legacy_mode or "butterfly"
    # Public surface: "butterfly" vs "stock". Legacy aliases remain accepted.
    if mode_arg == "wayfinder":
        mode_arg = "butterfly"
    if mode_arg in {"native", "hybrid", "dense", "stock"}:
        mode_arg = "stock"
    if mode_arg not in {"stock", "butterfly", "sparse"}:
        parser.error(
            f"--mode must be one of ['stock', 'butterfly', 'sparse'] plus legacy aliases ['native', 'hybrid', 'dense', 'wayfinder']; got {mode_arg!r}"
        )
    if legacy_mode is not None and legacy_mode != mode_arg:
        parser.error(
            f"Conflicting mode selectors: --mode {mode_arg} vs legacy selector {legacy_mode}. "
            "Use only --mode in primary flows."
        )
    return mode_arg


def _build_single_turn_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for row in rows:
        summary_rows.append(
            {
                "prompt_tokens": int(row.get("seq_len", 0)),
                "decode_tokens": int(row.get("decode_len", 0)),
                "repeat": int(row.get("repeat", 0)),
                "prefill_sec": row.get("prefill_sec"),
                "ttft_sec": row.get("ttft_sec"),
                "decode_tok_s": row.get("decode_tok_s"),
                "e2e_sec": row.get("e2e_sec"),
                "path_counts": dict(row.get("path_counts", {})),
                "stock_fallback_share_run": row.get("stock_fallback_share_run"),
                "stock_fallback_share_prefill_steps": row.get("stock_fallback_share_prefill_steps"),
                "stock_fallback_share_decode_steps": row.get("stock_fallback_share_decode_steps"),
            }
        )
    return summary_rows


def _render_summary_markdown(payload: Dict[str, Any]) -> str:
    butterfly_meta = payload.get("butterfly") or {}
    lines = [
        "# Qwen MLX Summary",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- butterfly decode backend: `{payload.get('butterfly_decode_backend')}`",
        f"- butterfly prefill scope: `{butterfly_meta.get('prefill_scope')}`",
        f"- butterfly prefill target layers: `{butterfly_meta.get('target_prefill_layer_indices')}`",
        f"- butterfly prefill active layers: `{butterfly_meta.get('active_prefill_layer_indices')}`",
        "",
        "| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _build_single_turn_summary_rows(payload.get("single_turn") or []):
        stock_share = row.get("stock_fallback_share_run")
        stock_share_str = "n/a" if stock_share is None else f"{float(stock_share):.2f}"
        decode_tok_s = row.get("decode_tok_s")
        decode_tok_s_str = "n/a" if decode_tok_s is None else f"{float(decode_tok_s):.2f}"
        lines.append(
            "| "
            f"{int(row.get('prompt_tokens', 0))} | "
            f"{float(row.get('prefill_sec', 0.0)):.4f} | "
            f"{float(row.get('ttft_sec', 0.0)):.4f} | "
            f"{decode_tok_s_str} | "
            f"{float(row.get('e2e_sec', 0.0)):.4f} | "
            f"{stock_share_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Qwen MLX benchmark with stock vs Butterfly single-turn, multi-turn, and quality flows."
    )
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-1.7B-4bit")
    p.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Hugging Face cache home (defaults to /Volumes/VIXinSSD/hf_cache if present).",
    )
    p.add_argument(
        "--hf-hub-cache",
        type=str,
        default=None,
        help="Hugging Face hub cache directory (defaults to <hf-home>/hub).",
    )
    p.add_argument(
        "--hf-offline",
        action="store_true",
        default=False,
        help="Enable HF offline mode and require all model files to already exist in cache.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        help=(
            "Primary attention mode selector. "
            "'stock' = default Qwen 3.5 attention path. "
            "'butterfly' = Butterfly attention for the 8 swapped prefill layers on Qwen 3.5. "
            "Legacy aliases remain accepted. "
            "'sparse' = legacy sparse-gather path."
        ),
    )
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--decode-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--multi-decode-len", type=int, default=128)
    p.add_argument("--multi-target-context", type=int, default=65536)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--kv-step", type=int, default=4096)
    p.add_argument("--cooldown-sec", type=float, default=60.0)
    p.add_argument(
        "--stage-timeout-sec",
        type=float,
        default=0.0,
        help="Per-stage timeout for warmup/prompt-build/prefill/decode (<=0 disables).",
    )
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=30.0,
        help="Emit heartbeat logs every N seconds during prompt-build/prefill/decode (<=0 disables).",
    )
    p.add_argument("--path", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=0)
    p.add_argument("--num-cycles", type=int, default=1)
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
    p.add_argument("--head-chunk-size", type=int, default=2)
    p.add_argument("--query-chunk-size", type=int, default=384)
    p.add_argument(
        "--butterfly-decode-backend",
        type=str,
        default="stock",
        help="Decode backend policy. 'stock' (legacy alias: 'dense') keeps the default stock decode path on swapped layers. 'experimental' (legacy alias: 'active_permute') enables Butterfly decode experiments.",
    )
    p.add_argument("--debug-wayfinder-decode-backend", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--wayfinder-decode-backend", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument(
        "--debug-disable-fused-dispatch",
        action="store_true",
        default=False,
        help="Debug-only: disable fused Butterfly dispatch.",
    )
    p.add_argument(
        "--disable-fused-dispatch", action="store_true", default=False, help=argparse.SUPPRESS
    )
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-swap", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument(
        "--debug-swap-first-n-layers",
        type=int,
        default=0,
        help="Debug-only: swap only first N layers.",
    )
    p.add_argument(
        "--debug-swap-last-n-layers",
        type=int,
        default=0,
        help="Debug-only: swap only last N layers.",
    )
    p.add_argument(
        "--debug-swap-layer-indices",
        type=str,
        default="",
        help="Debug-only: swap explicit comma-separated layer indices.",
    )
    p.add_argument("--swap-first-n-layers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--swap-last-n-layers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--swap-layer-indices", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument(
        "--quality-dataset",
        type=Path,
        default=Path("benchmarks/mlx/qwen3_1_7b_wayfinder/quality_eval_qwen3_consumer_v1.json"),
    )
    p.add_argument("--quality-task-id-filter", type=str, default="")
    p.add_argument("--trace-quality-task-id", type=str, default="")
    p.add_argument("--trace-topk", type=int, default=5)
    p.add_argument("--trace-max-steps", type=int, default=0)
    p.add_argument(
        "--hsa-trace",
        action="store_true",
        help="Record per-step Wayfinder attention path snapshots.",
    )
    p.add_argument(
        "--hsa-trace-max-layers",
        type=int,
        default=8,
        help="Max swapped layers to include in each HSA snapshot (<=0 means all).",
    )
    p.add_argument(
        "--hsa-trace-max-steps",
        type=int,
        default=0,
        help="Max decode steps to capture for HSA tracing (<=0 means all decode steps).",
    )
    p.add_argument("--skip-single-turn", action="store_true")
    p.add_argument("--skip-multi-turn", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    mode = _resolve_primary_mode(args, p)
    if args.mode is None:
        _log(f"Resolved mode={mode} from legacy/default flags (preferred: --mode {mode}).")
    debug_disable_fused_dispatch = bool(args.debug_disable_fused_dispatch) or bool(
        args.disable_fused_dispatch
    )
    debug_swap_first_n_layers = (
        int(args.debug_swap_first_n_layers)
        if int(args.debug_swap_first_n_layers) > 0
        else int(args.swap_first_n_layers)
    )
    debug_swap_last_n_layers = (
        int(args.debug_swap_last_n_layers)
        if int(args.debug_swap_last_n_layers) > 0
        else int(args.swap_last_n_layers)
    )
    debug_swap_layer_indices = (
        str(args.debug_swap_layer_indices).strip() or str(args.swap_layer_indices).strip()
    )

    try:
        mx.random.seed(int(args.seed))
    except Exception:
        pass

    decode_backend = (
        str(args.butterfly_decode_backend).strip()
        if str(args.butterfly_decode_backend).strip()
        else (
            str(args.wayfinder_decode_backend).strip()
            if str(args.wayfinder_decode_backend).strip()
            else str(args.debug_wayfinder_decode_backend).strip()
        )
    )
    if decode_backend == "stock":
        decode_backend = "dense"
    if decode_backend == "experimental":
        decode_backend = "active_permute"
    if decode_backend not in {"active_permute", "dense"}:
        p.error(
            "--butterfly-decode-backend must be one of ['stock', 'experimental'] plus legacy aliases ['dense', 'active_permute']"
        )
    if mode == "butterfly" and int(args.chunk_size) > int(max(1, args.query_chunk_size)):
        p.error(
            "--chunk-size exceeds --query-chunk-size in butterfly mode; "
            "prefill chunks after the first will fall back to stock attention "
            "(active_large_q), so this is not a valid butterfly benchmark."
        )
    cache_cfg = _configure_hf_cache(
        hf_home=(str(args.hf_home).strip() or None),
        hf_hub_cache=(str(args.hf_hub_cache).strip() or None),
        hf_offline=bool(args.hf_offline),
    )
    _log(
        "HF cache config: "
        f"HF_HOME={cache_cfg['hf_home']} HF_HUB_CACHE={cache_cfg['hf_hub_cache']} "
        f"HF_HUB_OFFLINE={cache_cfg['hf_hub_offline']}"
    )

    resolved_model_path = resolve_qwen_mlx_model_path(args.model_path, prefer_text=True)
    if str(resolved_model_path) != str(Path(args.model_path).expanduser().resolve()):
        _log(f"Resolved text-only checkpoint: {resolved_model_path}")

    _log(f"Loading model {resolved_model_path}")
    model, tokenizer, model_config = load_qwen_mlx_model(
        resolved_model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded")
    position_limit = _resolve_position_limit(model_config, tokenizer)
    if position_limit is not None:
        _log(f"Model max_position_embeddings={position_limit}")
        if not args.skip_single_turn:
            invalid_seq_lens = [
                int(seq_len)
                for seq_len in args.seq_lens
                if int(seq_len) + int(args.decode_len) > int(position_limit)
            ]
            if invalid_seq_lens:
                p.error(
                    "--seq-lens plus --decode-len exceed model max_position_embeddings="
                    f"{int(position_limit)} for seq_lens={invalid_seq_lens}."
                )
        if not args.skip_multi_turn and int(args.multi_decode_len) >= int(position_limit):
            p.error(
                "--multi-decode-len must be smaller than model max_position_embeddings="
                f"{int(position_limit)}."
            )
    else:
        _log("WARNING: unable to resolve max_position_embeddings; decode headroom will not be validated.")

    try:
        full_attention_layer_indices = validate_qwen35_full_attention_layers(
            model,
            allow_mismatch=False,
        )
        butterfly_prefill_scope = "qwen35_full_attention_layers"
    except ValueError as exc:
        full_attention_layer_indices = get_qwen_full_attention_layer_indices(model)
        butterfly_prefill_scope = "discovered_full_attention_layers"
        _log(f"WARNING: {exc}. Continuing with discovered full-attention layers for this benchmark.")

    baseline_mode = mode == "stock"
    wf_cfg: Optional[QwenWayfinderConfig] = None
    if not baseline_mode:
        qwen_path = "permute" if mode == "butterfly" else "sparse"
        wf_cfg = QwenWayfinderConfig(
            path=qwen_path,  # type: ignore[arg-type]
            strategy="random",
            window=int(args.window),
            landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
            num_cycles=int(max(0, args.num_cycles)),
            edge_disjoint=bool(args.edge_disjoint),
            enforce_hamiltonian=bool(args.enforce_hamiltonian),
            seed=int(args.seed),
            edge_bias=True,
            window_drop=float(args.window_drop),
            permute_head_chunk_size=int(max(1, args.head_chunk_size)),
            query_chunk_size=int(max(1, args.query_chunk_size)),
            compute_edge_utilization_proxy=False,
            compute_graph_metrics=False,
            retro_backfill_enabled=bool(args.retro_backfill),
            retro_backfill_alpha=float(args.retro_alpha),
            retro_backfill_training_only=bool(args.retro_training_only),
            retro_backfill_causal_only=bool(args.retro_causal_only),
            use_fused_dispatch=not bool(debug_disable_fused_dispatch),
            wayfinder_decode_backend=decode_backend,  # type: ignore[arg-type]
        )

    replaced_layers = 0
    replaced_indices: List[int] = []
    requested_layer_indices: Optional[List[int]] = None
    active_prefill_layer_indices: List[int] = []
    active_prefill_scope = butterfly_prefill_scope
    if baseline_mode:
        _log(
            "mode=stock: default Qwen 3.5 attention path "
            f"(Butterfly target scope={butterfly_prefill_scope}, layers={full_attention_layer_indices})"
        )
    else:
        assert wf_cfg is not None
        layers = list(_iter_model_layers(model))
        requested_layer_indices = _resolve_swap_layers(
            total_layers=len(layers),
            swap_first_n_layers=int(debug_swap_first_n_layers),
            swap_last_n_layers=int(debug_swap_last_n_layers),
            swap_layer_indices=debug_swap_layer_indices,
        )
        active_prefill_layer_indices = (
            requested_layer_indices if requested_layer_indices is not None else full_attention_layer_indices
        )
        if requested_layer_indices is not None:
            active_prefill_scope = "explicit_debug_layer_selection"
        replaced = swap_qwen_attention_with_wayfinder(
            model,
            cfg=wf_cfg,
            layer_indices=active_prefill_layer_indices,
        )
        replaced_indices = [int(x) for x in replaced]
        replaced_layers = len(replaced_indices)
        for idx in replaced_indices:
            layer_attn = layers[idx].self_attn
            if isinstance(layer_attn, QwenWayfinderAttention):
                layer_attn.compute_edge_utilization_proxy = False
                layer_attn.compute_graph_metrics = False
        _log(
            f"mode={mode}: Butterfly swap active on {replaced_layers} layers "
            f"(scope={active_prefill_scope}, layers={replaced_indices}, "
            f"decode_backend={'stock' if decode_backend == 'dense' else 'experimental'})"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    butterfly_decode_backend_public = (
        "stock" if baseline_mode or decode_backend == "dense" else "experimental"
    )
    payload: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "model_config": {
            "max_position_embeddings": int(position_limit) if position_limit is not None else None,
        },
        "hf_cache": cache_cfg,
        "mode": mode,
        "butterfly_decode_backend": butterfly_decode_backend_public,
        "retro_backfill_enabled": bool(args.retro_backfill),
        "butterfly": {
            "enabled": bool(not baseline_mode),
            "decode_backend": butterfly_decode_backend_public,
            "prefill_scope": active_prefill_scope,
            "target_prefill_layer_indices": [int(x) for x in full_attention_layer_indices],
            "active_prefill_layer_indices": [int(x) for x in replaced_indices],
            "active_prefill_layer_count": int(replaced_layers),
            "debug_override_layer_indices": requested_layer_indices,
            "config": (
                None
                if baseline_mode
                else {
                    "window": int(args.window),
                    "landmark_stride": (
                        None if int(args.landmark_stride) <= 0 else int(args.landmark_stride)
                    ),
                    "num_cycles": int(max(0, args.num_cycles)),
                    "head_chunk_size": int(max(1, args.head_chunk_size)),
                    "query_chunk_size": int(max(1, args.query_chunk_size)),
                    "decode_backend": butterfly_decode_backend_public,
                    "edge_disjoint": bool(args.edge_disjoint),
                    "enforce_hamiltonian": bool(args.enforce_hamiltonian),
                    "fused_dispatch": bool(not debug_disable_fused_dispatch),
                }
            ),
        },
        "observability": {
            "runtime_summary_always_on": True,
            "hsa_trace_samples_enabled": bool(args.hsa_trace),
            "hsa_trace_max_layers": int(args.hsa_trace_max_layers),
            "hsa_trace_max_steps": int(args.hsa_trace_max_steps),
            "hsa_trace_layer_indices": replaced_indices,
        },
        "status": "interrupted",
        "status_reason": None,
        "progress": {
            "single_turn_rows_completed": 0,
            "multi_turns_completed": 0,
            "quality_tasks_completed": 0,
            "last_stage": None,
            "last_seq_len": None,
            "last_repeat": None,
        },
        "timing": {
            "writeout_sec_last": 0.0,
            "writeout_sec_total": 0.0,
            "script_start_unix_sec": float(time.time()),
        },
        "single_turn": None,
        "multi_turn": None,
        "quality": None,
    }

    results_path = args.out_dir / "results.json"
    summary_json_path = args.out_dir / "summary.json"
    summary_md_path = args.out_dir / "summary.md"

    def _flush() -> float:
        t0 = time.perf_counter()
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_payload = {
            "created_at": payload.get("created_at"),
            "mode": payload.get("mode"),
            "butterfly_decode_backend": payload.get("butterfly_decode_backend"),
            "butterfly": payload.get("butterfly"),
            "status": payload.get("status"),
            "status_reason": payload.get("status_reason"),
            "single_turn": _build_single_turn_summary_rows(payload.get("single_turn") or []),
        }
        summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        summary_md_path.write_text(_render_summary_markdown(payload), encoding="utf-8")
        dt = float(time.perf_counter() - t0)
        timing = payload.setdefault("timing", {})
        timing["writeout_sec_last"] = dt
        timing["writeout_sec_total"] = float(timing.get("writeout_sec_total", 0.0)) + dt
        return dt

    _flush()

    def _set_stage(
        stage: str,
        *,
        seq_len: Optional[int] = None,
        repeat: Optional[int] = None,
    ) -> None:
        progress = payload.setdefault("progress", {})
        progress["last_stage"] = str(stage)
        if seq_len is not None:
            progress["last_seq_len"] = int(seq_len)
        if repeat is not None:
            progress["last_repeat"] = int(repeat)

    caught: Optional[BaseException] = None
    try:
        if not args.skip_single_turn:
            payload["single_turn"] = []
            _set_stage("single_turn")

            def _on_single_row(row: Dict[str, Any]) -> None:
                payload["single_turn"].append(dict(row))
                progress = payload.setdefault("progress", {})
                progress["single_turn_rows_completed"] = int(len(payload["single_turn"]))
                _set_stage(
                    "single_turn_row_complete",
                    seq_len=int(row.get("seq_len", 0)),
                    repeat=int(row.get("repeat", 0)),
                )
                write_sec = _flush()
                payload["single_turn"][-1].setdefault("stage_timing_sec", {})["writeout_sec"] = (
                    write_sec
                )
                _flush()

            _run_single_turn(
                model,
                tokenizer,
                mode=mode,
                seq_lens=[int(x) for x in args.seq_lens],
                decode_len=int(args.decode_len),
                repeats=int(args.repeats),
                chunk_size=int(args.chunk_size),
                kv_step=int(args.kv_step),
                cooldown_sec=float(args.cooldown_sec),
                stage_timeout_sec=float(args.stage_timeout_sec),
                heartbeat_sec=float(args.heartbeat_sec),
                hsa_trace=bool(args.hsa_trace),
                hsa_trace_layer_indices=replaced_indices,
                hsa_trace_max_layers=int(args.hsa_trace_max_layers),
                hsa_trace_max_steps=int(args.hsa_trace_max_steps),
                on_row=_on_single_row,
            )

        if not args.skip_multi_turn:
            _set_stage("multi_turn")
            payload["multi_turn"] = {
                "turns": int(args.turns),
                "target_total_context": int(args.multi_target_context),
                "decode_len_per_turn": int(args.multi_decode_len),
                "max_position_embeddings": int(position_limit) if position_limit is not None else None,
                "prompt_budget_limit": (
                    int(position_limit) - int(args.multi_decode_len)
                    if position_limit is not None
                    else None
                ),
                "session_e2e_sec": 0.0,
                "per_turn": [],
            }

            def _on_multi_turn(turn_row: Dict[str, Any]) -> None:
                multi_turn_payload = payload.setdefault("multi_turn", {})
                per_turn_rows = multi_turn_payload.setdefault("per_turn", [])
                per_turn_rows.append(dict(turn_row))
                multi_turn_payload["session_e2e_sec"] = float(
                    multi_turn_payload.get("session_e2e_sec", 0.0)
                ) + float(turn_row.get("e2e_sec", 0.0))
                payload.setdefault("progress", {})["multi_turns_completed"] = int(
                    len(per_turn_rows)
                )
                _set_stage("multi_turn_turn_complete")
                _flush()

            payload["multi_turn"] = _run_multi_turn(
                model,
                tokenizer,
                turns=int(args.turns),
                target_total_context=int(args.multi_target_context),
                decode_len=int(args.multi_decode_len),
                chunk_size=int(args.chunk_size),
                kv_step=int(args.kv_step),
                position_limit=position_limit,
                on_turn=_on_multi_turn,
            )
            _flush()

        if not args.skip_quality:
            _set_stage("quality")
            task_id_filter = {
                x.strip() for x in str(args.quality_task_id_filter).split(",") if x.strip()
            }
            payload["quality"] = _run_quality(
                model,
                tokenizer,
                dataset_path=args.quality_dataset,
                decode_len=min(64, int(args.decode_len)),
                chunk_size=int(args.chunk_size),
                kv_step=int(args.kv_step),
                task_id_filter=(task_id_filter or None),
                trace_task_id=(str(args.trace_quality_task_id).strip() or None),
                trace_topk=int(args.trace_topk),
                trace_max_steps=int(args.trace_max_steps),
                on_task=lambda _row, _c, _n: (
                    payload.setdefault("progress", {}).__setitem__(
                        "quality_tasks_completed",
                        int(_n),
                    ),
                    _set_stage("quality_task_complete"),
                    _flush(),
                ),
            )
            _flush()
        payload["status"] = "completed"
        payload["status_reason"] = None
    except TimeoutError as exc:
        payload["status"] = "timeout"
        payload["status_reason"] = str(exc)
        caught = exc
    except KeyboardInterrupt as exc:
        progress = payload.get("progress", {})
        rows_done = int(progress.get("single_turn_rows_completed", 0))
        payload["status"] = "hang_suspected" if rows_done <= 0 else "interrupted"
        payload["status_reason"] = "KeyboardInterrupt"
        caught = exc
    except Exception as exc:  # pragma: no cover - safety capture for long runs
        payload["status"] = "interrupted"
        payload["status_reason"] = f"{type(exc).__name__}: {exc}"
        caught = exc
    finally:
        payload["timing"]["script_end_unix_sec"] = float(time.time())
        _flush()
        _log(f"Wrote {results_path}")

    if caught is not None:
        raise caught


if __name__ == "__main__":
    main()
