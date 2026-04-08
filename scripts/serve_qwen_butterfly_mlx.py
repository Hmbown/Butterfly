#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.mlx_kv_quant import (  # noqa: E402
    MLXKVQuantizationConfig,
    maybe_quantize_mlx_prompt_cache,
    summarize_mlx_prompt_cache_quantization,
    validate_mlx_kv_quantization_config,
)
from bna.integrations.qwen_mlx import (  # noqa: E402
    QwenWayfinderAttention,
    QwenWayfinderConfig,
    swap_qwen_attention_with_wayfinder,
    validate_qwen35_full_attention_layers,
)
from bna.integrations.qwen_mlx_loader import (  # noqa: E402
    load_qwen_mlx_model,
    resolve_qwen_mlx_model_path,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / float(1024**3), 3)


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


def _clear_workspace() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _active_memory() -> int:
    if hasattr(mx, "get_active_memory"):
        return int(mx.get_active_memory())
    return int(mx.metal.get_active_memory())


def _cache_memory() -> int:
    if hasattr(mx, "get_cache_memory"):
        return int(mx.get_cache_memory())
    return int(mx.metal.get_cache_memory())


def _make_openai_sampler(temperature: float, top_p: float):
    temp = float(max(0.0, temperature))
    p = float(max(0.0, min(1.0, top_p)))
    if p >= 1.0:
        p = 0.0
    return make_sampler(temp=temp, top_p=p)


def _normalize_num_cycles(raw: str) -> int | str:
    value = str(raw).strip().lower()
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--num-cycles must be >= 0 or 'auto'")
    return parsed


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"text", "input_text"}:
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def _build_prompt(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except TypeError:
            try:
                return str(tokenizer.apply_chat_template(messages, tokenize=False))
            except Exception:
                pass
        except Exception:
            pass

    lines: List[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _coerce_message_content(message.get("content", ""))
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


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


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if q <= 0.0:
        return xs[0]
    if q >= 100.0:
        return xs[-1]
    idx = int(round((q / 100.0) * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


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
            "quantized_kv_cache": "quantized_kv_stock_decode",
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


def _estimate_tree_nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, mx.array):
        try:
            mx.eval(obj)
        except Exception:
            pass
        if hasattr(obj, "nbytes"):
            return int(obj.nbytes)
        try:
            return int(obj.size * obj.itemsize)
        except Exception:
            return 0
    if isinstance(obj, dict):
        return sum(_estimate_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_tree_nbytes(v) for v in obj)
    if hasattr(obj, "state"):
        try:
            return _estimate_tree_nbytes(obj.state)
        except Exception:
            return 0
    return 0


def _memory_snapshot(memory_budget_bytes: int) -> Dict[str, Any]:
    device_info = mx.device_info()
    memory_size = int(device_info.get("memory_size", 0))
    recommended = int(device_info.get("max_recommended_working_set_size", memory_size))
    active_bytes = _active_memory()
    cache_bytes = _cache_memory()
    peak_bytes = _peak_memory()
    used_bytes = active_bytes + cache_bytes
    available_bytes = max(0, int(memory_budget_bytes) - used_bytes)
    return {
        "device": "metal" if mx.metal.is_available() else "cpu",
        "device_name": device_info.get("device_name"),
        "architecture": device_info.get("architecture"),
        "memory_size_bytes": memory_size,
        "memory_size_gb": _bytes_to_gib(memory_size),
        "recommended_working_set_bytes": recommended,
        "recommended_working_set_gb": _bytes_to_gib(recommended),
        "memory_budget_bytes": int(memory_budget_bytes),
        "memory_budget_gb": _bytes_to_gib(memory_budget_bytes),
        "active_memory_bytes": active_bytes,
        "active_memory_gb": _bytes_to_gib(active_bytes),
        "cache_memory_bytes": cache_bytes,
        "cache_memory_gb": _bytes_to_gib(cache_bytes),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gb": _bytes_to_gib(peak_bytes),
        "available_memory_bytes": available_bytes,
        "available_memory_gb": _bytes_to_gib(available_bytes),
    }


def _resolve_cache_dtype_nbytes(model: Any) -> int:
    layers = getattr(model, "layers", None)
    if not layers:
        return 2
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for weight_name in ("k_proj", "q_proj", "v_proj"):
            weight_owner = getattr(attn, weight_name, None)
            weight = getattr(weight_owner, "weight", None)
            if weight is None:
                continue
            if hasattr(weight, "dtype"):
                try:
                    return int(mx.array([0], dtype=weight.dtype).nbytes)
                except Exception:
                    pass
    return 2


def _estimate_kv_bytes_per_token(
    model: Any,
    *,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
) -> int:
    full_attention_layers = []
    for layer in getattr(model, "layers", []) or []:
        if getattr(layer, "is_linear", None) is False and getattr(layer, "self_attn", None) is not None:
            full_attention_layers.append(layer.self_attn)
    if not full_attention_layers:
        return 0
    sample_attn = full_attention_layers[0]
    n_kv_heads = int(
        getattr(sample_attn, "n_kv_heads", None)
        or getattr(sample_attn, "num_key_value_heads", None)
        or getattr(sample_attn, "num_kv_heads", None)
        or 1
    )
    head_dim = int(getattr(sample_attn, "head_dim", 0) or 0)
    if head_dim <= 0:
        scale = getattr(sample_attn, "scale", None) or getattr(sample_attn, "scaling", None)
        if scale is not None:
            head_dim = int(round(float(scale) ** -2))
    if head_dim <= 0:
        return 0
    dtype_nbytes = _resolve_cache_dtype_nbytes(model)
    element_nbytes = float(dtype_nbytes)
    if kv_bits is not None:
        element_nbytes = (float(kv_bits) / 8.0) + (
            2.0 * float(dtype_nbytes) / float(max(1, kv_group_size))
        )
    total = 2.0 * float(len(full_attention_layers) * n_kv_heads * head_dim) * element_nbytes
    return int(round(total))


def _collect_wayfinder_trace_snapshot(
    model: Any,
    *,
    layer_indices: Optional[Sequence[int]] = None,
    max_layers: int = 0,
) -> List[Dict[str, Any]]:
    layers = list(getattr(model, "layers", []) or [])
    if layer_indices is not None:
        selected = [int(idx) for idx in layer_indices if 0 <= int(idx) < len(layers)]
    else:
        selected = []
        for idx, layer in enumerate(layers):
            if isinstance(getattr(layer, "self_attn", None), QwenWayfinderAttention):
                selected.append(int(idx))
    if int(max_layers) > 0:
        selected = selected[: int(max_layers)]

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
        "quantized_kv_cache",
        "wayfinder_decode_dense_triggered",
        "wayfinder_decode_backend",
        "adaptive_graph_reuse",
        "permute_head_chunk_effective",
        "permute_query_chunk_effective",
        "sparse_active_mode",
    )
    out: List[Dict[str, Any]] = []
    for idx in selected:
        attn = getattr(layers[idx], "self_attn", None)
        if not isinstance(attn, QwenWayfinderAttention):
            continue
        profile = attn.last_profile.to_dict() if hasattr(attn, "last_profile") else {}
        notes_obj = profile.get("notes")
        notes_source = notes_obj if isinstance(notes_obj, dict) and notes_obj else profile
        notes_subset = {key: notes_source.get(key) for key in note_keys if key in notes_source}
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


def _summarize_wayfinder_trace(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    path_counts: Counter[str] = Counter()
    cache_source_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
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
            if phase == "decode":
                decode_layer_obs += 1
            elif phase == "prefill":
                prefill_layer_obs += 1

            raw_path = str(layer_row.get("path") or "unknown")
            public_path = _public_path_label(raw_path)
            path_counts[public_path] += 1
            notes = layer_row.get("notes") or {}
            stock_reason = _public_stock_fallback_reason(
                raw_reason=notes.get("dense_fallback_reason"),
                notes=notes,
                raw_path=raw_path,
            )
            is_stock_fallback = stock_reason is not None
            if is_stock_fallback:
                stock_reason_counts[stock_reason] += 1
                fallback_layer_obs += 1
                sample_has_fallback = True
                if phase == "decode":
                    decode_fallback_layer_obs += 1
                elif phase == "prefill":
                    prefill_fallback_layer_obs += 1

            cache_source = notes.get("cache_source")
            if cache_source is not None:
                cache_source_counts[str(cache_source)] += 1
            if bool(notes.get("cache_hit")):
                cache_hit_obs += 1

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
        "path_counts": dict(path_counts),
        "cache_source_counts": dict(cache_source_counts),
        "phase_counts": dict(phase_counts),
        "stock_fallback_reason_counts": dict(stock_reason_counts),
        "stock_fallback_layer_observations": int(fallback_layer_obs),
        "stock_fallback_decode_layer_observations": int(decode_fallback_layer_obs),
        "stock_fallback_prefill_layer_observations": int(prefill_fallback_layer_obs),
        "stock_fallback_decode_steps": int(decode_fallback_steps),
        "stock_fallback_prefill_steps": int(prefill_fallback_steps),
        "stock_fallback_share_run": stock_fallback_share_run,
        "stock_fallback_share_decode_layers": stock_fallback_share_decode_layers,
        "stock_fallback_share_prefill_layers": stock_fallback_share_prefill_layers,
        "stock_fallback_share_decode_steps": stock_fallback_share_decode_steps,
        "stock_fallback_share_prefill_steps": stock_fallback_share_prefill_steps,
        "fallback_share_known": bool(total_obs > 0),
        "cache_hit_ratio": float(cache_hit_obs / total_obs) if total_obs else 0.0,
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
            float(sum(total_ms_values) / len(total_ms_values)) if total_ms_values else None
        ),
        "total_ms_p95": _percentile(total_ms_values, 95.0),
    }


def _prefill_prompt_tokens(
    model: Any,
    *,
    token_ids: Sequence[int],
    prompt_cache: Sequence[Any],
    chunk_size: int,
    trace_samples: Optional[List[Dict[str, Any]]] = None,
    trace_layer_indices: Optional[Sequence[int]] = None,
) -> float:
    if not token_ids:
        return 0.0

    tokens = mx.array([list(token_ids)], dtype=mx.int32)
    seq_len = int(tokens.shape[1])
    t0 = time.perf_counter()
    chunk_idx = 0
    for start in range(0, seq_len, int(max(1, chunk_size))):
        end = min(seq_len, start + int(max(1, chunk_size)))
        chunk_idx += 1
        logits = model(tokens[:, start:end], cache=prompt_cache)
        mx.eval(logits)
        if trace_samples is not None:
            layer_rows = _collect_wayfinder_trace_snapshot(
                model,
                layer_indices=trace_layer_indices,
            )
            if layer_rows:
                trace_samples.append(
                    {
                        "phase": "prefill",
                        "chunk": int(chunk_idx),
                        "chunk_start": int(start),
                        "chunk_end": int(end),
                        "layers": layer_rows,
                    }
                )
        _clear_workspace()
    return float(time.perf_counter() - t0)


def _classify_generation_failure(exc: Exception) -> Tuple[int, str]:
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    if "memory" in lowered or "out of" in lowered:
        return 507, message
    return 500, message


@dataclass
class PromptCacheEntry:
    tokens: Tuple[int, ...]
    prompt_cache: List[Any]
    bytes_estimate: int
    created_at: float
    last_used_at: float
    hits: int = 0


class PromptCacheStore:
    def __init__(self, *, max_entries: int, max_bytes: int):
        self.max_entries = int(max(0, max_entries))
        self.max_bytes = int(max(0, max_bytes))
        self._entries: "OrderedDict[Tuple[int, ...], PromptCacheEntry]" = OrderedDict()
        self._bytes = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    @property
    def total_bytes(self) -> int:
        return int(self._bytes)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "entries": int(len(self._entries)),
            "bytes": int(self._bytes),
            "bytes_gb": _bytes_to_gib(self._bytes),
            "max_entries": int(self.max_entries),
            "max_bytes": int(self.max_bytes),
            "max_bytes_gb": _bytes_to_gib(self.max_bytes),
            "token_lengths": [int(len(key)) for key in self._entries.keys()],
        }

    def _evict_key(self, key: Tuple[int, ...]) -> Optional[PromptCacheEntry]:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes = max(0, int(self._bytes - entry.bytes_estimate))
        return entry

    def evict_all(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def _find_best_key(self, prompt_ids: Sequence[int]) -> Optional[Tuple[int, ...]]:
        if not self.enabled:
            return None
        prompt_tuple = tuple(int(x) for x in prompt_ids)
        best_key: Optional[Tuple[int, ...]] = None
        best_len = 0
        for key in self._entries.keys():
            key_len = len(key)
            if key_len <= best_len or key_len > len(prompt_tuple):
                continue
            if key == prompt_tuple[:key_len]:
                best_key = key
                best_len = key_len
        return best_key

    def best_prefix_len(self, prompt_ids: Sequence[int]) -> int:
        key = self._find_best_key(prompt_ids)
        return int(len(key)) if key is not None else 0

    def lookup(self, prompt_ids: Sequence[int]) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
        key = self._find_best_key(prompt_ids)
        if key is None:
            return None, {
                "cache_hit": False,
                "prefix_tokens": 0,
                "match_type": "miss",
                "bytes_estimate": 0,
            }
        entry = self._entries.get(key)
        if entry is None:
            return None, {
                "cache_hit": False,
                "prefix_tokens": 0,
                "match_type": "miss",
                "bytes_estimate": 0,
            }
        self._entries.move_to_end(key)
        entry.last_used_at = time.time()
        entry.hits += 1
        try:
            working_cache = copy.deepcopy(entry.prompt_cache)
        except Exception:
            self._evict_key(key)
            return None, {
                "cache_hit": False,
                "prefix_tokens": 0,
                "match_type": "stale",
                "bytes_estimate": 0,
            }
        match_type = "exact" if len(key) == max(0, len(prompt_ids) - 1) else "prefix"
        return working_cache, {
            "cache_hit": True,
            "prefix_tokens": int(len(key)),
            "match_type": match_type,
            "bytes_estimate": int(entry.bytes_estimate),
        }

    def put(self, tokens: Sequence[int], prompt_cache: List[Any]) -> Dict[str, Any]:
        token_tuple = tuple(int(x) for x in tokens)
        if not self.enabled:
            return {"stored": False, "reason": "disabled"}
        if not token_tuple:
            return {"stored": False, "reason": "empty"}
        bytes_estimate = _estimate_tree_nbytes(prompt_cache)
        if bytes_estimate <= 0:
            return {"stored": False, "reason": "empty_bytes"}
        if bytes_estimate > self.max_bytes:
            return {
                "stored": False,
                "reason": "too_large",
                "bytes_estimate": int(bytes_estimate),
            }
        if token_tuple in self._entries:
            self._evict_key(token_tuple)
        while self._entries and (
            len(self._entries) >= self.max_entries or (self._bytes + bytes_estimate) > self.max_bytes
        ):
            _old_key, old_entry = self._entries.popitem(last=False)
            self._bytes = max(0, int(self._bytes - old_entry.bytes_estimate))
        now = time.time()
        self._entries[token_tuple] = PromptCacheEntry(
            tokens=token_tuple,
            prompt_cache=prompt_cache,
            bytes_estimate=int(bytes_estimate),
            created_at=now,
            last_used_at=now,
        )
        self._bytes = sum(entry.bytes_estimate for entry in self._entries.values())
        return {
            "stored": True,
            "bytes_estimate": int(bytes_estimate),
            "entries": int(len(self._entries)),
        }


@dataclass
class ServerState:
    model: Any
    tokenizer: Any
    model_path: str
    model_id: str
    mode: str
    decode_backend: str
    lock: threading.Lock
    replaced_layer_indices: List[int]
    full_attention_layer_indices: List[int]
    max_input_tokens: Optional[int]
    max_total_tokens: Optional[int]
    prefill_chunk_size: int
    query_chunk_size: int
    memory_budget_bytes: int
    memory_reserve_bytes: int
    memory_estimate_multiplier: float
    kv_quantization: MLXKVQuantizationConfig
    kv_bytes_per_token_dense: int
    kv_bytes_per_token_quantized: Optional[int]
    prompt_cache_store: PromptCacheStore
    warmup_seq_lens: List[int]
    warmup_status: Dict[str, Any] = field(default_factory=dict)


def _build_wayfinder_config(args: argparse.Namespace) -> QwenWayfinderConfig:
    return QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=_normalize_num_cycles(str(args.num_cycles)),
        edge_disjoint=not bool(args.disable_edge_disjoint),
        enforce_hamiltonian=not bool(args.allow_non_hamiltonian),
        seed=int(args.seed),
        edge_bias=True,
        window_drop=0.0,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        permute_head_chunk_size=int(max(1, args.head_chunk_size)),
        query_chunk_size=int(max(1, args.query_chunk_size)),
        use_fused_dispatch=not bool(args.disable_fused_dispatch),
        wayfinder_decode_backend=str(args.butterfly_decode_backend),
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
        permute_stream_o_proj=bool(args.butterfly_stream_o_proj),
    )


def _check_layer_layout(
    *,
    model: Any,
    allow_layer_mismatch: bool,
) -> List[int]:
    try:
        return validate_qwen35_full_attention_layers(
            model,
            allow_mismatch=allow_layer_mismatch,
        )
    except ValueError as exc:
        raise SystemExit(
            f"{exc}. Refusing Butterfly swap by default; pass --allow-layer-mismatch only if "
            "you intentionally want to bypass the Qwen 3.5 stock-layout check."
        ) from exc


def _admission_check(
    state: ServerState,
    *,
    prompt_tokens: int,
    max_tokens: int,
    cached_prefix_tokens: int,
) -> Dict[str, Any]:
    snapshot = _memory_snapshot(state.memory_budget_bytes)
    incremental_tokens = max(1, int(prompt_tokens) - int(cached_prefix_tokens)) + int(max_tokens)
    estimated_request_bytes = int(
        max(0.0, state.memory_estimate_multiplier)
        * float(state.kv_bytes_per_token_dense)
        * float(incremental_tokens)
    )
    required_bytes = (
        int(snapshot["active_memory_bytes"])
        + int(snapshot["cache_memory_bytes"])
        + int(estimated_request_bytes)
        + int(state.memory_reserve_bytes)
    )
    admitted = required_bytes <= int(state.memory_budget_bytes)
    return {
        "admitted": bool(admitted),
        "cached_prefix_tokens": int(cached_prefix_tokens),
        "estimated_request_bytes": int(estimated_request_bytes),
        "estimated_request_gb": _bytes_to_gib(estimated_request_bytes),
        "required_bytes": int(required_bytes),
        "required_gb": _bytes_to_gib(required_bytes),
        "memory": snapshot,
    }


def _build_request_meta(
    *,
    state: ServerState,
    trace_samples: Sequence[Dict[str, Any]],
    working_cache: Sequence[Any],
    prompt_tokens: int,
    completion_tokens: int,
    cache_meta: Dict[str, Any],
    cache_store_meta: Dict[str, Any],
    admission_meta: Dict[str, Any],
    finish_reason: Optional[str],
    request_started_at: float,
    ttft_sec: Optional[float],
    itl_values: Sequence[float],
) -> Dict[str, Any]:
    trace_summary = _summarize_wayfinder_trace(trace_samples)
    kv_quantization_meta = summarize_mlx_prompt_cache_quantization(
        working_cache,
        config=state.kv_quantization,
    )
    peak_memory_bytes = _peak_memory()
    e2e_sec = float(time.perf_counter() - request_started_at)
    return {
        "mode": state.mode,
        "decode_backend": state.decode_backend,
        "finish_reason": finish_reason,
        "single_inference_gate": True,
        "ttft_sec": ttft_sec,
        "itl_p50_sec": _percentile(itl_values, 50.0),
        "itl_p95_sec": _percentile(itl_values, 95.0),
        "e2e_sec": e2e_sec,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "stock_fallback_share_run": trace_summary.get("stock_fallback_share_run"),
        "stock_fallback_share_decode_layers": trace_summary.get("stock_fallback_share_decode_layers"),
        "stock_fallback_share_prefill_layers": trace_summary.get("stock_fallback_share_prefill_layers"),
        "stock_fallback_share_decode_steps": trace_summary.get("stock_fallback_share_decode_steps"),
        "stock_fallback_share_prefill_steps": trace_summary.get("stock_fallback_share_prefill_steps"),
        "stock_fallback_reason_counts": dict(trace_summary.get("stock_fallback_reason_counts", {})),
        "fallback_share_known": bool(trace_summary.get("fallback_share_known")),
        "graph_build_ms_mean": trace_summary.get("graph_build_ms_mean"),
        "butterfly_ms_mean": trace_summary.get("butterfly_ms_mean"),
        "butterfly_ms_decode_mean": trace_summary.get("butterfly_ms_decode_mean"),
        "butterfly_ms_prefill_mean": trace_summary.get("butterfly_ms_prefill_mean"),
        "attention_ms_mean": trace_summary.get("attention_ms_mean"),
        "trace_summary": trace_summary,
        "butterfly": {
            "enabled": bool(state.mode == "butterfly"),
            "decode_backend": state.decode_backend,
            "prefill_scope": "qwen35_full_attention_layers",
            "target_prefill_layer_indices": [int(x) for x in state.full_attention_layer_indices],
            "active_prefill_layer_indices": [int(x) for x in state.replaced_layer_indices],
            "active_prefill_layer_count": int(len(state.replaced_layer_indices)),
            "prefill_chunk_size": int(state.prefill_chunk_size),
            "query_chunk_size": int(state.query_chunk_size),
        },
        "kv_quantization": {
            **state.kv_quantization.to_dict(),
            "policy": "post_prefill_before_decode",
            "preserve_dense_prefix_cache": True,
            "working_cache": kv_quantization_meta,
            "estimated_full_attention_kv_bytes_per_token_dense": int(
                state.kv_bytes_per_token_dense
            ),
            "estimated_full_attention_kv_bytes_per_token_quantized": (
                None
                if state.kv_bytes_per_token_quantized is None
                else int(state.kv_bytes_per_token_quantized)
            ),
        },
        "cache": {
            "hit": bool(cache_meta.get("cache_hit", False)),
            "match_type": cache_meta.get("match_type"),
            "prefix_tokens": int(cache_meta.get("prefix_tokens", 0)),
            "entry_bytes_estimate": int(cache_meta.get("bytes_estimate", 0)),
            "store": cache_store_meta,
            "store_stats": state.prompt_cache_store.stats(),
        },
        "admission": admission_meta,
        "memory": {
            **_memory_snapshot(state.memory_budget_bytes),
            "peak_memory_bytes": int(peak_memory_bytes),
            "peak_memory_gb": _bytes_to_gib(peak_memory_bytes),
        },
    }


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Butterfly MLX OpenAI Bridge", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        status = {
            "started_at": time.time(),
            "completed_at": None,
            "ok": False,
            "warmed_seq_lens": [],
            "error": None,
        }
        state.warmup_status = status
        if not state.warmup_seq_lens:
            status["ok"] = True
            status["completed_at"] = time.time()
            return

        _log(
            "Startup warmup: "
            f"seq_lens={state.warmup_seq_lens} prefill_chunk_size={state.prefill_chunk_size}"
        )
        try:
            seed_ids = _encode_text(
                state.tokenizer,
                "System: Butterfly warmup.\nUser: Prime the graph cache for deployment.\nAssistant:",
            )
            if not seed_ids:
                seed_ids = [1]
            for seq_len in state.warmup_seq_lens:
                warm_ids = list(seed_ids)
                while len(warm_ids) < int(seq_len):
                    warm_ids.extend(seed_ids)
                warm_ids = warm_ids[: int(seq_len)]
                if not warm_ids:
                    continue
                prompt_cache = list(make_prompt_cache(state.model))
                prefix_ids = warm_ids[:-1]
                if prefix_ids:
                    _prefill_prompt_tokens(
                        state.model,
                        token_ids=prefix_ids,
                        prompt_cache=prompt_cache,
                        chunk_size=state.prefill_chunk_size,
                    )
                maybe_quantize_mlx_prompt_cache(prompt_cache, config=state.kv_quantization)
                warm_iter = stream_generate(
                    state.model,
                    state.tokenizer,
                    prompt=warm_ids[-1:],
                    max_tokens=1,
                    sampler=_make_openai_sampler(0.0, 1.0),
                    prompt_cache=prompt_cache,
                    kv_bits=state.kv_quantization.bits,
                    kv_group_size=state.kv_quantization.group_size,
                    quantized_kv_start=state.kv_quantization.quantized_kv_start,
                )
                for _ in warm_iter:
                    pass
                _clear_workspace()
                status["warmed_seq_lens"].append(int(seq_len))
            status["ok"] = True
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            _log(f"Warmup failed: {status['error']}")
        finally:
            status["completed_at"] = time.time()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model": state.model_id,
            "model_path": state.model_path,
            "mode": state.mode,
            "decode_backend": state.decode_backend,
            "butterfly": {
                "enabled": bool(state.mode == "butterfly"),
                "decode_backend": state.decode_backend,
                "prefill_scope": "qwen35_full_attention_layers",
                "target_prefill_layer_indices": [int(x) for x in state.full_attention_layer_indices],
                "active_prefill_layer_indices": [int(x) for x in state.replaced_layer_indices],
                "active_prefill_layer_count": int(len(state.replaced_layer_indices)),
                "prefill_chunk_size": int(state.prefill_chunk_size),
                "query_chunk_size": int(state.query_chunk_size),
            },
            "busy": bool(state.lock.locked()),
            "caps": {
                "max_input_tokens": state.max_input_tokens,
                "max_total_tokens": state.max_total_tokens,
                "prefill_chunk_size": state.prefill_chunk_size,
                "query_chunk_size": state.query_chunk_size,
                "prefill_chunk_valid_for_butterfly": bool(
                    state.mode != "butterfly"
                    or state.prefill_chunk_size <= state.query_chunk_size
                ),
            },
            "kv_quantization": {
                **state.kv_quantization.to_dict(),
                "policy": "post_prefill_before_decode",
                "preserve_dense_prefix_cache": True,
                "estimated_full_attention_kv_bytes_per_token_dense": int(
                    state.kv_bytes_per_token_dense
                ),
                "estimated_full_attention_kv_bytes_per_token_quantized": (
                    None
                    if state.kv_bytes_per_token_quantized is None
                    else int(state.kv_bytes_per_token_quantized)
                ),
            },
            "memory": _memory_snapshot(state.memory_budget_bytes),
            "prompt_cache": state.prompt_cache_store.stats(),
            "warmup": dict(state.warmup_status),
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "butterfly-local",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")

        messages: List[Dict[str, Any]] = []
        for message in messages_raw:
            if not isinstance(message, dict):
                continue
            messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": message.get("content", ""),
                }
            )
        if not messages:
            raise HTTPException(status_code=400, detail="No valid chat messages found.")

        request_model = str(payload.get("model", "") or "").strip()
        if request_model and request_model != state.model_id:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model {request_model!r} does not match server model {state.model_id!r}.",
            )

        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 256)
        max_tokens = max(1, min(max_tokens, 4096))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        stream = bool(payload.get("stream", False))
        stream_options = payload.get("stream_options")
        include_usage = isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))

        prompt_text = _build_prompt(state.tokenizer, messages)
        prompt_ids = _encode_text(state.tokenizer, prompt_text)
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="Prompt tokenized to zero tokens.")
        prompt_len = int(len(prompt_ids))
        if state.max_input_tokens is not None and prompt_len > state.max_input_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt too long: {prompt_len} tokens exceeds server cap "
                    f"{state.max_input_tokens}. Reduce the prompt or raise --max-input-tokens."
                ),
            )
        if (
            state.max_total_tokens is not None
            and (prompt_len + max_tokens) > state.max_total_tokens
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested total tokens {prompt_len + max_tokens} exceeds server cap "
                    f"{state.max_total_tokens}. Reduce max_tokens or raise --max-total-tokens."
                ),
            )

        if not state.lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Server is busy. Butterfly MLX runs serial inference only.",
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        sampler = _make_openai_sampler(temperature, top_p)
        request_started_at = time.perf_counter()
        trace_samples: List[Dict[str, Any]] = []
        working_cache: List[Any] = []
        cache_meta: Dict[str, Any] = {
            "cache_hit": False,
            "prefix_tokens": 0,
            "match_type": "miss",
            "bytes_estimate": 0,
        }
        cache_store_meta: Dict[str, Any] = {"stored": False, "reason": "skipped"}
        admission_meta: Dict[str, Any] = {}

        def _release_request_resources() -> None:
            _clear_workspace()
            if state.lock.locked():
                state.lock.release()

        try:
            prefix_hint = state.prompt_cache_store.best_prefix_len(prompt_ids)
            admission_meta = _admission_check(
                state,
                prompt_tokens=prompt_len,
                max_tokens=max_tokens,
                cached_prefix_tokens=prefix_hint,
            )
            if not admission_meta.get("admitted", False):
                detail = (
                    "Request rejected by admission control. "
                    f"estimated_required_gb={admission_meta.get('required_gb')} "
                    f"budget_gb={admission_meta.get('memory', {}).get('memory_budget_gb')} "
                    f"available_gb={admission_meta.get('memory', {}).get('available_memory_gb')}"
                )
                raise HTTPException(status_code=503, detail=detail)

            _reset_peak_memory()

            working_cache, cache_meta = state.prompt_cache_store.lookup(prompt_ids)
            if working_cache is None:
                working_cache = list(make_prompt_cache(state.model))
            prefix_tokens = int(cache_meta.get("prefix_tokens", 0))

            prefix_prefill_ids = prompt_ids[prefix_tokens:-1] if prompt_len > 1 else []
            prefill_sec = _prefill_prompt_tokens(
                state.model,
                token_ids=prefix_prefill_ids,
                prompt_cache=working_cache,
                chunk_size=state.prefill_chunk_size,
                trace_samples=trace_samples,
                trace_layer_indices=state.replaced_layer_indices,
            )

            if prompt_len > 1 and prefix_tokens != (prompt_len - 1):
                cache_store_meta = state.prompt_cache_store.put(
                    prompt_ids[:-1],
                    copy.deepcopy(working_cache),
                )
            elif prompt_len <= 1:
                cache_store_meta = {"stored": False, "reason": "single_token_prompt"}
            else:
                cache_store_meta = {"stored": False, "reason": "exact_prefix_hit"}

            maybe_quantize_mlx_prompt_cache(working_cache, config=state.kv_quantization)
            generation_prompt = prompt_ids[-1:] if prompt_len > 1 else prompt_ids

            def _iter_generation():
                return stream_generate(
                    state.model,
                    state.tokenizer,
                    prompt=generation_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=working_cache,
                    kv_bits=state.kv_quantization.bits,
                    kv_group_size=state.kv_quantization.group_size,
                    quantized_kv_start=state.kv_quantization.quantized_kv_start,
                )

            eos_token_ids = set(int(x) for x in getattr(state.tokenizer, "eos_token_ids", []) or [])

            if not stream:
                text_parts: List[str] = []
                finish_reason: Optional[str] = None
                token_event_intervals: List[float] = []
                previous_token_at = request_started_at
                ttft_sec: Optional[float] = None
                final_generation_tokens = 0
                final_response_token = 0
                for response in _iter_generation():
                    now = time.perf_counter()
                    layer_rows = _collect_wayfinder_trace_snapshot(
                        state.model,
                        layer_indices=state.replaced_layer_indices,
                    )
                    if layer_rows:
                        phase_name = "decode" if response.finish_reason is None or response.finish_reason == "length" else "finalize"
                        trace_samples.append(
                            {
                                "phase": phase_name,
                                "step": int(response.generation_tokens),
                                "layers": layer_rows,
                            }
                        )
                    if response.text:
                        text_parts.append(response.text)
                    token_emitted = response.finish_reason is None or response.finish_reason == "length"
                    if token_emitted:
                        interval = float(now - previous_token_at)
                        token_event_intervals.append(interval)
                        previous_token_at = now
                        if ttft_sec is None:
                            ttft_sec = float(now - request_started_at)
                    finish_reason = response.finish_reason or finish_reason
                    final_generation_tokens = int(response.generation_tokens)
                    final_response_token = int(response.token)

                completion_tokens = int(final_generation_tokens)
                if finish_reason == "stop" and final_response_token in eos_token_ids:
                    completion_tokens = max(0, completion_tokens - 1)
                usage = {
                    "prompt_tokens": int(prompt_len),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_len + completion_tokens),
                }
                butterfly_meta = _build_request_meta(
                    state=state,
                    trace_samples=trace_samples,
                    working_cache=working_cache,
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    cache_meta=cache_meta,
                    cache_store_meta=cache_store_meta,
                    admission_meta=admission_meta,
                    finish_reason=finish_reason,
                    request_started_at=request_started_at,
                    ttft_sec=ttft_sec,
                    itl_values=token_event_intervals[1:],
                )
                butterfly_meta["prefill_sec_prefix_delta"] = float(prefill_sec)
                return JSONResponse(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": state.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "".join(text_parts)},
                                "finish_reason": finish_reason or "stop",
                            }
                        ],
                        "usage": usage,
                        "butterfly_meta": butterfly_meta,
                    }
                )

            async def _event_stream():
                finish_reason: Optional[str] = None
                token_event_intervals: List[float] = []
                previous_token_at = request_started_at
                ttft_sec: Optional[float] = None
                final_generation_tokens = 0
                final_response_token = 0

                role_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                try:
                    for response in _iter_generation():
                        now = time.perf_counter()
                        layer_rows = _collect_wayfinder_trace_snapshot(
                            state.model,
                            layer_indices=state.replaced_layer_indices,
                        )
                        if layer_rows:
                            phase_name = "decode" if response.finish_reason is None or response.finish_reason == "length" else "finalize"
                            trace_samples.append(
                                {
                                    "phase": phase_name,
                                    "step": int(response.generation_tokens),
                                    "layers": layer_rows,
                                }
                            )
                        token_emitted = response.finish_reason is None or response.finish_reason == "length"
                        if token_emitted:
                            interval = float(now - previous_token_at)
                            token_event_intervals.append(interval)
                            previous_token_at = now
                            if ttft_sec is None:
                                ttft_sec = float(now - request_started_at)
                        finish_reason = response.finish_reason or finish_reason
                        final_generation_tokens = int(response.generation_tokens)
                        final_response_token = int(response.token)

                        if response.text or response.finish_reason is not None:
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": (
                                            {"content": response.text}
                                            if response.text
                                            else {}
                                        ),
                                        "finish_reason": response.finish_reason,
                                    }
                                ],
                            }
                            if response.finish_reason is not None:
                                completion_tokens = int(final_generation_tokens)
                                if response.finish_reason == "stop" and final_response_token in eos_token_ids:
                                    completion_tokens = max(0, completion_tokens - 1)
                                usage = {
                                    "prompt_tokens": int(prompt_len),
                                    "completion_tokens": int(completion_tokens),
                                    "total_tokens": int(prompt_len + completion_tokens),
                                }
                                butterfly_meta = _build_request_meta(
                                    state=state,
                                    trace_samples=trace_samples,
                                    working_cache=working_cache,
                                    prompt_tokens=prompt_len,
                                    completion_tokens=completion_tokens,
                                    cache_meta=cache_meta,
                                    cache_store_meta=cache_store_meta,
                                    admission_meta=admission_meta,
                                    finish_reason=response.finish_reason,
                                    request_started_at=request_started_at,
                                    ttft_sec=ttft_sec,
                                    itl_values=token_event_intervals[1:],
                                )
                                butterfly_meta["prefill_sec_prefix_delta"] = float(prefill_sec)
                                chunk["butterfly_meta"] = butterfly_meta
                                if include_usage:
                                    chunk["usage"] = usage
                            yield f"data: {json.dumps(chunk)}\n\n"

                        if await request.is_disconnected():
                            _log("Client disconnected during streaming response; stopping generation.")
                            break
                    yield "data: [DONE]\n\n"
                finally:
                    _release_request_resources()

            return StreamingResponse(_event_stream(), media_type="text/event-stream")

        except HTTPException:
            _release_request_resources()
            raise
        except Exception as exc:
            state.prompt_cache_store.evict_all()
            _release_request_resources()
            status_code, detail = _classify_generation_failure(exc)
            raise HTTPException(status_code=status_code, detail=detail) from exc
        finally:
            if not stream and state.lock.locked():
                _release_request_resources()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Qwen 3.5 with stock or Butterfly attention on MLX as an OpenAI-compatible endpoint."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument(
        "--mode",
        type=str,
        default="butterfly",
        help="'butterfly' = Butterfly attention prefill on the 8 swapped Qwen 3.5 layers. 'stock' = default Qwen 3.5 attention path. Legacy aliases remain accepted.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--hf-home", type=str, default="/Volumes/VIXinSSD/hf_cache")
    parser.add_argument("--hf-hub-cache", type=str, default="/Volumes/VIXinSSD/hf_cache/hub")
    parser.add_argument("--hf-offline", action="store_true", default=False)

    parser.add_argument("--max-input-tokens", type=int, default=131072)
    parser.add_argument("--max-total-tokens", type=int, default=131584)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Enable MLX-LM KV cache quantization for full-attention layers only.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for MLX KV cache quantization.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="Quantize the full-attention KV cache once offset >= this token count.",
    )

    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--memory-budget-fraction", type=float, default=0.9)
    parser.add_argument("--memory-reserve-gb", type=float, default=1.5)
    parser.add_argument("--memory-estimate-multiplier", type=float, default=1.35)

    parser.add_argument("--prompt-cache-max-entries", type=int, default=4)
    parser.add_argument("--prompt-cache-max-bytes-gb", type=float, default=2.5)

    parser.add_argument("--disable-startup-warmup", action="store_true", default=False)
    parser.add_argument("--warmup-seq-lens", type=int, nargs="*", default=[2048, 8192, 16384])

    parser.add_argument("--allow-layer-mismatch", action="store_true", default=False)

    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--landmark-stride", type=int, default=64)
    parser.add_argument("--num-cycles", type=str, default="1")
    parser.add_argument("--head-chunk-size", type=int, default=2)
    parser.add_argument("--query-chunk-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-fused-dispatch", action="store_true", default=False)
    parser.add_argument("--allow-non-hamiltonian", action="store_true", default=False)
    parser.add_argument("--disable-edge-disjoint", action="store_true", default=False)
    parser.add_argument(
        "--butterfly-decode-backend",
        type=str,
        default="stock",
        help="Decode policy for swapped layers. 'stock' (legacy alias: 'dense') keeps the default stock decode path. 'experimental' (legacy alias: 'active_permute') enables Butterfly decode experiments.",
    )
    parser.add_argument("--wayfinder-decode-backend", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--butterfly-stream-o-proj",
        action="store_true",
        default=False,
        help="Enable streamed output projection on the Butterfly prefill path.",
    )
    parser.add_argument("--permute-stream-o-proj", action="store_true", default=False, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        mx.random.seed(int(args.seed))
    except Exception:
        pass

    mode = str(args.mode).strip().lower()
    if mode == "wayfinder":
        mode = "butterfly"
    elif mode in {"native", "hybrid", "dense", "stock"}:
        mode = "stock"
    elif mode != "butterfly":
        parser.error("--mode must be one of ['stock', 'butterfly'] plus legacy aliases")

    decode_backend_arg = (
        str(args.butterfly_decode_backend).strip().lower()
        if str(args.butterfly_decode_backend).strip()
        else str(args.wayfinder_decode_backend).strip().lower()
    )
    if decode_backend_arg == "stock":
        decode_backend_arg = "dense"
    if decode_backend_arg == "experimental":
        decode_backend_arg = "active_permute"
    if decode_backend_arg not in {"active_permute", "dense"}:
        parser.error("--butterfly-decode-backend must be one of ['stock', 'experimental'] plus legacy aliases ['dense', 'active_permute']")
    kv_quantization = MLXKVQuantizationConfig(
        bits=(None if args.kv_bits is None else int(args.kv_bits)),
        group_size=int(args.kv_group_size),
        quantized_kv_start=int(args.quantized_kv_start),
    )
    try:
        validate_mlx_kv_quantization_config(kv_quantization)
    except ValueError as exc:
        parser.error(str(exc))
    if kv_quantization.enabled and mode == "butterfly" and decode_backend_arg != "dense":
        parser.error(
            "--kv-bits currently requires --butterfly-decode-backend stock "
            "because the experimental Butterfly decode path does not support quantized KV."
        )
    if mode == "butterfly" and int(args.prefill_chunk_size) > int(max(1, args.query_chunk_size)):
        parser.error(
            "--prefill-chunk-size exceeds --query-chunk-size in butterfly mode; "
            "later prefill chunks would fall back to stock attention (active_large_q), "
            "so this is not a valid Butterfly prefill server configuration."
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

    _log(f"Loading model: {resolved_model_path}")
    model, tokenizer, _cfg = load_qwen_mlx_model(
        resolved_model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _log("Model loaded")

    full_attention_layer_indices = _check_layer_layout(
        model=model,
        allow_layer_mismatch=bool(args.allow_layer_mismatch) or mode == "stock",
    )

    replaced_layer_indices: List[int] = []
    if mode == "butterfly":
        args.butterfly_decode_backend = decode_backend_arg
        args.butterfly_stream_o_proj = bool(args.butterfly_stream_o_proj) or bool(
            args.permute_stream_o_proj
        )
        wf_cfg = _build_wayfinder_config(args)
        replaced_layer_indices = swap_qwen_attention_with_wayfinder(
            model,
            cfg=wf_cfg,
            layer_indices=full_attention_layer_indices,
        )
        _log(
            "Butterfly swap complete: "
            f"layers_replaced={len(replaced_layer_indices)} "
            f"indices={replaced_layer_indices} "
            "scope=qwen35_full_attention_layers "
            f"decode_backend={'stock' if wf_cfg.wayfinder_decode_backend == 'dense' else 'experimental'}"
        )
    else:
        _log(
            "Stock mode selected: using the default Qwen 3.5 attention path "
            "(Butterfly target scope remains qwen35_full_attention_layers)."
        )

    device_info = mx.device_info()
    recommended = int(device_info.get("max_recommended_working_set_size", device_info.get("memory_size", 0)))
    if float(args.memory_budget_gb) > 0.0:
        memory_budget_bytes = int(float(args.memory_budget_gb) * (1024**3))
    else:
        memory_budget_bytes = int(float(max(1, recommended)) * float(max(0.1, args.memory_budget_fraction)))
    memory_reserve_bytes = int(float(max(0.0, args.memory_reserve_gb)) * float(1024**3))
    prompt_cache_max_bytes = int(float(max(0.0, args.prompt_cache_max_bytes_gb)) * float(1024**3))
    kv_bytes_per_token_dense = _estimate_kv_bytes_per_token(model)
    kv_bytes_per_token_quantized = (
        _estimate_kv_bytes_per_token(
            model,
            kv_bits=int(kv_quantization.bits),
            kv_group_size=int(kv_quantization.group_size),
        )
        if kv_quantization.enabled
        else None
    )

    model_id = str(args.model_id).strip() or f"{Path(args.model_path).name}-{mode}"
    warmup_seq_lens = [] if bool(args.disable_startup_warmup) else [int(x) for x in args.warmup_seq_lens if int(x) > 0]

    state = ServerState(
        model=model,
        tokenizer=tokenizer,
        model_path=str(resolved_model_path),
        model_id=model_id,
        mode=mode,
        decode_backend=(
            ("stock" if decode_backend_arg == "dense" else "experimental")
            if mode == "butterfly"
            else "stock"
        ),
        lock=threading.Lock(),
        replaced_layer_indices=[int(x) for x in replaced_layer_indices],
        full_attention_layer_indices=[int(x) for x in full_attention_layer_indices],
        max_input_tokens=(int(args.max_input_tokens) if int(args.max_input_tokens) > 0 else None),
        max_total_tokens=(int(args.max_total_tokens) if int(args.max_total_tokens) > 0 else None),
        prefill_chunk_size=int(max(1, args.prefill_chunk_size)),
        query_chunk_size=int(max(1, args.query_chunk_size)),
        memory_budget_bytes=int(memory_budget_bytes),
        memory_reserve_bytes=int(memory_reserve_bytes),
        memory_estimate_multiplier=float(max(0.0, args.memory_estimate_multiplier)),
        kv_quantization=kv_quantization,
        kv_bytes_per_token_dense=int(kv_bytes_per_token_dense),
        kv_bytes_per_token_quantized=(
            None if kv_bytes_per_token_quantized is None else int(kv_bytes_per_token_quantized)
        ),
        prompt_cache_store=PromptCacheStore(
            max_entries=int(args.prompt_cache_max_entries),
            max_bytes=int(prompt_cache_max_bytes),
        ),
        warmup_seq_lens=warmup_seq_lens,
        warmup_status={},
    )

    _log(
        "Butterfly MLX server config: "
        f"model_id={state.model_id} mode={state.mode} "
        f"max_input_tokens={state.max_input_tokens} max_total_tokens={state.max_total_tokens} "
        f"prefill_chunk_size={state.prefill_chunk_size} "
        f"memory_budget_gb={_bytes_to_gib(state.memory_budget_bytes)} "
        f"memory_reserve_gb={_bytes_to_gib(state.memory_reserve_bytes)} "
        f"kv_bytes_per_token_dense={state.kv_bytes_per_token_dense} "
        f"kv_bytes_per_token_quantized={state.kv_bytes_per_token_quantized}"
    )
    _log(
        "KV quantization: "
        f"enabled={state.kv_quantization.enabled} bits={state.kv_quantization.bits} "
        f"group_size={state.kv_quantization.group_size} "
        f"quantized_kv_start={state.kv_quantization.quantized_kv_start} "
        "policy=post_prefill_before_decode preserve_dense_prefix_cache=True"
    )
    _log(
        "Prompt cache store: "
        f"entries={state.prompt_cache_store.max_entries} "
        f"max_bytes_gb={_bytes_to_gib(state.prompt_cache_store.max_bytes)}"
    )

    app = create_app(state)
    _log(
        f"Serving Butterfly MLX OpenAI endpoint on http://{args.host}:{args.port}/v1 "
        f"(model_id={state.model_id})"
    )
    _log("Endpoints: POST /v1/chat/completions  |  GET /health  |  GET /v1/models")
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level))


if __name__ == "__main__":
    main()
