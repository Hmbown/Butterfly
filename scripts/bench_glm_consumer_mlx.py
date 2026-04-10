#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import numpy as np
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.glm_mlx import (
    GLMButterflyAttention,
    GLMButterflyConfig,
    swap_glm_attention_with_butterfly,
)


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


def _clear_workspace() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _iter_model_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unable to locate model layers for GLM benchmark.")


def _collect_hsa_layer_snapshot(
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
            if isinstance(getattr(layer, "self_attn", None), GLMButterflyAttention)
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
        if not isinstance(attn, GLMButterflyAttention):
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
                "attention_ms": profile.get("attention_ms"),
                "total_ms": profile.get("total_ms"),
                "notes": notes_subset,
            }
        )
    return out


def _summarize_hsa_trace(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    path_counts: Counter[str] = Counter()
    cache_source_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    graph_seq_len_counts: Counter[str] = Counter()
    q_len_counts: Counter[str] = Counter()
    dense_reason_counts: Counter[str] = Counter()

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
    timing_acc: Dict[str, Dict[str, float]] = {
        "overall": {
            "graph_build_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "observations": 0.0,
        },
        "prefill": {
            "graph_build_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "observations": 0.0,
        },
        "decode": {
            "graph_build_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "observations": 0.0,
        },
        "unknown": {
            "graph_build_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "observations": 0.0,
        },
    }

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
            path = str(layer_row.get("path") or "unknown")
            path_counts[path] += 1
            notes = layer_row.get("notes") or {}
            is_dense_fallback = False
            graph_ms = float(layer_row.get("graph_build_ms") or 0.0)
            attention_ms = float(layer_row.get("attention_ms") or 0.0)
            total_ms = float(layer_row.get("total_ms") or 0.0)
            phase_key = phase if phase in {"prefill", "decode"} else "unknown"
            for bucket_name in ("overall", phase_key):
                bucket = timing_acc[bucket_name]
                bucket["graph_build_ms"] += graph_ms
                bucket["attention_ms"] += attention_ms
                bucket["total_ms"] += total_ms
                bucket["observations"] += 1.0

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

            dense_reason = notes.get("dense_fallback_reason")
            dense_reason_s = "" if dense_reason is None else str(dense_reason).strip()
            if dense_reason_s and dense_reason_s.lower() not in {"none", "null"}:
                dense_reason_counts[dense_reason_s] += 1
                is_dense_fallback = True
            else:
                if bool(notes.get("active_dense_triggered")):
                    dense_reason_counts["active_dense_threshold"] += 1
                    is_dense_fallback = True
                if bool(notes.get("active_large_q_dense_triggered")):
                    dense_reason_counts["active_large_q"] += 1
                    is_dense_fallback = True
                if "dense_fallback" in path and not is_dense_fallback:
                    dense_reason_counts["unspecified"] += 1
                    is_dense_fallback = True

            if phase == "decode":
                decode_layer_obs += 1
            elif phase == "prefill":
                prefill_layer_obs += 1
            if is_dense_fallback:
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

    def _finalize_timing_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
        graph_ms = float(bucket.get("graph_build_ms", 0.0))
        attention_ms = float(bucket.get("attention_ms", 0.0))
        total_profile_ms = float(bucket.get("total_ms", 0.0))
        denom = graph_ms + attention_ms
        return {
            "graph_build_ms": graph_ms,
            "attention_ms": attention_ms,
            "total_profile_ms": total_profile_ms,
            "graph_vs_attention_total_ms": float(denom),
            "graph_build_share_of_graph_plus_attention": (
                float(graph_ms / denom) if denom > 0.0 else None
            ),
            "attention_share_of_graph_plus_attention": (
                float(attention_ms / denom) if denom > 0.0 else None
            ),
            "layer_observations_with_timing": int(round(bucket.get("observations", 0.0))),
        }

    timing_ms_by_phase = {
        "overall": _finalize_timing_bucket(timing_acc["overall"]),
        "prefill": _finalize_timing_bucket(timing_acc["prefill"]),
        "decode": _finalize_timing_bucket(timing_acc["decode"]),
    }

    return {
        "sample_count": int(len(samples)),
        "layer_observations": int(total_obs),
        "phase_counts": dict(phase_counts),
        "path_counts": dict(path_counts),
        "cache_source_counts": dict(cache_source_counts),
        "graph_seq_len_counts": dict(graph_seq_len_counts),
        "q_len_counts": dict(q_len_counts),
        "dense_fallback_reason_counts": dict(dense_reason_counts),
        "fallback_layer_observations": int(fallback_layer_obs),
        "decode_layer_observations": int(decode_layer_obs),
        "decode_fallback_layer_observations": int(decode_fallback_layer_obs),
        "prefill_layer_observations": int(prefill_layer_obs),
        "prefill_fallback_layer_observations": int(prefill_fallback_layer_obs),
        "decode_step_observations": int(decode_step_obs),
        "decode_fallback_steps": int(decode_fallback_steps),
        "prefill_step_observations": int(prefill_step_obs),
        "prefill_fallback_steps": int(prefill_fallback_steps),
        "dense_fallback_share_run": (float(fallback_layer_obs / total_obs) if total_obs else None),
        "dense_fallback_share_decode_layers": (
            float(decode_fallback_layer_obs / decode_layer_obs) if decode_layer_obs else None
        ),
        "dense_fallback_share_prefill_layers": (
            float(prefill_fallback_layer_obs / prefill_layer_obs) if prefill_layer_obs else None
        ),
        "dense_fallback_share_decode_steps": (
            float(decode_fallback_steps / decode_step_obs) if decode_step_obs else None
        ),
        "dense_fallback_share_prefill_steps": (
            float(prefill_fallback_steps / prefill_step_obs) if prefill_step_obs else None
        ),
        "timing_ms_by_phase": timing_ms_by_phase,
        "fallback_share_known": bool(total_obs > 0),
        "active_query_ratio": (float(active_query_obs / total_obs) if total_obs else 0.0),
        "adaptive_graph_reuse_ratio": (float(adaptive_reuse_obs / total_obs) if total_obs else 0.0),
        "cache_hit_ratio": (float(cache_hit_obs / total_obs) if total_obs else 0.0),
    }


def _expected_primary_path_for_mode(mode: str) -> str:
    mode_s = str(mode).strip().lower()
    if mode_s in {"wayfinder", "butterfly"}:
        return "permute"
    if mode_s == "sparse":
        return "sparse"
    return "dense"


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
    cache = list(make_prompt_cache(model, max_kv_size=None))
    if kv_step > 0 and target_seq_len > 0:
        prealloc_size = target_seq_len
        if prealloc_size % kv_step != 0:
            prealloc_size = ((prealloc_size // kv_step) + 1) * kv_step
        _log(f"    kv_step={kv_step}, target prealloc={prealloc_size} tokens")
    return cache


def _run_chunked_prefill(
    model: Any,
    *,
    prompt_tokens: mx.array,
    chunk_size: int,
    cache: Sequence[Any],
    hsa_trace_samples: Optional[List[Dict[str, Any]]] = None,
    hsa_trace_layer_indices: Optional[Sequence[int]] = None,
    hsa_trace_max_layers: int = 0,
    observability_default_path: Optional[str] = None,
) -> Dict[str, Any]:
    batch = int(prompt_tokens.shape[0])
    seq_len = int(prompt_tokens.shape[1])

    t0 = time.perf_counter()
    chunk_idx = 0
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        chunk_idx += 1
        logits = model(prompt_tokens[:, start:end], cache=cache)
        mx.eval(logits)
        if hsa_trace_samples is not None:
            layer_rows = _collect_hsa_layer_snapshot(
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
            layer_rows = _collect_hsa_layer_snapshot(
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


def _build_prompt_tokens(tokenizer: Any, target_len: int, *, seed_text: str) -> List[int]:
    # Build deterministic chat-like prompt text and trim to target token length.
    text = seed_text
    while True:
        ids = _encode_text(tokenizer, text)
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
        warm_tokens = _build_prompt_tokens(
            tokenizer,
            max(256, min(2048, seq_len)),
            seed_text="User: Warmup chat prompt.\nAssistant:",
        )
        cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=seq_len + decode_len)
        _reset_peak_memory()
        _run_chunked_prefill(
            model,
            prompt_tokens=mx.array([warm_tokens], dtype=mx.int32),
            chunk_size=chunk_size,
            cache=cache,
        )
        _run_decode(model, batch=1, decode_len=min(16, decode_len), cache=cache)
        _clear_workspace()

        for r in range(1, repeats + 1):
            prompt_tokens = _build_prompt_tokens(
                tokenizer,
                seq_len,
                seed_text=(
                    "System: You are a helpful assistant in a long-running session.\n"
                    "User: Please read all previous notes and answer precisely with concise output.\nAssistant:"
                ),
            )
            cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=seq_len + decode_len)
            _reset_peak_memory()
            hsa_trace_samples: List[Dict[str, Any]] = []
            pre = _run_chunked_prefill(
                model,
                prompt_tokens=mx.array([prompt_tokens], dtype=mx.int32),
                chunk_size=chunk_size,
                cache=cache,
                hsa_trace_samples=hsa_trace_samples,
                hsa_trace_layer_indices=hsa_trace_layer_indices,
                hsa_trace_max_layers=hsa_trace_max_layers,
                observability_default_path=expected_primary_path,
            )
            dec = _run_decode(
                model,
                batch=1,
                decode_len=decode_len,
                cache=cache,
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
                "dense_fallback_reason_counts": dict(
                    hsa_summary.get("dense_fallback_reason_counts", {})
                ),
                "dense_fallback_share_run": hsa_summary.get("dense_fallback_share_run"),
                "dense_fallback_share_prefill_steps": hsa_summary.get(
                    "dense_fallback_share_prefill_steps"
                ),
                "dense_fallback_share_decode_steps": hsa_summary.get(
                    "dense_fallback_share_decode_steps"
                ),
                "timing_ms_by_phase": hsa_summary.get("timing_ms_by_phase"),
                "observability_fallback_share_known": bool(hsa_summary.get("fallback_share_known")),
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
    on_turn: Any = None,
) -> Dict[str, Any]:
    per_turn: List[Dict[str, Any]] = []
    history = "System: You are a concise assistant.\n"
    total_e2e = 0.0
    for t in range(1, turns + 1):
        target_len = max(1024, int(target_total_context * t / turns))
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
        legacy_mode = "dense"
    elif path_arg == "sparse":
        legacy_mode = "sparse"
    elif path_arg == "permute":
        legacy_mode = "butterfly"

    if mode_arg is None:
        return legacy_mode or "butterfly"
    if mode_arg == "wayfinder":
        mode_arg = "butterfly"
    if mode_arg not in {"dense", "butterfly", "sparse"}:
        parser.error(
            f"--mode must be one of ['dense', 'butterfly', 'sparse'] plus legacy alias ['wayfinder']; got {mode_arg!r}"
        )
    if legacy_mode is not None and legacy_mode != mode_arg:
        parser.error(
            f"Conflicting mode selectors: --mode {mode_arg} vs legacy selector {legacy_mode}. "
            "Use only --mode in primary flows."
        )
    return mode_arg


def main() -> None:
    p = argparse.ArgumentParser(
        description="GLM consumer-like benchmark (single-turn/multi-turn/quality)"
    )
    p.add_argument("--model-path", type=str, default="mlx-community/GLM-4.7-Flash-4bit")
    p.add_argument(
        "--mode",
        type=str,
        choices=["dense", "butterfly", "wayfinder", "sparse"],
        default=None,
        help="Primary attention mode selector for benchmark UX.",
    )
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    p.add_argument("--decode-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--multi-decode-len", type=int, default=128)
    p.add_argument("--multi-target-context", type=int, default=65536)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--kv-step", type=int, default=4096)
    p.add_argument("--cooldown-sec", type=float, default=60.0)
    p.add_argument("--path", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=0)
    p.add_argument("--head-chunk-size", type=int, default=2)
    p.add_argument("--query-chunk-size", type=int, default=384)
    p.add_argument(
        "--debug-active-dense-threshold",
        type=int,
        default=0,
        help="Debug-only: force dense fallback in active decode when K_len <= threshold (<=0 disables).",
    )
    p.add_argument(
        "--debug-disable-discovered-active-row-kernel",
        action="store_true",
        default=False,
        help="Debug-only: disable discovered active-row kernel route.",
    )
    p.add_argument(
        "--debug-disable-fused-dispatch",
        action="store_true",
        default=False,
        help="Debug-only: disable fused permute dispatch.",
    )
    p.add_argument(
        "--debug-enable-decode-local-tail-fastpath",
        action="store_true",
        default=False,
        help="Debug-only: re-enable dense-like decode local-tail shortcut.",
    )
    p.add_argument(
        "--debug-butterfly-decode-backend",
        dest="debug_butterfly_decode_backend",
        type=str,
        default="dense",
        choices=["active_permute", "dense"],
        help="Debug-only: Butterfly decode backend policy (default: dense).",
    )
    p.add_argument(
        "--debug-wayfinder-decode-backend",
        dest="debug_butterfly_decode_backend",
        type=str,
        default=argparse.SUPPRESS,
        choices=["active_permute", "dense"],
        help=argparse.SUPPRESS,
    )
    p.add_argument("--active-dense-threshold", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument(
        "--disable-discovered-active-row-kernel",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--disable-fused-dispatch", action="store_true", default=False, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--disable-decode-local-tail-fastpath",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
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
        default=Path(
            "benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_v1.json"
        ),
    )
    p.add_argument("--quality-task-id-filter", type=str, default="")
    p.add_argument("--trace-quality-task-id", type=str, default="")
    p.add_argument("--trace-topk", type=int, default=5)
    p.add_argument("--trace-max-steps", type=int, default=0)
    p.add_argument(
        "--hsa-trace",
        action="store_true",
        help="Record per-step Butterfly attention path snapshots.",
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
    debug_active_dense_threshold = int(args.debug_active_dense_threshold)
    if int(args.active_dense_threshold) > 0:
        debug_active_dense_threshold = int(args.active_dense_threshold)
    debug_disable_discovered_active_row_kernel = bool(
        args.debug_disable_discovered_active_row_kernel
    ) or bool(args.disable_discovered_active_row_kernel)
    debug_disable_fused_dispatch = bool(args.debug_disable_fused_dispatch) or bool(
        args.disable_fused_dispatch
    )
    debug_enable_decode_local_tail_fastpath = bool(args.debug_enable_decode_local_tail_fastpath)
    if bool(args.disable_decode_local_tail_fastpath):
        debug_enable_decode_local_tail_fastpath = False
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

    _log(f"Loading model {args.model_path}")
    model, tokenizer, _ = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
        model_config={"trust_remote_code": True},
    )
    _log("Model loaded")

    dense_mode = mode == "dense"
    butterfly_cfg: Optional[GLMButterflyConfig] = None
    butterfly_decode_backend_public = "stock"
    if not dense_mode:
        butterfly_cfg = GLMButterflyConfig(
            path=mode,  # type: ignore[arg-type]
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
            permute_prepermute_mode="auto",
            permute_log_chunks=False,
            compute_edge_utilization_proxy=False,
            compute_graph_metrics=False,
            retro_backfill_enabled=False,
            retro_backfill_alpha=0.0,
            retro_backfill_training_only=True,
            retro_backfill_causal_only=True,
            permute_memory_budget_bytes=None,
            active_dense_threshold=(
                None if debug_active_dense_threshold <= 0 else int(debug_active_dense_threshold)
            ),
            use_discovered_active_row_kernel=not bool(debug_disable_discovered_active_row_kernel),
            use_fused_dispatch=not bool(debug_disable_fused_dispatch),
            enable_decode_local_tail_fastpath=bool(debug_enable_decode_local_tail_fastpath),
            wayfinder_decode_backend=str(args.debug_butterfly_decode_backend),
        )
        butterfly_decode_backend_public = (
            "stock"
            if str(args.debug_butterfly_decode_backend).strip().lower() == "dense"
            else "experimental"
        )
        if mode == "butterfly" and butterfly_cfg.enable_decode_local_tail_fastpath:
            _log("Debug override: decode local-tail fastpath enabled for butterfly mode.")
        if mode == "butterfly" and butterfly_cfg.active_dense_threshold is not None:
            _log(
                "Debug override: active_dense_threshold set in butterfly mode "
                f"({butterfly_cfg.active_dense_threshold})."
            )

    replaced_layers = 0
    replaced_indices: List[int] = []
    requested_layer_indices: Optional[List[int]] = None
    if dense_mode:
        _log("mode=dense: stock GLM attention baseline (no swap)")
    else:
        assert butterfly_cfg is not None
        layers = list(_iter_model_layers(model))
        requested_layer_indices = _resolve_swap_layers(
            total_layers=len(layers),
            swap_first_n_layers=int(debug_swap_first_n_layers),
            swap_last_n_layers=int(debug_swap_last_n_layers),
            swap_layer_indices=debug_swap_layer_indices,
        )
        replaced = swap_glm_attention_with_butterfly(
            model,
            cfg=butterfly_cfg,
            layer_indices=requested_layer_indices,
        )
        replaced_indices = [int(x) for x in replaced]
        replaced_layers = len(replaced_indices)
        for idx in replaced_indices:
            layer_attn = layers[idx].self_attn
            if isinstance(layer_attn, GLMButterflyAttention):
                layer_attn.compute_edge_utilization_proxy = False
                layer_attn.compute_graph_metrics = False
        scope = (
            "all-layers"
            if requested_layer_indices is None
            else f"selected-layers={requested_layer_indices}"
        )
        _log(f"mode={mode}: swapped attention on {replaced_layers} layers ({scope})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    butterfly_config_payload = (
        None
        if dense_mode or butterfly_cfg is None
        else {
            "path": str(butterfly_cfg.path),
            "strategy": str(butterfly_cfg.strategy),
            "window": int(butterfly_cfg.window),
            "landmark_stride": butterfly_cfg.landmark_stride,
            "num_cycles": butterfly_cfg.num_cycles,
            "edge_disjoint": bool(butterfly_cfg.edge_disjoint),
            "enforce_hamiltonian": bool(butterfly_cfg.enforce_hamiltonian),
            "regular_num_clusters": int(butterfly_cfg.regular_num_clusters),
            "seed": int(butterfly_cfg.seed),
            "edge_bias": bool(butterfly_cfg.edge_bias),
            "window_drop": float(butterfly_cfg.window_drop),
            "permute_head_chunk_size": int(butterfly_cfg.permute_head_chunk_size),
            "query_chunk_size": int(butterfly_cfg.query_chunk_size),
            "permute_prepermute_mode": str(butterfly_cfg.permute_prepermute_mode),
            "active_dense_threshold": butterfly_cfg.active_dense_threshold,
            "use_discovered_active_row_kernel": bool(
                butterfly_cfg.use_discovered_active_row_kernel
            ),
            "circular": bool(butterfly_cfg.circular),
            "multi_cycle_mode": str(butterfly_cfg.multi_cycle_mode),
            "use_fused_dispatch": bool(butterfly_cfg.use_fused_dispatch),
            "enable_decode_local_tail_fastpath": bool(
                butterfly_cfg.enable_decode_local_tail_fastpath
            ),
            "decode_backend": butterfly_decode_backend_public,
            "butterfly_decode_backend": butterfly_decode_backend_public,
        }
    )
    wayfinder_config_payload = (
        None
        if butterfly_config_payload is None
        else {
            **butterfly_config_payload,
            "wayfinder_decode_backend": butterfly_decode_backend_public,
        }
    )
    payload: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "mode": mode,
        "butterfly_decode_backend": butterfly_decode_backend_public,
        "wayfinder_decode_backend": butterfly_decode_backend_public,
        "no_swap": bool(dense_mode),
        "retro_backfill_enabled": False,
        "butterfly_config": butterfly_config_payload,
        "wayfinder_config": wayfinder_config_payload,
        "swap": {
            "replaced_layers": int(replaced_layers),
            "replaced_layer_indices": replaced_indices,
            "requested_layer_indices": requested_layer_indices,
            "debug_swap_first_n_layers": int(debug_swap_first_n_layers),
            "debug_swap_last_n_layers": int(debug_swap_last_n_layers),
            "debug_swap_layer_indices_arg": str(debug_swap_layer_indices),
            "debug_disable_discovered_active_row_kernel": bool(
                debug_disable_discovered_active_row_kernel
            ),
            "debug_disable_fused_dispatch": bool(debug_disable_fused_dispatch),
            "debug_enable_decode_local_tail_fastpath": bool(
                debug_enable_decode_local_tail_fastpath
            ),
            "debug_active_dense_threshold": (
                None if debug_active_dense_threshold <= 0 else int(debug_active_dense_threshold)
            ),
        },
        "observability": {
            "runtime_summary_always_on": True,
            "hsa_trace_samples_enabled": bool(args.hsa_trace),
            "hsa_trace_max_layers": int(args.hsa_trace_max_layers),
            "hsa_trace_max_steps": int(args.hsa_trace_max_steps),
            "hsa_trace_layer_indices": replaced_indices,
        },
        "single_turn": None,
        "multi_turn": None,
        "quality": None,
    }

    results_path = args.out_dir / "results.json"

    def _flush() -> None:
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _flush()

    if not args.skip_single_turn:
        payload["single_turn"] = []

        def _on_single_row(row: Dict[str, Any]) -> None:
            payload["single_turn"].append(dict(row))
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
            hsa_trace=bool(args.hsa_trace),
            hsa_trace_layer_indices=replaced_indices,
            hsa_trace_max_layers=int(args.hsa_trace_max_layers),
            hsa_trace_max_steps=int(args.hsa_trace_max_steps),
            on_row=_on_single_row,
        )

    if not args.skip_multi_turn:
        payload["multi_turn"] = _run_multi_turn(
            model,
            tokenizer,
            turns=int(args.turns),
            target_total_context=int(args.multi_target_context),
            decode_len=int(args.multi_decode_len),
            chunk_size=int(args.chunk_size),
            kv_step=int(args.kv_step),
            on_turn=lambda _turn: _flush(),
        )
        _flush()

    if not args.skip_quality:
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
            on_task=lambda _row, _c, _n: _flush(),
        )
        _flush()

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
