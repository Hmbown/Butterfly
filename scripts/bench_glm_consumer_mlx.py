#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def _run_chunked_prefill(model: Any, *, prompt_tokens: mx.array, chunk_size: int, cache: Sequence[Any]) -> Dict[str, Any]:
    batch = int(prompt_tokens.shape[0])
    seq_len = int(prompt_tokens.shape[1])

    t0 = time.perf_counter()
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        logits = model(prompt_tokens[:, start:end], cache=cache)
        mx.eval(logits)
        _clear_workspace()

    prefill_sec = time.perf_counter() - t0
    total_tokens = batch * seq_len
    return {
        "prefill_sec": float(prefill_sec),
        "prefill_tok_s": float(total_tokens / max(prefill_sec, 1e-12)),
        "peak_memory_bytes": int(_peak_memory()),
    }


def _run_decode(model: Any, *, batch: int, decode_len: int, cache: Sequence[Any]) -> Dict[str, Any]:
    if decode_len <= 0:
        return {
            "decode_sec": 0.0,
            "decode_tok_s": None,
            "ttft_sec": None,
            "itl_p50_sec": None,
            "itl_p95_sec": None,
            "token_ids": [],
        }

    per_token_sec: List[float] = []
    token_ids: List[int] = []
    next_token = mx.zeros((batch, 1), dtype=mx.int32)
    t0 = time.perf_counter()
    for _ in range(int(decode_len)):
        t_step = time.perf_counter()
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1).astype(mx.int32)
        mx.eval(next_token)
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
    seq_lens: List[int],
    decode_len: int,
    repeats: int,
    chunk_size: int,
    kv_step: int,
    cooldown_sec: float,
    on_row: Any = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
            pre = _run_chunked_prefill(
                model,
                prompt_tokens=mx.array([prompt_tokens], dtype=mx.int32),
                chunk_size=chunk_size,
                cache=cache,
            )
            dec = _run_decode(model, batch=1, decode_len=decode_len, cache=cache)
            e2e = float(pre["prefill_sec"] + dec["decode_sec"])
            rows.append(
                {
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
                }
            )
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
    on_task: Any = None,
) -> Dict[str, Any]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    rows: List[Dict[str, Any]] = []
    correct = 0
    for task in tasks:
        prompt = str(task["prompt"])
        expected = str(task["expected"])
        prompt_ids = _encode_text(tokenizer, prompt)
        cache = _prepare_cache(model, kv_step=kv_step, target_seq_len=len(prompt_ids) + decode_len)
        _reset_peak_memory()
        _run_chunked_prefill(
            model,
            prompt_tokens=mx.array([prompt_ids], dtype=mx.int32),
            chunk_size=chunk_size,
            cache=cache,
        )
        dec = _run_decode(model, batch=1, decode_len=decode_len, cache=cache)
        out_text = _decode_tokens(tokenizer, dec["token_ids"])
        ok = _normalize_text(expected) in _normalize_text(out_text)
        if ok:
            correct += 1
        rows.append(
            {
                "id": task.get("id"),
                "expected": expected,
                "output": out_text,
                "correct": bool(ok),
                "ttft_sec": dec["ttft_sec"],
                "itl_p95_sec": dec["itl_p95_sec"],
                "peak_memory_bytes": int(_peak_memory()),
            }
        )
        if on_task is not None:
            on_task(rows[-1], int(correct), int(len(rows)))
    accuracy = float(correct / max(1, len(tasks)))
    return {
        "dataset_path": str(dataset_path),
        "num_tasks": int(len(tasks)),
        "correct": int(correct),
        "accuracy": accuracy,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GLM consumer-like benchmark (single-turn/multi-turn/quality)")
    p.add_argument("--model-path", type=str, default="mlx-community/GLM-4.7-Flash-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    p.add_argument("--decode-len", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--multi-decode-len", type=int, default=128)
    p.add_argument("--multi-target-context", type=int, default=65536)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--kv-step", type=int, default=4096)
    p.add_argument("--cooldown-sec", type=float, default=60.0)
    p.add_argument("--path", type=str, choices=["sparse", "permute"], default="permute")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=0)
    p.add_argument("--head-chunk-size", type=int, default=2)
    p.add_argument("--query-chunk-size", type=int, default=384)
    p.add_argument("--active-dense-threshold", type=int, default=49152)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-swap", action="store_true", default=False)
    p.add_argument("--quality-dataset", type=Path, default=Path("benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_v1.json"))
    p.add_argument("--skip-single-turn", action="store_true")
    p.add_argument("--skip-multi-turn", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

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
        permute_prepermute_mode="auto",
        permute_log_chunks=False,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
        permute_memory_budget_bytes=None,
        active_dense_threshold=(None if int(args.active_dense_threshold) <= 0 else int(args.active_dense_threshold)),
    )

    replaced_layers = 0
    if args.no_swap:
        _log("--no-swap: stock GLM attention for dense chunked control")
    else:
        layers = list(_iter_model_layers(model))
        replaced = swap_glm_attention_with_wayfinder(model, cfg=wf_cfg, layer_indices=None)
        replaced_layers = len(replaced)
        for idx in replaced:
            layer_attn = layers[idx].self_attn
            if isinstance(layer_attn, GLMWayfinderAttention):
                layer_attn.compute_edge_utilization_proxy = False
                layer_attn.compute_graph_metrics = False
        _log(f"Swapped Wayfinder attention on {replaced_layers} layers")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "no_swap": bool(args.no_swap),
        "retro_backfill_enabled": False,
        "wayfinder_config": None if args.no_swap else wf_cfg.__dict__,
        "swap": {"replaced_layers": int(replaced_layers)},
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
            seq_lens=[int(x) for x in args.seq_lens],
            decode_len=int(args.decode_len),
            repeats=int(args.repeats),
            chunk_size=int(args.chunk_size),
            kv_step=int(args.kv_step),
            cooldown_sec=float(args.cooldown_sec),
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
        payload["quality"] = _run_quality(
            model,
            tokenizer,
            dataset_path=args.quality_dataset,
            decode_len=min(64, int(args.decode_len)),
            chunk_size=int(args.chunk_size),
            kv_step=int(args.kv_step),
            on_task=lambda _row, _c, _n: _flush(),
        )
        _flush()

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
