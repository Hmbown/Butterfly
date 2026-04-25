"""Quality smoke for compressed_butterfly vs stock attention on Qwen 3.5 0.8B.

Loads the model twice (sequentially, never both in memory at once), once for
each mode, runs prefill on a deterministic synthetic prompt, then greedy-
decodes a short suffix. Compares:

* prefix logit divergence (KL of next-token distributions at end of prefill)
* top-1 agreement on the next token after prefill
* exact-match rate of the greedy-decoded suffix
* token-by-token agreement of the greedy suffix

The smoke is descriptive: it does not assert pass/fail. Outputs JSON with
the metrics and the decoded text from both modes for human inspection.

Usage::

    python scripts/quality_smoke.py \\
        --model-path /path/to/qwen3.5-0.8b-4bit \\
        --seq-len 8192 --decode-len 32 \\
        --out-path results/benchmarks/qwen35_0p8b_mlx/quality_smoke_8192/result.json

Run with both modes is automatic; a single invocation produces one combined
JSON file.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np


def _log(msg: str) -> None:
    print(msg, flush=True)


def _build_synthetic_prompt(seq_len: int, seed: int = 42) -> str:
    """A deterministic synthetic prompt: simple repeated structured text.

    The model under both modes should produce near-identical continuations
    for a well-conditioned input. We use simple structured text so that
    correctly-behaving attention has a strong signal for the next token.
    """
    # A natural-sounding base passage; we'll repeat it to fill the context.
    base = (
        "The compressed attention mechanism keeps the most recent tokens in "
        "raw form and summarizes older blocks into a single mean-pooled "
        "representation. This allows long-context inference to use a much "
        "smaller key-value cache while preserving local detail. The "
        "deterministic routing ensures that the same set of historical "
        "blocks is attended to from each query position, regardless of "
        "content. This makes the architecture predictable and efficient.\n"
        "\n"
        "Question: which two properties make this architecture effective?\n"
        "Answer: 1) recent tokens are kept in raw form for local detail; "
        "2) older blocks are mean-pooled summaries routed by Butterfly. "
        "The combination yields a small KV cache and a fast forward pass.\n"
        "\n"
    )
    # Pad to roughly seq_len tokens by repetition. The tokenizer will trim
    # later; we just need enough text.
    rough_chars_per_token = 4.0
    target_chars = int(seq_len * rough_chars_per_token * 1.2)
    repeats = max(1, target_chars // max(1, len(base)) + 1)
    return base * repeats


def _run_one_mode(
    *,
    mode: str,
    model_path: str,
    seq_len: int,
    decode_len: int,
    chunk_size: int,
    block_size: int,
    local_window_tokens: int,
    query_chunk_size: int,
    block_partner_rule: str,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Load model, run prefill + greedy decode, return tokens + logits stats."""
    # Lazy imports so model load only happens when this function runs.
    from mlx_lm.utils import load
    from mlx_lm.models.cache import make_prompt_cache

    if mode == "compressed_butterfly":
        from bna.integrations.qwen_mlx import (
            QwenButterflyConfig,
            install_compressed_kv_caches,
            swap_qwen_attention_with_butterfly,
            get_qwen_full_attention_layer_indices,
        )

    _log(f"[{mode}] loading model {model_path}")
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    _log(f"[{mode}] loaded in {time.perf_counter() - t0:.1f}s")

    if mode == "compressed_butterfly":
        cfg = QwenButterflyConfig(
            path="block_sparse",
            window=int(local_window_tokens),
            num_cycles=1,
            edge_disjoint=True,
            enforce_hamiltonian=True,
            seed=0,
            edge_bias=True,
            block_size=int(block_size),
            block_local_window_blocks=1,
            block_partner_count=1,
            block_sink_blocks=1,
            block_partner_rule=str(block_partner_rule),
            block_compression="mean",
            compressed_local_window_tokens=int(local_window_tokens),
            query_chunk_size=int(query_chunk_size),
            compute_edge_utilization_proxy=False,
            compute_graph_metrics=False,
            wayfinder_decode_backend="dense",
        )
        layer_indices = get_qwen_full_attention_layer_indices(model)
        replaced = swap_qwen_attention_with_butterfly(
            model, cfg=cfg, layer_indices=layer_indices,
        )
        _log(f"[{mode}] swapped layers={replaced}")

    # Build prompt and tokenize.
    text = _build_synthetic_prompt(seq_len)
    ids = tokenizer.encode(text)
    # Trim or pad-truncate to seq_len.
    if len(ids) >= seq_len:
        prompt_ids = ids[:seq_len]
    else:
        # Repeat last few tokens to fill (only happens if base text is short).
        prompt_ids = ids + ids[-(seq_len - len(ids)):]
        prompt_ids = prompt_ids[:seq_len]
    prompt = mx.array([prompt_ids], dtype=mx.int32)

    # Cache.
    prealloc = ((seq_len + decode_len + chunk_size - 1) // chunk_size) * chunk_size
    cache = list(make_prompt_cache(model, max_kv_size=prealloc))
    if mode == "compressed_butterfly":
        install_compressed_kv_caches(
            model, cache,
            block_size=int(block_size),
            local_window_tokens=int(local_window_tokens),
            max_kv_size=prealloc,
            max_chunk_size=max(int(chunk_size), int(query_chunk_size)),
        )
        # Wire the eviction hook so we don't OOM on long prefill.
        from bna.integrations.qwen_mlx import _qwen_graph_cache_drop_other_keys
    else:
        _qwen_graph_cache_drop_other_keys = None

    # Chunked prefill, last chunk's logits captured.
    _log(f"[{mode}] prefill seq_len={seq_len} chunk_size={chunk_size}")
    t0 = time.perf_counter()
    final_logits = None
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        if _qwen_graph_cache_drop_other_keys is not None:
            _qwen_graph_cache_drop_other_keys(int(end))
        logits = model(prompt[:, start:end], cache=cache)
        mx.eval(logits)
        if end == seq_len:
            final_logits = logits[:, -1, :]
            mx.eval(final_logits)
    prefill_sec = time.perf_counter() - t0

    # Top-K next-token distribution at end of prefill.
    final_logits_np = np.asarray(final_logits.astype(mx.float32), dtype=np.float32).reshape(-1)
    top_idx = np.argsort(-final_logits_np)[: int(top_k)]
    top_logits = final_logits_np[top_idx]
    # Normalize over the full vocab for an honest distribution.
    log_softmax = final_logits_np - logsumexp(final_logits_np)

    # Greedy decode for decode_len steps.
    _log(f"[{mode}] greedy decode {decode_len} tokens")
    decoded: List[int] = []
    decoded_top_tokens: List[List[Tuple[int, float]]] = []
    next_id = int(top_idx[0])
    decoded.append(next_id)
    decoded_top_tokens.append([(int(top_idx[i]), float(top_logits[i])) for i in range(min(8, top_k))])
    for _ in range(int(decode_len) - 1):
        cur = mx.array([[next_id]], dtype=mx.int32)
        if _qwen_graph_cache_drop_other_keys is not None:
            # Decode reuses the same T target; eviction effectively no-ops once
            # cache is fully grown.
            pass
        out = model(cur, cache=cache)
        mx.eval(out)
        out_np = np.asarray(out[:, -1, :].astype(mx.float32), dtype=np.float32).reshape(-1)
        idx = np.argsort(-out_np)[: int(top_k)]
        next_id = int(idx[0])
        decoded.append(next_id)
        decoded_top_tokens.append([(int(idx[i]), float(out_np[idx[i]])) for i in range(min(8, top_k))])

    decoded_text = tokenizer.decode(decoded)
    return {
        "mode": mode,
        "seq_len": int(seq_len),
        "decode_len": int(decode_len),
        "prefill_sec": float(prefill_sec),
        "prompt_token_count": int(len(prompt_ids)),
        "final_log_softmax_top_k": [
            {"token_id": int(top_idx[i]), "logit": float(top_logits[i]), "log_prob": float(log_softmax[int(top_idx[i])])}
            for i in range(int(top_k))
        ],
        "decoded_token_ids": decoded,
        "decoded_text": decoded_text,
        "decoded_top_8_per_step": decoded_top_tokens,
    }


def logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def _kl(p_log: Dict[int, float], q_log: Dict[int, float]) -> float:
    """KL(p || q) where both are dictionaries of token_id -> log_prob. Tokens
    only in p contribute; tokens only in q are ignored. Returns NaN if no
    overlap."""
    overlap = 0
    kl = 0.0
    for tok, lp in p_log.items():
        if tok in q_log:
            kl += math.exp(lp) * (lp - q_log[tok])
            overlap += 1
    return kl if overlap > 0 else float("nan")


def _compare(stock: Dict[str, Any], compressed: Dict[str, Any]) -> Dict[str, Any]:
    p = {e["token_id"]: e["log_prob"] for e in stock["final_log_softmax_top_k"]}
    q = {e["token_id"]: e["log_prob"] for e in compressed["final_log_softmax_top_k"]}
    kl_pq = _kl(p, q)
    kl_qp = _kl(q, p)
    overlap = len(set(p) & set(q))
    top1_agree = (stock["final_log_softmax_top_k"][0]["token_id"]
                  == compressed["final_log_softmax_top_k"][0]["token_id"])
    decoded_agree = sum(
        1 for a, b in zip(stock["decoded_token_ids"], compressed["decoded_token_ids"]) if a == b
    )
    return {
        "top_k_overlap": int(overlap),
        "top_k_total": len(p),
        "kl_stock_compressed": float(kl_pq),
        "kl_compressed_stock": float(kl_qp),
        "top1_agree": bool(top1_agree),
        "decoded_token_agreement": float(decoded_agree / max(1, len(stock["decoded_token_ids"]))),
        "decoded_token_agreement_count": int(decoded_agree),
        "decoded_token_total": int(len(stock["decoded_token_ids"])),
    }


def _cmd_run(args: argparse.Namespace) -> None:
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out = _run_one_mode(
        mode=str(args.mode),
        model_path=str(args.model_path),
        seq_len=int(args.seq_len),
        decode_len=int(args.decode_len),
        chunk_size=int(args.chunk_size),
        block_size=int(args.block_size),
        local_window_tokens=int(args.compressed_local_window_tokens),
        query_chunk_size=int(args.query_chunk_size),
        block_partner_rule=str(args.block_partner_rule),
    )
    out["config"] = {
        "seq_len": int(args.seq_len),
        "decode_len": int(args.decode_len),
        "chunk_size": int(args.chunk_size),
        "block_size": int(args.block_size),
        "compressed_local_window_tokens": int(args.compressed_local_window_tokens),
        "query_chunk_size": int(args.query_chunk_size),
        "block_partner_rule": str(args.block_partner_rule),
        "model_path": str(args.model_path),
    }
    with args.out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    _log(f"wrote {args.out_path}")


def _cmd_compare(args: argparse.Namespace) -> None:
    with args.stock_path.open() as fh:
        stock = json.load(fh)
    with args.compressed_path.open() as fh:
        compressed = json.load(fh)
    cmp = _compare(stock, compressed)
    out = {
        "stock_config": stock.get("config"),
        "compressed_config": compressed.get("config"),
        "compare": cmp,
        "stock_decoded_text": stock.get("decoded_text"),
        "compressed_decoded_text": compressed.get("decoded_text"),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    _log(f"wrote {args.out_path}")
    _log(json.dumps(cmp, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--mode", choices=["stock", "compressed_butterfly"], required=True)
    pr.add_argument("--model-path", required=True)
    pr.add_argument("--seq-len", type=int, default=8192)
    pr.add_argument("--decode-len", type=int, default=32)
    pr.add_argument("--chunk-size", type=int, default=384)
    pr.add_argument("--block-size", type=int, default=128)
    pr.add_argument("--compressed-local-window-tokens", type=int, default=64)
    pr.add_argument("--query-chunk-size", type=int, default=64)
    pr.add_argument("--block-partner-rule", type=str, default="causal_shift")
    pr.add_argument("--out-path", type=Path, required=True)
    pr.set_defaults(func=_cmd_run)

    pc = sub.add_parser("compare")
    pc.add_argument("--stock-path", type=Path, required=True)
    pc.add_argument("--compressed-path", type=Path, required=True)
    pc.add_argument("--out-path", type=Path, required=True)
    pc.set_defaults(func=_cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
