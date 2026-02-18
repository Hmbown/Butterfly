#!/usr/bin/env python3
"""End-to-end correctness check: fused vs non-fused HCSA on real models.

Dense and HCSA use fundamentally different attention masks, so their outputs
WILL differ on a pretrained dense model. That's expected — sparse attention
approximates dense, it doesn't replicate it.

The real correctness question: does the new optimized fused dispatch path
produce the SAME output as the old per-head chunked path? Both compute the
same sparse attention pattern, just via different code paths.

Tests:
  1. Full-model forward: fused=True vs fused=False, same HCSA config
  2. Top-1 token agreement: should be ~100%
  3. Perplexity agreement: should be identical
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mae(a: mx.array, b: mx.array) -> float:
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    val = mx.mean(diff)
    mx.eval(val)
    return float(val.item())


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    val = mx.max(diff)
    mx.eval(val)
    return float(val.item())


def _cos_sim(a: mx.array, b: mx.array) -> float:
    a_f = a.astype(mx.float32).reshape(-1)
    b_f = b.astype(mx.float32).reshape(-1)
    dot = mx.sum(a_f * b_f)
    norm_a = mx.sqrt(mx.sum(a_f * a_f))
    norm_b = mx.sqrt(mx.sum(b_f * b_f))
    sim = dot / (norm_a * norm_b + 1e-12)
    mx.eval(sim)
    return float(sim.item())


def _has_nan(a: mx.array) -> bool:
    val = mx.any(mx.isnan(a))
    mx.eval(val)
    return bool(val.item())


def _report(name: str, a: mx.array, b: mx.array):
    mae = _mae(a, b)
    max_diff = _max_abs_diff(a, b)
    cos = _cos_sim(a, b)
    has_nan_a = _has_nan(a)
    has_nan_b = _has_nan(b)
    print(f"  {name}:")
    print(f"    MAE:          {mae:.6f}")
    print(f"    Max abs diff: {max_diff:.6f}")
    print(f"    Cosine sim:   {cos:.8f}")
    print(f"    NaN (fused):  {has_nan_a}  NaN (chunked): {has_nan_b}")
    return {"mae": mae, "max_diff": max_diff, "cos_sim": cos,
            "nan_a": has_nan_a, "nan_b": has_nan_b}


def _ppl(logits: mx.array, target_ids: mx.array) -> float:
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    log_probs = flat - mx.logsumexp(flat, axis=-1, keepdims=True)
    tgt_lp = mx.take_along_axis(
        log_probs, target_ids.reshape(-1, 1), axis=-1
    ).squeeze(-1)
    loss = -mx.mean(tgt_lp)
    mx.eval(loss)
    return math.exp(float(loss.item()))


# ---------------------------------------------------------------------------
# GPT-2: fused vs non-fused
# ---------------------------------------------------------------------------

def test_gpt2(seq_len: int = 512, window: int = 64, seed: int = 42):
    print("=" * 60)
    print(f"GPT-2: fused vs non-fused (T={seq_len}, W={window})")
    print("=" * 60)

    from mlx_lm.utils import load as mlx_load
    from hcsa.integrations.gpt2_mlx import (
        GPT2WayfinderConfig,
        swap_gpt2_attention_with_wayfinder,
    )

    model_path = "openai-community/gpt2"
    print(f"Loading {model_path}...")
    model, tokenizer = mlx_load(model_path, tokenizer_config={"trust_remote_code": True})
    mx.eval(model.parameters())

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning was the Word, and the Word was with God. "
        "To be or not to be, that is the question. "
    )
    tokens = tokenizer.encode(text)
    tokens = (tokens * ((seq_len + 1) // len(tokens) + 1))[:seq_len + 1]
    input_ids = mx.array(tokens[:seq_len]).reshape(1, -1)
    target_ids = mx.array(tokens[1:seq_len + 1]).reshape(1, -1)
    mx.eval(input_ids, target_ids)

    # --- Run with fused=True (new optimized path) ---
    print("\n[1] HCSA with fused dispatch (optimized path)...")
    wf_cfg_fused = GPT2WayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        permute_head_chunk_size=8,
        query_chunk_size=256,
        permute_prepermute_mode="auto",
        use_fused_dispatch=True,
    )
    swap_gpt2_attention_with_wayfinder(model, cfg=wf_cfg_fused)
    logits_fused = model(input_ids)
    mx.eval(logits_fused)
    print(f"  Shape: {logits_fused.shape}")

    # --- Run with fused=False (old per-head chunked path) ---
    print("\n[2] HCSA with chunked dispatch (reference path)...")
    wf_cfg_chunked = GPT2WayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        permute_head_chunk_size=1,
        query_chunk_size=256,
        permute_prepermute_mode="auto",
        use_fused_dispatch=False,
    )
    swap_gpt2_attention_with_wayfinder(model, cfg=wf_cfg_chunked)
    logits_chunked = model(input_ids)
    mx.eval(logits_chunked)
    print(f"  Shape: {logits_chunked.shape}")

    # --- Compare ---
    print("\n[3] Comparison (fused vs chunked):")
    r = _report("Full-model logits", logits_fused, logits_chunked)

    top1_fused = mx.argmax(logits_fused, axis=-1)
    top1_chunked = mx.argmax(logits_chunked, axis=-1)
    mx.eval(top1_fused, top1_chunked)
    agree = float(mx.mean((top1_fused == top1_chunked).astype(mx.float32)).item())
    print(f"    Top-1 agreement: {agree:.2%}")

    ppl_fused = _ppl(logits_fused, target_ids)
    ppl_chunked = _ppl(logits_chunked, target_ids)
    print(f"    Perplexity (fused):   {ppl_fused:.4f}")
    print(f"    Perplexity (chunked): {ppl_chunked:.4f}")

    print("\n--- Verdict ---")
    passed = True
    if r["nan_a"] or r["nan_b"]:
        print("  FAIL: NaN detected")
        passed = False
    if r["cos_sim"] < 0.999:
        print(f"  FAIL: Cosine similarity {r['cos_sim']:.6f} < 0.999")
        passed = False
    if agree < 0.95:
        print(f"  FAIL: Top-1 agreement {agree:.2%} < 95%")
        passed = False
    if r["max_diff"] > 1.0:
        print(f"  FAIL: Max abs diff {r['max_diff']:.4f} > 1.0")
        passed = False
    if passed:
        print(f"  PASS: cos={r['cos_sim']:.8f}, agree={agree:.2%}, "
              f"max_diff={r['max_diff']:.6f}")
    return passed


# ---------------------------------------------------------------------------
# Qwen3: fused vs non-fused
# ---------------------------------------------------------------------------

def test_qwen(seq_len: int = 512, window: int = 64, seed: int = 42):
    print("\n" + "=" * 60)
    print(f"Qwen3-1.7B-4bit: fused vs non-fused (T={seq_len}, W={window})")
    print("=" * 60)

    from mlx_lm.utils import load as mlx_load
    from hcsa.integrations.qwen_mlx import (
        QwenWayfinderConfig,
        swap_qwen_attention_with_wayfinder,
    )

    model_path = "mlx-community/Qwen3-1.7B-4bit"
    print(f"Loading {model_path}...")
    model, tokenizer = mlx_load(model_path, tokenizer_config={"trust_remote_code": True})
    mx.eval(model.parameters())

    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer.encode(text)
    tokens = (tokens * ((seq_len + 1) // len(tokens) + 1))[:seq_len + 1]
    input_ids = mx.array(tokens[:seq_len]).reshape(1, -1)
    target_ids = mx.array(tokens[1:seq_len + 1]).reshape(1, -1)
    mx.eval(input_ids, target_ids)

    # --- Fused ---
    print("\n[1] HCSA fused dispatch...")
    wf_cfg_fused = QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        edge_bias=False,
        window_drop=0.0,
        permute_head_chunk_size=8,
        query_chunk_size=256,
        use_fused_dispatch=True,
    )
    swap_qwen_attention_with_wayfinder(model, cfg=wf_cfg_fused)
    logits_fused = model(input_ids)
    mx.eval(logits_fused)

    # --- Chunked ---
    print("[2] HCSA chunked dispatch...")
    wf_cfg_chunked = QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        edge_bias=False,
        window_drop=0.0,
        permute_head_chunk_size=1,
        query_chunk_size=256,
        use_fused_dispatch=False,
    )
    swap_qwen_attention_with_wayfinder(model, cfg=wf_cfg_chunked)
    logits_chunked = model(input_ids)
    mx.eval(logits_chunked)

    # --- Compare ---
    print("[3] Comparison:")
    r = _report("Full-model logits", logits_fused, logits_chunked)

    top1_fused = mx.argmax(logits_fused, axis=-1)
    top1_chunked = mx.argmax(logits_chunked, axis=-1)
    mx.eval(top1_fused, top1_chunked)
    agree = float(mx.mean((top1_fused == top1_chunked).astype(mx.float32)).item())
    print(f"    Top-1 agreement: {agree:.2%}")

    ppl_fused = _ppl(logits_fused, target_ids)
    ppl_chunked = _ppl(logits_chunked, target_ids)
    print(f"    Perplexity (fused):   {ppl_fused:.4f}")
    print(f"    Perplexity (chunked): {ppl_chunked:.4f}")

    print("\n--- Verdict ---")
    passed = True
    if r["nan_a"] or r["nan_b"]:
        print("  FAIL: NaN detected")
        passed = False
    if r["cos_sim"] < 0.999:
        print(f"  FAIL: Cosine similarity {r['cos_sim']:.6f} < 0.999")
        passed = False
    if agree < 0.95:
        print(f"  FAIL: Top-1 agreement {agree:.2%} < 95%")
        passed = False
    if r["max_diff"] > 1.0:
        print(f"  FAIL: Max abs diff {r['max_diff']:.4f} > 1.0")
        passed = False
    if passed:
        print(f"  PASS: cos={r['cos_sim']:.8f}, agree={agree:.2%}, "
              f"max_diff={r['max_diff']:.6f}")
    return passed


# ---------------------------------------------------------------------------
# GLM: fused vs non-fused
# ---------------------------------------------------------------------------

def test_glm(seq_len: int = 2048, window: int = 64, seed: int = 42):
    print("\n" + "=" * 60)
    print(f"GLM-4.7-Flash: fused vs non-fused (T={seq_len}, W={window})")
    print("=" * 60)

    from mlx_lm.utils import load as mlx_load
    from hcsa.integrations.glm_mlx import (
        GLMWayfinderConfig,
        swap_glm_attention_with_wayfinder,
    )

    model_path = "mlx-community/GLM-4.7-Flash-4bit"
    print(f"Loading {model_path}...")
    model, tokenizer = mlx_load(model_path, tokenizer_config={"trust_remote_code": True})
    mx.eval(model.parameters())

    text = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(text)
    tokens = (tokens * ((seq_len + 1) // len(tokens) + 1))[:seq_len + 1]
    input_ids = mx.array(tokens[:seq_len]).reshape(1, -1)
    target_ids = mx.array(tokens[1:seq_len + 1]).reshape(1, -1)
    mx.eval(input_ids, target_ids)

    # --- Fused ---
    print("\n[1] HCSA fused dispatch...")
    wf_cfg_fused = GLMWayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        edge_bias=False,
        window_drop=0.0,
        permute_head_chunk_size=8,
        query_chunk_size=256,
        use_fused_dispatch=True,
    )
    swap_glm_attention_with_wayfinder(model, cfg=wf_cfg_fused)
    logits_fused = model(input_ids)
    mx.eval(logits_fused)

    # --- Chunked ---
    print("[2] HCSA chunked dispatch...")
    wf_cfg_chunked = GLMWayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        edge_bias=False,
        window_drop=0.0,
        permute_head_chunk_size=1,
        query_chunk_size=256,
        use_fused_dispatch=False,
    )
    swap_glm_attention_with_wayfinder(model, cfg=wf_cfg_chunked)
    logits_chunked = model(input_ids)
    mx.eval(logits_chunked)

    # --- Compare ---
    print("[3] Comparison:")
    r = _report("Full-model logits", logits_fused, logits_chunked)

    top1_fused = mx.argmax(logits_fused, axis=-1)
    top1_chunked = mx.argmax(logits_chunked, axis=-1)
    mx.eval(top1_fused, top1_chunked)
    agree = float(mx.mean((top1_fused == top1_chunked).astype(mx.float32)).item())
    print(f"    Top-1 agreement: {agree:.2%}")

    ppl_fused = _ppl(logits_fused, target_ids)
    ppl_chunked = _ppl(logits_chunked, target_ids)
    print(f"    Perplexity (fused):   {ppl_fused:.4f}")
    print(f"    Perplexity (chunked): {ppl_chunked:.4f}")

    print("\n--- Verdict ---")
    passed = True
    if r["nan_a"] or r["nan_b"]:
        print("  FAIL: NaN detected")
        passed = False
    if r["cos_sim"] < 0.999:
        print(f"  FAIL: Cosine similarity {r['cos_sim']:.6f} < 0.999")
        passed = False
    if agree < 0.95:
        print(f"  FAIL: Top-1 agreement {agree:.2%} < 95%")
        passed = False
    if r["max_diff"] > 1.0:
        print(f"  FAIL: Max abs diff {r['max_diff']:.4f} > 1.0")
        passed = False
    if passed:
        print(f"  PASS: cos={r['cos_sim']:.8f}, agree={agree:.2%}, "
              f"max_diff={r['max_diff']:.6f}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="E2E correctness: fused vs non-fused HCSA")
    p.add_argument("--models", nargs="*", default=["gpt2", "qwen"],
                   choices=["gpt2", "qwen", "glm"],
                   help="Which models to test")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results = {}
    for m in args.models:
        if m == "gpt2":
            results["gpt2"] = test_gpt2(args.seq_len, args.window, args.seed)
        elif m == "qwen":
            results["qwen"] = test_qwen(args.seq_len, args.window, args.seed)
        elif m == "glm":
            results["glm"] = test_glm(
                seq_len=args.seq_len,
                window=args.window,
                seed=args.seed,
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll e2e checks passed.")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
