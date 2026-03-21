#!/usr/bin/env python3
"""Quick quality check: compare GPT-2 perplexity between dense and HCSA permute attention.

Runs a forward pass over a sample of text and computes cross-entropy loss (perplexity)
for both dense baseline and Wayfinder/HCSA permute path. Reports the delta to verify
that the lazy eval optimization does not degrade output quality.
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
import mlx.nn as nn
import numpy as np


def _load_gpt2(model_path: str):
    """Load GPT-2 model and tokenizer via mlx_lm."""
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(model_path)
    return model, tokenizer


def _cross_entropy_loss(logits: mx.array, targets: mx.array) -> float:
    """Compute mean cross-entropy loss."""
    # logits: [B, T, V], targets: [B, T]
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    # gather log probs at target indices
    target_log_probs = mx.take_along_axis(
        log_probs, targets_flat[:, None], axis=-1
    ).squeeze(-1)
    loss = -mx.mean(target_log_probs)
    mx.eval(loss)
    return float(loss.item())


def _swap_to_wayfinder(model, *, window: int, seed: int):
    """Swap GPT-2 attention layers to Wayfinder/HCSA permute."""
    from bna.integrations.gpt2_mlx import GPT2WayfinderAttention, GPT2WayfinderConfig

    cfg = GPT2WayfinderConfig(
        path="permute",
        strategy="random",
        window=window,
        landmark_stride=window,
        num_cycles=1,
        seed=seed,
        permute_head_chunk_size=8,
        query_chunk_size=256,
        permute_prepermute_mode="auto",
    )

    for i, block in enumerate(model.model.h):
        orig_attn = block.attn
        wf_attn = GPT2WayfinderAttention(orig_attn, cfg)
        block.attn = wf_attn

    return model


def main():
    p = argparse.ArgumentParser(description="GPT-2 quality check: dense vs HCSA perplexity")
    p.add_argument("--model-path", type=str, default="openai-community/gpt2")
    p.add_argument("--seq-len", type=int, default=512, help="Sequence length for eval")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text", type=str, default=None, help="Custom text to evaluate on")
    args = p.parse_args()

    print(f"Loading model: {args.model_path}")
    model, tokenizer = _load_gpt2(args.model_path)

    # Prepare input text
    if args.text:
        text = args.text
    else:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "In the beginning was the Word, and the Word was with God, and the Word was God. "
            "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles. "
            "All happy families are alike; each unhappy family is unhappy in its own way. "
            "It was the best of times, it was the worst of times, it was the age of wisdom, "
            "it was the age of foolishness. Call me Ishmael. Some years ago, never mind how long "
            "precisely, having little or no money in my purse, and nothing particular to interest "
            "me on shore, I thought I would sail about a little and see the watery part of the world. "
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
            "must be in want of a wife. In a hole in the ground there lived a hobbit. Not a nasty, "
            "dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, "
            "sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that "
            "means comfort. Mr and Mrs Dursley of number four, Privet Drive, were proud to say that "
            "they were perfectly normal, thank you very much."
        )

    tokens = tokenizer.encode(text)
    if len(tokens) < args.seq_len + 1:
        # Repeat to fill
        tokens = (tokens * ((args.seq_len + 1) // len(tokens) + 1))
    tokens = tokens[: args.seq_len + 1]

    input_ids = mx.array(tokens[:-1]).reshape(1, -1)
    target_ids = mx.array(tokens[1:]).reshape(1, -1)
    mx.eval(input_ids, target_ids)

    T = input_ids.shape[1]
    print(f"Eval sequence length: {T}")

    # --- Dense forward pass ---
    print("\n--- Dense baseline ---")
    logits_dense = model(input_ids)
    mx.eval(logits_dense)
    loss_dense = _cross_entropy_loss(logits_dense, target_ids)
    ppl_dense = math.exp(loss_dense)
    print(f"  Loss: {loss_dense:.4f}")
    print(f"  Perplexity: {ppl_dense:.2f}")

    # --- Swap to Wayfinder ---
    print(f"\n--- HCSA permute (window={args.window}) ---")
    model = _swap_to_wayfinder(model, window=args.window, seed=args.seed)
    logits_wf = model(input_ids)
    mx.eval(logits_wf)
    loss_wf = _cross_entropy_loss(logits_wf, target_ids)
    ppl_wf = math.exp(loss_wf)
    print(f"  Loss: {loss_wf:.4f}")
    print(f"  Perplexity: {ppl_wf:.2f}")

    # --- Compare ---
    print("\n--- Comparison ---")
    logit_diff = mx.max(mx.abs(logits_dense - logits_wf))
    mx.eval(logit_diff)
    print(f"  Max logit abs diff: {float(logit_diff.item()):.6f}")
    print(f"  Loss delta: {loss_wf - loss_dense:+.4f}")
    print(f"  Perplexity delta: {ppl_wf - ppl_dense:+.2f}")
    ppl_pct = 100.0 * (ppl_wf - ppl_dense) / ppl_dense
    print(f"  Perplexity % change: {ppl_pct:+.2f}%")

    if abs(ppl_pct) < 5.0:
        print("\n  PASS: Perplexity within 5% of dense baseline.")
    else:
        print(f"\n  WARNING: Perplexity differs by {ppl_pct:+.2f}% from dense.")
        print("  This is expected for sparse attention — fewer edges means some information loss.")
        print("  Verify this is consistent with pre-optimization behavior.")


if __name__ == "__main__":
    main()
