#!/usr/bin/env python3
"""Interactive chat with GLM-4.7-Flash using Wayfinder sparse attention.

Wayfinder replaces attention only during PREFILL (where the 40% speedup comes
from). Decode uses standard dense SDPA — no quality trade-off.

Usage (if you already have the model downloaded):

    python3 scripts/chat_glm_wayfinder.py --model-path /path/to/GLM-4.7-Flash-4bit

Usage (downloads from HuggingFace on first run, ~4 GB):

    python3 scripts/chat_glm_wayfinder.py

Compare against dense baseline:

    python3 scripts/chat_glm_wayfinder.py --mode dense

Type 'quit' / 'exit' to stop.  Type 'reset' to clear conversation history.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_lm import load, stream_generate

from bna.integrations.glm_mlx import GLMWayfinderConfig, swap_glm_attention_with_wayfinder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive chat with GLM-4.7-Flash + Wayfinder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        default="mlx-community/GLM-4.7-Flash-4bit",
        metavar="PATH_OR_HF_ID",
        help=(
            "Local directory or HuggingFace repo ID. "
            "If you already ran `mlx_lm.convert` or downloaded the model, "
            "pass that path here to skip the download. "
            "(default: mlx-community/GLM-4.7-Flash-4bit)"
        ),
    )
    p.add_argument(
        "--mode",
        choices=["wayfinder", "dense"],
        default="wayfinder",
        help="'wayfinder' (sparse prefill + dense decode, default) or 'dense' (full baseline)",
    )
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per turn (default: 512)")
    p.add_argument("--window", type=int, default=64, help="Wayfinder local window size (default: 64)")
    p.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    p.add_argument("--temp", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    return p.parse_args()


def build_model(args: argparse.Namespace):
    print(f"\nLoading {args.model_path} ...", flush=True)
    result = load(args.model_path)
    model, tokenizer = result[0], result[1]

    if args.mode == "wayfinder":
        cfg = GLMWayfinderConfig(
            path="wayfinder",
            window=args.window,
            landmark_stride=64,
            num_cycles=1,
            strategy="random",
            # Decode always uses dense SDPA — this is the default and matches
            # what the benchmarks show: zero decode regression.
            wayfinder_decode_backend="dense",
        )
        swap_glm_attention_with_wayfinder(model, cfg=cfg)
        print(f"Wayfinder active (window={args.window}, decode=dense).", flush=True)
    else:
        print("Dense attention active (baseline).", flush=True)

    return model, tokenizer


def build_prompt(tokenizer, messages: List[Dict[str, str]], system: str) -> str:
    full = [{"role": "system", "content": system}] + messages
    try:
        return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Minimal fallback for tokenizers without a chat template
        lines = [f"[System]: {system}"]
        for m in messages:
            lines.append(f"[{m['role'].capitalize()}]: {m['content']}")
        lines.append("[Assistant]:")
        return "\n".join(lines)


def chat(model, tokenizer, args: argparse.Namespace) -> None:
    history: List[Dict[str, str]] = []
    mode_label = f"Wayfinder window={args.window}" if args.mode == "wayfinder" else "Dense"

    print(f"\n{'─' * 56}")
    print(f"  GLM-4.7-Flash  |  {mode_label}")
    print(f"  'reset' = clear history   'quit' = exit")
    print(f"{'─' * 56}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            history = []
            print("[History cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(tokenizer, history, args.system)

        print("Assistant: ", end="", flush=True)
        reply_parts: List[str] = []
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
        ):
            text = chunk.text
            print(text, end="", flush=True)
            reply_parts.append(text)

        print()  # newline after streamed output
        history.append({"role": "assistant", "content": "".join(reply_parts).strip()})


def main() -> None:
    args = parse_args()
    model, tokenizer = build_model(args)
    chat(model, tokenizer, args)


if __name__ == "__main__":
    main()
