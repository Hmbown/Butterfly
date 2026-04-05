#!/usr/bin/env python3
"""Interactive chat with GLM or Qwen using Butterfly sparse attention.

Butterfly replaces attention only during PREFILL (where the 40% speedup comes
from). Decode uses standard dense SDPA — no quality trade-off.

Usage (if you already have the model downloaded):

    python3 scripts/chat_glm_wayfinder.py --model-path /path/to/GLM-4.7-Flash-4bit

Usage (downloads from HuggingFace on first run, ~4 GB):

    python3 scripts/chat_glm_wayfinder.py

Compare against stock (native model) baseline:

    python3 scripts/chat_glm_wayfinder.py --mode stock

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
from mlx_lm.sample_utils import make_sampler

from bna.integrations.glm_mlx import GLMWayfinderConfig, swap_glm_attention_with_wayfinder
from bna.integrations.qwen_mlx import QwenWayfinderConfig, swap_qwen_attention_with_wayfinder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive chat with GLM-4.7-Flash + Butterfly",
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
        choices=["wayfinder", "dense", "stock", "butterfly"],
        default="wayfinder",
        help="'butterfly'/'wayfinder' = BNA sparse prefill. 'stock'/'dense' = native model attention.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=512, help="Max new tokens per turn (default: 512)"
    )
    p.add_argument(
        "--window", type=int, default=64, help="Butterfly local window size (default: 64)"
    )
    p.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    p.add_argument("--temp", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    return p.parse_args()


def build_model(args: argparse.Namespace):
    print(f"\nLoading {args.model_path} ...", flush=True)
    result = load(args.model_path, return_config=True, tokenizer_config={"trust_remote_code": True})
    model, tokenizer, config = result[0], result[1], result[2]
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    is_qwen = "qwen" in model_type or "qwen" in str(args.model_path).lower()

    mode = args.mode
    if mode == "butterfly":
        mode = "wayfinder"
    if mode == "wayfinder":
        if is_qwen:
            cfg = QwenWayfinderConfig(
                path="permute",
                strategy="random",
                window=int(args.window),
                landmark_stride=None,
                num_cycles=1,
                edge_disjoint=True,
                enforce_hamiltonian=True,
                seed=42,
                edge_bias=True,
                window_drop=0.0,
                compute_edge_utilization_proxy=False,
                compute_graph_metrics=False,
                permute_head_chunk_size=2,
                query_chunk_size=384,
                use_fused_dispatch=True,
                wayfinder_decode_backend="dense",
                retro_backfill_enabled=False,
                retro_backfill_alpha=0.0,
                retro_backfill_training_only=True,
                retro_backfill_causal_only=True,
            )
            swap_qwen_attention_with_wayfinder(model, cfg=cfg)
        else:
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
        print(f"Butterfly active (window={args.window}, decode=dense).", flush=True)
    else:
        print("Dense attention active (baseline).", flush=True)

    model_label = Path(str(args.model_path)).name or str(args.model_path)
    return model, tokenizer, model_label


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


def chat(model, tokenizer, args: argparse.Namespace, model_label: str) -> None:
    history: List[Dict[str, str]] = []
    active_mode = "wayfinder" if args.mode in ("wayfinder", "butterfly") else "dense"  # "stock" aliases to "dense" internally
    mode_label = f"Butterfly window={args.window}" if active_mode == "wayfinder" else "Dense"

    print(f"\n{'─' * 56}")
    print(f"  {model_label}  |  {mode_label}")
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
        sampler = make_sampler(temp=float(args.temp), top_p=float(args.top_p))
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            text = chunk.text
            print(text, end="", flush=True)
            reply_parts.append(text)

        print()  # newline after streamed output
        history.append({"role": "assistant", "content": "".join(reply_parts).strip()})


def main() -> None:
    args = parse_args()
    model, tokenizer, model_label = build_model(args)
    chat(model, tokenizer, args, model_label)


if __name__ == "__main__":
    main()
