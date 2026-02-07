#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _emit_setup_instructions(model_id: str) -> None:
    print(
        json.dumps(
            {
                "status": "missing_runtime",
                "message": "mlx-lm runtime not available; run setup commands below",
                "setup_commands": [
                    "pip install mlx-lm",
                    f"python -m mlx_lm.convert --hf-path {model_id} --mlx-path mlx_models/{model_id.split('/')[-1]}",
                    "python scripts/qwen3_mlx_attention_microbench.py --seq-len 2048 --batch 1 --heads 32 --embd 4096",
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-4B MLX scaffold benchmark")
    p.add_argument("--model-id", type=str, default="mlx-community/Qwen3-4B-Instruct-4bit")
    p.add_argument("--prompt", type=str, default="Wayfinder MLX microbench")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    try:
        from mlx_lm import generate, load
    except Exception:
        _emit_setup_instructions(args.model_id)
        return

    model, tokenizer = load(args.model_id)
    text = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)

    payload = {
        "status": "ok",
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "sample": text,
        "next_step": "Run scripts/qwen3_mlx_attention_microbench.py for controlled tensor-shape perf/memory.",
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
