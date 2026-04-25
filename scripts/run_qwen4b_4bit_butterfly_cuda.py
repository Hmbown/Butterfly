#!/usr/bin/env python3
"""Run Qwen3-4B in NF4 4-bit on CUDA with Butterfly attention swapped in.

Inference-only. Requires a CUDA device (e.g. RTX 3080 10GB). On non-CUDA
hosts this script exits cleanly so it can still be imported in CI.

Usage:
    python scripts/run_qwen4b_4bit_butterfly_cuda.py \
        --model-id Qwen/Qwen3-4B \
        --prompt "Explain butterfly attention in three sentences." \
        --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument(
        "--prompt",
        default="Explain butterfly attention in three sentences.",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--no-butterfly",
        action="store_true",
        help="Disable butterfly swap; run stock NF4 Qwen.",
    )
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument(
        "--compressed",
        action="store_true",
        help="Use compressed Butterfly: exact local tokens plus one KV summary per routed block.",
    )
    p.add_argument("--compressed-local-window-tokens", type=int, default=128)
    p.add_argument("--local-window", type=int, default=2)
    p.add_argument(
        "--partner-rule",
        default="xor",
        choices=["xor", "bit_reversal", "benes", "causal_shift"],
    )
    p.add_argument("--sink-count", type=int, default=1)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def _build_butterfly_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "path": "block_sparse",
        "strategy": "random",
        "block_size": int(args.block_size),
        "block_local_window_blocks": int(args.local_window),
        "block_partner_rule": args.partner_rule,
        "block_sink_blocks": int(args.sink_count),
        "block_compression": "mean" if bool(args.compressed) else "none",
        "compressed_local_window_tokens": int(args.compressed_local_window_tokens),
    }


def main() -> int:
    args = _parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed; cannot run.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("This script requires CUDA; skipping.")
        return 0

    from bna.integrations.qwen_torch import load_qwen_4bit_cuda

    butterfly_cfg = _build_butterfly_config(args)
    swap = not args.no_butterfly

    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_qwen_4bit_cuda(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.dtype,
        attn_implementation="eager",
        swap_butterfly=swap,
        butterfly_config=butterfly_cfg if swap else None,
    )

    print(f"model_id           = {args.model_id}")
    print(f"dtype              = {args.dtype}")
    print(f"butterfly_active   = {swap}")
    if swap:
        print(
            "butterfly_layout   = "
            f"block_size={butterfly_cfg['block_size']}, "
            f"partner_rule={butterfly_cfg['block_partner_rule']}, "
            f"local_window={butterfly_cfg['block_local_window_blocks']}, "
            f"sink_count={butterfly_cfg['block_sink_blocks']}, "
            f"compression={butterfly_cfg['block_compression']}"
        )

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device="cuda"
    )
    input_len = int(input_ids.shape[1])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = int(out.shape[1]) - input_len
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"peak_cuda_mem_gb   = {peak_gb:.3f}")
    print(f"latency_s          = {elapsed:.3f}")
    print(f"new_tokens         = {new_tokens}")
    print(f"tokens_per_sec     = {tokens_per_sec:.2f}")
    print("--- generation ---")
    print(tokenizer.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
