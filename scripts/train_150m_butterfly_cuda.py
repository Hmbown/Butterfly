"""Train a ~150M-param Butterfly GPT from scratch on an RTX 3080.

Reads a packed mmap .bin (see :mod:`bna.data_hf`) plus its ``.meta.json``,
builds a ``GPTConfigTorch.tiny_150m`` model with the wayfinder-permute path,
and trains with bf16 autocast + AdamW + linear warmup.

Example:

    python scripts/train_150m_butterfly_cuda.py \\
        --data data/nemotron_cc_train.bin \\
        --max-steps 1000 --batch-size 4 --grad-accum 4

This script is a no-op on non-CUDA machines.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


def _load_meta(bin_path: str) -> dict:
    meta_path = bin_path + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)


def _warmup_lr(step: int, warmup: int, peak: float) -> float:
    if warmup <= 0 or step >= warmup:
        return peak
    return peak * float(step) / float(max(1, warmup))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/nemotron_cc_train.bin")
    parser.add_argument("--tokenizer", default=None, help="Override; default: infer from meta")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override; default: infer from tokenizer")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-out", default="checkpoints/butterfly_150m.pt")
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("This script requires CUDA; skipping.")
        return 0

    from bna.data_mmap import MemmapDataset
    from bna.torch.model import GPTConfigTorch, GPTTorch

    torch.manual_seed(args.seed)

    meta = _load_meta(args.data)
    dtype = meta.get("dtype", "uint32")
    data = MemmapDataset(args.data, dtype=dtype)
    print(f"Loaded {data.n_tokens:,} tokens from {args.data} (dtype={dtype})")

    if args.vocab_size is not None:
        vocab_size = int(args.vocab_size)
    else:
        tok_kind = args.tokenizer or ("qwen3" if meta.get("tokenizer_type") == "HFAutoTokenizer" else "gpt2")
        from bna.tokenizers import build_tokenizer

        tok = build_tokenizer(tok_kind)
        vocab_size = int(tok.vocab_size)
    print(f"vocab_size = {vocab_size:,}")

    cfg = GPTConfigTorch.tiny_150m(vocab_size=vocab_size)
    model = GPTTorch(cfg).to("cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built GPTTorch: {n_params/1e6:.1f}M params (attn={cfg.attn}, seq_len={cfg.seq_len})")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)) or ".", exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    seq_len = cfg.seq_len
    t0 = time.perf_counter()
    last = t0
    running_loss = 0.0
    running_count = 0

    def save_checkpoint(step: int) -> None:
        payload = {
            "model": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "step": int(step),
            "meta": meta,
        }
        torch.save(payload, args.ckpt_out)
        print(f"  saved checkpoint -> {args.ckpt_out} (step={step})")

    for step in range(1, args.max_steps + 1):
        lr = _warmup_lr(step, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _micro in range(args.grad_accum):
            x, y = data.get_batch(args.batch_size, seq_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, y)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            step_loss += float(loss.item())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        running_loss += step_loss
        running_count += 1

        if step == 1 or step % args.log_every == 0:
            now = time.perf_counter()
            dt = max(1e-6, now - last)
            tokens = args.batch_size * seq_len * args.grad_accum * args.log_every
            toks_per_s = tokens / dt if step != 1 else (args.batch_size * seq_len * args.grad_accum) / dt
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            avg = running_loss / max(1, running_count)
            print(
                f"step {step:5d} | loss {avg:.4f} | ppl {math.exp(min(avg, 20)):.2f} "
                f"| lr {lr:.2e} | {toks_per_s:,.0f} tok/s | peak {peak_gb:.2f} GB"
            )
            last = now
            running_loss = 0.0
            running_count = 0

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            save_checkpoint(step)

    save_checkpoint(args.max_steps)
    total_dt = time.perf_counter() - t0
    print(f"done in {total_dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
