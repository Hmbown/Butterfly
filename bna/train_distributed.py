"""Distributed Data Parallel (DDP) training.

Usage:
    torchrun --nproc_per_node=2 -m hcsa.train_distributed --data data/tinyshakespeare.txt

This module wraps the standard training loop with:
- DistributedDataParallel
- Deterministic cycle seeds synchronized across ranks
- Proper gradient synchronization
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import build_datasets, get_batch, load_text
from .model import GPT, GPTConfig
from .tokenizers import build_tokenizer
from .train import estimate_loss, build_argparser, _cosine_lr
from .utils import (
    collect_run_metadata,
    ensure_dir,
    format_bytes,
    peak_memory_bytes,
    reset_peak_memory_stats,
    save_json,
    set_seed,
)


def _setup_ddp() -> tuple[int, int, torch.device]:
    """Initialize distributed process group.

    Returns (rank, world_size, device).
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def _cleanup_ddp() -> None:
    dist.destroy_process_group()


def run_distributed_training(args: argparse.Namespace) -> Path | None:
    """DDP training loop."""
    rank, world_size, device = _setup_ddp()
    is_master = rank == 0

    # Use the same seed across all ranks for deterministic cycle construction
    set_seed(args.seed)

    # Load data (all ranks load the same data)
    text = load_text(args.data)
    tok = build_tokenizer(args.tokenizer, text_for_char_vocab=text if args.tokenizer == "char" else None)
    data = build_datasets(text, tok)

    landmark_stride = None if args.landmark_stride <= 0 else int(args.landmark_stride)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        n_layers=args.layers,
        n_heads=args.heads,
        n_embd=args.embd,
        dropout=args.dropout,
        attn=args.attn,
        cycle=args.cycle,
        window=args.window,
        landmark_stride=landmark_stride,
        num_cycles=args.num_cycles,
        seed=args.seed,  # Same seed for all ranks = same cycles
    )

    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[device])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    use_amp = getattr(args, "amp", False) and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else None
    scaler = torch.amp.GradScaler(enabled=use_amp)

    warmup_steps = getattr(args, "warmup_steps", 0)
    lr_min = getattr(args, "lr_min", 1e-5)
    use_lr_schedule = warmup_steps > 0

    run_dir = None
    if is_master:
        run_root = ensure_dir(args.outdir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = args.run_name.strip() or f"{stamp}_{args.attn}_ddp{world_size}"
        run_dir = ensure_dir(run_root / name)
        save_json(run_dir / "config.json", {
            "args": vars(args),
            "model": asdict(cfg),
            "metadata": asdict(collect_run_metadata()),
            "tokenizer": tok.state_dict(),
            "world_size": world_size,
        })

    reset_peak_memory_stats(device)
    model.train()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        if use_lr_schedule:
            lr = _cosine_lr(step, warmup_steps, args.steps, args.lr, lr_min)
            for pg in opt.param_groups:
                pg["lr"] = lr

        xb, yb = get_batch(data.train, args.batch_size, args.seq_len, device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(xb, yb)
                loss = out["loss"]
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(xb, yb)
            loss = out["loss"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        opt.zero_grad(set_to_none=True)

        if is_master and (step == 1 or step % 10 == 0):
            now = time.perf_counter()
            dt = now - t0
            tokens = args.batch_size * args.seq_len * world_size * step
            toks_per_s = tokens / max(dt, 1e-8)
            print(
                f"step {step:5d} | loss {loss.item():.4f}"
                f" | {toks_per_s:,.0f} tok/s (x{world_size} GPUs)"
                f" | peak {format_bytes(peak_memory_bytes(device))}"
            )

        if is_master and (step == args.steps or (args.eval_every > 0 and step % args.eval_every == 0)):
            raw_model = model.module
            val_loss = estimate_loss(
                raw_model, data.val, batch_size=args.batch_size,
                seq_len=args.seq_len, device=device,
                amp_dtype=amp_dtype,
            )
            print(f"eval step {step:5d} | val {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.2f})")

            if run_dir is not None:
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": asdict(cfg),
                    "tokenizer": tok.state_dict(),
                    "step": step,
                }, run_dir / "ckpt.pt")

    _cleanup_ddp()
    return run_dir


def main() -> None:
    p = build_argparser()
    args = p.parse_args()
    run_dir = run_distributed_training(args)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and run_dir:
        print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
