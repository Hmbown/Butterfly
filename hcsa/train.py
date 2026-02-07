from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import trange

from .data import build_datasets, get_batch, load_text
from .model import GPT, GPTConfig
from .tokenizers import build_tokenizer
from .utils import (
    collect_run_metadata,
    auto_device,
    ensure_dir,
    format_bytes,
    peak_memory_bytes,
    reset_peak_memory_stats,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# LR scheduling helpers
# ---------------------------------------------------------------------------


def _cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    iters: int = 20,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(data, batch_size, seq_len, device)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(xb, yb)
        else:
            out = model(xb, yb)
        losses.append(out["loss"].item())
    return float(sum(losses) / len(losses))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a tiny GPT with dense or Hamiltonian-cycle sparse attention.")
    p.add_argument("--data", type=str, required=True, help="Path to a text file.")
    p.add_argument(
        "--hf-dataset",
        type=str,
        default="",
        help="Optional HuggingFace dataset name (requires datasets). If set, --data is ignored.",
    )
    p.add_argument(
        "--hf-config",
        type=str,
        default="",
        help="Optional HuggingFace dataset config name (e.g. wikitext-2-raw-v1).",
    )
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe", "gpt2"], help="Tokenizer kind.")
    p.add_argument("--attn", type=str, default="dense", choices=["dense", "hcsa"], help="Attention type.")
    p.add_argument("--cycle", type=str, default="random", choices=["random", "greedy", "online_insertion"], help="Cycle strategy for HCSA.")
    p.add_argument("--window", type=int, default=64, help="Local causal window size for HCSA.")
    p.add_argument("--landmark-stride", type=int, default=64, help="Landmark stride (<=0 disables landmarks).")
    p.add_argument("--num-cycles", type=int, default=1, help="Union of this many cycles (random/greedy).")

    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5, help="Minimum LR for cosine annealing.")
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup steps (0 = no warmup).")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation micro-batches.")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    p.add_argument("--seq-len-warmup", type=int, default=0,
                    help="Warmup seq_len from 64 to --seq-len over this many steps (0 = disabled).")

    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")

    p.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--outdir", type=str, default="runs", help="Base directory for runs")
    p.add_argument("--run-name", type=str, default="", help="Optional run name")
    return p


def run_training(args: argparse.Namespace) -> Path:
    device = auto_device(args.device)
    set_seed(args.seed)

    if args.hf_dataset:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise ImportError(
                "HuggingFace datasets support requires `datasets`. Install with: pip install -e .[hf]"
            ) from e

        ds = load_dataset(args.hf_dataset, args.hf_config or None)
        # Common fields: 'text'
        def _join(split: str) -> str:
            if split not in ds:
                raise ValueError(f"Dataset has no split {split!r}. Available: {list(ds.keys())}")
            ex = ds[split]
            if "text" not in ex.column_names:
                raise ValueError(f"Expected a 'text' column, got {ex.column_names}")
            return "\n".join(ex["text"])

        train_text = _join("train")
        val_text = _join("validation") if "validation" in ds else _join("test")
        text = train_text + "\n" + val_text
    else:
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
        seed=args.seed,
    )

    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP setup
    use_amp = getattr(args, "amp", False) and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if use_amp else None
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # Gradient accumulation
    grad_accum_steps = max(1, getattr(args, "grad_accum_steps", 1))

    # LR scheduling
    warmup_steps = getattr(args, "warmup_steps", 0)
    lr_min = getattr(args, "lr_min", 1e-5)
    use_lr_schedule = warmup_steps > 0 or lr_min < args.lr

    # Curriculum learning: sequence length warmup
    seq_len_warmup = getattr(args, "seq_len_warmup", 0)
    min_seq_len = 64

    run_root = ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.run_name.strip() or f"{stamp}_{args.attn}_{args.cycle if args.attn=='hcsa' else 'dense'}"
    run_dir = ensure_dir(run_root / name)

    meta = collect_run_metadata()
    save_json(run_dir / "config.json", {"args": vars(args), "model": asdict(cfg), "metadata": asdict(meta), "tokenizer": tok.state_dict()})

    reset_peak_memory_stats(device)

    model.train()
    t0 = time.perf_counter()
    last = t0

    use_tqdm = not bool(getattr(args, "no_tqdm", False))
    iterator = trange(1, args.steps + 1, desc=f"train[{device.type}]", disable=not use_tqdm)

    for step in iterator:
        # Curriculum: sequence length warmup
        if seq_len_warmup > 0 and step <= seq_len_warmup:
            frac = step / seq_len_warmup
            effective_seq_len = int(min_seq_len + (args.seq_len - min_seq_len) * frac)
            # Round to multiple of 8 for efficiency
            effective_seq_len = max(min_seq_len, (effective_seq_len // 8) * 8)
        else:
            effective_seq_len = args.seq_len

        # LR scheduling
        if use_lr_schedule:
            lr = _cosine_lr(step, warmup_steps, args.steps, args.lr, lr_min)
            for pg in opt.param_groups:
                pg["lr"] = lr

        # Gradient accumulation loop
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _micro in range(grad_accum_steps):
            xb, yb = get_batch(data.train, args.batch_size, effective_seq_len, device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(xb, yb)
                    loss = out["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(xb, yb)
                loss = out["loss"] / grad_accum_steps
                loss.backward()

            accum_loss += loss.item()

        if args.grad_clip > 0:
            if use_amp:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        # Logging
        if step == 1 or step % 10 == 0:
            now = time.perf_counter()
            dt = now - last
            tokens = args.batch_size * effective_seq_len * grad_accum_steps
            toks_per_s = tokens / max(1e-8, dt)
            mem = peak_memory_bytes(device)
            current_lr = opt.param_groups[0]["lr"]
            msg = (
                f"step {step:5d} | loss {accum_loss:.4f} | ppl {math.exp(min(accum_loss, 20)):.2f}"
                f" | lr {current_lr:.2e} | {toks_per_s:,.0f} tok/s | peak {format_bytes(mem)}"
            )
            (iterator.write if hasattr(iterator, "write") else print)(msg)
            last = now

        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = estimate_loss(
                model, data.train, batch_size=args.batch_size, seq_len=effective_seq_len,
                device=device, iters=min(args.eval_iters, 10), amp_dtype=amp_dtype,
            )
            val_loss = estimate_loss(
                model, data.val, batch_size=args.batch_size, seq_len=effective_seq_len,
                device=device, iters=args.eval_iters, amp_dtype=amp_dtype,
            )
            msg = (
                f"eval step {step:5d} | train {train_loss:.4f} (ppl {math.exp(min(train_loss, 20)):.2f})"
                f" | val {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.2f})"
            )
            (iterator.write if hasattr(iterator, "write") else print)(msg)
            model.train()

        if step == args.steps or (args.eval_every > 0 and step % args.eval_every == 0):
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": asdict(cfg),
                "tokenizer": tok.state_dict(),
                "step": step,
            }
            torch.save(ckpt, run_dir / "ckpt.pt")

    return run_dir


def main() -> None:
    p = build_argparser()
    args = p.parse_args()
    run_dir = run_training(args)
    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
