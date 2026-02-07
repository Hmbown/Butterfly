#!/usr/bin/env python3
"""Long-range dependency probes.

Tests whether sparse attention can support:
1. Copy task: reproduce a token seen N steps ago
2. Induction heads: detect [A][B]...[A] and predict [B]
3. Associative recall: key-value pairs stored at distance

Usage:
    python scripts/probe_long_range.py --task copy --seq-len 128 --distance 64
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from hcsa.model import GPT, GPTConfig
from hcsa.utils import auto_device, set_seed


def make_copy_task_data(
    batch_size: int,
    seq_len: int,
    distance: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate copy task data.

    Pattern: [random tokens...][marker][content][...padding...][marker][expected: content]
    The model must copy tokens from `distance` positions back.
    """
    assert distance < seq_len // 2
    content_len = min(distance // 2, 8)

    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # -100 = ignore in CE loss

    marker = 0
    pad = 1

    for b in range(batch_size):
        # Place content at start
        content_start = 2
        x[b, content_start - 1] = marker
        content = torch.randint(2, vocab_size, (content_len,))
        x[b, content_start : content_start + content_len] = content

        # Place recall marker
        recall_start = content_start + distance
        if recall_start + content_len >= seq_len:
            continue
        x[b, recall_start - 1] = marker

        # Target: predict content after recall marker
        y[b, recall_start : recall_start + content_len] = content

    return x, y


def make_induction_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate induction head data.

    Pattern: ...A B ... A -> predict B
    """
    x = torch.randint(2, vocab_size, (batch_size, seq_len))
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        # Place pattern in first half
        pos1 = seq_len // 4
        a_token = torch.randint(2, vocab_size, (1,)).item()
        b_token = torch.randint(2, vocab_size, (1,)).item()
        x[b, pos1] = a_token
        x[b, pos1 + 1] = b_token

        # Place trigger in second half
        pos2 = 3 * seq_len // 4
        x[b, pos2] = a_token
        y[b, pos2] = b_token  # should predict b_token after seeing a_token

    return x, y


def evaluate_probe(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on probe data."""
    model.eval()
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits = model(x)["logits"]

    # Only evaluate on non-ignored positions
    mask = y != -100
    if not mask.any():
        return {"accuracy": 0.0, "loss": float("inf")}

    logits_flat = logits[mask]
    y_flat = y[mask]

    loss = F.cross_entropy(logits_flat, y_flat).item()
    preds = logits_flat.argmax(dim=-1)
    accuracy = (preds == y_flat).float().mean().item()

    return {"accuracy": accuracy, "loss": loss}


def run_probe(
    task: str,
    attn: str,
    seq_len: int = 128,
    distance: int = 64,
    n_steps: int = 500,
    device: torch.device | None = None,
) -> dict:
    """Train a model on a probe task and evaluate."""
    if device is None:
        device = auto_device("auto")

    set_seed(42)
    vocab_size = 32

    cfg = GPTConfig(
        vocab_size=vocab_size, seq_len=seq_len, n_layers=4, n_heads=4, n_embd=128,
        attn=attn, cycle="random", window=16, landmark_stride=16, seed=42,  # type: ignore[arg-type]
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, n_steps + 1):
        if task == "copy":
            x, y = make_copy_task_data(16, seq_len, distance, vocab_size)
        elif task == "induction":
            x, y = make_induction_data(16, seq_len, vocab_size)
        else:
            raise ValueError(f"Unknown task: {task}")

        x, y = x.to(device), y.to(device)
        logits = model(x)["logits"]

        mask = y != -100
        if mask.any():
            loss = F.cross_entropy(logits[mask], y[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Final evaluation
    if task == "copy":
        x_eval, y_eval = make_copy_task_data(64, seq_len, distance, vocab_size)
    else:
        x_eval, y_eval = make_induction_data(64, seq_len, vocab_size)

    results = evaluate_probe(model, x_eval, y_eval, device)
    results["task"] = task
    results["attn"] = attn
    results["seq_len"] = seq_len
    results["distance"] = distance
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="copy", choices=["copy", "induction"])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--distance", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = auto_device(args.device)

    for attn in ["dense", "hcsa"]:
        print(f"\n--- {attn.upper()} ---")
        results = run_probe(
            args.task, attn, args.seq_len, args.distance,
            n_steps=args.steps, device=device,
        )
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Loss: {results['loss']:.4f}")


if __name__ == "__main__":
    main()
