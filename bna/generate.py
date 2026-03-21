from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizers import tokenizer_from_state_dict
from .utils import auto_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text from an HCSA checkpoint.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt")
    p.add_argument("--prompt", type=str, default="", help="Prompt text")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    p.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = auto_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"])
    tok = tokenizer_from_state_dict(ckpt["tokenizer"])

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    prompt = args.prompt
    if prompt == "":
        prompt = "\n"

    ids = tok.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    top_k = None if args.top_k <= 0 else args.top_k
    out = model.generate(idx, max_new_tokens=args.max_new, temperature=args.temperature, top_k=top_k)

    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
