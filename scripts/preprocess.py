#!/usr/bin/env python3
"""Pre-tokenize a text file or HF dataset into a binary memmap file.

Usage:
    python scripts/preprocess.py --data data/tinyshakespeare.txt --tokenizer char --out data/tinyshakespeare.bin
    python scripts/preprocess.py --hf-dataset wikitext --hf-config wikitext-2-raw-v1 --tokenizer gpt2 --out data/wikitext2.bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hcsa.data import load_text
from hcsa.tokenizers import build_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-tokenize data for memmap training.")
    p.add_argument("--data", type=str, default="", help="Path to a text file.")
    p.add_argument("--hf-dataset", type=str, default="")
    p.add_argument("--hf-config", type=str, default="")
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe", "gpt2"])
    p.add_argument("--out", type=str, required=True, help="Output .bin file path.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    if args.hf_dataset:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(args.hf_dataset, args.hf_config or None)
        text = "\n".join(ds["train"]["text"])
        val_text = "\n".join(ds.get("validation", ds.get("test", ds["train"]))["text"])
    elif args.data:
        full_text = load_text(args.data)
        n_val = int(len(full_text) * args.val_fraction)
        text = full_text[:-n_val] if n_val > 0 else full_text
        val_text = full_text[-n_val:] if n_val > 0 else ""
    else:
        raise ValueError("Provide --data or --hf-dataset")

    tok = build_tokenizer(args.tokenizer, text_for_char_vocab=text if args.tokenizer == "char" else None)

    # Tokenize
    train_ids = tok.encode(text)
    val_ids = tok.encode(val_text) if val_text else []

    print(f"Vocab size: {tok.vocab_size}")
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Write train
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    np.array(train_ids, dtype=dtype).tofile(out_path)

    # Write val
    if val_ids:
        val_path = out_path.with_suffix(".val.bin")
        np.array(val_ids, dtype=dtype).tofile(val_path)
        print(f"Saved val to: {val_path}")

    # Metadata
    meta = {
        "dtype": str(dtype),
        "vocab_size": tok.vocab_size,
        "n_train_tokens": len(train_ids),
        "n_val_tokens": len(val_ids),
        "tokenizer": tok.state_dict(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved train to: {out_path}")
    print(f"Saved meta to: {meta_path}")


if __name__ == "__main__":
    main()
