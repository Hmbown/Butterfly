#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from mlx_lm.utils import load_tokenizer


TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".go",
    ".rs",
    ".sh",
    ".sql",
    ".html",
    ".css",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_").lower()


def _iter_local_docs(path: Path) -> Iterable[Tuple[str, str]]:
    if path.is_file():
        yield (str(path), path.read_text(encoding="utf-8", errors="ignore"))
        return

    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS]
    for fp in sorted(files):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            yield (str(fp), text)


def _pick_text_field(sample: Dict[str, Any]) -> str:
    preferred = ["text", "content", "code", "markdown", "document", "body"]
    for key in preferred:
        if key in sample and isinstance(sample[key], str):
            return key
    for key, value in sample.items():
        if isinstance(value, str):
            return key
    raise ValueError("Could not infer text field from HF sample.")


def _load_hf_docs(dataset_id: str, split: str, max_docs: int | None) -> List[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    if len(ds) == 0:
        raise ValueError(f"HF dataset split is empty: {dataset_id}:{split}")
    text_key = _pick_text_field(ds[0])

    docs: List[Tuple[str, str]] = []
    for i, row in enumerate(ds):
        txt = row.get(text_key)
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt:
            continue
        docs.append((f"{dataset_id}:{split}:{i}", txt))
        if max_docs is not None and len(docs) >= max_docs:
            break
    return docs


def _eos_token_id(tokenizer) -> int:
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids:
        return int(tokenizer.eos_token_ids[0])
    return 0


def _tokenizer_hash(tokenizer) -> str:
    name = str(getattr(tokenizer, "name_or_path", "unknown"))
    eos = _eos_token_id(tokenizer)
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(f":eos:{eos}".encode("utf-8"))
    return h.hexdigest()[:16]


def _pack_tokens_to_fixed(ids: np.ndarray, seq_len: int) -> np.ndarray:
    usable = (len(ids) // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}; only {len(ids)} available")
    return ids[:usable].reshape(-1, seq_len)


def _write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.uint32, copy=False).ravel().tofile(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess long-context dataset for Qwen3-4B")
    p.add_argument("--dataset", type=str, required=True, help="hf:<id> or local:<path>")
    p.add_argument(
        "--tokenizer",
        type=str,
        default="mlx-community/Qwen3-4B-4bit",
        help="Tokenizer source (mlx path or HF id)",
    )
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--local-fallback", type=Path, default=Path("."))
    p.add_argument("--out-root", type=Path, default=Path("datasets/qwen3_4b"))
    args = p.parse_args()

    tokenizer = load_tokenizer(args.tokenizer, tokenizer_config_extra={"trust_remote_code": True})
    eos_id = _eos_token_id(tokenizer)

    docs: List[Tuple[str, str]] = []
    provenance: Dict[str, Any] = {"requested_dataset": args.dataset}

    max_docs = None if int(args.max_docs) <= 0 else int(args.max_docs)
    if args.dataset.startswith("hf:"):
        ds_id = args.dataset.split("hf:", 1)[1]
        try:
            docs = _load_hf_docs(ds_id, args.hf_split, max_docs=max_docs)
            provenance["resolved_dataset"] = f"hf:{ds_id}"
            provenance["fallback_used"] = False
        except Exception as exc:
            fb = args.local_fallback.resolve()
            docs = list(_iter_local_docs(fb))
            provenance["resolved_dataset"] = f"local:{fb}"
            provenance["fallback_used"] = True
            provenance["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    elif args.dataset.startswith("local:"):
        local_path = Path(args.dataset.split("local:", 1)[1]).resolve()
        docs = list(_iter_local_docs(local_path))
        provenance["resolved_dataset"] = f"local:{local_path}"
        provenance["fallback_used"] = False
    else:
        raise ValueError("--dataset must start with hf: or local:")

    if not docs:
        raise ValueError("No text documents found to preprocess.")

    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(len(docs))
    docs = [docs[int(i)] for i in order]

    if 0.0 < float(args.data_fraction) < 1.0:
        keep = max(1, int(len(docs) * float(args.data_fraction)))
        docs = docs[:keep]

    token_ids: list[int] = []
    used_docs = 0
    for doc_id, text in docs:
        _ = doc_id
        ids = tokenizer.encode(text)
        if not ids:
            continue
        token_ids.extend(int(t) for t in ids)
        token_ids.append(int(eos_id))
        used_docs += 1
        if args.max_tokens > 0 and len(token_ids) >= int(args.max_tokens):
            token_ids = token_ids[: int(args.max_tokens)]
            break

    ids_np = np.asarray(token_ids, dtype=np.uint32)
    packed = _pack_tokens_to_fixed(ids_np, int(args.seq_len))

    seq_order = rng.permutation(len(packed))
    packed = packed[seq_order]

    n_val = max(1, int(len(packed) * float(args.val_fraction)))
    if len(packed) - n_val < 1:
        n_val = max(1, len(packed) - 1)

    train_arr = packed[:-n_val]
    val_arr = packed[-n_val:]
    val_long_arr = val_arr.copy()

    dataset_name = provenance["resolved_dataset"].replace(":", "_")
    dataset_dir = args.out_root / _slug(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_bin(dataset_dir / "train.bin", train_arr)
    _write_bin(dataset_dir / "val.bin", val_arr)
    _write_bin(dataset_dir / "val_long.bin", val_long_arr)

    meta = {
        "seq_len": int(args.seq_len),
        "dtype": "uint32",
        "packing": "fixed_length_non_overlapping",
        "tokenizer_source": args.tokenizer,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "eos_token_id": int(eos_id),
        "seed": int(args.seed),
        "data_fraction": float(args.data_fraction),
        "val_fraction": float(args.val_fraction),
        "num_docs_used": int(used_docs),
        "num_tokens_raw": int(len(token_ids)),
        "num_train_sequences": int(train_arr.shape[0]),
        "num_val_sequences": int(val_arr.shape[0]),
        "num_val_long_sequences": int(val_long_arr.shape[0]),
        "provenance": provenance,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"dataset_dir={dataset_dir}")
    print(f"train_sequences={train_arr.shape[0]}")
    print(f"val_sequences={val_arr.shape[0]}")
    print(f"seq_len={args.seq_len}")
    print(f"tokens_raw={len(token_ids)}")


if __name__ == "__main__":
    main()

