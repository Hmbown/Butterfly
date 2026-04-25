"""Stream HuggingFace text datasets and pack them into mmap .bin files.

The output format is compatible with :class:`bna.data_mmap.MemmapDataset`:
a flat little-endian array of token IDs (uint16 if vocab < 65536, uint32 else)
written contiguously to disk, plus a ``<bin>.meta.json`` sidecar describing the
provenance and dtype.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Optional

import numpy as np
from tqdm import tqdm


def _resolve_dtype(dtype: str) -> np.dtype:
    np_dtype = np.dtype(dtype)
    if np_dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        raise ValueError(f"Unsupported dtype {dtype!r}; expected uint16 or uint32")
    return np_dtype


def _max_token_for_dtype(np_dtype: np.dtype) -> int:
    return int(np.iinfo(np_dtype).max)


def stream_hf_to_mmap_bin(
    dataset_id: str,
    *,
    output_path: str,
    tokenizer,
    text_field: str = "text",
    split: str = "train",
    max_tokens: Optional[int] = None,
    dataset_config: Optional[str] = None,
    trust_remote_code: bool = True,
    append_eos: bool = True,
    eos_id: Optional[int] = None,
    dtype: str = "uint32",
) -> dict:
    """Stream an HF dataset, tokenize, and pack tokens into ``output_path``.

    The companion ``<output_path>.meta.json`` is written alongside the bin.
    Returns the meta dict.
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError(
            "stream_hf_to_mmap_bin requires the `datasets` extra. "
            "Install with: pip install -e .[hf]"
        ) from e

    np_dtype = _resolve_dtype(dtype)
    max_tok_value = _max_token_for_dtype(np_dtype)

    if append_eos and eos_id is None:
        eos_id = getattr(tokenizer, "eos_id", None)
        if eos_id is None:
            inner = getattr(tokenizer, "_tok", None)
            eos_id = getattr(inner, "eos_token_id", None) if inner is not None else None
    if append_eos and eos_id is None:
        raise ValueError(
            "append_eos=True but tokenizer has no eos_id; pass eos_id=... explicitly"
        )

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    ds = load_dataset(
        dataset_id,
        name=dataset_config,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    written = 0
    tokenizer_type = type(tokenizer).__name__

    pbar = tqdm(
        total=int(max_tokens) if max_tokens is not None else None,
        unit="tok",
        unit_scale=True,
        desc=f"tokenize:{dataset_id}",
    )
    with open(output_path, "wb") as fh:
        for example in ds:
            text = example.get(text_field)
            if not text:
                continue
            ids = tokenizer.encode(text)
            if not ids:
                continue
            if append_eos:
                ids = list(ids) + [int(eos_id)]
            if max_tokens is not None and written + len(ids) > max_tokens:
                ids = ids[: max(0, max_tokens - written)]
                if not ids:
                    break
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size and int(arr.max()) > max_tok_value:
                raise ValueError(
                    f"Token id {int(arr.max())} exceeds {dtype} max ({max_tok_value}); "
                    "use dtype='uint32'"
                )
            arr = arr.astype(np_dtype, copy=False)
            fh.write(arr.tobytes())
            written += int(arr.size)
            pbar.update(int(arr.size))
            if max_tokens is not None and written >= max_tokens:
                break
    pbar.close()

    meta: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "split": split,
        "text_field": text_field,
        "num_tokens": int(written),
        "dtype": str(np_dtype),
        "eos_id": int(eos_id) if eos_id is not None else None,
        "tokenizer_type": tokenizer_type,
        "produced_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(output_path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta
