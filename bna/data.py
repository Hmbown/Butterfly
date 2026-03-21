from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import torch

from .tokenizers import Tokenizer


@dataclass
class LMData:
    train: torch.Tensor  # [N]
    val: torch.Tensor  # [M]


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_datasets(text: str, tok: Tokenizer, val_fraction: float = 0.1) -> LMData:
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = ids.numel()
    n_val = int(n * val_fraction)
    if n_val < 1:
        raise ValueError(
            f"Not enough tokens ({n}) to make a validation split. Provide more text."
        )
    return LMData(train=ids[:-n_val], val=ids[-n_val:])


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of sequences (x, y) from a 1D token stream."""
    # Sample random starting positions
    max_start = data.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small for seq_len={seq_len}: need at least {seq_len+2} tokens."
        )
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)
