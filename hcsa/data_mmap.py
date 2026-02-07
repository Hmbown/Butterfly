"""Memory-mapped dataset for pre-tokenized data.

For large-scale training, pre-tokenize the data once with ``scripts/preprocess.py``
and stream from the resulting ``.bin`` file via numpy memmap.  This avoids loading
the entire dataset into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class MemmapDataset:
    """Memory-mapped pre-tokenized dataset.

    The ``.bin`` file contains a flat array of uint16 (or uint32) token IDs.
    A small ``.meta.json`` sidecar stores ``{dtype, vocab_size, n_tokens}``.

    Parameters
    ----------
    path : str | Path
        Path to the ``.bin`` file.
    dtype : str
        Numpy dtype of the stored tokens (``"uint16"`` or ``"uint32"``).
    """

    def __init__(self, path: str | Path, dtype: str = "uint16"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        np_dtype = np.dtype(dtype)
        self.data = np.memmap(str(self.path), dtype=np_dtype, mode="r")
        self.n_tokens = len(self.data)

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (x, y) pairs."""
        max_start = self.n_tokens - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset too small ({self.n_tokens} tokens) for seq_len={seq_len}"
            )
        starts = np.random.randint(0, max_start, size=(batch_size,))
        x = np.stack([self.data[s : s + seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + seq_len + 1] for s in starts])
        return (
            torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device),
        )

    def get_fraction(self, fraction: float) -> "MemmapDataset":
        """Return a view of the first ``fraction`` of the data."""
        n = max(1, int(self.n_tokens * fraction))
        ds = MemmapDataset.__new__(MemmapDataset)
        ds.path = self.path
        ds.data = self.data[:n]
        ds.n_tokens = n
        return ds
