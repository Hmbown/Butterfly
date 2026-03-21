"""KV Cache for efficient autoregressive generation.

During generation, we cache K and V tensors from previous positions to avoid
recomputation.  For dense attention this is straightforward.  For HCSA, the
cache is indexed by the neighbor indices (neigh_idx[t_new, :]) to gather only
the relevant cached K,V entries for the new token position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class KVCache:
    """Key-Value cache for a single attention layer.

    Stores cached K, V tensors with shape [B, H, T_cached, dh].
    """

    k: Optional[torch.Tensor] = None  # [B, H, T_cached, dh]
    v: Optional[torch.Tensor] = None  # [B, H, T_cached, dh]

    @property
    def seq_len(self) -> int:
        if self.k is None:
            return 0
        return self.k.shape[2]

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "KVCache":
        """Append new K, V to the cache.

        Parameters
        ----------
        k_new, v_new : Tensor [B, H, T_new, dh]
            New key/value tensors to append.

        Returns
        -------
        KVCache
            Updated cache (self).
        """
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)  # type: ignore[list-item]
            self.v = torch.cat([self.v, v_new], dim=2)  # type: ignore[list-item]
        return self

    def gather_for_sparse(
        self,
        neigh_idx: torch.Tensor,
        head_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather cached K, V for sparse HCSA neighbors.

        Parameters
        ----------
        neigh_idx : Tensor [D]
            Neighbor indices for the new position (0-based, -1 = invalid).
        head_idx : int
            Which head to gather from.

        Returns
        -------
        (k_neigh, v_neigh) : tuple of Tensor [B, D, dh]
            Gathered key/value for the neighbors.
        """
        assert self.k is not None and self.v is not None
        B = self.k.shape[0]
        dh = self.k.shape[3]
        D = neigh_idx.shape[0]

        safe_idx = neigh_idx.clamp(min=0)  # [D]
        # Gather from [B, T_cached, dh] for the given head
        k_h = self.k[:, head_idx]  # [B, T_cached, dh]
        v_h = self.v[:, head_idx]  # [B, T_cached, dh]

        idx = safe_idx.view(1, D, 1).expand(B, D, dh)
        k_neigh = torch.gather(k_h, dim=1, index=idx)  # [B, D, dh]
        v_neigh = torch.gather(v_h, dim=1, index=idx)  # [B, D, dh]

        return k_neigh, v_neigh

    def reset(self) -> None:
        """Clear the cache."""
        self.k = None
        self.v = None


@dataclass
class LayerCaches:
    """Collection of KV caches for all layers."""

    caches: list[KVCache] = field(default_factory=list)

    @classmethod
    def create(cls, n_layers: int) -> "LayerCaches":
        return cls(caches=[KVCache() for _ in range(n_layers)])

    def __getitem__(self, idx: int) -> KVCache:
        return self.caches[idx]

    def __len__(self) -> int:
        return len(self.caches)

    def reset(self) -> None:
        for cache in self.caches:
            cache.reset()

    @property
    def seq_len(self) -> int:
        if not self.caches:
            return 0
        return self.caches[0].seq_len
