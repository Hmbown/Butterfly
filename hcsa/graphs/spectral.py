"""Spectral sparsification graph strategy.

Connects positions based on alignment in the top-k eigenspace of the
routing similarity matrix.  Positions that are close in eigenspace get
connected, creating a structure that captures the dominant modes of the
learned routing function.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ..graph_strategies import register_strategy


class SpectralStrategy:
    """Spectral sparsification via eigenvector alignment.

    Parameters
    ----------
    k_eigenvectors : int
        Number of top eigenvectors to use for connectivity.
    n_neighbors : int
        Number of nearest neighbours per node in eigenspace.
    """

    def __init__(self, k_eigenvectors: int = 8, n_neighbors: int = 4):
        self.k_eigenvectors = k_eigenvectors
        self.n_neighbors = n_neighbors

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        if r is None:
            raise ValueError("SpectralStrategy requires routing embeddings r")

        # Compute similarity matrix
        d = r.shape[1]
        S = (r @ r.t()) / (d ** 0.5)  # [T, T]

        # Eigendecomposition (use top-k eigenvalues)
        k = min(self.k_eigenvectors, T - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        # eigh returns ascending order; take top-k
        top_k_vecs = eigenvectors[:, -k:]  # [T, k]

        # Compute distance in eigenspace
        # dist[i,j] = ||top_k_vecs[i] - top_k_vecs[j]||
        diff = top_k_vecs.unsqueeze(0) - top_k_vecs.unsqueeze(1)  # [T, T, k]
        dist = diff.pow(2).sum(-1)  # [T, T]

        # For each node, find n_neighbors nearest in eigenspace
        n = min(self.n_neighbors, T - 1)
        # Set self-distance to infinity
        dist.fill_diagonal_(float("inf"))

        adj: List[List[int]] = []
        for i in range(T):
            _, indices = dist[i].topk(n, largest=False)
            adj.append(sorted(indices.tolist()))

        return adj

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError("SpectralStrategy does not support incremental updates.")


register_strategy("spectral", SpectralStrategy)
