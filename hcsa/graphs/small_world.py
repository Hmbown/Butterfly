"""Watts-Strogatz small-world network strategy.

Starts with a ring lattice (= local window) and randomly rewires each edge
with probability *p*.  This produces a graph that has both:
- High clustering (local structure)
- Short average path length (long-range shortcuts)

The ring lattice component overlaps with the "window" in HCSA, while the
rewired edges provide the "highway" connections similar to landmarks but
distributed more naturally.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Any, List, Optional

import torch

from ..graph_strategies import register_strategy


class SmallWorldStrategy:
    """Watts-Strogatz small-world network.

    Parameters
    ----------
    k : int
        Each node connects to *k* nearest ring neighbours (k/2 on each side).
        Must be even.
    p : float
        Rewiring probability in [0, 1].  p=0 gives a regular ring lattice,
        p=1 gives a random graph.
    seed : int
        RNG seed for reproducible rewiring.
    """

    def __init__(self, k: int = 4, p: float = 0.1, seed: int = 0):
        if k % 2 != 0:
            raise ValueError("k must be even")
        self.k = k
        self.p = p
        self.seed = seed

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        rng = stdlib_random.Random(self.seed + head_idx)
        k = min(self.k, T - 1)
        half_k = k // 2

        # Start with ring lattice
        adj: List[set[int]] = [set() for _ in range(T)]
        for i in range(T):
            for j in range(1, half_k + 1):
                fwd = (i + j) % T
                bwd = (i - j) % T
                adj[i].add(fwd)
                adj[i].add(bwd)
                adj[fwd].add(i)
                adj[bwd].add(i)

        # Rewire with probability p
        for i in range(T):
            for j in range(1, half_k + 1):
                if rng.random() < self.p:
                    target = (i + j) % T
                    # Remove original edge
                    adj[i].discard(target)
                    adj[target].discard(i)
                    # Pick random new target (not self, not existing neighbour)
                    candidates = [n for n in range(T) if n != i and n not in adj[i]]
                    if candidates:
                        new_target = rng.choice(candidates)
                        adj[i].add(new_target)
                        adj[new_target].add(i)

        return [sorted(s) for s in adj]

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError(
            "SmallWorldStrategy does not support incremental updates."
        )


register_strategy("small_world", SmallWorldStrategy)
