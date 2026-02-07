"""Hierarchical multi-scale cycle strategy.

Creates a "highway system" with multiple levels:
- Level 0: local window (always-on causal)
- Level 1: Hamiltonian cycle over all positions
- Level 2+: coarse cycles over super-nodes (groups of positions)

Each level provides connections at a different scale, ensuring both local
detail and global reach.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ..cycles import cycle_prev_next_from_perm, random_cycle, greedy_cycle
from ..graph_strategies import register_strategy


class HierarchicalStrategy:
    """Multi-scale hierarchical cycle strategy.

    Parameters
    ----------
    n_levels : int
        Number of hierarchy levels.  Level 0 is the base cycle,
        each subsequent level groups positions into super-nodes.
    block_size : int
        Size of blocks for super-node grouping at each level.
        Level k groups into blocks of (block_size ** k).
    cycle_type : str
        Base cycle strategy for each level.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        n_levels: int = 3,
        block_size: int = 4,
        cycle_type: str = "random",
        seed: int = 0,
    ):
        self.n_levels = n_levels
        self.block_size = block_size
        self.cycle_type = cycle_type
        self.seed = seed

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        rng = torch.Generator(device="cpu").manual_seed(self.seed + head_idx)
        adj: List[set[int]] = [set() for _ in range(T)]

        for level in range(self.n_levels):
            group_size = self.block_size ** level
            # Number of super-nodes at this level
            n_super = max(1, (T + group_size - 1) // group_size)

            if n_super < 2:
                continue

            # Build cycle over super-nodes
            if self.cycle_type == "greedy" and r is not None:
                # Average routing embeddings within each super-node
                r_super = []
                for g in range(n_super):
                    start = g * group_size
                    end = min(start + group_size, T)
                    r_super.append(r[start:end].mean(dim=0))
                r_super_t = torch.stack(r_super)
                perm = greedy_cycle(r_super_t, start=head_idx % n_super)
            else:
                perm = random_cycle(n_super, generator=rng, device=torch.device("cpu"))

            prev, nxt = cycle_prev_next_from_perm(perm)

            # Translate super-node edges to position-level edges
            for g in range(n_super):
                g_prev = int(prev[g])
                g_next = int(nxt[g])

                # Representative positions for this super-node
                rep = g * group_size  # first position in group
                rep_prev = g_prev * group_size
                rep_next = g_next * group_size

                if rep < T and rep_prev < T:
                    adj[rep].add(rep_prev)
                    adj[rep_prev].add(rep)
                if rep < T and rep_next < T:
                    adj[rep].add(rep_next)
                    adj[rep_next].add(rep)

        return [sorted(s) for s in adj]

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError(
            "HierarchicalStrategy does not support incremental updates."
        )


register_strategy("hierarchical", HierarchicalStrategy)
