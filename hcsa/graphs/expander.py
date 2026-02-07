"""Expander graph strategy using Cayley-style cyclic shifts.

Connects each node to neighbours at distances {1, 2, 4, 8, ...} (powers of 2).
This creates an O(log T) degree graph with O(log T) diameter, ensuring rapid
information flow across the full sequence.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ..graph_strategies import register_strategy


class ExpanderGraphStrategy:
    """Expander graph via cyclic power-of-two shifts.

    For each node i, connect to i +/- {1, 2, 4, 8, ...} mod T.
    Degree = O(log T), diameter = O(log T).
    """

    def __init__(self, max_degree: Optional[int] = None):
        """
        Parameters
        ----------
        max_degree : int, optional
            Cap the number of shift distances.  None = use all powers of 2 up to T/2.
        """
        self.max_degree = max_degree

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        # Compute shift distances: {1, 2, 4, ..., <= T//2}
        shifts = []
        s = 1
        while s <= T // 2:
            shifts.append(s)
            s *= 2

        if self.max_degree is not None and len(shifts) > self.max_degree:
            shifts = shifts[: self.max_degree]

        adj: List[List[int]] = [[] for _ in range(T)]
        for i in range(T):
            neighbours: set[int] = set()
            for d in shifts:
                fwd = (i + d) % T
                bwd = (i - d) % T
                neighbours.add(fwd)
                neighbours.add(bwd)
            neighbours.discard(i)
            adj[i] = sorted(neighbours)

        return adj

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError("Expander graphs do not support incremental updates.")


# Register so it can be used via build_strategy("expander")
register_strategy("expander", ExpanderGraphStrategy)
