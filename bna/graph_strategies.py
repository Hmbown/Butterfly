"""Graph strategy abstraction for sparse attention neighborhoods.

Provides a unified Protocol for constructing adjacency lists and
incrementally updating them.  Existing cycle strategies in ``cycles.py``
are wrapped as concrete implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from .cycles import (
    OnlineInsertionState,
    cycle_prev_next_from_perm,
    edge_disjoint_random_cycles,
    greedy_cycle,
    online_insertion_cycle,
    online_insertion_step,
    random_cycle,
    regular_partition_cycle,
)
from bna.graph.abi import WayfinderGraphABI, build_graph_abi_from_adjacency


@runtime_checkable
class GraphStrategy(Protocol):
    """Protocol for graph construction strategies.

    A strategy builds an adjacency list for *T* token-positions given
    optional routing embeddings and a head index.
    """

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        """Return adjacency list ``adj[i]`` = list of neighbours of node *i*.

        Parameters
        ----------
        T : int
            Number of token positions.
        r : Tensor [T, d], optional
            Routing embeddings.  Some strategies ignore this (e.g. random cycle).
        head_idx : int
            Used to diversify cycles across heads.

        Returns
        -------
        adj : list[list[int]]
            ``adj[i]`` contains the (undirected) neighbour indices of *i*.
        """
        ...

    def update_incremental(
        self,
        state: Any,
        r: torch.Tensor,
        new_node: int,
    ) -> Any:
        """Optionally update the graph state when a new node is added.

        Not all strategies support incremental updates; those that don't
        should raise ``NotImplementedError``.
        """
        ...

    def build(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
        *,
        window: int = 0,
        landmark_stride: int | None = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        """Build a graph ABI object consumable by both Torch and MLX."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations wrapping the existing cycles.py functions
# ---------------------------------------------------------------------------


def _adj_from_perms(perms: List[torch.Tensor], T: int) -> List[List[int]]:
    """Convert one or more cycle permutations into a union adjacency list."""
    adj: List[set[int]] = [set() for _ in range(T)]
    for perm in perms:
        prev, nxt = cycle_prev_next_from_perm(perm)
        for i in range(T):
            adj[i].add(int(prev[i].item()))
            adj[i].add(int(nxt[i].item()))
    return [sorted(s) for s in adj]


class RandomCycleStrategy:
    """One or more random Hamiltonian cycles."""

    def __init__(self, num_cycles: int = 1, seed: int = 0, edge_disjoint: bool = True):
        self.num_cycles = num_cycles
        self.edge_disjoint = bool(edge_disjoint)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

    def _sample_perms(self, T: int, device: torch.device | None = None) -> List[torch.Tensor]:
        if self.num_cycles > 1 and self.edge_disjoint:
            perms_np = edge_disjoint_random_cycles(
                T,
                self.num_cycles,
                generator=self._rng,
            )
            perms = [
                torch.from_numpy(p.astype("int64", copy=False)).to(torch.long)
                for p in perms_np
            ]
        else:
            perms = [
                random_cycle(T, generator=self._rng, device=torch.device("cpu"))
                for _ in range(self.num_cycles)
            ]
        if device is not None:
            perms = [p.to(device) for p in perms]
        return perms

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        perms = self._sample_perms(T, r.device if r is not None else None)
        return _adj_from_perms(perms, T)

    def build(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
        *,
        window: int = 0,
        landmark_stride: int | None = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        perms = self._sample_perms(T, r.device if r is not None else None)
        adj = _adj_from_perms(perms, T)
        cycle_perm = perms[0].detach().cpu().tolist() if perms else None
        all_cycle_perms = [p.detach().cpu().tolist() for p in perms] if perms else None
        return build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=adj,
            window=window,
            landmark_stride=landmark_stride,
            include_self=include_self,
            cycle_perm=cycle_perm,
            all_cycle_perms=all_cycle_perms,
            strategy="random",
            head_idx=head_idx,
            num_cycles=self.num_cycles,
        )

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError("Random cycles do not support incremental updates.")


class GreedyCycleStrategy:
    """Nearest-neighbour greedy Hamiltonian cycle(s)."""

    def __init__(self, num_cycles: int = 1):
        self.num_cycles = num_cycles

    def _sample_perms(
        self, T: int, r: Optional[torch.Tensor], head_idx: int
    ) -> List[torch.Tensor]:
        if r is None:
            raise ValueError("GreedyCycleStrategy requires routing embeddings r")
        perms = []
        for k in range(self.num_cycles):
            start = (k * 997 + head_idx) % T
            perms.append(greedy_cycle(r, start=start))
        return perms

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        perms = self._sample_perms(T, r, head_idx)
        return _adj_from_perms(perms, T)

    def build(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
        *,
        window: int = 0,
        landmark_stride: int | None = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        perms = self._sample_perms(T, r, head_idx)
        adj = _adj_from_perms(perms, T)
        cycle_perm = perms[0].detach().cpu().tolist() if perms else None
        all_cycle_perms = [p.detach().cpu().tolist() for p in perms] if perms else None
        return build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=adj,
            window=window,
            landmark_stride=landmark_stride,
            include_self=include_self,
            cycle_perm=cycle_perm,
            all_cycle_perms=all_cycle_perms,
            strategy="greedy",
            head_idx=head_idx,
            num_cycles=self.num_cycles,
        )

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError("Greedy cycles do not support incremental updates.")


class OnlineInsertionStrategy:
    """Online insertion Hamiltonian cycle with optional state caching."""

    def __init__(self, seed: int = 0):
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        self._state: Optional[OnlineInsertionState] = None

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        if r is None:
            raise ValueError("OnlineInsertionStrategy requires routing embeddings r")
        device = r.device

        if self._state is not None and self._state.perm.numel() == T - 1:
            state = online_insertion_step(self._state, r)
        else:
            if T == 1:
                state = OnlineInsertionState(
                    perm=torch.zeros((1,), dtype=torch.long, device=device)
                )
            elif T == 2:
                state = OnlineInsertionState(
                    perm=torch.tensor([0, 1], dtype=torch.long, device=device)
                )
            else:
                base = random_cycle(
                    T - 1, generator=self._rng, device=torch.device("cpu")
                ).to(device)
                state = online_insertion_step(
                    OnlineInsertionState(perm=base), r
                )
        self._state = state
        return _adj_from_perms([state.perm], T)

    def build(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
        *,
        window: int = 0,
        landmark_stride: int | None = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        adj = self.build_adjacency(T=T, r=r, head_idx=head_idx)
        cycle_perm = (
            self._state.perm.detach().cpu().tolist()
            if self._state is not None
            else None
        )
        all_cycle_perms = [cycle_perm] if cycle_perm is not None else None
        return build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=adj,
            window=window,
            landmark_stride=landmark_stride,
            include_self=include_self,
            cycle_perm=cycle_perm,
            all_cycle_perms=all_cycle_perms,
            strategy="online_insertion",
            head_idx=head_idx,
            num_cycles=1,
        )

    def update_incremental(
        self,
        state: OnlineInsertionState,
        r: torch.Tensor,
        new_node: int,
    ) -> OnlineInsertionState:
        return online_insertion_step(state, r)


class RegularPartitionStrategy:
    """Cluster-balanced Hamiltonian cycle strategy."""

    def __init__(self, num_clusters: int = 8, num_cycles: int = 1, seed: int = 0):
        self.num_clusters = int(max(1, num_clusters))
        self.num_cycles = int(max(1, num_cycles))
        self._rng = np.random.default_rng(int(seed))

    def _sample_perms(self, T: int, device: torch.device | None = None) -> List[torch.Tensor]:
        perms = [
            torch.from_numpy(
                regular_partition_cycle(
                    T,
                    num_clusters=self.num_clusters,
                    generator=self._rng,
                ).astype("int64", copy=False)
            ).to(torch.long)
            for _ in range(self.num_cycles)
        ]
        if device is not None:
            perms = [p.to(device) for p in perms]
        return perms

    def build_adjacency(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
    ) -> List[List[int]]:
        del r, head_idx
        perms = self._sample_perms(T)
        return _adj_from_perms(perms, T)

    def build(
        self,
        T: int,
        r: Optional[torch.Tensor] = None,
        head_idx: int = 0,
        *,
        window: int = 0,
        landmark_stride: int | None = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        del r
        perms = self._sample_perms(T)
        adj = _adj_from_perms(perms, T)
        cycle_perm = perms[0].detach().cpu().tolist() if perms else None
        all_cycle_perms = [p.detach().cpu().tolist() for p in perms] if perms else None
        return build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=adj,
            window=window,
            landmark_stride=landmark_stride,
            include_self=include_self,
            cycle_perm=cycle_perm,
            all_cycle_perms=all_cycle_perms,
            strategy="regular_partition",
            head_idx=head_idx,
            num_cycles=self.num_cycles,
        )

    def update_incremental(self, state: Any, r: torch.Tensor, new_node: int) -> Any:
        raise NotImplementedError("RegularPartitionStrategy does not support incremental updates.")


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "random": RandomCycleStrategy,
    "greedy": GreedyCycleStrategy,
    "online_insertion": OnlineInsertionStrategy,
    "regular_partition": RegularPartitionStrategy,
}


def build_strategy(name: str, **kwargs: Any) -> GraphStrategy:
    """Instantiate a graph strategy by name.

    Parameters
    ----------
    name : str
        One of the registered strategy names.
    **kwargs
        Forwarded to the strategy constructor.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy {name!r}.  Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls(**kwargs)  # type: ignore[call-arg]


def register_strategy(name: str, cls: type) -> None:
    """Register a new graph strategy class under *name*."""
    STRATEGY_REGISTRY[name] = cls
