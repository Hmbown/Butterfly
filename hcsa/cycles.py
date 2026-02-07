from __future__ import annotations

"""Hamiltonian cycle construction utilities.

We treat token positions as nodes {0..T-1}. A Hamiltonian cycle is a permutation
pi of these nodes; cycle edges connect consecutive nodes in pi, plus wrap-around.

Cycle strategies:
- random_cycle: sample a random permutation
- greedy_cycle: nearest-neighbor greedy on routing similarity (TSP-ish heuristic)
- online_insertion_cycle: update an existing cycle by inserting a new node at the
  best edge (a,b) maximizing delta = s(a,u)+s(u,b)-s(a,b)

All cycle edges are undirected; causality is enforced later in attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def validate_cycle_perm(perm: torch.Tensor, T: int) -> None:
    if perm.ndim != 1 or perm.numel() != T:
        raise ValueError(f"perm must be 1D of length {T}, got {tuple(perm.shape)}")
    if perm.dtype != torch.long:
        raise ValueError(f"perm must be torch.long, got {perm.dtype}")
    # On GPU, using bincount may not be available depending on device;
    # do a conservative check on CPU.
    p = perm.detach().cpu()
    if int(p.min()) != 0 or int(p.max()) != T - 1:
        raise ValueError("perm must contain values in [0, T-1]")
    counts = torch.bincount(p, minlength=T)
    if not torch.all(counts == 1):
        raise ValueError("perm is not a permutation of 0..T-1")


def cycle_prev_next_from_perm(perm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a permutation pi, return prev/next arrays for each node.

    prev[node] = predecessor of node in the cycle
    next[node] = successor of node in the cycle
    """
    T = perm.numel()
    validate_cycle_perm(perm, T)
    device = perm.device
    prev = torch.empty((T,), dtype=torch.long, device=device)
    nxt = torch.empty((T,), dtype=torch.long, device=device)

    # Wrap-around edges
    src = perm
    dst = torch.roll(perm, shifts=-1)
    # src -> dst defines next
    nxt[src] = dst
    prev[dst] = src
    return prev, nxt


def routing_similarity(r: torch.Tensor) -> torch.Tensor:
    """Compute similarity matrix s(i,j) = (r_i dot r_j) / sqrt(d)."""
    if r.ndim != 2:
        raise ValueError(f"r must be [T, d], got {tuple(r.shape)}")
    d = r.shape[1]
    return (r @ r.t()) / (d**0.5)


def random_cycle(T: int, *, generator: Optional[torch.Generator] = None, device: torch.device | None = None) -> torch.Tensor:
    """Random Hamiltonian cycle as a permutation pi of [0..T-1]."""
    device = device or torch.device("cpu")
    perm = torch.randperm(T, generator=generator, device=device)
    return perm.to(torch.long)


def greedy_cycle(r: torch.Tensor, *, start: int = 0) -> torch.Tensor:
    """Nearest-neighbor greedy Hamiltonian cycle.

    Uses routing similarity computed from r [T, d]. Starting at `start`,
    repeatedly pick the unvisited node with maximum similarity to current.

    Complexity: O(T^2) for T <= 1024 is acceptable for experiments.
    """
    T = r.shape[0]
    if not (0 <= start < T):
        raise ValueError(f"start must be in [0,{T-1}]")

    # For determinism and speed, compute the full similarity matrix once.
    s = routing_similarity(r)
    visited = torch.zeros((T,), dtype=torch.bool, device=r.device)
    perm = torch.empty((T,), dtype=torch.long, device=r.device)

    cur = torch.tensor(start, device=r.device)
    for t in range(T):
        perm[t] = cur
        visited[cur] = True
        if t == T - 1:
            break
        # pick best unvisited neighbor
        scores = s[cur].clone()
        scores[visited] = -1e9
        # tie-breaker: argmax returns first max index
        cur = torch.argmax(scores)

    validate_cycle_perm(perm, T)
    return perm


@dataclass
class OnlineInsertionState:
    """Cycle state for online insertion."""

    perm: torch.Tensor  # [t] permutation of nodes 0..t-1


def online_insertion_step(state: OnlineInsertionState, r: torch.Tensor) -> OnlineInsertionState:
    """Insert the newest node u = T-1 into the existing cycle.

    This assumes `state.perm` is a permutation of 0..T-2, and `r` is [T, d]
    for the *current* representations.

    Returns an updated state with a length-T permutation.
    """
    T = r.shape[0]
    if state.perm.numel() != T - 1:
        raise ValueError(f"state.perm must have length T-1={T-1}, got {state.perm.numel()}")
    validate_cycle_perm(state.perm, T - 1)

    perm = state.perm
    u = torch.tensor(T - 1, dtype=torch.long, device=r.device)

    # Evaluate all insertion positions along the cycle edges.
    d = r.shape[1]
    scale = d**0.5
    r_u = r[u]

    a = perm
    b = torch.roll(perm, shifts=-1)

    s_au = (r[a] @ r_u) / scale
    s_ub = (r[b] @ r_u) / scale
    s_ab = (r[a] * r[b]).sum(dim=-1) / scale
    delta = s_au + s_ub - s_ab

    best = int(torch.argmax(delta).item())

    # Insert u between a[best] and b[best]
    new_perm = torch.cat([perm[: best + 1], u.view(1), perm[best + 1 :]], dim=0)
    validate_cycle_perm(new_perm, T)
    return OnlineInsertionState(perm=new_perm)


def online_insertion_cycle(r: torch.Tensor, *, init: str = "sequential") -> OnlineInsertionState:
    """Build a cycle incrementally by repeated online insertion.

    - init='sequential': start with [0] then insert 1..T-1
    - init='random': start from a random 3-cycle and insert the rest

    Returns the final OnlineInsertionState.
    """
    T = r.shape[0]
    if T <= 0:
        raise ValueError("T must be positive")
    device = r.device

    if T == 1:
        return OnlineInsertionState(perm=torch.zeros((1,), dtype=torch.long, device=device))

    if init == "sequential":
        perm = torch.tensor([0], dtype=torch.long, device=device)
        state = OnlineInsertionState(perm=perm)
        for t in range(1, T):
            # Build temporary r for first t+1 nodes.
            r_sub = r[: t + 1]
            state = online_insertion_step(OnlineInsertionState(perm=state.perm), r_sub)
        return state

    raise ValueError(f"Unknown init: {init}")
