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

import numpy as np
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


def num_blocks_for_seq_len(seq_len: int, block_size: int) -> int:
    """Return ceil(seq_len / block_size) for block-sparse attention."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return int((int(seq_len) + int(block_size) - 1) // int(block_size))


def log_landmark_blocks(num_blocks: int) -> list[int]:
    """Return fixed log-spaced landmark block ids: 0, 1, 2, 4, 8, ..."""
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    landmarks = [0]
    stride = 1
    while stride < int(num_blocks):
        landmarks.append(int(stride))
        stride *= 2
    return sorted(set(landmarks))


def block_hamiltonian_cycles(
    seq_len: int,
    block_size: int,
    *,
    strategy: str = "random",
    num_cycles: int = 1,
    edge_disjoint: bool = True,
    regular_num_clusters: int = 8,
    seed: int = 0,
    head_idx: int = 0,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Sample Hamiltonian cycles over blocks instead of individual tokens.

    The returned permutations are over block ids ``0..ceil(seq_len / block_size)-1``.
    This keeps the Hamiltonian machinery reusable while shifting the attention
    pattern onto GPU-friendly contiguous tiles.
    """
    if num_cycles <= 0:
        raise ValueError("num_cycles must be positive")

    num_blocks = num_blocks_for_seq_len(seq_len, block_size)
    target_device = device or torch.device("cpu")
    if num_blocks == 1:
        return [torch.zeros((1,), dtype=torch.long, device=target_device) for _ in range(num_cycles)]

    strategy_name = str(strategy)
    seeded_head = int(seed) + 7919 * int(head_idx)

    if strategy_name == "random":
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seeded_head)
        if num_cycles > 1 and edge_disjoint:
            perms_np = edge_disjoint_random_cycles(
                num_blocks,
                num_cycles,
                generator=rng,
            )
            perms = [
                torch.from_numpy(np.asarray(perm, dtype=np.int64)).to(dtype=torch.long, device=target_device)
                for perm in perms_np
            ]
        else:
            perms = [
                random_cycle(num_blocks, generator=rng, device=target_device)
                for _ in range(num_cycles)
            ]
    elif strategy_name == "regular_partition":
        rng = np.random.default_rng(seeded_head)
        perms = [
            torch.from_numpy(
                regular_partition_cycle(
                    num_blocks,
                    num_clusters=regular_num_clusters,
                    generator=rng,
                ).astype(np.int64, copy=False)
            ).to(dtype=torch.long, device=target_device)
            for _ in range(num_cycles)
        ]
    else:
        raise ValueError(
            "block_hamiltonian_cycles currently supports only static strategies "
            f"('random', 'regular_partition'); got {strategy_name!r}"
        )

    for perm in perms:
        validate_cycle_perm(perm, num_blocks)
    return perms


def _edge_set_from_perm_np(perm: np.ndarray) -> set[tuple[int, int]]:
    p = np.asarray(perm, dtype=np.int64).reshape(-1)
    t = int(p.shape[0])
    edges: set[tuple[int, int]] = set()
    if t <= 1:
        return edges
    for i in range(t):
        u = int(p[i])
        v = int(p[(i + 1) % t])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return edges


def _as_numpy_rng(generator: Optional[object]) -> np.random.Generator:
    if isinstance(generator, np.random.Generator):
        return generator
    if isinstance(generator, torch.Generator):
        seed = int(
            torch.randint(
                low=0,
                high=2**31 - 1,
                size=(1,),
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.int64,
            ).item()
        )
        return np.random.default_rng(seed)
    return np.random.default_rng()


def _sample_cycle_np(T: int, *, generator: Optional[object] = None) -> np.ndarray:
    if isinstance(generator, torch.Generator):
        return torch.randperm(T, generator=generator, device=torch.device("cpu")).cpu().numpy()
    if isinstance(generator, np.random.Generator):
        return generator.permutation(T).astype(np.int64)
    return np.random.default_rng().permutation(T).astype(np.int64)


def _walecki_even_cycles(T: int) -> list[np.ndarray]:
    """Construct edge-disjoint Hamilton cycles for complete graph K_T (T even)."""
    if T % 2 != 0 or T < 4:
        return []
    m = T // 2
    mod = T - 1
    inf = T - 1
    out: list[np.ndarray] = []
    for i in range(m - 1):
        seq: list[int] = [inf]
        for t in range(1, m):
            seq.append((i - (t - 1)) % mod)
            seq.append((i + t) % mod)
        seq.append((i - (m - 1)) % mod)
        out.append(np.asarray(seq, dtype=np.int64))
    return out


def edge_disjoint_random_cycles(
    T: int,
    num_cycles: int,
    *,
    max_retries: int = 100,
    generator=None,
) -> list[np.ndarray]:
    """Generate edge-disjoint random Hamiltonian cycles."""
    if T <= 0:
        raise ValueError("T must be positive")
    if num_cycles <= 0:
        raise ValueError("num_cycles must be positive")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")

    cycles: list[np.ndarray] = []
    used_edges: set[tuple[int, int]] = set()

    for cycle_idx in range(num_cycles):
        accepted = False
        for _retry in range(max_retries):
            candidate = _sample_cycle_np(T, generator=generator).astype(np.int64, copy=False)
            edges = _edge_set_from_perm_np(candidate)
            if used_edges.isdisjoint(edges):
                cycles.append(candidate)
                used_edges |= edges
                accepted = True
                break
        if not accepted:
            raise ValueError(
                "Failed to generate edge-disjoint cycles "
                f"(T={T}, num_cycles={num_cycles}, cycle_idx={cycle_idx}, max_retries={max_retries})"
            )

    return cycles


def verify_edge_disjoint(cycles: list[np.ndarray]) -> tuple[bool, int]:
    """Check if cycles are pairwise edge-disjoint."""
    edge_counts: dict[tuple[int, int], int] = {}
    for perm in cycles:
        for edge in _edge_set_from_perm_np(np.asarray(perm, dtype=np.int64)):
            edge_counts[edge] = int(edge_counts.get(edge, 0) + 1)
    shared = int(sum(1 for c in edge_counts.values() if c > 1))
    return shared == 0, shared


def regular_partition_cycle(
    T: int,
    num_clusters: int = 8,
    *,
    generator=None,
) -> np.ndarray:
    """Construct a cluster-balanced Hamiltonian cycle permutation."""
    if T <= 0:
        raise ValueError("T must be positive")
    k = int(max(1, min(num_clusters, T)))
    rng = _as_numpy_rng(generator)

    vertices = np.arange(T, dtype=np.int64)
    clusters = [np.asarray(c, dtype=np.int64) for c in np.array_split(vertices, k)]
    shuffled: list[np.ndarray] = []
    for c in clusters:
        if c.size <= 1:
            shuffled.append(c.copy())
        else:
            shuffled.append(rng.permutation(c))

    cursor = [0 for _ in range(k)]
    out: list[int] = []
    while len(out) < T:
        progressed = False
        cluster_order = rng.permutation(k).tolist()
        for cidx in cluster_order:
            pos = cursor[cidx]
            if pos >= int(shuffled[cidx].shape[0]):
                continue
            out.append(int(shuffled[cidx][pos]))
            cursor[cidx] = pos + 1
            progressed = True
        if not progressed:  # pragma: no cover
            break

    perm = np.asarray(out, dtype=np.int64)
    if perm.shape[0] != T:
        raise RuntimeError(f"regular_partition_cycle failed to produce length {T}, got {perm.shape[0]}")
    if np.unique(perm).shape[0] != T:
        raise RuntimeError("regular_partition_cycle produced duplicate vertices")
    return perm


def covering_cycles(
    T: int,
    *,
    max_cycles: int = 20,
    coverage_target: float = 0.99,
    generator=None,
) -> tuple[list[np.ndarray], float]:
    """Generate random cycles until target edge coverage is reached."""
    if T <= 1:
        return [np.arange(max(T, 1), dtype=np.int64)], 1.0
    max_c = int(max(1, max_cycles))
    target = float(min(1.0, max(0.0, coverage_target)))

    total_possible = int(T * (T - 1) // 2)
    covered: set[tuple[int, int]] = set()
    cycles: list[np.ndarray] = []
    seeded = _walecki_even_cycles(T)
    for perm in seeded:
        if len(cycles) >= max_c:
            break
        cycles.append(perm)
        covered |= _edge_set_from_perm_np(perm)
        frac = float(len(covered)) / float(max(total_possible, 1))
        if frac >= target:
            return cycles, frac

    if T <= 64:
        candidate_trials = 8192
    elif T <= 128:
        candidate_trials = 4096
    else:
        candidate_trials = 64

    for _ in range(max(0, max_c - len(cycles))):
        best_perm: np.ndarray | None = None
        best_edges: set[tuple[int, int]] | None = None
        best_gain = -1
        for _c in range(candidate_trials):
            perm = _sample_cycle_np(T, generator=generator).astype(np.int64, copy=False)
            edges = _edge_set_from_perm_np(perm)
            gain = int(len(edges - covered))
            if gain > best_gain:
                best_gain = gain
                best_perm = perm
                best_edges = edges
                if gain == T:
                    break
        if best_perm is None or best_edges is None:  # pragma: no cover
            break
        cycles.append(best_perm)
        covered |= best_edges
        frac = float(len(covered)) / float(max(total_possible, 1))
        if frac >= target:
            return cycles, frac

    frac = float(len(covered)) / float(max(total_possible, 1))
    return cycles, frac


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


def recommended_num_cycles(T: int, *, expansion_constant: float = 2.0) -> int:
    """Return the theoretically-motivated number of edge-disjoint cycles.

    Based on expander graph theory: d = ceil(c * log2(T)) edge-disjoint
    Hamiltonian cycles give an (n, 2d, O(sqrt(d)))-graph, guaranteeing:
    - Spectral gap d/lambda = Omega(sqrt(d))
    - Diameter O(log T)
    - Resilience: survives dropping up to ~half the edges

    The expansion_constant c controls the trade-off:
    - c=1: minimal expansion, d~=log2(T)
    - c=2: good expansion (recommended default)
    - c=3: strong expansion, higher compute cost

    Returns:
        int: recommended number of cycles, >= 1
    """
    import math

    return max(1, math.ceil(expansion_constant * math.log2(max(2, T))))


def max_edge_disjoint_cycles(T: int) -> int:
    """Theoretical maximum edge-disjoint Hamiltonian cycles for K_T.

    For T even: floor((T-1)/2) (Walecki decomposition)
    For T odd:  floor(T/2)
    """
    return (T - 1) // 2 if T >= 4 else (1 if T >= 3 else 0)
