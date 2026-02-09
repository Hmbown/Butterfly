from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


class EdgeType(IntEnum):
    PAD = 0
    CYCLE = 1
    WINDOW = 2
    LANDMARK = 3
    REWIRE = 4


EDGE_PRIORITY: dict[EdgeType, int] = {
    EdgeType.WINDOW: 0,
    EdgeType.LANDMARK: 1,
    EdgeType.REWIRE: 2,
    EdgeType.CYCLE: 3,
}


@dataclass(frozen=True)
class WayfinderGraphABI:
    """Language-agnostic graph ABI.

    Shapes:
    - single-head: neigh_idx [T, D], edge_type [T, D]
    - multi-head:  neigh_idx [H, T, D], edge_type [H, T, D]
    """

    neigh_idx: np.ndarray
    edge_type: np.ndarray
    meta: Dict[str, Any]

    @property
    def ndim(self) -> int:
        return int(self.neigh_idx.ndim)

    @property
    def n_heads(self) -> int:
        return 1 if self.neigh_idx.ndim == 2 else int(self.neigh_idx.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.neigh_idx.shape[-2])

    @property
    def max_degree(self) -> int:
        return int(self.neigh_idx.shape[-1])


def _coerce_np(neigh_idx: np.ndarray | Sequence[Any], edge_type: np.ndarray | Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    ni = np.asarray(neigh_idx)
    et = np.asarray(edge_type)

    if ni.dtype not in (np.int32, np.int64):
        ni = ni.astype(np.int32)
    if et.dtype != np.uint8:
        et = et.astype(np.uint8)

    if ni.shape != et.shape:
        raise ValueError(f"neigh_idx and edge_type must have same shape, got {ni.shape} vs {et.shape}")
    if ni.ndim not in (2, 3):
        raise ValueError(f"Expected [T,D] or [H,T,D], got {ni.shape}")

    return ni, et


def _update_edge_type(edge_map: dict[int, EdgeType], neigh_order: list[int], j: int, edge_type: EdgeType) -> None:
    existing = edge_map.get(j)
    if existing is None:
        edge_map[j] = edge_type
        neigh_order.append(j)
        return
    if EDGE_PRIORITY[edge_type] > EDGE_PRIORITY[existing]:
        edge_map[j] = edge_type


def build_graph_abi_from_adjacency(
    *,
    T: int,
    cycle_adj: Sequence[Sequence[int]],
    window: int,
    landmark_stride: int | None,
    include_self: bool = True,
    rewire_adj: Mapping[int, Sequence[int]] | None = None,
    max_degree: int | None = None,
    cycle_perm: Sequence[int] | None = None,
    all_cycle_perms: Sequence[Sequence[int]] | None = None,
    strategy: str | None = None,
    head_idx: int | None = None,
    num_cycles: int | None = None,
    track_multiplicity: bool = False,
) -> WayfinderGraphABI:
    if T <= 0:
        raise ValueError("T must be positive")
    if len(cycle_adj) != T:
        raise ValueError(f"cycle_adj length must equal T={T}, got {len(cycle_adj)}")
    if window < 0:
        raise ValueError("window must be >= 0")
    if landmark_stride is not None and landmark_stride <= 0:
        raise ValueError("landmark_stride must be > 0 or None")

    rows: list[list[int]] = []
    edge_rows: list[list[int]] = []
    mult_rows: list[list[int]] = [] if track_multiplicity else []
    max_row = 0

    for i in range(T):
        edge_map: dict[int, EdgeType] = {}
        neigh_order: list[int] = []
        mult_count: dict[int, int] = {} if track_multiplicity else {}

        for j in cycle_adj[i]:
            if 0 <= int(j) < T:
                _update_edge_type(edge_map, neigh_order, int(j), EdgeType.CYCLE)
                if track_multiplicity:
                    mult_count[int(j)] = mult_count.get(int(j), 0) + 1

        if window > 0:
            for j in range(max(0, i - window), i):
                _update_edge_type(edge_map, neigh_order, int(j), EdgeType.WINDOW)

        if landmark_stride is not None:
            for j in range(0, i, landmark_stride):
                _update_edge_type(edge_map, neigh_order, int(j), EdgeType.LANDMARK)

        if rewire_adj is not None:
            for j in rewire_adj.get(i, []):
                if 0 <= int(j) < T:
                    _update_edge_type(edge_map, neigh_order, int(j), EdgeType.REWIRE)

        if include_self:
            _update_edge_type(edge_map, neigh_order, i, EdgeType.WINDOW)

        row = neigh_order
        et_row = [int(edge_map[j]) for j in neigh_order]

        if max_degree is not None:
            row = row[:max_degree]
            et_row = et_row[:max_degree]

        rows.append(row)
        edge_rows.append(et_row)
        if track_multiplicity:
            m_row = [mult_count.get(j, 1) for j in row]
            if max_degree is not None:
                m_row = m_row[:max_degree]
            mult_rows.append(m_row)
        max_row = max(max_row, len(row))

    D = max_row if max_degree is None else min(max_row, max_degree)
    if D <= 0:
        D = 1

    neigh_idx = np.full((T, D), -1, dtype=np.int32)
    edge_type = np.full((T, D), int(EdgeType.PAD), dtype=np.uint8)
    multiplicity = np.zeros((T, D), dtype=np.int32) if track_multiplicity else None

    for i, (row, et_row) in enumerate(zip(rows, edge_rows)):
        take = min(D, len(row))
        if take == 0:
            continue
        neigh_idx[i, :take] = np.asarray(row[:take], dtype=np.int32)
        edge_type[i, :take] = np.asarray(et_row[:take], dtype=np.uint8)
        if track_multiplicity:
            multiplicity[i, :take] = np.asarray(  # type: ignore[index]
                mult_rows[i][:take], dtype=np.int32,
            )

    deg = (neigh_idx >= 0).sum(axis=-1)
    meta: Dict[str, Any] = {
        "seq_len": int(T),
        "max_degree": int(D),
        "window": int(window),
        "landmark_stride": None if landmark_stride is None else int(landmark_stride),
        "include_self": bool(include_self),
        "strategy": strategy,
        "head_idx": head_idx,
        "num_cycles": num_cycles,
        "degree_min": int(deg.min()) if deg.size else 0,
        "degree_max": int(deg.max()) if deg.size else 0,
        "degree_mean": float(deg.mean()) if deg.size else 0.0,
    }
    if cycle_perm is not None:
        meta["cycle_perm"] = [int(x) for x in cycle_perm]
    if all_cycle_perms is not None:
        meta["all_cycle_perms"] = [
            [int(x) for x in perm]
            for perm in all_cycle_perms
        ]
    if track_multiplicity and multiplicity is not None:
        meta["multiplicity"] = multiplicity

    return WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)


def stack_head_abis(head_abis: Sequence[WayfinderGraphABI]) -> WayfinderGraphABI:
    if not head_abis:
        raise ValueError("head_abis must be non-empty")

    T = head_abis[0].seq_len
    for abi in head_abis:
        if abi.seq_len != T:
            raise ValueError("All head ABIs must share the same seq_len")

    D = max(abi.max_degree for abi in head_abis)
    H = len(head_abis)

    neigh = np.full((H, T, D), -1, dtype=np.int32)
    etype = np.full((H, T, D), int(EdgeType.PAD), dtype=np.uint8)

    per_head_meta: list[Dict[str, Any]] = []
    cycle_perms: list[list[int] | None] = []
    all_cycle_perms: list[list[list[int]] | None] = []

    for h, abi in enumerate(head_abis):
        ni, et = _coerce_np(abi.neigh_idx, abi.edge_type)
        if ni.ndim != 2:
            raise ValueError("stack_head_abis expects single-head [T,D] ABIs")
        d_h = ni.shape[1]
        neigh[h, :, :d_h] = ni
        etype[h, :, :d_h] = et
        per_head_meta.append(dict(abi.meta))
        all_h = abi.meta.get("all_cycle_perms")
        if isinstance(all_h, list) and all_h:
            norm_all_h = [
                [int(x) for x in perm]
                for perm in all_h
            ]
            all_cycle_perms.append(norm_all_h)
            cycle_perms.append(norm_all_h[0])
        else:
            cperm = abi.meta.get("cycle_perm")
            if isinstance(cperm, list):
                cycle_perms.append([int(x) for x in cperm])
            else:
                cycle_perms.append(None)
            all_cycle_perms.append(None)

    meta: Dict[str, Any] = {
        "n_heads": H,
        "seq_len": T,
        "max_degree": D,
        "heads": per_head_meta,
        "cycle_perms": cycle_perms,
        "all_cycle_perms": all_cycle_perms,
    }
    return WayfinderGraphABI(neigh_idx=neigh, edge_type=etype, meta=meta)


def _iter_heads(abi: WayfinderGraphABI) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    ni, et = _coerce_np(abi.neigh_idx, abi.edge_type)
    if ni.ndim == 2:
        yield ni, et
        return
    for h in range(ni.shape[0]):
        yield ni[h], et[h]


def validate_graph_abi(
    abi: WayfinderGraphABI,
    *,
    expect_heads: int | None = None,
    expect_tokens: int | None = None,
    enforce_hamiltonian: bool = True,
) -> None:
    ni, et = _coerce_np(abi.neigh_idx, abi.edge_type)

    if expect_heads is not None and (1 if ni.ndim == 2 else ni.shape[0]) != expect_heads:
        raise ValueError(f"Expected {expect_heads} heads, got shape {ni.shape}")
    if expect_tokens is not None and ni.shape[-2] != expect_tokens:
        raise ValueError(f"Expected T={expect_tokens}, got shape {ni.shape}")

    T = ni.shape[-2]
    if T <= 0:
        raise ValueError("T must be positive")

    valid_edge_codes = {int(x) for x in EdgeType}
    if not np.isin(et, list(valid_edge_codes)).all():
        bad = np.unique(et[~np.isin(et, list(valid_edge_codes))])
        raise ValueError(f"Invalid edge_type values: {bad.tolist()}")

    valid_range = ((ni == -1) | ((ni >= 0) & (ni < T)))
    if not valid_range.all():
        raise ValueError("neigh_idx has out-of-range entries")

    pad_mask = ni == -1
    if not np.all(et[pad_mask] == int(EdgeType.PAD)):
        raise ValueError("edge_type must be PAD where neigh_idx == -1")
    if np.any((~pad_mask) & (et == int(EdgeType.PAD))):
        raise ValueError("Non-pad neighbors must not use PAD edge_type")

    for head_ni, _head_et in _iter_heads(abi):
        for i in range(T):
            row = head_ni[i]
            vals = row[row >= 0]
            if vals.size != np.unique(vals).size:
                raise ValueError(f"Duplicate neighbors found in row i={i}")

    if enforce_hamiltonian:
        validate_hamiltonian_backbone(abi)


def validate_hamiltonian_backbone(abi: WayfinderGraphABI) -> None:
    T = abi.seq_len
    for head_idx, (ni, et) in enumerate(_iter_heads(abi)):
        cycle_adj: list[list[int]] = [[] for _ in range(T)]
        for i in range(T):
            row = ni[i]
            et_row = et[i]
            cycle_neighbors = [
                int(j) for j, t in zip(row.tolist(), et_row.tolist())
                if j >= 0 and int(t) == int(EdgeType.CYCLE)
            ]
            cycle_adj[i] = cycle_neighbors

            if T == 1:
                continue
            if T == 2:
                if len(cycle_neighbors) < 1:
                    raise ValueError(f"Head {head_idx} token {i} missing cycle edge")
            elif len(cycle_neighbors) < 2:
                raise ValueError(f"Head {head_idx} token {i} has <2 cycle neighbors")

        # Connectivity over cycle edges
        seen = set([0])
        q: deque[int] = deque([0])
        while q:
            u = q.popleft()
            for v in cycle_adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(seen) != T:
            raise ValueError(
                f"Head {head_idx} cycle backbone disconnected: reachable={len(seen)}/{T}"
            )


def graph_metrics(abi: WayfinderGraphABI, *, bfs_hops: int = 4) -> Dict[str, Any]:
    """Minimal substrate visibility metrics."""
    ni, et = _coerce_np(abi.neigh_idx, abi.edge_type)
    if ni.ndim == 2:
        ni = ni[None, ...]
        et = et[None, ...]

    H, T, _D = ni.shape
    nonpad = ni >= 0
    deg = nonpad.sum(axis=-1)

    # shortcut = long-range cycle/rewire edges in original index space
    i_idx = np.arange(T, dtype=np.int32)[None, :, None]
    dist = np.abs(ni - i_idx)
    long_range = (dist > 1) & nonpad

    cycle_or_rewire = (et == int(EdgeType.CYCLE)) | (et == int(EdgeType.REWIRE))
    shortcut_edges = long_range & cycle_or_rewire
    shortcut_rate = float(shortcut_edges.sum()) / float(max(1, cycle_or_rewire.sum()))

    # Reachability proxy: mean unique nodes reachable within bfs_hops from each node
    reach_counts: list[int] = []
    for h in range(H):
        adj = [[] for _ in range(T)]
        for i in range(T):
            row = ni[h, i]
            for j in row[row >= 0].tolist():
                adj[i].append(int(j))
        for src in range(T):
            seen = {src}
            frontier = {src}
            for _ in range(bfs_hops):
                nxt = set()
                for u in frontier:
                    nxt.update(adj[u])
                nxt -= seen
                if not nxt:
                    break
                seen |= nxt
                frontier = nxt
            reach_counts.append(len(seen))

    edge_type_counts = {
        "cycle": int((et == int(EdgeType.CYCLE)).sum()),
        "window": int((et == int(EdgeType.WINDOW)).sum()),
        "landmark": int((et == int(EdgeType.LANDMARK)).sum()),
        "rewire": int((et == int(EdgeType.REWIRE)).sum()),
    }

    return {
        "heads": int(H),
        "tokens": int(T),
        "max_degree": int(ni.shape[-1]),
        "degree_mean": float(deg.mean()),
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "shortcut_rate": shortcut_rate,
        "reachability_proxy": float(np.mean(reach_counts)) if reach_counts else 0.0,
        "edge_type_counts": edge_type_counts,
    }
