from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from bna.graph.abi import WayfinderGraphABI, stack_head_abis, validate_graph_abi
from bna.graph_strategies import build_strategy


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class TopologyGraph:
    """First-class graph object used by attention backends."""

    abi: WayfinderGraphABI
    source: str = "runtime"
    artifact_dir: str | None = None
    notes: Dict[str, Any] = field(default_factory=dict)


class Topology:
    """Topology runtime responsible for graph construction and persistence."""

    def __init__(
        self,
        *,
        n_heads: int,
        strategy: str = "random",
        num_cycles: int = 1,
        edge_disjoint: bool = True,
        regular_num_clusters: int = 8,
        seed: int = 0,
        window: int = 64,
        landmark_stride: Optional[int] = 64,
        enforce_hamiltonian: bool = True,
    ):
        self.n_heads = int(n_heads)
        self.strategy = str(strategy)
        self.num_cycles = int(num_cycles)
        self.edge_disjoint = bool(edge_disjoint)
        self.regular_num_clusters = int(max(1, regular_num_clusters))
        self.seed = int(seed)
        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.enforce_hamiltonian = bool(enforce_hamiltonian)
        self._strategies = [self._make_strategy(h) for h in range(self.n_heads)]

    @property
    def cache_mode(self) -> str:
        return "static" if self.strategy in {"random", "regular_partition"} else "dynamic"

    def _make_strategy(self, head_idx: int):
        if self.strategy == "random":
            return build_strategy(
                "random",
                num_cycles=self.num_cycles,
                edge_disjoint=self.edge_disjoint,
                seed=self.seed + 7919 * head_idx,
            )
        if self.strategy == "greedy":
            return build_strategy("greedy", num_cycles=self.num_cycles)
        if self.strategy == "online_insertion":
            return build_strategy("online_insertion", seed=self.seed + 7919 * head_idx)
        if self.strategy == "regular_partition":
            return build_strategy(
                "regular_partition",
                num_clusters=self.regular_num_clusters,
                num_cycles=self.num_cycles,
                seed=self.seed + 7919 * head_idx,
            )
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _normalize_routing(
        self,
        *,
        routing_by_head: Optional[Sequence[Any]],
    ) -> List[Any | None]:
        if routing_by_head is None:
            return [None for _ in range(self.n_heads)]
        if len(routing_by_head) != self.n_heads:
            raise ValueError(
                f"routing_by_head must have length {self.n_heads}, got {len(routing_by_head)}"
            )
        return [routing_by_head[h] for h in range(self.n_heads)]

    def construct_abi(
        self,
        *,
        T: int,
        routing_by_head: Optional[Sequence[Any]] = None,
        include_self: bool = True,
    ) -> WayfinderGraphABI:
        t = int(T)
        if t <= 0:
            raise ValueError(f"T must be > 0, got {t}")

        routing = self._normalize_routing(routing_by_head=routing_by_head)
        head_abis: list[WayfinderGraphABI] = []
        for h in range(self.n_heads):
            abi_h = self._strategies[h].build(
                T=t,
                r=routing[h],
                head_idx=h,
                window=self.window,
                landmark_stride=self.landmark_stride,
                include_self=bool(include_self),
            )
            head_abis.append(abi_h)

        abi = stack_head_abis(head_abis)
        validate_graph_abi(
            abi,
            expect_heads=self.n_heads,
            expect_tokens=t,
            enforce_hamiltonian=self.enforce_hamiltonian,
        )
        return abi

    def construct_perms_only(self, T: int) -> TopologyGraph:
        """Fast path: generate only cycle permutations, skip full ABI.

        Returns a minimal ``TopologyGraph`` whose ``abi`` has D=0
        (no neighbor indices / edge types) but carries per-head
        permutation lists in ``meta["cycle_perms"]`` and
        ``meta["all_cycle_perms"]``.  This is ~100x faster than
        ``construct()`` for large T because it avoids the O(T*D)
        Python loop in ``build_graph_abi_from_adjacency``.
        """
        t = int(T)
        if t <= 0:
            raise ValueError(f"T must be > 0, got {t}")

        cycle_perms: list[list[int] | None] = []
        all_cycle_perms: list[list[list[int]] | None] = []

        for h in range(self.n_heads):
            perms = self._strategies[h]._sample_perms(t)
            all_h = [p.detach().cpu().tolist() for p in perms]
            all_cycle_perms.append(all_h)
            cycle_perms.append(all_h[0] if all_h else None)

        neigh_idx = np.zeros((self.n_heads, t, 0), dtype=np.int32)
        edge_type = np.zeros((self.n_heads, t, 0), dtype=np.uint8)

        meta: Dict[str, Any] = {
            "n_heads": self.n_heads,
            "seq_len": t,
            "max_degree": 0,
            "window": self.window,
            "landmark_stride": self.landmark_stride,
            "strategy": self.strategy,
            "num_cycles": self.num_cycles,
            "cycle_perms": cycle_perms,
            "all_cycle_perms": all_cycle_perms,
        }

        abi = WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)
        return TopologyGraph(
            abi=abi,
            source="runtime_perms_only",
            notes={
                "strategy": self.strategy,
                "window": self.window,
                "landmark_stride": self.landmark_stride,
                "num_cycles": self.num_cycles,
                "edge_disjoint": self.edge_disjoint,
                "regular_num_clusters": self.regular_num_clusters,
            },
        )

    def construct(
        self,
        x: int | Mapping[str, Any],
        *,
        routing_by_head: Optional[Sequence[Any]] = None,
        include_self: bool = True,
    ) -> TopologyGraph:
        """Build a first-class graph from sequence length or a query mapping."""
        if isinstance(x, Mapping):
            if "T" in x:
                t = int(x["T"])
            elif "seq_len" in x:
                t = int(x["seq_len"])
            else:
                raise ValueError("Topology.construct mapping must include 'T' or 'seq_len'")
            if routing_by_head is None and "routing_by_head" in x:
                maybe_routing = x["routing_by_head"]
                if isinstance(maybe_routing, Sequence):
                    routing_by_head = maybe_routing  # type: ignore[assignment]
            include_self = bool(x.get("include_self", include_self))
        else:
            t = int(x)

        abi = self.construct_abi(
            T=t,
            routing_by_head=routing_by_head,
            include_self=include_self,
        )
        return TopologyGraph(
            abi=abi,
            source="runtime",
            notes={
                "strategy": self.strategy,
                "window": self.window,
                "landmark_stride": self.landmark_stride,
                "num_cycles": self.num_cycles,
                "edge_disjoint": self.edge_disjoint,
                "regular_num_clusters": self.regular_num_clusters,
            },
        )

    def rewire(
        self,
        query: Mapping[str, Any],
        *,
        graph: Optional[TopologyGraph] = None,
    ) -> TopologyGraph:
        """Rebuild graph from a query; placeholder for future query-driven rewiring."""
        t = int(query.get("T", query.get("seq_len", graph.abi.seq_len if graph else 0)))
        if t <= 0:
            raise ValueError("rewire requires T/seq_len or an existing graph")
        routing_by_head = query.get("routing_by_head")
        include_self = bool(query.get("include_self", True))
        out = self.construct(
            {"T": t, "include_self": include_self},
            routing_by_head=routing_by_head,
            include_self=include_self,
        )
        return TopologyGraph(
            abi=out.abi,
            source="rewire",
            notes={"query": _jsonable(dict(query))},
        )

    def save(self, graph: TopologyGraph | WayfinderGraphABI, path: str | Path) -> Path:
        abi = graph.abi if isinstance(graph, TopologyGraph) else graph
        out = Path(path)
        if out.suffix.lower() == ".npz":
            npz_path = out
            meta_path = out.with_suffix(".json")
            npz_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            npz_path = out / "neighborindex.npz"
            meta_path = out / "meta.json"

        np.savez(
            npz_path,
            neigh_idx=np.asarray(abi.neigh_idx, dtype=np.int32),
            edge_type=np.asarray(abi.edge_type, dtype=np.uint8),
        )
        meta_path.write_text(json.dumps(_jsonable(dict(abi.meta)), indent=2), encoding="utf-8")
        return npz_path

    def load(
        self,
        path: str | Path,
        *,
        expect_heads: Optional[int] = None,
        expect_tokens: Optional[int] = None,
    ) -> TopologyGraph:
        src = Path(path)
        if src.is_dir():
            npz_path = src / "neighborindex.npz"
            meta_path = src / "meta.json"
        else:
            npz_path = src
            meta_path = src.with_suffix(".json")

        payload = np.load(npz_path)
        neigh_idx = np.asarray(payload["neigh_idx"], dtype=np.int32)
        edge_type = np.asarray(payload["edge_type"], dtype=np.uint8)
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}

        h = int(expect_heads or self.n_heads)
        if neigh_idx.ndim == 2:
            neigh_idx = np.broadcast_to(neigh_idx[None, :, :], (h, *neigh_idx.shape)).copy()
            edge_type = np.broadcast_to(edge_type[None, :, :], (h, *edge_type.shape)).copy()

        abi = WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)
        validate_graph_abi(
            abi,
            expect_heads=expect_heads or self.n_heads,
            expect_tokens=expect_tokens,
            enforce_hamiltonian=self.enforce_hamiltonian,
        )
        return TopologyGraph(
            abi=abi,
            source="compiled",
            artifact_dir=str(src if src.is_dir() else src.parent),
        )
