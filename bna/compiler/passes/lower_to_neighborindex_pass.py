from __future__ import annotations

from typing import Any, Dict

import numpy as np

from bna.graph_strategies import build_strategy
from bna.compiler.graph_ir import GraphIR
from bna.graph.abi import WayfinderGraphABI, graph_metrics, stack_head_abis, validate_graph_abi


def lower_to_neighborindex_pass(ir: GraphIR, *, T: int, H: int) -> Dict[str, Any]:
    head_abis: list[WayfinderGraphABI] = []
    for h in range(H):
        strategy = build_strategy(
            ir.strategy,
            num_cycles=ir.num_cycles,
            seed=ir.seed + 7919 * h,
        )
        abi_h = strategy.build(
            T=T,
            r=None,
            head_idx=h,
            window=ir.window_size,
            landmark_stride=ir.landmark_stride,
            include_self=True,
        )
        head_abis.append(abi_h)

    abi = stack_head_abis(head_abis)
    validate_graph_abi(abi, expect_heads=H, expect_tokens=T, enforce_hamiltonian=True)

    neigh_idx = np.asarray(abi.neigh_idx, dtype=np.int32)
    edge_type = np.asarray(abi.edge_type, dtype=np.uint8)

    return {
        "abi": abi,
        "neigh_idx": neigh_idx,
        "edge_type": edge_type,
        "graph_metrics": graph_metrics(abi),
    }
