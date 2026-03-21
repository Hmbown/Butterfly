"""Expander-based MoE (Mixture of Experts) routing.

This module applies the same combinatorial toolkit used in HCSA sparse
attention -- Hamiltonian cycles, expander graphs, spectral analysis --
to the MoE routing problem.

MoE routing is fundamentally a bipartite matching problem:
  - Left side: tokens (B * T positions)
  - Right side: experts (E experts, each with capacity C)
  - Edges: gating scores from a learned router
  - Goal: find a good matching that balances load while respecting affinity

Expander theory connects these two domains:
  1. Sparse attention via Hamiltonian cycles -> expander attention graphs
  2. MoE routing via bipartite expanders -> load-balanced routing

The key classes and functions:

  ExpanderRouter
    Threshold-based routing that builds a bipartite assignment graph from
    gating scores, checks its expansion quality, optionally augments with
    random edges to boost expansion, and routes via greedy matching.

  CyclicMatchingRouter
    Loss-free router that maintains K random perfect matchings.  At each
    forward pass, selects the matching best aligned with gating scores.
    K >= 3 random matchings form a bipartite expander w.h.p., guaranteeing
    near-perfect load balance without auxiliary losses.

  moe_load_balance_via_expansion
    Standalone function: given gating scores, build the assignment graph,
    measure expansion, augment if poor, and route.

  BipartiteExpanderCheck
    Spectral and combinatorial diagnostics for bipartite graphs:
    spectral gap, Hall's margin, vertex expansion.

  RoutingResult
    Dataclass holding assignments, load, loss, and metrics.
"""

from .expander_router import (
    BipartiteExpanderCheck,
    CyclicMatchingRouter,
    ExpanderRouter,
    RoutingResult,
    moe_load_balance_via_expansion,
)

__all__ = [
    "BipartiteExpanderCheck",
    "CyclicMatchingRouter",
    "ExpanderRouter",
    "RoutingResult",
    "moe_load_balance_via_expansion",
]
