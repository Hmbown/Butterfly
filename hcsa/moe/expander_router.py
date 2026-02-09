"""Expander-based MoE routing via bipartite matchings.

Core insight: MoE routing (assigning N tokens to E experts) is a
bipartite matching problem.  If the assignment graph is an expander,
Hall's condition is satisfied with margin and simple greedy matching
achieves near-perfect load balance.

Three routers:
  ExpanderRouter       — threshold-based with spectral quality check
  CyclicMatchingRouter — union of K random matchings = expander w.h.p.
  moe_load_balance_via_expansion — standalone function
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "BipartiteExpanderCheck",
    "CyclicMatchingRouter",
    "ExpanderRouter",
    "RoutingResult",
    "moe_load_balance_via_expansion",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Result of MoE routing."""
    assignments: np.ndarray  # [B, T] or [T] — expert index per token
    load: np.ndarray         # [B, E] or [E] — tokens per expert
    load_balance_loss: float  # CV of load (0 = perfect)
    metrics: dict = field(default_factory=dict)


def _load_balance_loss(load: np.ndarray) -> float:
    """Coefficient of variation of expert load. 0 = perfect balance."""
    if load.size == 0:
        return 0.0
    mean = float(load.mean())
    if mean < 1e-12:
        return 0.0
    return float(load.std() / mean)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def _greedy_bipartite_matching(
    scores: np.ndarray,
    capacity: int,
) -> np.ndarray:
    """Priority-greedy matching: highest-score tokens pick first.

    Args:
        scores: [N, E] affinity scores (higher = better).
        capacity: max tokens per expert.

    Returns:
        assignments: [N] int array, expert index (-1 if unassigned).
    """
    N, E = scores.shape
    assignments = np.full(N, -1, dtype=np.int64)
    expert_load = np.zeros(E, dtype=np.int64)

    # sort tokens by their best score (descending)
    best_scores = scores.max(axis=1)
    order = np.argsort(-best_scores)

    for idx in order:
        # try experts in order of preference for this token
        pref = np.argsort(-scores[idx])
        for e in pref:
            if expert_load[e] < capacity:
                assignments[idx] = int(e)
                expert_load[e] += 1
                break

    return assignments


# ---------------------------------------------------------------------------
# BipartiteExpanderCheck
# ---------------------------------------------------------------------------

class BipartiteExpanderCheck:
    """Utility for checking expander quality of bipartite graphs."""

    @staticmethod
    def spectral_gap(biadj: np.ndarray) -> float:
        """Spectral gap: sigma_1 - sigma_2 of the biadjacency matrix.

        Larger gap → better bipartite expansion.
        """
        B = np.asarray(biadj, dtype=np.float64)
        if B.size == 0:
            return 0.0
        svs = np.linalg.svd(B, compute_uv=False)
        svs = np.sort(svs)[::-1]
        if len(svs) < 2:
            return float(svs[0]) if len(svs) else 0.0
        return float(svs[0] - svs[1])

    @staticmethod
    def is_expander(
        biadj: np.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """Is the bipartite spectral gap above threshold?"""
        return BipartiteExpanderCheck.spectral_gap(biadj) >= threshold

    @staticmethod
    def halls_margin(biadj: np.ndarray) -> float:
        """Estimate min (|N(S)| - |S|) / |S| over left-vertex subsets.

        Exact for N <= 16, sampled for larger inputs.
        """
        B = np.asarray(biadj, dtype=np.float64)
        N = B.shape[0]
        if N == 0:
            return 0.0

        def neighborhood_size(subset_mask: np.ndarray) -> int:
            cols_hit = (B[subset_mask] > 0).any(axis=0)
            return int(cols_hit.sum())

        min_margin = float("inf")
        if N <= 16:
            # enumerate all non-empty subsets
            for bits in range(1, 1 << N):
                mask = np.array(
                    [(bits >> i) & 1 for i in range(N)], dtype=bool
                )
                sz = int(mask.sum())
                if sz > N // 2:
                    continue
                ns = neighborhood_size(mask)
                margin = (ns - sz) / sz
                min_margin = min(min_margin, margin)
        else:
            rng = np.random.default_rng()
            for _ in range(500):
                sz = rng.integers(1, max(2, N // 2 + 1))
                idxs = rng.choice(N, size=sz, replace=False)
                mask = np.zeros(N, dtype=bool)
                mask[idxs] = True
                ns = neighborhood_size(mask)
                margin = (ns - int(mask.sum())) / int(mask.sum())
                min_margin = min(min_margin, margin)

        return float(min_margin) if np.isfinite(min_margin) else 0.0


# ---------------------------------------------------------------------------
# ExpanderRouter
# ---------------------------------------------------------------------------

class ExpanderRouter:
    """MoE router with expander-quality bipartite assignment graph.

    1. Softmax gating logits → scores
    2. Build allowed-assignment bipartite graph via top-p threshold
    3. Check spectral gap; augment with random edges if poor
    4. Greedy matching on the subgraph
    """

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.2,
        top_p: float = 0.9,
        min_edges_per_token: int = 2,
        spectral_threshold: float = 0.5,
        augment_random_edges: int = 2,
        seed: int | None = None,
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.top_p = top_p
        self.min_edges = min_edges_per_token
        self.spectral_threshold = spectral_threshold
        self.augment_random_edges = augment_random_edges
        self.rng = np.random.default_rng(seed)

    def route(self, gating_logits: np.ndarray) -> RoutingResult:
        """Route tokens to experts.

        Args:
            gating_logits: [B, T, E] or [T, E] raw logits.

        Returns:
            RoutingResult with assignments and metrics.
        """
        logits = np.asarray(gating_logits, dtype=np.float64)
        single_batch = logits.ndim == 2
        if single_batch:
            logits = logits[None]

        B, T, E = logits.shape
        capacity = int(self.capacity_factor * T / E)

        all_assign = np.full((B, T), -1, dtype=np.int64)
        all_load = np.zeros((B, E), dtype=np.int64)

        for b in range(B):
            scores = _softmax(logits[b])
            allowed = self._build_allowed_graph(scores)
            gap = BipartiteExpanderCheck.spectral_gap(
                allowed.astype(np.float64)
            )

            if gap < self.spectral_threshold:
                allowed = self._augment(allowed)
                gap = BipartiteExpanderCheck.spectral_gap(
                    allowed.astype(np.float64)
                )

            masked = np.where(allowed, scores, -np.inf)
            assigns = _greedy_bipartite_matching(masked, capacity)
            load = np.bincount(
                assigns[assigns >= 0], minlength=E
            ).astype(np.int64)

            all_assign[b] = assigns
            all_load[b] = load

        if single_batch:
            all_assign = all_assign[0]
            all_load = all_load[0]

        return RoutingResult(
            assignments=all_assign,
            load=all_load,
            load_balance_loss=_load_balance_loss(all_load),
        )

    def _build_allowed_graph(self, scores: np.ndarray) -> np.ndarray:
        """Build bipartite allowed-assignment graph via top-p."""
        N, E = scores.shape
        allowed = np.zeros((N, E), dtype=bool)

        for i in range(N):
            order = np.argsort(-scores[i])
            cumsum = np.cumsum(scores[i, order])
            cutoff = np.searchsorted(cumsum, self.top_p) + 1
            cutoff = max(cutoff, self.min_edges)
            cutoff = min(cutoff, E)
            allowed[i, order[:cutoff]] = True

        return allowed

    def _augment(self, allowed: np.ndarray) -> np.ndarray:
        """Add random edges to boost expansion."""
        n = allowed.shape[0]
        aug = allowed.copy()
        for i in range(n):
            disallowed = np.where(~aug[i])[0]
            if disallowed.size == 0:
                continue
            k = min(self.augment_random_edges, disallowed.size)
            chosen = self.rng.choice(disallowed, size=k, replace=False)
            aug[i, chosen] = True
        return aug


# ---------------------------------------------------------------------------
# CyclicMatchingRouter
# ---------------------------------------------------------------------------

class CyclicMatchingRouter:
    """MoE router via union of K random perfect matchings.

    Theorem: K >= 3 random perfect matchings of a bipartite graph
    form an expander w.h.p.  This gives perfect load balance by
    construction — no auxiliary loss needed.
    """

    def __init__(
        self,
        num_experts: int,
        num_matchings: int = 5,
        seed: int | None = None,
    ):
        self.num_experts = num_experts
        self.num_matchings = num_matchings
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, list[np.ndarray]] = {}

    def _generate_matchings(self, T: int) -> list[np.ndarray]:
        """Generate K random perfect matchings between T slots and E experts."""
        E = self.num_experts
        matchings = []
        for _ in range(self.num_matchings):
            # build balanced slot pool
            slots = np.tile(np.arange(E), T // E + 1)[:T]
            self.rng.shuffle(slots)
            matchings.append(slots.copy())
        return matchings

    def _get_matchings(self, T: int) -> list[np.ndarray]:
        if T not in self._cache:
            self._cache[T] = self._generate_matchings(T)
        return self._cache[T]

    def regenerate_matchings(self, T: int) -> None:
        """Force regeneration of cached matchings."""
        self._cache[T] = self._generate_matchings(T)

    def route(self, gating_logits: np.ndarray) -> RoutingResult:
        """Pick the matching best aligned with gating scores."""
        logits = np.asarray(gating_logits, dtype=np.float64)
        single_batch = logits.ndim == 2
        if single_batch:
            logits = logits[None]

        B, T, E = logits.shape
        matchings = self._get_matchings(T)

        all_assign = np.full((B, T), -1, dtype=np.int64)
        all_load = np.zeros((B, E), dtype=np.int64)

        for b in range(B):
            scores = _softmax(logits[b])
            # score each matching by alignment
            best_score = -np.inf
            best_m = matchings[0]
            for m in matchings:
                alignment = sum(
                    float(scores[i, m[i]]) for i in range(T)
                )
                if alignment > best_score:
                    best_score = alignment
                    best_m = m

            all_assign[b] = best_m
            all_load[b] = np.bincount(best_m, minlength=E)

        if single_batch:
            all_assign = all_assign[0]
            all_load = all_load[0]

        return RoutingResult(
            assignments=all_assign,
            load=all_load,
            load_balance_loss=_load_balance_loss(all_load),
        )


# ---------------------------------------------------------------------------
# Standalone function
# ---------------------------------------------------------------------------

def moe_load_balance_via_expansion(
    gating_scores: np.ndarray,
    num_experts: int,
    capacity: int,
    *,
    spectral_threshold: float = 1.0,
    max_augment_rounds: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Standalone expander-based routing.

    1. Build top-2 assignment graph
    2. Check spectral gap
    3. Augment with random edges until expansion is good
    4. Greedy matching

    Returns (assignments [N], metrics dict).
    """
    rng = rng or np.random.default_rng()
    scores = np.asarray(gating_scores, dtype=np.float64)
    N, E = scores.shape

    # top-2 per token
    allowed = np.zeros((N, E), dtype=bool)
    for i in range(N):
        top2 = np.argsort(-scores[i])[:2]
        allowed[i, top2] = True

    gap = BipartiteExpanderCheck.spectral_gap(
        allowed.astype(np.float64)
    )

    rounds = 0
    while gap < spectral_threshold and rounds < max_augment_rounds:
        for i in range(N):
            off = np.where(~allowed[i])[0]
            if off.size > 0:
                allowed[i, rng.choice(off)] = True
        gap = BipartiteExpanderCheck.spectral_gap(
            allowed.astype(np.float64)
        )
        rounds += 1

    masked = np.where(allowed, scores, -np.inf)
    assignments = _greedy_bipartite_matching(masked, capacity)
    load = np.bincount(assignments[assignments >= 0], minlength=E)

    return assignments, {
        "spectral_gap": float(gap),
        "augment_rounds": rounds,
        "load_balance_loss": _load_balance_loss(load),
        "load": load.tolist(),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    # Quick demo
    T, E = 16, 4
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((T, E))

    print("=== CyclicMatchingRouter ===")
    cmr = CyclicMatchingRouter(E, num_matchings=5, seed=42)
    res = cmr.route(logits)
    print(f"  Load: {res.load}")
    print(f"  Balance loss (CV): {res.load_balance_loss:.4f}")

    print("\n=== ExpanderRouter ===")
    er = ExpanderRouter(E, seed=42)
    res = er.route(logits)
    print(f"  Load: {res.load}")
    print(f"  Balance loss (CV): {res.load_balance_loss:.4f}")

    print("\n=== Standalone ===")
    scores = _softmax(logits)
    assigns, metrics = moe_load_balance_via_expansion(
        scores, E, capacity=6, rng=rng,
    )
    print(f"  Gap: {metrics['spectral_gap']:.3f}")
    print(f"  Rounds: {metrics['augment_rounds']}")
    print(f"  Balance loss: {metrics['load_balance_loss']:.4f}")
