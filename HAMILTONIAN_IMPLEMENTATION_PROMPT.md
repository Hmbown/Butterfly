# Actually Hamiltonian: Closing the Cycle, Unifying the Multigraph, and Principled Expansion

You are the AI/ML implementation lead for the HCSA (Hamiltonian Cycle Sparse Attention) project.

**Working directory**: `/Volumes/VIXinSSD/wayfinder`

**Companion document**: `KERNEL_DISCOVERY_EXECUTION_PROMPT.md` covers the Metal kernel discovery work (K4 active-row fused, K1 permute-window fused). That prompt fixes the **performance** problem (chunked-prefill dense fallback). This prompt fixes the **mathematical** problem (the cycle isn't actually a cycle).

**Read first** (in order):
1. `CLAUDE.md` — project overview, architecture, commands
2. `AGENTS.md` — mandatory sub-agent policy, Bell Labs notebook protocol
3. This file — the implementation specification
4. `notes/LAB_NOTEBOOK.md` — experiment history (append-only)
5. `notes/experiments.ndjson` — machine-readable experiment log (append-only)

**Non-negotiable constraints**:
- Follow `AGENTS.md` exactly (sub-agents, Bell Labs protocol).
- Every code change gets a hypothesis entry BEFORE and a result entry AFTER in both `notes/LAB_NOTEBOOK.md` and `notes/experiments.ndjson`.
- Keep Hamiltonian path active (`path=permute`). Do not redesign away from Hamiltonian.
- Retro/backfill OFF by default for inference.
- Must use `python3` not `python` on this system.
- `ruff` line-length = 100 chars.
- Run `pytest` after each code change to verify no regressions.
- All existing tests must continue to pass. New behavior defaults to OFF (opt-in via flags).

---

## Executive Summary

HCSA claims to use **Hamiltonian cycles** for sparse attention, but three critical bugs/design gaps mean the system is not actually Hamiltonian in practice:

| Problem | Severity | Location |
|---------|----------|----------|
| **Linear clamping** instead of circular wrap-around | HIGH | `attention.py` lines 302-304, 689-690, 1011-1013 |
| **Multi-cycle averaging** instead of union-graph single pass | MEDIUM | `attention.py` lines 509-553, 904-940, 407-434 |
| **Ad-hoc d selection** instead of principled `d = O(log T)` | LOW | `num_cycles` param across codebase, always user-specified |

This prompt specifies three staged changes to fix all three, plus integration of existing Laplacian/Cheeger diagnostics and Fiedler-bridge rewiring.

---

## Problem 1: The Cycle Is Not Closed (Linear Clamping)

### What's wrong

A Hamiltonian cycle on T vertices has the defining property that vertex `perm[T-1]` connects back to `perm[0]` — the **wrap-around edge**. This is the single structural feature that distinguishes a Hamiltonian cycle from a Hamiltonian path.

The current permute-window kernel uses **linear clamping** (`clip(0, T-1)`) to handle boundary positions. When a query at permuted position `i` looks at its window `[i-w, i+w]`, positions that would be negative or >= T are clamped to 0 or T-1 respectively. This means:

- Position `perm[0]` cannot see `perm[T-1]` (its cycle predecessor)
- Position `perm[T-1]` cannot see `perm[0]` (its cycle successor)
- The wrap-around edge — the defining feature of a cycle vs. a path — is **silently discarded**

The attention graph is actually a Hamiltonian **path** with clamped boundary duplicates, not a Hamiltonian cycle.

### Exact locations

**Function 1: `permute_cycle_window_attention_single`** (`hcsa/mlx/attention.py` lines 300-304)

```python
# Current (BROKEN — linear clamping, discards wrap-around):
W = 2 * window + 1
offsets = mx.arange(-window, window + 1, dtype=mx.int32)
pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
valid = (pi_idx >= 0) & (pi_idx < T)
pi_idx_clamped = mx.clip(pi_idx, 0, T - 1)
```

Should be:
```python
# Fixed — circular wrap-around, cycle edge preserved:
W = 2 * window + 1
offsets = mx.arange(-window, window + 1, dtype=mx.int32)
pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
pi_idx_wrapped = pi_idx % T
valid = mx.ones((T, W), dtype=mx.bool_)  # All positions valid under circular wrap
```

**Function 2: `wayfinder_permute_window_attention_batched`** (`hcsa/mlx/attention.py` lines 689-690, 714-718)

```python
# Current (BROKEN — linear clamping on chunk boundaries):
ks = max(0, s - window)
ke = min(T, e + window)
# ...later...
q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, e - s, 1)
k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, ke - ks)
rel = k_pos - q_pos
in_window = (rel >= -window) & (rel <= window)
```

The chunk boundary logic (`ks = max(0, s - window)`, `ke = min(T, e + window)`) clips at array boundaries instead of wrapping. For the first chunk (`s=0`), `ks = max(0, 0 - window) = 0` — the window doesn't wrap to see the end of the array. For the last chunk, `ke = min(T, ...)` — the window doesn't wrap to see the beginning.

**Function 3: `wayfinder_permute_window_attention_active_batched`** (`hcsa/mlx/attention.py` lines 1011-1013)

```python
# Current (BROKEN — linear clamping):
k_rank = q_rank.reshape(-1, 1) + offsets  # [Qblk, W]
valid = (k_rank >= 0) & (k_rank < Tg)
k_rank_clipped = mx.clip(k_rank, 0, Tg - 1).astype(mx.int32)
```

Should be:
```python
# Fixed — circular wrap-around:
k_rank = q_rank.reshape(-1, 1) + offsets  # [Qblk, W]
k_rank_wrapped = k_rank % Tg
valid = mx.ones_like(k_rank, dtype=mx.bool_)  # All valid under wrap
```

### Causality is preserved

The cycle is **undirected** — every edge `(perm[i], perm[(i+1) % T])` exists regardless of order. Causality is enforced separately via the `causal = orig_k <= orig_q` mask (line 721 in batched, line 1018 in active). The causal mask operates on **original token indices**, not permuted indices, so circular wrapping does not break causality:

- If `perm[0] = 47` and `perm[T-1] = 12`, then position 47 can see position 12 (causal: 12 <= 47) but position 12 cannot see position 47 (acausal: 47 > 12). This is correct.

### Implementation approach

**Phase 1a: Add `circular` flag (default `False` for backward compat)**

Add a `circular: bool = False` parameter to all three functions. When `True`, use `% T` instead of `clip`. When `False`, preserve existing behavior exactly.

**Phase 1b: Fix `permute_cycle_window_attention_single`**

Simplest function. When `circular=True`:
- Replace `pi_idx_clamped = mx.clip(pi_idx, 0, T - 1)` with `pi_idx_wrapped = pi_idx % T`
- Replace `valid = (pi_idx >= 0) & (pi_idx < T)` with `valid = mx.ones((T, W), dtype=mx.bool_)`
- The causal mask (`perm_mx[pi_idx_wrapped] <= perm_mx[mx.arange(T).reshape(T,1)]`) still applies

**Phase 1c: Fix `wayfinder_permute_window_attention_batched`**

The chunked batched path is more complex because chunks are contiguous slices of the permuted sequence. For circular wrapping at chunk boundaries:

Option A (recommended): **Circular padding of the pre-permuted K/V buffers**. Before chunking, create a circularly-padded version of K_pi and V_pi:
```python
# Pad K_pi circularly: append first `window` elements to end, last `window` to beginning
k_pi_padded = mx.concatenate([k_pi[:, :, -window:, :], k_pi, k_pi[:, :, :window, :]], axis=2)
v_pi_padded = mx.concatenate([v_pi[:, :, -window:, :], v_pi, v_pi[:, :, :window, :]], axis=2)
# Now k_pi_padded has shape [B, hc, T + 2*window, dh]
# Chunk slicing becomes: ks = s, ke = e + 2*window (no clamping needed)
```

Then adjust the chunk boundary logic:
```python
# Old: ks = max(0, s - window), ke = min(T, e + window)
# New: ks = s, ke = e + 2*window (offset by the prepended window)
```

And the position arrays and causal mask need to reference the original (unwrapped) permutation indices for proper causality.

Option B: **Wrap-around index computation**. Replace `ks = max(0, s-window)` with modular arithmetic. More complex, more error-prone.

**Phase 1d: Fix `wayfinder_permute_window_attention_active_batched`**

Active-row path. Replace clip with modular:
```python
k_rank_wrapped = k_rank % Tg
# Remove: valid = (k_rank >= 0) & (k_rank < Tg)
# Replace with: valid = mx.ones_like(k_rank, dtype=mx.bool_)
```

Keep the `available = k_orig < Tk` check (ensures we don't look at positions beyond the current KV cache length during generation).

### Test requirements (Phase 1)

1. **Wrap-around edge test**: Construct a known permutation where `perm[0]=5, perm[T-1]=3`. Verify that with `circular=True`, token 5 attends to token 3 (and vice versa, subject to causality). Verify with `circular=False`, this edge is missing.

2. **Numerical equivalence**: For `circular=False`, output must be **identical** to current code (bit-for-bit).

3. **Causality test**: With `circular=True`, verify that no token attends to a future token. Use the existing `tests/pytorch/test_causality.py` pattern.

4. **Degree count test**: With `circular=True`, every permuted position should have exactly `min(2*window+1, T)` valid neighbors (no boundary dropout). With `circular=False`, boundary positions have fewer.

5. **All three functions agree**: For the same inputs and `circular=True`, `permute_cycle_window_attention_single`, `wayfinder_permute_window_attention_batched` (with query_chunk_size=T to avoid chunking), and `wayfinder_permute_window_attention_active_batched` (with query_positions = all positions) produce the same output.

---

## Problem 2: Multi-Cycle Averaging Discards Multigraph Structure

### What's wrong

When `num_cycles > 1` (i.e., `all_perms.ndim == 3` with shape `[H, d, T]`), all three paths run `d` independent attention passes and average:

```python
# wayfinder_permute_window_attention_batched, lines 509-553:
d = int(all_perms.shape[1])
ys: list[mx.array] = []
for c in range(d):
    y_c, _ = wayfinder_permute_window_attention_batched(
        q, k, v, all_perms=all_perms[:, c, :], all_inv_perms=all_inv_perms[:, c, :], ...
    )
    ys.append(y_c.astype(mx.float32))
return mx.mean(mx.stack(ys, axis=0), axis=0).astype(v.dtype), None
```

Same pattern in `wayfinder_permute_window_attention_active_batched` (lines 904-940) and `permute_cycle_window_attention` (lines 407-434).

This is wasteful:
- Each pass runs a full attention computation: O(d * T * W) total
- The averaging destroys attention weight structure (a token might attend strongly to neighbor X in cycle 1 and neighbor Y in cycle 2, but averaging smooths both out)
- Edge multiplicity information is lost (if two cycles share an edge, that edge should be **stronger**, not averaged away)

### Proposed fix: Union multigraph with multiplicity bias

Instead of d separate passes, build a **single attention graph** whose neighbor set is the **union** of all d cycle windows, with edge multiplicity tracked as a bias term.

**Approach:**

For each query position `i` in original space, compute the set of keys reachable through any of the d cycles:

```
neighbors(i) = UNION over c in [0,d): { perm_c[j] : |inv_perm_c[i] - j| <= window (mod T) }
```

For each neighbor `j` in this union, compute its **multiplicity** `m(i,j)` = number of cycles where `j` appears in `i`'s window. Add `log(m(i,j))` or `alpha * m(i,j)` as a bias to the attention logit for that entry.

**Why this is better:**
- Single attention pass: O(T * W_union) where W_union <= d * W (usually much less due to overlap)
- Multiplicity bias preserves the information that multiply-connected neighbors are structurally important
- Softmax operates over the full union, not per-cycle

**Implementation plan:**

1. Add a utility function `build_union_multigraph_index(all_perms, all_inv_perms, window, T)` that returns:
   - `union_neigh_idx: [H, T, D_union]` — neighbor indices for the union graph
   - `multiplicity: [H, T, D_union]` — edge multiplicity per neighbor
   - `valid_mask: [H, T, D_union]` — padding mask

2. Modify the multi-cycle branch to call this utility, then run a single gather-based attention pass on the union graph (reusing `sparse_gather_attention` logic).

3. The `build_graph_abi_from_adjacency` function in `hcsa/graph/abi.py` already builds adjacency from cycle edges but **discards multiplicity** (line 116-118: `_update_edge_type` overwrites duplicate entries by priority). Add an optional `track_multiplicity=True` flag that counts how many times each neighbor appears.

**Backward compatibility:**
- New flag `multi_cycle_mode: Literal["average", "union"] = "average"` on the attention functions
- Default `"average"` preserves current behavior exactly
- `"union"` enables the new path
- The `"union"` path should produce output within reasonable numerical tolerance of `"average"` (they solve different optimization problems, but both are valid)

### Test requirements (Phase 2)

1. **Union graph correctness**: For known 2-cycle setup, verify union neighbor set is correct (union of both windows).

2. **Multiplicity correctness**: For two cycles sharing edges, verify multiplicity > 1 for shared edges.

3. **Average mode unchanged**: `multi_cycle_mode="average"` produces identical output to current code.

4. **Union output is valid**: No NaN, correct shape, all attention weights sum to 1.

5. **Performance comparison**: Union mode should be faster than average mode for d >= 2 (fewer total FLOPS).

---

## Problem 3: Ad-Hoc d Selection

### What's wrong

`num_cycles` is always user-specified with a default of 1. There is no guidance on how to choose it. From expander graph theory, the number of edge-disjoint Hamiltonian cycles needed for guaranteed O(log T) diameter is:

```
d = ceil(c * log2(T))   where c ~= 2-3
```

For T=2048: d ~= 22-33 (with c=2-3)
For T=65536: d ~= 32-48

### Proposed fix: `recommended_num_cycles(T)` utility

Add to `hcsa/cycles.py`:

```python
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
```

Also add Walecki-based upper bound check:

```python
def max_edge_disjoint_cycles(T: int) -> int:
    """Theoretical maximum edge-disjoint Hamiltonian cycles for K_T.

    For T even: floor((T-1)/2) (Walecki decomposition, already in _walecki_even_cycles)
    For T odd:  floor(T/2) (same structure)
    """
    return (T - 1) // 2 if T >= 4 else (1 if T >= 3 else 0)
```

Wire `recommended_num_cycles` into configs:

```python
# In QwenWayfinderConfig, GLMWayfinderConfig, etc.:
num_cycles: int | Literal["auto"] = 1
# When "auto": num_cycles = recommended_num_cycles(T)
```

### Test requirements (Phase 3)

1. `recommended_num_cycles(T=1024)` returns a value in [11, 30] (for c=2: ceil(2*10) = 20)
2. `recommended_num_cycles(T=2)` returns 1
3. `max_edge_disjoint_cycles(T=100)` returns 49
4. When `num_cycles="auto"`, the runtime correctly computes `d` from `T`

---

## Problem 4: Integrate Laplacian/Cheeger Diagnostics

### What already exists

`hcsa/graph/analysis.py` already has:
- `spectral_gap(cycle_perm, *, include_window, window, expander_threshold)` — full eigenvalue computation
- `expansion_proxy(cycle_perm, *, window, num_walks, walk_len, rng)` — random-walk mixing estimate
- `check_resilience(cycle_perm, window, drop_rate, *, num_trials, rng)` — edge-drop survival
- `check_regularity(cycle_perm, num_clusters, *, epsilon)` — epsilon-regularity
- `compute_edge_coverage(cycles, T, *, causal_only)` — union edge coverage

The `spectral_gap` function already computes the adjacency matrix eigenvalues and the d/lambda ratio. The `verify_spectral_gap` flag is already wired through configs.

### What to add

**4a: Laplacian spectral gap (Cheeger constant proxy)**

Add to `hcsa/graph/analysis.py`:

```python
def laplacian_spectral_gap(
    cycle_perm: np.ndarray,
    *,
    include_window: bool = False,
    window: int = 0,
) -> dict[str, Any]:
    """Compute Laplacian spectral gap (algebraic connectivity / Fiedler value).

    The Laplacian L = D - A where D is the degree matrix.
    The second-smallest eigenvalue lambda_2(L) (Fiedler value) bounds the
    Cheeger constant: lambda_2/2 <= h(G) <= sqrt(2 * lambda_2).

    A larger Fiedler value means better connectivity / harder to cut.

    Returns:
    - fiedler_value: lambda_2(L), the algebraic connectivity
    - cheeger_lower: lambda_2 / 2
    - cheeger_upper: sqrt(2 * lambda_2)
    - fiedler_vector: eigenvector for lambda_2 (useful for identifying bottlenecks)
    - is_well_connected: fiedler_value > threshold
    """
```

**4b: Fiedler-bridge rewiring**

The Fiedler vector (eigenvector of the second-smallest Laplacian eigenvalue) identifies the **bottleneck cut** in the graph. Vertices where the Fiedler vector changes sign are at the connectivity boundary. Adding edges across this boundary (rewire edges) maximally improves connectivity.

Add to `hcsa/graph/analysis.py`:

```python
def fiedler_bridge_candidates(
    cycle_perm: np.ndarray,
    *,
    window: int = 0,
    num_bridges: int = 10,
) -> list[tuple[int, int]]:
    """Identify the best bridge edges to add for connectivity improvement.

    Uses the Fiedler vector to find the graph bottleneck, then proposes
    edges between the most positive and most negative Fiedler-vector vertices.

    These edges can be added as REWIRE edges in the graph ABI.
    """
```

Wire into the existing `rewire_adj` parameter of `build_graph_abi_from_adjacency` (already supports `EdgeType.REWIRE`).

**4c: Diagnostic integration**

After graph construction, optionally compute and log:
```python
diag = {
    "spectral_gap": spectral_gap(perm, include_window=True, window=w),
    "laplacian": laplacian_spectral_gap(perm, include_window=True, window=w),
    "expansion_proxy": expansion_proxy(perm, window=w),
}
```

Add a `log_diagnostics: bool = False` flag to `WayfinderAttentionMLX` and integration configs.

### Test requirements (Phase 4)

1. `laplacian_spectral_gap` on a connected graph returns `fiedler_value > 0`
2. `laplacian_spectral_gap` on a disconnected graph returns `fiedler_value ~= 0`
3. `fiedler_bridge_candidates` returns edges that actually improve the Fiedler value
4. Adding rewire bridges to a bottlenecked graph measurably increases `fiedler_value`

---

## Implementation Stages

Execute in this order. Each stage is independently testable and committable.

### Stage 1: Circular windowing (Problem 1)
**Files to modify:**
- `hcsa/mlx/attention.py`: `permute_cycle_window_attention_single`, `wayfinder_permute_window_attention_batched`, `wayfinder_permute_window_attention_active_batched`
- New test file: `tests/mlx/test_circular_wrap.py`

**Risk:** Medium. The batched chunked path (Function 2) is the most complex due to circular padding at chunk boundaries. Start with Function 1 (simplest), verify, then tackle Function 2 and 3.

**Flag:** `circular: bool = False` (default preserves existing behavior)

### Stage 2: Union multigraph (Problem 2)
**Files to modify:**
- `hcsa/mlx/attention.py`: multi-cycle branches in all three functions
- `hcsa/graph/abi.py`: optional multiplicity tracking in `build_graph_abi_from_adjacency`
- New test file: `tests/mlx/test_union_multigraph.py`

**Risk:** Medium. The union graph has variable degree per position (unlike the fixed-width window), requiring gather-based attention rather than the contiguous-window fast path.

**Flag:** `multi_cycle_mode: Literal["average", "union"] = "average"` (default preserves existing behavior)

### Stage 3: Principled d selection (Problem 3)
**Files to modify:**
- `hcsa/cycles.py`: add `recommended_num_cycles`, `max_edge_disjoint_cycles`
- All config dataclasses: support `num_cycles: int | Literal["auto"]`
- `hcsa/topology/core.py`: resolve "auto" at construction time

**Risk:** Low. Purely additive, no existing behavior changes.

### Stage 4: Laplacian/Cheeger diagnostics and Fiedler rewiring (Problem 4)
**Files to modify:**
- `hcsa/graph/analysis.py`: add `laplacian_spectral_gap`, `fiedler_bridge_candidates`
- `hcsa/mlx/attention.py`: wire `log_diagnostics` flag
- New test file: `tests/test_laplacian_diagnostics.py`

**Risk:** Low. Purely additive diagnostic tools.

### Stage 5: End-to-end validation
**Benchmark:**
- Run tiny model at T=2048, window=32, comparing:
  - `circular=False, num_cycles=1` (current baseline)
  - `circular=True, num_cycles=1` (closed cycle)
  - `circular=True, num_cycles=2, multi_cycle_mode="average"` (current multi-cycle)
  - `circular=True, num_cycles=2, multi_cycle_mode="union"` (new union path)
  - `circular=True, num_cycles="auto"` (principled d)
- Measure: throughput, memory, spectral gap, Fiedler value
- Record all results in lab notebook

---

## File Location Guide

```
hcsa/mlx/attention.py                  # PRIMARY TARGET — circular wrap, union multigraph
hcsa/graph/abi.py                      # Multiplicity tracking for union graph
hcsa/graph/analysis.py                 # Laplacian, Fiedler diagnostics (existing + new)
hcsa/cycles.py                         # recommended_num_cycles, max_edge_disjoint_cycles
hcsa/topology/core.py                  # Wire "auto" num_cycles
hcsa/mlx/model.py                      # Config updates (circular, multi_cycle_mode)
hcsa/integrations/qwen_mlx.py          # Config updates for Qwen integration
hcsa/integrations/glm_mlx.py           # Config updates for GLM integration
hcsa/integrations/gpt2_mlx.py          # Config updates for GPT-2 integration
tests/mlx/test_circular_wrap.py        # NEW — circular windowing tests
tests/mlx/test_union_multigraph.py     # NEW — union multigraph tests
tests/test_laplacian_diagnostics.py    # NEW — Laplacian/Fiedler tests
```

## Safety Constraints

1. **All new behavior is opt-in via flags.** Defaults match current behavior exactly.
2. **No changes to the sparse-gather reference path.** Only the permute fast path is modified.
3. **Causality must be verified for every change.** The causal mask operates on original token indices, not permuted indices. Circular wrapping does not violate causality because the causal check (`k_orig <= q_orig`) is independent of permutation order.
4. **The PyTorch backend is out of scope.** Only `hcsa/mlx/attention.py` is modified. The PyTorch backend (`hcsa/torch/`) uses different code paths and should be updated in a separate PR.
5. **No performance regression when flags are off.** `circular=False, multi_cycle_mode="average"` must produce identical output and identical performance to the current code. Verify with benchmarks.
6. **Memory budget awareness.** The union multigraph may have higher degree than a single cycle window. Ensure `memory_budget_bytes` planner accounts for variable-width union graphs.

## Key Insight: Why This Matters

The **spectral gap** of a union of `d` edge-disjoint random Hamiltonian cycles on `T` vertices is:

- **Degree**: `2d` (each cycle contributes 2 edges per vertex)
- **Second eigenvalue**: `lambda_2 = O(sqrt(d))` (concentration result)
- **Expansion ratio**: `d/lambda = Omega(sqrt(d))`

With `d = O(log T)`:
- Expansion ratio: `Omega(sqrt(log T))` — provably good expander
- Diameter: `O(T / (d * W))` hops through sparse attention — logarithmic mixing
- Resilience: survives dropping up to `(1/2 - epsilon)` of edges (Theorem 1.5)

But **none of this holds** if:
- The cycle is actually a path (clamping breaks the wrap-around edge) — drops 1 edge per vertex from the spectral budget
- The d cycles are averaged instead of unioned — the effective degree is still 2 (not 2d)
- d is chosen ad-hoc instead of scaling with T — no expansion guarantee

Fixing all three turns HCSA from "attention on a random permutation path" into "attention on a provably expanding multigraph" — a qualitative improvement in the theoretical foundation.

---

## Important Reminders

- `python3` not `python`
- Package imports: `from hcsa import ...`, `from hcsa.mlx.attention import ...`
- The graph ABI bridge: strategies produce cycles on CPU (numpy), ABI wraps them, MLX converts to `mx.array`
- The permute fast path requires `cycle_perms` in ABI meta — any new cycle strategy must populate this
- Keep all existing tests passing throughout
- Memory sign convention: `memory_reduction_pct = 100 * (1 - wayfinder/dense)`
- For benchmarks, report: absolute metric, delta vs baseline, percentage delta vs baseline
- Walecki decomposition for edge-disjoint cycles: `hcsa/cycles.py:117` (`_walecki_even_cycles`)
- Edge-disjoint generation already exists: `hcsa/cycles.py:135` (`edge_disjoint_random_cycles`)
- Spectral diagnostics already exist: `hcsa/graph/analysis.py:85` (`spectral_gap`), `hcsa/graph/analysis.py:161` (`expansion_proxy`)
- The `verify_spectral_gap` flag is already wired through all integration configs
