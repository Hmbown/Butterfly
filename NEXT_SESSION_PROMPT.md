# Next Session: Integrate Hamiltonian Cycle Theory into HCSA

You are the AI/ML research lead for the HCSA (Hamiltonian Cycle Sparse Attention) project.

**Working directory**: `/Volumes/VIXinSSD/wayfinder`

**Read first** (in order):
1. `CLAUDE.md` — project overview, architecture, commands
2. `AGENTS.md` — mandatory sub-agent policy, Bell Labs notebook protocol
3. `docs/ROADMAP.md` — current state, overnight execution package, phase plan
4. `notes/LAB_NOTEBOOK.md` — experiment history (append-only)
5. `notes/experiments.ndjson` — machine-readable experiment log (append-only)
6. The referenced paper: Draganić, Kim, Lee, Munhá Correia, Pavez-Signé, Sudakov (2025). "Hamilton cycles in pseudorandom graphs." arXiv:2507.22807

**Non-negotiable constraints**:
- Follow `AGENTS.md` exactly (sub-agents, Bell Labs protocol).
- Every code change gets a hypothesis entry BEFORE and a result entry AFTER in both `notes/LAB_NOTEBOOK.md` and `notes/experiments.ndjson`.
- Keep Hamiltonian path active (`path=permute`). Do not redesign away from Hamiltonian.
- Retro/backfill OFF by default for inference.
- Must use `python3` not `python` on this system.
- `ruff` line-length = 100 chars.
- Run `pytest` after each code change to verify no regressions.

---

## Context: What HCSA Does Today

HCSA replaces dense O(T²) attention with sparse O(T·W) attention. Each token attends to:
- **Cycle neighbors**: 2 adjacent nodes in a Hamiltonian cycle (long-range shortcuts)
- **Local window**: the W preceding tokens (local context)
- **Self**: always included

The **permute fast path** reorders Q/K/V into cycle order so cycle-neighbor attention becomes contiguous local-window attention. This is the production path.

Current cycle construction: `hcsa/cycles.py` generates a single random permutation per head. The `num_cycles` parameter exists in `hcsa/graph_strategies.py` but when `num_cycles > 1`, the additional cycles are **not verified for edge-disjointness** — they can share edges, wasting the sparsity budget. The permute fast path in `hcsa/mlx/attention.py` only uses `perms[0]` from the ABI metadata, **silently discarding additional cycles**.

---

## Your Mission

Integrate five ideas from Draganić et al. (2025) into the HCSA codebase. Each idea must be:
1. **Implemented** with clean, tested code
2. **Validated** with unit tests and at least one benchmark measurement
3. **Documented** in the lab notebook with hypothesis/result entries

Only after ALL five ideas are integrated, tested, and documented should you return to the GLM-4.7-Flash consumer benchmark campaign (see final section).

---

## Idea 1: Edge-Disjoint Cycle Packing

### Theory
Theorems 1.6 and 1.8 prove that (n,d,λ)-graphs contain ~d/2 edge-disjoint Hamilton cycles. Edge-disjoint means no two cycles share any edge — every edge in the union graph provides unique connectivity. Using multiple edge-disjoint cycles per head gives more long-range connections within the same total degree budget.

### What exists today
- `hcsa/cycles.py`: `random_cycle(T, generator)` returns a single random permutation
- `hcsa/graph_strategies.py`: `RandomCycleStrategy.build()` calls `random_cycle()` `num_cycles` times independently. No disjointness check. The resulting adjacency is the union, but overlapping edges are wasted.
- `hcsa/graph/abi.py`: `WayfinderGraphABI` stores `neigh_idx` (neighbor indices) and `edge_types`. The `meta` dict can hold `cycle_perm` (single perm) or `all_cycle_perms` (list of perms).
- `hcsa/mlx/attention.py`: `wayfinder_permute_window_attention_batched()` only uses the first permutation. Multiple cycles are not utilized in the permute path.

### What to implement

#### 1a. Edge-disjoint cycle generator
In `hcsa/cycles.py`, add:
```python
def edge_disjoint_random_cycles(T: int, num_cycles: int, *, max_retries: int = 100, generator=None) -> list[np.ndarray]:
    """Generate num_cycles edge-disjoint Hamiltonian cycles on T vertices.

    Strategy: generate cycles one at a time. For each new cycle, verify that
    no edge in the new cycle appears in any previously accepted cycle. If
    collision, regenerate with a new random seed. After max_retries failures
    for a single cycle, raise ValueError.

    An edge is an unordered pair {perm[i], perm[(i+1) % T]} for i in [0, T).

    Returns: list of num_cycles permutation arrays, each shape [T].
    """
```

Edge collision check: build a set of frozenset pairs from each accepted cycle. For a candidate cycle, check if any of its T edges appear in the global set. This is O(T) per check, O(num_cycles * T * retries) total — fast for T ≤ 200k and num_cycles ≤ 4.

#### 1b. Disjointness verification utility
```python
def verify_edge_disjoint(cycles: list[np.ndarray]) -> tuple[bool, int]:
    """Check that cycles are pairwise edge-disjoint.
    Returns (is_disjoint, num_shared_edges).
    """
```

#### 1c. Wire into GraphStrategy
In `hcsa/graph_strategies.py`, modify `RandomCycleStrategy.build()`:
- When `num_cycles > 1` and a new flag `edge_disjoint=True` (default True), call `edge_disjoint_random_cycles` instead of calling `random_cycle` independently num_cycles times.
- Store ALL permutations in the ABI metadata as `all_cycle_perms`.

#### 1d. Multi-cycle permute path
In `hcsa/mlx/attention.py`, modify the permute attention to support multiple cycles:
- Accept `all_perms` as `[H, T]` (d=1, current behavior) or `[H, d, T]` (d>1).
- For d>1: run d independent permute-window passes (each using one cycle's permutation), average the outputs.
- Each pass is the existing single-cycle logic — no new kernel needed.
- Keep d=1 as default with zero overhead.

This is already sketched in `docs/ROADMAP.md` section 3b-bis. Follow that design.

#### 1e. Tests
- `tests/test_edge_disjoint_cycles.py`:
  - Test that `edge_disjoint_random_cycles(T=128, num_cycles=2)` returns 2 cycles with no shared edges.
  - Test that `edge_disjoint_random_cycles(T=64, num_cycles=3)` works.
  - Test that `verify_edge_disjoint` correctly identifies non-disjoint cycles.
  - Test that the multi-cycle permute path produces output with correct shape and no NaN.
  - Test that d=1 produces identical output to the existing single-cycle path.

#### 1f. Benchmark measurement
Run the tiny MLX benchmark with `num_cycles=1` vs `num_cycles=2` (edge-disjoint) at T=2048 and T=4096:
```bash
PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32 --warmup 2 --iters 4 \
  --num-cycles 1 --out-dir benchmarks/mlx/tiny_wayfinder/disjoint_d1

PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32 --warmup 2 --iters 4 \
  --num-cycles 2 --out-dir benchmarks/mlx/tiny_wayfinder/disjoint_d2
```
Record throughput ratio and memory for d=1 vs d=2. Expected: d=2 is ~2x slower (two passes) but provides denser long-range connectivity. The quality story matters more than speed here — this is a topology investment.

If `--num-cycles` doesn't exist as a CLI flag on the tiny benchmark script, add it and wire it through.

---

## Idea 2: Resilience Guarantees for Window-Drop

### Theory
Theorem 1.5 (Resilience): For any spanning subgraph H of an (n,d,λ)-graph G with minimum degree δ(H) ≥ (1/2 + γ)d, H contains a Hamilton cycle. This means: if you randomly remove up to ~half the edges from a good expander, the Hamiltonian cycle structure **provably survives**.

### What exists today
- `window_drop` parameter in attention configs: during training, randomly drops edges from the attention window with some probability. This is a regularization technique.
- No theoretical justification is cited. No verification that the remaining graph still has a Hamilton cycle.

### What to implement

#### 2a. Resilience verification utility
In `hcsa/graph/analysis.py` (new file):
```python
def check_resilience(
    cycle_perm: np.ndarray,
    window: int,
    drop_rate: float,
    *,
    num_trials: int = 100,
    rng: np.random.Generator | None = None,
) -> dict:
    """Empirically verify that window-drop at the given rate preserves
    Hamiltonian cycle structure.

    For each trial:
    1. Build the full neighbor graph (cycle + window edges)
    2. Drop edges independently with probability drop_rate
    3. Check if the remaining graph is Hamiltonian
       (use a heuristic: check that min degree >= T//2 for small T,
        or check connectivity + min-degree bound for larger T)
    4. Record whether the cycle survived

    Returns dict with:
    - survival_rate: fraction of trials where Hamiltonian structure survived
    - min_degree_mean: average minimum degree across trials
    - min_degree_min: worst-case minimum degree
    - theoretical_threshold: (1/2 + gamma) * d for the graph
    """
```

For practical purposes, checking full Hamiltonicity is NP-hard for large T. Instead, verify the **necessary conditions**:
- The graph remains connected (BFS/DFS, O(T+E))
- Minimum degree ≥ T/2 (Dirac's theorem gives Hamiltonicity for dense graphs)
- Or for sparse graphs: verify the spectral gap condition (see Idea 4)

#### 2b. Documentation
Add a section to `docs/ROADMAP.md` or a new `docs/resilience.md` explaining:
- The theoretical backing from Theorem 1.5
- What window_drop rates are provably safe
- Empirical verification results from 2a

#### 2c. Tests
- `tests/test_resilience.py`:
  - At T=128, window=32: verify that drop_rate=0.3 has survival_rate > 0.95
  - At T=128, window=32: verify that drop_rate=0.8 has survival_rate < 0.5 (too aggressive)
  - Verify that min_degree stays above the theoretical threshold at safe drop rates

#### 2d. Cite in code
Add a comment in the attention code where window_drop is applied, referencing Theorem 1.5 and the verified safe range.

---

## Idea 3: Edge Covering — Approximate Dense via Cycle Decomposition

### Theory
Theorem 1.7: At most (1+ε)d/2 Hamilton cycles suffice to **cover all edges** of an (n,d,λ)-graph. This means full dense attention can be decomposed into a small number of sparse Hamiltonian attention passes whose union visits every possible token-token connection.

### What to implement

#### 3a. Edge-covering cycle generator
In `hcsa/cycles.py`:
```python
def covering_cycles(T: int, *, max_cycles: int = 20, coverage_target: float = 0.99, generator=None) -> tuple[list[np.ndarray], float]:
    """Generate Hamilton cycles until the union covers ≥ coverage_target fraction
    of all T*(T-1)/2 possible undirected edges (causal edges only: j < i).

    Strategy: greedily add random cycles, tracking the set of covered edges.
    Stop when coverage_target is reached or max_cycles is hit.

    Returns: (list of cycle permutations, actual coverage fraction)
    """
```

Note: For T=1024, there are ~524k possible causal edges. Each cycle contributes T edges. So ~524 cycles would be needed for full coverage at T=1024. For T=64, ~2016 edges, ~32 cycles. This is only practical for small T or as a theoretical validation tool.

#### 3b. Coverage analysis utility
```python
def compute_edge_coverage(cycles: list[np.ndarray], T: int, *, causal_only: bool = True) -> dict:
    """Compute what fraction of possible edges are covered by the union of cycles.

    Returns:
    - total_possible_edges: T*(T-1)/2 for causal
    - covered_edges: number of unique edges across all cycles
    - coverage_fraction: covered/total
    - edges_per_cycle: T (each cycle contributes exactly T edges)
    - theoretical_min_cycles: ceil(total_possible_edges / T) lower bound
    """
```

#### 3c. Multi-pass covering attention mode
In `hcsa/mlx/attention.py`, add a new attention mode or extend the existing permute path:
```python
def wayfinder_covering_attention(
    q, k, v, all_perms, all_inv_perms, window, scale, mask=None
):
    """Run one permute-window attention pass per cycle, then average outputs.

    This approximates dense attention by covering all edges across passes.
    Cost: O(d * T * W) where d is the number of covering cycles.
    Memory: O(T * W) per pass (same as single-cycle).

    For d = O(T/W) cycles, total cost is O(T²) — same as dense.
    The value is that for d << T/W, you get a controlled approximation
    with provable coverage guarantees.
    """
```

#### 3d. Tests
- `tests/test_covering.py`:
  - At T=64: verify that `covering_cycles` reaches >0.95 coverage within 50 cycles
  - Verify coverage monotonically increases with more cycles
  - Verify `wayfinder_covering_attention` with 1 cycle matches existing single-cycle output
  - Verify `wayfinder_covering_attention` with "enough" cycles produces output closer to dense attention than a single cycle does (measure cosine similarity or L2 distance to dense output)

#### 3e. Benchmark
Run a small comparison: at T=256 or T=512 with a tiny model config, compute:
- Dense attention output
- Single-cycle permute output
- 4-cycle covering output
- 8-cycle covering output

Measure L2 distance to dense for each. This validates whether cycle covering actually converges toward dense attention.

---

## Idea 4: Spectral Gap Verification

### Theory
The core condition for Hamiltonicity in pseudorandom graphs is d/λ ≥ C, where d is the degree and λ is the second-largest eigenvalue of the adjacency matrix. This spectral gap measures expansion quality. Larger d/λ → better expander → guaranteed Hamiltonian cycles.

Random permutation cycles on T vertices form a graph with d=2 and expected λ = O(1/√T), giving d/λ = O(√T) — good expansion. But we never verify this.

### What to implement

#### 4a. Spectral gap computation
In `hcsa/graph/analysis.py`:
```python
def spectral_gap(cycle_perm: np.ndarray, *, include_window: bool = False, window: int = 0) -> dict:
    """Compute spectral properties of the attention graph defined by the cycle
    (and optionally the local window).

    Build the adjacency matrix A of the undirected graph.
    Compute eigenvalues using numpy.linalg.eigvalsh (symmetric real matrix).

    Returns:
    - degree: average vertex degree
    - lambda_1: largest eigenvalue (= degree for regular graphs)
    - lambda_2: second-largest eigenvalue (in absolute value)
    - spectral_gap: lambda_1 - |lambda_2|
    - expansion_ratio: degree / |lambda_2| (the d/λ ratio)
    - is_good_expander: expansion_ratio >= some threshold (e.g., 4.0)

    Note: eigenvalue computation is O(T³) — only feasible for T ≤ ~4096.
    For larger T, use scipy.sparse.linalg.eigsh with k=2 for O(T·nnz) cost.
    """
```

#### 4b. Cheap expansion proxy for large T
For T > 4096, full eigendecomposition is too expensive. Implement a cheaper proxy:
```python
def expansion_proxy(cycle_perm: np.ndarray, *, window: int = 0, num_walks: int = 1000, walk_len: int = 20, rng=None) -> dict:
    """Estimate mixing time via random walks on the attention graph.

    Start num_walks random walks from random vertices. After walk_len steps,
    measure how uniformly the endpoints are distributed (chi-squared test
    against uniform distribution).

    A good expander mixes quickly (endpoints are near-uniform after O(log T) steps).
    A poor expander has clustered endpoints even after many steps.

    Returns:
    - mixing_time_estimate: number of steps until chi-squared < threshold
    - endpoint_uniformity: chi-squared statistic at walk_len
    - is_fast_mixer: mixing_time_estimate <= 2 * log2(T)
    """
```

#### 4c. Integration into graph construction
In the graph runtime (`_QwenGraphRuntime._build_graph_abi` or wherever graphs are constructed), add an optional verification step:
```python
if cfg.verify_spectral_gap:
    gap_info = spectral_gap(perm, include_window=True, window=cfg.window)
    if not gap_info['is_good_expander']:
        warnings.warn(f"Cycle has poor expansion: d/λ = {gap_info['expansion_ratio']:.2f}")
        # Optionally: regenerate the cycle
```

This should be OFF by default (it's expensive) but available for research/debugging.

#### 4d. Tests
- `tests/test_spectral.py`:
  - At T=128: verify random cycle has expansion_ratio > 2.0
  - At T=128: verify a deliberately bad cycle (e.g., identity permutation = no shuffle) has expansion_ratio close to 0
  - At T=256: verify that adding window edges improves the spectral gap
  - Verify the random walk proxy agrees directionally with exact eigenvalue computation at T=128

---

## Idea 5: Sparse Regularity and Structured Graph Construction

### Theory
The Draganić et al. proof uses Szemerédi's regularity lemma for sparse graphs to decompose the vertex set into clusters where edge density is approximately uniform between every pair of clusters. This gives structured, balanced connectivity.

### What to implement

#### 5a. Regularity-informed cycle construction
In `hcsa/cycles.py`, add a new cycle strategy:
```python
def regular_partition_cycle(T: int, num_clusters: int = 8, *, generator=None) -> np.ndarray:
    """Build a Hamiltonian cycle that is regular by construction.

    Strategy:
    1. Partition [0, T) into num_clusters approximately equal clusters.
    2. Within each cluster, create a random sub-path (random permutation of cluster members).
    3. Connect clusters in a random order, linking the end of one cluster's
       sub-path to the start of the next.
    4. The result is a Hamiltonian cycle that by construction visits clusters
       in a balanced interleaving pattern.

    The key property: every cluster-pair has approximately T/num_clusters²
    crossing edges in the cycle, giving uniform inter-cluster connectivity.
    This approximates the ε-regular partition the paper uses.

    Parameters:
    - T: sequence length
    - num_clusters: number of clusters (paper suggests O(1/ε²) for ε-regularity)
    - generator: numpy random generator for reproducibility

    Returns: permutation array of shape [T]
    """
```

#### 5b. Regularity verification
```python
def check_regularity(cycle_perm: np.ndarray, num_clusters: int = 8) -> dict:
    """Check how ε-regular the cycle's edge distribution is across clusters.

    Partition vertices into num_clusters equal groups.
    For each cluster pair (A, B), count edges between them.
    Compute the deviation from the expected count (|A|*|B| * density).

    Returns:
    - max_deviation: max |actual - expected| / expected across all pairs
    - mean_deviation: average deviation
    - is_epsilon_regular: max_deviation < epsilon (e.g., 0.25)
    - cluster_pair_densities: matrix of inter-cluster edge densities
    """
```

#### 5c. Comparison benchmark
Run the tiny model with three cycle strategies at T=2048 and T=4096:
1. `random` (current default)
2. `regular_partition` (new, num_clusters=8)
3. `regular_partition` (new, num_clusters=16)

Measure: throughput, memory, and regularity metrics. The regularity-informed cycles won't be faster (same O(T·W) attention), but they should have better expansion properties (verify with spectral gap from Idea 4).

#### 5d. Tests
- `tests/test_regularity.py`:
  - `regular_partition_cycle` returns a valid permutation (all elements in [0,T), no duplicates)
  - The cycle visits every vertex exactly once
  - `check_regularity` reports lower deviation for `regular_partition_cycle` than for a worst-case cycle (e.g., identity)
  - Verify at T=256 that `regular_partition_cycle` with 8 clusters has `max_deviation < 0.5`

---

## Integration Order

Execute in this order:
1. **Idea 4** (spectral gap) — provides verification tools needed by all other ideas
2. **Idea 1** (edge-disjoint packing) — most impactful for attention quality
3. **Idea 2** (resilience) — quick win, mostly analysis + documentation
4. **Idea 5** (regularity) — new cycle strategy, uses tools from Idea 4
5. **Idea 3** (covering) — most ambitious, builds on everything above

For each idea:
1. Write hypothesis entry in `notes/LAB_NOTEBOOK.md` and `notes/experiments.ndjson`
2. Implement the code
3. Run `pytest` to verify no regressions
4. Run the specified benchmark/measurement
5. Write result entry in both logs
6. Commit with descriptive message

---

## File Location Guide

Key files you'll modify or create:
```
hcsa/cycles.py                          # Add edge_disjoint_random_cycles, covering_cycles, regular_partition_cycle
hcsa/graph_strategies.py                # Wire edge_disjoint flag, new strategies
hcsa/graph/abi.py                       # Ensure all_cycle_perms stored correctly for d>1
hcsa/graph/analysis.py                  # NEW: spectral_gap, expansion_proxy, check_resilience, check_regularity, compute_edge_coverage, verify_edge_disjoint
hcsa/mlx/attention.py                   # Multi-cycle permute path, covering attention mode
hcsa/integrations/qwen_mlx.py           # Wire num_cycles, edge_disjoint through _QwenGraphRuntime
hcsa/integrations/glm_mlx.py            # Wire same through GLMWayfinderAttention
tests/test_edge_disjoint_cycles.py      # NEW
tests/test_spectral.py                  # NEW
tests/test_resilience.py                # NEW
tests/test_regularity.py                # NEW
tests/test_covering.py                  # NEW
```

Existing test directories:
- `tests/pytorch/` — PyTorch tests
- `tests/mlx/` — MLX tests
- New graph-theory tests can go in `tests/` top-level since they're backend-agnostic

---

## After All Five Ideas Are Integrated and Tested

Only then, return to the GLM-4.7-Flash consumer benchmark campaign:

1. Re-read `README.md` section "2026-02-08 GLM-4.7 Consumer Benchmark Status" for current state
2. The victory gates require:
   - seq=65536, decode=256: HCSA E2E median >=10% better than dense, TTFT >=10% better, ITL p95 not worse by >5%
   - Peak memory reduction >=8%
   - Quality parity (not worse by >2 percentage points)
   - Reproducibility (>=3 repeats + 2-run confirmation within ±5%)
3. Promoted config: `path=permute, active_dense_threshold=49152, query_chunk_size=384, head_chunk_size=2, kv_step=4096`
4. Consumer benchmark script: `scripts/bench_glm_consumer_mlx.py`
5. Quality dataset: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_v1.json`

Consider whether any of the newly integrated ideas (especially edge-disjoint multi-cycle or regularity-informed cycles) should be tested as part of the GLM benchmark campaign. If `num_cycles=2` with edge-disjoint cycles shows quality improvement in the tiny benchmarks, it would be worth testing on GLM as well.

---

## Important Reminders

- `python3` not `python`
- Package imports: `from hcsa import ...`, `from hcsa.mlx.attention import ...`
- The graph ABI bridge: strategies produce cycles on CPU (numpy), ABI wraps them, MLX converts to `mx.array`
- The permute fast path requires `cycle_perms` in ABI meta — any new cycle strategy must populate this
- Keep all existing tests passing throughout
- Memory sign convention: `memory_reduction_pct = 100 * (1 - wayfinder/dense)`
- For benchmarks, report: absolute metric, delta vs baseline, percentage delta vs baseline
