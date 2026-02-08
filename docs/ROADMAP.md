# HCSA/Wayfinder Production Roadmap

> From research prototype to production sparse attention on MLX.
>
> Status as of 2026-02-08: Memory savings proven (10-23% depending on
> hybrid threshold). Chunked prefill dense-fallback bug fixed. Active-query
> kernel operational with matmul + adaptive graph reuse. Latency crossover
> not yet reached at T<=32K. The open question is where it crosses.

---

## Table of Contents

1. [Current State](#current-state)
2. [Overnight Execution Package](#overnight-execution-package)
3. [Phase 0: Structural Cleanup](#phase-0-structural-cleanup)
4. [Phase 1: Graph Construction Amortization](#phase-1-graph-construction-amortization)
5. [Phase 2: Kill the Python Loop](#phase-2-kill-the-python-loop)
6. [Phase 3: Better Topologies](#phase-3-better-topologies)
7. [Phase 4: Custom Metal Kernel](#phase-4-custom-metal-kernel)
8. [Phase 5: Production Serving](#phase-5-production-serving)
9. [Phase 6: Training from Scratch](#phase-6-training-from-scratch)
10. [Appendix: Cost Model & Crossover Analysis](#appendix-cost-model--crossover-analysis)

---

## Current State

### What Works

- **Memory reduction is real**: 22.9% at T=32768 with threshold=0 (always Hamiltonian),
  22.4% at threshold=16384 (hybrid), 10.1% at threshold=32768 (all dense active mode).
  The Pareto frontier is smooth and controllable.
- **Chunked prefill bug is fixed**: Active-query kernel
  (`wayfinder_permute_window_attention_active_batched`) handles Q_len < K_len correctly.
  All chunks now route through Hamiltonian sparse attention instead of falling back to
  O(T^2) dense MLA. This was the root cause of the original 6.9x regression at T=65536.
- **Graph amortization works**: Adaptive horizon (`_adaptive_graph_seq_len`) builds the
  graph once at an overestimated target length. Subsequent chunks get `cache_hit=True`
  with `graph_build_ms` near zero.
- **Hybrid threshold gate works**: `active_dense_threshold` parameter gates between dense
  (for early low-K chunks) and Hamiltonian permute (for later high-K chunks). Provides a
  smooth latency/memory Pareto frontier.
- **Active-query kernel uses matmul**: Score computation uses `mx.matmul` instead of
  elementwise `mx.sum(q * k)`, routing through Metal AMX units.
- **Three model integrations**: GLM-4.7-Flash (MLA), Qwen3, GPT-2 on MLX.
- **Tiny model validation**: At T=4096, batched permute achieves 3.2x throughput ratio
  and 62% memory reduction vs fused SDPA dense (fair comparison).

### What Doesn't Work Yet

- **Latency crossover not reached at T<=32K**: Even with the active-query fix, matmul,
  and graph amortization, Hamiltonian chunks are still slower than dense chunks at every
  K value tested. The per-chunk Wayfinder constant factor exceeds the dense SDPA constant
  factor, so the curves haven't crossed. More dense chunks = faster total prefill.
- **Python per-head loop remains**: The active-query kernel still loops over heads
  individually with `mx.eval` sync barriers between head-chunks. This is the dominant
  remaining source of the constant-factor gap.
- **No production serving API**, no KV cache quantization, no eviction on graph caches.
- **~1500 lines of duplicated code** across three integration files.
- **torch is a hard dependency** even for MLX-only users (cycles.py uses torch.randperm).

### The Key Numbers

```
GLM-4.7-Flash-4bit, T=32768, chunk=4096, MLX on Apple Silicon:

Reference points:
  Monolithic baseline (stock, no chunking):     61.29s    534.66 tok/s    28.93 GB
  Stock chunked dense (no swap, chunk=4096):    165.53s    197.95 tok/s    26.02 GB

Hybrid threshold sweep (active-query + matmul + adaptive graph):
  threshold=0     (8 permute, 0 dense):  267.87s   122.33 tok/s   22.31 GB   22.86% mem saved
  threshold=4096  (7 permute, 1 dense):  245.80s   133.31 tok/s   22.31 GB   22.88% mem saved
  threshold=8192  (6 permute, 2 dense):  228.55s   143.38 tok/s   22.31 GB   22.88% mem saved
  threshold=16384 (4 permute, 4 dense):  197.32s   166.07 tok/s   22.45 GB   22.41% mem saved
  threshold=32768 (0 permute, 8 dense):  164.15s   199.62 tok/s   26.02 GB   10.06% mem saved

65k triangulation (active-query + matmul + adaptive graph):
  monolithic baseline:                         126.69s   517.30 tok/s   44.89 GB
  threshold=16384 (12 permute, 4 dense):      959.90s    68.27 tok/s   24.14 GB   46.22% mem saved
  threshold=32768 (8 permute, 8 dense):       923.29s    70.98 tok/s   26.02 GB   42.04% mem saved
  threshold=49152 (4 permute, 12 dense):      890.70s    73.58 tok/s   29.59 GB   34.09% mem saved

65k chunked-dense matched control (`--no-swap`, chunk=4096):
  run A:                                       785.10s    83.47 tok/s   33.16 GB   26.13% mem saved
  run B:                                       626.80s   104.56 tok/s   33.16 GB   26.13% mem saved
  median:                                      705.95s    94.02 tok/s   33.16 GB

65k tuned long-context config (`threshold=49152`, `q=384`, `h=2`):
  run A:                                       483.50s   135.54 tok/s   29.59 GB   34.09% mem saved
  run B:                                       485.45s   135.00 tok/s   29.59 GB   34.09% mem saved
  median:                                      484.48s   135.27 tok/s   29.59 GB

65k retune probe (same `q=384`, `h=2`):
  threshold=45056 (run A):                     464.72s   141.02 tok/s   28.70 GB
  threshold=45056 (repeat):                    658.55s    99.52 tok/s   28.70 GB
  threshold=45056 (median):                    561.64s   120.27 tok/s   28.70 GB
  threshold=53248:                             638.10s   102.71 tok/s   30.48 GB

Key observations:
  - threshold=32768 (all dense active) matches stock chunked dense (164s vs 166s)
    confirming minimal overhead from the chunking infrastructure itself
  - Each Hamiltonian chunk adds ~13s overhead vs a dense chunk (267.87-164.15)/8 = ~13s
  - Memory reduction is ~23% for any threshold that uses at least one permute chunk
  - The latency/memory tradeoff is monotonic: more permute = more memory saved, more time
  - At T=65536 after active-query fix + matmul + graph reuse, latency remains far from
    monolithic baseline, but 34-46% memory reduction is sustained across thresholds
  - Matched dense controls at 65k are also very slow (626-785s), confirming chunking/KV
    growth is a dominant absolute cost; Wayfinder overhead is additive on top
  - A head-chunk vectorization attempt regressed 32k (`197s -> 943s`) and was reverted
  - Query-chunk tuning was high-impact at 65k: tuned config now beats chunked-dense
    control median by `221.47s` and `+41.26 tok/s` while using `10.77%` less peak memory
  - Threshold `45056` can be faster on single runs but currently shows high variance;
    `49152` is the reproducible default based on pair stability
```

### Completed Items

These were identified in the original roadmap and have been implemented:

1. **Chunked prefill dense-fallback fix** (Phase 2 prerequisite): `glm_mlx.py` line 259
   had `Q_len != K_len` guard that routed all chunks after chunk 0 to O(T^2) dense MLA.
   Implemented `wayfinder_permute_window_attention_active_batched` in `attention.py` to
   handle Q_len < K_len by computing query positions' ranks in the graph permutation.

2. **Active-query matmul** (Phase 2a): Replaced elementwise `mx.sum(q * k)` with
   `mx.matmul` in the active-query kernel. Eliminates 56MB intermediate per query-chunk
   at dh=576 for MLA.

3. **Adaptive graph reuse** (Phase 1a partial): `_adaptive_graph_seq_len()` builds graph
   at an overestimated target and reuses across chunks. `cache_hit=True` after chunk 0,
   `graph_build_ms` near zero for subsequent chunks.

4. **Hybrid dense/permute threshold gate** (Phase 2e): `active_dense_threshold` parameter
   on `GLMWayfinderAttention`. When `k_len < threshold`, the active-query path uses dense
   attention instead of Hamiltonian permute. Provides controllable latency/memory tradeoff.

---

## Overnight Execution Package

This section is designed as a direct handoff prompt scaffold for an overnight agent.

### North Star (GLM-4.7 Positive Benchmark)

At `T=65536`, `chunk=4096`, prefill-only:

- Latency target:
  - `tok/s` is **at least parity** with chunked dense control at the same chunking regime,
    or within `-5%` if memory win is substantial.
- Memory target:
  - Peak memory reduction vs monolithic baseline remains **>= 15%** using
    `100 * (1 - wayfinder / dense)`.
- Correctness target:
  - No causality breakage and sanity MAE remains within current GLM tolerance envelope.

### Immediate Facts To Ground Work

- 32k crossover extraction shows dense is faster on every chunk, but the gap collapses:
  - `k_len=4096`: permute `+33.37s` slower
  - `k_len=32768`: permute `+0.18s` slower
- `corr(k_len, permute_attention_ms) ~= 0.97` while `graph_seq_len` is fixed and
  `cache_hit=true` after chunk 0.
- Conclusion: graph rebuild is largely solved; remaining gap is active-kernel constant factor.
- Current reproducible 65k config:
  - `threshold=49152`, `query_chunk_size=384`, `head_chunk_size=2`
  - median `484.48s`, `135.27 tok/s`, peak `29.59 GB`
  - beats chunked dense control median (`705.95s`, `94.02 tok/s`) by `221.47s`
    and `+41.26 tok/s`, with `10.77%` lower peak memory.

### Overnight Workstream (Ordered)

1. Stabilize and tighten reproducible long-context default.
  - Keep `query_chunk_size=384`, `head_chunk_size=2`.
  - Use `threshold=49152` as default until a lower threshold beats it on replicated runs.
  - For new threshold candidates, require at least 2 runs and compare medians/stdev.
  - Report: absolute prefill sec/tok-s/peak bytes, delta vs monolithic baseline, and
    delta vs chunked dense control median.

2. Implement active-kernel constant-factor reduction.
  - Primary: remove Python per-head loop in active path by batching heads where possible.
  - Keep existing matmul path; do not regress graph cache-hit behavior.
  - Ensure per-chunk profile still includes:
    `path`, `k_len`, `graph_build_ms`, `attention_ms`, `cache_hit`, `graph_seq_len`.
  - Constraint from latest attempt: avoid large gather/broadcast tensor expansion patterns
    that traded Python overhead for much worse kernel-level memory traffic.

3. Run minimal post-change verification matrix.
  - 32k thresholds: `0`, `16384`, `32768`.
  - 65k thresholds: `16384`, `32768`, `49152`.
  - Include one chunked dense control (`--no-swap`) at 65k for fairness sanity.

4. Promote threshold default only if supported by measured crossover.
  - If crossover still not reached by 65k, keep hybrid default conservative and document
    that active kernel remains the blocker.
  - If crossover appears in late chunks, choose threshold that keeps memory >=15% while
    maximizing tok/s.

### Deliverables Required From Overnight Agent

- Updated:
  - `notes/LAB_NOTEBOOK.md` with planned + result entries.
  - `notes/experiments.ndjson` with full experiment records.
- Artifact paths:
  - Benchmark result directories for each threshold run.
- Decision memo:
  - One paragraph: keep direction vs pivot, backed by crossover evidence.

---

## Phase 0: Structural Cleanup

> Prerequisite for everything else. Unblocks parallel work on Phases 1-4.

### 0a. Extract `WayfinderAttentionBase`

The three integration files (`qwen_mlx.py`, `glm_mlx.py`, `gpt2_mlx.py`) share
70-80% identical code. Extract the shared dispatch logic into a base class.

**What goes in the base class:**
- `__call__` dispatch: graph cache lookup, dense fallback check, path selection
  (sparse vs permute vs active-permute), profiling
- Runtime controls: `set_runtime_controls()`, `clear_runtime_controls()`,
  `cache_persistent_bytes()`
- Window drop mask construction
- Edge type bias scalar extraction
- Value dim padding (`_pad_value_dim`)
- Edge utilization proxy computation
- `_effective_permute_chunking()`
- `_adaptive_graph_seq_len()`

**What stays in subclasses (model-specific):**
- `extract_qkv(x, cache)` -> queries, keys, values
- `project_output(y_h)` -> final output tensor
- Model-specific config fields (e.g., MLA dimensions, RoPE parameters)

**Concrete structure:**
```python
# hcsa/integrations/base.py
class WayfinderAttentionBase(nn.Module):
    def __init__(self, cfg: WayfinderConfig, n_heads: int, ...):
        ...
    def extract_qkv(self, x, cache) -> Tuple[mx.array, mx.array, mx.array]:
        raise NotImplementedError
    def project_output(self, y_h) -> mx.array:
        raise NotImplementedError
    def __call__(self, x, mask, cache):
        queries, keys, values = self.extract_qkv(x, cache)
        # ... all shared dispatch logic ...
        return self.project_output(y_h)

# hcsa/integrations/glm_mlx.py
class GLMWayfinderAttention(WayfinderAttentionBase):
    def extract_qkv(self, x, cache):
        return extract_qkv_from_glm_attention(self, x, cache=cache)
    def project_output(self, y_h):
        y_latent = y_h[..., :self.value_dim]
        y_proj = self.unembed_out(y_latent)
        return self.o_proj(y_proj.transpose(0, 2, 1, 3).reshape(...))
```

**Impact**: ~1500 lines removed. Bug fixes propagate to all models. The active-query
chunked prefill fix (currently GLM-only) automatically works for Qwen and GPT-2.

### 0b. Rename and Relocate `_QwenGraphRuntime`

```
_QwenGraphRuntime         -> WayfinderGraphRuntime
_QWEN_GRAPH_CACHE_STORE   -> _GRAPH_CACHE_BY_OWNER
_QWEN_GRAPH_CACHE_BY_KEY  -> _GRAPH_CACHE_BY_KEY
```

Move to `hcsa/integrations/graph_runtime.py` or `hcsa/mlx/graph_runtime.py`.
Unify with the separate `_GRAPH_CACHE_STORE` in `attention.py` so standalone
`WayfinderAttentionMLX` uses the same infrastructure.

### 0c. Make torch Optional for MLX Path

In `hcsa/cycles.py`, replace the core random cycle generation:

```python
# Before (requires torch)
def random_cycle(T, generator=None):
    perm = torch.randperm(T, generator=generator)
    return perm

# After (numpy for MLX path, torch for PyTorch path)
def random_cycle(T, generator=None, backend="auto"):
    if backend == "numpy" or (backend == "auto" and not _torch_available()):
        rng = np.random.default_rng(generator)
        return rng.permutation(T).astype(np.int64)
    else:
        perm = torch.randperm(T, generator=generator)
        return perm
```

The `WayfinderGraphRuntime._build_graph_abi()` already converts to numpy before
passing to MLX, so this is a clean boundary.

### 0d. Consolidate Graph Subsystem

Current fragmentation:
```
hcsa/graph/abi.py          # ABI dataclass + validation
hcsa/graphs/               # 5 experimental strategies (DEAD CODE for integrations)
hcsa/graph_strategies.py   # Protocol + Random/Greedy/OnlineInsertion
hcsa/mlx/graph_abi.py      # MLX conversions
hcsa/topology/core.py      # Yet another wrapper
```

Target:
```
hcsa/graph/
    abi.py                 # ABI dataclass (keep as-is)
    strategies.py          # Protocol + all strategies (merge graph_strategies.py)
    mlx.py                 # MLX conversions (move from hcsa/mlx/graph_abi.py)
    cache.py               # Unified cache (from WayfinderGraphRuntime)
hcsa/graph/experimental/   # Move hcsa/graphs/ here
```

---

## Phase 1: Graph Construction Amortization

> Make graph construction effectively free at inference time.
>
> **Status**: Partially done. Adaptive graph reuse eliminates rebuild cost after chunk 0.
> Remaining items are optimizations for decode and cache management.

### Done: Adaptive Graph Horizon

`_adaptive_graph_seq_len()` in `glm_mlx.py` builds the graph at an overestimated
target length and reuses it across chunks. After chunk 0, `graph_build_ms` is near
zero. This is the primary graph amortization mechanism for chunked prefill.

### 1a. Precomputed Cycle Table (Prewarm at Startup)

For the `random` strategy with a fixed seed, the permutation for a given T is
deterministic. Precompute and cache permutations for common sequence lengths.

**Memory cost**: A permutation of T=32768 with H=20 heads is `20 * 32768 * 4 bytes
= 2.5 MB`. Even caching 20 different sequence lengths costs only 50 MB.

**Implementation in `WayfinderGraphRuntime`:**
```python
def prewarm(self, seq_lengths: List[int]) -> None:
    """Pre-build graph caches for expected sequence lengths at server startup."""
    for T in seq_lengths:
        self.get_or_build_cache(owner_id=0, T=T)
```

**Where to call it**: At model load time, before first inference request. The
`swap_X_attention_with_wayfinder()` functions should accept a `prewarm_lengths`
parameter.

### 1b. Incremental Cycle Extension for Autoregressive Decode

During token-by-token generation, don't rebuild the full cycle. Insert the new
token into the existing Hamiltonian cycle in O(1):

1. Pick an edge (u, v) in the current cycle
2. Splice new token `w` as u -> w -> v
3. Update the permutation and inverse permutation arrays

**Edge selection strategies** (in order of complexity):
- **Random**: Pick a random edge. O(1), no quality consideration.
- **Positional midpoint**: Pick the edge (u, v) where `(u + v) / 2` is closest
  to the new token's position. O(log T) with a sorted index, or O(1) amortized
  with a bucketed structure.
- **Routing-aware**: If routing embeddings are available, pick the edge (u, v)
  that maximizes `s(u, w) + s(w, v) - s(u, v)`. This is the `online_insertion_step`
  already in `cycles.py` (line 147+). O(T) scan over edges, but only runs once per
  generated token (not per layer).

### 1c. Fixed Structured Cycles for Short Sequences

For T below the crossover threshold, cycle quality barely matters because the
window alone covers a large fraction of tokens. Use a deterministic O(1)-per-element
cycle:

**Bit-reversal permutation**: Position i maps to bit_reverse(i). Properties: O(1) to
compute per element (no storage needed), excellent long-range mixing (guaranteed
O(log T) diameter), fully deterministic.

```python
def bit_reversal_cycle(T: int) -> np.ndarray:
    bits = int(np.ceil(np.log2(max(T, 2))))
    perm = np.zeros(T, dtype=np.int64)
    for i in range(T):
        rev = int(bin(i)[2:].zfill(bits)[::-1], 2)
        perm[i] = min(rev, T - 1)
    return perm
```

### 1d. LRU Eviction on Graph Caches

Currently `_QWEN_GRAPH_CACHE_BY_KEY` grows unbounded. Add eviction:

```python
from collections import OrderedDict

class GraphCacheLRU:
    def __init__(self, max_entries: int = 32, max_bytes: Optional[int] = None):
        self._cache: OrderedDict[tuple, _GraphCache] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes

    def get(self, key: tuple) -> Optional[_GraphCache]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, cache: _GraphCache) -> None:
        self._cache[key] = cache
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
```

---

## Phase 2: Kill the Python Loop

> The single highest-impact remaining performance fix. Turns ~220 Python
> iterations per layer (10 head-chunks x 22 query-chunks for GLM) into
> ~1 batched operation per layer.
>
> **Status**: 2a (matmul) and 2e (hybrid dispatch) are done. The remaining
> items (2b, 2c, 2d) target the per-head loop, which is the dominant source
> of the constant-factor gap between Hamiltonian and dense per-chunk cost.

### Done: 2a. Replace Elementwise Attention with matmul

Implemented in `wayfinder_permute_window_attention_active_batched`. Score
computation now uses `mx.matmul` instead of `mx.sum(q * k)`. Value aggregation
uses `mx.matmul(weights, v)` instead of `mx.sum(w * v)`. This eliminates the
56MB intermediate per query-chunk at dh=576 for MLA and routes computation
through Metal's matrix multiply hardware.

### Done: 2e. Hybrid Dispatch: Dense for Small K, Wayfinder for Large K

Implemented as `active_dense_threshold` parameter on `GLMWayfinderAttention`.
When `k_len < threshold`, the active-query path uses stock dense attention.
The threshold sweep at T=32768 shows a smooth Pareto frontier:

```
threshold=0     -> 267.87s, 22.86% mem saved  (all Hamiltonian)
threshold=16384 -> 197.32s, 22.41% mem saved  (recommended default)
threshold=32768 -> 164.15s, 10.06% mem saved  (all dense active)
```

The recommended default is threshold=16384, which retains ~22% memory reduction
with a 26% latency improvement vs always-permute. The memory reduction only drops
significantly when ALL chunks use dense (threshold=32768).

### 2b. Enable Pre-Permute for Tg > Tk (Unlock Fused SDPA)

The active-query path disables pre-permute when `Tg != Tk` (`attention.py:813-814`):
```python
if Tg != Tk:
    prepermute_kv = False
```

This forces every query chunk into the scatter-gather path. But you CAN pre-permute
when `Tg > Tk` -- just use the first Tk entries of the permutation:

```python
perm_truncated = all_perms[:, :Tk]  # [Hq, Tk] -- positions in graph order
# K_permuted[rank] = K[perm[rank]] for rank in [0, Tk)
# Window neighborhoods in permuted space are contiguous slices
```

Once K/V are in permuted order, the window lookup for each query becomes:
1. Find query's rank: `q_rank = inv_perm[q_position]`
2. Window neighbors are `K_permuted[q_rank - W : q_rank + W]` -- a contiguous slice
3. This is exactly the pattern the batched path uses, including `mx.fast.sdpa`

**Implementation**: Factor out the contiguous window attention logic from the batched
path (lines 615-694) into a shared helper that both paths use.

**Expected impact**: Eliminates the gather overhead per query-chunk and enables fused
SDPA dispatch. This is the largest single remaining optimization opportunity.

### 2c. Vectorize Across Heads

For MQA/GQA models (GLM has Hkv=1), all query heads share the same K/V. The
current code loops over heads individually. Instead:

1. Expand K/V once: `K_expanded = mx.broadcast_to(K, [B, Hq, Tk, dh])`
2. Compute ALL heads' permutation ranks simultaneously
3. Build window indices for all heads: `[Hq, Tq, 2W+1]`
4. Single batched gather + single `mx.fast.sdpa` call

This eliminates the head loop entirely. Combined with 2b, the active-query path
becomes ~3 operations per layer vs the current ~220.

### 2d. Remove Per-Head-Chunk mx.eval Barriers

`attention.py:929`: `mx.eval(y_local)` forces GPU synchronization after every
head-chunk. Once heads are vectorized (2c), this becomes a single eval at the end.

Current: ~10 `mx.eval` barriers per layer for GLM.
After: 1 `mx.eval` per layer.

---

## Phase 3: Better Topologies

> Beyond random Hamiltonian cycles. The permute trick generalizes to any
> permutation-decomposable graph.

### 3a. Butterfly/FFT-Style Connectivity

Each token at position i connects to position `i XOR 2^k` for k = 0, 1, ..., log2(T).

Properties:
- O(log T) degree per token
- O(log T) diameter (information propagates in log T layers)
- **Each bit level is an independent perfect matching** -- each is a permutation
- The permute trick works: do log(T) rounds of permute-attend-unpermute

**Cost**: O(T * W * log T) per layer instead of O(T * W), but with the same
memory efficiency per round. At T=32768, log2(T)=15, so 15 permute-window rounds.

**When to use**: For models where information routing quality matters more than
raw throughput. Good candidate for every-Nth-layer sparse attention (see 3e).

### 3b. Random Regular Graphs (Degree 4)

A random 4-regular graph on T nodes:
- Is an expander with high probability (spectral gap > 0.5 for d >= 3)
- Has O(log T) mixing time vs O(T) for Hamiltonian cycles
- Decomposes into exactly 2 perfect matchings (Petersen's theorem for even degree)

**Implementation**: Generate 2 independent random permutations. Each defines a
perfect matching. Together they define a 4-regular graph. Apply the permute trick
twice.

**Cost**: 2 permute-window rounds per layer (2x compute vs Hamiltonian cycle).

**Benchmark question**: Does degree-4 random regular (2 passes, better routing)
outperform Hamiltonian cycle (1 pass, worse routing) on downstream quality?

### 3b-bis. Multi-Cycle Expander Graphs (Exphormer-Informed)

Exphormer (Shirzad et al., ICML 2023) shows that using d>1 independent Hamiltonian
cycles per head creates near-Ramanujan expander graphs with O(log n) mixing time
vs O(n) for a single cycle. The spectral gap improves dramatically: a single random
Hamiltonian cycle on n nodes has spectral gap O(1/n^2), while the union of d>=2
independent cycles gives spectral gap Omega(1) with high probability (Friedman's
theorem applied to random regular graphs).

**Current bug**: The codebase already supports `num_cycles` for graph construction
(adjacency union works in `graph_strategies.py`), but `build_graph_abi_from_adjacency`
only stores `perms[0]` in the ABI meta. The permute fast path can only use 1 cycle,
silently discarding additional cycles. This means `num_cycles=2` builds a correct
degree-4 adjacency matrix for the gather path, but the permute path ignores the
second cycle entirely.

**Fix**: Store all d permutations in the ABI (`all_cycle_perms: list[list[int]]`).
The permute kernel runs d independent passes (each using the existing single-cycle
logic) and averages their outputs. Each pass is O(T*W), total is O(d*T*W).

**Implementation plan**:

1. `graph_strategies.py`: Store all perms instead of just `perms[0]`
2. `graph/abi.py`: Extend `build_graph_abi_from_adjacency` to accept
   `all_cycle_perms: list[list[int]]`; keep `cycle_perm` (singular) for backward compat
3. `mlx/attention.py`: Accept `all_perms` as `[H, T]` (d=1) or `[H, d, T]` (d>1);
   extract loop body into `_single_cycle_permute_pass()` helper; multi-cycle case
   loops over d and averages
4. Integration files (`qwen_mlx.py`, `glm_mlx.py`, `gpt2_mlx.py`): Read
   `all_cycle_perms` from ABI meta, stack into `[H, d, T]` tensor

**Configuration**: `num_cycles=2` recommended for quality-critical applications.
Default remains `num_cycles=1` (zero overhead, identical to current behavior).

**Compute tradeoff**: d=2 doubles the per-layer permute cost (2 passes instead of 1).
At current constants this is prohibitive, but after Phase 2 (head vectorization +
fused SDPA) or Phase 4 (Metal kernel), the per-pass cost drops enough that 2x is
acceptable for the routing quality improvement.

**Quality hypothesis**: For models trained natively with HCSA (Phase 6), d=2 expander
graphs should show measurable perplexity improvement over d=1 on long-range retrieval
tasks (RULER passkey, multi-hop QA), because the O(log n) mixing time means
information can propagate across the full context in O(log n) layers instead of O(n).

### 3c. Hierarchical Cycles

Two-level structure:
1. Partition T tokens into blocks of size B (e.g., B=256)
2. Build a Hamiltonian cycle within each block (local routing)
3. Build a Hamiltonian cycle across block representatives (long-range transport)

This is what landmark tokens already approximate, but making the hierarchy explicit
lets you control the long-range hop frequency and reason about information flow
bounds (provable O(B + T/B) diameter).

### 3d. LSH-Guided Cycles

Use locality-sensitive hashing to build cycles with better routing quality in O(T).
This is the Reformer idea applied to cycle construction instead of attention
selection. The greedy strategy already uses routing similarity but is O(T^2);
LSH gets the same quality trend in O(T).

### 3e. Interleaved Dense-Sparse (Apple's Approach)

Apple's own on-device models use alternating global-local attention. Same philosophy:

```
Layers 0-3:   Dense (early layers need global context)
Layers 4-43:  Wayfinder sparse (bulk of computation, sparse is sufficient)
Layers 44-46: Dense (final layers refine with global attention)
```

The `swap_X_attention_with_wayfinder()` function already accepts a layer selection.
Just pass `layers=[4,5,6,...,43]` to skip early/late layers.

---

## Phase 4: Custom Metal Kernel

> The real unlock. Replaces the Python per-head loop with 1 kernel dispatch.

### 4a. MLX Custom Metal Kernel API

MLX v0.30+ supports custom Metal kernels via `mx.fast.metal_kernel()`. This allows
inline Metal shader code that integrates with MLX's lazy evaluation system.

**Target kernel signature** (extends the plan in `docs/mlx_kernels_plan.md`):

```
wayfinder_sparse_attention_kernel(
    q: bfloat16/float16 [B, Hq, Tq, dh],     // Active queries
    k: bfloat16/float16 [B, Hkv, Tk, dh],     // Full KV cache
    v: bfloat16/float16 [B, Hkv, Tk, dh],
    perms: int32 [Hq, Tg],                     // Cycle permutations
    inv_perms: int32 [Hq, Tg],                 // Inverse permutations
    query_positions: int32 [Tq],               // Original positions of active queries
    window: int32,                              // Half-window size W
) -> bfloat16/float16 [B, Hq, Tq, dh]
```

**What this eliminates**:
- All Python loops (heads, query chunks)
- All mx.eval sync barriers
- All intermediate tensor allocations (scores, masks, weights)
- All gather/scatter through MLX's graph compiler
- Total kernel dispatches per layer: **1** (vs current ~220 graph nodes + 10 evals)

### 4b. Metal FlashAttention Block-Sparse Backend

Alternative to writing a custom kernel from scratch: adapt the community
Metal FlashAttention 2.0 project, which already has a block-sparse algorithm.

Precompute the attention mask from the Hamiltonian cycle permutation as a
block-sparse descriptor, then feed it to the block-sparse kernel. Less control,
but significantly less development effort.

### 4c. Fused Permute-Project Kernel

Fuse the index remapping into the Q/K/V projection kernels. The model's projection
layer outputs tensors that are already in cycle order. Zero additional memory
traffic for the permute step.

---

## Phase 5: Production Serving

> From research library to drop-in serving component.

### 5a. Unified Swap API

```python
import hcsa

result = hcsa.swap(
    model,
    config=hcsa.WayfinderConfig(
        window=64,
        strategy="random",
        seed=42,
        active_dense_threshold=16384,  # hybrid gate
    ),
    layers="auto",
    prewarm_lengths=[4096, 8192, 16384, 32768],
)

print(f"Replaced {result.num_replaced} of {result.num_total} attention layers")
print(f"Estimated memory reduction: {result.estimated_memory_reduction_pct:.1f}%")
```

**Auto-detection**: Inspect `model.__class__.__name__` and layer structure to
determine which model family. Support at minimum: Qwen, GLM, LLaMA, Mistral, GPT-2.

### 5b. vllm-mlx Plugin Integration

[vllm-mlx](https://github.com/waybarrios/vllm-mlx) (Jan 2026) provides
OpenAI-compatible serving with continuous batching on MLX. Integration path:

1. Write a model loader hook that swaps attention post-load
2. Ensure Wayfinder attention is compatible with vllm-mlx's `make_prompt_cache`
   and incremental KV cache update
3. Handle variable batch sizes (graph is batch-independent)
4. Support prompt caching / prefix sharing (graph caches are sequence-length keyed)

### 5c. PagedAttention Compatibility

vLLM-style paged KV cache stores K/V in fixed-size pages. Options:

**Option A**: Permute within pages. If W < page_size, all window neighbors are
within O(1) pages.

**Option B**: Virtual address mapping. Page table maps permuted ranks to physical
page locations.

**Option C**: Don't permute the KV cache at all. Use the current active-query
approach (logical mapping + physical gather). Post-Phase 4, the gather overhead
is hidden inside the fused kernel.

### 5d. KV Cache Quantization

Production serving typically quantizes the KV cache to INT8 or INT4. The permute
trick works with quantized data (permutation is index-based, dtype agnostic).
Dequantization happens after gathering W neighbors, so only O(Q * W * dh) values
are dequantized per query block instead of O(Q * T * dh).

**Known issue**: MLX's `QuantizedKVCache.update_and_fetch()` returns packed
weights/scales/biases as tuple trees. GLM's MLA extraction path slices dense key
tensors (`keys[..., :-qk_rope_head_dim]`), which is incompatible. This was
observed in EXP-20260207-230429. Requires integration work before KV quantization
is usable with Wayfinder.

### 5e. Graph Pre-Warming and Adaptive Warm-Up

```python
# At server startup
runtime.prewarm(
    seq_lengths=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    strategies=["random"],
    seeds=[42],
)

# During serving: adaptive extension
# If a request arrives at T=50000 (not pre-warmed),
# extend the nearest smaller cached graph incrementally
# using the O(1) splice method from Phase 1b
```

---

## Phase 6: Training from Scratch

> The ultimate validation: train a model natively with HCSA attention.

### 6a. The Quality Story

HCSA's positioning against other sparse approaches:

| Approach | Complexity | Exact softmax? | Content retrieval? | Hardware efficient? |
|----------|-----------|----------------|-------------------|-------------------|
| Dense | O(T^2) | Yes | Yes | Yes (fused SDPA) |
| Sliding Window | O(T*W) | Yes | No (local only) | Yes |
| SSM (Mamba) | O(T) | No | Poor | Yes |
| Linear Attn (RWKV) | O(T) | No | Approximate | Yes |
| HCSA | O(T*W) | Yes | Yes (cycle edges) | Pending (Phase 4) |

HCSA is the only approach that preserves exact softmax attention semantics,
achieves sub-quadratic complexity, AND provides long-range content-based retrieval
via cycle edges. The hardware efficiency column is honest: HCSA is not yet
hardware-efficient at the implementation level. Phase 4 (custom Metal kernel) is
required to make the "hardware efficient" claim truthful.

### 6b. Hybrid Architecture (Jamba-Style)

AI21's Jamba uses Mamba layers + dense attention layers. The natural extension:

```
Mamba layer       (smooth sequential processing, O(T))
HCSA layer        (sparse content-based retrieval, O(T*W))
Mamba layer
HCSA layer
...
Dense attn layer  (full global context, every Nth layer)
```

### 6c. Proof-of-Concept Training

Target: Train a 1-3B parameter model from scratch with HCSA attention on standard
benchmarks (C4, SlimPajama). Show competitive perplexity against:
- Dense baseline (same params, same data, same compute)
- Sliding window baseline (same window size, no cycle edges)
- Mamba baseline (same params)

**The key comparison**: HCSA vs sliding window with the same W. The cycle edges
should provide measurably better perplexity on tasks requiring long-range retrieval
(RULER, LongBench passkey, multi-hop QA). If this delta is significant, it
validates the entire approach.

### Existing Training Signal

Tiny-long training (TinyShakespeare, 1000 steps) shows:
- Wayfinder permute val ppl: 91.0 (retro off) / 79.9 (retro on, -12.2%)
- Retro backfill is a training-time feature (causal-safe default: off at inference)
- Throughput cost of retro: ~7%, memory cost: ~1.5%

---

## Appendix: Cost Model & Crossover Analysis

### The Crossover Equation

```
Dense cost per chunk:     C_dense  = c_d * Q * K
Wayfinder cost per chunk: C_sparse = c_s * Q * W + c_o

Crossover when C_dense = C_sparse:
K_cross = (c_s * W + c_o / Q) / c_d
```

Where:
- `c_d`: Cost per element of dense SDPA (highly optimized fused Metal kernel)
- `c_s`: Cost per element of sparse attention (includes Python loop overhead,
  gather cost, per-head-chunk eval barriers)
- `c_o`: Fixed overhead per chunk (Python dispatch, kernel launch)
- `W`: Window size (2 * window + 1 = 129 for window=64)
- `Q`: Query chunk size (4096 in benchmarks)

### Current Constants (Post-Matmul, from Threshold Sweep)

Measured from the GLM-4.7-Flash threshold sweep at T=32768, chunk=4096, 47 layers:

```
Stock chunked dense per chunk (avg):
  165.53s / 8 chunks = 20.7s per chunk
  c_d_chunk = 20.7 / 47 = 0.44s per layer per chunk

Wayfinder per chunk (from threshold=0, avg):
  267.87s / 8 chunks = 33.5s per chunk
  c_s_chunk = 33.5 / 47 = 0.71s per layer per chunk

Overhead per Hamiltonian chunk vs dense chunk:
  (267.87 - 164.15) / 8 = 12.97s per chunk
  Per layer: 12.97 / 47 = 0.28s additional per Hamiltonian chunk per layer
```

The per-Hamiltonian-chunk overhead is ~0.28s per layer. This is the constant factor
that must be reduced below zero for a latency crossover. At current constants:

```
Dense chunk cost grows with K:    c_d * Q * K
Wayfinder chunk cost is ~flat:    c_s * Q * W  (W=129, fixed)

For crossover: c_d * Q * K_cross = c_s * Q * W
K_cross = (c_s / c_d) * W

From the data, dense chunk times grow roughly linearly with K:
  Chunk 0 (K=4096):  ~22s dense, but Wayfinder is slower
  Chunk 7 (K=32768): ~33s dense, Wayfinder still slower at ~45s

The ratio per chunk: Wayfinder/Dense goes from ~3.3x (chunk 0) to ~1.35x (chunk 7)
If extrapolated linearly, crossover occurs at K ~ 44000-50000
```

### What the Threshold Sweep Tells Us

The threshold sweep is the most informative data point for the crossover question:

```
threshold=0     (all permute):  267.87s, 22.86% mem saved
threshold=32768 (all dense):    164.15s, 10.06% mem saved

Per Hamiltonian chunk penalty: (267.87 - 164.15) / 8 = 12.97s
Per dense chunk saved: same

The penalty is CONSTANT per chunk -- it doesn't depend on K in a way that
inverts. This means the crossover is NOT within the K=4096..32768 range.
```

The monotonic improvement from threshold=0 to threshold=32768 confirms that at
T=32768, **every chunk** is faster with dense than with Hamiltonian permute.
The crossover point (where Hamiltonian chunks become individually faster) has
not been reached.

### Projected Constants After Phase 2 Remaining (2b+2c+2d)

Eliminating the Python per-head loop, enabling fused SDPA via pre-permute, and
removing eval barriers should reduce the per-element sparse cost dramatically:

```
Current c_s / c_d ratio: ~1.6x (from 0.71/0.44 per layer per chunk)
Projected after 2b-2d:   ~1.05-1.15x (gather overhead + mask handling only)

This would put the per-chunk penalty at:
  (1.1 * c_d - c_d) * Q * W_avg = 0.1 * c_d * Q * W_avg

vs per-chunk dense advantage at K:
  c_d * Q * (K - W_avg)

Crossover: K where 0.1 * W_avg > K - W_avg
  -> K_cross = 1.1 * W_avg = 1.1 * 129 ~ 142 tokens

This is optimistic. A more conservative estimate of c_s/c_d = 1.3x gives:
  K_cross = 1.3 * 129 / (1 - 1/1.3) ~ 560 tokens
```

Even conservatively, the crossover drops to K < 1000 after Phase 2 completion.
This would make Hamiltonian chunks individually faster than dense for essentially
all chunks beyond the first 1-2 in any long-context prefill.

### Projected Constants After Phase 4 (Custom Metal Kernel)

With a fused Metal kernel:

```
Projected c_s ~ c_d * 1.2-1.5x  (memory access pattern overhead only)
K_cross ~ 150-200 tokens
```

At this point, Wayfinder is strictly faster than dense for sequences beyond ~200
tokens.

### The Honest Assessment

**Today (post-matmul, pre-head-vectorization):**
- HCSA is a memory optimization that trades latency for memory at T<=32K
- The hybrid threshold gate makes this tradeoff controllable and smooth
- Recommended operating point: threshold=16384, getting 22% memory reduction
  at 1.19x latency vs stock chunked dense (197s vs 166s)

**After Phase 2 completion (estimated):**
- Per-chunk latency crossover likely at K=500-1000
- At T=32768 chunk=4096, chunks 1-7 (K=8192..32768) should all be faster with
  Hamiltonian than dense
- Total prefill should be faster than stock chunked dense AND use less memory
- This is the first milestone where HCSA is strictly better on both axes

**After Phase 4 (custom Metal kernel):**
- Crossover at K~200
- Competitive with or faster than monolithic dense at T>4096
- Memory savings scale with T while latency improves

### Levers for Lowering the Crossover

Ranked by expected impact:

1. **Vectorize heads + enable fused SDPA** (Phase 2b+2c): Eliminate the Python
   per-head loop. This is ~80% of the remaining constant-factor gap. A single
   batched SDPA call replaces ~220 Python iterations per layer.

2. **Remove eval barriers** (Phase 2d): Eliminate ~10 GPU sync points per layer.
   Unlocked by head vectorization.

3. **Custom Metal kernel** (Phase 4): Fuse permute+gather+attend+scatter into one
   kernel dispatch. Eliminates all remaining Python-level overhead.

4. **Better topologies** (Phase 3): Reduce effective W needed for same routing
   quality. Expander graphs may achieve equivalent information flow with smaller
   windows.

5. **Increase T**: The crossover is K-dependent. At longer sequences, more chunks
   have K > K_cross. The T=65536 benchmark (not yet re-run post-fix) should show
   a more favorable ratio. Re-running at T=65536 with current implementation is
   the next empirical step.

---

## Summary: The Path to Production

```
Phase 0 (1-2 weeks)    Structural cleanup, deduplicate integrations
Phase 1 (remaining)    Prewarm, incremental decode extension, LRU eviction
Phase 2 (remaining)    Vectorize heads (2b+2c+2d) -- the critical path
Phase 3 (ongoing)      Explore better topologies, benchmark against random cycles
Phase 4 (4-6 weeks)    Custom Metal kernel -- the real unlock
Phase 5 (2-3 weeks)    vllm-mlx plugin, serving API, KV quantization
Phase 6 (ongoing)      Train from scratch, validate quality story
```

**The critical path is Phase 2b+2c+2d.** These three items -- pre-permute for
Tg>Tk, head vectorization, and eval barrier removal -- are the difference between
"HCSA is a memory optimization" and "HCSA is faster AND uses less memory." They
are pure engineering work with no research risk.

Phase 4 is the long-term unlock but requires Metal shader development expertise.
Phase 2 can be done entirely within MLX's Python API.

**Next empirical step**: Re-run the threshold sweep at T=65536 with thresholds
{0, 16384, 32768} to see if the per-chunk crossover is closer at longer contexts,
and to validate that the 26% memory reduction observed pre-fix is preserved.
