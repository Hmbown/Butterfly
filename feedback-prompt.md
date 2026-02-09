# HCSA/Wayfinder: Fused Kernel + Quality Validation Prompt

> **Context:** This prompt is for an AI assistant working on the HCSA (Hamiltonian Cycle Sparse Attention) codebase. The primary goal is **fused Metal kernels that distribute compute across Apple Silicon in a Hamiltonian pattern** — eliminating the Python-level serialization that currently bottlenecks performance. Quality validation is the secondary gate. The codebase lives at the repo root with core library in `hcsa/`, MLX backend in `hcsa/mlx/`, integrations in `hcsa/integrations/`, and a proven LLM-assisted kernel discovery system in the sibling project at `/Volumes/VIXinSSD/ZMLX/`.

---

## The Core Problem: Serialized Compute on Parallel Hardware

HCSA replaces dense O(T^2) attention with O(T*W) sparse attention using Hamiltonian cycles. The algorithmic complexity is right. But the implementation **feeds Apple Silicon through a straw**:

### Current Bottlenecks in `hcsa/mlx/attention.py`

**1. Per-head Python loops (lines 230, 987, 1266)**
```python
for h in range(H):           # sparse_gather_attention — one head at a time
for h in range(h0, h1):      # active_batched — per-head inside each chunk
for h in range(self.n_heads): # graph construction — serial
```
Each iteration launches a tiny GPU kernel, waits for it, returns to Python, launches the next. On M-series chips with 800+ GB/s bandwidth and 32-wide SIMD, this is criminal underutilization.

**2. Forced synchronization inside loops (lines 813, 1094)**
```python
mx.eval(y_h)      # batched permute: blocks GPU after EACH head chunk
mx.eval(y_local)   # active path: same — kills pipelining
```
MLX's lazy evaluation is designed to build a large compute graph and execute it efficiently. These mid-loop `mx.eval` calls flush the pipeline after every chunk, turning one big efficient dispatch into H separate tiny ones.

**3. Multi-step attention as separate kernel launches**
The permute-window path does 4 separate operations per chunk:
1. Permute Q/K/V into cycle order (`mx.take_along_axis`)
2. Slice local windows (`k_pi[:, pi_idx_clamped]`)
3. Matmul + softmax (`mx.matmul` / `mx.fast.scaled_dot_product_attention`)
4. Unpermute output (`mx.take_along_axis` with inverse)

Each is a separate Metal kernel launch with its own memory round-trip. Data flows GPU→memory→GPU→memory→GPU→memory→GPU instead of staying in threadgroup memory.

**4. Graph construction is CPU-blocking**
All cycle permutations and neighbor indexing run in NumPy on CPU. Zero overlap with GPU attention compute. The Metal kernel stub at `hcsa/mlx/kernels/metal/__init__.py` literally raises `NotImplementedError`.

### The Fix: Hamiltonian Compute Distribution

The Hamiltonian cycle isn't just the attention pattern — it's the **model for how to distribute compute**. A Hamiltonian path visits every node exactly once with no revisits. The fused kernel should do the same: each GPU thread visits each element in its workgroup exactly once, in cycle order, with no redundant memory loads.

**Target architecture:**
- One Metal kernel launch per layer (not per head, not per chunk)
- All heads processed in parallel across threadgroups
- Permute + window-gather + matmul + unpermute fused in threadgroup memory
- Cycle permutation indices live in constant/threadgroup memory, not reloaded per op
- No Python-level loops, no mid-computation `mx.eval`

---

## Empirical Baseline: What the E2E Validation Shows

A full E2E validation pass was completed on 2026-02-08. Key findings from `benchmarks/mlx/e2e_validation_20260208_192216/`:

### Where HCSA dominates

| Scenario | Joint Utility | Key Delta |
|----------|:---:|-----------|
| GLM consumer @ 8K | **8.28** | TTFT: 99.6s -> 0.19s (-99.8%), decode tok/s: 1.6 -> 44.0 |
| GLM chunked 32K (thr49152) | **3.91** | Prefill: +291% tok/s |
| GLM chunked 65K (thr49152) | **1.88** | Prefill: +68% tok/s, memory: -10.8% |
| Qwen block 32K | **2.20** | +108% tok/s, -5.8% memory |
| GLM attention-swap micro 8K | **4.49** | +285% tok/s, -14.2% memory |

### Where HCSA loses

| Scenario | Joint Utility | Why |
|----------|:---:|-----|
| Tiny training (short) | 0.71 | -27% tok/s, +2% memory. Overhead dominates when model is tiny. |
| Tiny training (long) | 0.43 | -46% tok/s, +26% memory. Graph build cost is proportionally huge. |
| GLM consumer 2K | 0.91 | -10% E2E. Graph construction overhead exceeds attention savings at short context. |

### Critical observations for kernel work

1. **Graph build cost is severe.** Qwen @ 32K: 45.7 seconds first-call build. Even @ 8K: 4.9s. This CPU work blocks everything. K3 (GPU graph construct) is more important than initially assumed.

2. **Consumer E2E memory is flat.** Peak memory reduction is ~0% in consumer mode because KV cache preallocation dominates. HCSA's attention-level savings are invisible at the system level. This means: either the fused kernels need to shrink the attention buffers dramatically, or a complementary approach (like TTT-based KV compression) is needed to move system-level memory.

3. **Variance is high.** The tuned 65K campaign swung between 548s and 813s across two runs (std=132s). Environment/thermal sensitivity is real — benchmark results need >= 3 repeats.

4. **Dense fails at 32K on 36GB.** The dense baseline at GLM seq=32768 attention-swap tried to allocate 43GB in one Metal buffer — physically impossible on 36GB. HCSA's sparse buffers sidestep this limit, which is an underappreciated structural advantage.

5. **Quality: 3/6 = 3/6.** Identical accuracy on 6 trivial tasks. No catastrophic collapse, but insufficient to claim parity. Same tasks fail on both (math, one lookup). Deeper quality validation (Q1-Q5 below) remains essential.

6. **Idea 5 (regular partition) failed.** Throughput direction was unstable across repeats. Random cycle strategy remains the robust default.

7. **The wins are already real despite Python bottleneck.** All of the above numbers include the serialization overhead. Fused kernels would amplify every win and reduce every loss (especially the short-context overhead where graph build dominates).

### Kimi K2.5 Projection

The verdict includes a falsifiable hypothesis for the North Star target (Kimi K2.5, 256K context, 512GB Mac Studio):
- Projected speedup: 1.25x-2.3x E2E
- Projected memory reduction: 5%-20%
- Falsifiers: reject if < 1.15x speedup, < 5% memory reduction, or expert paging I/O dominates > 30% wall-time

---

## Phase 1: Fused Metal Kernels via ZMLX Discover

### The Tool: LLM-Guided Kernel Search

The sibling project ZMLX (`/Volumes/VIXinSSD/ZMLX/`) has a **proven LLM-assisted kernel discovery system** that uses PUCT tree search to find optimized Metal kernels. It has already produced real speedups:

| Kernel | Baseline | Discovered | Speedup |
|--------|----------|------------|---------|
| `glm_fused_swiglu` | 141.3 us | 102.8 us | **1.37x** |
| `ttt_linear_decode` | 381.8 us | 222.4 us | **1.72x** |
| `glm_rmsnorm` | — | — | **1.12x** |

**Workflow:**
1. Define a `SearchSpace` in `targets.py` with: baseline Metal source, Python reference function, I/O specs, grid computation, correctness constraints
2. Run `python -m zmlx.discover search <target> --llm claude-code --steps 10`
3. The system calls Claude to generate Metal kernel variants, compiles them, checks correctness against the Python reference, benchmarks timing, and uses PUCT to explore the optimization space
4. Export the winning kernel

**Key files:**
- `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/targets.py` — Target registration pattern
- `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/prompts.py` — System/user prompt engineering
- `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/tree.py` — PUCT search tree
- `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/evaluate.py` — Compile + correctness + benchmark
- `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/llm.py` — LLM backends (ClaudeCodeBackend shells to local `claude` CLI)
- `/Volumes/VIXinSSD/ZMLX/docs/DISCOVER_PLAYBOOK.md` — Full workflow documentation

### Target Kernels to Discover

Define these as ZMLX Discover targets. Each gets its own search space with reference implementation from the current Python code.

#### Target K1: `hcsa_permute_window_fused`

**The big one.** Fuses the entire permute-window attention path into a single Metal kernel.

**Current Python reference** (from `attention.py:260-350`):
```
Input:  q_h [B, T, dh], k_h [B, T, dh], v_h [B, T, dh], perm [T], window W
Output: y_h [B, T, dh]

Steps:
1. q_pi = q_h[:, perm]          # permute Q into cycle order
2. k_pi = k_h[:, perm]          # permute K
3. v_pi = v_h[:, perm]          # permute V
4. For each position i in cycle-order:
   a. Gather W neighbors: k_win = k_pi[:, i-w:i+w+1]
   b. scores = q_pi[:, i] @ k_win.T / sqrt(dh)
   c. Apply causal mask (original positions)
   d. softmax
   e. y_pi[:, i] = weights @ v_win
5. y_h = y_pi[:, inv_perm]      # unpermute output
```

**Metal kernel structure (seed for Discover):**
```
- Grid: (T, H, 1) — one thread per (position, head) pair
- Threadgroup: (W, 1, 1) — W threads cooperate on one position's window
- Each threadgroup:
  1. Load perm[i] from constant memory → know original position
  2. Load q_pi[i] into registers (dh floats)
  3. Cooperative load: each thread loads one k_pi[i + offset] into threadgroup memory
  4. Each thread computes one score = dot(q, k_neighbor) / sqrt(dh)
  5. Causal mask: compare original positions via perm[]
  6. SIMD reduce for softmax normalization
  7. Cooperative load v_pi neighbors, weighted sum
  8. Write output to y[inv_perm[i]]
```

**Hamiltonian compute pattern:** Thread i processes cycle-position i. Its window neighbors are cycle-positions i-w through i+w — contiguous in threadgroup layout. The permutation itself makes the sparse attention *physically local* in GPU memory. This is the key insight: the Hamiltonian cycle converts random sparse access into contiguous sequential access.

**Correctness reference:** `permute_cycle_window_attention_single()` at `attention.py:260`

**Concrete test shapes:**
- T=512, dh=64, W=129 (window=64), H=8 (GPT-2 scale)
- T=4096, dh=128, W=129, H=32 (GLM scale)
- T=8192, dh=128, W=129, H=40 (Qwen3 scale)

#### Target K2: `hcsa_sparse_gather_fused`

Fuses the sparse-gather reference path: gather neighbors by index, compute attention, all in one kernel.

**Current Python reference** (from `attention.py:170-257`):
```
Input:  q [B, H, T, dh], k [B, H, T, dh], v [B, H, T, dh],
        neigh_idx [H, T, D], edge_type [H, T, D]
Output: y [B, H, T, dh]

Steps (per head, per position):
1. Gather: k_g = k[:, idx]  for D neighbor indices
2. scores = q @ k_g.T / sqrt(dh)
3. Apply causal mask + padding mask
4. Optional edge-type bias
5. softmax over D neighbors
6. y = weights @ v_g
```

**Metal kernel structure:**
```
- Grid: (T, H, B) — one threadgroup per (position, head, batch)
- Threadgroup: (D, 1, 1) — D threads, one per neighbor
- Each thread loads one neighbor's K and V via neigh_idx
- SIMD reduction for softmax
- Weighted V accumulation
```

This kernel is simpler than K1 but handles the general case (arbitrary neighbor lists, not just cycle-window).

#### Target K3: `hcsa_graph_construct`

Move cycle permutation generation from CPU to GPU.

**Current Python reference:** `hcsa/cycles.py:random_cycle()` — Fisher-Yates shuffle producing a random permutation.

**Metal kernel approach:**
- Use parallel random number generation (each position generates its own random swap target)
- Or: generate random values per position, argsort on GPU to get permutation
- Pre-compute `inv_perm = argsort(perm)` and `pi_idx_clamped` on GPU
- Eliminates the CPU→GPU sync for graph construction

**Note:** This is lower priority than K1/K2 unless profiling shows graph_build_ms > 10% of total at T=65K.

#### Target K4: `hcsa_active_row_fused`

The chunked-prefill kernel for Q_len < K_len (the dense fallback bug path).

**Current state:** `glm_mlx.py:259` falls back to dense MLA when Q_len != K_len. The `wayfinder_permute_window_attention_active_batched` function at `attention.py:866` implements the logic in Python but with per-head loops.

**Metal kernel approach:**
- Accept query_positions array (active rows)
- For each active query, look up its cycle-rank, compute window in cycle space
- Gather K/V from cache at cycle-neighbor positions
- Compute attention in one fused pass
- This directly fixes the chunked prefill dense fallback bug

#### Target K5: `hcsa_wayfinder_ttt_fused`

**The novel fusion.** Combines HCSA sparse attention with Test-Time Training (TTT) weight updates, both operating in cycle-permuted order within a single kernel.

**Motivation:** The E2E validation shows that HCSA's attention-level memory savings are invisible at the consumer level because KV cache preallocation dominates. TTT-Linear replaces the KV cache with a learned weight matrix that compresses the full sequence — O(F^2) memory instead of O(T*dh). Combined with HCSA's O(T*W) attention, this creates a system where both compute AND memory scale subquadratically.

The Hamiltonian cycle is also the TTT weight update order. The same permutation that makes attention contiguous also structures the gradient flow:

**Combined algorithm (per layer, per head):**
```
Input:  x [B, T, dh] — token representations
        W_ttt [F, F] — TTT inner model weights (persistent across tokens)
        perm [T] — Hamiltonian cycle permutation
Output: y [B, T, dh] — attended + TTT-compressed output

For each position i in cycle order (permuted space):
  1. HCSA Attention:
     a. q = x[perm[i]], k_win = x[perm[i-w:i+w+1]]
     b. scores = q @ k_win.T / sqrt(dh), causal mask, softmax
     c. y_attn = weights @ v_win

  2. TTT Forward + Update:
     a. z = x[perm[i]] @ W_ttt                   # inner model forward
     b. target = y_attn                            # attention output as target
     c. loss = ||z - target||^2                    # L2 loss
     d. grad = d(loss)/d(W_ttt)                    # inner gradient
     e. W_ttt -= lr * grad                         # weight update
     f. z_updated = x[perm[i]] @ W_ttt             # updated forward

  3. Combine:
     y[perm[i]] = y_attn + alpha * z_updated       # residual combination
```

**Why cycle-order TTT is powerful:**
- Token at cycle-position j sees weight updates from cycle-positions 0..j-1
- These predecessors are from *random* original positions (scattered across the sequence)
- After T updates, W_ttt has seen every token exactly once, with information from the full sequence mixed in cycle order
- The weight matrix becomes a *global compressed summary* that compensates for HCSA's local attention window
- This directly addresses the causal diameter concern: distant information propagates through W_ttt, not through multi-hop attention paths

**Causal correctness options:**
- Option A (causal TTT): Only apply weight updates from cycle-predecessors whose original index <= current original index. Safe but some updates get skipped.
- Option B (bidirectional TTT for prefill): Let W_ttt see all tokens during prefill (all positions are "past" at prefill time). Fix W_ttt for decoding. The weight matrix acts as a read-only compressed cache. This is the practical choice for inference.

**Metal kernel structure:**
```
- Same grid as K1: (T, H, 1)
- Threadgroup: (max(W, F), 1, 1)
- Each threadgroup processes one cycle-position:
  Steps 1a-1c: identical to K1 (attention in threadgroup memory)
  Step 2a: matmul x @ W_ttt in registers (F is small, ~64)
  Steps 2b-2e: L2 loss, gradient, weight update — all register/threadgroup ops
  Step 2f: updated forward with new W_ttt
  Step 3: combine and write output
- W_ttt lives in threadgroup memory, updated in-place per position
- Total extra compute: O(F^2) per position = O(T*F^2) per layer
```

**Complexity:** O(T * (W*dh + F^2)) per layer. With W=129, dh=128, F=64: attention=16,512, TTT=4,096. TTT adds ~25% compute for global information propagation. Memory for W_ttt: F^2 * sizeof(float) = 16KB per head — trivially fits in threadgroup memory.

**Reference implementation:** ZMLX TTT-Linear at `/Volumes/VIXinSSD/ZMLX/src/zmlx/ttt/linear.py` (MLX reference) and `/Volumes/VIXinSSD/ZMLX/src/zmlx/ttt/kernel.py` (12-step fused Metal kernel). The HCSA-TTT kernel extends the TTT pattern by embedding it within the cycle-order traversal.

**Concrete test shapes:**
- T=512, dh=64, F=64, W=129, H=8 (prototype scale)
- T=4096, dh=128, F=64, W=129, H=32 (GLM scale)

**Open questions for experimentation:**
- Does the inner learning rate need to scale with cycle-order position? (Later positions have seen more updates.)
- Does bidirectional TTT during prefill (Option B) outperform causal TTT (Option A)?
- What is the optimal F (TTT feature dim) vs W (window size) tradeoff? Larger F compresses more but costs more compute.
- Can W_ttt replace the KV cache entirely for decoding, or is it complementary?

### How to Register These Targets

Create `hcsa/discover/targets.py` following the ZMLX pattern:

```python
from zmlx.discover.candidates import InputSpec, KernelCandidate, KernelSpec, OutputSpec, SearchSpace

def hcsa_permute_window_target(T: int = 4096, dh: int = 128, window: int = 64) -> SearchSpace:
    W = 2 * window + 1
    source = """
        // Seed kernel: naive permute-window attention
        constexpr uint DH = {dh};
        constexpr uint W = {W};
        uint pos = thread_position_in_grid.x;   // cycle-order position
        uint head = thread_position_in_grid.y;   // head index
        // ... baseline implementation
    """

    ref_python = """
import mlx.core as mx
from hcsa.mlx.attention import permute_cycle_window_attention_single
def reference(*inputs):
    q, k, v, perm = inputs
    # ... call existing Python implementation
"""

    return SearchSpace(
        name="hcsa_permute_window",
        description=f"Fused permute-window attention: T={T}, dh={dh}, W={W}",
        # ... full spec following ZMLX pattern
    )
```

### Running Discovery

```bash
# From ZMLX project (or adapt to run from HCSA)
python -m zmlx.discover search hcsa_permute_window --llm claude-code --steps 15 -v

# With multiple backends for diversity
python -m zmlx.discover autorun \
  --targets hcsa_permute_window hcsa_sparse_gather hcsa_active_row \
  --backends "claude-code,claude" \
  --steps 10

# Export winner
python -m zmlx.discover export discover_sessions/hcsa_permute_window_session.json \
  -o hcsa/mlx/kernels/metal/permute_window.py
```

### Integration into HCSA

Once a kernel is discovered and validated:

1. Place the Metal source in `hcsa/mlx/kernels/metal/`
2. Wire it into `attention.py` via `mx.fast.metal_kernel()`:
```python
kernel = mx.fast.metal_kernel(
    name="hcsa_permute_window_fused",
    source=DISCOVERED_SOURCE,
    input_names=("q", "k", "v", "perm", "inv_perm"),
    output_names=("out",),
)
```
3. Add a dispatch check: use fused kernel when available, fall back to Python path otherwise
4. Benchmark end-to-end on GLM-4.7-Flash at T=32K and T=65K

### Discover System Prompt Extension

The ZMLX system prompt at `discover/prompts.py` should be extended with HCSA-specific Metal expertise:

```
## HCSA-Specific Optimization Strategies

- **Hamiltonian locality**: The permutation converts random sparse access into
  contiguous sequential access. Exploit this — cycle-neighbors at offsets +/-1
  in permuted space are the most important edges.

- **Causal mask via original-index comparison**: In cycle-permuted space,
  causality requires comparing original positions: perm[i] vs perm[j].
  Store perm[] in constant memory for fast broadcast.

- **Window = local in permuted space**: After permutation, the W-sized window
  is physically contiguous. Use threadgroup memory for the window, not
  scattered global loads.

- **GQA/MQA awareness**: K/V may have fewer heads than Q (GLM uses Hkv=1).
  The kernel must handle head index mapping: kv_head = q_head // kv_repeat.

- **32-wide SIMD on Apple Silicon**: Window sizes W that are multiples of 32
  get free SIMD utilization. W=129 (window=64) wastes 31 lanes. Consider
  W=128 (window=63) or W=96 (window=47) for better occupancy.

- **Non-power-of-2 window sizes are fine on Metal**: Unlike CUDA tensor cores
  that prefer 64/128, Apple Silicon's SIMD groups work with any width.
  Tune W for SIMD alignment (multiples of 32), not power-of-2.

- **4-bit quantization fusion**: For quantized models, fuse dequantization
  into the gather. Don't materialize full-precision K/V — dequantize on-the-fly
  during the window gather. This saves memory bandwidth proportional to the
  sparsity ratio (only W/T of K/V is accessed).
```

---

## Phase 2: Eliminate Python-Level Serialization

Even before custom Metal kernels are ready, remove the worst serialization:

### Refactor 2A: Remove mid-loop `mx.eval`

**Files:** `hcsa/mlx/attention.py` lines 813, 1094

The `mx.eval(y_h)` and `mx.eval(y_local)` calls were added to limit peak memory by forcing each chunk to complete before the next starts. But this kills pipelining.

**Fix:** Remove the `mx.eval` calls. Let MLX's lazy evaluation build the full compute graph across all head chunks. If memory pressure is a concern, use MLX's stream-based memory management instead of manual eval fencing.

**Risk:** Peak memory may increase if all chunks are live simultaneously. Monitor with `mx.metal.get_peak_memory()`. If it blows the budget, add eval at wider intervals (every 4 chunks instead of every 1).

### Refactor 2B: Vectorize the sparse-gather path

**File:** `hcsa/mlx/attention.py` line 230

The `for h in range(H)` loop in `sparse_gather_attention` processes one head at a time. Replace with batched ops:

```python
# Before (serialized):
for h in range(H):
    k_g = k_h[:, idx_h]  # [B, T, D, dh] — one head
    ...

# After (vectorized):
k_g = mx.take(k, s_idx, axis=2)  # [B, H, T, D, dh] — all heads at once
scores = mx.sum(q[:, :, :, None, :] * k_g, axis=-1) / sqrt(dh)  # [B, H, T, D]
w = stable_masked_softmax(scores, mask[None], axis=-1)
y = mx.sum(w[..., None] * v_g, axis=3)  # [B, H, T, dh]
```

This turns H sequential GPU kernel launches into one large launch.

### Refactor 2C: Vectorize the active-row path

**File:** `hcsa/mlx/attention.py` line 987

Same problem: `for h in range(h0, h1)` inside each head chunk. The per-head loop does individual `mx.take` calls. Batch all heads together using advanced indexing.

### Refactor 2D: Async graph construction

**File:** `hcsa/mlx/attention.py` line 1266+

Currently: build graph for all heads → convert to MLX → run attention.

Better: Build graph for head 0, start attention for head 0 with `mx.async_eval`, build graph for head 1, start attention for head 1, etc. Graph construction on CPU overlaps with attention compute on GPU.

Even better (with K3 kernel): move graph construction to GPU entirely.

---

## Phase 3: Quality Validation

Quality validation gates whether the work is *useful*, but doesn't need to gate kernel development. Kernel correctness is validated against the Python reference by the Discover system. Quality (does sparse attention degrade model output?) is a separate question that can run in parallel with kernel work.

### Experiment Q1: Perplexity on Real Corpora (swap-in)
- Models: GLM-4.7-Flash-4bit, Qwen3-1.7B-4bit
- Corpus: WikiText-2 validation set
- Protocol: Dense forward pass vs HCSA swap, measure cross-entropy loss -> perplexity
- Sweep window sizes: w=32, 48, 64, 96, 128
- Success gate: < 5% perplexity degradation at w=64
- Build on `scripts/eval_gpt2_quality.py`

### Experiment Q2: Passkey Retrieval
- Synthetic document with embedded passkey at random position
- Vary context length: T=2K, 4K, 8K, 16K, 32K, 65K
- Vary position: 5%, 50%, 90% of context
- 10 trials per config, report accuracy
- Success gate: > 95% where dense succeeds
- **Confound warning:** Chunked prefill dense fallback bug may mask HCSA's true behavior at long lengths

### Experiment Q3: Window-Only Baseline
- Compare HCSA (window + cycle + landmarks) vs window-only (no cycle)
- If window-only matches HCSA quality, the cycle adds complexity without value
- If HCSA wins, the cycle earns its keep
- Run with num_cycles=0 to disable

### Experiment Q4: Attention Coverage Analysis
- Record dense attention weights for GLM-4.7-Flash on representative inputs (T=4096)
- Compute what fraction of attention mass falls within the HCSA neighbor set
- If coverage > 90%, swap-in is likely safe; if < 70%, critical info is dropped
- This explains *why* quality holds or doesn't

### Experiment Q5: Effective Causal Diameter
- Build HCSA graph, apply causal masking (remove j > i edges)
- Compute reachability, effective diameter, bottleneck positions
- Compare to log(T) theoretical prediction
- Validates the theoretical foundation of using Hamiltonian cycles under causality

---

## Phase 4: Apple Silicon-Specific Optimizations

These become relevant once fused kernels exist and quality is validated.

### Opt 1: Non-Power-of-2 Window Sizes
- Apple Silicon SIMD = 32 lanes. Test w=47 (W=95, near 3*32), w=63 (W=127, near 4*32), w=79 (W=159, near 5*32)
- Compare throughput against w=32, 64, 128
- Non-power-of-2 sizes that would destroy CUDA occupancy may work fine on Metal

### Opt 2: Quantization-Aware Sparse Gather
- With 4-bit models, K/V is stored quantized. Current path dequantizes all of K before sparse gather
- Fused dequant-gather kernel: only dequantize the W/T fraction that's actually accessed
- At w=64, T=65536: only 0.2% of K is gathered → 99.8% of dequantization is wasted
- This is potentially a massive bandwidth win for the fused kernel

### Opt 3: `mx.async_eval` for Expert Paging
- GLM-4.7 has MoE FFN — experts can be paged from SSD
- HCSA's O(T*W) attention is the "latency hiding window" for expert loading
- Profile: is attention_time > expert_load_time? If so, async overlap hides loading completely
- Pin attention layers + active experts in HBM, page cold experts from NVMe

### Opt 4: Memory Aliasing for Permuted Views
- MLX uses zero-copy buffers on unified memory
- Permuted Q/K/V views could alias the original buffers via strided access
- Eliminates the `mx.take_along_axis` materialization step entirely
- Requires fused kernel that reads Q/K/V through permutation indirection

---

## Priority Order

| Priority | What | Why |
|----------|------|-----|
| **P0** | K1: `hcsa_permute_window_fused` via Discover | The primary kernel — eliminates all serialization |
| **P0** | Refactors 2A-2B: Remove `mx.eval` + vectorize | Immediate wins without custom kernels |
| **P1** | K4: `hcsa_active_row_fused` | Fixes the chunked prefill dense fallback bug |
| **P1** | K3: `hcsa_graph_construct` | E2E shows 45.7s graph build at 32K — this is a top bottleneck |
| **P1** | Q1-Q2: Perplexity + Passkey retrieval | Quality gates before production use |
| **P2** | K5: `hcsa_wayfinder_ttt_fused` | Novel fusion — addresses system-level memory (KV cache) and causal diameter |
| **P2** | K2: `hcsa_sparse_gather_fused` | General-case kernel for non-permute path |
| **P2** | Q3-Q4: Window-only baseline + attention coverage | Validates the cycle's contribution |
| **P3** | Opt 1-2: Window tuning + quantization fusion | Performance tuning |
| **P3** | Opt 3-4: Expert paging + memory aliasing | Advanced optimizations |
| **P4** | Q5: Causal diameter analysis | Theoretical validation |

---

## Key Files

| File | Role |
|------|------|
| `hcsa/mlx/attention.py` | **Primary target.** All attention paths, graph cache, Python loops to eliminate |
| `hcsa/mlx/kernels/metal/__init__.py` | Metal kernel stub — wire discovered kernels here |
| `hcsa/integrations/glm_mlx.py` | GLM integration (chunked prefill dense fallback at line 259) |
| `hcsa/integrations/qwen_mlx.py` | Qwen3 integration, shared `_QwenGraphRuntime` |
| `hcsa/graph/abi.py` | Graph ABI: `WayfinderGraphABI`, `EdgeType` enum |
| `hcsa/topology/core.py` | Topology runtime: graph construction dispatch |
| `hcsa/cycles.py` | Cycle algorithms: random, greedy, online_insertion |
| `/Volumes/VIXinSSD/ZMLX/src/zmlx/discover/` | **Discover system** — targets, prompts, tree, evaluate, llm |
| `/Volumes/VIXinSSD/ZMLX/docs/DISCOVER_PLAYBOOK.md` | Discover workflow documentation |
| `scripts/eval_gpt2_quality.py` | Existing quality eval (extend for Q1) |
| `notes/LAB_NOTEBOOK.md` | Lab notebook |
| `notes/experiments.ndjson` | Structured experiment log |

---

## Known Bug: Chunked Prefill Dense Fallback

`glm_mlx.py:259` has a guard: when Q_len != K_len, all chunks (except the first) fall back to O(T^2) dense MLA instead of O(T*W) HCSA sparse. This is the root cause of the 6.9x latency regression at T=65536 with chunk=4096. The K4 fused kernel directly fixes this by implementing Q_len < K_len support in Metal. Until K4 is ready, test with chunk_size >= seq_len to avoid chunking.

---

## Decision Gates

**After K1 kernel + Refactors 2A-2B:**
- Benchmark end-to-end on GLM-4.7-Flash at T=32K. If > 2x faster than current Python path: proceed to K4, quality validation.
- If < 1.5x: the kernel needs more Discover iterations or architectural rethinking.

**After Q1-Q2 quality validation:**
- ppl degradation < 5% AND passkey > 95%: Full steam ahead. Ship it.
- ppl 5-15% OR passkey 80-95%: Tune window, add cycles, try greedy strategy.
- ppl > 15% OR passkey < 80%: Swap-in is dead. Retrain from scratch.

---

## Output Format

For each piece of work, produce:
1. **Kernels:** Metal source in `hcsa/mlx/kernels/metal/`, Discover session JSON in `discover_sessions/`
2. **Refactors:** Direct edits to `hcsa/mlx/attention.py` with before/after benchmarks
3. **Quality experiments:** Script in `scripts/eval_<name>.py`, results in `benchmarks/mlx/<model>/quality_<name>.json`
4. **Lab entries:** Summary in `notes/experiments.ndjson` with `{hypothesis, metric, result, decision}`
