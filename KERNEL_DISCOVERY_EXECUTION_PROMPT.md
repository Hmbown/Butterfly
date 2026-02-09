# Kernel Discovery Execution: K4 Active-Row Fused + K1 Permute-Window

You are continuing work in `/Volumes/VIXinSSD/wayfinder`.

**Companion document**: `HAMILTONIAN_IMPLEMENTATION_PROMPT.md` covers the mathematical/topology fixes (circular windowing, union multigraph, principled d selection). This prompt covers the **Metal kernel discovery and integration** work.

---

## Feasibility Stance

This is feasible. Expect engineering iteration and benchmark variance; do not treat first-run results as final. Proceed with staged validation gates and keep fallback paths intact.

---

## Read First (in order)

1. `CLAUDE.md` — project overview, architecture, commands
2. `AGENTS.md` — mandatory sub-agent policy, Bell Labs notebook protocol
3. This file — the kernel execution specification
4. `notes/LAB_NOTEBOOK.md` — experiment history (append-only)
5. `notes/experiments.ndjson` — machine-readable experiment log (append-only)

---

## Hard Constraints

- Follow `/Volumes/VIXinSSD/wayfinder/AGENTS.md` exactly.
- Spawn at least one sub-agent before non-trivial edits.
- Add pre-hypothesis and post-result entries to both notebook logs for every benchmark/compute run.
- Retro/backfill must remain default-off for inference.
- Compare every benchmark against a named baseline with absolute + delta + % delta.
- Do NOT revert unrelated dirty files in this repo.
- Must use `python3` not `python` on this system.
- `ruff` line-length = 100 chars.

---

## Goal

Start real compute/inference/memory execution from the completed setup scaffold, with immediate priority on **K4 (active-row fused; fixes known chunked-prefill dense fallback)**, then **K1**.

---

## Context Already Completed

The setup scaffold exists and is ready:

| File | Purpose | Status |
|------|---------|--------|
| `discover_sessions/manifest.json` | Consolidated setup metadata | `ready=true` |
| `docs/discover_setup.md` | Documentation | Complete |
| `hcsa/discover/targets.py` | K1-K5 target registry | Complete |
| `hcsa/discover/readiness.py` | Environment validation | Complete |
| `hcsa/discover/session.py` | Workspace scaffolding | Complete |
| `scripts/wayc.py` | CLI (`discover-targets`/`discover-setup`/`discover-status`) | Complete |
| `hcsa/mlx/kernels/metal/seeds/` | Placeholder Metal seed kernels | Complete |
| `runs/mlx/discover_setup_20260208_setup.json` | Setup run artifact | Complete |

Bell Labs setup entries already logged in:
- `notes/LAB_NOTEBOOK.md`
- `notes/experiments.ndjson`

### Existing Discovered K4 Artifact (Partial)

A preliminary K4 artifact already exists but is **incomplete/mock**:
- `hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.metal` — mock variant, hardcoded D=128, W=65
- `hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.py` — Python wrapper importing from `zmlx.metal`

This artifact has `Speedup: 1.34x` from session `958dc0e78c1d4161` but uses hardcoded dimensions and is not production-ready. It needs to be replaced with a proper implementation.

---

## Critical Bug: Chunked Prefill Dense Fallback

### Root Cause

In `hcsa/integrations/glm_mlx.py` line 299-314, the GLM integration has a guard:

```python
active_mode = self.path == "permute" and cache is not None and q_len < k_len
```

When `active_mode` is true (chunked prefill: Q_len < K_len), the code checks for:
1. A discovered active-row kernel (`use_discovered_active_row_kernel`)
2. Whether K_len is below `active_dense_threshold` (currently 49152)

If neither condition enables the HCSA path, it falls back to `_dense_fallback()` — full O(T^2) dense MLA attention. This means **ALL chunked prefill chunks (except the first where Q_len == K_len) run dense attention**, negating HCSA's O(T*W) complexity.

This is the root cause of the 6.9x latency regression observed at T=65536 with chunk=4096.

The **memory reduction IS real** (26% at T=65536) because the KV cache structure is still sparse. The **latency regression is a fixable bug**, not a fundamental limitation.

### The Fix

The `wayfinder_permute_window_attention_active_batched` function (`hcsa/mlx/attention.py:866`) already implements Q_len < K_len support in Python. The issue is:
1. It's not wired as the default path in the GLM integration
2. Performance without a fused kernel is worse than dense (Python dispatch overhead)
3. The discovered kernel artifact is a mock, not a real implementation

K4's goal: make the active-row path **faster than dense fallback** via a fused Metal kernel, then wire it as the default.

---

## Execution Plan

### Step 1: Confirm Readiness and Target Metadata

```bash
cd /Volumes/VIXinSSD/wayfinder
python3 scripts/wayc.py discover-status --manifest discover_sessions/manifest.json
python3 scripts/wayc.py discover-targets --targets K4 K1
```

Verify `ready=true` in manifest. Verify K4 and K1 target specs are correct.

### Step 2: K4 — Active-Row Fused Kernel (Primary)

#### 2a. Session stub

Use: `discover_sessions/hcsa_active_row_session.stub.json`

The session stub contains:
- Target: K4 (`hcsa_active_row_fused`)
- Question: "Can we support Q_len < K_len without dense fallback in chunked prefill?"
- Quality gates: Q1 (correctness vs reference), Q2 (latency vs dense)

#### 2b. Bridge/Register K4 into ZMLX Discover Flow

If the ZMLX discover engine at `/Volumes/VIXinSSD/ZMLX` needs the target registered:
```bash
# Check if ZMLX target registry already has K4
python3 -c "from zmlx.discover.targets import list_targets; print(list_targets())"
```

If K4 is not registered, bridge it by creating a ZMLX-compatible target spec from the HCSA target definition.

#### 2c. Run Real Discovery Search

```bash
python -m zmlx.discover search hcsa_active_row --llm claude-code --steps 10
```

The discovery search should:
1. Start from the seed kernel in `hcsa/mlx/kernels/metal/seeds/hcsa_active_row_fused.metal`
2. Generate candidate Metal kernels
3. Evaluate each candidate against the Python reference path
4. Export the best candidate to: `hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.metal`

#### 2d. Wire Dispatch

In `hcsa/mlx/kernels/metal/__init__.py`, the seam already exists:

```python
_ACTIVE_ROW_DISCOVERED = _KERNEL_DIR / "hcsa_active_row_fused_discovered.metal"

def has_discovered_active_row_kernel() -> bool:
    return _ACTIVE_ROW_DISCOVERED.exists()
```

Wire the discovered kernel into the attention dispatch so `Q_len < K_len` uses the fused candidate when available, with safe fallback to `wayfinder_permute_window_attention_active_batched`:

```python
# In glm_mlx.py or attention.py:
if has_discovered_active_row_kernel():
    # Use fused Metal kernel
    y = active_row_fused_kernel(q_active, k_cache, v_cache, ...)
else:
    # Fallback to Python active-row path
    y, _ = wayfinder_permute_window_attention_active_batched(...)
```

Preserve the dense fallback as a last resort (controlled by `active_dense_threshold`).

### Step 3: Validate K4

#### 3a. Correctness vs Python Reference

Compare fused kernel output against `wayfinder_permute_window_attention_active_batched` output:
- Max absolute error < 1e-3 (half precision)
- Mean absolute error < 1e-4
- Test at multiple (Q_len, K_len) combinations: (1, 1024), (32, 4096), (256, 16384), (4096, 65536)

#### 3b. Benchmark: Chunked Prefill Bug Regime

Run on the **original non-Qwen path first** (AGENTS.md: "reproduce and optimize on the original non-Qwen path first"):

```bash
PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 4096 16384 65536 --batch 1 --heads 8 --embd 256 \
  --window 32 --warmup 2 --iters 4 \
  --out-dir benchmarks/mlx/tiny_wayfinder/k4_baseline

# Then with K4 active-row kernel:
PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 4096 16384 65536 --batch 1 --heads 8 --embd 256 \
  --window 32 --warmup 2 --iters 4 \
  --out-dir benchmarks/mlx/tiny_wayfinder/k4_fused
```

#### 3c. Report

For each sequence length, report:
- **Absolute**: latency_ms, peak_memory_MB, throughput_tok/s
- **Delta vs baseline**: latency_ms_diff, memory_MB_diff
- **Delta %**: `100 * (1 - fused/baseline)` for memory, `baseline/fused` for speedup

Expected outcome: K4 fused should be **at least 2x faster** than dense fallback at T=65536 (sparse O(T*W) vs dense O(T^2)), with equivalent or better memory.

### Step 4: K1 — Permute-Window Fused Kernel

#### 4a. Session stub

Use: `discover_sessions/hcsa_permute_window_session.stub.json`

Target: K1 (`hcsa_permute_window_fused`)
Question: "Can we fuse permute+window+attention+unpermute into one Metal kernel?"
Priority: P0 (highest payoff)

#### 4b. Discovery + Export + Integration

Same flow as K4:
1. Run discovery search from seed kernel
2. Export to `hcsa/mlx/kernels/metal/hcsa_permute_window_fused_discovered.metal`
3. Wire dispatch via `has_discovered_permute_window_kernel()` in `__init__.py`

#### 4c. Correctness

Compare against `permute_cycle_window_attention_single` and `wayfinder_permute_window_attention_batched`:
- Same error tolerances as K4
- Test at T=1024, 4096, 16384, 65536

#### 4d. Benchmark

```bash
PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 1024 4096 16384 65536 --batch 1 --heads 8 --embd 256 \
  --window 32 --warmup 2 --iters 4 \
  --out-dir benchmarks/mlx/tiny_wayfinder/k1_baseline

PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 1024 4096 16384 65536 --batch 1 --heads 8 --embd 256 \
  --window 32 --warmup 2 --iters 4 \
  --out-dir benchmarks/mlx/tiny_wayfinder/k1_fused
```

### Step 5: Logging Discipline

Before each run:
```json
// notes/experiments.ndjson
{
  "id": "YYYYMMDD_HHMMSS_k4_fused_correctness",
  "question": "Does the K4 fused kernel match the Python reference?",
  "hypothesis": "Fused kernel output should match Python within half-precision tolerance",
  "change_set": ["hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.metal"],
  "command": "python3 -m pytest tests/mlx/test_k4_correctness.py -v",
  "controls": {"retro_backfill": "off", "window": 32, "backend": "mlx"},
  "metrics": {},
  "decision": "pending",
  "next_action": "pending"
}
```

After each run: fill in `metrics`, `decision`, `next_action`.

If conflicting outcomes occur, run tie-break experiment and record both hypotheses/results.

### Step 6: Final Summary

After completing K4 and K1, produce a summary with:
- Files changed (list)
- Commands run (list)
- Metrics vs baseline (table: absolute, delta, delta%)
- Decision: keep / revert / follow-up
- Exact next action

---

## Architecture: Discover System

### Target Registry

Five kernel targets defined in `hcsa/discover/targets.py`:

| ID | Name | Priority | Purpose |
|----|------|----------|---------|
| K1 | `hcsa_permute_window_fused` | P0 | Fuse permute+window+attn+unpermute |
| K2 | `hcsa_sparse_gather_fused` | P2 | Fuse general sparse gather attention |
| K3 | `hcsa_graph_construct` | P1 | GPU-side cycle generation |
| K4 | `hcsa_active_row_fused` | P1 | Q_len < K_len without dense fallback |
| K5 | `hcsa_wayfinder_ttt_fused` | P2 | Fuse HCSA + TTT updates |

### Runtime Integration Seam

`hcsa/mlx/kernels/metal/__init__.py` defines:
- `has_discovered_active_row_kernel() -> bool` — checks for K4 artifact at `hcsa_active_row_fused_discovered.metal`
- `has_discovered_permute_window_kernel() -> bool` — checks for K1 artifact at `hcsa_permute_window_fused_discovered.metal`
- `sparse_row_attention_fused(q, k, v, neigh_idx, *, edge_type)` — future hook, currently `NotImplementedError`

### Discovery Workflow

```
1. wayc.py discover-setup (already done)
   ↓ writes session stubs + seed kernels
2. ZMLX Discover search (this session's work)
   ↓ LLM-guided kernel generation + evaluation
3. Export best candidate to hcsa/mlx/kernels/metal/
   ↓ discovered .metal + .py artifacts
4. Wire dispatch in attention.py / glm_mlx.py
   ↓ has_discovered_*_kernel() gates
5. Validate correctness + benchmark
   ↓ hypothesis/result entries
6. Ship or iterate
```

### GLM Integration Dense Fallback Logic

`hcsa/integrations/glm_mlx.py` lines 289-332:

```python
def __call__(self, x, mask=None, cache=None):
    queries, keys, values = extract_qkv_from_glm_attention(self, x, cache=cache)
    q_len = int(queries.shape[2])
    k_len = int(keys.shape[2])
    active_mode = self.path == "permute" and cache is not None and q_len < k_len

    # K4 discovered kernel check
    discovered_active_available = (
        active_mode
        and self.use_discovered_active_row_kernel
        and has_discovered_active_row_kernel()
    )

    # Dense fallback if K_len below threshold and no discovered kernel
    force_dense_active = (
        active_mode
        and self.active_dense_threshold is not None
        and k_len <= self.active_dense_threshold
        and (not discovered_active_available)
    )

    use_active_permute = active_mode and not force_dense_active

    if force_dense_active or (q_len != k_len and not use_active_permute):
        return self._dense_fallback(queries, keys, values, mask, cache)
    # ... HCSA path continues
```

The goal of K4 is to make `discovered_active_available = True` with a real (not mock) kernel, so the dense fallback is never triggered for K_len > `active_dense_threshold`.

---

## Key File Reference

```
# Discover system
hcsa/discover/targets.py                    # K1-K5 target definitions
hcsa/discover/readiness.py                  # Environment validation
hcsa/discover/session.py                    # Session scaffolding
scripts/wayc.py                             # CLI interface
discover_sessions/manifest.json             # Setup output
discover_sessions/*.stub.json               # Per-target session templates

# Metal kernel seam
hcsa/mlx/kernels/metal/__init__.py          # Runtime integration (has_discovered_*())
hcsa/mlx/kernels/metal/seeds/              # Placeholder seed kernels
hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.metal   # K4 (mock, replace)
hcsa/mlx/kernels/metal/hcsa_active_row_fused_discovered.py      # K4 Python wrapper (mock)

# Attention kernels (Python reference paths)
hcsa/mlx/attention.py                       # All three attention functions
  :260  permute_cycle_window_attention_single      # K1 reference
  :448  wayfinder_permute_window_attention_batched # K1 batched reference
  :866  wayfinder_permute_window_attention_active_batched  # K4 reference

# Integration (where dense fallback lives)
hcsa/integrations/glm_mlx.py               # GLM-4.7 integration, dense fallback at :289-332

# Benchmarks
scripts/bench_mlx_wayfinder_scale.py        # Tiny model benchmark
scripts/bench_glm_consumer_mlx.py           # GLM consumer benchmark
scripts/bench_glm_chunked_prefill_mlx.py    # Chunked prefill benchmark (has --no-swap, --kv-step)
```

---

## After K4 + K1 Are Validated

Return to the GLM-4.7-Flash consumer benchmark campaign (see `README.md` section "2026-02-08 GLM-4.7 Consumer Benchmark Status"):

Victory gates:
- seq=65536, decode=256: HCSA E2E median >= 10% better than dense, TTFT >= 10% better, ITL p95 not worse by > 5%
- Peak memory reduction >= 8%
- Quality parity (not worse by > 2 percentage points)
- Reproducibility (>= 3 repeats + 2-run confirmation within +/- 5%)

Promoted config: `path=permute, active_dense_threshold=49152, query_chunk_size=384, head_chunk_size=2, kv_step=4096`

Consumer benchmark: `scripts/bench_glm_consumer_mlx.py`

Consider whether the "Actually Hamiltonian" fixes from `HAMILTONIAN_IMPLEMENTATION_PROMPT.md` (circular windowing, union multigraph) should be tested as part of the GLM benchmark campaign after K4/K1 are landed.

---

## Use Aleph Tools

Use Aleph tools where helpful for:
- Large log/result analysis
- Benchmark result comparison
- Metal kernel source inspection
- Session transcript review
