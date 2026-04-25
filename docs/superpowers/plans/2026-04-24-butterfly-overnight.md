# Butterfly Overnight: Real Long-Context Wins, Strict Claims

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Butterfly compressed-attention path either a measured win on Apple-silicon MLX long-context inference, or honestly downgrade the claim — backed by per-call profiling, ladder benchmarks, and a written blocker memo if the evidence does not arrive.

**Architecture:** Three implementation routes attack three different layers (kernel, decode-only cache, prefill streaming) plus a quality eval. Each route is gated by stop rules. The plan is iterative: every route ends in a benchmark and a decision (proceed / pivot / stop) before moving to the next.

**Tech Stack:** MLX 0.20+ (`mlx.core`, `mx.fast.scaled_dot_product_attention`), `mlx_lm` Qwen 3.5 0.8B 4-bit, Python 3.14 with the `.venv-macos-metal` venv, existing modules `bna/mlx/attention.py`, `bna/mlx/compressed_cache.py`, `bna/integrations/qwen_mlx.py`, harness `scripts/bench_qwen_consumer_mlx.py`.

---

## Current Truth Boundary

This is what the artifacts prove **today (2026-04-24, HEAD `f2bdbed`)**, not what they suggest.

**Branch state (verified, not assumed):**
- `git status --short --branch` → `## main...origin/main [ahead 3]` (the prior assumption of "ahead by 2" is stale by one commit; the third commit is `f2bdbed` "Add query chunking to compressed Butterfly attention").
- Recent log: `f2bdbed`, `3576dc6`, `0748b48`, `568c484` (origin/main), `f88cb0a`.

**Verification commands run for this plan:**
- `.venv-macos-metal/bin/python -m py_compile scripts/bench_qwen_consumer_mlx.py bna/mlx/attention.py bna/integrations/qwen_mlx.py` → exit 0.
- `.venv-macos-metal/bin/python -m pytest tests/mlx/test_compressed_butterfly_attention.py tests/mlx/test_compressed_butterfly_from_cache.py tests/mlx/test_compressed_kv_cache.py tests/pytorch/test_compressed_butterfly_attention.py -q` → **8 passed, 0 failed**.

**Model facts (read from `config.json` of `mlx-community/Qwen3.5-0.8B-4bit`):**
- `num_hidden_layers: 24`, `full_attention_interval: 4`.
- `layer_types` lists exactly 6 `full_attention` slots at indices `[3, 7, 11, 15, 19, 23]`. The other 18 layers are `linear_attention` (SSM/mamba-style, not classical KV).
- `EXPECTED_QWEN35_FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)` in `bna/integrations/qwen_mlx.py:48` is the constant for a deeper Qwen 3.5 variant (32 layers). On 0.8B the discovery rule (`bna/integrations/qwen_mlx.py:1806-1816`) correctly returns 6, not 8. The "missing 2" layers are not a bug — they do not exist on this checkpoint. Use `validate_qwen35_full_attention_layers(model, allow_mismatch=True)` for 0.8B or fix the constant.

**What the latest 32k benchmark proves:**
- Stock 32k (`stock_instrumented_32768/results.json`): `e2e=9.34s`, `peak=1.77 GB`, `cache_storage_after_prefill≈414 MB`.
- Best compressed_butterfly 32k (`compressed_butterfly_w64_qchunk64_instrumented_32768/results.json`, `--compressed-local-window-tokens 64 --query-chunk-size 64`): `e2e=14.64s`, `peak=6.27 GB`, `cache_storage_after_prefill≈16.4 MB`.
- **Retained KV win**: ~25× smaller resident cache after prefill. This is a real, artifact-backed advantage.
- **MLX peak/speed loss**: 3.53× more peak memory, 1.57× slower e2e at 32k. The compressed path is currently a net loss on speed and peak.

**Decode is dense fallback.** Confirmed by reading source:
- `scripts/bench_qwen_consumer_mlx.py:1578-1579` maps the CLI default `--butterfly-decode-backend stock` to internal value `dense`.
- `bna/integrations/qwen_mlx.py:1293-1306` triggers `force_dense_butterfly_decode` when `path ∈ {permute, block_sparse}` and backend=="dense" and we are decoding.
- The dense fallback at `bna/integrations/qwen_mlx.py:1234-1254` calls `scaled_dot_product_attention(queries, keys, values, …)` on the keys/values returned by `cache.update_and_fetch` — and `CompressedKVCache.update_and_fetch` (`bna/mlx/compressed_cache.py:34-59`) returns only the **trimmed local-window tail**.
- **Consequence:** under default flags, decode under compressed_butterfly attends to the local-window tail plus the new token only. Block summaries are not consulted at decode. This is information-lossy. Speed/peak numbers are still measurable; quality numbers under this configuration are **not** valid evidence of anything except a windowed-attention proxy.

**Therefore, today we may claim:**
- Compressed Butterfly prefill on swapped full-attention layers is structurally Butterfly (block-sparse routed neighbors, mean per-block summaries, causal-shift partner rule).
- Retained KV storage after prefill is materially smaller (~25× at 32k on 6 layers).
- Tests pass; the math agrees with a NumPy reference at small sizes.

**We may not yet claim:**
- That Butterfly is faster than stock at any tested context length on MLX.
- That Butterfly uses less peak MLX memory than stock at any tested context length.
- That decode quality is preserved under the current default flags.
- That the path scales to 1M / 2M without further work — we have not measured beyond 32k under the new code.

---

## Is Prefill Truly Butterfly?

**Yes, on the swapped full-attention layers, during prefill.** Specifically:

- **Layout.** `BlockSparseButterflyLayout` (`bna/mlx/attention.py:105`) holds `block_neighbors[Hq, N, K]`, `block_token_idx`, and a per-block causal mask. `build_block_butterfly_layout(... partner_rule="causal_shift", local_window_blocks, sink_count, partner_count)` (`bna/mlx/attention.py:195`) constructs it. This is the routed-neighbors data structure that makes the path Butterfly rather than dense.
- **Prefill kernel.** `compressed_butterfly_attention()` (`bna/mlx/attention.py:1108`) iterates over query blocks; per query block it gathers (a) raw tokens inside the local window via `mx.take(k/v, raw_idx, axis=2)` (lines 1169-1170) and (b) `block_size`-mean summaries of routed neighbor blocks via `mx.take(k_summary/v_summary, summary_idx, axis=2)` (lines 1175-1176). Outputs are concatenated (line 1180-1181), masked (line 1185), and run through `mx.fast.scaled_dot_product_attention` (line 1192) with a manual fallback at line 1200.
- **Phase usage.**
  - **Prefill**: True Butterfly, on swapped layers only.
  - **Cache compaction**: True per-block mean compression (`CompressedKVCache._append_summary` at `bna/mlx/compressed_cache.py:23-32`) — happens both during chunked prefill and during decode steps.
  - **Decode**: **Dense fallback** by default (above). The cache-aware path `compressed_butterfly_attention_from_cache` (`bna/mlx/attention.py:1414`) exists and is exercised by tests, but the bench harness sets `butterfly_decode_backend == "dense"` for `block_sparse`/`permute` paths.
- **Stock fallback.** Linear-attention layers (18 of 24 on 0.8B) are not swapped at all. Their attention is whatever `mlx_lm` ships. They use the standard MLX cache, not `CompressedKVCache`.

**Bottom line:** prefill claim is honest. Decode claim is currently *windowed*, not Butterfly.

---

## Scaling Target — What Has To Be True For 1M / 2M

Three independent things scale with sequence length T. Any one of them turning quadratic kills 1M/2M.

**(1) Retained KV cache after prefill** — what `cache_storage_after_prefill.total_bytes` measures.
- Stock (today): ~414 MB at 32k on 6 full-attention layers → ~26 MB per layer per 32k tokens. Linear in T. 1M extrapolates to ~13 GB; 2M to ~26 GB. **Already a memory-wall problem at 1M.**
- Compressed Butterfly (today): ~16.4 MB at 32k → ~2.7 MB per layer per 32k. Linear in T but with much smaller constant, dominated by `block_size`-mean summaries (≈T/block_size summary tokens × 2 KV × dh × bytes). 1M extrapolates to ~512 MB; 2M to ~1 GB. **Acceptable.**
- Required for 1M: stay on compressed.

**(2) Execution intermediates / per-call peak** — what `peak_memory_bytes` measures.
- Stock: SDPA buffers grow linearly during chunked prefill but fused at the kernel level. Today: 1.77 GB at 32k. Past 64k, MLX SDPA over a `[1, H, T, T]` attention matrix is the wall — at 1M with H=8, T=1M, the score tile alone is ~32 GB if not chunked. Stock prefill **does not scale to 1M without further chunking work** that is outside this plan.
- Compressed Butterfly (today): 6.27 GB at 32k. The retained KV is only 16.4 MB; the rest is *transient*: gathered raw+summary K/V tiles, concat tensors, output buffer, layout numpy arrays. If transient cost is constant per query block (block_size queries × (raw_slots+summary_slots) keys), peak should grow only as O(B·H·block_size·(raw_slots+summary_slots)·dh) — i.e. flat in T at fixed window/partner_count. **If we cannot make peak flat in T, compressed Butterfly does not scale to 1M either.** Phase 1 instrumentation must answer this directly.
- Required for 1M: gather peak must be measured constant in T at fixed window/partner_count. If Phase 1 shows it growing with T, that is the actual blocker — not stock vs compressed at 32k.

**(3) Layout/graph construction cost.**
- `_compressed_butterfly_block_indices` / `_compressed_butterfly_active_indices(_from_tail)` (`bna/mlx/attention.py:955-1077, 1346-1411`) build numpy index arrays per call, not cached. The inner numpy loops over `num_blocks` are O(num_blocks²) at worst. At 1M / block_size=128, num_blocks≈8000; an O(N²) loop runs 64M iterations *per call per layer* — minutes of pure Python work. **Layout caching is mandatory for 1M.**
- `BlockSparseButterflyLayout` itself (`bna/mlx/attention.py:195-361`) is also rebuilt per `kv_len`. With chunked prefill emitting many distinct `kv_len` values, this currently rebuilds every call. The static-mode `ButterflyAttentionMLX` already has a graph cache (`_GRAPH_CACHE_STORE`, lines 40-42, 3042-3071); the compressed path needs the equivalent.
- Required for 1M: layout build cost amortized to once per layer per Pythonic-key (not per call), and per-T-bucketed if T varies.

**Implication for the overnight plan:** Phase 1 is the gate. If `peak_gather_bytes` rises with T (from a 32k probe alone we cannot tell), Route C (block-streamed prefill) is mandatory. If layout/index recomputation dominates `attn_ms`, none of Routes A/B/C will close the gap without also adding a layout cache — call this Route C-prime and add it to the streamed implementation.

---

## Verification Commands And Results

```bash
# 1. Compile sanity
.venv-macos-metal/bin/python -m py_compile \
  scripts/bench_qwen_consumer_mlx.py \
  bna/mlx/attention.py \
  bna/integrations/qwen_mlx.py
# → exit 0

# 2. Compressed-Butterfly numerical & cache tests
.venv-macos-metal/bin/python -m pytest \
  tests/mlx/test_compressed_butterfly_attention.py \
  tests/mlx/test_compressed_butterfly_from_cache.py \
  tests/mlx/test_compressed_kv_cache.py \
  tests/pytorch/test_compressed_butterfly_attention.py \
  -q
# → 8 passed, 0 failed
```

Captured at the top of `Current Truth Boundary`. These two commands gate every other task in this plan and must keep passing after each implementation step.

---

## Overnight Plan

The night is structured into seven phases. Each phase is a self-contained set of tasks with explicit commit points and a decision gate. Use `BNA_COMPRESS_PROFILE=1` and `BNA_COMPRESS_FORCE_MANUAL=1` from `bna/mlx/attention.py:46-47` as diagnostic switches.

### Phase 0 — Confirm baselines and lock the floor (≈30 min)

**Files:**
- Read: `results/benchmarks/qwen35_0p8b_mlx/stock_instrumented_32768/results.json`
- Read: `results/benchmarks/qwen35_0p8b_mlx/compressed_butterfly_w64_qchunk64_instrumented_32768/results.json`
- Read: `bna/integrations/qwen_mlx.py:1806-1816` (`get_qwen_full_attention_layer_indices`)

- [ ] **Step 1: Re-run pytest gate**
  ```bash
  .venv-macos-metal/bin/python -m pytest \
    tests/mlx/test_compressed_butterfly_attention.py \
    tests/mlx/test_compressed_butterfly_from_cache.py \
    tests/mlx/test_compressed_kv_cache.py \
    tests/pytorch/test_compressed_butterfly_attention.py -q
  ```
  Expected: `8 passed`.

- [ ] **Step 2: Reconfirm 32k stock baseline (one process)**
  ```bash
  .venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
    --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
    --hf-home /Volumes/VIXinSSD/hf_cache --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub --hf-offline \
    --mode stock --seq-lens 32768 --decode-len 8 --repeats 1 \
    --chunk-size 384 --skip-multi-turn --skip-quality \
    --out-dir results/benchmarks/qwen35_0p8b_mlx/p0_stock_32768
  ```
  Capture `prefill_sec`, `e2e_sec`, `peak_memory_bytes`, `cache_storage_after_prefill.total_bytes`. These are the floor.

- [ ] **Step 3: Reconfirm 32k compressed_butterfly best-known**
  ```bash
  .venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
    --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
    --hf-home /Volumes/VIXinSSD/hf_cache --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub --hf-offline \
    --mode compressed_butterfly --block-partner-rule causal_shift \
    --compressed-local-window-tokens 64 --seq-lens 32768 --decode-len 8 --repeats 1 \
    --chunk-size 384 --kv-step 384 --query-chunk-size 64 --block-size 128 \
    --butterfly-decode-backend stock --skip-multi-turn --skip-quality \
    --out-dir results/benchmarks/qwen35_0p8b_mlx/p0_compressed_w64_qc64_32768
  ```

- [ ] **Step 4: Lock the floor in a small markdown table**
  Append to `results/benchmarks/qwen35_0p8b_mlx/OVERNIGHT_LOG.md` (create if absent):
  - timestamp
  - phase 0 reproducible numbers from steps 2-3
  Decision gate: if step 2 differs from `9.34s/1.77 GB` by >15% or step 3 differs from `14.64s/6.27 GB` by >15%, halt and write a "machine drift" note before proceeding.

- [ ] **Step 5: Commit Phase 0 baseline**
  ```bash
  git add docs/superpowers/plans/2026-04-24-butterfly-overnight.md \
          results/benchmarks/qwen35_0p8b_mlx/OVERNIGHT_LOG.md \
          results/benchmarks/qwen35_0p8b_mlx/p0_stock_32768 \
          results/benchmarks/qwen35_0p8b_mlx/p0_compressed_w64_qc64_32768
  git commit -m "bench: lock 32k stock and compressed_butterfly Phase 0 baselines"
  ```

### Phase 1 — Per-call instrumentation of compressed attention (≈90 min)

**Goal:** find which sub-step of the compressed kernel contributes most to the 6.27 GB peak.

**Files:**
- Modify: `bna/mlx/attention.py` (extend `_compress_stats` collection in `compressed_butterfly_attention_from_cache`, `compressed_butterfly_attention_active`, and the prefill `compressed_butterfly_attention`)
- Modify: `scripts/bench_qwen_consumer_mlx.py` (dump compressed_profile to JSON, already partly wired per `COMPRESSED_BUTTERFLY_FIX_REPORT.md` §9)
- Test: `tests/mlx/test_compress_profile_instrumentation.py` (new)

- [ ] **Step 1: Write a failing test for new instrumentation**
  ```python
  # tests/mlx/test_compress_profile_instrumentation.py
  import os, math, numpy as np, pytest
  mx = pytest.importorskip("mlx.core")
  os.environ["BNA_COMPRESS_PROFILE"] = "1"
  from bna.mlx import attention as A
  from bna.mlx.attention import (
      build_block_butterfly_layout,
      compressed_butterfly_attention_active,
      _compress_profile_reset, _compress_profile_dump,
  )

  def test_profile_records_take_concat_attn_time():
      _compress_profile_reset()
      B, H, Tk, Tq, dh = 1, 2, 32, 4, 8
      layout = build_block_butterfly_layout(
          seq_len=Tk, block_size=4, num_key_value_heads=H,
          num_key_value_groups=1, layer_idx=2,
          local_window_blocks=1, sink_count=1, partner_count=1,
          partner_rule="causal_shift",
      )
      q = mx.array(np.random.randn(B,H,Tq,dh).astype(np.float32))
      k = mx.array(np.random.randn(B,H,Tk,dh).astype(np.float32))
      v = mx.array(np.random.randn(B,H,Tk,dh).astype(np.float32))
      compressed_butterfly_attention_active(
          q, k, v, layout=layout,
          query_positions=mx.array([21,25,29,31], dtype=mx.int32),
          local_window_tokens=4,
      )
      stats = _compress_profile_dump()
      assert "take_ms" in stats and stats["take_ms"] > 0
      assert "concat_ms" in stats and stats["concat_ms"] >= 0
      assert "attn_ms" in stats and stats["attn_ms"] > 0
      assert "peak_gather_bytes" in stats
  ```
  Run: `pytest tests/mlx/test_compress_profile_instrumentation.py -v` → expected FAIL with `KeyError: 'take_ms'` (or similar).

- [ ] **Step 2: Add instrumentation to `_compress_stats`**
  In `bna/mlx/attention.py`, extend the `_compress_stats` dict with fields `take_ms`, `concat_ms`, `attn_ms`, `peak_gather_bytes`. Wrap each of the calls at lines 1297-1303 (active) and 1473-1478 (from_cache) with `time.perf_counter()` deltas, accumulate into stats. Compute `peak_gather_bytes = max(prev, raw_k.nbytes + summary_k.nbytes + raw_v.nbytes + summary_v.nbytes)` after each gather. Use `mx.eval()` before timing reads to ensure the work has actually happened.

- [ ] **Step 3: Run the test, expect PASS**
  ```bash
  .venv-macos-metal/bin/python -m pytest tests/mlx/test_compress_profile_instrumentation.py -v
  ```

- [ ] **Step 4: Run an instrumented 32k profile**
  ```bash
  BNA_COMPRESS_PROFILE=1 .venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
    --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
    --hf-home /Volumes/VIXinSSD/hf_cache --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub --hf-offline \
    --mode compressed_butterfly --block-partner-rule causal_shift \
    --compressed-local-window-tokens 64 --seq-lens 32768 --decode-len 8 --repeats 1 \
    --chunk-size 384 --kv-step 384 --query-chunk-size 64 --block-size 128 \
    --butterfly-decode-backend stock --skip-multi-turn --skip-quality \
    --out-dir results/benchmarks/qwen35_0p8b_mlx/p1_profile_32768
  ```
  Expected artifact: `compress_profile.json` containing `take_ms`, `concat_ms`, `attn_ms`, `peak_gather_bytes`, `calls`.

- [ ] **Step 5: Decide which route to attack first**
  Read `compress_profile.json`. Apply this rule and write the decision into `OVERNIGHT_LOG.md`:
  - If `peak_gather_bytes > 1.5 GB` → **Route C** (block-streamed prefill) is most likely to move peak.
  - If `attn_ms / total_e2e > 0.45` → **Route A** (fused kernel) is most likely to move speed.
  - If `take_ms / attn_ms > 0.4` → mention gather-bandwidth bottleneck explicitly; Route C still wins.
  - Otherwise pick the route with the largest single component.

- [ ] **Step 6: Commit Phase 1**
  ```bash
  git add bna/mlx/attention.py tests/mlx/test_compress_profile_instrumentation.py \
          results/benchmarks/qwen35_0p8b_mlx/p1_profile_32768
  git commit -m "instrument compressed butterfly take/concat/attn timing and gather bytes"
  ```

### Phase 2 — Implement the chosen route prototype (≈3 h)

Pick one of A/B/C from the ranking below. Stop rules apply.

#### Route A: Fused MLX compressed-attention kernel (one query chunk, no concat)

**Files:**
- Modify: `bna/mlx/attention.py` — add `compressed_butterfly_attention_fused_active`
- Test: `tests/mlx/test_compressed_butterfly_fused.py`

The idea: compute attention separately over (a) the raw local window and (b) the routed summary blocks, with online softmax recombination. Avoid the gather→concat→SDPA path; never materialize the unioned K/V tensor.

- [ ] **Step 1: Failing parity test** — write `tests/mlx/test_compressed_butterfly_fused.py`:
  ```python
  from __future__ import annotations
  import numpy as np, pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import (
      build_block_butterfly_layout,
      compressed_butterfly_attention_active,
      compressed_butterfly_attention_fused_active,
  )

  def test_fused_matches_active_reference():
      rng = np.random.default_rng(19)
      B, H, Tk, Tq, dh = 1, 2, 18, 3, 8
      query_positions = np.asarray([9, 13, 17], dtype=np.int32)
      layout = build_block_butterfly_layout(
          seq_len=Tk, block_size=4, num_key_value_heads=H,
          num_key_value_groups=1, layer_idx=3,
          local_window_blocks=1, sink_count=1, partner_count=1,
          partner_rule="causal_shift",
      )
      q_np = rng.standard_normal((B, H, Tq, dh), dtype=np.float32)
      k_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
      v_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
      y_ref, _ = compressed_butterfly_attention_active(
          mx.array(q_np), mx.array(k_np), mx.array(v_np),
          layout=layout, query_positions=mx.array(query_positions, dtype=mx.int32),
          local_window_tokens=4,
      )
      y_fused, _ = compressed_butterfly_attention_fused_active(
          mx.array(q_np), mx.array(k_np), mx.array(v_np),
          layout=layout, query_positions=mx.array(query_positions, dtype=mx.int32),
          local_window_tokens=4,
      )
      mx.eval(y_ref, y_fused)
      assert np.allclose(
          np.asarray(y_fused, dtype=np.float32),
          np.asarray(y_ref, dtype=np.float32),
          atol=3e-4, rtol=3e-4,
      )
  ```
  Run: `pytest tests/mlx/test_compressed_butterfly_fused.py -v` → expected FAIL with `ImportError: cannot import name 'compressed_butterfly_attention_fused_active'`.
- [ ] **Step 2: Implement the fused path with online softmax**
  Two SDPA calls per query block — one over `raw_k/raw_v[block_size]`, one over `summary_k/summary_v[K]` — merged by max-stabilized softmax (`m_total = max(m_raw, m_sum); l_total = exp(m_raw-m_total)*l_raw + exp(m_sum-m_total)*l_sum; out = (exp(m_raw-m_total)*o_raw*l_raw + exp(m_sum-m_total)*o_sum*l_sum) / l_total`). Return weights when requested.
- [ ] **Step 3: Make the test pass**
- [ ] **Step 4: A/B benchmark at 32k**
  Run with the fused path enabled (CLI flag `--compressed-attn-impl fused` to be added to `scripts/bench_qwen_consumer_mlx.py`). Compare to Phase 1 baseline.
- [ ] **Step 5: Commit**

#### Route B: Stock prefill, decode-only Butterfly compressed cache

**Files:**
- Modify: `scripts/bench_qwen_consumer_mlx.py` — add `--mode butterfly_decode_only`
- Modify: `bna/integrations/qwen_mlx.py` — after prefill completes, re-summarize the stock cache into a `CompressedKVCache` per swapped layer; switch decode dispatch to `compressed_butterfly_attention_from_cache`.

The idea: take the speed and peak win of stock prefill (it is already 1.77 GB / 9.34 s at 32k), then compact the cache for decode-only long-context. Only honest if `--butterfly-decode-backend experimental` (which maps to `active_permute`) wires the compressed-from-cache path during decode rather than dense fallback.

- [ ] **Step 1: Failing test on swap-after-prefill correctness** — write `tests/mlx/test_decode_only_cache_compaction.py`:
  ```python
  from __future__ import annotations
  import numpy as np, pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import (
      build_block_butterfly_layout,
      compressed_butterfly_attention_from_cache,
  )
  from bna.mlx.compressed_cache import CompressedKVCache
  from bna.integrations.qwen_mlx import compact_cache_after_prefill

  def test_compact_cache_after_prefill_matches_streaming_cache():
      rng = np.random.default_rng(31)
      B, H, Tk, dh = 1, 2, 24, 8
      block_size, local_window = 4, 4
      k_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
      v_np = rng.standard_normal((B, H, Tk, dh), dtype=np.float32)
      streaming = CompressedKVCache(block_size=block_size, local_window_tokens=local_window)
      for s in range(0, Tk, 3):
          streaming.update_and_fetch(
              mx.array(k_np[:, :, s : s + 3, :]),
              mx.array(v_np[:, :, s : s + 3, :]),
          )
      tk_s, tv_s, ks_s, vs_s, tail_start_s, off_s = streaming.get_compressed_state()
      compacted = compact_cache_after_prefill(
          mx.array(k_np), mx.array(v_np),
          block_size=block_size, local_window_tokens=local_window,
      )
      tk_c, tv_c, ks_c, vs_c, tail_start_c, off_c = compacted.get_compressed_state()
      assert off_c == off_s and tail_start_c == tail_start_s
      assert np.allclose(np.asarray(tk_c, dtype=np.float32), np.asarray(tk_s, dtype=np.float32), atol=1e-5)
      assert np.allclose(np.asarray(ks_c, dtype=np.float32), np.asarray(ks_s, dtype=np.float32), atol=1e-5)
  ```
  Expected FAIL with `ImportError: cannot import name 'compact_cache_after_prefill'`.
- [ ] **Step 2: Implement post-prefill cache compaction**
  In `qwen_mlx.py`, add a function `compact_cache_after_prefill(prompt_cache, layer_indices, block_size, local_window_tokens)` that for each swapped layer:
  1. Reads `cache.keys`, `cache.values` from the stock cache.
  2. Constructs a new `CompressedKVCache` and calls `_append_summary` per `block_size`-block of the historical keys, then trims the raw keys to the local-window tail.
  3. Replaces `prompt_cache[idx]` with the compacted cache.
- [ ] **Step 3: Wire decode path to use compressed-from-cache**
  When `force_dense_butterfly_decode` would have triggered, instead route to `compressed_butterfly_attention_from_cache` if `cache` is `CompressedKVCache` and `butterfly_decode_backend != "dense"` — explicitly opt-in via `experimental` to keep `stock` semantics unchanged.
- [ ] **Step 4: Bench 32k → 64k → 128k decode-only**
- [ ] **Step 5: Commit**

#### Route C: Block-streamed Butterfly prefill (no per-query gather retention)

**Files:**
- Modify: `bna/mlx/attention.py` — add `compressed_butterfly_attention_streamed`
- Modify: `bna/integrations/qwen_mlx.py` — call streamed variant when `--compressed-attn-impl streamed`

Idea: emit per-block output to a pre-allocated tensor, never hold the full `[B, n_chunk, raw_slots+summary_slots, dh]` materialized buffer across blocks. `mx.eval()` between blocks to force scheduler flush; rely on out-of-place write into a pre-allocated `out` tensor with `mx.scatter`/concatenate-by-slice semantics. This directly attacks `peak_gather_bytes`.

- [ ] **Step 1: Failing parity test** — write `tests/mlx/test_compressed_butterfly_streamed.py`:
  ```python
  from __future__ import annotations
  import numpy as np, pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import (
      build_block_butterfly_layout,
      compressed_butterfly_attention,
      compressed_butterfly_attention_streamed,
  )

  def test_streamed_matches_prefill_reference():
      rng = np.random.default_rng(41)
      B, H, T, dh = 1, 2, 32, 8
      layout = build_block_butterfly_layout(
          seq_len=T, block_size=4, num_key_value_heads=H,
          num_key_value_groups=1, layer_idx=2,
          local_window_blocks=1, sink_count=1, partner_count=1,
          partner_rule="causal_shift",
      )
      q_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      k_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      v_np = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      y_ref, _ = compressed_butterfly_attention(
          mx.array(q_np), mx.array(k_np), mx.array(v_np),
          layout=layout, local_window_tokens=4,
      )
      y_streamed, _ = compressed_butterfly_attention_streamed(
          mx.array(q_np), mx.array(k_np), mx.array(v_np),
          layout=layout, local_window_tokens=4,
      )
      mx.eval(y_ref, y_streamed)
      assert np.allclose(
          np.asarray(y_streamed, dtype=np.float32),
          np.asarray(y_ref, dtype=np.float32),
          atol=3e-4, rtol=3e-4,
      )
  ```
  Expected FAIL with `ImportError: cannot import name 'compressed_butterfly_attention_streamed'`.
- [ ] **Step 2: Implement block-streamed path**
- [ ] **Step 3: Pass the test**
- [ ] **Step 4: 32k bench, compare peak**
- [ ] **Step 5: Commit**

### Phase 3 — 32k decision gate (≈30 min)

- [ ] **Step 1: Read both Phase-2 results JSON files.**
- [ ] **Step 2: Apply the stop rules below.** If neither A/B/C beat the floor at 32k by ≥15% on either e2e or peak, **stop coding**. Do not climb the ladder. Skip directly to Phase 6 (blocker memo) and Phase 7 (quality eval — Route D — which is independent).
- [ ] **Step 3: If a route wins, write the chosen route's name and the deltas into `OVERNIGHT_LOG.md`.**

### Phase 4 — Ladder up: 64k, 128k, 256k (≈2 h)

Only enter this phase if Phase 3 chose a winning route.

- [ ] **Step 1: 64k stock + winning compressed**
  Two one-process-per-config runs at `--seq-lens 65536 --decode-len 8 --repeats 1`. Output dirs: `p4_stock_65536`, `p4_compressed_65536`.
- [ ] **Step 2: 128k**, only if 64k peaks/times look healthy (compressed peak < 12 GB, e2e < 60 s).
- [ ] **Step 3: 256k**, only if 128k healthy (compressed peak < 24 GB, e2e < 240 s).
- [ ] **Step 4: For each rung, capture peak, e2e, retained KV, and the `compress_profile.json`.**
- [ ] **Step 5: Commit per rung.**

### Phase 5 — Quality probe (Route D, independent of A/B/C)

Even if speed/memory improvements stall, we still need to know whether the compressed cache is *useful*. Independent task.

**Files:**
- Create: `scripts/eval_butterfly_retrieval.py`
- Create: `tests/mlx/test_butterfly_retrieval_smoke.py`

- [ ] **Step 1: Write a smoke test for a needle-in-a-haystack-style retrieval probe**
  Synthesize a 16k-token sequence with one inserted unique sentinel; read out of the model under stock vs compressed cache; assert top-1 token logit at the answer position is recovered above a threshold under stock, and report the value under compressed.
- [ ] **Step 2: Run the probe at 8k, 16k, 32k.**
  This is descriptive; it produces JSON, not a pass/fail.
- [ ] **Step 3: Commit `results/benchmarks/qwen35_0p8b_mlx/p5_retrieval_probe.json`.**

### Phase 6 — Blocker memo (only if Phase 3 said stop)

- [ ] **Step 1: Write `results/benchmarks/qwen35_0p8b_mlx/BLOCKER_MEMO_2026-04-24.md`**
  Required sections: (a) what was tried, (b) the exact two route prototypes built and their before/after deltas, (c) what `compress_profile.json` shows is dominant, (d) what kernel/runtime change would actually move the number (e.g., a Metal-level fused kernel, or graph hoisting `BlockSparseButterflyLayout` outside the per-call path), (e) a recommendation: ship the retained-cache claim as the only honest claim, or do not ship.
- [ ] **Step 2: Commit memo.**

### Phase 7 — Final overnight summary

- [ ] **Step 1: Append to `OVERNIGHT_LOG.md` a short table** of all phases, decisions, and final claim status.
- [ ] **Step 2: Commit log.**
- [ ] **Step 3: Do not auto-push. Leave for human review.**

---

## Implementation Routes Ranked

Ranking is provisional until Phase 1 instrumentation lands. Re-rank after `compress_profile.json` exists.

1. **Route C — Block-streamed prefill (highest expected peak win).** The 6.27 GB peak with only 16.4 MB retained cache implies most of the memory is *transient* gather/intermediate, not stored cache. Streaming output per query block — write straight into `out[:,:,start:end,:]` via slice assignment, drop the `out_chunks` list — is the lowest-risk way to drop peak. If `peak_gather_bytes` from Phase 1 is the dominant component, pick this.
2. **Route A — Fused kernel via online softmax.** Doubles the SDPA dispatches but eliminates the union K/V tensor entirely. Strong on peak *and* potentially on speed because two narrow SDPA calls can be faster than one wide gathered SDPA on Metal. Higher implementation risk: numerical parity must be guarded by the existing reference test.
3. **Route B — Decode-only compaction after stock prefill.** Sidesteps the prefill gap entirely and is the only route that yields a memory-wall win at very long contexts (1M+) without first solving prefill. Compatibility-only label: this is *not* Butterfly prefill — it is stock prefill plus Butterfly decode cache. Must be reported under that exact label.
4. **Route D — Retrieval/quality probe.** Cheap, independent, and the only thing that proves the compressed cache preserves useful behavior. Run regardless of which compute route we pick.

---

## Benchmark Matrix

| Phase | Mode | Seq | Decode | Window | Q-chunk | Notes |
|------:|:-----|----:|------:|------:|--------:|:------|
| P0 | stock | 32768 | 8 | n/a | n/a | baseline floor |
| P0 | compressed_butterfly | 32768 | 8 | 64 | 64 | best-known compressed |
| P1 | compressed_butterfly + `BNA_COMPRESS_PROFILE=1` | 32768 | 8 | 64 | 64 | dumps `compress_profile.json` |
| P2 | (winning route prototype) | 32768 | 8 | 64 | 64 | A, B, or C |
| P4 | stock | 65536 | 8 | n/a | n/a | only after P3 says go |
| P4 | (winning route) | 65536 | 8 | 64 | 64 | only after P3 says go |
| P4 | (winning route) | 131072 | 8 | 64 | 64 | only if 65k healthy |
| P4 | (winning route) | 262144 | 8 | 64 | 64 | only if 128k healthy |
| P5 | retrieval probe | 8192 / 16384 / 32768 | n/a | 64 | 64 | quality only |

Each row is a separate Python invocation (one process per config). `--repeats 1`. Quantization off unless specified.

---

## Stop Rules

These are bright lines. Do not negotiate with them at 3 AM.

1. **Speed/peak floor.** If after two implementation attempts (any combination of Routes A/B/C) the compressed_butterfly **peak at 32k remains > 3× stock peak**, stop coding and write the blocker memo. Do not advance to 64k.
2. **Semantic drift.** If a route changes prefill semantics away from Butterfly (e.g., Route B), label every artifact and every claim with `compatibility-only` or `decode-only`. Do not put it under the "Butterfly prefill superior" heading.
3. **OOM / stall.** If a benchmark stalls past `--stage-timeout-sec 900` or OOMs, capture exactly: the full CLI command, the last 50 lines of stdout, MLX peak memory at point of failure. Do **not** rerun with a longer context. Move to the next planned config or to the blocker memo.
4. **Stock fallback hidden as Butterfly.** Forbidden phrasing in any artifact this overnight: "Butterfly decode beats stock decode" or "compressed Butterfly faster overall". Use exactly the floor numbers we measured. If decode_backend is `stock`, write `stock decode (compressed cache)`, not `Butterfly decode`.
5. **Test regressions.** If at any point the four pytest files at the top of this plan stop passing, halt the implementation step, revert the offending change, and do not proceed.
6. **Layer-count drift.** If `validate_qwen35_full_attention_layers(model, allow_mismatch=True)` returns anything other than `[3,7,11,15,19,23]` for the 0.8B 4-bit checkpoint, halt and investigate before benchmarking — a different swap set invalidates the comparison.

---

## Expected Artifacts

By the morning, the following must exist on disk and be committed:

- `docs/superpowers/plans/2026-04-24-butterfly-overnight.md` (this file).
- `results/benchmarks/qwen35_0p8b_mlx/OVERNIGHT_LOG.md` (running log of decisions and numbers).
- `results/benchmarks/qwen35_0p8b_mlx/p0_stock_32768/results.json`
- `results/benchmarks/qwen35_0p8b_mlx/p0_compressed_w64_qc64_32768/results.json`
- `results/benchmarks/qwen35_0p8b_mlx/p1_profile_32768/results.json`
- `results/benchmarks/qwen35_0p8b_mlx/p1_profile_32768/compress_profile.json`
- `tests/mlx/test_compress_profile_instrumentation.py`
- `tests/mlx/test_butterfly_retrieval_smoke.py`
- For the chosen route in Phase 2: a parity test file (`tests/mlx/test_compressed_butterfly_<routename>.py`), a 32k bench dir.
- If Phase 3 said go: P4 dirs at the rungs we reached.
- Either `BLOCKER_MEMO_2026-04-24.md` (if we stopped) or a one-paragraph note in `OVERNIGHT_LOG.md` declaring which claim has been verified and which has not.

Every artifact must be reachable from a single grep of `OVERNIGHT_LOG.md`.

---

## What Not To Claim Yet

Forbidden, until the artifacts say otherwise:

- "Butterfly is faster than stock at 32k on MLX." — false at HEAD; 14.64 s vs 9.34 s.
- "Butterfly uses less peak memory than stock at 32k on MLX." — false at HEAD; 6.27 GB vs 1.77 GB.
- "Butterfly decode is superior." — currently dense fallback over a trimmed window. There is no Butterfly-decode-vs-stock-decode comparison in the artifacts.
- "Butterfly preserves long-context retrieval quality." — never measured. Phase 5 will produce a *first* descriptive number; that is not yet a claim of preservation.
- "8 full-attention layers swapped." — false on Qwen 3.5 0.8B 4-bit; 6 is correct (`layer_types` in the checkpoint config). 8 belongs to a deeper variant.
- "Scales to 1M / 2M context." — not measured under the new code at all. Even the retained-cache extrapolation (≈16.4 MB × 1M/32k ≈ 0.5 GB) ignores intermediate buffers and graph state.

What we *may* claim today, and only this:

- The compressed Butterfly prefill path on swapped full-attention layers is structurally Butterfly (block-routed neighbors + mean per-block summaries + causal-shift partner rule), and its math agrees with a NumPy reference at small sizes (8 tests pass).
- After 32k prefill, the retained KV cache footprint is ~25× smaller than stock under the same harness (16.4 MB vs 414 MB on the 6 swapped layers).
- Everything else is open and gated by Phases 1 through 7 of this plan.
