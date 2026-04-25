# Butterfly Cutover: V4 Systems Architecture + Deterministic Butterfly Routing

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task in this session, with checkpoints between phases. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut over the existing `compressed_butterfly` MLX attention path to a V4-systems-shaped implementation while preserving Butterfly's deterministic `causal_shift` routing as the unique sparse-selector mechanism. The code on disk continues to be called `compressed_butterfly_attention` / `CompressedKVCache` / `--mode compressed_butterfly`; only the internals change. No `_v2` names, no parallel modules.

**Unique contribution (this is what we are actually shipping):** A V4-architected sparse-KV attention path whose indexer is **deterministic Butterfly `causal_shift` routing over compressed blocks**, not a learned top-k. The training-required pieces of V4 (learned compressor, learned indexer) are replaced with parameter-free structural equivalents that work on a **frozen pretrained checkpoint** — Qwen 3.5 0.8B 4-bit MLX as the witness. V4 cannot retrofit; this can.

**Architecture (one-paragraph summary).** Per swapped full-attention layer, on long context: maintain a two-pool KV cache (state cache for the recent SWA tail and the in-progress compression block; classical cache for completed compressed blocks). At forward time, run two attention streams in parallel — (1) SWA over the `n_win` raw recent tokens, (2) sparse-compressed over compressed blocks selected by Butterfly's deterministic `causal_shift` routing topology — and merge their per-query outputs by online softmax `(m, l, o)` accumulation. Apply V4's counter-RoPE on the compressed-stream output. Never materialize the union of the two streams' K/V tensors as a Python-level concatenation. Compressor stays parameter-free (mean pool for now; positional-biased weighted sum as a no-train upgrade lane), with explicit acknowledgment that a learned compressor would require training and is out of scope for this cutover.

**Tech Stack:** MLX 0.20+ on Apple silicon. Python 3.14, `.venv-macos-metal`. Worktree at `/Volumes/VIXinSSD/butterfly/.worktrees/butterfly-overnight` on branch `butterfly/overnight-2026-04-24`. Reference paper: `docs/DeepSeek_V4.pdf` (read; brief in conversation history).

---

## Why a Cutover, Not an Iteration

Phase 0 of the prior overnight measured the existing `compressed_butterfly` path at 32k on Qwen 3.5 0.8B 4-bit MLX:

| | prefill | e2e | peak | retained KV |
|---|---:|---:|---:|---:|
| stock | 9.47 s | 9.52 s | 1.77 GB | 414 MB |
| compressed_butterfly (current code) | 15.20 s | 15.29 s | **6.27 GB** | 16 MB |

The retained-KV win is real. The peak-memory loss is structural. The current kernel does `mx.take(k, raw_idx) → mx.take(k_summary, summary_idx) → mx.concatenate(raw_k, summary_k) → SDPA(q, union_k, union_v)` per query block, accumulates outputs into a Python list, and concatenates at the end. MLX is lazy: until `mx.eval` runs, the computation graph holds references to source tensors and intermediate gathers across all chunks. Tuning `query_chunk_size` and `local_window_tokens` shifts when the cost is paid; it does not avoid materializing the union K. **The existing code path cannot win on peak memory at long context regardless of parameters.**

DeepSeek V4 (`docs/DeepSeek_V4.pdf`, §2.3, §3.6, Figure 6) describes the architecture that *does* win. The Aleph brief in the conversation produced these load-bearing facts:

- V4 keeps two **parallel KV streams** within a layer — compressed history + SWA tail of `n_win = 128` raw recent tokens — and concatenates them inside one attention call (line 738–747). On Apple silicon without a custom Metal kernel, the equivalent is two SDPA calls merged by online softmax — same math, no union tensor.
- V4's compressor is **`softmax(H·W^Z + B) ⊙ (H·W^KV)` summed over an `m`-block** (Eqs. 9–12, 20–23). Compression ratio block_size → 1 (m=4 for CSA, m'=128 for HCA).
- V4's KV cache is **two-pool, custom-paged** (Figure 6): state cache for `n_win` SWA tail and the in-progress compression block; classical cache for completed compressed blocks at `lcm(m, m')` granularity.
- Compressed-KV used as both K and V requires a **counter-RoPE** on the output at position `−i` so the output carries relative rather than absolute positional content (line 733–737).
- V4's **learned top-k indexer** is the load-bearing differentiator vs deterministic routing (Eqs. 13–17). It is not specified how it is trained in V4; the contract lives in V3.2 / DSA.
- V4 cannot retrofit. It is pretrained from scratch with this attention; the paper has no continued-pretraining or distillation discussion.

The cutover takes V4's *systems* design (two streams, two-pool cache, online softmax, counter-RoPE, paged compressed cache) **wholesale**, and replaces V4's *learned* pieces with deterministic Butterfly equivalents that work on a frozen model.

---

## What Stays Unique To Butterfly (the actual contribution)

**Deterministic `causal_shift` routing as a no-train sparse selector.** V4 chooses which compressed blocks to attend to via a learned multi-head ReLU-scored top-k (`q^I_t · K^IComp_s`, take top-k). This requires training the indexer projections `W^DQ, W^IUQ, W^w` end-to-end. Our path uses `BlockSparseButterflyLayout.build_block_butterfly_layout(... partner_rule="causal_shift", local_window_blocks, sink_count, partner_count)` which produces a **fixed deterministic adjacency** — every query block attends to a structurally-defined set of compressed neighbor blocks. No training. No learned scorer. Same downstream attention math.

The Butterfly topology is not a degraded substitute for a learned indexer; it is a **different design point** with two real properties V4 lacks:

1. **Retrofit-ready on a frozen checkpoint.** The whole point of this cutover. V4 cannot do this; the brief is explicit (§Compatibility Surface, "no discussion of post-hoc adaptation").
2. **Provable mixing.** Butterfly graphs (the FFT-style routing inherited from `bna/butterfly_graph.py`) have logarithmic-depth all-pairs mixing. With `causal_shift`, every pair of compressed blocks is connected by a path of length `O(log(num_blocks))`. The learned indexer offers no such guarantee.

**What we keep from existing code:** `BlockSparseButterflyLayout`, `build_block_butterfly_layout`, `_compressed_butterfly_block_indices`, `CompressedKVCache` (the *outer* class shape — its internals get rewritten), and the public function names `compressed_butterfly_attention`, `compressed_butterfly_attention_active`, `compressed_butterfly_attention_from_cache`. `--mode compressed_butterfly` stays.

**What gets replaced:** the entire body of those three functions; the body of `CompressedKVCache.update_and_fetch`; the prefill/decode dispatch in `bna/integrations/qwen_mlx.py`; the bench harness's mode handling stays as-is at the surface but learns about the new state-cache fields.

---

## Honest Up-Front Boundaries

What this cutover *can* prove if it works:
- That the V4 systems architecture is portable to MLX on Apple silicon without a custom Metal kernel.
- That deterministic Butterfly routing can replace V4's learned top-k indexer in retrofit settings, with measurable peak-memory and retained-KV wins on a frozen pretrained checkpoint.
- That mean-pool compression is sufficient (or insufficient) at the quality bar we set, separate from the systems claim.

What this cutover *cannot* prove:
- Quality parity with V4-Pro (we do not have V4-Pro's pretrained weights).
- That a learned compressor wouldn't be strictly better (we are not training one).
- That this beats stock at very-long context (1M+) without further work; we'll go as far as the artifacts support.
- Anything about decode quality if SWA + sparse-compressed loses to dense at the model's attention sink dynamics on this checkpoint.

---

## File Structure

The cutover touches three core files. No new modules. No `_v2` siblings.

| File | Role | What changes |
|------|------|--------------|
| `bna/mlx/attention.py` | Attention kernels | `compressed_butterfly_attention`, `compressed_butterfly_attention_active`, `compressed_butterfly_attention_from_cache` rewritten to multi-stream online-softmax. New private helpers: `_swa_stream_attention`, `_compressed_stream_attention`, `_online_softmax_merge`, `_apply_counter_rope_to_output`, `_compress_block_v4_style`. Existing `_compress_stats` profiling kept and extended. Existing `BlockSparseButterflyLayout` and `build_block_butterfly_layout` reused as-is — they already give us the routing topology we need. |
| `bna/mlx/compressed_cache.py` | KV cache | `CompressedKVCache` gains a real state-cache (`swa_keys`, `swa_values` for the `n_win` tail; `partial_block_keys`, `partial_block_values` for the in-progress compression block) and a real classical cache (`block_keys`, `block_values` of shape `[B, H, num_completed_blocks, dh]` after compression). `update_and_fetch` is replaced by `append` + accessor methods that return `(swa_k, swa_v, partial_k, partial_v, block_k, block_v)` tuples. The compatibility surface for `mlx_lm.utils.maybe_quantize_kv_cache` is preserved (existing methods like `nbytes`, `state`, `meta_state` stay). |
| `bna/integrations/qwen_mlx.py` | Qwen attention swap | `QwenButterflyAttention.__call__` rewritten for compressed-butterfly path: routes to the new multi-stream functions; removes `force_dense_butterfly_decode` for compressed mode (decode now uses the same multi-stream path as prefill); handles per-layer summary cache via the new cache structure; preserves the layer-swap and config-passthrough mechanics. |

Tests:

| File | Role | What changes |
|------|------|--------------|
| `tests/mlx/test_compressed_butterfly_attention.py` | Numerical parity | Existing tests stay (they assert against a NumPy reference at small sizes); all must pass after cutover. |
| `tests/mlx/test_compressed_butterfly_from_cache.py` | Cache-aware parity | Same — must pass. |
| `tests/mlx/test_compressed_kv_cache.py` | Cache mechanics | Will need updates because `update_and_fetch` semantics change. |
| `tests/mlx/test_online_softmax_merge.py` | NEW | Numerical test for the `(m, l, o)` accumulator merge. |
| `tests/mlx/test_compressed_butterfly_swa_stream.py` | NEW | SWA stream alone matches dense-over-window reference. |
| `tests/mlx/test_compressed_butterfly_counter_rope.py` | NEW | Counter-RoPE on compressed-stream output matches the dense-equivalent computation. |
| `tests/pytorch/test_compressed_butterfly_attention.py` | CUDA mirror | Existing PyTorch CUDA mirror tests stay; we do not cut over the CUDA path in this plan. They must continue to pass (they test a different, untouched module). |

Reuse:
- `bna/mlx/attention.py:105` `BlockSparseButterflyLayout` — reused.
- `bna/mlx/attention.py:195` `build_block_butterfly_layout(... partner_rule="causal_shift" ...)` — reused.
- `bna/mlx/attention.py:955` `_compressed_butterfly_block_indices` — replaced (current version computes raw_idx as raw-token positions; new version computes indices into the compressed-block array, far smaller).
- `bna/butterfly_graph.py` — unchanged; provides graph-theoretic primitives for the layout.

---

## Phases

Six phases. Each is gated by either a passing test or a measured number. Stop rules apply between phases. The user's hardware constraint (one MLX model in memory at a time) holds throughout — no parallel benches.

### Phase A — Multi-Stream Online-Softmax Beachhead (HCA-style first, no Butterfly routing yet)

**Why first.** Sanity check that the architectural fix (no union K, online softmax over two streams) actually drops peak memory. Use HCA-shape (dense over *all* compressed blocks; no Butterfly routing yet) because that's the simplest variant and the brief identifies it as the cleanest first beachhead. If this doesn't beat the floor, the architecture is broken and Phase B onward is moot.

**Files:**
- Modify: `bna/mlx/attention.py` — add private helpers `_swa_stream_attention`, `_compressed_stream_attention`, `_online_softmax_merge`. Rewrite `compressed_butterfly_attention` body to call them with HCA-shape (all compressed blocks).
- Modify: `bna/mlx/compressed_cache.py` — extend with `swa_keys`, `swa_values`, `partial_block_keys`, `partial_block_values`, `block_keys`, `block_values`.
- Test: `tests/mlx/test_online_softmax_merge.py` (new), `tests/mlx/test_compressed_butterfly_swa_stream.py` (new).

- [ ] **Step A.1: Failing test for online softmax merge primitive.** Write `tests/mlx/test_online_softmax_merge.py`:
  ```python
  from __future__ import annotations
  import math
  import numpy as np
  import pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import _online_softmax_merge

  def test_online_softmax_merge_two_streams_matches_dense_softmax():
      rng = np.random.default_rng(7)
      B, H, Tq, dh = 1, 2, 4, 8
      Ka, Kb = 5, 7
      q = rng.standard_normal((B, H, Tq, dh), dtype=np.float32)
      ka = rng.standard_normal((B, H, Ka, dh), dtype=np.float32)
      va = rng.standard_normal((B, H, Ka, dh), dtype=np.float32)
      kb = rng.standard_normal((B, H, Kb, dh), dtype=np.float32)
      vb = rng.standard_normal((B, H, Kb, dh), dtype=np.float32)
      scale = 1.0 / math.sqrt(dh)
      sa = (q @ ka.transpose(0, 1, 3, 2)) * scale
      sb = (q @ kb.transpose(0, 1, 3, 2)) * scale
      ma, mb = sa.max(axis=-1, keepdims=True), sb.max(axis=-1, keepdims=True)
      la = np.exp(sa - ma).sum(axis=-1, keepdims=True)
      lb = np.exp(sb - mb).sum(axis=-1, keepdims=True)
      oa = (np.exp(sa - ma) @ va) / la
      ob = (np.exp(sb - mb) @ vb) / lb
      out = _online_softmax_merge(
          (mx.array(oa), mx.array(la), mx.array(ma)),
          (mx.array(ob), mx.array(lb), mx.array(mb)),
      )
      mx.eval(out)
      ref_full = np.concatenate([sa, sb], axis=-1)
      ref_v = np.concatenate([va, vb], axis=-2)
      ref_w = np.exp(ref_full - ref_full.max(axis=-1, keepdims=True))
      ref_w = ref_w / ref_w.sum(axis=-1, keepdims=True)
      ref = ref_w @ ref_v
      assert np.allclose(np.asarray(out, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)
  ```
  Run: `.venv-macos-metal/bin/python -m pytest tests/mlx/test_online_softmax_merge.py -v` → expected FAIL with `ImportError: cannot import name '_online_softmax_merge'`.

- [ ] **Step A.2: Implement `_online_softmax_merge`.** In `bna/mlx/attention.py` near the other private helpers:
  ```python
  def _online_softmax_merge(
      a: tuple[mx.array, mx.array, mx.array],
      b: tuple[mx.array, mx.array, mx.array],
  ) -> mx.array:
      """Merge two stream outputs by online softmax (m, l, o) accumulators.

      Each stream provides (output, l, m) where:
          output: weighted-sum-of-V already divided by stream's l
          l:      sum_j exp(score_j - m)
          m:      max score for the stream
      Returns merged output identical to a single softmax over the union.
      """
      o_a, l_a, m_a = a
      o_b, l_b, m_b = b
      m_total = mx.maximum(m_a, m_b)
      coeff_a = mx.exp(m_a - m_total)
      coeff_b = mx.exp(m_b - m_total)
      l_total = coeff_a * l_a + coeff_b * l_b
      out = (coeff_a * l_a * o_a + coeff_b * l_b * o_b) / l_total
      return out
  ```
  Run the test: `pytest tests/mlx/test_online_softmax_merge.py -v` → PASS.

- [ ] **Step A.3: Failing test for SWA stream alone.** Write `tests/mlx/test_compressed_butterfly_swa_stream.py`:
  ```python
  from __future__ import annotations
  import math
  import numpy as np
  import pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import _swa_stream_attention

  def test_swa_stream_matches_dense_over_window_for_each_query():
      rng = np.random.default_rng(11)
      B, H, T, dh = 1, 2, 32, 8
      n_win = 8
      q = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      k = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      v = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      scale = 1.0 / math.sqrt(dh)
      out_streamed, l_streamed, m_streamed = _swa_stream_attention(
          mx.array(q), mx.array(k), mx.array(v),
          n_win=n_win, scale=scale,
      )
      mx.eval(out_streamed, l_streamed, m_streamed)
      ref = np.zeros_like(q)
      for t in range(T):
          start = max(0, t - n_win + 1)
          kk = k[:, :, start : t + 1, :]
          vv = v[:, :, start : t + 1, :]
          scores = (q[:, :, t : t + 1, :] @ kk.transpose(0, 1, 3, 2)) * scale
          w = np.exp(scores - scores.max(axis=-1, keepdims=True))
          w = w / w.sum(axis=-1, keepdims=True)
          ref[:, :, t : t + 1, :] = w @ vv
      assert np.allclose(np.asarray(out_streamed, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)
  ```
  Run → expected FAIL with `ImportError: cannot import name '_swa_stream_attention'`.

- [ ] **Step A.4: Implement `_swa_stream_attention`.** In `bna/mlx/attention.py`, returning `(o, l, m)` tuples (not normalized to final out). Use `mx.fast.scaled_dot_product_attention` if available with `mask=mx.array(causal_local_mask)` to constrain to the `n_win` window; otherwise fall back to manual stable_masked_softmax. Compute `m, l` separately from `o` so the caller can merge. Critically, run `mx.eval(o, l, m)` before returning to break the lazy graph — the whole point of streaming. Run the test → PASS.

- [ ] **Step A.5: Failing test for HCA-style compressed stream.** Write `tests/mlx/test_compressed_butterfly_hca_stream.py`:
  ```python
  from __future__ import annotations
  import math
  import numpy as np
  import pytest
  mx = pytest.importorskip("mlx.core")
  from bna.mlx.attention import _compressed_stream_attention

  def test_hca_stream_dense_over_compressed_blocks():
      rng = np.random.default_rng(13)
      B, H, T, dh = 1, 2, 32, 8
      block_size = 8
      num_blocks = T // block_size
      q = rng.standard_normal((B, H, T, dh), dtype=np.float32)
      k_summary = rng.standard_normal((B, H, num_blocks, dh), dtype=np.float32)
      v_summary = rng.standard_normal((B, H, num_blocks, dh), dtype=np.float32)
      scale = 1.0 / math.sqrt(dh)
      out_s, l_s, m_s = _compressed_stream_attention(
          mx.array(q), mx.array(k_summary), mx.array(v_summary),
          block_size=block_size, scale=scale, routed_indices=None,
      )
      mx.eval(out_s, l_s, m_s)
      ref = np.zeros_like(q)
      for t in range(T):
          q_block = t // block_size
          allowed = q_block
          if allowed <= 0:
              continue
          kk = k_summary[:, :, :allowed, :]
          vv = v_summary[:, :, :allowed, :]
          scores = (q[:, :, t : t + 1, :] @ kk.transpose(0, 1, 3, 2)) * scale
          w = np.exp(scores - scores.max(axis=-1, keepdims=True))
          w = w / w.sum(axis=-1, keepdims=True)
          ref[:, :, t : t + 1, :] = w @ vv
      mask = np.ones((B, H, T, dh), dtype=bool)
      for t in range(T):
          if (t // block_size) <= 0:
              mask[:, :, t, :] = False
      assert np.allclose(np.asarray(out_s, dtype=np.float32)[mask], ref[mask], atol=3e-4, rtol=3e-4)
  ```
  Run → FAIL on import.

- [ ] **Step A.6: Implement `_compressed_stream_attention`.** In `bna/mlx/attention.py`. Two cases on `routed_indices`:
  - `routed_indices is None` (HCA-style): for each query block, keys are *all* compressed blocks strictly earlier than the query block (causal mask). Single SDPA per query block over a tiny K = q_block_idx. Return `(o, l, m)`.
  - `routed_indices is not None` (Phase B): per-query-block selection by Butterfly causal_shift; gather compressed blocks per query block via `mx.take`. Return `(o, l, m)`.
  Run `mx.eval(o, l, m)` before return. Test passes.

- [ ] **Step A.7: Rewrite `compressed_butterfly_attention` body to call the two streams + merge.** In `bna/mlx/attention.py:1108-1213`, replace the gather→concat→SDPA body with:
  ```python
  swa_o, swa_l, swa_m = _swa_stream_attention(q, k, v, n_win=local_window_tokens, scale=scale)
  k_summary, v_summary = _compress_block_v4_style(k, v, block_size=block_size)  # mean for now
  hca_o, hca_l, hca_m = _compressed_stream_attention(
      q, k_summary, v_summary, block_size=block_size, scale=scale, routed_indices=None,
  )
  out = _online_softmax_merge((swa_o, swa_l, swa_m), (hca_o, hca_l, hca_m))
  return out, None
  ```
  `_compress_block_v4_style` is initially mean pool (Phase A); Phase B leaves it parameter-free; later phases or scope additions can replace it. Run `tests/mlx/test_compressed_butterfly_attention.py` → MUST still pass at the existing `atol=3e-4` (this is the small-size NumPy parity test from the existing test file, lines 90–124). If it doesn't, the merge math or mask handling is wrong; debug before continuing.

- [ ] **Step A.8: 32k bench** with the multi-stream HCA-shape implementation:
  ```bash
  /Volumes/VIXinSSD/butterfly/.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
    --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
    --hf-home /Volumes/VIXinSSD/hf_cache --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub --hf-offline \
    --mode compressed_butterfly --block-partner-rule causal_shift \
    --compressed-local-window-tokens 128 --seq-lens 32768 --decode-len 8 --repeats 1 \
    --chunk-size 384 --kv-step 384 --query-chunk-size 64 --block-size 128 \
    --butterfly-decode-backend stock --skip-multi-turn --skip-quality \
    --stage-timeout-sec 900 \
    --out-dir results/benchmarks/qwen35_0p8b_mlx/pA_hca_stream_32768
  ```
  Pre-flight: vm_stat must show ≥ 9 GB reclaimable (the new path SHOULD use less; if for some reason it OOMs, the architecture is wrong and Phase A failed).

- [ ] **Step A.9: Phase A decision gate.** Read `pA_hca_stream_32768/results.json` and apply:
  - If `peak_memory_bytes ≤ 1.10 × stock_peak (1.95 GB)` AND `e2e_sec ≤ 1.20 × stock_e2e (11.40 s)`: PASS — proceed to Phase B.
  - If `peak_memory_bytes > stock_peak` (any improvement at all, even partial): RECORD as "partial — investigate" and proceed to Phase B with a flagged concern.
  - If `peak_memory_bytes > 3 × stock_peak (5.31 GB)` OR e2e regresses by >50%: FAIL — halt. Write blocker memo. The MLX-Python online-softmax approach is insufficient and we'd need a Metal kernel.

- [ ] **Step A.10: Append Phase A row to OVERNIGHT_LOG.md and commit.**
  ```bash
  git add bna/mlx/attention.py bna/mlx/compressed_cache.py \
          tests/mlx/test_online_softmax_merge.py \
          tests/mlx/test_compressed_butterfly_swa_stream.py \
          tests/mlx/test_compressed_butterfly_hca_stream.py \
          results/benchmarks/qwen35_0p8b_mlx/pA_hca_stream_32768 \
          results/benchmarks/qwen35_0p8b_mlx/OVERNIGHT_LOG.md
  git commit -m "$(cat <<'EOF'
  phase A: multi-stream online-softmax cutover (HCA-shape, no routing)

  Replace gather→concat→SDPA with two parallel streams:
  - SWA over n_win raw tokens
  - Dense over all causally-prior compressed blocks (HCA-style)
  Merge via online softmax (m, l, o) accumulators.
  No union K/V tensor materialized.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```

### Phase B — Re-introduce Butterfly `causal_shift` Routing Over Compressed Blocks (CSA-shape replacement)

**Why.** This is where Butterfly's unique contribution lives: the routed sparse selector, deterministic instead of learned. We replace HCA's "all causal-prior compressed blocks" with `BlockSparseButterflyLayout`'s routed neighbor blocks per query block. Same multi-stream, same online softmax merge.

**Files:**
- Modify: `bna/mlx/attention.py` — `compressed_butterfly_attention` now passes `routed_indices` derived from `BlockSparseButterflyLayout` to `_compressed_stream_attention`. The layout is built once per layer per `kv_len` bucket and cached on the module instance.
- Test: `tests/mlx/test_compressed_butterfly_attention.py:90-124` (existing) — already tests against a NumPy reference that uses the *Butterfly-routed* candidate set. Must continue to pass.

- [ ] **Step B.1: Confirm the existing parity test exercises Butterfly routing.** Read `tests/mlx/test_compressed_butterfly_attention.py:70-87` — `_reference_compressed` walks `_valid_block_neighbors(layout, block_idx)` (line 70) and only includes summaries from those neighbors (lines 73–76). The reference IS Butterfly-routed. Good.

- [ ] **Step B.2: Wire `routed_indices` through.** In `compressed_butterfly_attention`, build `routed_indices` from `layout.block_neighbors` (shape `[H, num_blocks, partner_count]`) — for each query block, the list of valid neighbor block indices, padded with `-1`. Pass into `_compressed_stream_attention`.

- [ ] **Step B.3: Implement routed branch in `_compressed_stream_attention`.** When `routed_indices is not None`, per query block: gather `k_summary[:, :, routed_indices[h, q_block, :valid_count]]` (shape `[B, H, valid_count, dh]`), run SDPA with appropriate mask. **Critical:** do not loop in Python over query blocks at framework level — vectorize with `mx.take` over the `[H, num_blocks, partner_count]` index tensor and a per-block mask. Run the existing test → PASS.

- [ ] **Step B.4: Layout cache.** `compressed_butterfly_attention` currently rebuilds `BlockSparseButterflyLayout` per call. Add a module-level `_layout_cache: dict[tuple, BlockSparseButterflyLayout]` keyed by `(seq_len, block_size, num_kv_heads, layer_idx, partner_rule, local_window_blocks, sink_count, partner_count)`. Replace the `build_block_butterfly_layout(...)` call with cached lookup. This addresses the layout-construction-per-call overhead noted in the prior plan's Scaling Target section.

- [ ] **Step B.5: 32k bench with Butterfly routing.**
  ```bash
  /Volumes/VIXinSSD/butterfly/.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
    --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
    --hf-home /Volumes/VIXinSSD/hf_cache --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub --hf-offline \
    --mode compressed_butterfly --block-partner-rule causal_shift \
    --compressed-local-window-tokens 128 --seq-lens 32768 --decode-len 8 --repeats 1 \
    --chunk-size 384 --kv-step 384 --query-chunk-size 64 --block-size 128 \
    --butterfly-decode-backend stock --skip-multi-turn --skip-quality \
    --stage-timeout-sec 900 \
    --out-dir results/benchmarks/qwen35_0p8b_mlx/pB_butterfly_routed_32768
  ```

- [ ] **Step B.6: Phase B decision gate.**
  - If peak ≤ Phase A peak (Butterfly routing didn't *increase* peak vs HCA): proceed to Phase C.
  - If peak > 1.5 × Phase A peak: investigate and fix the routed-gather pattern before proceeding.

- [ ] **Step B.7: Append Phase B row to OVERNIGHT_LOG.md and commit.**
  ```
  phase B: deterministic Butterfly causal_shift routing as sparse selector
  ```

### Phase C — Counter-RoPE on Compressed-Stream Output

**Why.** V4 §2.3.3 (lines 733–737): when compressed KV serves as both K and V, applying a counter-RoPE at position `−i` to the output `o_{t,i}` is necessary to make the output carry *relative* rather than *absolute* positional content. Skipping this would silently degrade quality over long context. Aleph brief flagged this as the non-obvious detail when retrofitting any compression scheme over RoPE.

**Files:**
- Modify: `bna/mlx/attention.py` — add `_apply_counter_rope_to_output(out, query_positions, rope_dims, base, partial_rotary_factor)`; call after `_online_softmax_merge` on the compressed-stream contribution (NOT the SWA stream — SWA tokens already have correct RoPE).
- Test: `tests/mlx/test_compressed_butterfly_counter_rope.py` (new).

- [ ] **Step C.1: Failing test.** Write `tests/mlx/test_compressed_butterfly_counter_rope.py`. The contract: for a single query at position `t` attending to a single compressed-block summary built from raw tokens at positions `[b*m, ..., b*m + m - 1]` whose K is the position-`b*m + (m-1)/2` mean of raw RoPE'd K, applying counter-RoPE at position `−t` to the post-attention output makes the result depend only on `t − (b*m + (m-1)/2)` rather than on absolute `t`. Test that two queries at different absolute positions but the same relative offset produce the same output:
  ```python
  from __future__ import annotations
  import math
  import numpy as np
  import pytest
  mx = pytest.importorskip("mlx.core")
  from mlx_lm.models.rope_utils import initialize_rope
  from bna.mlx.attention import _apply_counter_rope_to_output

  def _rotate_half(x):
      half = x.shape[-1] // 2
      return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

  def _apply_rope_np(x, pos, base, dim):
      half = dim // 2
      freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
      angles = pos[:, None] * freqs[None, :]
      cos = np.repeat(np.cos(angles), 2, axis=-1)
      sin = np.repeat(np.sin(angles), 2, axis=-1)
      return x * cos + _rotate_half(x) * sin

  def test_counter_rope_makes_output_relative():
      rng = np.random.default_rng(29)
      B, H, dh = 1, 2, 8
      base = 10000.0
      m = 4
      v_block = rng.standard_normal((B, H, 1, dh), dtype=np.float32)
      out_at_t = {}
      for t in (12, 20, 28):
          out_no_rope = v_block.copy()
          out = _apply_counter_rope_to_output(
              mx.array(out_no_rope),
              query_positions=mx.array(np.array([t], dtype=np.int32)),
              rope_dims=dh, base=base, partial_rotary_factor=1.0,
          )
          mx.eval(out)
          out_at_t[t] = np.asarray(out, dtype=np.float32)
      ref = _apply_rope_np(v_block.reshape(-1, dh), np.array([-12], dtype=np.float32), base, dh).reshape(B, H, 1, dh)
      assert np.allclose(out_at_t[12], ref, atol=3e-4, rtol=3e-4)
      diff_relative = out_at_t[20] - _apply_rope_np(v_block.reshape(-1, dh), np.array([-20], dtype=np.float32), base, dh).reshape(B, H, 1, dh)
      assert np.allclose(diff_relative, 0.0, atol=3e-4)
  ```
  Run → expected FAIL with `ImportError: cannot import name '_apply_counter_rope_to_output'`.

- [ ] **Step C.2: Implement `_apply_counter_rope_to_output`.** Use the same RoPE primitive that `_attn_rope` in `bna/integrations/qwen_mlx.py:311` already uses, but with negated positions. Run test → PASS.

- [ ] **Step C.3: Wire into `compressed_butterfly_attention`.** After computing `hca_o` (the compressed stream output, before merging), apply counter-RoPE to it. The SWA stream is left alone. Then merge.

- [ ] **Step C.4: 32k bench (correctness + perf check).**
  ```
  out-dir: pC_counter_rope_32768
  ```

- [ ] **Step C.5: Phase C decision gate.** Counter-RoPE adds compute. Acceptable: peak unchanged, e2e within 10% of Phase B. If e2e regresses by >25%, optimize the RoPE application before proceeding.

- [ ] **Step C.6: Append Phase C row + commit.**

### Phase D — Two-Pool Cache Layout

**Why.** Until this phase, the cache still mostly looks like the old single-pool `CompressedKVCache`. Phase D adopts V4's two-pool design (Figure 6): explicit `state_cache` for the SWA tail + in-progress compression block, and a `classical_cache` for completed compressed blocks. This is the cache shape the multi-stream attention actually wants.

**Files:**
- Modify: `bna/mlx/compressed_cache.py` — `CompressedKVCache.__init__` declares the new fields. `update_and_fetch` is replaced with `append_tokens(keys, values)` returning a tuple of references the multi-stream attention needs.
- Modify: `bna/integrations/qwen_mlx.py` — call sites updated.
- Modify: `tests/mlx/test_compressed_kv_cache.py` — update existing tests to the new API.
- Test: `tests/mlx/test_compressed_kv_cache.py` (existing, edited).

- [ ] **Step D.1: Update tests for new API.** Edit `tests/mlx/test_compressed_kv_cache.py` to assert the new fields and methods. Run → expected FAIL (new methods don't exist yet).
- [ ] **Step D.2: Implement two-pool fields.** Add `swa_keys, swa_values` (rolling fixed-size buffer of `n_win`); `partial_block_keys, partial_block_values` (size 0 to `block_size − 1`); `block_keys, block_values` (`[B, H, num_completed_blocks, dh]`).
- [ ] **Step D.3: Implement `append_tokens`.** Append to partial; when partial reaches `block_size`, compress via `_compress_block_v4_style`, append to `block_keys/values`, clear partial. Maintain SWA window separately.
- [ ] **Step D.4: Implement state accessors.** `get_state()` returns `(swa_k, swa_v, partial_k, partial_v, block_k, block_v)`.
- [ ] **Step D.5: Update `bna/integrations/qwen_mlx.py`.** Replace `cache.update_and_fetch(keys, values)` with `cache.append_tokens(keys, values); state = cache.get_state()`.
- [ ] **Step D.6: All tests pass.** Run the full test suite (the four existing test files plus the new ones).
- [ ] **Step D.7: 32k bench.**
  ```
  out-dir: pD_two_pool_cache_32768
  ```
- [ ] **Step D.8: Phase D decision gate.** Retained KV in `cache_storage_after_prefill.total_bytes` should still be ~16 MB or smaller. Peak should not regress vs Phase C.
- [ ] **Step D.9: Append Phase D row + commit.**

### Phase E — Decode-Time Multi-Stream (no more dense fallback)

**Why.** Until this phase, decode under `--butterfly-decode-backend stock` falls back to dense SDPA over the trimmed local-window tail (the "lossy decode" issue from the prior plan's Truth Boundary). Phase E removes that fallback for the compressed_butterfly path: decode uses the same multi-stream attention as prefill. The decode `Tq=1` case is just the limit of the prefill code path with `Tq=1`.

**Files:**
- Modify: `bna/integrations/qwen_mlx.py:1293-1306` — remove `force_dense_butterfly_decode` for compressed_butterfly path; route Tq=1 calls into `compressed_butterfly_attention_from_cache`.
- Modify: `bna/mlx/attention.py:1414` `compressed_butterfly_attention_from_cache` — body now mirrors the multi-stream prefill, just for `Tq=1`.

- [ ] **Step E.1: Existing test must keep passing.** `tests/mlx/test_compressed_butterfly_from_cache.py` already tests this function against a NumPy reference. Run after each edit.
- [ ] **Step E.2: Rewrite `compressed_butterfly_attention_from_cache` body** as a thin call into the two-stream + merge + counter-RoPE path with `Tq=1`.
- [ ] **Step E.3: Update `qwen_mlx.py` decode dispatch.** When `cache` is `CompressedKVCache` and we are decoding, no longer call `_dense_fallback`; call `compressed_butterfly_attention_from_cache`.
- [ ] **Step E.4: All tests pass.**
- [ ] **Step E.5: 32k bench (decode now goes through real multi-stream).**
  ```
  out-dir: pE_real_decode_32768
  ```
- [ ] **Step E.6: Phase E decision gate.** Decode tokens/sec must be reasonable (within 3× stock decode speed; we're not optimizing decode aggressively yet). If decode is catastrophically slow (>10× stock), there's a per-token overhead bug; fix before proceeding.
- [ ] **Step E.7: Append Phase E row + commit.**

### Phase F — Honest Bench

**Why.** With Phases A–E complete, run a clean comparison vs stock at 32k and (only if 32k wins) a small ladder.

- [ ] **Step F.1: Re-run stock 32k baseline** (drift check; the floor must still be the same):
  ```
  out-dir: pF_stock_32768
  ```
- [ ] **Step F.2: Re-run compressed_butterfly 32k** with the now-fully-cutover code:
  ```
  out-dir: pF_compressed_butterfly_32768
  ```
- [ ] **Step F.3: Phase F decision gate at 32k.** Compute deltas:
  - peak_memory_bytes: compressed vs stock
  - e2e_sec: compressed vs stock
  - cache_storage_after_prefill: compressed vs stock
  Three buckets:
  - **Win on peak AND retained KV, no worse than 1.20× e2e:** the cutover succeeded. Proceed to Step F.4 (ladder). Update `docs/COMPRESSED_BUTTERFLY_ATTENTION.md` Claim Boundary section honestly.
  - **Win on retained KV only, peak ≤ stock:** partial win. Proceed to F.4 cautiously. Document that quality eval is the next gate.
  - **Peak still > stock:** the cutover failed at the architecture level on MLX. Stop. Write `BLOCKER_MEMO_2026-04-25_post_cutover.md` documenting what V4 systems pieces were ported and why they still don't suffice on MLX without a Metal kernel.

- [ ] **Step F.4: Ladder up cautiously**, only if F.3 said go: 64k, then 128k, then 256k. For each rung: stock + compressed_butterfly, sequential, vm_stat pre-flight, stage-timeout 900s, no-retry-on-OOM.
- [ ] **Step F.5: Final commit + final OVERNIGHT_LOG.md row.**
- [ ] **Step F.6: Update `docs/COMPRESSED_BUTTERFLY_ATTENTION.md` Claim Boundary section** with the actual measured deltas. No promotion of unmeasured claims.

---

## Stop Rules

These bind throughout the cutover.

1. **Phase A is the architecture gate.** If multi-stream online-softmax + HCA-shape doesn't drop peak below 3× stock at 32k, the architecture cannot be made to work in MLX-Python and the cutover stops. We do not proceed to Phase B/C/D/E in that case; we write a blocker memo and the user decides whether a Metal kernel is in scope.
2. **OOM = halt, not retry.** Stop Rule #3 from the prior plan applies verbatim. Capture exact CLI, stdout tail, vm_stat. Do not increase context.
3. **Test regressions = revert.** The four existing test files (`tests/mlx/test_compressed_butterfly_attention.py`, `tests/mlx/test_compressed_butterfly_from_cache.py`, `tests/mlx/test_compressed_kv_cache.py`, `tests/pytorch/test_compressed_butterfly_attention.py`) must pass at every commit. If any phase breaks them and the fix is not obvious within the phase budget, revert and re-evaluate.
4. **No new module names.** No `_v2`, no `compressed_butterfly_2`, no parallel implementation files. Inplace cutover.
5. **One model in inference at a time.** Sequential benches only. No parallel processes loading MLX models.
6. **No quality claims without a quality run.** Phases A–F measure speed/peak/retained-KV. Quality preservation is unproven until a retrieval probe runs (out of scope for this cutover; can be a future scoped task).
7. **No claim that this is V4.** It is V4-systems + Butterfly-routing. The Aleph brief showed V4's learned indexer is load-bearing for V4's own quality claims; we deliberately replace it with structural routing and accept the consequence.
8. **Counter-RoPE is non-negotiable.** If Phase C is skipped or broken, document it explicitly in the OVERNIGHT_LOG; do not silently ship.

---

## Expected Artifacts

By the end of execution, on this branch (`butterfly/overnight-2026-04-24`) and in this worktree:

- This plan committed at `docs/superpowers/plans/2026-04-25-butterfly-cutover.md`.
- `bna/mlx/attention.py`, `bna/mlx/compressed_cache.py`, `bna/integrations/qwen_mlx.py` cutover-edited.
- New tests:
  - `tests/mlx/test_online_softmax_merge.py`
  - `tests/mlx/test_compressed_butterfly_swa_stream.py`
  - `tests/mlx/test_compressed_butterfly_hca_stream.py`
  - `tests/mlx/test_compressed_butterfly_counter_rope.py`
- Updated tests: `tests/mlx/test_compressed_kv_cache.py` (new fields/methods).
- All four pre-existing test files still passing.
- Per-phase results dirs in `results/benchmarks/qwen35_0p8b_mlx/`:
  - `pA_hca_stream_32768/`, `pB_butterfly_routed_32768/`, `pC_counter_rope_32768/`, `pD_two_pool_cache_32768/`, `pE_real_decode_32768/`, `pF_stock_32768/`, `pF_compressed_butterfly_32768/`
  - If F.4 ran: `pF_stock_65536/`, `pF_compressed_butterfly_65536/`, etc.
- `OVERNIGHT_LOG.md` with one row per phase including peak/e2e/cache deltas vs stock.
- Either an updated `docs/COMPRESSED_BUTTERFLY_ATTENTION.md` Claim Boundary (if the cutover wins) or `BLOCKER_MEMO_2026-04-25_post_cutover.md` (if Phase A or F.3 says stop).

---

## What This Proves (the unique claim, if F.3 says win)

Three things — separate them and don't conflate:

1. **Systems portability.** The V4 attention systems design (two-pool KV, multi-stream, online-softmax merge, counter-RoPE) ports to MLX on Apple silicon without a custom Metal kernel. The artifacts are the per-phase bench JSONs.
2. **Deterministic Butterfly routing as a viable indexer.** A no-train sparse selector built from `causal_shift` Butterfly topology produces a measurable retained-KV / peak-memory profile, on a frozen pretrained checkpoint. The artifact is Phase B's bench delta vs Phase A.
3. **Retrofit feasibility.** This works on Qwen 3.5 0.8B 4-bit MLX *without* retraining. V4 cannot do this.

---

## What Not To Claim (still)

- That this matches V4-Pro quality. We do not have V4-Pro weights and we are not training a learned indexer.
- That our compressor is competitive with V4's learned compressor. Mean pool is a placeholder.
- That decoding quality is preserved. No quality eval in this plan.
- That this scales to 1M without further work. We measure to 256k at most, and only if 32k and 64k both win cleanly.
- That Butterfly routing is "better" than learned top-k in any general sense. We claim it is *viable on a frozen model* — a different design point.
