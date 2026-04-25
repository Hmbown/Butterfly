# Butterfly Overnight Log — 2026-04-25

## Phase 0: Baselines (locked)

Apple Silicon, MLX, Qwen3.5-0.8B-4bit, decode_len=8, repeats=1.

| Variant | prefill_sec | e2e_sec | peak_memory_bytes | cache_after_prefill (bytes) |
|---|---:|---:|---:|---:|
| stock 32768 | 9.472026957999333 | 9.516935833002208 | 1774884158 | 414326784 |
| compressed_butterfly w64 qc64 32768 | 15.240854582996690 | 15.287026457997854 | 6273298171 | 16379904 |

**Floor confirmation gate (15% drift threshold):**
- stock e2e drift vs 9.34s: +1.89%
- compressed e2e drift vs 14.64s: +4.42%
- decision: PROCEED

Pre-flight free memory snapshots:
- before stock: 10.32 GB reclaimable (free=169481 + inactive=472606 + speculative=34011 pages × 16 KiB)
- before compressed: 10.48 GB reclaimable (free=167719 + inactive=487465 + speculative=31576 pages × 16 KiB)

## Phase A: Multi-stream online-softmax cutover (HALTED at gate)

Architectural cutover landed in `bna/mlx/attention.py`:
- `_online_softmax_merge((o_a, l_a, m_a), (o_b, l_b, m_b))` — merge streams without materializing union K/V.
- `_swa_stream_attention(q, k, v, n_win, scale)` — sliding-window stream returning `(o, l, m)`.
- `_swa_stream_from_cache(q, tail_k, tail_v, query_positions, tail_start, n_win, scale)` — SWA against the cached tail.
- `_compressed_stream_attention(q, k_summary, v_summary, block_size, scale, routed_indices)` — HCA dense or routed CSA over compressed blocks.
- `_compressed_stream_from_cache(...)` — routed CSA over the cache's compressed summaries.
- `_compressed_butterfly_routed_indices(layout, kv_len, local_window_tokens)` — Butterfly `causal_shift` indices, filtered against the SWA window so streams do not double-count.
- `compressed_butterfly_attention` and `compressed_butterfly_attention_from_cache` rewritten to two streams + online-softmax merge. No union K/V tensor materialized.

Tests (all green):
- `tests/mlx/test_online_softmax_merge.py` (new, 2 tests)
- `tests/mlx/test_compressed_butterfly_swa_stream.py` (new, 2 tests)
- `tests/mlx/test_compressed_butterfly_hca_stream.py` (new, 2 tests)
- `tests/mlx/test_compressed_butterfly_attention.py` (existing, 3 tests, still pass against NumPy reference)
- `tests/mlx/test_compressed_butterfly_from_cache.py` (existing, 1 test, still pass)
- `tests/mlx/test_compressed_kv_cache.py` (existing, 2 tests, still pass)

| Variant | prefill_sec | e2e_sec | peak_memory_bytes | cache_after_prefill (bytes) |
|---|---:|---:|---:|---:|
| compressed_butterfly multi-stream w64 qc64 32768 | (in stdout: 11.151) | 11.201 | 6,209,013,346 | 16,379,904 |

**vs Phase 0 compressed (same flags):**
- e2e: -27% (15.29 → 11.20 s) — multi-stream wins on speed.
- peak: -1.0% (6.27 GB → 6.21 GB) — peak essentially unchanged.
- retained KV: identical (16.4 MB).

**vs stock floor:**
- e2e: 1.18× stock (still slower).
- peak: 3.50× stock (still 3.5× more memory).
- retained KV: 25× smaller than stock.

**Stop rule #1 from `2026-04-25-butterfly-cutover.md` triggered:** "If multi-stream online-softmax doesn't drop peak below 3× stock at 32k, the architecture cannot be made to work in MLX-Python." We are at 3.50×, above the 3× ceiling. **Halt the cutover.** Phases B-F not executed.

**Diagnostic finding:** the multi-stream rewrite eliminates the union K/V materialization in the compute path and saves 4 s of compute time, but the 4.5 GB of "extra working memory" (compressed peak − stock peak − retained KV) is not in the attention compute itself. Likely candidates (none yet measured):
- `mx.concatenate` in `CompressedKVCache.update_and_fetch` repeatedly allocating new full-history tensors before the trim step inside the same call.
- Lazy-graph retention of activations across the 24 layers' chunked-prefill pass (compressed mode triggers per-layer divergent code paths that may compose differently from stock's monolithic SDPA).
- Per-call layout / index numpy buffers (`_compressed_butterfly_routed_indices` allocates a new `[H, num_blocks, summary_slots]` array per call; not the dominant cost but adds up).

The ~4 s wall-clock improvement from the architectural cutover is real and worth keeping. The peak gap is the remaining problem and requires either:
(a) a Metal kernel for the compressed-stream attention (a fused tile-streaming implementation à la FlashAttention),
(b) instrumentation to identify and fix whichever non-attention component is consuming the missing 4.5 GB,
(c) accepting the current Pareto tradeoff: 25× retained-KV win at the cost of 3.5× peak (only useful for use cases where retained KV matters more than peak, e.g., serving many concurrent long sessions on the same device).

## Phase B-F: not executed (stop rule)

See plan section "Stop Rules" item #1.

## Diagnostic: per-chunk peak memory profile (post-Phase A)

Added `BNA_PEAK_JOURNAL_PATH` env var to `_run_chunked_prefill` in
`scripts/bench_qwen_consumer_mlx.py`. With this set, the bench resets MLX peak
memory before each prefill chunk and records peak after each chunk's
`mx.eval(logits)`. Output is a JSON list of `{chunk_idx, tokens_end,
peak_memory_bytes_chunk, cache_bytes_after_chunk, chunk_sec}`.

### Per-chunk peak: stock vs compressed at 32k

| chunk | tokens_end | stock peak (MB) | compressed peak (MB) | excess (MB) |
|---:|---:|---:|---:|---:|
| 1 | 384 | 1243 | 752 | -491 |
| 5 | 1920 | 1378 | 886 | -492 |
| 20 | 7680 | 1560 | 1171 | -388 |
| 30 | 11520 | 1460 | 1535 | +75 |
| 50 | 19200 | 1398 | 2696 | +1298 |
| 70 | 26880 | 1609 | 4479 | +2871 |
| 80 | 30720 | 1717 | 5595 | +3878 |
| 86 | 32768 | 1281 | 6195 | +4914 |

**Stock per-chunk peak is flat (~1500 MB regardless of T). Compressed grows
linearly+ with cumulative T.**

### Microbenchmarks (component isolation)

Standalone scripts at `/tmp/diag_cache_only.py` and `/tmp/diag_attn_only.py`:

- **`CompressedKVCache.update_and_fetch` alone:** per-chunk peak 2.4 → 9.6 MB
  across 86 chunks of 384 tokens with growing summary buffer. Cache itself is
  bounded.
- **`compressed_butterfly_attention_from_cache` alone:** per-chunk peak 23.3 →
  24.2 MB regardless of growing `k_summary` size. Attention itself is bounded.

### Bypass diagnostic

Added `BNA_DIAG_BYPASS_COMPRESSED_ATTN=1` (now removed) which replaced
`compressed_butterfly_attention_from_cache(...)` with stock
`scaled_dot_product_attention(queries, tail_k_q, tail_v_q, ...)` — same
swapped-layer dispatch path, but the compressed attention math is bypassed.

| chunk | tokens_end | bypassed peak (MB) | full compressed peak (MB) |
|---:|---:|---:|---:|
| 1 | 384 | 752 | 752 |
| 11 | 4224 | 957 | 957 |
| 21 | 8064 | 1202 | 1202 |
| 31 | 11904 | 1578 | (16k bench was earlier) |
| 43 | 16384 | 2145 | 2152 |

**Bypassing the compressed attention compute ENTIRELY does not change the
per-chunk peak growth.** The growth is not in `compressed_butterfly_attention_from_cache`.

### Conclusion

The 4.5 GB excess peak at 32k is **not** in any of:
- `CompressedKVCache` data structure (bounded in isolation),
- `compressed_butterfly_attention_from_cache` math (bounded in isolation),
- the lazy-graph evaluation pattern of the multi-stream helpers (removing inner
  `mx.eval` calls did not change the growth pattern).

It IS specific to having `CompressedKVCache` in place of the standard
`KVCache` for the swapped layers, but emerges only inside the full model
forward pass — i.e., from some interaction between the cache type and how MLX
schedules the rest of the layer's compute (q/k/v projections, RoPE,
`_repeat_kv_to_q_heads`, residual stream, MLP, the next layer's input...).

Most plausible remaining hypothesis: `mx.concatenate`-based summary growth in
`CompressedKVCache._append_summary` produces a chain of intermediate tensors
that downstream graph nodes hold references to until the outer `mx.eval(logits)`
fires; under the standard `KVCache` (which uses fixed-size pre-allocated
buffers and `slice_update`) this chain doesn't exist.

### Recommended next experiment

Rebuild `CompressedKVCache` to use pre-allocated fixed-size buffers with
`slice_update` (no `mx.concatenate`) — match V4's "state cache + classical
cache" design from the Aleph brief (Figure 6, paper §3.6.1). If this hypothesis
is correct, peak should track stock. If not, the issue is somewhere else and
this whole approach is the wrong shape for MLX.

### Phase A2: fixed-buffer cache rebuild (hypothesis FALSIFIED)

`CompressedKVCache` rewritten to V4-style two-pool fixed-size buffers:

- `_tail_buf_keys/values: [B, H, tail_capacity, dh]`, slice-updated in place;
  `tail_capacity = local_window + 2*max_chunk_size + 2*block_size`.
- `_summary_buf_keys/values: [B, H, summary_capacity, dh]`, one slot written
  per completed compression block.
- All `mx.concatenate` replaced with `mx.slice_update`.
- `max_kv_size` and `max_chunk_size` plumbed through `install_compressed_kv_caches`
  and `_prepare_cache` so the buffers can be sized exactly.
- Public API and invariants preserved: `keys`, `values`, `k_summary`,
  `v_summary` are sliced views of the buffers; `state`, `meta_state`,
  `nbytes`, `make_mask`, `is_trimmable`, `trim`, `empty` unchanged.

All 12 tests pass.

| Variant | prefill_sec | e2e_sec | peak_memory_bytes | cache_after_prefill (bytes) |
|---|---:|---:|---:|---:|
| compressed_butterfly fixed-buffer cache, w64 qc64 32768 | (10.61) | 11.410 | 6,213,002,977 | (cache slice valid only) |

Per-chunk peak vs the old concat-based cache:

| chunk | tokens | concat-cache peak (MB) | fixed-buf peak (MB) | diff (MB) |
|---:|---:|---:|---:|---:|
| 1 | 384 | 752 | 767 | +15 |
| 21 | 8064 | 1202 | 1218 | +16 |
| 41 | 15744 | 2096 | 2115 | +19 |
| 61 | 23424 | 3603 | 3627 | +24 |
| 81 | 31104 | 5715 | 5744 | +29 |
| 86 | 32768 | 6195 | 6213 | +18 |

**The growth is identical.** Replacing `mx.concatenate` with `mx.slice_update`
into pre-allocated buffers did NOT change the per-chunk peak growth. The cache
implementation was not the source of the 4.5 GB excess.

### Final diagnostic conclusion

We have eliminated, in order:
1. `CompressedKVCache` data structure (microbench: 2-10 MB).
2. `compressed_butterfly_attention_from_cache` math (microbench: 23 MB).
3. The integrated compressed attention path (bypass test: peak still grows).
4. Inner `mx.eval` calls in the stream helpers (removing did not help).
5. `mx.concatenate`-based cache growth (replacing with `mx.slice_update`
   into fixed buffers did not help).

**The 4.5 GB excess peak at 32k in compressed mode is real, T-dependent, and
NOT in any individually testable component.** Most likely it lives in MLX's
lazy graph behavior when the compressed path's many small ops (multi-stream
SWA + compressed gather + online softmax merge in 6 swapped layers) are
composed with the rest of the model's forward pass — but we cannot pinpoint
the exact site with the diagnostic tools available in pure Python.

### Recommendation

Resolving the peak gap from here requires one of:
- A fused Metal kernel for the compressed-stream attention (FlashAttention-2
  style tile-streaming over routed compressed blocks + SWA tail; out of scope
  in this run).
- Switching to a different MLX abstraction (e.g., `mx.compile` with a static
  graph wrapping the multi-stream attention) — speculative; depends on whether
  MLX's compilation layer would coalesce the intermediates.
- Accepting the current Pareto: 25× retained-KV win + 27% e2e improvement at
  the cost of 3.5× peak. Useful for serving many concurrent long sessions on
  one device; not a peak-memory-wall win on Apple silicon.

### Diagnostic artifacts

- `results/benchmarks/qwen35_0p8b_mlx/diag_stock_32768/peak_journal.json`
- `results/benchmarks/qwen35_0p8b_mlx/diag_compressed_32768/peak_journal.json`
- `results/benchmarks/qwen35_0p8b_mlx/diag_compressed_no_inner_eval_16384/peak_journal.json`
- `results/benchmarks/qwen35_0p8b_mlx/diag_compressed_bypass_attn_16384/peak_journal.json`

## Phase B (overnight 04-25 follow-up): mx.compile + Metal kernel — both BLOCKER

### Path A: mx.compile of the multi-stream inner kernels

Wrapped `_swa_stream_inner_eager`, `_compressed_stream_inner_eager`, and
`_online_softmax_merge_eager` with `mx.compile(..., shapeless=True)` so the
inner score+softmax+matmul fuses into a single static graph and intermediate
buffers can be freed eagerly. Toggle: `BNA_COMPRESS_DISABLE_COMPILE=1`.

12 tests pass with the compiled inner kernels.

Bench (`pB_mxcompile_32768`):
- e2e=11.087s (down from 15.24s in the locked baseline; matches the post-A2
  fixed-buffer measurement).
- peak=6.21 GB. **Unchanged** vs the prior compressed-Butterfly best.

Per-chunk peak journal trajectory mirrors the pre-compile run almost
exactly (chunk 1: 732 MB; chunk 43: 2119 MB; chunk 86: 5917 MB), so
mx.compile improved end-to-end latency but did **not** affect the peak gate.

### Path B: fused Metal kernel for the multi-stream attention

Wrote a single `mx.fast.metal_kernel` that fuses the SWA stream (over
`tail_k`/`tail_v`) and the compressed-routed stream (over `k_summary`/
`v_summary` with `routed_idx[Tq, summary_slots]`) plus online-softmax merge.
Source lives in `bna/mlx/compressed_butterfly_kernel.py`. Toggle:
`BNA_COMPRESSED_BUTTERFLY_KERNEL=1`. Two parity tests added in
`tests/mlx/test_compressed_butterfly_kernel.py` confirm bit-for-bit numerical
parity with the Python multi-stream reference (max diff 1.5e-8 on f32).

Bench (`pB_kernel_32768`):
- e2e=11.92s (a hair slower than the Python compiled path, since the kernel
  is straight scalar-style code; SWA + routed loops × heads × queries).
- peak=6.20 GB. **Unchanged.** Per-chunk trajectory is identical to the
  Python path (chunk 1: 731 MB; chunk 43: 2119 MB; chunk 86: 5917 MB).

### Diagnostic conclusion (updated)

The kernel collapses every Python-level intermediate of the multi-stream
attention into a single output tensor. Despite that, the per-chunk peak is
unchanged. **The 4.5 GB excess in compressed mode is not in
`compressed_butterfly_attention_from_cache` at all.** Earlier OVERNIGHT_LOG
entries already noted that bypassing the compressed attention with stock
SDPA over the trimmed tail did not change the peak; this run is a stronger
form of the same evidence — even when the attention is run as one fused
Metal call writing only `[B, H, Tq, dh]`, the per-chunk peak is the same.

The peak therefore lives outside the swapped attention call. Candidates that
remain to investigate:
- `_repeat_kv_to_q_heads` GQA broadcasts in `bna/integrations/qwen_mlx.py`
  (8x reshape from `[1, 2, *, 128]` to `[1, 16, *, 128]`) firing once per
  compressed layer per chunk.
- The o_proj quantized matmul activation on `y_bt = [B, T, D]` in the
  compressed layer: this is bf16 output of size `1*384*896*2 = 0.7 MB`,
  so unlikely to be the dominant term unless an upstream broadcast
  is being held alive.
- Some interaction between the compressed-cache layer's `mx.eval(out)` and
  the lazy graph for the next stock layer's KV materialization, where the
  stock cache for that next layer transiently holds an O(T) intermediate.

What stayed working:
- mx.compile inner kernels are fine — they shave ~30% off compressed e2e.
- Custom Metal kernel matches Python to machine precision and is safe to
  ship behind the env var if peak relief comes from elsewhere.

Per the task's "Neither works" branch: changes reverted, including the
mx.compile additions and the metal kernel module/test. Bench artifacts
retained under `results/benchmarks/qwen35_0p8b_mlx/pB_mxcompile_*` and
`pB_kernel_*` for future reference.

### Suggested next attempt

The peak almost certainly lives in `bna/integrations/qwen_mlx.py` around the
compressed layer's pre/post-attention plumbing, NOT in the attention math.
Specifically: instrument peak memory before and after each line of the
`elif self.path == "block_sparse" / isinstance(cache, CompressedKVCache):`
branch, paying attention to the four `_repeat_kv_to_q_heads` calls and the
`y_h.transpose(0, 2, 1, 3).reshape(...)` post-attention. If the spike is in
the GQA broadcast, the fix is to run the kernel WITHOUT pre-expanding K/V to
H_q heads — pass `[B, H_kv, *, dh]` and have the kernel index by
`h // (H_q // H_kv)` to gather the correct kv-head, saving an 8x materialized
broadcast per call.

### Path B follow-up: GQA-natural kernel (`pB_kernel_gqa_32768`)

Hypothesis: maybe the four `_repeat_kv_to_q_heads` calls in
`bna/integrations/qwen_mlx.py` (8x materialized broadcast from H_kv=2 to
H_q=16 per `tail_k`/`tail_v`/`k_summary`/`v_summary` per call per
compressed layer per chunk) are what scales with T.

Test: rewrote the Metal kernel to accept GQA-natural K/V at H_kv=2 and
index per-thread `kv_head = h_q // (H_q // H_kv)`. Bypassed the host-side
`_repeat_kv_to_q_heads` calls entirely when the kernel is enabled. Added a
GQA parity test (`test_kernel_handles_gqa_kv_natural_shape`) — passes.

Bench:
- e2e=11.97s, peak=6.21 GB. **Identical** trajectory to the H_q-shape
  kernel run. The GQA broadcast is not the source.

This is decisive: the leak is not in the compressed_butterfly attention
call OR its immediate input preparation. The `_repeat_kv_to_q_heads`
broadcasts are evaluated lazily and apparently fold into the SDPA-style
read pattern without accumulating intermediates.

### Final BLOCKER status

Both Path A (mx.compile) and Path B (Metal kernel, with and without GQA
broadcast bypass) leave the 32k peak at 6.21 GB. Acceptance gate of
5.31 GB is unmet. Per task instructions, all source changes have been
reverted on `bna/mlx/attention.py` and `bna/integrations/qwen_mlx.py`,
and the kernel module/test files have been removed. Bench artifacts under
`pB_mxcompile_*` and `pB_kernel_*` (and `pB_kernel_gqa_32768`) preserved.

The peak excess is somewhere in the model forward pass that is invariant
to attention internals — most likely the post-attention `o_proj` quantized
matmul, the residual add, or the next stock layer's `cache.update_and_fetch`
on a stock cache that accumulates differently when interleaved with
CompressedKVCache layers. A targeted, layer-by-layer peak-memory probe
across the 32-layer prefill would be the next-most-useful diagnostic.
