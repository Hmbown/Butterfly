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
