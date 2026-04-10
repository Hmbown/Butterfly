# Qwen 3.5 4B MLX Benchmark Report
Date: 2026-04-08

Base Butterfly claim, in one slide

- The MLX evidence supports staged Butterfly routing as the durable part of the design.
- Local-only remains a required baseline, but it is not the target story.
- The strongest current MLX result is Butterfly prefill with stock decode at longer context.
- This report does not support a generic "mixing is better" claim.
- This report does not support a Butterfly decode claim.

Scope

- Butterfly / BNA on Apple Silicon only.
- Harness: `scripts/bench_qwen_consumer_mlx.py`
- Checkpoint: `/Volumes/VIXinSSD/models/Qwen3.5-4B-MLX-4bit`
- Core report inputs:
  - `/tmp/butterfly_full_kv_eval/stock`
  - `/tmp/butterfly_full_kv_eval/stock_kv4`
  - `/tmp/butterfly_full_kv_eval/butterfly`
  - `/tmp/butterfly_full_kv_eval/butterfly_kv4`
- Exploratory 64k inputs:
  - `/tmp/butterfly_full_kv_eval_explore/stock_65536`
  - `/tmp/butterfly_full_kv_eval_explore/stock_kv4_65536`
  - `/tmp/butterfly_full_kv_eval_explore/butterfly_65536`
  - `/tmp/butterfly_full_kv_eval_explore/butterfly_kv4_65536`

Configuration guardrails

- Core matrix: `seq_len={2048,8192,32768}`, `decode_len=32`, `repeats=3`
- Exploratory matrix: `seq_len=65536`, `decode_len=32`, `repeats=1`
- Butterfly rows use `decode_backend=stock`. This report does not claim a Butterfly decode win.
- Live benchmark writers now emit Butterfly-first summary/config fields such as `butterfly_decode_backend` and `butterfly_config`. Legacy `wayfinder_*` mirrors remain only for compatibility with older readers.
- Butterfly prefill config from the stored artifacts:
  - `window=64`
  - `landmark_stride=64`
  - `num_cycles=1`
  - `prefill_chunk_size=384`
  - `query_chunk_size=384`
  - `head_chunk_size=2`
  - `fused_dispatch=true`
- `chunk-size <= query-chunk-size` is satisfied in the measured Butterfly runs.
- KV quantization is MLX-LM KV quantization on the full-attention KV cache only:
  - `bits=4`
  - `group_size=64`
  - `quantized_kv_start=0`
  - `policy=post_prefill_before_decode`
  - `preserve_dense_prefix_cache=true`
- Stored prefix caches remain dense. This is not a new persistent cache format.

Core matrix

Arithmetic mean across the three repeats for each row. `peak_memory_bytes` uses the mean of the per-repeat MLX peak counter.

| Variant | Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Peak bytes | Decode fallback steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 2048 | 2.1147 | 0.0153 | 75.18 | 2.5561 | 3576240294 | 0.00 |
| stock+kv4 | 2048 | 1.9904 | 0.0137 | 83.15 | 2.3755 | 3576240294 | 0.00 |
| butterfly | 2048 | 2.0450 | 0.0127 | 85.98 | 2.4172 | 3563786534 | 1.00 |
| butterfly+kv4 | 2048 | 2.0388 | 0.0130 | 87.30 | 2.4054 | 3563786534 | 1.00 |
| stock | 8192 | 9.0243 | 0.0153 | 68.86 | 9.4999 | 3687328382 | 0.00 |
| stock+kv4 | 8192 | 8.3058 | 0.0176 | 76.93 | 8.7217 | 3687328382 | 0.00 |
| butterfly | 8192 | 8.3155 | 0.0138 | 82.21 | 8.7049 | 3727181411 | 1.00 |
| butterfly+kv4 | 8192 | 8.3009 | 0.0178 | 79.28 | 8.7046 | 3727181411 | 1.00 |
| stock | 32768 | 40.4172 | 0.0174 | 66.55 | 40.8981 | 5058225226 | 0.00 |
| stock+kv4 | 32768 | 39.6951 | 0.0288 | 59.67 | 40.2314 | 5058225226 | 0.00 |
| butterfly | 32768 | 34.0861 | 0.0169 | 63.41 | 34.5990 | 4978360363 | 1.00 |
| butterfly+kv4 | 32768 | 33.8331 | 0.0291 | 56.64 | 34.4067 | 4978360363 | 1.00 |

Exploratory 64k

This section is exploratory only. It is useful for long-context direction, but it is not yet a release-ready memory story.

| Variant | Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Peak bytes | Decode fallback steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | 65536 | 95.2526 | 0.0240 | 57.01 | 95.8140 | 6877619276 | 0.00 |
| stock+kv4 | 65536 | 90.4247 | 0.0416 | 50.57 | 91.0575 | 6877619276 | 0.00 |
| butterfly | 65536 | 70.2775 | 0.0182 | 47.41 | 70.9524 | 7472408586 | 1.00 |
| butterfly+kv4 | 65536 | 71.1148 | 0.0453 | 29.56 | 72.1974 | 7472408586 | 1.00 |

Measured facts

- Within the core `2k -> 8k -> 32k` matrix, Butterfly prefill gains grow with context:
  - `2048`: prefill `-3.30%`, e2e `-5.43%` versus stock
  - `8192`: prefill `-7.85%`, e2e `-8.37%`
  - `32768`: prefill `-15.66%`, e2e `-15.40%`
- The strongest measured 32k lever is still Butterfly prefill. At `32768`, Butterfly beats stock on mean prefill (`34.0861s` vs `40.4172s`) and mean e2e (`34.5990s` vs `40.8981s`).
- In plain `butterfly`, decode is fully stock fallback in every measured row:
  - `stock_fallback_share_decode_steps = 1.0`
  - `stock_fallback_reason_counts = {decode_backend_stock: 256}` in the stored `results.json` traces
- In `butterfly+kv4`, decode is also fully stock fallback:
  - `stock_fallback_share_decode_steps = 1.0`
  - `stock_fallback_reason_counts = {quantized_kv_stock_decode: 256}` in the stored `results.json` traces
- KV quantization is genuinely active on the full-attention cache in the kv4 rows:
  - `quantized_entries = 8`
  - `active = true`
- KV quantization did not reduce the measured peak-memory counter in this pass:
  - stock and stock+kv4 have identical measured peaks at `2048`, `8192`, `32768`, and `65536`
  - butterfly and butterfly+kv4 also match exactly within each prompt length
- KV quantization worsened TTFT and decode behavior at longer context in this pass:
  - `stock_kv4` is slower than `stock` on TTFT and decode at `32768` and `65536`
  - `butterfly_kv4` is slower than `butterfly` on TTFT and decode at `32768` and `65536`
- The exploratory `65536` row shows a large Butterfly e2e win versus stock (`70.9524s` vs `95.8140s`, `-25.95%`) while also showing a higher measured peak (`7472408586` vs `6877619276`, `+594789310 bytes`, about `+8.65%`).

Inference and interpretation

- Based on the current data, Butterfly-first remains the stronger practical lever than the current KV architecture for Qwen 3.5 4B on MLX.
- The long-context speed story is real enough to keep exploring: Butterfly is materially faster than stock at `32768` and `65536`.
- The long-context memory story is not resolved enough to over-claim.
- The `65536` peak difference is unlikely to be caused by the kv4 toggle itself because:
  - stock and stock+kv4 land on the exact same measured peak
  - butterfly and butterfly+kv4 also land on the exact same measured peak
- The most defensible interpretation today is that the extra `65536` peak is tied to Butterfly prefill working-set behavior, allocator/prealloc behavior, or both.
- The current harness records one MLX process-level peak counter for the whole measured prefill+decode run. It does not attribute bytes between:
  - prompt-cache preallocation
  - graph-cache persistence
  - transient Butterfly prefill buffers
  - transient decode buffers
- Because of that, the `65536` memory anomaly is narrowed but still ambiguous. It should stay documented as ambiguity rather than being presented as a proven persistent-cache penalty.

Near-term recommendation

- For MLX / Apple Silicon work, prioritize Butterfly prefill with stock decode.
- Treat KV quantization as an exploratory secondary knob until it shows a repeatable memory or latency benefit under this harness.
- Keep the repo story centered on measured prefill/e2e gains, not on unproven decode or memory claims.

Reproducible reporting

The helper below reads the external `results.json` artifacts directly and renders a compact markdown summary without copying benchmark artifacts into the repo:

```bash
python scripts/report_qwen_mlx_bench.py \
  --run stock=/tmp/butterfly_full_kv_eval/stock \
  --run stock_kv4=/tmp/butterfly_full_kv_eval/stock_kv4 \
  --run butterfly=/tmp/butterfly_full_kv_eval/butterfly \
  --run butterfly_kv4=/tmp/butterfly_full_kv_eval/butterfly_kv4 \
  --seq-lens 2048 8192 32768 \
  --format markdown
```

The report reader is intentionally tolerant of both the current Butterfly-first payload keys and older `wayfinder_*` mirrors so live summaries can read mixed-generation MLX result folders safely.

Exploratory 64k summary:

```bash
python scripts/report_qwen_mlx_bench.py \
  --run stock=/tmp/butterfly_full_kv_eval_explore/stock_65536 \
  --run stock_kv4=/tmp/butterfly_full_kv_eval_explore/stock_kv4_65536 \
  --run butterfly=/tmp/butterfly_full_kv_eval_explore/butterfly_65536 \
  --run butterfly_kv4=/tmp/butterfly_full_kv_eval_explore/butterfly_kv4_65536 \
  --seq-lens 65536 \
  --format markdown
```

Recommended next step after this report

Add stage-level memory attribution before making a stronger `65536+` memory claim. The lowest-risk follow-up is a rerun that records separate peaks for prefill and decode, plus a small graph-cache and prompt-cache summary, rather than changing model behavior.
