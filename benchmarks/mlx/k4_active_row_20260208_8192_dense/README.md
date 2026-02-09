# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T23:09:19.540027+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `0`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 8192 | prefill_only | 108.406942 | 75.567117 | n/a | 20660795044 | n/a | n/a | n/a | n/a |
| normal | 4096 | 8192 | prefill_plus_1 | 108.935305 | 75.567117 | 1.892636 | 20660795044 | n/a | n/a | n/a | n/a |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
