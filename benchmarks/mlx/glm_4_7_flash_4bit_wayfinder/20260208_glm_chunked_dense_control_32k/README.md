# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T03:33:56.953132+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `0`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 162.553654 | 201.582672 | n/a | 26018070388 | 28928288404 | -2910218016 | -10.060111 | 10.060111 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
