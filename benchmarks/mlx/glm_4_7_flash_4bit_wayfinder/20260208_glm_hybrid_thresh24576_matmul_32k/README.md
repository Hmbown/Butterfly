# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T04:23:16.321319+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 181.883106 | 180.159668 | n/a | 24232344896 | 28928288404 | -4695943508 | -16.233050 | 16.233050 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
