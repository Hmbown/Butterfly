# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T08:57:04.496181+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `0`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 65536 | prefill_only | 679.145047 | 96.497796 | n/a | 33161104180 | 44890891940 | -11729787760 | -26.129549 | 26.129549 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
