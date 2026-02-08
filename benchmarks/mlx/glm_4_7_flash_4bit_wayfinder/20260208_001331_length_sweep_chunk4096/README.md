# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T00:31:50.103455+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 163.331554 | 200.622594 | n/a | 26021806692 | 28928288404 | -2906481712 | -10.047196 | 10.047196 |
| normal | 4096 | 32768 | prefill_plus_1 | 164.967735 | 200.622594 | 0.611179 | 26021806692 | 29053543348 | -3031736656 | -10.434998 | 10.434998 |
| normal | 4096 | 32768 | prefill_plus_64 | 165.530018 | 200.622594 | 29.111235 | 26021806692 | n/a | n/a | n/a | n/a |
| normal | 4096 | 65536 | prefill_only | 876.314248 | 74.785957 | n/a | 33164840500 | 44890891940 | -11726051440 | -26.121226 | 26.121226 |
| normal | 4096 | 65536 | prefill_plus_1 | 903.317596 | 74.785957 | 0.037032 | 33164840500 | 44890891940 | -11726051440 | -26.121226 | 26.121226 |
| normal | 4096 | 65536 | prefill_plus_64 | 904.306472 | 74.785957 | 2.286349 | 33164840500 | n/a | n/a | n/a | n/a |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
