# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T23:11:50.647542+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/k4_active_row_20260208_8192_dense/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 8192 | prefill_only | 68.579336 | 119.452891 | n/a | 21441617192 | 20660795044 | 780822148 | 3.779245 | -3.779245 |
| normal | 4096 | 8192 | prefill_plus_1 | 68.964741 | 119.452891 | 2.594675 | 21441617192 | 20660795044 | 780822148 | 3.779245 | -3.779245 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
