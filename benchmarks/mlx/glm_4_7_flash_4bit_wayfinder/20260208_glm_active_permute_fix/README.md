# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T03:21:14.202985+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 447.824442 | 73.171531 | n/a | 22348113740 | 28928288404 | -6580174664 | -22.746505 | 22.746505 |
| normal | 4096 | 32768 | prefill_plus_1 | 462.633991 | 73.171531 | 0.067524 | 22348113740 | 29053543348 | -6705429608 | -23.079559 | 23.079559 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
