# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T20:49:22.587197+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 174.282331 | 188.016765 | n/a | 26018070576 | 28928288404 | -2910217828 | -10.060111 | 10.060111 |
| normal | 4096 | 32768 | prefill_plus_1 | 180.742260 | 188.016765 | 0.154800 | 26018070576 | 29053543348 | -3035472772 | -10.447857 | 10.447857 |
| normal | 4096 | 65536 | prefill_only | 511.355989 | 128.161205 | n/a | 29589653016 | 44890891940 | -15301238924 | -34.085397 | 34.085397 |
| normal | 4096 | 65536 | prefill_plus_1 | 541.881939 | 128.161205 | 0.032759 | 29589653016 | 44890891940 | -15301238924 | -34.085397 | 34.085397 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
