# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T00:13:01.837083+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 163.511077 | 200.402326 | n/a | 26021806692 | 28928288404 | -2906481712 | -10.047196 | 10.047196 |
| normal | 4096 | 32768 | prefill_plus_1 | 165.160118 | 200.402326 | 0.606413 | 26021806692 | 29053543348 | -3031736656 | -10.434998 | 10.434998 |
| normal | 4096 | 32768 | prefill_plus_64 | 165.713457 | 200.402326 | 29.059468 | 26021806692 | n/a | n/a | n/a | n/a |
| normal | 8192 | 32768 | prefill_only | 231.863675 | 141.324423 | n/a | 30798216294 | 28928288404 | 1869927890 | 6.464012 | -6.464012 |
| normal | 8192 | 32768 | prefill_plus_1 | 244.497756 | 141.324423 | 0.079151 | 30798216294 | 29053543348 | 1744672946 | 6.005026 | -6.005026 |
| normal | 8192 | 32768 | prefill_plus_64 | 245.061457 | 141.324423 | 4.849299 | 30798216294 | n/a | n/a | n/a | n/a |
| normal | 16384 | 32768 | prefill_only | 212.844382 | 153.952853 | n/a | 41954965882 | 28928288404 | 13026677478 | 45.030931 | -45.030931 |
| normal | 16384 | 32768 | prefill_plus_1 | 256.894309 | 153.952853 | 0.022702 | 41954965882 | 29053543348 | 12901422534 | 44.405677 | -44.405677 |
| normal | 16384 | 32768 | prefill_plus_64 | 257.448221 | 153.952853 | 1.434854 | 41954965882 | n/a | n/a | n/a | n/a |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
