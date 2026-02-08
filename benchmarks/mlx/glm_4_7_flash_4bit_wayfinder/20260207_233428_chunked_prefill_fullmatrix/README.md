# GLM Chunked Prefill Benchmark

- created_at: `2026-02-07T23:47:51.969533+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 167.741951 | 195.347674 | n/a | 26021806692 | 28928288404 | -2906481712 | -10.047196 | 10.047196 |
| normal | 4096 | 32768 | prefill_plus_1 | 171.836471 | 195.347674 | 0.244229 | 26021806692 | 29053543348 | -3031736656 | -10.434998 | 10.434998 |
| normal | 4096 | 32768 | prefill_plus_64 | 172.392326 | 195.347674 | 13.762331 | 26021806692 | n/a | n/a | n/a | n/a |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
