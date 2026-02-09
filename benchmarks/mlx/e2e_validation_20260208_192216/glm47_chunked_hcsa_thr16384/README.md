# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T20:36:42.209656+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `47`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 217.592607 | 150.593352 | n/a | 22446619216 | 28928288404 | -6481669188 | -22.405989 | 22.405989 |
| normal | 4096 | 32768 | prefill_plus_1 | 236.288382 | 150.593352 | 0.053488 | 22446619216 | 29053543348 | -6606924132 | -22.740511 | 22.740511 |
| normal | 4096 | 65536 | prefill_only | 575.701718 | 113.836728 | n/a | 24155149252 | 44890891940 | -20735742688 | -46.191425 | 46.191425 |
| normal | 4096 | 65536 | prefill_plus_1 | 666.279439 | 113.836728 | 0.011040 | 24155149252 | 44890891940 | -20735742688 | -46.191425 | 46.191425 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
