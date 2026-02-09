# GLM Chunked Prefill Benchmark

- created_at: `2026-02-08T20:20:48.470516+00:00`
- model_path: `mlx-community/GLM-4.7-Flash-4bit`
- baseline_path: `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- replaced_layers: `0`

| cache_mode | chunk | seq_len | scenario | latency_sec | prefill_tok_s | decode_tok_s | peak_memory_bytes | baseline_peak | delta_peak | delta_peak_pct | reduction_pct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 4096 | 32768 | prefill_only | 680.855877 | 48.127660 | n/a | 26018037620 | 28928288404 | -2910250784 | -10.060225 | 10.060225 |
| normal | 4096 | 32768 | prefill_plus_1 | 696.708412 | 48.127660 | 0.063081 | 26018037620 | 29053543348 | -3035505728 | -10.447971 | 10.447971 |
| normal | 4096 | 65536 | prefill_only | 858.419846 | 76.344926 | n/a | 33161071420 | 44890891940 | -11729820520 | -26.129622 | 26.129622 |
| normal | 4096 | 65536 | prefill_plus_1 | 878.244155 | 76.344926 | 0.050443 | 33161071420 | 44890891940 | -11729820520 | -26.129622 | 26.129622 |

Memory reduction uses: `100 * (1 - candidate / baseline)`.
