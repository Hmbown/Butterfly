# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T19:25:59.273296+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 16 --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea5_reg16`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1349551.0 | 0.0000 | 0.0000 | 3.0351 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 433582.6 | 190.3509 | 0.0057 | 0.0878 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 3210397.4 | 172.4186 | 0.0028 | 1.0894 | 1.0891 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 692661.0 | 0.0000 | 0.0000 | 11.8269 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 269471.1 | 541.1028 | 0.0062 | 0.1153 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3459794.0 | 565.8653 | 0.0059 | 2.0663 | 2.0657 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1115753.1 | 3.6711 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 405682.6 | 10.0966 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2219953.7 | 1.8451 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 634131.4 | 12.9185 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 261592.7 | 31.3159 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2385353.7 | 3.4343 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
