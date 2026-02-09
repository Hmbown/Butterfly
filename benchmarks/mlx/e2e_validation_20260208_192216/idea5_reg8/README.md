# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T19:25:52.734348+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 8 --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea5_reg8`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1360995.2 | 0.0000 | 0.0000 | 3.0096 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 438880.6 | 191.0013 | 0.0039 | 0.0831 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2931866.0 | 173.4620 | 0.0017 | 1.1875 | 1.1872 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 720768.5 | 0.0000 | 0.0000 | 11.3656 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 271430.4 | 490.3011 | 0.0062 | 0.0978 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3637419.6 | 493.5466 | 0.0018 | 2.0130 | 2.0126 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1116881.5 | 3.6674 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 415523.8 | 9.8574 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2179061.3 | 1.8797 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 661835.0 | 12.3777 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 264979.5 | 30.9156 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2398185.1 | 3.4159 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
