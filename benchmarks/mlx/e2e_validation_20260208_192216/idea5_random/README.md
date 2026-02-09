# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T19:25:46.209847+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy random --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea5_random`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1329258.8 | 0.0000 | 0.0000 | 3.0814 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 427523.9 | 189.0565 | 0.0083 | 0.1228 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2915260.0 | 173.9525 | 0.0024 | 1.1121 | 1.1117 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 736977.9 | 0.0000 | 0.0000 | 11.1157 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 272503.1 | 476.3882 | 0.0054 | 0.0962 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3829155.7 | 485.2979 | 0.0018 | 1.9084 | 1.9082 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1097577.8 | 3.7319 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 410483.4 | 9.9785 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2192694.8 | 1.8680 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 667948.6 | 12.2644 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 263888.5 | 31.0434 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2451868.8 | 3.3411 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
