# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T18:41:59.845830+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy random --num-cycles 1 --out-dir benchmarks/mlx/tiny_wayfinder/regularity_random`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1354786.5 | 0.0000 | 0.0000 | 3.0234 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 425225.3 | 192.6400 | 0.0048 | 0.0881 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2773893.2 | 177.7053 | 0.0025 | 1.2494 | 1.2488 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 703206.2 | 0.0000 | 0.0000 | 11.6495 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 266553.9 | 500.0661 | 0.0081 | 0.1195 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3251707.9 | 506.8016 | 0.0022 | 2.2836 | 2.2830 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1139895.3 | 3.5933 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 402178.1 | 10.1845 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 1862541.2 | 2.1991 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 657954.5 | 12.4507 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 255815.0 | 32.0231 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2458906.3 | 3.3316 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
