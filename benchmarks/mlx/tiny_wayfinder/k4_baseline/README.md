# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T23:32:23.918039+00:00`
- command: `/Volumes/VIXinSSD/wayfinder/scripts/bench_mlx_wayfinder_scale.py --seq-lens 4096 8192 16384 --batch 1 --heads 8 --embd 256 --window 32 --warmup 2 --iters 4 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/tiny_wayfinder/k4_baseline`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 4096 | 384024.0 | 0.0000 | 0.0000 | 10.6660 | 0.0000 | 563.0MB | 0.0B | 563.0MB |
| wayfinder_sparse | 4096 | 250886.5 | 828.4597 | 0.0062 | 0.1717 | 0.0000 | 971.6MB | 43.6MB | 927.9MB |
| wayfinder_permute | 4096 | 1334890.5 | 727.9476 | 0.0032 | 2.7377 | 2.7372 | 156.5MB | 43.6MB | 112.8MB |
| dense | 8192 | 194653.1 | 0.0000 | 0.0000 | 42.0851 | 0.0000 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 8192 | 164359.7 | 1969.1555 | 0.0081 | 0.2220 | 0.0000 | 2.5GB | 127.2MB | 2.4GB |
| wayfinder_permute | 8192 | 1485936.7 | 1958.6481 | 0.0043 | 5.0721 | 5.0716 | 351.8MB | 127.2MB | 224.6MB |
| dense | 16384 | 92872.9 | 0.0000 | 0.0000 | 176.4131 | 0.0000 | 8.5GB | 0.0B | 8.5GB |
| wayfinder_sparse | 16384 | 98021.8 | 6193.9805 | 0.0060 | 0.1627 | 0.0000 | 5.7GB | 414.5MB | 5.3GB |
| wayfinder_permute | 16384 | 2027074.8 | 6154.7586 | 0.0059 | 7.4926 | 7.4921 | 747.3MB | 414.5MB | 332.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 4096 | 363475.7 | 11.2690 | 603.6MB | 0.0B | 603.6MB |
| wayfinder_sparse | 4096 | 247810.9 | 16.5287 | 984.1MB | 43.6MB | 940.5MB |
| wayfinder_permute | 4096 | 1324717.7 | 3.0920 | 173.0MB | 43.6MB | 129.4MB |
| dense | 8192 | 179129.4 | 45.7323 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 8192 | 159691.5 | 51.2989 | 2.5GB | 127.2MB | 2.4GB |
| wayfinder_permute | 8192 | 1229391.5 | 6.6635 | 382.4MB | 127.2MB | 255.1MB |
| dense | 16384 | 92429.6 | 177.2592 | 8.7GB | 0.0B | 8.7GB |
| wayfinder_sparse | 16384 | 94173.8 | 173.9762 | 5.8GB | 414.5MB | 5.4GB |
| wayfinder_permute | 16384 | 1454432.5 | 11.2649 | 815.7MB | 414.5MB | 401.2MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
