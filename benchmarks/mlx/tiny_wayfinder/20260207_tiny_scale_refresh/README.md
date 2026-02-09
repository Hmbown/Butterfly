# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:26:12.403633+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 1024 2048 4096 --batch 1 --heads 8 --embd 512 --window 64 --landmark-stride 64 --warmup 1 --iters 1 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_scale_refresh`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1024 | 391849.2 | 0.0000 | 0.0000 | 2.6132 | 0.0000 | 148.1MB | 0.0B | 148.1MB |
| wayfinder_sparse | 1024 | 151040.8 | 179.1789 | 0.0069 | 0.1470 | 0.0000 | 386.0MB | 12.5MB | 373.5MB |
| wayfinder_permute | 1024 | 99904.5 | 168.6229 | 0.0035 | 0.1780 | 0.0000 | 615.6MB | 12.5MB | 603.1MB |
| dense | 2048 | 265170.5 | 0.0000 | 0.0000 | 7.7233 | 0.0000 | 601.5MB | 0.0B | 601.5MB |
| wayfinder_sparse | 2048 | 136090.3 | 364.0658 | 0.0032 | 0.1375 | 0.0000 | 932.8MB | 27.5MB | 905.3MB |
| wayfinder_permute | 2048 | 103441.3 | 366.1597 | 0.0041 | 0.1872 | 0.0000 | 1.2GB | 27.5MB | 1.2GB |
| dense | 4096 | 138294.1 | 0.0000 | 0.0000 | 29.6180 | 0.0000 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 4096 | 105749.8 | 867.1574 | 0.0038 | 0.1540 | 0.0000 | 2.0GB | 65.1MB | 2.0GB |
| wayfinder_permute | 4096 | 105264.6 | 854.5832 | 0.0038 | 0.1903 | 0.0000 | 2.0GB | 65.1MB | 2.0GB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1024 | 256430.1 | 3.9933 | 175.1MB | 0.0B | 175.1MB |
| wayfinder_sparse | 1024 | 138245.7 | 7.4071 | 425.9MB | 12.5MB | 413.4MB |
| wayfinder_permute | 1024 | 93882.3 | 10.9073 | 653.5MB | 12.5MB | 641.0MB |
| dense | 2048 | 231885.2 | 8.8320 | 646.5MB | 0.0B | 646.5MB |
| wayfinder_sparse | 2048 | 126916.1 | 16.1366 | 951.9MB | 27.5MB | 924.3MB |
| wayfinder_permute | 2048 | 98123.8 | 20.8716 | 1.2GB | 27.5MB | 1.2GB |
| dense | 4096 | 127636.9 | 32.0910 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 4096 | 90272.6 | 45.3737 | 2.1GB | 65.1MB | 2.0GB |
| wayfinder_permute | 4096 | 100253.4 | 40.8565 | 2.1GB | 65.1MB | 2.0GB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
