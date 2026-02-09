# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:54:40.798320+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1220318.8 | 0.0000 | 0.0000 | 0.4196 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| wayfinder_sparse | 256 | 369519.5 | 20.2766 | 0.0023 | 0.0809 | 0.0000 | 27.0MB | 826.0KB | 26.2MB |
| wayfinder_permute | 256 | 932922.5 | 16.3323 | 0.0015 | 0.3571 | 0.3569 | 13.3MB | 826.0KB | 12.5MB |
| dense | 512 | 1914018.7 | 0.0000 | 0.0000 | 0.5350 | 0.0000 | 37.3MB | 0.0B | 37.3MB |
| wayfinder_sparse | 512 | 651580.8 | 32.6682 | 0.0013 | 0.0690 | 0.0000 | 62.1MB | 1.8MB | 60.4MB |
| wayfinder_permute | 512 | 1394818.9 | 32.2003 | 0.0011 | 0.5592 | 0.5590 | 28.9MB | 1.8MB | 27.1MB |
| dense | 1024 | 1123294.1 | 0.0000 | 0.0000 | 1.8232 | 0.0000 | 138.8MB | 0.0B | 138.8MB |
| wayfinder_sparse | 1024 | 560619.1 | 78.1867 | 0.0037 | 0.0740 | 0.0000 | 159.1MB | 4.2MB | 155.0MB |
| wayfinder_permute | 1024 | 2001995.2 | 72.2146 | 0.0016 | 0.8471 | 0.8468 | 60.0MB | 4.2MB | 55.8MB |
| dense | 2048 | 532143.8 | 0.0000 | 0.0000 | 7.6972 | 0.0000 | 535.9MB | 0.0B | 535.9MB |
| wayfinder_sparse | 2048 | 432580.9 | 176.0932 | 0.0044 | 0.0850 | 0.0000 | 461.0MB | 10.8MB | 450.1MB |
| wayfinder_permute | 2048 | 2278641.9 | 168.8298 | 0.0017 | 1.5792 | 1.5790 | 123.8MB | 10.8MB | 113.0MB |
| dense | 4096 | 261828.2 | 0.0000 | 0.0000 | 31.2877 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 270228.7 | 480.2481 | 0.0059 | 0.0988 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2420297.4 | 485.5935 | 0.0026 | 3.1685 | 3.1681 | 248.5MB | 31.7MB | 216.9MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 759126.9 | 0.6745 | 13.8MB | 0.0B | 13.8MB |
| wayfinder_sparse | 256 | 403666.1 | 1.2684 | 30.6MB | 826.0KB | 29.8MB |
| wayfinder_permute | 256 | 812053.9 | 0.6305 | 15.0MB | 826.0KB | 14.2MB |
| dense | 512 | 1322426.8 | 0.7743 | 42.9MB | 0.0B | 42.9MB |
| wayfinder_sparse | 512 | 606739.7 | 1.6877 | 67.6MB | 1.8MB | 65.8MB |
| wayfinder_permute | 512 | 1042681.7 | 0.9821 | 31.4MB | 1.8MB | 29.6MB |
| dense | 1024 | 1000183.1 | 2.0476 | 149.1MB | 0.0B | 149.1MB |
| wayfinder_sparse | 1024 | 547361.9 | 3.7416 | 169.4MB | 4.2MB | 165.2MB |
| wayfinder_permute | 1024 | 1618598.7 | 1.2653 | 64.3MB | 4.2MB | 60.1MB |
| dense | 2048 | 481599.1 | 8.5050 | 555.7MB | 0.0B | 555.7MB |
| wayfinder_sparse | 2048 | 420321.7 | 9.7449 | 480.8MB | 10.8MB | 469.9MB |
| wayfinder_permute | 2048 | 1712702.5 | 2.3915 | 131.6MB | 10.8MB | 120.8MB |
| dense | 4096 | 257045.6 | 31.8698 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 262863.9 | 31.1644 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 1935842.1 | 4.2317 | 269.9MB | 31.7MB | 238.2MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
