# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:30:47.687894+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_readme_repro`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 714834.2 | 0.0000 | 0.0000 | 0.7163 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| wayfinder_sparse | 256 | 445588.6 | 42.4382 | 0.0016 | 0.0718 | 0.0000 | 27.0MB | 818.0KB | 26.2MB |
| wayfinder_permute | 256 | 304565.5 | 14.9105 | 0.0015 | 0.0901 | 0.0000 | 42.6MB | 818.0KB | 41.8MB |
| dense | 512 | 1660091.4 | 0.0000 | 0.0000 | 0.6168 | 0.0000 | 40.5MB | 0.0B | 40.5MB |
| wayfinder_sparse | 512 | 487807.9 | 31.5121 | 0.0017 | 0.0742 | 0.0000 | 65.3MB | 1.8MB | 63.6MB |
| wayfinder_permute | 512 | 410750.1 | 31.2265 | 0.0023 | 0.1124 | 0.0000 | 85.7MB | 1.8MB | 84.0MB |
| dense | 1024 | 1102816.9 | 0.0000 | 0.0000 | 1.8571 | 0.0000 | 144.1MB | 0.0B | 144.1MB |
| wayfinder_sparse | 1024 | 605917.2 | 79.5284 | 0.0021 | 0.0748 | 0.0000 | 164.3MB | 4.1MB | 160.2MB |
| wayfinder_permute | 1024 | 562112.9 | 69.3916 | 0.0041 | 0.1486 | 0.0000 | 165.5MB | 4.1MB | 161.4MB |
| dense | 2048 | 522088.3 | 0.0000 | 0.0000 | 7.8454 | 0.0000 | 541.1MB | 0.0B | 541.1MB |
| wayfinder_sparse | 2048 | 459344.1 | 172.5959 | 0.0040 | 0.0974 | 0.0000 | 466.1MB | 10.8MB | 455.4MB |
| wayfinder_permute | 2048 | 649961.0 | 168.2954 | 0.0034 | 0.1302 | 0.0000 | 326.9MB | 10.8MB | 316.1MB |
| dense | 4096 | 268514.2 | 0.0000 | 0.0000 | 30.5086 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 281390.6 | 476.2549 | 0.0054 | 0.1172 | 0.0000 | 1.5GB | 31.5MB | 1.4GB |
| wayfinder_permute | 4096 | 671359.6 | 491.7831 | 0.0046 | 0.1505 | 0.0000 | 689.4MB | 31.5MB | 657.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 794799.6 | 0.6442 | 13.3MB | 0.0B | 13.3MB |
| wayfinder_sparse | 256 | 366860.9 | 1.3956 | 31.2MB | 818.0KB | 30.4MB |
| wayfinder_permute | 256 | 275902.3 | 1.8557 | 46.8MB | 818.0KB | 46.0MB |
| dense | 512 | 933118.5 | 1.0974 | 46.0MB | 0.0B | 46.0MB |
| wayfinder_sparse | 512 | 400306.2 | 2.5580 | 72.9MB | 1.8MB | 71.2MB |
| wayfinder_permute | 512 | 517842.0 | 1.9774 | 90.5MB | 1.8MB | 88.7MB |
| dense | 1024 | 977730.6 | 2.0946 | 154.4MB | 0.0B | 154.4MB |
| wayfinder_sparse | 1024 | 561224.0 | 3.6492 | 174.5MB | 4.1MB | 170.4MB |
| wayfinder_permute | 1024 | 522632.3 | 3.9186 | 175.8MB | 4.1MB | 171.6MB |
| dense | 2048 | 510473.9 | 8.0239 | 560.8MB | 0.0B | 560.8MB |
| wayfinder_sparse | 2048 | 437579.6 | 9.3606 | 485.9MB | 10.8MB | 475.1MB |
| wayfinder_permute | 2048 | 611421.3 | 6.6991 | 346.7MB | 10.8MB | 335.9MB |
| dense | 4096 | 258477.4 | 31.6933 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 279036.5 | 29.3582 | 1.5GB | 31.5MB | 1.5GB |
| wayfinder_permute | 4096 | 646465.7 | 12.6720 | 728.2MB | 31.5MB | 696.6MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
