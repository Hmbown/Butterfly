# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:37:58.682835+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_readme_repro_topology`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1283745.3 | 0.0000 | 0.0000 | 0.3988 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| wayfinder_sparse | 256 | 408347.7 | 19.5706 | 0.0023 | 0.0837 | 0.0000 | 27.0MB | 818.0KB | 26.2MB |
| wayfinder_permute | 256 | 470227.8 | 15.6198 | 0.0022 | 0.0945 | 0.0000 | 42.6MB | 818.0KB | 41.8MB |
| dense | 512 | 1912678.0 | 0.0000 | 0.0000 | 0.5354 | 0.0000 | 40.5MB | 0.0B | 40.5MB |
| wayfinder_sparse | 512 | 628992.6 | 76.3425 | 0.0027 | 0.0811 | 0.0000 | 65.3MB | 1.8MB | 63.6MB |
| wayfinder_permute | 512 | 350644.9 | 31.4001 | 0.0039 | 0.1108 | 0.0000 | 85.7MB | 1.8MB | 84.0MB |
| dense | 1024 | 958894.9 | 0.0000 | 0.0000 | 2.1358 | 0.0000 | 144.1MB | 0.0B | 144.1MB |
| wayfinder_sparse | 1024 | 518136.7 | 93.3129 | 0.0043 | 0.0850 | 0.0000 | 164.3MB | 4.1MB | 160.2MB |
| wayfinder_permute | 1024 | 566851.8 | 69.9695 | 0.0051 | 0.1384 | 0.0000 | 165.5MB | 4.1MB | 161.4MB |
| dense | 2048 | 493769.1 | 0.0000 | 0.0000 | 8.2954 | 0.0000 | 541.1MB | 0.0B | 541.1MB |
| wayfinder_sparse | 2048 | 362809.7 | 179.6719 | 0.0055 | 0.0980 | 0.0000 | 466.1MB | 10.8MB | 455.4MB |
| wayfinder_permute | 2048 | 483854.5 | 175.5147 | 0.0054 | 0.1453 | 0.0000 | 326.9MB | 10.8MB | 316.1MB |
| dense | 4096 | 229872.2 | 0.0000 | 0.0000 | 35.6372 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 213582.4 | 515.9404 | 0.0070 | 0.1320 | 0.0000 | 1.5GB | 31.5MB | 1.4GB |
| wayfinder_permute | 4096 | 503344.2 | 508.6738 | 0.0054 | 0.1514 | 0.0000 | 689.4MB | 31.5MB | 657.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 787792.9 | 0.6499 | 13.3MB | 0.0B | 13.3MB |
| wayfinder_sparse | 256 | 547789.0 | 0.9347 | 31.2MB | 818.0KB | 30.4MB |
| wayfinder_permute | 256 | 404796.4 | 1.2648 | 46.4MB | 818.0KB | 45.6MB |
| dense | 512 | 1526174.9 | 0.6710 | 46.0MB | 0.0B | 46.0MB |
| wayfinder_sparse | 512 | 488909.3 | 2.0945 | 72.9MB | 1.8MB | 71.2MB |
| wayfinder_permute | 512 | 389748.8 | 2.6273 | 91.2MB | 1.8MB | 89.5MB |
| dense | 1024 | 845742.3 | 2.4215 | 154.4MB | 0.0B | 154.4MB |
| wayfinder_sparse | 1024 | 536269.1 | 3.8190 | 174.5MB | 4.1MB | 170.4MB |
| wayfinder_permute | 1024 | 505897.6 | 4.0483 | 175.8MB | 4.1MB | 171.6MB |
| dense | 2048 | 481597.9 | 8.5050 | 560.8MB | 0.0B | 560.8MB |
| wayfinder_sparse | 2048 | 329573.4 | 12.4282 | 485.9MB | 10.8MB | 475.1MB |
| wayfinder_permute | 2048 | 472428.0 | 8.6701 | 346.7MB | 10.8MB | 335.9MB |
| dense | 4096 | 225063.3 | 36.3986 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 206247.2 | 39.7193 | 1.5GB | 31.5MB | 1.5GB |
| wayfinder_permute | 4096 | 484422.0 | 16.9109 | 728.2MB | 31.5MB | 696.6MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
