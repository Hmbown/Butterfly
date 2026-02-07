# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T04:59:38.585052+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1254902.0 | 0.0000 | 0.0000 | 0.4080 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| hha_sparse | 256 | 440020.0 | 59.9899 | 0.0022 | 0.0671 | 0.0000 | 27.0MB | 818.0KB | 26.2MB |
| hha_permute | 256 | 487648.1 | 15.8355 | 0.0016 | 0.0963 | 0.0000 | 41.6MB | 818.0KB | 40.8MB |
| dense | 512 | 2131481.6 | 0.0000 | 0.0000 | 0.4804 | 0.0000 | 39.4MB | 0.0B | 39.4MB |
| hha_sparse | 512 | 738184.3 | 46.0481 | 0.0021 | 0.0761 | 0.0000 | 65.3MB | 1.8MB | 63.6MB |
| hha_permute | 512 | 526134.3 | 32.1438 | 0.0039 | 0.1304 | 0.0000 | 84.7MB | 1.8MB | 82.9MB |
| dense | 1024 | 1173526.7 | 0.0000 | 0.0000 | 1.7452 | 0.0000 | 145.2MB | 0.0B | 145.2MB |
| hha_sparse | 1024 | 547523.5 | 73.0165 | 0.0040 | 0.0743 | 0.0000 | 166.2MB | 4.1MB | 162.1MB |
| hha_permute | 1024 | 546819.9 | 72.0512 | 0.0032 | 0.1029 | 0.0000 | 171.9MB | 4.1MB | 167.7MB |
| dense | 2048 | 480136.0 | 0.0000 | 0.0000 | 8.5309 | 0.0000 | 547.5MB | 0.0B | 547.5MB |
| hha_sparse | 2048 | 393128.7 | 195.9354 | 0.0052 | 0.1072 | 0.0000 | 476.0MB | 10.8MB | 465.2MB |
| hha_permute | 2048 | 570452.0 | 194.0505 | 0.0040 | 0.1292 | 0.0000 | 343.2MB | 10.8MB | 332.5MB |
| dense | 4096 | 241769.7 | 0.0000 | 0.0000 | 33.8835 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| hha_sparse | 4096 | 250698.8 | 566.8686 | 0.0048 | 0.0974 | 0.0000 | 1.5GB | 31.5MB | 1.5GB |
| hha_permute | 4096 | 592406.0 | 500.6850 | 0.0046 | 0.1371 | 0.0000 | 710.1MB | 31.5MB | 678.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 838341.7 | 0.6107 | 13.8MB | 0.0B | 13.8MB |
| hha_sparse | 256 | 382803.9 | 1.3375 | 31.7MB | 818.0KB | 30.9MB |
| hha_permute | 256 | 334049.2 | 1.5327 | 45.8MB | 818.0KB | 45.1MB |
| dense | 512 | 1193212.4 | 0.8582 | 45.0MB | 0.0B | 45.0MB |
| hha_sparse | 512 | 679486.4 | 1.5070 | 71.9MB | 1.8MB | 70.1MB |
| hha_permute | 512 | 536137.3 | 1.9100 | 91.4MB | 1.8MB | 89.6MB |
| dense | 1024 | 806504.2 | 2.5394 | 155.4MB | 0.0B | 155.4MB |
| hha_sparse | 1024 | 475806.5 | 4.3043 | 180.9MB | 4.1MB | 176.8MB |
| hha_permute | 1024 | 478593.2 | 4.2792 | 182.2MB | 4.1MB | 178.0MB |
| dense | 2048 | 468808.6 | 8.7370 | 567.3MB | 0.0B | 567.3MB |
| hha_sparse | 2048 | 394936.3 | 10.3713 | 502.2MB | 10.8MB | 491.5MB |
| hha_permute | 2048 | 545368.3 | 7.5105 | 369.5MB | 10.8MB | 358.7MB |
| dense | 4096 | 225689.2 | 36.2977 | 2.1GB | 0.0B | 2.1GB |
| hha_sparse | 4096 | 248117.7 | 33.0166 | 1.5GB | 31.5MB | 1.5GB |
| hha_permute | 4096 | 578365.1 | 14.1641 | 780.5MB | 31.5MB | 749.0MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
