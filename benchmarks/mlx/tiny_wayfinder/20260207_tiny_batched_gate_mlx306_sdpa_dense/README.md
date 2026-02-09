# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:58:08.291487+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306_sdpa_dense`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1586672.0 | 0.0000 | 0.0000 | 0.3227 | 0.0000 | 4.4MB | 0.0B | 4.4MB |
| wayfinder_sparse | 256 | 427111.4 | 19.8348 | 0.0020 | 0.0762 | 0.0000 | 27.0MB | 826.0KB | 26.2MB |
| wayfinder_permute | 256 | 1016377.2 | 17.5533 | 0.0010 | 0.3519 | 0.3517 | 13.3MB | 826.0KB | 12.5MB |
| dense | 512 | 3168645.3 | 0.0000 | 0.0000 | 0.3232 | 0.0000 | 13.8MB | 0.0B | 13.8MB |
| wayfinder_sparse | 512 | 713878.9 | 34.4550 | 0.0016 | 0.0701 | 0.0000 | 62.1MB | 1.8MB | 60.4MB |
| wayfinder_permute | 512 | 1341594.3 | 35.0487 | 0.0023 | 0.5702 | 0.5699 | 28.9MB | 1.8MB | 27.1MB |
| dense | 1024 | 3030518.3 | 0.0000 | 0.0000 | 0.6758 | 0.0000 | 43.7MB | 0.0B | 43.7MB |
| wayfinder_sparse | 1024 | 547721.7 | 91.2662 | 0.0042 | 0.0947 | 0.0000 | 159.1MB | 4.2MB | 155.0MB |
| wayfinder_permute | 1024 | 1930217.2 | 70.0490 | 0.0025 | 0.8719 | 0.8715 | 60.0MB | 4.2MB | 55.8MB |
| dense | 2048 | 1323736.7 | 0.0000 | 0.0000 | 3.0943 | 0.0000 | 153.8MB | 0.0B | 153.8MB |
| wayfinder_sparse | 2048 | 427090.2 | 171.1038 | 0.0056 | 0.1091 | 0.0000 | 461.0MB | 10.8MB | 450.1MB |
| wayfinder_permute | 2048 | 2081234.7 | 172.2754 | 0.0032 | 1.6915 | 1.6910 | 123.8MB | 10.8MB | 113.0MB |
| dense | 4096 | 710205.6 | 0.0000 | 0.0000 | 11.5347 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 272796.9 | 486.8653 | 0.0053 | 0.0971 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2287003.6 | 499.4473 | 0.0030 | 3.2910 | 3.2903 | 248.5MB | 31.7MB | 216.9MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1035913.0 | 0.4943 | 8.1MB | 0.0B | 8.1MB |
| wayfinder_sparse | 256 | 585031.2 | 0.8752 | 30.6MB | 826.0KB | 29.8MB |
| wayfinder_permute | 256 | 702652.9 | 0.7287 | 15.0MB | 826.0KB | 14.2MB |
| dense | 512 | 2003259.2 | 0.5112 | 19.3MB | 0.0B | 19.3MB |
| wayfinder_sparse | 512 | 636815.9 | 1.6080 | 67.6MB | 1.8MB | 65.9MB |
| wayfinder_permute | 512 | 902202.6 | 1.1350 | 31.4MB | 1.8MB | 29.6MB |
| dense | 1024 | 1742608.0 | 1.1752 | 54.0MB | 0.0B | 54.0MB |
| wayfinder_sparse | 1024 | 518000.3 | 3.9537 | 169.4MB | 4.2MB | 165.2MB |
| wayfinder_permute | 1024 | 1482939.3 | 1.3810 | 64.3MB | 4.2MB | 60.1MB |
| dense | 2048 | 1099966.4 | 3.7238 | 173.5MB | 0.0B | 173.5MB |
| wayfinder_sparse | 2048 | 401109.9 | 10.2117 | 480.8MB | 10.8MB | 469.9MB |
| wayfinder_permute | 2048 | 1740510.2 | 2.3533 | 131.6MB | 10.8MB | 120.8MB |
| dense | 4096 | 662332.1 | 12.3684 | 612.1MB | 0.0B | 612.1MB |
| wayfinder_sparse | 4096 | 264900.7 | 30.9248 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 1795326.6 | 4.5630 | 274.5MB | 31.7MB | 242.8MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
