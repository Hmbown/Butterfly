# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T18:42:13.530267+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 16 --num-cycles 1 --out-dir benchmarks/mlx/tiny_wayfinder/regularity_reg16`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1427851.3 | 0.0000 | 0.0000 | 2.8686 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 426761.1 | 191.8892 | 0.0062 | 0.0973 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2938483.0 | 179.7795 | 0.0031 | 1.1185 | 1.1181 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 729271.7 | 0.0000 | 0.0000 | 11.2331 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 267150.9 | 501.0112 | 0.0065 | 0.2542 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3660410.3 | 508.9352 | 0.0024 | 2.0226 | 2.0224 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1126983.9 | 3.6345 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 404347.7 | 10.1299 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2127654.1 | 1.9251 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 667903.2 | 12.2653 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 259811.9 | 31.5305 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2370213.3 | 3.4562 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
