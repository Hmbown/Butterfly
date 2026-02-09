# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T18:42:06.351962+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 8 --num-cycles 1 --out-dir benchmarks/mlx/tiny_wayfinder/regularity_reg8`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1303567.5 | 0.0000 | 0.0000 | 3.1421 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 423792.6 | 192.1599 | 0.0063 | 0.1116 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 3138146.7 | 179.7907 | 0.0035 | 1.0589 | 1.0586 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 720026.8 | 0.0000 | 0.0000 | 11.3774 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 267005.5 | 501.1530 | 0.0062 | 0.1047 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3470573.9 | 510.7731 | 0.0033 | 2.1313 | 2.1310 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1203577.5 | 3.4032 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 407920.7 | 10.0412 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2310316.8 | 1.7729 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 667320.6 | 12.2760 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 257869.0 | 31.7681 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2388962.2 | 3.4291 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
