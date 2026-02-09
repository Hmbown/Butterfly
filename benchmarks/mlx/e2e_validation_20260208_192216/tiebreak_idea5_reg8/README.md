# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T20:50:18.305564+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 8 --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/tiebreak_idea5_reg8`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1329708.2 | 0.0000 | 0.0000 | 3.0804 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 473697.1 | 188.7247 | 0.0023 | 0.0695 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2398857.5 | 173.9355 | 0.0014 | 1.4842 | 1.4840 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 805962.1 | 0.0000 | 0.0000 | 10.1642 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 296155.9 | 485.6767 | 0.0019 | 0.0625 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3190626.3 | 489.9061 | 0.0015 | 2.3206 | 2.3204 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1275292.4 | 3.2118 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 453432.8 | 9.0333 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 1499576.7 | 2.7314 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 749117.0 | 10.9355 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 287990.4 | 28.4454 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2581106.2 | 3.1738 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
