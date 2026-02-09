# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T19:24:38.400242+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea1_disjoint_d1`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1165733.9 | 0.0000 | 0.0000 | 3.5137 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 422444.1 | 188.6958 | 0.0071 | 0.1084 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 3361625.9 | 173.0160 | 0.0027 | 1.0256 | 1.0253 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 682438.0 | 0.0000 | 0.0000 | 12.0040 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 259827.0 | 490.9048 | 0.0071 | 0.1137 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3449595.8 | 498.8107 | 0.0039 | 2.1065 | 2.1061 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1109056.6 | 3.6932 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 390446.2 | 10.4906 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2160836.2 | 1.8956 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 635202.9 | 12.8967 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 258042.5 | 31.7467 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2504688.3 | 3.2707 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
