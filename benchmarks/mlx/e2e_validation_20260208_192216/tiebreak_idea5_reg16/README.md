# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T20:50:20.728210+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy regular_partition --regular-num-clusters 16 --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/tiebreak_idea5_reg16`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1405748.6 | 0.0000 | 0.0000 | 2.9137 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 474717.4 | 186.6114 | 0.0022 | 0.0644 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2362991.5 | 173.4317 | 0.0014 | 1.5123 | 1.5121 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 808020.7 | 0.0000 | 0.0000 | 10.1384 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 296035.7 | 485.7236 | 0.0026 | 0.0694 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2614503.0 | 490.9065 | 0.0012 | 2.8359 | 2.8355 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1270323.5 | 3.2244 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 453352.3 | 9.0349 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 1461769.7 | 2.8021 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 754492.8 | 10.8576 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 287842.8 | 28.4600 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2203396.8 | 3.7179 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
