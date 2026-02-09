# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T18:30:00.781940+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --num-cycles 1 --out-dir benchmarks/mlx/tiny_wayfinder/disjoint_d1`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1180900.0 | 0.0000 | 0.0000 | 3.4685 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 428646.2 | 187.0320 | 0.0055 | 0.0922 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 3189825.0 | 173.1944 | 0.0017 | 1.0730 | 1.0727 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 715881.9 | 0.0000 | 0.0000 | 11.4432 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 267802.7 | 489.3495 | 0.0059 | 0.1069 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 3312325.4 | 497.1883 | 0.0017 | 2.2282 | 2.2276 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1054701.8 | 3.8836 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 413225.2 | 9.9123 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 2210047.3 | 1.8534 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 657392.4 | 12.4614 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 258148.2 | 31.7337 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2225154.6 | 3.6815 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
