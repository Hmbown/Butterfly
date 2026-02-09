# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T20:50:15.882982+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --strategy random --num-cycles 1 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/tiebreak_idea5_random`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1469680.4 | 0.0000 | 0.0000 | 2.7870 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 471978.9 | 187.4963 | 0.0023 | 0.0736 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 2382462.6 | 172.4793 | 0.0011 | 1.5091 | 1.5090 | 64.1MB | 10.8MB | 53.3MB |
| dense | 4096 | 810166.6 | 0.0000 | 0.0000 | 10.1115 | 0.0000 | 573.3MB | 0.0B | 573.3MB |
| wayfinder_sparse | 4096 | 295833.5 | 483.4295 | 0.0026 | 0.0755 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2652778.0 | 490.0389 | 0.0015 | 2.7990 | 2.7985 | 138.3MB | 31.7MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1272955.8 | 3.2177 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 453085.9 | 9.0402 | 480.8MB | 10.8MB | 470.0MB |
| wayfinder_permute | 2048 | 1497041.5 | 2.7361 | 71.9MB | 10.8MB | 61.1MB |
| dense | 4096 | 748691.9 | 10.9417 | 612.0MB | 0.0B | 612.0MB |
| wayfinder_sparse | 4096 | 287770.3 | 28.4671 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 2645958.9 | 3.0960 | 153.1MB | 31.7MB | 121.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
