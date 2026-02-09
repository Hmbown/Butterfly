# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T18:30:09.324913+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --num-cycles 2 --out-dir benchmarks/mlx/tiny_wayfinder/disjoint_d2`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1369766.1 | 0.0000 | 0.0000 | 2.9903 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 425014.8 | 214.1988 | 0.0047 | 0.0910 | 0.0000 | 470.1MB | 11.0MB | 459.0MB |
| wayfinder_permute | 2048 | 1785284.3 | 200.7652 | 0.0018 | 2.0653 | 2.0651 | 64.4MB | 11.0MB | 53.3MB |
| dense | 4096 | 732746.6 | 0.0000 | 0.0000 | 11.1799 | 0.0000 | 573.5MB | 0.0B | 573.5MB |
| wayfinder_sparse | 4096 | 264492.7 | 528.1739 | 0.0056 | 0.0974 | 0.0000 | 1.5GB | 32.1MB | 1.4GB |
| wayfinder_permute | 4096 | 1909139.9 | 541.0134 | 0.0023 | 4.0113 | 4.0109 | 138.7MB | 32.1MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1116900.6 | 3.6673 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 395682.7 | 10.3517 | 489.8MB | 11.0MB | 478.8MB |
| wayfinder_permute | 2048 | 1462204.4 | 2.8012 | 74.1MB | 11.0MB | 63.1MB |
| dense | 4096 | 664070.4 | 12.3360 | 612.3MB | 0.0B | 612.3MB |
| wayfinder_sparse | 4096 | 260724.1 | 31.4202 | 1.5GB | 32.1MB | 1.5GB |
| wayfinder_permute | 4096 | 1512700.8 | 5.4155 | 157.5MB | 32.1MB | 125.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
