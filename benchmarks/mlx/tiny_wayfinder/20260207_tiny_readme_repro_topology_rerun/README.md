# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:38:29.867096+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_readme_repro_topology_rerun`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 467223.2 | 0.0000 | 0.0000 | 8.7667 | 0.0000 | 531.4MB | 0.0B | 531.4MB |
| wayfinder_sparse | 2048 | 353606.4 | 202.5305 | 0.0051 | 0.1221 | 0.0000 | 461.0MB | 10.8MB | 450.2MB |
| wayfinder_permute | 2048 | 503579.5 | 179.5763 | 0.0054 | 0.1534 | 0.0000 | 343.3MB | 10.8MB | 332.5MB |
| dense | 4096 | 235079.5 | 0.0000 | 0.0000 | 34.8478 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 223324.3 | 552.4132 | 0.0057 | 0.1263 | 0.0000 | 1.5GB | 31.5MB | 1.4GB |
| wayfinder_permute | 4096 | 516239.5 | 516.1745 | 0.0096 | 0.1693 | 0.0000 | 716.9MB | 31.5MB | 685.4MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 440779.7 | 9.2926 | 551.2MB | 0.0B | 551.2MB |
| wayfinder_sparse | 2048 | 336917.1 | 12.1573 | 491.7MB | 10.8MB | 480.9MB |
| wayfinder_permute | 2048 | 465628.7 | 8.7967 | 374.0MB | 10.8MB | 363.2MB |
| dense | 4096 | 227926.2 | 35.9415 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 206383.3 | 39.6931 | 1.5GB | 31.5MB | 1.5GB |
| wayfinder_permute | 4096 | 484311.0 | 16.9148 | 755.7MB | 31.5MB | 724.2MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
