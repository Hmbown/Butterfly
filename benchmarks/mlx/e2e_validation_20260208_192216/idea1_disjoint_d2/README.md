# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T19:24:44.793106+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --num-cycles 2 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea1_disjoint_d2`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1243323.5 | 0.0000 | 0.0000 | 3.2944 | 0.0000 | 149.3MB | 0.0B | 149.3MB |
| wayfinder_sparse | 2048 | 417159.3 | 220.9410 | 0.0058 | 0.0974 | 0.0000 | 470.1MB | 11.0MB | 459.0MB |
| wayfinder_permute | 2048 | 1655270.2 | 201.8081 | 0.0021 | 2.1087 | 2.1083 | 64.4MB | 11.0MB | 53.3MB |
| dense | 4096 | 733379.3 | 0.0000 | 0.0000 | 11.1702 | 0.0000 | 573.5MB | 0.0B | 573.5MB |
| wayfinder_sparse | 4096 | 268251.2 | 540.4849 | 0.0060 | 0.1005 | 0.0000 | 1.5GB | 32.1MB | 1.4GB |
| wayfinder_permute | 4096 | 1771466.5 | 541.8642 | 0.0032 | 4.3094 | 4.3089 | 138.7MB | 32.1MB | 106.6MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 1145006.9 | 3.5773 | 169.0MB | 0.0B | 169.0MB |
| wayfinder_sparse | 2048 | 406619.8 | 10.0733 | 489.8MB | 11.0MB | 478.8MB |
| wayfinder_permute | 2048 | 1402679.7 | 2.9201 | 74.1MB | 11.0MB | 63.1MB |
| dense | 4096 | 669491.8 | 12.2361 | 612.3MB | 0.0B | 612.3MB |
| wayfinder_sparse | 4096 | 262448.6 | 31.2137 | 1.5GB | 32.1MB | 1.5GB |
| wayfinder_permute | 4096 | 1528049.3 | 5.3611 | 157.5MB | 32.1MB | 125.4MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
