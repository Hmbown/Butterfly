# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-08T23:46:45.152683+00:00`
- command: `/Volumes/VIXinSSD/wayfinder/scripts/bench_mlx_wayfinder_scale.py --seq-lens 4096 8192 16384 --batch 1 --heads 8 --embd 256 --window 32 --warmup 2 --iters 4 --out-dir /Volumes/VIXinSSD/wayfinder/benchmarks/mlx/tiny_wayfinder/k1_fused`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 4096 | 397378.1 | 0.0000 | 0.0000 | 10.3076 | 0.0000 | 563.0MB | 0.0B | 563.0MB |
| wayfinder_sparse | 4096 | 265322.9 | 719.9174 | 0.0047 | 0.1438 | 0.0000 | 971.6MB | 43.6MB | 927.9MB |
| wayfinder_permute | 4096 | 1617254.1 | 714.5466 | 0.0018 | 2.2717 | 2.2714 | 156.5MB | 43.6MB | 112.8MB |
| dense | 8192 | 199598.5 | 0.0000 | 0.0000 | 41.0424 | 0.0000 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 8192 | 172718.8 | 1968.4909 | 0.0059 | 0.1581 | 0.0000 | 2.5GB | 127.2MB | 2.4GB |
| wayfinder_permute | 8192 | 2052339.7 | 1974.9967 | 0.0027 | 3.6768 | 3.6763 | 351.8MB | 127.2MB | 224.6MB |
| dense | 16384 | 97449.6 | 0.0000 | 0.0000 | 168.1280 | 0.0000 | 8.5GB | 0.0B | 8.5GB |
| wayfinder_sparse | 16384 | 99255.7 | 6264.4650 | 0.0063 | 0.1760 | 0.0000 | 5.7GB | 414.5MB | 5.3GB |
| wayfinder_permute | 16384 | 2161750.9 | 6190.9361 | 0.0038 | 7.1063 | 7.1059 | 747.3MB | 414.5MB | 332.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 4096 | 368026.8 | 11.1296 | 603.6MB | 0.0B | 603.6MB |
| wayfinder_sparse | 4096 | 255689.0 | 16.0195 | 984.1MB | 43.6MB | 940.5MB |
| wayfinder_permute | 4096 | 1330311.1 | 3.0790 | 173.0MB | 43.6MB | 129.4MB |
| dense | 8192 | 192878.7 | 42.4723 | 2.2GB | 0.0B | 2.2GB |
| wayfinder_sparse | 8192 | 167912.7 | 48.7873 | 2.5GB | 127.2MB | 2.4GB |
| wayfinder_permute | 8192 | 1470065.3 | 5.5725 | 382.4MB | 127.2MB | 255.1MB |
| dense | 16384 | 95319.5 | 171.8850 | 8.7GB | 0.0B | 8.7GB |
| wayfinder_sparse | 16384 | 97526.8 | 167.9948 | 5.8GB | 414.5MB | 5.4GB |
| wayfinder_permute | 16384 | 1559780.5 | 10.5040 | 815.7MB | 414.5MB | 401.2MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
