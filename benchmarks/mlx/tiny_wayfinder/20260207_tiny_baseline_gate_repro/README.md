# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:45:50.884189+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_baseline_gate_repro`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 769611.8 | 0.0000 | 0.0000 | 0.6653 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| wayfinder_sparse | 256 | 433217.6 | 20.7892 | 0.0012 | 0.0622 | 0.0000 | 27.0MB | 818.0KB | 26.2MB |
| wayfinder_permute | 256 | 411183.2 | 15.3160 | 0.0016 | 0.0922 | 0.0000 | 42.6MB | 818.0KB | 41.8MB |
| dense | 512 | 1999755.9 | 0.0000 | 0.0000 | 0.5121 | 0.0000 | 40.5MB | 0.0B | 40.5MB |
| wayfinder_sparse | 512 | 763194.1 | 31.6960 | 0.0010 | 0.0598 | 0.0000 | 65.3MB | 1.8MB | 63.6MB |
| wayfinder_permute | 512 | 560539.0 | 31.0963 | 0.0024 | 0.1092 | 0.0000 | 85.7MB | 1.8MB | 84.0MB |
| dense | 1024 | 1154304.2 | 0.0000 | 0.0000 | 1.7742 | 0.0000 | 144.1MB | 0.0B | 144.1MB |
| wayfinder_sparse | 1024 | 559126.8 | 76.8278 | 0.0037 | 0.0819 | 0.0000 | 164.3MB | 4.1MB | 160.2MB |
| wayfinder_permute | 1024 | 585976.5 | 68.4557 | 0.0036 | 0.1091 | 0.0000 | 165.5MB | 4.1MB | 161.4MB |
| dense | 2048 | 524707.8 | 0.0000 | 0.0000 | 7.8062 | 0.0000 | 541.1MB | 0.0B | 541.1MB |
| wayfinder_sparse | 2048 | 440031.9 | 169.6322 | 0.0029 | 0.0816 | 0.0000 | 466.1MB | 10.8MB | 455.4MB |
| wayfinder_permute | 2048 | 620484.6 | 168.2737 | 0.0040 | 0.1436 | 0.0000 | 326.9MB | 10.8MB | 316.1MB |
| dense | 4096 | 267980.3 | 0.0000 | 0.0000 | 30.5694 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 276021.0 | 483.5109 | 0.0042 | 0.0971 | 0.0000 | 1.5GB | 31.5MB | 1.4GB |
| wayfinder_permute | 4096 | 648394.1 | 471.2212 | 0.0042 | 0.1208 | 0.0000 | 689.4MB | 31.5MB | 657.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 780339.1 | 0.6561 | 13.3MB | 0.0B | 13.3MB |
| wayfinder_sparse | 256 | 520568.0 | 0.9835 | 31.2MB | 818.0KB | 30.4MB |
| wayfinder_permute | 256 | 395048.9 | 1.2960 | 46.8MB | 818.0KB | 46.0MB |
| dense | 512 | 1392723.6 | 0.7353 | 46.0MB | 0.0B | 46.0MB |
| wayfinder_sparse | 512 | 638437.0 | 1.6039 | 72.9MB | 1.8MB | 71.2MB |
| wayfinder_permute | 512 | 530799.3 | 1.9292 | 90.5MB | 1.8MB | 88.7MB |
| dense | 1024 | 1049808.0 | 1.9508 | 154.4MB | 0.0B | 154.4MB |
| wayfinder_sparse | 1024 | 516242.9 | 3.9671 | 174.5MB | 4.1MB | 170.4MB |
| wayfinder_permute | 1024 | 534970.3 | 3.8283 | 175.8MB | 4.1MB | 171.6MB |
| dense | 2048 | 496048.9 | 8.2573 | 560.8MB | 0.0B | 560.8MB |
| wayfinder_sparse | 2048 | 418055.2 | 9.7978 | 485.9MB | 10.8MB | 475.1MB |
| wayfinder_permute | 2048 | 583025.9 | 7.0254 | 346.7MB | 10.8MB | 335.9MB |
| dense | 4096 | 259794.4 | 31.5326 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 266500.4 | 30.7392 | 1.5GB | 31.5MB | 1.5GB |
| wayfinder_permute | 4096 | 621445.6 | 13.1822 | 728.2MB | 31.5MB | 696.6MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
