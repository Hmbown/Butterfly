# MLX Wayfinder Scaling Benchmark

- created_at: `2026-02-07T16:48:54.283392+00:00`
- command: `scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 256 | 1300456.2 | 0.0000 | 0.0000 | 0.3937 | 0.0000 | 10.2MB | 0.0B | 10.2MB |
| wayfinder_sparse | 256 | 432561.8 | 18.8744 | 0.0014 | 0.0639 | 0.0000 | 27.0MB | 826.0KB | 26.2MB |
| wayfinder_permute | 256 | 793516.7 | 15.7606 | 0.0010 | 0.4998 | 0.4996 | 15.3MB | 826.0KB | 14.5MB |
| dense | 512 | 2121270.8 | 0.0000 | 0.0000 | 0.4827 | 0.0000 | 37.3MB | 0.0B | 37.3MB |
| wayfinder_sparse | 512 | 736613.3 | 32.3158 | 0.0014 | 0.0670 | 0.0000 | 62.1MB | 1.8MB | 60.4MB |
| wayfinder_permute | 512 | 1045876.6 | 30.6708 | 0.0012 | 0.8131 | 0.8129 | 19.8MB | 1.8MB | 18.0MB |
| dense | 1024 | 1180020.5 | 0.0000 | 0.0000 | 1.7356 | 0.0000 | 138.8MB | 0.0B | 138.8MB |
| wayfinder_sparse | 1024 | 575179.9 | 80.4193 | 0.0039 | 0.0800 | 0.0000 | 159.1MB | 4.2MB | 154.9MB |
| wayfinder_permute | 1024 | 1168504.9 | 68.4467 | 0.0013 | 1.5882 | 1.5879 | 27.0MB | 4.2MB | 22.8MB |
| dense | 2048 | 522529.5 | 0.0000 | 0.0000 | 7.8388 | 0.0000 | 535.9MB | 0.0B | 535.9MB |
| wayfinder_sparse | 2048 | 444616.3 | 171.7690 | 0.0028 | 0.0709 | 0.0000 | 461.0MB | 10.8MB | 450.1MB |
| wayfinder_permute | 2048 | 1187260.8 | 168.5436 | 0.0017 | 3.2591 | 3.2587 | 41.6MB | 10.8MB | 30.7MB |
| dense | 4096 | 260379.2 | 0.0000 | 0.0000 | 31.4618 | 0.0000 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 276022.2 | 483.0754 | 0.0040 | 0.0865 | 0.0000 | 1.4GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 1227139.5 | 480.9728 | 0.0026 | 6.4204 | 6.4199 | 79.5MB | 31.7MB | 47.8MB |

## 1-Block End-to-End Memory

| mode | T | tok/s | latency ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 628638.9 | 0.8145 | 13.3MB | 0.0B | 13.3MB |
| wayfinder_sparse | 256 | 559843.1 | 0.9145 | 30.1MB | 826.0KB | 29.3MB |
| wayfinder_permute | 256 | 661250.1 | 0.7743 | 17.2MB | 826.0KB | 16.4MB |
| dense | 512 | 1415017.4 | 0.7237 | 42.9MB | 0.0B | 42.9MB |
| wayfinder_sparse | 512 | 669810.3 | 1.5288 | 67.6MB | 1.8MB | 65.8MB |
| wayfinder_permute | 512 | 805902.6 | 1.2706 | 22.8MB | 1.8MB | 21.0MB |
| dense | 1024 | 1034496.0 | 1.9797 | 149.1MB | 0.0B | 149.1MB |
| wayfinder_sparse | 1024 | 576198.1 | 3.5543 | 169.3MB | 4.2MB | 165.2MB |
| wayfinder_permute | 1024 | 980354.1 | 2.0890 | 32.2MB | 4.2MB | 28.1MB |
| dense | 2048 | 505676.5 | 8.1000 | 555.6MB | 0.0B | 555.6MB |
| wayfinder_sparse | 2048 | 419523.5 | 9.7635 | 480.8MB | 10.8MB | 469.9MB |
| wayfinder_permute | 2048 | 1073961.5 | 3.8139 | 51.3MB | 10.8MB | 40.5MB |
| dense | 4096 | 260713.9 | 31.4214 | 2.1GB | 0.0B | 2.1GB |
| wayfinder_sparse | 4096 | 268174.7 | 30.5473 | 1.5GB | 31.7MB | 1.4GB |
| wayfinder_permute | 4096 | 1072508.6 | 7.6382 | 98.2MB | 31.7MB | 66.5MB |

## Notes
- `graph_build_ms_cached` should be near zero on cache hits.
- `step intermediates` is reported as `peak_memory_bytes - persistent_cache_bytes` for consistent decomposition.
