# Torch Wayfinder CUDA Scaling Benchmark

- created_at: `2026-02-07T05:49:02.034506+00:00`
- command: `scripts/bench_torch_wayfinder_scale.py --device cpu --dtype float32 --seq-lens 32,64 --batch 1 --heads 4 --embd 128 --window 16 --landmark-stride 16 --warmup 1 --iters 2 --graph-spec configs/graph_specs/default.wf`
- device: `cpu`

## Attention Throughput & Memory

| mode | T | tok/s | graph first ms | graph cached ms | attention ms | permute ms | peak mem | persistent cache | step intermediates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 32 | 32582.4 | 0.0000 | 0.0000 | 0.9821 | 0.0000 | 0.0B | 0.0B | 0.0B |
| hha_sparse | 32 | 21061.6 | 2.4592 | 0.0849 | 0.5244 | 0.0000 | 0.0B | 115.2KB | 0.0B |
| hha_permute | 32 | 16353.6 | 1.3243 | 0.0583 | 1.4586 | 0.0000 | 0.0B | 115.2KB | 0.0B |
| dense | 64 | 431764.4 | 0.0000 | 0.0000 | 0.1482 | 0.0000 | 0.0B | 0.0B | 0.0B |
| hha_sparse | 64 | 77335.5 | 2.2511 | 0.0493 | 0.6890 | 0.0000 | 0.0B | 374.5KB | 0.0B |
| hha_permute | 64 | 34602.8 | 2.2425 | 0.0603 | 1.6677 | 0.0000 | 0.0B | 374.5KB | 0.0B |

## Notes

- `graph_build_ms_cached` should be near zero for static/random strategy on cache hits.
- `hha_sparse` is the correctness/reference path; `hha_permute` is the fast path.

