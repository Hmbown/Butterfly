# Wayfinder v0.3 Overnight Start Proof

## Source Artifacts
- runs/mlx/20260207_043737 (baseline)
- runs/mlx/20260207_043756 (edge-bias + window-drop)
- benchmarks/mlx/bench_results.json

## Tiny Train Summary (200 steps, seq_len=128)

| Run | Dense val ppl | Wayfinder val ppl | Gap | Dense tok/s | Wayfinder tok/s | Ratio | Cycle% |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 27.032 | 29.957 | +2.925 | 111583.0 | 56923.5 | 0.510 | 3.05 |
| edge_bias_drop | 27.041 | 28.624 | +1.583 | 100468.6 | 95633.8 | 0.952 | 2.98 |

## MLX Bench @ T=2048

| Mode | Attention tok/s | Attention peak MB | Block tok/s | Block peak MB | Graph build first ms | Graph build cached ms |
|---|---:|---:|---:|---:|---:|---:|
| dense | 80975.5 | 531.4 | 191185.8 | 551.2 | 0.0000 | 0.0000 |
| wayfinder_sparse | 122665.3 | 457.7 | 76440.2 | 477.5 | 393.0603 | 0.0054 |
| wayfinder_permute | 103871.9 | 317.8 | 119767.1 | 337.6 | 228.6488 | 0.0044 |
