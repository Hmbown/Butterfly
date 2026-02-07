# MLX Reality Report

## Environment
- Repo: `hcsa` (local clone)
- Python: `3.12` virtual env at `.venv312`
- MLX: `0.30.6`
- Device: Apple Silicon + Metal backend
- Date: 2026-02-07

## Tree Snapshot (Relevant)
- `hcsa/graph/abi.py` (Graph ABI + validators + metrics)
- `hcsa/mlx/graph_abi.py` (MLX conversions)
- `hcsa/mlx/attention.py` (dense/sparse/permute paths)
- `hcsa/mlx/model.py` (MLX GPT wiring)
- `scripts/bench_mlx_wayfinder.py` (MLX benchmark)
- `scripts/run_mlx_experiment_tiny.py` (dense vs wayfinder experiment)
- `tests_mlx/` (MLX-specific tests)
- `hcsa/graph_strategies.py` (GraphStrategy `build()` now returns ABI)

## Smoke Checks
- `GPTMLX` forward works for:
  - `dense`
  - `wayfinder_sparse`
  - `wayfinder_permute`
- MLX generate loop works on Wayfinder model.
- Measured smoke times:
  - `dense` forward (`B=2,T=64,C=128`): `12.82 ms`
  - `wayfinder_sparse` forward (`B=2,T=64,C=128`): `27.78 ms`
  - `wayfinder_permute` forward (`B=2,T=64,C=128`): `18.40 ms`
  - generate 6 tokens (`wayfinder_permute`, start length 8): `23.64 ms`

## Current Hotspots (Expected)
- `graph_build_ms`: CPU graph construction dominates at small T.
- `sparse_gather_attention`: irregular gather pressure at larger T.
- `wayfinder_permute`: lower irregular memory traffic; intended MLX fast path.
- Initial blocker fixed: root-level `mlx/` package shadowed Apple MLX imports (`import mlx.core` failed). Runtime is now consolidated under `hcsa/mlx/`.

## Status
- ABI-first MLX wiring complete.
- Benchmark + experiment + tests added.

## Test Results
- `python -m pytest -q tests_mlx` -> `6 passed`.

## Benchmark Results (scripts/bench_mlx_wayfinder.py)
Config used:
- `B=2, H=4, C=128, window=32, landmark_stride=32, seq_len=[128,256,512,1024,2048], warmup=2, iters=4`

Selected attention-path results:
- `T=128`: dense `119,029 tok/s`, sparse `28,776 tok/s`, permute `29,043 tok/s`
- `T=1024`: dense `200,938 tok/s`, sparse `22,494 tok/s`, permute `20,864 tok/s`
- `T=2048`: dense `167,378 tok/s`, sparse `17,061 tok/s`, permute `17,042 tok/s`
- At `T=2048`, peak memory:
  - dense: `531.4MB`
  - sparse: `461.5MB`
  - permute: `322.0MB`

Breakdown trend:
- `graph_build_ms` is the dominant cost for Wayfinder path in this reference build.
- `attention_ms` for sparse/permute kernels remains sub-millisecond to low-millisecond in this config.

## Tiny Experiment Results (scripts/run_mlx_experiment_tiny.py)
Run directory:
- `runs/mlx/20260207_035224`

Config used:
- TinyShakespeare, char tokenizer, `steps=200`, `B=8`, `T=128`, `layers=2`, `heads=4`, `embd=128`, wayfinder mode=`wayfinder_sparse`.

Final metrics:
- Dense final validation PPL: `26.840`
- Wayfinder final validation PPL: `28.768`
- Dense avg throughput: `96,040.5 tok/s`
- Wayfinder avg throughput: `19,831.7 tok/s`
- Graph metrics (Wayfinder):
  - `shortcut_rate=0.979`
  - `reachability_proxy=128.00`
- Edge utilization (Wayfinder sparse attention mass):
  - `cycle=0.0299`
  - `window=0.9420`
  - `landmark=0.0281`
  - `rewire=0.0000`

Artifacts generated:
- `runs/mlx/20260207_035224/config.json`
- `runs/mlx/20260207_035224/metrics.jsonl`
- `runs/mlx/20260207_035224/summary.json`
- `runs/mlx/20260207_035224/wayfinder_graph_debug.npz`
