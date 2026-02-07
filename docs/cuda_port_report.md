# Wayfinder CUDA / PyTorch Port Report

## Architecture Mapping

This port adds a dedicated PyTorch backend under `hcsa/torch` while preserving the existing MLX path.

Implemented components:

- `hcsa/torch/attention_dense.py`
  - Dense causal baseline using `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`.
  - Manual masked-matmul fallback for environments where SDPA backend is unavailable.

- `hcsa/torch/attention_wayfinder_sparse.py`
  - Wayfinder sparse-row reference path:
    - accepts graph ABI neighbor index and edge types,
    - gathers K/V by neighbor index,
    - computes logits in FP32,
    - uses stable masked softmax,
    - returns zero rows for all-masked queries (no NaNs),
    - optional attention-weight return for metrics.
  - `WayfinderAttentionTorch` module mirrors MLX runtime controls and profile fields:
    - `graph_build_ms`, `attention_ms`, `permute_ms`, `cache_hit`, `cache_source`, `cache_persistent_bytes`.
  - Supports loading compiled graph artifacts from `wayc` output (`neighborindex.npz` + `meta.json`).

- `hcsa/torch/attention_wayfinder_permute.py`
  - Wayfinder permute-window fast path:
    - permute to cycle order,
    - local window attention in permuted space,
    - original-position causal enforcement,
    - inverse permute back to original order,
    - no T x T mask materialization in forward path.
  - Uses precomputed per-head permute/cache tensors when available.

- `hcsa/torch/model.py`
  - Torch GPT model parity surface with attention modes:
    - `dense`, `wayfinder_sparse`, `wayfinder_permute`.

- `hcsa/torch/bench_utils.py`
  - Shared timing, mask/softmax utilities, graph normalization, compiled ABI loading, graph/edge metrics helpers.

## Graph ABI / Compiler Artifact Compatibility

The torch backend uses the same ABI structures and compiler artifacts:

- ABI: `hcsa.graph.abi.WayfinderGraphABI`, `EdgeType`.
- Neighbor index shape support:
  - `[T,D]`, `[H,T,D]`, and `[B,H,T,D]` with `-1` padding.
- Compiler artifact compatibility:
  - consumes `neighborindex.npz` and `meta.json` produced by `scripts/wayc.py` / `hcsa.compiler.compile_graph_spec`.

## Parity & Correctness Tests

Added under `tests/pytorch/`:

- `test_torch_causality.py`
  - sparse-row masks future neighbors,
  - permute path enforces causality in original token order.

- `test_torch_no_nan.py`
  - all-masked neighborhoods produce finite zero outputs,
  - zero-degree neighborhoods are stable.

- `test_torch_dense_limit_equivalence.py`
  - sparse-row with full past-neighbor set matches dense causal attention.

- `test_torch_permute_roundtrip.py`
  - permutation/inverse roundtrip correctness,
  - full-window permute path matches dense causal output.

- `test_torch_matches_mlx_reference_small.py`
  - torch sparse-row matches a shared NumPy reference,
  - optional MLX direct parity check when MLX is installed.

Validation run in this environment:

- Command: `PYTHONPATH=. .venv312/bin/python -m pytest -q tests/pytorch`
- Result: `9 passed`
- Note: CUDA device was not available in this environment (`torch.cuda.is_available() == False`), so CUDA execution was not validated here.

## Benchmark Suite

Added CUDA-focused benchmark script:

- `scripts/bench_torch_wayfinder_scale.py`
  - Compares: `dense` vs `wayfinder_sparse` (reference) vs `wayfinder_permute` (fast path)
  - Reports per mode and sequence length:
    - `tokens_per_sec`, `attention_ms`, `graph_build_ms_first`, `graph_build_ms_cached`,
    - cache hit rate,
    - peak memory (`torch.cuda.max_memory_allocated`) on CUDA,
    - intermediate-memory proxy,
    - compiled graph artifact path used.
  - Saves artifacts to: `benchmarks/cuda/scale_<timestamp>/`.

Smoke run (CPU only in this environment):

- Command:
  - `PYTHONPATH=. .venv312/bin/python scripts/bench_torch_wayfinder_scale.py --device cpu --dtype float32 --seq-lens 32,64 --batch 1 --heads 4 --embd 128 --window 16 --landmark-stride 16 --warmup 1 --iters 2 --graph-spec configs/graph_specs/default.wf`
- Artifacts:
  - `benchmarks/cuda/scale_20260207_054901/results.json`
  - `benchmarks/cuda/scale_20260207_054901/README.md`
- Observed cache behavior:
  - cached graph build was near-zero (`~0.05-0.08 ms`) after first build (`~1.3-2.5 ms`) for random/static strategy.

## HF Integration Target (Microbench)

Implemented:

- `scripts/qwen3_torch_attention_microbench.py`
  - Runs dense causal attention vs Wayfinder-permute on Qwen3-shaped tensors.
  - Two modes:
    - synthetic/config-shaped QKV (default),
    - real HF QKV extraction (`--load-model`) when model weights are available.

Smoke artifact (CPU/config-shaped):

- `benchmarks/cuda/qwen3_torch_microbench_smoke.json`

## Differences vs MLX Path

- MLX backend remains unchanged under `hcsa/mlx/*`.
- Torch backend mirrors semantics but is implemented with PyTorch tensor ops and SDPA.
- Both paths share ABI/compiler contract and graph-substrate visibility fields in debug/profile outputs.

## Next Steps

1. Run the new benchmark script on real CUDA hardware at long sequence lengths (`256..4096+`) and collect throughput/memory curves.
2. Add Triton kernel(s) under `hcsa/torch/kernels/` for sparse-row and/or permute-window once CUDA parity baselines are locked.
3. Add optional HF attention-module swap script if full-model integration beyond QKV microbench is required.
