# Wayfinder (HCSA) - graph-sparse causal attention for inference

Wayfinder implements Hamiltonian Cycle Sparse Attention (HCSA): an inference-time runtime that replaces dense causal self-attention with a bounded-degree graph neighborhood.

The graph has three ingredients:
- local causal window
- Hamiltonian-cycle backbone from a random permutation of token positions
- optional landmark tokens (every k-th position)

In this repo, "Hamiltonian cycle" means a permutation-induced cycle over token indices (each token gets two cycle neighbors, then causal masking is applied), not a metric-space TSP claim.

Status for this release:
- Validated path: MLX backend (`hcsa/integrations/glm_mlx.py`) on Apple Silicon, model `mlx-community/GLM-4.7-Flash-4bit`, `T=8192`, `decode_len=32`, single-turn benchmark.
- Decode policy: dense by default (`wayfinder_decode_backend="dense"`), and decode steps (`q_len <= 2`) route to standard dense SDPA. Wayfinder is intentionally a prefill optimization path.
- Experimental/non-default: Qwen and Nanbeige integrations.
- PyTorch backend exists (`hcsa/torch/`) but is not the validated focus of this release.

Important: if a model was trained with dense attention, Wayfinder changes the computation. It is an approximation. Evaluate quality on your workload.

## Where this fits

This is:
- A training-free sparse-attention runtime + ABI + benchmark harness.
- In the same research space as XAttention and SpargeAttn (training-free sparse inference acceleration), but with a different mechanism: Wayfinder uses a graph-defined neighborhood (window + permutation-cycle + landmarks) instead of their selection schemes.
- A real cross-backend graph ABI: `neigh_idx` int32 with shape `[T,D]` or `[H,T,D]`, padded with `-1`; `edge_type` uint8 enum `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`.

This is not:
- A quality-parity guarantee for every model or prompt.
- Exact dense attention with a faster kernel (that is the FlashAttention / FlashInfer category).

Mental model: dense causal attention is a complete causal DAG over positions; HCSA is a bounded-degree sparse causal DAG.

## Attention pattern comparison

Five causal attention patterns at sequence length `T=64` (illustrative mask size, not the benchmark length):

| Pattern | Edges | Out-degree | Notes |
|---|---|---|---|
| Dense causal | `T(T+1)/2` | `O(T)` | Baseline |
| Sliding window | `O(T*w)` | `O(w)` | Local only |
| Longformer-style | `O(T*w + T*g)` | `O(w+g)` | Window + global prefix tokens |
| BigBird-style | `O(T*w + T*g + T*r)` | `O(w+g+r)` | Window + global + random keys |
| HCSA (Wayfinder) | `O(T*w + T + T/s)` | `O(w+2+T/s)` | Window + cycle backbone + landmarks |

The HCSA panel uses the same mask logic as the implementation: random Hamiltonian permutation + causal cycle-neighbor edges + window + landmark stride.

Reproduce:
```bash
python3 scripts/viz/attention_pattern_comparison.py --seq-len 64 --window 8
python3 scripts/viz/graph_viz.py --seq-len 32 --out docs/assets/hcsa_graph_circle.png
```

## Results: GLM-4.7-Flash-4bit (Apple Silicon, MLX)

Validated run: `EXP-20260218T151213Z-STABLE-PROFILE` (`T=8192`, `decode_len=32`, single turn).

| Metric | Unit | Dense | Wayfinder | Delta (Wayfinder vs Dense) |
|---|---:|---:|---:|---:|
| End-to-end time | `s` | `17.1473 s` | `10.5563 s` | `-38.44%` |
| Prefill time | `s` | `16.3586 s` | `9.7533 s` | `-40.38%` |
| Decode time | `s` | `0.7886 s` | `0.8030 s` | `+1.82%` |
| Decode throughput | `tok/s` | `40.5762 tok/s` | `39.8499 tok/s` | `-1.79%` |
| Peak memory | `GB` | `~20.66 GB` | `~20.07 GB` | `-2.85%` |

Takeaway: speedup comes from prefill. Decode behavior is intentionally dense-first, which avoids decode-quality regression from sparse decode routing.

Full evidence and artifacts: `docs/FIRST_RELEASE.md`.

## Research context

Sparse structured attention references:
- Longformer: <https://arxiv.org/abs/2004.05150>
- BigBird: <https://arxiv.org/abs/2007.14062>
- Mistral 7B (sliding-window attention): <https://arxiv.org/abs/2310.06825>

Training-free sparse inference papers in the same space:
- XAttention (ICML 2025): <https://arxiv.org/abs/2503.16428>
- SpargeAttn (ICML 2025): <https://arxiv.org/abs/2502.18137>

Different bucket (exact attention, faster kernels):
- FlashAttention: <https://arxiv.org/abs/2205.14135>
- FlashInfer: <https://arxiv.org/abs/2501.01005>

## Try It Now (Interactive Chat)

This script is a thin wrapper over `mlx_lm.load` and `mlx_lm.stream_generate`:
- `scripts/chat_glm_wayfinder.py`

```bash
git clone https://github.com/Hmbown/Wayfinder && cd Wayfinder
pip install -e ".[mlx]"
python3 scripts/chat_glm_wayfinder.py --model-path /path/to/GLM-4.7-Flash-4bit
```

Or let `mlx_lm` download it automatically:

```bash
python3 scripts/chat_glm_wayfinder.py
```

Dense baseline:

```bash
python3 scripts/chat_glm_wayfinder.py --mode dense
```

## Quick Start (5 Minutes)

```bash
git clone <this-repo> && cd <this-repo>
pip install -e ".[dev]"
pip install -e ".[mlx]"
pip install -e ".[viz]"

./scripts/verify_install_and_preflight.sh \
  --run-id EXP-YYYYMMDDTHHMMSSZ-VERIFY-INSTALL \
  --out-dir benchmarks/mlx/preflight
```

Expected verify artifacts:
- `benchmarks/mlx/preflight/<RUN_ID>_env_check_mlx.json`
- `benchmarks/mlx/preflight/<RUN_ID>_summary.json`
- `benchmarks/mlx/preflight/<RUN_ID>_raw.txt`

## First Successful Run

```bash
python3 scripts/bench_glm_consumer_mlx.py \
  --mode dense \
  --seq-lens 2048 \
  --decode-len 8 \
  --repeats 1 \
  --skip-multi-turn \
  --skip-quality \
  --out-dir benchmarks/mlx/first_release/first_run_dense_t2048
```

Success criteria:
- exit code is `0`
- `benchmarks/mlx/first_release/first_run_dense_t2048/results.json` exists

## Stable Public Profile (Default)

```bash
./scripts/run_public_stable_profile_glm.sh
```

Output artifacts:
- `<out-root>/<run-id>/dense/results.json`
- `<out-root>/<run-id>/wayfinder/results.json`
- `<out-root>/<run-id>/stable_profile_summary.json`
- `<out-root>/<run-id>/stable_profile_summary.md`

## Support Matrix (Validated vs Experimental)

| Tier | Status | Scope | Default | Evidence |
|---|---|---|---|---|
| Validated | Recommended | GLM-4.7 stable wrapper path (`T=8192`, `decode_len=32`) | Yes | `docs/FIRST_RELEASE.md` |
| Experimental | Opt-in only | Qwen and Nanbeige diagnostic slices | No | `docs/FIRST_RELEASE.md` |
| Known regression | Non-default | Nanbeige `T=131072, decode_len=256` | No | `docs/FIRST_RELEASE.md` |

Boundary note: Nanbeige `T=131072, decode_len=32` has diagnostic data but remains experimental/non-default.

## Troubleshooting

- If verify or benchmark exits nonzero, treat the run as failed and fix environment first.
- If `results.json` is missing, do not compute deltas from partial outputs.
- For fallback diagnostics, add `--hsa-trace`.
- Keep Nanbeige `T=131072` slices non-default unless release evidence changes.

Release evidence: `docs/FIRST_RELEASE.md`. Architecture details: `docs/ARCHITECTURE.md`. Research notes: `docs/RESEARCH.md`.

## License

MIT. See `LICENSE`.
