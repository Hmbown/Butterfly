# Wayfinder (HCSA): graph-sparse causal attention for inference

Wayfinder implements Hamiltonian Cycle Sparse Attention (HCSA), an inference-time runtime that replaces dense causal self-attention with a bounded-degree graph neighborhood.

The graph has three ingredients:
- local causal window
- Hamiltonian-cycle backbone from a random permutation of token positions
- optional landmark tokens (every k-th position)

In this repo, "Hamiltonian cycle" means a permutation-induced cycle over token indices (each token gets two cycle neighbors, then causal masking is applied), not a metric-space TSP claim.

Release status:
- Validated path: MLX backend (`hcsa/integrations/glm_mlx.py`) on Apple Silicon, model `mlx-community/GLM-4.7-Flash-4bit`, `T=8192`, `decode_len=32`, single-turn benchmark.
- Decode policy: dense by default (`wayfinder_decode_backend="dense"`). Decode steps (`q_len <= 2`) route to standard dense SDPA; Wayfinder is currently a prefill optimization path.
- Strict path-audit follow-up (`EXP-20260219T040010Z-GLM47-STRICT-OBS-RERUN`) passed with explicit decode fallback reasons (`wayfinder_decode_dense`, no `unspecified`) and no OOM (`20.07 GB` at `T=8192`, `21.98 GB` at `T=32768`, both below a `28 GB` safety cap).
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
Terminology map: `docs/GLOSSARY.md`.

## Visual summary

Suggested order:
1. Pattern landscape (`docs/assets/attention_comparison_5panel.png`)
2. Hamiltonian mechanism (`docs/assets/hcsa_graph_circle.png`)
3. Validated metric card (`docs/assets/wayfinder_metric_card.png`)

Storyboard and narration guidance: `docs/VISUAL_STORYBOARD.md`.

![Wayfinder attention pattern comparison](docs/assets/attention_comparison_5panel.png)
![Wayfinder Hamiltonian cycle graph](docs/assets/hcsa_graph_circle.png)
![Wayfinder validated metric card](docs/assets/wayfinder_metric_card.png)

## Attention pattern comparison details

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
python3 scripts/viz/attention_pattern_comparison.py --seq-len 64 --window 8 --out docs/assets/attention_comparison_5panel.png
python3 scripts/viz/graph_viz.py --seq-len 32 --window 4 --landmark-stride 8 --out docs/assets/hcsa_graph_circle.png
python3 scripts/viz/wayfinder_metric_card.py \
  --stable-summary-json benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/stable_profile_summary.json \
  --out docs/assets/wayfinder_metric_card.png
```

## Validated results: GLM-4.7-Flash-4bit (Apple Silicon, MLX)

Validated run: `EXP-20260218T151213Z-STABLE-PROFILE` (`T=8192`, `decode_len=32`, single turn).

| Metric | Unit | Dense | Wayfinder | Delta (Wayfinder vs Dense) |
|---|---:|---:|---:|---:|
| End-to-end time | `s` | `17.1473 s` | `10.5563 s` | `-38.44%` |
| Prefill time | `s` | `16.3586 s` | `9.7533 s` | `-40.38%` |
| Decode time | `s` | `0.7886 s` | `0.8030 s` | `+1.82%` |
| Decode throughput | `tok/s` | `40.5762 tok/s` | `39.8499 tok/s` | `-1.79%` |
| Peak memory | `GB` | `~20.66 GB` | `~20.07 GB` | `-2.85%` |

Takeaway: the measured improvement is a prefill effect. Decode remains dense-first by design, which keeps the validated path conservative.

Full evidence and artifacts: `docs/FIRST_RELEASE.md`.

Follow-up token-length sweep (`EXP-20260218T183512Z-GLM47-TOKENLEN-SWEEP`, single-turn, `decode_len=32`, `repeats=1`):

| Seq Len (T) | Dense E2E (s) | Wayfinder E2E (s) | E2E Delta | Dense Prefill (s) | Wayfinder Prefill (s) | Prefill Delta | Dense Decode (s) | Wayfinder Decode (s) | Decode Delta | Dense Decode tok/s | Wayfinder Decode tok/s | Decode tok/s Delta | Dense Peak (GB) | Wayfinder Peak (GB) | Peak Memory Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 3.1354 | 2.5831 | -17.62% | 2.4644 | 2.0189 | -18.08% | 0.6710 | 0.5642 | -15.93% | 47.6869 | 56.7206 | +18.94% | 18.28 | 18.32 | -0.19% |
| 8192 | 16.8978 | 9.3819 | -44.48% | 16.0745 | 8.6491 | -46.19% | 0.8234 | 0.7328 | -11.00% | 38.8649 | 43.6663 | +12.35% | 20.66 | 20.07 | +2.85% |
| 32768 | 203.3019 | 112.0960 | -44.86% | 193.3550 | 110.7578 | -42.72% | 9.9469 | 1.3382 | -86.55% | 3.2171 | 23.9132 | +643.32% | 26.02 | 21.98 | +15.52% |
| 65536 | 990.0789 | 268.3590 | -72.90% | 961.6474 | 264.4541 | -72.50% | 28.4316 | 3.9049 | -86.27% | 1.1255 | 8.1948 | +628.10% | 33.16 | 23.80 | +28.24% |

Sweep artifacts:
- `benchmarks/mlx/first_release/EXP-20260218T183512Z-GLM47-TOKENLEN-SWEEP/token_length_summary.json`
- `benchmarks/mlx/first_release/EXP-20260219T040010Z-GLM47-STRICT-OBS-RERUN/strict_obs_gate_report.json`

## Current Qwen 3.5 work

Qwen 3.5 support is experimental and is not part of the validated public release path.

Current work includes custom kernels needed to get HCSA/Wayfinder working correctly on that model family.

The present engineering objective is to keep prefill speed roughly consistent as context length increases. That objective is still under investigation and should not be read as a validated performance claim.

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

## Try it now (interactive chat)

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

## Quick start (5 minutes)

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

## First successful run

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

## Stable public profile (default)

```bash
./scripts/run_public_stable_profile_glm.sh
```

Output artifacts:
- `<out-root>/<run-id>/dense/results.json`
- `<out-root>/<run-id>/wayfinder/results.json`
- `<out-root>/<run-id>/stable_profile_summary.json`
- `<out-root>/<run-id>/stable_profile_summary.md`

## Support matrix

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

- `docs/FIRST_RELEASE.md` — release evidence and benchmark artifacts
- `docs/ARCHITECTURE.md` — architecture details
- `docs/RESEARCH.md` — research notes
- `docs/PUBLIC_POSITIONING.md` — positioning
- `docs/RELEASE_GATE.md` — release gate criteria
- `SECURITY.md` — security reporting
- `CONTRIBUTING.md` — contributor guide

## License

MIT. See `LICENSE`.
