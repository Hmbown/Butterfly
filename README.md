# Wayfinder (HCSA)

Wayfinder implements **Hamiltonian Cycle Sparse Attention (HCSA)**: causal self-attention where each token attends to a bounded neighborhood defined by a graph over positions `0..T-1` (local window + Hamiltonian cycle + optional landmarks/rewires). The Hamiltonian cycle is a degree-2 backbone; any mixing comes from the union of edges (and optional multiple cycles/rewires) plus composition across layers. Execution targets **PyTorch** and **MLX** via a backend-agnostic Graph ABI.

**Graph ABI** (`hcsa/graph/abi.py`):
- `neigh_idx`: padded `int32` adjacency list; `-1` denotes padding. Shape `[T, D]` or `[H, T, D]`
- `edge_type`: `uint8` edge labels in `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`

## Definition And Cost Model

Let the sequence positions be `0..T-1`. Construct a Hamiltonian cycle `C` over these positions. For token `i`, candidate neighbors are:

1. `W` most recent predecessors (local causal window)
2. the two neighbors of `i` in `C` (cycle backbone)
3. landmark positions at stride `s`

Apply the causal mask `j < i`, yielding a directed sparse neighborhood for token `i`.

**Cost model** (`d` = head/embedding dim, `s` = landmark stride):

Dense causal: per-token fan-in is `T-1`, so total edges are `O(T^2)` and attention work is `O(T^2 d)`.
HCSA: per-token fan-in is bounded by `D <= W + 2 + T/s`; after causal masking on the cycle, typical average is closer to `D ~= W + 1 + T/s`. Total edges are `O(TD)` and attention work is `O(T D d)`.

At `T=4096`, `W=64`, `s=64`: `D=130` vs `4095`, a **31x** reduction in edges per token.

## First-Release Support Matrix (2026-02-18)

| Tier | Status | Scope | Evidence |
|---|---|---|---|
| Validated (default) | Recommended | GLM-4.7-Flash-4bit, consumer benchmark path (`--mode wayfinder`) via stable wrapper | `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_dense32/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_wayfinder32/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_dense256/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_wayfinder256/results.json` |
| Experimental (opt-in) | Not default | Qwen and Nanbeige diagnostic slices, decode-length and q-chunk sweeps | `benchmarks/mlx/qwen3_1_7b_wayfinder/diag_decode_len_sweep_20260217T183144Z/*/results.json`, `benchmarks/mlx/nanbeige4_1_3b_wayfinder/diag_qchunk_sweep_20260217T183144Z/*/results.json` |
| Known regression / non-default | Do not use as public default | Nanbeige `T=131072, decode_len=256` Wayfinder boundary | `EXP-20260218T074600Z-NANBEIGE-INSTRUMENTED-DENSE256-COMPANION-RERUN-RESULT` in `notes/LAB_NOTEBOOK.md`, paired artifacts under `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218/` |

Release focus for 2026-02-18: this is primarily a GLM-4.7-Flash-4bit stable-path release; Qwen/Nanbeige remain opt-in diagnostic paths.

First-release decision and reproduction commands: `docs/FIRST_RELEASE.md`.

## Install

### 5-minute install + verify

```bash
git clone <this-repo> && cd <this-repo>
pip install -e ".[dev]"
pip install -e ".[mlx]"

# optional diagnostics dependencies
pip install -e ".[viz]"

# sequential install + preflight verification
./scripts/verify_install_and_preflight.sh \
  --run-id EXP-YYYYMMDDTXXXXXXZ-VERIFY-INSTALL \
  --out-dir benchmarks/mlx/preflight
```

Expected verification artifacts:
- `benchmarks/mlx/preflight/<RUN_ID>_env_check_mlx.json`
- `benchmarks/mlx/preflight/<RUN_ID>_summary.json`
- `benchmarks/mlx/preflight/<RUN_ID>_raw.txt`

### First successful run (dense sanity)

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

### Stable public profile (safe default)

This is the release-default benchmark path for public users.

```bash
./scripts/run_public_stable_profile_glm.sh
```

Optional overrides:

```bash
./scripts/run_public_stable_profile_glm.sh \
  --run-id EXP-YYYYMMDDTXXXXXXZ-STABLE-PROFILE \
  --out-root benchmarks/mlx/first_release \
  --model-path mlx-community/GLM-4.7-Flash-4bit \
  --seq-len 8192 \
  --decode-len 32 \
  --repeats 1
```

The wrapper runs dense then wayfinder sequentially and writes:
- `<out-root>/<run-id>/dense/results.json`
- `<out-root>/<run-id>/wayfinder/results.json`
- `<out-root>/<run-id>/stable_profile_summary.json`
- `<out-root>/<run-id>/stable_profile_summary.md`

### Optional standalone preflight verification

Run this before any benchmark campaign to verify script entrypoints and capture host memory snapshots (swap/compressor):

```bash
./scripts/bench_protocol_preflight_setup.sh \
  --run-id EXP-YYYYMMDDTXXXXXXZ-BENCH-PROTOCOL \
  --out-dir benchmarks/mlx/preflight
```

### Environment health check

Record MLX/package/system readiness into the same run tree used by benchmark artifacts:

```bash
python3 scripts/env_check_mlx.py \
  --json-out benchmarks/mlx/<run-tag>/env_check_mlx.json
```

## Quickstart

Run tests:

```bash
pytest
```

Tiny train (PyTorch core):

```bash
python3 -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char \
  --attn hcsa --cycle random --window 32 --landmark-stride 32 --steps 200
```

MLX scaling benchmark:

```bash
python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32
```

## Workflow (Steps)

1. Propose: write a hypothesis in `notes/LAB_NOTEBOOK.md` (required for benchmarks).
2. Change: edit code/config (often `configs/graph_specs/*.wf`).
3. Benchmark: run a targeted script under `scripts/` (or `python3 scripts/wayc.py bench` for a tiny MLX microbench).
4. Record: append results to `notes/experiments.ndjson`.
5. Decide: keep/revert/follow-up in `notes/LAB_NOTEBOOK.md`.

Protocol details: see `AGENTS.md`.

### Benchmark execution protocol (sequential only)

Use one process at a time to avoid OOM and preserve reproducibility:

1. Run preflight and env-check commands above before launching benchmarks.
2. Execute baseline and variant commands strictly sequentially (for example: dense -> wayfinder -> sweeps). Do not overlap runs.
3. After each run, enforce stop-gates before starting the next:
   - command exit code must be zero and `results.json` must exist
   - fallback diagnostics must be present (`dense_fallback_reason_counts` and related share fields)
   - throughput/memory gates in the corresponding `notes/experiments.ndjson` PRERUN entry must pass
4. If any gate fails, stop the queue, record the failure in `notes/LAB_NOTEBOOK.md` and `notes/experiments.ndjson`, then adjust one variable at a time.

For the built-in sequential queue runner with stop-gates, use PRERUN ids from `notes/experiments.ndjson`:

```bash
python3 scripts/run_section4_queue.py \
  --nanbeige-exp-id <NANBEIGE_PRERUN_ID> \
  --qwen-exp-id <QWEN_PRERUN_ID> \
  --timeout-sec 1800 \
  --dry-run
# remove --dry-run to execute; add --overwrite only when intentionally reusing out_dir paths
```

### Troubleshooting

- OOM or swap growth:
  - run `./scripts/verify_install_and_preflight.sh --run-id <ID> --out-dir benchmarks/mlx/preflight`
  - check swap/compressor deltas in `<RUN_ID>_summary.json`
  - reduce `--seq-lens` and retry one command at a time.
- Fallback diagnostics missing:
  - ensure Wayfinder runs include `--hsa-trace` when diagnosing path behavior.
  - stop and record if `dense_fallback_reason_counts` is empty or unspecified.
- Queue dry-run fails on existing paths:
  - `scripts/run_section4_queue.py --dry-run` refuses reused out dirs unless `--overwrite` is passed.
- Missing artifacts:
  - every completed run should have `<out-dir>/results.json`; if absent, treat as failed run and do not compute deltas.
- Known non-default slice:
  - Nanbeige `T=131072, decode_len=256` currently regresses vs dense (`+44.63%` e2e, `-82.24%` decode tok/s, `+51.25%` memory); keep this slice experimental/non-default.

## Benchmarks

All numbers: Apple M4 Max, MLX backend, 4-bit weights, `W=64`, chunk=`4096`. Measured Feb 16-18, 2026.

### GLM-4.7-Flash-4bit - full model (47 layers, MoE)

End-to-end chunked prefill through all 47 layers of a production MoE model (9B total, ~4B active).

**decode_len=32 (short decode):**

| T | Dense e2e (s) | Wayfinder e2e (s) | Delta | Dense peak | Wayfinder peak | Mem Δ |
|---|----------:|----------:|------:|--------:|--------:|------:|
| 2,048 | 3.22 | 2.58 | **-19.76%** | 18.28 GB | 18.32 GB | -0.19% |
| 8,192 | 16.94 | 9.66 | **-42.94%** | 20.66 GB | 20.16 GB | **+2.42%** |
| 32,768 | 282.95 | 112.31 | **-60.31%** | 26.02 GB | 21.98 GB | **+15.52%** |
| 65,536 | 1156.53 | 727.04 | **-37.14%** | 33.16 GB | 23.80 GB | **+28.24%** |

**decode_len=256 (long decode):**

| T | Dense e2e (s) | Wayfinder e2e (s) | Delta | Dense peak | Wayfinder peak | Mem Δ |
|---|----------:|----------:|------:|--------:|--------:|------:|
| 2,048 | 6.64 | 6.33 | **-4.66%** | 18.28 GB | 18.32 GB | -0.19% |
| 8,192 | 21.32 | 14.30 | **-32.95%** | 20.66 GB | 20.16 GB | **+2.42%** |
| 32,768 | 379.13 | 120.95 | **-68.10%** | 26.02 GB | 21.98 GB | **+15.52%** |
| 65,536 | 1469.30 | 430.77 | **-70.68%** | 33.16 GB | 23.80 GB | **+28.24%** |

Memory reduction convention: `100 * (1 - wayfinder/dense)`.

**8W Active Contiguous Path:** When a chunked-prefill active block spans at least half the KV cache (`2*Tq >= Tk`), the active-row dispatch uses the contiguous full-prefill path instead of scattered gathers. The SDPA chunk size is `8W` (window-proportional), giving each block a K dimension of `10W` with 20% fan-out overhead. This eliminates all `mx.eval` barriers and random-access gathers in the critical second-chunk case, making sparse attention competitive at intermediate scales (8k) without any dense fallback.

Artifacts: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_dense32/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_wayfinder32/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_dense256/results.json`, `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/8w_fix_wayfinder256/results.json`

### Qwen3-4B-4bit - isolated attention (single layer)

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 241,600 | 309,100 | **1.28x** |
| 8,192 | 176,000 | 313,600 | **1.78x** |
| 16,384 | 113,700 | 314,800 | **2.77x** |

### Qwen3-4B-4bit - full transformer block (attention + MLP + norms)

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 85,200 | 92,000 | **1.08x** |
| 8,192 | 76,300 | 94,000 | **1.23x** |
| 16,384 | 61,300 | 93,700 | **1.53x** |

### GPT-2 (MLX) - isolated attention

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 790,000 | 1,020,000 | **1.30x** |
| 8,192 | 500,000 | 1,050,000 | **2.12x** |
| 16,384 | 280,000 | 1,090,000 | **3.87x** |

Raw data: `results/`, `benchmarks/`

### Nanbeige boundary telemetry (experimental / non-default)

Latest completed paired boundary at `T=131072, decode_len=256` (instrumented):

| Metric | Dense | Wayfinder | Delta (wayfinder vs dense) |
|---|---:|---:|---:|
| e2e sec | 404.079 | 584.437 | +44.63% |
| prefill sec | 375.899 | 425.757 | +13.26% |
| decode sec | 28.181 | 158.680 | +463.08% |
| decode tok/s | 9.084 | 1.613 | -82.24% |
| peak memory bytes | 18,460,513,312 | 27,922,247,416 | +51.25% |

Fallback diagnostics for this slice:
- `path_counts={permute:2056, permute_dense_fallback:248}`
- `dense_fallback_reason_counts={active_large_q:248}`
- `dense_fallback_share_run=0.1076`

Source ledger: `EXP-20260218T074600Z-NANBEIGE-INSTRUMENTED-DENSE256-COMPANION-RERUN-RESULT`.

Remaining boundary gate closure at `T=131072, decode_len=32` (instrumented, completed 2026-02-18):

| Metric | Dense | Wayfinder | Delta (wayfinder vs dense) |
|---|---:|---:|---:|
| e2e sec | 467.110 | 471.744 | +0.99% |
| prefill sec | 435.884 | 425.229 | -2.44% |
| decode sec | 31.225 | 46.515 | +48.97% |
| decode tok/s | 1.025 | 0.688 | -32.87% |
| peak memory bytes | 18,460,513,312 | 18,474,229,212 | +0.07% |

Fallback diagnostics for this slice:
- `path_counts={permute:264, permute_dense_fallback:248}`
- `dense_fallback_reason_counts={active_large_q:248}`
- `dense_fallback_share_run=0.4844`

Decision: this boundary remains experimental/non-default.

## How It's Fast

Three optimizations combine to make HCSA sparse attention faster than dense on real models:

1. **Vectorized all-head dispatch** (`hcsa/mlx/fused_attention.py`): Processes all query heads simultaneously via flat-index arithmetic + MLX's fused SDPA kernel, eliminating per-head Python loops and eval barriers. Two code paths: contiguous-window (when graph matches data) and flat-index gather (oversized graphs).

2. **Fast perms-only graph build** (`hcsa/topology/core.py`): `Topology.construct_perms_only()` generates only cycle permutations in O(T) time, skipping the O(T*D) neighbor-index construction that is unused by the permute attention path. **1904x faster** at T=32,768 (24s to 12ms).

3. **Exact graph horizon** (`hcsa/integrations/glm_mlx.py`): Builds graphs at exactly the KV cache size (Tg=Tk) instead of rounding up to a power-of-two horizon. This enables the contiguous-window active-row path where pre-permuted K/V can be sliced without random-access gathers.

## Visuals

Dense attention (left) vs HCSA connectivity (right) at `T=64`, `W=8`, `s=16`:

![Dense vs HCSA attention matrices](docs/assets/attention_comparison.png)

HCSA graph on 32 tokens (circle layout; only causal edges shown):

![HCSA graph circle layout](docs/assets/hcsa_graph_circle.png)

## Project Map

| Path | Purpose |
|---|---|
| `hcsa/attention_hcsa.py`, `hcsa/model.py` | Core sparse attention + reference GPT |
| `hcsa/cycles.py`, `hcsa/graph_strategies.py` | Cycle construction + strategy wrappers |
| `hcsa/graph/abi.py` | Graph ABI (neighbor indices + edge typing) |
| `hcsa/graph/analysis.py` | Empirical diagnostics: spectral gap, random-walk mixing, resilience |
| `hcsa/topology/core.py` | Topology runtime (construct/save/load/rewire/`construct_perms_only`) |
| `hcsa/compiler/` + `configs/graph_specs/*.wf` | Graph-spec compiler + cache artifacts |
| `hcsa/mlx/attention.py` | MLX attention dispatch (dense, sparse-gather, permute-window) |
| `hcsa/mlx/fused_attention.py` | Vectorized all-head fused dispatch (full-prefill + active-row) |
| `hcsa/integrations/` | Model integrations (Qwen3, GLM-4, GPT-2) |
| `hcsa/torch/` | PyTorch/CUDA backend |
| `scripts/` | Benchmarks, training, ablations, visualization |
| `tests/` | 240 tests covering correctness + diagnostics |
| `benchmarks/`, `results/` | Experiment results + benchmark data |

## Research Questions

Can sparse attention learn like a slime mold - start with a structured backbone and discover which edges carry information at every scale?

Practical version:
- Start from an overcomplete candidate graph (window + landmarks + `k` cycles/rewires).
- Learn a degree-budgeted subgraph (per-token/head top-`D` edges) via gated edges or periodic prune/rewire using usage/gradient credit.
- Measure: (1) long-context quality, (2) layer depth needed for near-full-prefix receptive fields, (3) end-to-end throughput and peak memory at fixed `D`.

Related structural question: does multiscale cycle richness of the undirected skeleton predict the depth needed for full-prefix receptive fields under causal composition, beyond spectral gap alone?

## References

Sparse/structured attention and graph-based sparsity:
- BigBird: https://arxiv.org/abs/2007.14062
- Exphormer: https://arxiv.org/abs/2303.06147
- FlashInfer: https://arxiv.org/abs/2501.01005
- Flex Attention: https://arxiv.org/abs/2412.05496

## License

MIT. See `LICENSE`.
