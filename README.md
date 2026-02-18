# Wayfinder (HCSA)

## What It Is
Wayfinder implements **Hamiltonian Cycle Sparse Attention (HCSA)**: causal self-attention where each token attends to a bounded neighborhood defined by a graph over positions `0..T-1`.

Core neighborhood ingredients:
- local causal window
- Hamiltonian-cycle backbone
- optional landmarks/rewires

Execution targets:
- PyTorch path
- MLX path

Graph ABI (`hcsa/graph/abi.py`):
- `neigh_idx`: padded `int32` adjacency list, `-1` for padding, shape `[T, D]` or `[H, T, D]`
- `edge_type`: `uint8` edge labels in `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`

This first public release is primarily a **GLM-4.7-Flash-4bit stable-path** release.
Nanbeige long-boundary slices remain experimental/non-default.

## Quick Start (5 Minutes)
Use this exact sequence as a new user:

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

Interpretation:
- if verify exits `0` and all three artifacts exist, environment/setup is ready
- if not, fix environment issues before running benchmarks

## First Successful Run
Run a dense sanity benchmark first:

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
- command exits `0`
- `benchmarks/mlx/first_release/first_run_dense_t2048/results.json` exists

## Stable Public Profile (Default)
This is the default public benchmark path.

```bash
./scripts/run_public_stable_profile_glm.sh
```

Optional explicit form:

```bash
./scripts/run_public_stable_profile_glm.sh \
  --run-id EXP-YYYYMMDDTHHMMSSZ-STABLE-PROFILE \
  --out-root benchmarks/mlx/first_release \
  --model-path mlx-community/GLM-4.7-Flash-4bit \
  --seq-len 8192 \
  --decode-len 32 \
  --repeats 1
```

Default behavior:
- strict sequential execution (dense, then wayfinder)
- conservative flags (`--skip-multi-turn --skip-quality`)
- retro/backfill inference default remains off

Output artifacts per run:
- `<out-root>/<run-id>/dense/results.json`
- `<out-root>/<run-id>/wayfinder/results.json`
- `<out-root>/<run-id>/stable_profile_summary.json`
- `<out-root>/<run-id>/stable_profile_summary.md`

## Support Matrix (Validated vs Experimental)
| Tier | Status | Scope | Default | Evidence |
|---|---|---|---|---|
| Validated | Recommended | GLM-4.7 stable wrapper path | Yes | `docs/FIRST_RELEASE.md` |
| Experimental | Opt-in only | Qwen and Nanbeige diagnostic slices | No | `docs/FIRST_RELEASE.md` |
| Known regression | Non-default | Nanbeige `T=131072, decode_len=256` | No | `docs/FIRST_RELEASE.md` |

Additional boundary context:
- Nanbeige `T=131072, decode_len=32` completed with informative fallback diagnostics but remains experimental/non-default.
- Full measured tables, deltas, reproduction commands, and artifact paths are in `docs/FIRST_RELEASE.md`.

## Troubleshooting
If a run fails or quality/perf looks wrong, check these first:

- OOM or heavy memory pressure:
  - re-run `./scripts/verify_install_and_preflight.sh` and inspect swap/compressor deltas
  - reduce `--seq-lens` and retry one command at a time

- Missing artifacts:
  - treat missing `results.json` as failed run
  - do not compute deltas from partial outputs

- Queue dry-run path collisions:
  - `scripts/run_section4_queue.py --dry-run` rejects existing out dirs unless `--overwrite` is used

- Fallback diagnostics unclear:
  - include `--hsa-trace` for diagnostic wayfinder runs
  - if fallback appears but reasons are missing/unspecified, treat as follow-up

- Non-default reminder:
  - keep Nanbeige long-boundary slices (`T=131072`) experimental unless release evidence is updated

## Docs Map
Use these docs by audience:

- Release evidence and reproduction: `docs/FIRST_RELEASE.md`
- Architecture and internals: `docs/ARCHITECTURE.md`
- Research direction and citations: `docs/RESEARCH.md`

Related workflow policy:
- Experiment discipline and ledger protocol: `AGENTS.md`

## License
MIT. See `LICENSE`.
