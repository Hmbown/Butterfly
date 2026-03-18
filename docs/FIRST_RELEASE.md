# Benchmark Evidence (2026-02-18)

Primary validated path: GLM-4.7-Flash-4bit on MLX (Apple Silicon).

## Default Profile

- Stable command: `./scripts/run_public_stable_profile_glm.sh`
- Model default: `mlx-community/GLM-4.7-Flash-4bit`
- Runtime defaults: `seq_len=8192`, `decode_len=32`, `repeats=1`
- Conservative flags: `--skip-multi-turn --skip-quality`
- Retro/backfill inference default remains off.
- Execution policy: run one benchmark/inference command at a time.

### Validated evidence (2026-02-18)

Stable wrapper run `EXP-20260218T151213Z-STABLE-PROFILE`:

- Dense: `e2e=17.1473s`, `prefill=16.3586s`, `decode=0.7886s`, `decode_tok_s=40.5762`, `peak_memory=20,660,500,140`
- Wayfinder: `e2e=10.5563s`, `prefill=9.7533s`, `decode=0.8030s`, `decode_tok_s=39.8499`, `peak_memory=20,071,482,232`
- Delta (wayfinder vs dense):
  - `e2e=-38.44%`
  - `prefill=-40.38%`
  - `decode=+1.82%`
  - `decode_tok_s=-1.79%`
  - `peak_memory=-2.85%` (wayfinder lower)
- Memory convention: `100*(1-wayfinder/dense)=+2.8509%` (reduction)

Artifacts:
- `benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/dense/results.json`
- `benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/wayfinder/results.json`
- `benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/stable_profile_summary.json`
- `benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/stable_profile_summary.md`

### Experimental: Nanbeige boundary (2026-02-18)

Nanbeige instrumented pair `T=131072, decode_len=32`: 

- Dense: `e2e=467.1097s`, `prefill=435.8845s`, `decode=31.2253s`, `decode_tok_s=1.0248`, `peak_memory=18,460,513,312`
- Wayfinder trace: `e2e=471.7444s`, `prefill=425.2292s`, `decode=46.5151s`, `decode_tok_s=0.6879`, `peak_memory=18,474,229,212`
- Delta (wayfinder vs dense):
  - `e2e=+0.99%`
  - `prefill=-2.44%`
  - `decode=+48.97%`
  - `decode_tok_s=-32.87%`
  - `peak_memory=+0.07%`
- Memory convention: `100*(1-wayfinder/dense)=-0.0743%` (slight increase)
- Fallback diagnostics: `path_counts={permute:264, permute_dense_fallback:248}`, `dense_fallback_reason_counts={active_large_q:248}`, `dense_fallback_share_run=0.484375`

Decision: keep this long-boundary Nanbeige path experimental/non-default.

Artifacts:
- `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218T151213Z/instrumented_trace32_T131072/results.json`
- `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218T151213Z/instrumented_dense32_T131072/results.json`

### Known regression (2026-02-18)

Nanbeige `T=131072, decode_len=256` paired instrumented result:

- `e2e=+44.63%`
- `prefill=+13.26%`
- `decode=+463.08%`
- `decode_tok_s=-82.24%`
- `peak_memory=+51.25%`

Decision: non-default, do not promote.

Artifacts:
- `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218/instrumented_wayfinder256_T131072/results.json`
- `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218/instrumented_dense256_T131072_rerun/results.json`

## Exact Reproduction Commands

Run sequentially, one command at a time.

```bash
./scripts/verify_install_and_preflight.sh --run-id EXP-YYYYMMDDTHHMMSSZ-VERIFY-INSTALL --out-dir benchmarks/mlx/preflight

python3 scripts/bench_glm_consumer_mlx.py --mode dense --seq-lens 2048 --decode-len 8 --repeats 1 --skip-multi-turn --skip-quality --out-dir benchmarks/mlx/first_release/first_run_dense_t2048

./scripts/run_public_stable_profile_glm.sh
```

Nanbeige boundary instrumentation pair used for gate closure:

```bash
python3 scripts/bench_qwen_consumer_mlx.py --model-path Nanbeige/Nanbeige4.1-3B --mode wayfinder --seq-lens 131072 --decode-len 32 --repeats 1 --skip-multi-turn --skip-quality --hsa-trace --window 64 --head-chunk-size 2 --query-chunk-size 384 --stage-timeout-sec 900 --heartbeat-sec 30 --out-dir benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218T151213Z/instrumented_trace32_T131072

python3 scripts/bench_qwen_consumer_mlx.py --model-path Nanbeige/Nanbeige4.1-3B --mode dense --seq-lens 131072 --decode-len 32 --repeats 1 --skip-multi-turn --skip-quality --stage-timeout-sec 900 --heartbeat-sec 30 --out-dir benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_20260218T151213Z/instrumented_dense32_T131072
```

## Artifact Layout

- Install verify:
  - `<out_dir>/<RUN_ID>_env_check_mlx.json`
  - `<out_dir>/<RUN_ID>_summary.json`
  - `<out_dir>/<RUN_ID>_raw.txt`
- Stable profile:
  - `benchmarks/mlx/first_release/<RUN_ID>/dense/results.json`
  - `benchmarks/mlx/first_release/<RUN_ID>/wayfinder/results.json`
  - `benchmarks/mlx/first_release/<RUN_ID>/stable_profile_summary.json`
  - `benchmarks/mlx/first_release/<RUN_ID>/stable_profile_summary.md`
- Boundary diagnostics:
  - `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_<TS>/instrumented_trace32_T131072/results.json`
  - `benchmarks/mlx/nanbeige4_1_3b_wayfinder/hang_debug_<TS>/instrumented_dense32_T131072/results.json`

## Decision Rules

- If command exit is nonzero, missing `results.json`, or missing usable `single_turn` row: stop and classify as follow-up.
- If fallback is observed and reason counts are missing/unspecified: stop and classify as follow-up.
- If dense-relative regressions are material or unresolved at boundary slices: keep experimental/non-default.
- Only validated slices are eligible for default commands.

## Where Wayfinder Helps / Does Not Help

- Helps (validated default path): GLM stable profile at `T=8192, decode_len=32` with large e2e/prefill gains and lower memory (2026-02-18 run above).
- Does not help (current non-default boundaries): Nanbeige `T=131072` long-boundary slices remain decode-limited and fallback-heavy; `decode_len=256` is a confirmed regression and `decode_len=32` still loses decode throughput.
