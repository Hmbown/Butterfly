#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$REPO_ROOT/benchmarks/mlx/post_reboot_20260211_20260211T202821Z}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${RUN_ROOT:-$BASE_RUN_ROOT/quality_localization_diag_${STAMP}}"
MODEL_PATH="${MODEL_PATH:-mlx-community/GLM-4.7-Flash-4bit}"
QUALITY_DATASET="${QUALITY_DATASET:-$REPO_ROOT/benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/quality_eval_glm47_consumer_v1.json}"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --seq-lens 8192
  --decode-len 64
  --repeats 1
  --chunk-size 4096
  --kv-step 4096
  --cooldown-sec 0
  --path permute
  --window 64
  --landmark-stride 0
  --head-chunk-size 2
  --query-chunk-size 192
  --active-dense-threshold 0
  --seed 42
  --quality-dataset "$QUALITY_DATASET"
  --skip-single-turn
  --skip-multi-turn
)

TRACE_ARGS=(
  --seq-lens 8192
  --decode-len 32
  --repeats 1
  --chunk-size 4096
  --kv-step 4096
  --cooldown-sec 0
  --path permute
  --window 64
  --landmark-stride 0
  --head-chunk-size 2
  --query-chunk-size 192
  --active-dense-threshold 0
  --seed 42
  --quality-dataset "$QUALITY_DATASET"
  --quality-task-id-filter extract-01
  --trace-quality-task-id extract-01
  --trace-topk 8
  --trace-max-steps 16
  --skip-single-turn
  --skip-multi-turn
)

mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"

python3 "$REPO_ROOT/scripts/bench_glm_consumer_mlx.py" \
  "${COMMON_ARGS[@]}" \
  --swap-first-n-layers 8 \
  --out-dir "$RUN_ROOT/layer_first8"

python3 "$REPO_ROOT/scripts/bench_glm_consumer_mlx.py" \
  "${COMMON_ARGS[@]}" \
  --swap-last-n-layers 8 \
  --out-dir "$RUN_ROOT/layer_last8"

python3 "$REPO_ROOT/scripts/bench_glm_consumer_mlx.py" \
  --model-path "$MODEL_PATH" \
  "${TRACE_ARGS[@]}" \
  --no-swap \
  --out-dir "$RUN_ROOT/trace_dense_extract01"

python3 "$REPO_ROOT/scripts/bench_glm_consumer_mlx.py" \
  --model-path "$MODEL_PATH" \
  "${TRACE_ARGS[@]}" \
  --out-dir "$RUN_ROOT/trace_wayfinder_extract01"

python3 "$REPO_ROOT/scripts/analyze_glm_drift_followup.py" localization-summary \
  --baseline-dense "$BASE_RUN_ROOT/consumer_dense_quality/results.json" \
  --baseline-wayfinder "$BASE_RUN_ROOT/consumer_wayfinder_quality/results.json" \
  --candidate first8="$RUN_ROOT/layer_first8/results.json" \
  --candidate last8="$RUN_ROOT/layer_last8/results.json" \
  --out "$RUN_ROOT/layer_localization_summary.json"

python3 "$REPO_ROOT/scripts/analyze_glm_drift_followup.py" trace-diff \
  --dense "$RUN_ROOT/trace_dense_extract01/results.json" \
  --wayfinder "$RUN_ROOT/trace_wayfinder_extract01/results.json" \
  --task-id extract-01 \
  --out "$RUN_ROOT/extract01_trace_diff.json"

echo "Artifacts:"
echo "  $RUN_ROOT/layer_first8/results.json"
echo "  $RUN_ROOT/layer_last8/results.json"
echo "  $RUN_ROOT/trace_dense_extract01/results.json"
echo "  $RUN_ROOT/trace_wayfinder_extract01/results.json"
echo "  $RUN_ROOT/layer_localization_summary.json"
echo "  $RUN_ROOT/extract01_trace_diff.json"
