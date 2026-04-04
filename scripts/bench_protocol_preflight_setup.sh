#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BUTTERFLY_PYTHON:-${ROOT_DIR}/.venv-macos-metal/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${ALT_PYTHON_BIN:-python3}"
fi
RUN_ID=""
OUT_DIR_ARG="notes/preflight"

usage() {
  cat <<'EOF'
Usage: bench_protocol_preflight_setup.sh [--run-id ID] [--out-dir PATH]

Runs setup-only benchmark preflight checks without inference:
- verifies required benchmark script files exist
- runs CLI help checks only
- captures pre/post host memory snapshots (swap + compressor)

Outputs:
- ${out_dir}/${run_id}_summary.json
- ${out_dir}/${run_id}_raw.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR_ARG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="EXP-$(date -u +%Y%m%dT%H%M%SZ)-BENCH-PROTOCOL-SETUP"
fi

if [[ "$OUT_DIR_ARG" = /* ]]; then
  OUT_DIR="$OUT_DIR_ARG"
else
  OUT_DIR="$ROOT_DIR/$OUT_DIR_ARG"
fi
mkdir -p "$OUT_DIR"

START_TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

capture_snapshot() {
  local prefix="$1"
  local swap_line
  local vmstat
  local page_size
  local comp_pages
  local comp_stored
  local swap_used
  local swap_free

  swap_line="$(sysctl -n vm.swapusage 2>/dev/null || true)"
  vmstat="$(vm_stat 2>/dev/null || true)"
  page_size="$(printf '%s\n' "$vmstat" | awk '/page size of/ {for (i=1; i<=NF; i++) {if ($i ~ /[0-9]+/) {gsub("[^0-9]","",$i); print $i; exit}}}')"
  if [[ -z "$page_size" ]]; then
    page_size="$(sysctl -n hw.pagesize 2>/dev/null || echo 0)"
  fi
  comp_pages="$(printf '%s\n' "$vmstat" | awk -F: '/Pages occupied by compressor/ {gsub("[^0-9]","",$2); print $2; exit}')"
  comp_stored="$(printf '%s\n' "$vmstat" | awk -F: '/Pages stored in compressor/ {gsub("[^0-9]","",$2); print $2; exit}')"
  swap_used="$(printf '%s\n' "$swap_line" | sed -E 's/.*used = ([0-9.]+)M.*/\1/' || true)"
  swap_free="$(printf '%s\n' "$swap_line" | sed -E 's/.*free = ([0-9.]+)M.*/\1/' || true)"

  [[ "$swap_used" =~ ^[0-9]+([.][0-9]+)?$ ]] || swap_used="0"
  [[ "$swap_free" =~ ^[0-9]+([.][0-9]+)?$ ]] || swap_free="0"
  [[ "$page_size" =~ ^[0-9]+$ ]] || page_size="0"
  [[ "$comp_pages" =~ ^[0-9]+$ ]] || comp_pages="0"
  [[ "$comp_stored" =~ ^[0-9]+$ ]] || comp_stored="0"

  printf -v "${prefix}_SWAP_LINE" '%s' "$swap_line"
  printf -v "${prefix}_SWAP_USED_MB" '%s' "$swap_used"
  printf -v "${prefix}_SWAP_FREE_MB" '%s' "$swap_free"
  printf -v "${prefix}_PAGE_SIZE_BYTES" '%s' "$page_size"
  printf -v "${prefix}_COMPRESSOR_OCCUPIED_PAGES" '%s' "$comp_pages"
  printf -v "${prefix}_COMPRESSOR_STORED_PAGES" '%s' "$comp_stored"
}

check_file_status() {
  local relative_path="$1"
  if [[ -f "$ROOT_DIR/$relative_path" ]]; then
    printf 'ok'
  else
    printf 'missing'
  fi
}

check_help_status() {
  if "$@" >/dev/null 2>&1; then
    printf 'ok'
  else
    printf 'fail'
  fi
}

capture_snapshot "PRE"

FILE_BENCH_PY="$(check_file_status "scripts/bench.py")"
FILE_BENCH_MLX_SCALE_PY="$(check_file_status "scripts/bench_mlx_wayfinder_scale.py")"
FILE_BENCH_QWEN_WAYFINDER_PY="$(check_file_status "scripts/bench_qwen_wayfinder_mlx.py")"
FILE_BENCH_GLM_WAYFINDER_PY="$(check_file_status "scripts/bench_glm_wayfinder_mlx.py")"
FILE_WAYC_PY="$(check_file_status "scripts/wayc.py")"

HELP_BENCH_PY="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/bench.py" --help)"
HELP_BENCH_MLX_SCALE_PY="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/bench_mlx_wayfinder_scale.py" --help)"
HELP_BENCH_QWEN_WAYFINDER_PY="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/bench_qwen_wayfinder_mlx.py" --help)"
HELP_BENCH_GLM_WAYFINDER_PY="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/bench_glm_wayfinder_mlx.py" --help)"
HELP_WAYC_PY="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/wayc.py" --help)"
HELP_WAYC_BENCH="$(check_help_status "$PYTHON_BIN" "$ROOT_DIR/scripts/wayc.py" bench --help)"

capture_snapshot "POST"
END_TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

OVERALL_STATUS="pass"
for status in \
  "$FILE_BENCH_PY" \
  "$FILE_BENCH_MLX_SCALE_PY" \
  "$FILE_BENCH_QWEN_WAYFINDER_PY" \
  "$FILE_BENCH_GLM_WAYFINDER_PY" \
  "$FILE_WAYC_PY"; do
  if [[ "$status" != "ok" ]]; then
    OVERALL_STATUS="fail"
  fi
done

for status in \
  "$HELP_BENCH_PY" \
  "$HELP_BENCH_MLX_SCALE_PY" \
  "$HELP_BENCH_QWEN_WAYFINDER_PY" \
  "$HELP_BENCH_GLM_WAYFINDER_PY" \
  "$HELP_WAYC_PY" \
  "$HELP_WAYC_BENCH"; do
  if [[ "$status" != "ok" ]]; then
    OVERALL_STATUS="fail"
  fi
done

DELTA_SWAP_USED_MB="$(awk -v post="$POST_SWAP_USED_MB" -v pre="$PRE_SWAP_USED_MB" 'BEGIN { printf "%.2f", post-pre }')"
DELTA_SWAP_FREE_MB="$(awk -v post="$POST_SWAP_FREE_MB" -v pre="$PRE_SWAP_FREE_MB" 'BEGIN { printf "%.2f", post-pre }')"
DELTA_COMPRESSOR_OCCUPIED_PAGES=$((POST_COMPRESSOR_OCCUPIED_PAGES - PRE_COMPRESSOR_OCCUPIED_PAGES))

PRE_COMPRESSOR_OCCUPIED_BYTES=$((PRE_COMPRESSOR_OCCUPIED_PAGES * PRE_PAGE_SIZE_BYTES))
POST_COMPRESSOR_OCCUPIED_BYTES=$((POST_COMPRESSOR_OCCUPIED_PAGES * POST_PAGE_SIZE_BYTES))
DELTA_COMPRESSOR_OCCUPIED_BYTES=$((DELTA_COMPRESSOR_OCCUPIED_PAGES * POST_PAGE_SIZE_BYTES))

SUMMARY_FILE="$OUT_DIR/${RUN_ID}_summary.json"
RAW_FILE="$OUT_DIR/${RUN_ID}_raw.txt"

cat >"$SUMMARY_FILE" <<EOF
{
  "id": "$RUN_ID",
  "started_at_utc": "$START_TS_UTC",
  "ended_at_utc": "$END_TS_UTC",
  "status": "$OVERALL_STATUS",
  "checks": {
    "files": {
      "scripts/bench.py": "$FILE_BENCH_PY",
      "scripts/bench_mlx_wayfinder_scale.py": "$FILE_BENCH_MLX_SCALE_PY",
      "scripts/bench_qwen_wayfinder_mlx.py": "$FILE_BENCH_QWEN_WAYFINDER_PY",
      "scripts/bench_glm_wayfinder_mlx.py": "$FILE_BENCH_GLM_WAYFINDER_PY",
      "scripts/wayc.py": "$FILE_WAYC_PY"
    },
    "help": {
      "bench.py --help": "$HELP_BENCH_PY",
      "bench_mlx_wayfinder_scale.py --help": "$HELP_BENCH_MLX_SCALE_PY",
      "bench_qwen_wayfinder_mlx.py --help": "$HELP_BENCH_QWEN_WAYFINDER_PY",
      "bench_glm_wayfinder_mlx.py --help": "$HELP_BENCH_GLM_WAYFINDER_PY",
      "wayc.py --help": "$HELP_WAYC_PY",
      "wayc.py bench --help": "$HELP_WAYC_BENCH"
    }
  },
  "metrics": {
    "pre_run": {
      "swap_used_mb": $PRE_SWAP_USED_MB,
      "swap_free_mb": $PRE_SWAP_FREE_MB,
      "pages_occupied_by_compressor": $PRE_COMPRESSOR_OCCUPIED_PAGES,
      "page_size_bytes": $PRE_PAGE_SIZE_BYTES,
      "compressor_occupied_bytes": $PRE_COMPRESSOR_OCCUPIED_BYTES
    },
    "post_run": {
      "swap_used_mb": $POST_SWAP_USED_MB,
      "swap_free_mb": $POST_SWAP_FREE_MB,
      "pages_occupied_by_compressor": $POST_COMPRESSOR_OCCUPIED_PAGES,
      "page_size_bytes": $POST_PAGE_SIZE_BYTES,
      "compressor_occupied_bytes": $POST_COMPRESSOR_OCCUPIED_BYTES
    },
    "safety_deltas": {
      "swap_used_mb": $DELTA_SWAP_USED_MB,
      "swap_free_mb": $DELTA_SWAP_FREE_MB,
      "pages_occupied_by_compressor": $DELTA_COMPRESSOR_OCCUPIED_PAGES,
      "compressor_occupied_bytes": $DELTA_COMPRESSOR_OCCUPIED_BYTES
    }
  },
  "artifacts": {
    "raw": "$RAW_FILE"
  }
}
EOF

cat >"$RAW_FILE" <<EOF
id=$RUN_ID
started_at_utc=$START_TS_UTC
ended_at_utc=$END_TS_UTC
status=$OVERALL_STATUS
file_check_scripts_bench_py=$FILE_BENCH_PY
file_check_scripts_bench_mlx_wayfinder_scale_py=$FILE_BENCH_MLX_SCALE_PY
file_check_scripts_bench_qwen_wayfinder_mlx_py=$FILE_BENCH_QWEN_WAYFINDER_PY
file_check_scripts_bench_glm_wayfinder_mlx_py=$FILE_BENCH_GLM_WAYFINDER_PY
file_check_scripts_wayc_py=$FILE_WAYC_PY
help_check_bench_py=$HELP_BENCH_PY
help_check_bench_mlx_wayfinder_scale_py=$HELP_BENCH_MLX_SCALE_PY
help_check_bench_qwen_wayfinder_mlx_py=$HELP_BENCH_QWEN_WAYFINDER_PY
help_check_bench_glm_wayfinder_mlx_py=$HELP_BENCH_GLM_WAYFINDER_PY
help_check_wayc_py=$HELP_WAYC_PY
help_check_wayc_bench=$HELP_WAYC_BENCH
pre_swap_line=$PRE_SWAP_LINE
pre_swap_used_mb=$PRE_SWAP_USED_MB
pre_swap_free_mb=$PRE_SWAP_FREE_MB
pre_page_size_bytes=$PRE_PAGE_SIZE_BYTES
pre_pages_occupied_by_compressor=$PRE_COMPRESSOR_OCCUPIED_PAGES
pre_pages_stored_in_compressor=$PRE_COMPRESSOR_STORED_PAGES
post_swap_line=$POST_SWAP_LINE
post_swap_used_mb=$POST_SWAP_USED_MB
post_swap_free_mb=$POST_SWAP_FREE_MB
post_page_size_bytes=$POST_PAGE_SIZE_BYTES
post_pages_occupied_by_compressor=$POST_COMPRESSOR_OCCUPIED_PAGES
post_pages_stored_in_compressor=$POST_COMPRESSOR_STORED_PAGES
delta_swap_used_mb=$DELTA_SWAP_USED_MB
delta_swap_free_mb=$DELTA_SWAP_FREE_MB
delta_pages_occupied_by_compressor=$DELTA_COMPRESSOR_OCCUPIED_PAGES
delta_compressor_occupied_bytes=$DELTA_COMPRESSOR_OCCUPIED_BYTES
EOF

echo "RUN_ID=$RUN_ID"
echo "STATUS=$OVERALL_STATUS"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "RAW_FILE=$RAW_FILE"
echo "PRE_SWAP_USED_MB=$PRE_SWAP_USED_MB"
echo "PRE_SWAP_FREE_MB=$PRE_SWAP_FREE_MB"
echo "PRE_PAGES_OCCUPIED_BY_COMPRESSOR=$PRE_COMPRESSOR_OCCUPIED_PAGES"
echo "POST_SWAP_USED_MB=$POST_SWAP_USED_MB"
echo "POST_SWAP_FREE_MB=$POST_SWAP_FREE_MB"
echo "POST_PAGES_OCCUPIED_BY_COMPRESSOR=$POST_COMPRESSOR_OCCUPIED_PAGES"
echo "DELTA_SWAP_USED_MB=$DELTA_SWAP_USED_MB"
echo "DELTA_SWAP_FREE_MB=$DELTA_SWAP_FREE_MB"
echo "DELTA_PAGES_OCCUPIED_BY_COMPRESSOR=$DELTA_COMPRESSOR_OCCUPIED_PAGES"
