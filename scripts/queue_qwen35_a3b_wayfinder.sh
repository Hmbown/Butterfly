#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

execute=0
run_id="EXP-$(date -u +%Y%m%dT%H%M%SZ)-QWEN35A3B-CONSUMER-QUEUE"
model_path="/Volumes/VIXinSSD/models/Qwen3.5-35B-A3B-MLX-4bit"
hf_home="/Volumes/VIXinSSD/hf_cache"
hf_hub_cache=""

usage() {
  cat <<'USAGE'
Usage: queue_qwen35_a3b_wayfinder.sh [options]

No-inference default: prints the queued commands only.
Use --execute to run the queue sequentially.

Options:
  --execute                Execute the queue (runs model benchmarks).
  --run-id ID              Override run id used in output paths.
  --model-path PATH        Override local model path.
  --hf-home PATH           Hugging Face cache root (default: /Volumes/VIXinSSD/hf_cache).
  --hf-hub-cache PATH      Hugging Face hub cache dir (default: <hf-home>/hub).
  -h, --help               Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute=1
      shift
      ;;
    --run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-id" >&2; exit 2; }
      run_id="$2"
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --model-path" >&2; exit 2; }
      model_path="$2"
      shift 2
      ;;
    --hf-home)
      [[ $# -ge 2 ]] || { echo "Missing value for --hf-home" >&2; exit 2; }
      hf_home="$2"
      shift 2
      ;;
    --hf-hub-cache)
      [[ $# -ge 2 ]] || { echo "Missing value for --hf-hub-cache" >&2; exit 2; }
      hf_hub_cache="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$hf_hub_cache" ]]; then
  hf_hub_cache="${hf_home}/hub"
fi

python_bin_candidates=(
  "/opt/homebrew/bin/python3"
  "/usr/local/bin/python3"
)
python_bin=""
for candidate in "${python_bin_candidates[@]}"; do
  if [[ -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "ERROR: Homebrew python3 not found; install it so mlx-lm 0.30.7 is available." >&2
  exit 2
fi

require_mlx_lm_qwen35_support() {
  "${python_bin}" - <<'PY'
import importlib.metadata
import sys

required = (0, 30, 7)

try:
    have_raw = importlib.metadata.version("mlx-lm")
except importlib.metadata.PackageNotFoundError:
    print("ERROR: mlx-lm is not installed.", file=sys.stderr)
    sys.exit(2)

parts = []
for chunk in have_raw.split("."):
    digits = ""
    for ch in chunk:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        break
    parts.append(int(digits))
while len(parts) < 3:
    parts.append(0)
have = tuple(parts[:3])

if have < required:
    print(
        f"ERROR: mlx-lm {have_raw} is too old for Qwen3.5 model loading. "
        "Need >= 0.30.7.",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"mlx-lm version check OK: {have_raw}", file=sys.stderr)
PY
}

out_root="benchmarks/mlx/qwen3_5_35b_a3b_wayfinder/${run_id}"

queue=(
  "${python_bin} scripts/bench_qwen_consumer_mlx.py --model-path '${model_path}' --hf-home '${hf_home}' --hf-hub-cache '${hf_hub_cache}' --hf-offline --mode dense --seq-lens 2048 --decode-len 32 --repeats 1 --chunk-size 4096 --kv-step 4096 --cooldown-sec 10 --seed 42 --skip-multi-turn --skip-quality --stage-timeout-sec 1800 --heartbeat-sec 30 --out-dir '${out_root}/t2048/dense'"
  "${python_bin} scripts/bench_qwen_consumer_mlx.py --model-path '${model_path}' --hf-home '${hf_home}' --hf-hub-cache '${hf_hub_cache}' --hf-offline --mode wayfinder --debug-wayfinder-decode-backend dense --seq-lens 2048 --decode-len 32 --repeats 1 --chunk-size 4096 --kv-step 4096 --cooldown-sec 10 --window 64 --landmark-stride 64 --num-cycles 1 --head-chunk-size 2 --query-chunk-size 384 --seed 42 --skip-multi-turn --skip-quality --stage-timeout-sec 1800 --heartbeat-sec 30 --out-dir '${out_root}/t2048/wayfinder'"
  "${python_bin} scripts/bench_qwen_consumer_mlx.py --model-path '${model_path}' --hf-home '${hf_home}' --hf-hub-cache '${hf_hub_cache}' --hf-offline --mode dense --seq-lens 8192 --decode-len 32 --repeats 1 --chunk-size 4096 --kv-step 4096 --cooldown-sec 10 --seed 42 --skip-multi-turn --skip-quality --stage-timeout-sec 1800 --heartbeat-sec 30 --out-dir '${out_root}/t8192/dense'"
  "${python_bin} scripts/bench_qwen_consumer_mlx.py --model-path '${model_path}' --hf-home '${hf_home}' --hf-hub-cache '${hf_hub_cache}' --hf-offline --mode wayfinder --debug-wayfinder-decode-backend dense --seq-lens 8192 --decode-len 32 --repeats 1 --chunk-size 4096 --kv-step 4096 --cooldown-sec 10 --window 64 --landmark-stride 64 --num-cycles 1 --head-chunk-size 2 --query-chunk-size 384 --seed 42 --skip-multi-turn --skip-quality --stage-timeout-sec 1800 --heartbeat-sec 30 --out-dir '${out_root}/t8192/wayfinder'"
)

echo "[queue] repo_root=${repo_root}"
echo "[queue] python_bin=${python_bin}"
echo "[queue] run_id=${run_id}"
echo "[queue] model_path=${model_path}"
echo "[queue] hf_home=${hf_home}"
echo "[queue] hf_hub_cache=${hf_hub_cache}"
echo "[queue] out_root=${out_root}"

if [[ ${execute} -eq 0 ]]; then
  echo ""
  echo "[queue] no-inference mode: commands prepared but not executed."
  echo "[queue] run with --execute to execute later."
  echo ""
  for i in "${!queue[@]}"; do
    idx=$((i + 1))
    echo "${idx}. ${queue[$i]}"
  done
  exit 0
fi

if [[ ! -f "${model_path}/config.json" ]]; then
  echo "ERROR: model config not found at ${model_path}/config.json" >&2
  exit 2
fi

require_mlx_lm_qwen35_support

cd "${repo_root}"
for i in "${!queue[@]}"; do
  idx=$((i + 1))
  cmd="${queue[$i]}"
  echo ""
  echo "[queue][${idx}/${#queue[@]}] ${cmd}"
  bash -lc "${cmd}"
done

echo ""
echo "[queue] completed."
