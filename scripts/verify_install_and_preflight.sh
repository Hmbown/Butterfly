#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${BUTTERFLY_PYTHON:-${repo_root}/.venv-macos-metal/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${PYTHON_BIN:-python3}"
fi

run_id="EXP-$(date -u +%Y%m%dT%H%M%SZ)-VERIFY-INSTALL"
out_dir="benchmarks/mlx/preflight"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-id" >&2; exit 2; }
      run_id="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --out-dir" >&2; exit 2; }
      out_dir="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: verify_install_and_preflight.sh [--run-id ID] [--out-dir PATH]

Runs install verification in strict sequence:
1) MLX/package/system env check (JSON artifact)
2) benchmark preflight setup checks (summary + raw artifacts)
USAGE
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"
mkdir -p "$out_dir"

env_json="${out_dir}/${run_id}_env_check_mlx.json"

echo "[verify] run_id=${run_id}"
echo "[verify] out_dir=${out_dir}"
echo "[verify] python_bin=${python_bin}"
echo "[verify] step=env_check_mlx"
"${python_bin}" scripts/env_check_mlx.py --json-out "$env_json"

echo "[verify] step=bench_protocol_preflight_setup"
BUTTERFLY_PYTHON="${python_bin}" ./scripts/bench_protocol_preflight_setup.sh --run-id "$run_id" --out-dir "$out_dir"

echo "[verify] status=pass"
echo "[verify] env_json=${env_json}"
echo "[verify] summary_json=${out_dir}/${run_id}_summary.json"
echo "[verify] raw_txt=${out_dir}/${run_id}_raw.txt"
