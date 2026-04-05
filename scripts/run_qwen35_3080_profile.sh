#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/cuda_local_env.sh"

python_bin="${PYTHON_BIN:-${repo_root}/.venv-cuda/bin/python}"
model_path="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
quantize="${QUANTIZE:-bnb-4bit}"
dtype="${DTYPE:-float16}"
wayfinder_path="${WAYFINDER_PATH:-block_sparse}"
wayfinder_engine="${WAYFINDER_ENGINE:-triton}"
block_size="${BLOCK_SIZE:-128}"
warmup="${WARMUP:-1}"
repeats="${REPEATS:-3}"
forward_target="${FORWARD_TARGET:-backbone}"
output="${OUTPUT:-${repo_root}/benchmarks/cuda/qwen35_wayfinder/EXP-qwen35-2b-3080-bnb4bit.ndjson}"
seq_lens_raw="${SEQ_LENS:-8192 16384 32768 65536 131072 196608 262144}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  echo "Run ./scripts/bootstrap_cuda_linux.sh first or set PYTHON_BIN." >&2
  exit 1
fi

read -r -a seq_lens <<< "${seq_lens_raw}"

exec "${python_bin}" "${repo_root}/scripts/bench_qwen35_cuda_wayfinder.py" \
  --model-path "${model_path}" \
  --quantize "${quantize}" \
  --dtype "${dtype}" \
  --path "${wayfinder_path}" \
  --engine "${wayfinder_engine}" \
  --block-size "${block_size}" \
  --seq-lens "${seq_lens[@]}" \
  --warmup "${warmup}" \
  --repeats "${repeats}" \
  --forward-target "${forward_target}" \
  --phases wayfinder dense \
  --skip-divergence \
  --output "${output}" \
  "$@"
