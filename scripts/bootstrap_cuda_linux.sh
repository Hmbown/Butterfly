#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${repo_root}/.venv-cuda"
cache_root="${BUTTERFLY_SSD_ROOT:-${repo_root}/.cache/butterfly}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
skip_python_deps=0
skip_smoke=0
print_env_only=0

usage() {
  cat <<EOF
Usage: bootstrap_cuda_linux.sh [options]

Create a repo-local Linux/WSL CUDA environment for the 10GB consumer-GPU path.

Options:
  --python PATH            Python executable to use for the venv (default: ${python_bin})
  --venv-dir PATH          Venv path (default: ${venv_dir})
  --cache-root PATH        Cache/models root (default: ${cache_root})
  --torch-index-url URL    PyTorch wheel index URL (default: ${torch_index_url})
  --skip-python-deps       Skip venv creation and Python package installation
  --skip-smoke             Skip env_check_cuda.py
  --print-env-only         Print resolved environment values and exit
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_bin="$2"
      shift 2
      ;;
    --venv-dir)
      venv_dir="$2"
      shift 2
      ;;
    --cache-root)
      cache_root="$2"
      shift 2
      ;;
    --torch-index-url)
      torch_index_url="$2"
      shift 2
      ;;
    --skip-python-deps)
      skip_python_deps=1
      shift
      ;;
    --skip-smoke)
      skip_smoke=1
      shift
      ;;
    --print-env-only)
      print_env_only=1
      shift
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

export BUTTERFLY_REPO_ROOT="$repo_root"
export BUTTERFLY_SSD_ROOT="$cache_root"
# shellcheck source=/dev/null
source "${script_dir}/cuda_local_env.sh"

if [[ "${print_env_only}" -eq 1 ]]; then
  cat <<EOF
BUTTERFLY_REPO_ROOT=${BUTTERFLY_REPO_ROOT}
BUTTERFLY_SSD_ROOT=${BUTTERFLY_SSD_ROOT}
HF_HOME=${HF_HOME}
HF_HUB_CACHE=${HF_HUB_CACHE}
BUTTERFLY_MODELS_ROOT=${BUTTERFLY_MODELS_ROOT}
BUTTERFLY_TOOLCHAINS_ROOT=${BUTTERFLY_TOOLCHAINS_ROOT}
TORCH_INDEX_URL=${torch_index_url}
EOF
  exit 0
fi

if [[ "${skip_python_deps}" -eq 0 ]]; then
  "${python_bin}" -m venv "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${venv_dir}/bin/python" -m pip install --index-url "${torch_index_url}" torch torchvision torchaudio
  (
    cd "${repo_root}"
    HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
      "${venv_dir}/bin/python" -m pip install -e ".[cuda,dev,kernels]"
  )
fi

if [[ "${skip_smoke}" -eq 0 ]]; then
  HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
    "${venv_dir}/bin/python" "${repo_root}/scripts/env_check_cuda.py"
fi

cat <<EOF
Bootstrap complete.

Source the shared CUDA environment before downloading models or benchmarking:
  source ${repo_root}/scripts/cuda_local_env.sh

Repo venv:
  ${venv_dir}

Recommended model download:
  ${venv_dir}/bin/python ${repo_root}/scripts/model_catalog.py download qwen35_2b_hf
EOF
