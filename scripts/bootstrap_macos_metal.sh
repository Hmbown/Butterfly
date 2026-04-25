#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${repo_root}/.venv-macos-metal"
ssd_root="${BUTTERFLY_SSD_ROOT:-/Volumes/VIXinSSD}"
zmlx_root="${BUTTERFLY_ZMLX_ROOT:-${ssd_root}/ZMLX}"
llama_tag="${BUTTERFLY_LLAMA_CPP_TAG:-b8656}"
skip_python_deps=0
skip_llama_build=0
rebuild_llama=0
skip_smoke=0
print_env_only=0

usage() {
  cat <<EOF
Usage: bootstrap_macos_metal.sh [options]

Create a repo-local Apple Silicon environment for MLX + ZMLX + llama.cpp Metal.

Options:
  --python PATH           Python executable to use for the venv (default: ${python_bin})
  --venv-dir PATH         Venv path (default: ${venv_dir})
  --ssd-root PATH         SSD root used for caches/models/toolchains (default: ${ssd_root})
  --zmlx-root PATH        ZMLX checkout root (default: ${zmlx_root})
  --llama-tag TAG         llama.cpp tag/ref to build (default: ${llama_tag})
  --skip-python-deps      Skip venv creation and Python package installation
  --skip-llama-build      Skip llama.cpp clone/build/install
  --rebuild-llama         Remove the current llama.cpp build/install directories before rebuilding
  --skip-smoke            Skip env_check/zmlx/llama.cpp smoke checks
  --print-env-only        Print resolved environment values and exit
  -h, --help              Show this help text
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
    --ssd-root)
      ssd_root="$2"
      shift 2
      ;;
    --zmlx-root)
      zmlx_root="$2"
      shift 2
      ;;
    --llama-tag)
      llama_tag="$2"
      shift 2
      ;;
    --skip-python-deps)
      skip_python_deps=1
      shift
      ;;
    --skip-llama-build)
      skip_llama_build=1
      shift
      ;;
    --rebuild-llama)
      rebuild_llama=1
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
export BUTTERFLY_SSD_ROOT="$ssd_root"
export BUTTERFLY_ZMLX_ROOT="$zmlx_root"
export BUTTERFLY_LLAMA_CPP_TAG="$llama_tag"
# shellcheck source=/dev/null
source "${script_dir}/macos_local_env.sh"

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${BUTTERFLY_MODELS_ROOT}" \
  "${BUTTERFLY_TOOLCHAINS_ROOT}"

if [[ "${print_env_only}" -eq 1 ]]; then
  cat <<EOF
BUTTERFLY_REPO_ROOT=${BUTTERFLY_REPO_ROOT}
BUTTERFLY_SSD_ROOT=${BUTTERFLY_SSD_ROOT}
HF_HOME=${HF_HOME}
HF_HUB_CACHE=${HF_HUB_CACHE}
BUTTERFLY_MODELS_ROOT=${BUTTERFLY_MODELS_ROOT}
BUTTERFLY_ZMLX_ROOT=${BUTTERFLY_ZMLX_ROOT}
BUTTERFLY_LLAMA_CPP_TAG=${BUTTERFLY_LLAMA_CPP_TAG}
BUTTERFLY_LLAMA_CPP_SRC_ROOT=${BUTTERFLY_LLAMA_CPP_SRC_ROOT}
BUTTERFLY_LLAMA_CPP_BUILD_ROOT=${BUTTERFLY_LLAMA_CPP_BUILD_ROOT}
BUTTERFLY_LLAMA_CPP_ROOT=${BUTTERFLY_LLAMA_CPP_ROOT}
BUTTERFLY_LLAMA_CPP_CURRENT_ROOT=${BUTTERFLY_LLAMA_CPP_CURRENT_ROOT}
EOF
  exit 0
fi

link_zmlx_into_venv() {
  local python_exec="$1"
  "${python_exec}" - <<PY
import site
from pathlib import Path

site_packages = None
for candidate in site.getsitepackages():
    if candidate.endswith("site-packages"):
        site_packages = Path(candidate)
        break
if site_packages is None:
    raise SystemExit("site-packages directory not found")
pth = site_packages / "butterfly_zmlx_local.pth"
pth.write_text("${BUTTERFLY_ZMLX_ROOT}/src\n", encoding="utf-8")
print(pth)
PY
}

if [[ "${skip_python_deps}" -eq 0 ]]; then
  "${python_bin}" -m venv "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
  (
    cd "${repo_root}"
    HF_HOME="${HF_HOME}" HF_HUB_CACHE="${HF_HUB_CACHE}" \
      "${venv_dir}/bin/python" -m pip install \
        -e ".[hf,dev,viz,experiments,server]" \
        mlx==0.31.1 \
        mlx-lm==0.31.1 \
        datasets
  )
  link_zmlx_into_venv "${venv_dir}/bin/python"
fi

if [[ "${skip_llama_build}" -eq 0 ]]; then
  if [[ "${rebuild_llama}" -eq 1 ]]; then
    rm -rf "${BUTTERFLY_LLAMA_CPP_BUILD_ROOT}" "${BUTTERFLY_LLAMA_CPP_ROOT}"
  fi

  if [[ ! -d "${BUTTERFLY_LLAMA_CPP_SRC_ROOT}/.git" ]]; then
    git clone --quiet --branch "${BUTTERFLY_LLAMA_CPP_TAG}" --depth 1 \
      https://github.com/ggml-org/llama.cpp.git \
      "${BUTTERFLY_LLAMA_CPP_SRC_ROOT}"
  else
    git -C "${BUTTERFLY_LLAMA_CPP_SRC_ROOT}" fetch --quiet --depth 1 origin \
      "refs/tags/${BUTTERFLY_LLAMA_CPP_TAG}:refs/tags/${BUTTERFLY_LLAMA_CPP_TAG}"
    git -C "${BUTTERFLY_LLAMA_CPP_SRC_ROOT}" checkout -f "${BUTTERFLY_LLAMA_CPP_TAG}" >/dev/null
  fi

  cmake -S "${BUTTERFLY_LLAMA_CPP_SRC_ROOT}" \
    -B "${BUTTERFLY_LLAMA_CPP_BUILD_ROOT}" \
    -G Ninja \
    -DGGML_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUTTERFLY_LLAMA_CPP_ROOT}" \
    -DCMAKE_INSTALL_RPATH="@executable_path/../lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

  build_jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"
  cmake --build "${BUTTERFLY_LLAMA_CPP_BUILD_ROOT}" --target install -j "${build_jobs}"
  ln -sfn "${BUTTERFLY_LLAMA_CPP_ROOT}" "${BUTTERFLY_LLAMA_CPP_CURRENT_ROOT}"

  for bin_path in "${BUTTERFLY_LLAMA_CPP_ROOT}"/bin/*; do
    if file "${bin_path}" | grep -q "Mach-O"; then
      install_name_tool -add_rpath '@executable_path/../lib' "${bin_path}" 2>/dev/null || true
    fi
  done
fi

if [[ "${skip_smoke}" -eq 0 ]]; then
  "${venv_dir}/bin/python" "${repo_root}/scripts/env_check_mlx.py"
  "${venv_dir}/bin/python" - <<PY
import zmlx
print(zmlx.__file__)
PY
  "${BUTTERFLY_LLAMA_CPP_ROOT}/bin/llama-cli" --version
  "${BUTTERFLY_LLAMA_CPP_ROOT}/bin/llama-cli" --list-devices | sed -n '1,40p'
fi

cat <<EOF
Bootstrap complete.

Source the shared environment before benchmarking:
  source ${repo_root}/scripts/macos_local_env.sh

Repo venv:
  ${venv_dir}

llama.cpp install:
  ${BUTTERFLY_LLAMA_CPP_ROOT}
EOF
