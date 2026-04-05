#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export BUTTERFLY_REPO_ROOT="${BUTTERFLY_REPO_ROOT:-$repo_root}"
export BUTTERFLY_SSD_ROOT="${BUTTERFLY_SSD_ROOT:-${repo_root}/.cache/butterfly}"
export HF_HOME="${HF_HOME:-${BUTTERFLY_SSD_ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export BUTTERFLY_MODELS_ROOT="${BUTTERFLY_MODELS_ROOT:-${BUTTERFLY_SSD_ROOT}/models}"
export BUTTERFLY_TOOLCHAINS_ROOT="${BUTTERFLY_TOOLCHAINS_ROOT:-${BUTTERFLY_SSD_ROOT}/toolchains}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${BUTTERFLY_MODELS_ROOT}" "${BUTTERFLY_TOOLCHAINS_ROOT}"
