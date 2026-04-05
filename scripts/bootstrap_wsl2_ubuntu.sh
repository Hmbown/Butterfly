#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

apt_packages=(
  build-essential
  python3
  python3-pip
  python3-venv
)
skip_apt=0
skip_gpu_check=0

usage() {
  cat <<EOF
Usage: bootstrap_wsl2_ubuntu.sh [options] [-- bootstrap_cuda_linux args...]

Install the Ubuntu-side prerequisites for the Butterfly CUDA path on WSL2, then
delegate to scripts/bootstrap_cuda_linux.sh.

Options:
  --skip-apt         Skip apt package installation
  --skip-gpu-check   Skip the nvidia-smi check
  -h, --help         Show this help text

Any remaining arguments after -- are forwarded to bootstrap_cuda_linux.sh.
EOF
}

forwarded_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt)
      skip_apt=1
      shift
      ;;
    --skip-gpu-check)
      skip_gpu_check=1
      shift
      ;;
    --)
      shift
      forwarded_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is intended for Ubuntu running inside WSL2." >&2
  exit 1
fi

if [[ -z "${WSL_DISTRO_NAME:-}" ]] && ! grep -qiE "(microsoft|wsl)" /proc/version; then
  echo "WSL2 was not detected. Use scripts/bootstrap_cuda_linux.sh for native Linux." >&2
  exit 1
fi

if [[ "${skip_gpu_check}" -eq 0 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
nvidia-smi is not available inside WSL.
Install/update the Windows NVIDIA driver with WSL CUDA support first, then run:
  wsl --update
EOF
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
nvidia-smi failed inside WSL.
Confirm the Windows host driver is current and that CUDA-on-WSL is working before continuing.
EOF
    exit 1
  fi
fi

if [[ "${skip_apt}" -eq 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Ubuntu prerequisites. Re-run with --skip-apt if already installed." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y "${apt_packages[@]}"
fi

exec "${repo_root}/scripts/bootstrap_cuda_linux.sh" "${forwarded_args[@]}"
