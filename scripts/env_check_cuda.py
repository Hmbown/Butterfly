#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.qwen_torch import get_cuda_arch_support_diagnostics  # noqa: E402


def _version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def main() -> None:
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "bitsandbytes": _version("bitsandbytes"),
        "torchao": _version("torchao"),
        "triton": _version("triton"),
        "huggingface_hub": _version("huggingface_hub"),
        "cuda_available": bool(torch.cuda.is_available()),
        "bitsandbytes_importable": _module_importable("bitsandbytes"),
    }

    if torch.cuda.is_available():
        diag = get_cuda_arch_support_diagnostics()
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        payload.update(
            {
                "device_index": int(device_index),
                "device_name": torch.cuda.get_device_name(device_index),
                "device_capability": list(torch.cuda.get_device_capability(device_index)),
                "device_total_gb": round(float(props.total_memory) / float(1024**3), 2),
                "cuda_arch_diag": diag,
            }
        )

    print(json.dumps(payload, indent=2))

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False")
    if not payload["transformers"]:
        raise SystemExit("transformers is not installed")
    if not payload["accelerate"]:
        raise SystemExit("accelerate is not installed")
    if not payload["bitsandbytes"]:
        raise SystemExit("bitsandbytes is not installed")


if __name__ == "__main__":
    main()
