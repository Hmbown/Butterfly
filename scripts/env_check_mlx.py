#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.local_setup import default_local_paths


def _fmt_gb(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _probe_module(name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {
            "available": True,
            "version": getattr(mod, "__version__", None),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _recommend_defaults(device_info: Dict[str, Any]) -> Dict[str, Any]:
    mem_size = int(device_info.get("memory_size", 0))
    rec_ws = int(device_info.get("max_recommended_working_set_size", 0))
    budget = max(mem_size, rec_ws)
    budget_gb = _fmt_gb(budget)

    if budget_gb >= 30.0:
        return {
            "train_8k": {"batch_size": 1, "grad_accum": 8, "dtype": "bfloat16"},
            "train_32k": {"batch_size": 1, "grad_accum": 32, "dtype": "bfloat16"},
            "eval_32k": {"batch_size": 1, "dtype": "bfloat16"},
            "eval_64k": {"batch_size": 1, "dtype": "bfloat16", "note": "try eval-only first"},
        }
    if budget_gb >= 22.0:
        return {
            "train_8k": {"batch_size": 1, "grad_accum": 12, "dtype": "bfloat16"},
            "train_32k": {"batch_size": 1, "grad_accum": 48, "dtype": "bfloat16"},
            "eval_32k": {"batch_size": 1, "dtype": "bfloat16"},
            "eval_64k": {"batch_size": 1, "dtype": "float16", "note": "likely unstable"},
        }
    return {
        "train_8k": {"batch_size": 1, "grad_accum": 16, "dtype": "float16"},
        "train_32k": {
            "batch_size": 1,
            "grad_accum": 64,
            "dtype": "float16",
            "note": "use smoke runs only",
        },
        "eval_32k": {"batch_size": 1, "dtype": "float16"},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MLX environment and capacity check")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    local_paths = default_local_paths(REPO_ROOT)
    device_info = mx.device_info()
    disk_points = [
        Path.cwd(),
        Path.cwd() / "runs",
        Path.cwd() / "benchmarks",
        Path.cwd() / "results" / "benchmarks",
        Path.home(),
    ]
    disk_points = [p.resolve() for p in disk_points if p.exists()]

    disk_report: Dict[str, Any] = {}
    for point in disk_points:
        usage = shutil.disk_usage(point)
        disk_report[str(point)] = {
            "total_gb": round(_fmt_gb(usage.total), 2),
            "used_gb": round(_fmt_gb(usage.used), 2),
            "free_gb": round(_fmt_gb(usage.free), 2),
        }

    report: Dict[str, Any] = {
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
            "cwd": str(Path.cwd()),
        },
        "mlx": {
            "mlx_core_version": getattr(mx, "__version__", None),
            "device_info": device_info,
            "memory_gb": round(_fmt_gb(int(device_info.get("memory_size", 0))), 2),
            "recommended_working_set_gb": round(
                _fmt_gb(int(device_info.get("max_recommended_working_set_size", 0))),
                2,
            ),
        },
        "packages": {
            "mlx_lm": _probe_module("mlx_lm"),
            "tokenizers": _probe_module("tokenizers"),
            "datasets": _probe_module("datasets"),
            "huggingface_hub": _probe_module("huggingface_hub"),
            "zmlx": _probe_module("zmlx"),
        },
        "disk": disk_report,
        "recommended_defaults": _recommend_defaults(device_info),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "MLX_CACHE_DIR": os.environ.get("MLX_CACHE_DIR"),
        },
        "local_paths": local_paths.as_dict(),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
