from __future__ import annotations

import dataclasses
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set RNG seeds for python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device(requested: str = "auto") -> torch.device:
    """Pick a device: cuda if available, else mps, else cpu, unless overridden."""
    requested = requested.lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_memory_bytes(device: torch.device) -> int:
    """Best-effort peak memory for the current process, in bytes."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps"):
        # torch.mps.* exists in recent PyTorch; API may vary.
        get_peak = getattr(torch.mps, "current_allocated_memory", None)
        if callable(get_peak):
            return int(get_peak())
    return 0


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def format_bytes(n: int) -> str:
    if n <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f}{units[i]}"


@dataclass
class RunMetadata:
    timestamp: float
    host: str
    platform: str
    python: str
    torch: str
    cuda: str | None
    git_commit: str | None


def get_git_commit() -> Optional[str]:
    """Return current git commit hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def collect_run_metadata() -> RunMetadata:
    return RunMetadata(
        timestamp=time.time(),
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        git_commit=get_git_commit(),
    )


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
