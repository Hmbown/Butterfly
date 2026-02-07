#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]

    rc = run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "hcsa/mlx",
            "hcsa/graph",
            "hcsa/compiler",
            "scripts/bench_mlx_wayfinder_scale.py",
            "scripts/run_mlx_experiment_tiny.py",
            "scripts/run_mlx_experiment_tiny_long.py",
            "scripts/run_mlx_ablation_cycle_push.py",
            "scripts/wayc.py",
            "tests/mlx",
            "--ignore",
            "E402",
        ]
    )
    if rc != 0:
        return rc

    rc = run([sys.executable, "-m", "pytest", "-q", "tests/mlx"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
