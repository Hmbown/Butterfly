#!/usr/bin/env python3
"""Hyperparameter sweep runner.

Supports grid search and random search over training configurations.
Logs results to CSV (and optionally wandb).

Usage:
    python scripts/sweep.py --config configs/sweep_base.yaml
    python scripts/sweep.py --config configs/sweep_base.yaml --mode random --n-trials 10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import yaml
except ImportError:
    yaml = None

from hcsa.train import run_training, build_argparser
from hcsa.utils import set_seed


def load_sweep_config(path: str) -> Dict[str, Any]:
    """Load sweep config from YAML or JSON."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("pyyaml required: pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def _flatten(prefix: str, d: Dict, out: Dict) -> None:
    """Flatten nested dict for argparse."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v


def generate_grid(sweep_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations for grid search."""
    keys = sorted(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def generate_random(
    sweep_params: Dict[str, List],
    n_trials: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate random combinations."""
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        cfg = {}
        for k, values in sweep_params.items():
            cfg[k] = rng.choice(values)
        configs.append(cfg)
    return configs


def run_sweep(
    sweep_config: Dict[str, Any],
    mode: str = "grid",
    n_trials: int = 10,
    seed: int = 42,
    output_csv: str = "sweep_results.csv",
    use_wandb: bool = False,
) -> List[Dict[str, Any]]:
    """Run a hyperparameter sweep.

    Parameters
    ----------
    sweep_config : dict
        Must contain 'base' (default args) and 'sweep' (param lists).
    mode : str
        'grid' or 'random'.
    n_trials : int
        Number of trials for random search.
    seed : int
        RNG seed.
    output_csv : str
        Path to save results CSV.
    use_wandb : bool
        Whether to log to wandb.
    """
    base_args = sweep_config.get("base", {})
    sweep_params = sweep_config.get("sweep", {})

    if mode == "grid":
        configs = generate_grid(sweep_params)
    else:
        configs = generate_random(sweep_params, n_trials, seed)

    print(f"Sweep: {len(configs)} configurations ({mode} search)")

    # Build base argparse namespace
    parser = build_argparser()

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="hamcycle-sweep", config=sweep_config)
        except ImportError:
            print("wandb not available, skipping wandb logging")

    results = []
    csv_path = Path(output_csv)

    for i, trial_params in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(configs)}")
        print(f"Params: {trial_params}")
        print(f"{'='*60}")

        # Merge base + trial params
        merged = {**base_args, **trial_params}

        # Convert to argparse namespace
        # Map param names to argparse format
        arg_list = []
        for k, v in merged.items():
            arg_name = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    arg_list.append(arg_name)
            else:
                arg_list.append(arg_name)
                arg_list.append(str(v))

        try:
            args = parser.parse_args(arg_list)
        except SystemExit:
            print(f"  SKIP: invalid args for trial {i+1}")
            continue

        # Add trial info to run name
        trial_name = f"sweep_{i+1:03d}_" + "_".join(
            f"{k}={v}" for k, v in sorted(trial_params.items())
        )
        args.run_name = trial_name
        args.no_tqdm = True

        try:
            set_seed(seed + i)
            t0 = time.perf_counter()
            run_dir = run_training(args)
            elapsed = time.perf_counter() - t0

            # Load final checkpoint for results
            import torch
            ckpt_path = run_dir / "ckpt.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                final_step = ckpt.get("step", args.steps)
            else:
                final_step = args.steps

            result = {
                "trial": i + 1,
                **trial_params,
                "run_dir": str(run_dir),
                "elapsed_s": round(elapsed, 1),
                "final_step": final_step,
                "status": "ok",
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            result = {
                "trial": i + 1,
                **trial_params,
                "run_dir": "",
                "elapsed_s": 0,
                "final_step": 0,
                "status": f"error: {e}",
            }

        results.append(result)

        if wandb_run is not None:
            try:
                import wandb
                wandb.log(result)
            except Exception:
                pass

        # Append to CSV incrementally
        write_header = not csv_path.exists() or i == 0
        with open(csv_path, "a" if not write_header else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result)

        print(f"  Status: {result['status']} | Elapsed: {result['elapsed_s']}s")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\nSweep complete. Results saved to {csv_path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter sweep runner.")
    p.add_argument("--config", type=str, required=True, help="Path to sweep config YAML/JSON.")
    p.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    p.add_argument("--n-trials", type=int, default=10, help="Number of trials for random search.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="sweep_results.csv")
    p.add_argument("--wandb", action="store_true", help="Log to wandb.")
    args = p.parse_args()

    sweep_config = load_sweep_config(args.config)
    run_sweep(
        sweep_config,
        mode=args.mode,
        n_trials=args.n_trials,
        seed=args.seed,
        output_csv=args.output,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
