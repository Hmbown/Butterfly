#!/usr/bin/env python3
"""Experiment runner: YAML config -> reproducible run -> structured JSON results.

Usage:
    python scripts/run_experiment.py configs/experiments/dense_vs_hcsa.yaml
    python scripts/run_experiment.py configs/experiments/dense_vs_hcsa.yaml --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from hcsa.model import GPT, GPTConfig
from hcsa.data import build_datasets, get_batch, load_text
from hcsa.tokenizers import build_tokenizer
from hcsa.train import estimate_loss
from hcsa.utils import (
    auto_device,
    collect_run_metadata,
    ensure_dir,
    format_bytes,
    peak_memory_bytes,
    reset_peak_memory_stats,
    save_json,
    set_seed,
)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _config_hash(cfg: Dict[str, Any]) -> str:
    """Short hash of a config dict for deduplication."""
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _build_gpt_config(run_cfg: Dict[str, Any], vocab_size: int) -> GPTConfig:
    """Build a GPTConfig from an experiment run config dict."""
    landmark = run_cfg.get("landmark_stride", 64)
    if isinstance(landmark, int) and landmark <= 0:
        landmark = None
    return GPTConfig(
        vocab_size=vocab_size,
        seq_len=run_cfg.get("seq_len", 256),
        n_layers=run_cfg.get("n_layers", 6),
        n_heads=run_cfg.get("n_heads", 8),
        n_embd=run_cfg.get("n_embd", 512),
        dropout=run_cfg.get("dropout", 0.0),
        attn=run_cfg.get("attn", "dense"),
        cycle=run_cfg.get("cycle", "random"),
        window=run_cfg.get("window", 64),
        landmark_stride=landmark,
        num_cycles=run_cfg.get("num_cycles", 1),
        routing_dim=run_cfg.get("routing_dim", None),
        seed=run_cfg.get("seed", 0),
    )


def _train_single_run(
    run_cfg: Dict[str, Any],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single configuration and return metrics."""
    cfg = _build_gpt_config(run_cfg, vocab_size)
    model = GPT(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    lr = run_cfg.get("lr", 3e-4)
    steps = run_cfg.get("steps", 200)
    batch_size = run_cfg.get("batch_size", 32)
    grad_clip = run_cfg.get("grad_clip", 1.0)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=run_cfg.get("weight_decay", 0.1),
    )

    reset_peak_memory_stats(device)
    model.train()

    losses: List[float] = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        xb, yb = get_batch(train_data, batch_size, cfg.seq_len, device)
        out = model(xb, yb)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        losses.append(loss.item())

    elapsed = time.perf_counter() - t0
    total_tokens = steps * batch_size * cfg.seq_len
    tokens_per_sec = total_tokens / max(elapsed, 1e-8)

    # Evaluation
    model.eval()
    val_loss = estimate_loss(
        model, val_data, batch_size=batch_size, seq_len=cfg.seq_len, device=device
    )
    import math

    val_ppl = math.exp(min(val_loss, 20.0))  # cap to avoid overflow

    return {
        "config": run_cfg,
        "model_config": asdict(cfg),
        "n_params": n_params,
        "final_train_loss": losses[-1] if losses else float("nan"),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_bytes": peak_memory_bytes(device),
        "peak_memory": format_bytes(peak_memory_bytes(device)),
        "elapsed_s": elapsed,
        "steps": steps,
    }


def run_experiment(config_path: str | Path, resume: bool = False) -> Path:
    """Run a full experiment from a YAML config file."""
    exp_cfg = _load_yaml(config_path)

    name = exp_cfg.get("name", Path(config_path).stem)
    outdir = ensure_dir(Path(exp_cfg.get("outdir", "experiments")) / name)
    results_path = outdir / "results.json"

    # Load existing results for resume
    completed_hashes: set[str] = set()
    existing_results: List[Dict[str, Any]] = []
    if resume and results_path.exists():
        existing_results = json.loads(results_path.read_text())
        for r in existing_results:
            h = _config_hash(r.get("config", {}))
            completed_hashes.add(h)
        print(f"Resuming: {len(existing_results)} runs already completed.")

    # Device
    device = auto_device(exp_cfg.get("device", "auto"))
    print(f"Device: {device}")

    # Data
    data_cfg = exp_cfg.get("data", {})
    data_path = data_cfg.get("path", "data/tinyshakespeare.txt")
    tokenizer_kind = data_cfg.get("tokenizer", "char")

    text = load_text(data_path)
    tok = build_tokenizer(
        tokenizer_kind,
        text_for_char_vocab=text if tokenizer_kind == "char" else None,
    )
    data = build_datasets(text, tok)

    # Global seed
    global_seed = exp_cfg.get("seed", 1337)
    set_seed(global_seed)

    # Runs
    runs = exp_cfg.get("runs", [])
    all_results = list(existing_results)

    for i, run_cfg in enumerate(runs):
        run_hash = _config_hash(run_cfg)
        if run_hash in completed_hashes:
            print(f"[{i+1}/{len(runs)}] Skipping (already completed): {run_cfg.get('name', run_hash)}")
            continue

        run_name = run_cfg.get("name", f"run_{i}")
        print(f"\n[{i+1}/{len(runs)}] Running: {run_name}")

        set_seed(run_cfg.get("seed", global_seed))
        result = _train_single_run(
            run_cfg, data.train, data.val, tok.vocab_size, device
        )
        result["run_name"] = run_name
        result["run_hash"] = run_hash

        all_results.append(result)

        # Save incrementally
        save_json(results_path, all_results)

        print(
            f"  val_loss={result['val_loss']:.4f} "
            f"val_ppl={result['val_ppl']:.2f} "
            f"tok/s={result['tokens_per_sec']:,.0f} "
            f"mem={result['peak_memory']}"
        )

    # Save metadata
    save_json(
        outdir / "metadata.json",
        {
            "experiment": name,
            "config_path": str(config_path),
            "metadata": asdict(collect_run_metadata()),
            "device": str(device),
            "n_runs": len(all_results),
        },
    )

    print(f"\nExperiment complete. Results saved to: {results_path}")
    return outdir


def main() -> None:
    p = argparse.ArgumentParser(description="Run experiments from YAML config.")
    p.add_argument("config", type=str, help="Path to experiment YAML config.")
    p.add_argument("--resume", action="store_true", help="Resume from existing results.")
    args = p.parse_args()
    run_experiment(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
