#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from hcsa.tokenizers import CharTokenizer
from hcsa.compiler import compile_graph_spec
from hcsa.graph.abi import graph_metrics
from hcsa.mlx.metrics import edge_utilization_by_type
from hcsa.mlx.model import GPTConfigMLX, GPTMLX


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_data(text: str, val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray, CharTokenizer]:
    tok = CharTokenizer.from_text(text)
    ids = np.asarray(tok.encode(text), dtype=np.int32)
    n_val = max(1, int(len(ids) * val_fraction))
    train = ids[:-n_val]
    val = ids[-n_val:]
    return train, val, tok


def sample_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[mx.array, mx.array]:
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int32)
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def eval_loss(
    model: GPTMLX,
    data: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    rng: np.random.Generator,
) -> float:
    vals: list[float] = []
    for _ in range(eval_batches):
        xb, yb = sample_batch(data, batch_size=batch_size, seq_len=seq_len, rng=rng)
        out = model(xb, targets=yb)
        loss = out["loss"]
        mx.eval(loss)
        vals.append(float(loss.item()))
    return float(np.mean(vals))


def linear_schedule(step: int, *, start: float, end: float, steps: int) -> float:
    if steps <= 1:
        return float(end)
    alpha = min(1.0, max(0.0, float(step) / float(steps - 1)))
    return float(start + alpha * (end - start))


def _edge_mass_terms(attn_weights: mx.array, edge_type: mx.array) -> Dict[str, mx.array]:
    et = edge_type.astype(mx.int32)[None, ...]
    w = attn_weights.astype(mx.float32)
    total = mx.maximum(mx.sum(w), mx.array(1e-9, dtype=mx.float32))

    def mass(code: int) -> mx.array:
        m = (et == int(code)).astype(mx.float32)
        return mx.sum(w * m) / total

    return {
        "cycle": mass(1),
        "window": mass(2),
        "landmark": mass(3),
        "rewire": mass(4),
    }


def run_dense_baseline(
    cfg: GPTConfigMLX,
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    eval_batches: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    model = GPTMLX(cfg)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(m: GPTMLX, xb: mx.array, yb: mx.array) -> mx.array:
        return m(xb, targets=yb)["loss"]

    value_and_grad = nn.value_and_grad(model, loss_fn)
    tok_rates: list[float] = []
    for _ in range(steps):
        xb, yb = sample_batch(train_data, batch_size=batch_size, seq_len=cfg.seq_len, rng=rng)
        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, xb, yb)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        tok_rates.append(float((batch_size * cfg.seq_len) / max(time.perf_counter() - t0, 1e-12)))

    val_loss = eval_loss(
        model,
        val_data,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        eval_batches=eval_batches,
        rng=rng,
    )
    return {
        "val_ppl": float(math.exp(min(val_loss, 20.0))),
        "tok_s": float(np.mean(tok_rates)),
    }


def run_wayfinder_config(
    cfg: GPTConfigMLX,
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    eval_batches: int,
    seed: int,
    window_drop_max: float,
    bias_cycle_max: float,
    bias_landmark_max: float,
    reliance_reg_coeff: float,
    reliance_cycle_min: float,
    reliance_window_max: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    model = GPTMLX(cfg)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    warm_steps = max(1, int(steps * 0.6))

    def loss_fn(m: GPTMLX, xb: mx.array, yb: mx.array, step_idx: int) -> mx.array:
        out = m(xb, targets=yb, return_debug=True)
        loss = out["loss"]
        b0 = out["debug"]["blocks"][0]["attn"]
        w = b0.get("attn_weights")
        et = b0.get("edge_type")
        if w is None or et is None or reliance_reg_coeff <= 0.0:
            return loss

        masses = _edge_mass_terms(w, et)
        cycle_deficit = mx.maximum(mx.array(0.0), mx.array(reliance_cycle_min) - masses["cycle"])
        window_excess = mx.maximum(mx.array(0.0), masses["window"] - mx.array(reliance_window_max))
        reg = cycle_deficit + window_excess
        return loss + mx.array(reliance_reg_coeff, dtype=loss.dtype) * reg

    value_and_grad = nn.value_and_grad(model, lambda m, xb, yb, step_idx: loss_fn(m, xb, yb, step_idx))

    tok_rates: list[float] = []
    for step in range(1, steps + 1):
        xb, yb = sample_batch(train_data, batch_size=batch_size, seq_len=cfg.seq_len, rng=rng)

        model.set_wayfinder_runtime_controls(
            window_drop=linear_schedule(step, start=0.0, end=window_drop_max, steps=warm_steps),
            schedule_bias={
                "cycle": linear_schedule(step, start=0.0, end=bias_cycle_max, steps=warm_steps),
                "landmark": linear_schedule(step, start=0.0, end=bias_landmark_max, steps=warm_steps),
            },
        )

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, xb, yb, step)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        tok_rates.append(float((batch_size * cfg.seq_len) / max(time.perf_counter() - t0, 1e-12)))

    val_loss = eval_loss(
        model,
        val_data,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        eval_batches=eval_batches,
        rng=rng,
    )

    xb_probe, _ = sample_batch(train_data, batch_size=1, seq_len=cfg.seq_len, rng=rng)
    probe = model(xb_probe, return_debug=True)
    mx.eval(probe["logits"])
    b0 = probe["debug"]["blocks"][0]["attn"]

    w = b0.get("attn_weights")
    et = b0.get("edge_type")
    cycle = landmark = window = rewire = 0.0
    if w is not None and et is not None:
        mx.eval(w, et)
        util = edge_utilization_by_type(np.asarray(w, dtype=np.float32), np.asarray(et, dtype=np.uint8))
        cycle = float(util.get("cycle", 0.0))
        window = float(util.get("window", 0.0))
        landmark = float(util.get("landmark", 0.0))
        rewire = float(util.get("rewire", 0.0))

    gm = {}
    graph_abi = b0.get("graph_abi")
    if graph_abi is not None:
        gm = graph_metrics(graph_abi)

    cache_hit_rate = 0.0
    avg_graph_build_ms = 0.0
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if hasattr(attn, "last_profile"):
            prof = attn.last_profile.to_dict()
            cache_hit_rate = float(prof.get("cache_hit", 0.0))
            avg_graph_build_ms = float(prof.get("graph_build_ms", 0.0))
            break

    return {
        "val_ppl": float(math.exp(min(val_loss, 20.0))),
        "tok_s": float(np.mean(tok_rates)),
        "cycle_pct": cycle * 100.0,
        "window_pct": window * 100.0,
        "landmark_pct": landmark * 100.0,
        "rewire_pct": rewire * 100.0,
        "shortcut_rate": float(gm.get("shortcut_rate", 0.0)),
        "reachability": float(gm.get("reachability_proxy", 0.0)),
        "cache_hit_rate": cache_hit_rate,
        "avg_graph_build_ms": avg_graph_build_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Cycle-push ablations for MLX Wayfinder")
    p.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--strategy", type=str, default="random", choices=["random", "greedy", "online_insertion"])
    p.add_argument("--wayfinder-attn", type=str, default="wayfinder_sparse", choices=["wayfinder_sparse", "wayfinder_permute"])
    p.add_argument("--window-sizes", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--window-drop-max", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.7])
    p.add_argument("--bias-cycle-max", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.8])
    p.add_argument("--landmark-strides", type=str, nargs="+", default=["off", "64", "128"])
    p.add_argument("--steps-grid", type=int, nargs="+", default=[200, 500])
    p.add_argument("--max-runs", type=int, default=0, help="0 means full grid")
    p.add_argument("--reliance-reg-coeff", type=float, default=0.01)
    p.add_argument("--reliance-cycle-min", type=float, default=0.10)
    p.add_argument("--reliance-window-max", type=float, default=0.85)
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    args = p.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    text = load_text(args.data)
    train_data, val_data, tok = build_data(text)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/mlx/ablations_cycle_push") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = list(
        itertools.product(
            args.steps_grid,
            args.window_sizes,
            args.window_drop_max,
            args.bias_cycle_max,
            args.landmark_strides,
        )
    )
    if args.max_runs > 0:
        configs = configs[: args.max_runs]

    dense_cache: dict[int, Dict[str, float]] = {}
    rows: list[Dict[str, Any]] = []

    for run_idx, (steps, window_size, drop_max, cycle_bias_max, landmark_s) in enumerate(configs, start=1):
        landmark_stride = None if landmark_s == "off" else int(landmark_s)
        compiled_graph_dir: Optional[str] = None
        if args.graph_spec is not None:
            compiled = compile_graph_spec(
                args.graph_spec,
                T=args.seq_len,
                H=args.heads,
                out_root=args.graph_cache_root,
            )
            compiled_graph_dir = str(compiled["artifact"]["artifact_dir"])

        base_cfg = GPTConfigMLX(
            vocab_size=tok.vocab_size,
            seq_len=args.seq_len,
            n_layers=args.layers,
            n_heads=args.heads,
            n_embd=args.embd,
            dropout=0.0,
            strategy=args.strategy,
            window=window_size,
            landmark_stride=landmark_stride,
            num_cycles=1,
            seed=args.seed,
            graph_spec=None if args.graph_spec is None else str(args.graph_spec),
            compiled_graph_dir=compiled_graph_dir,
        )

        if steps not in dense_cache:
            dense_cfg = GPTConfigMLX(**{**asdict(base_cfg), "attn": "dense", "edge_bias": False})
            dense_cache[steps] = run_dense_baseline(
                dense_cfg,
                train_data=train_data,
                val_data=val_data,
                steps=steps,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_batches=args.eval_batches,
                seed=args.seed + 10_000 + steps,
            )

        dense_ref = dense_cache[steps]

        way_cfg = GPTConfigMLX(**{**asdict(base_cfg), "attn": args.wayfinder_attn, "edge_bias": True})
        way = run_wayfinder_config(
            way_cfg,
            train_data=train_data,
            val_data=val_data,
            steps=steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_batches=args.eval_batches,
            seed=args.seed + run_idx,
            window_drop_max=drop_max,
            bias_cycle_max=cycle_bias_max,
            bias_landmark_max=0.1,
            reliance_reg_coeff=args.reliance_reg_coeff,
            reliance_cycle_min=args.reliance_cycle_min,
            reliance_window_max=args.reliance_window_max,
        )

        row = {
            "run_idx": run_idx,
            "steps": steps,
            "window_size": window_size,
            "window_drop_max": drop_max,
            "bias_cycle_max": cycle_bias_max,
            "landmark_stride": "off" if landmark_stride is None else landmark_stride,
            "dense_val_ppl": dense_ref["val_ppl"],
            "val_ppl": way["val_ppl"],
            "ppl_gap": float(way["val_ppl"] - dense_ref["val_ppl"]),
            "dense_tok_s": dense_ref["tok_s"],
            "tok_s": way["tok_s"],
            "tok_s_ratio": float(way["tok_s"] / max(dense_ref["tok_s"], 1e-12)),
            "cycle_pct": way["cycle_pct"],
            "window_pct": way["window_pct"],
            "landmark_pct": way["landmark_pct"],
            "rewire_pct": way["rewire_pct"],
            "reachability": way["reachability"],
            "shortcut_rate": way["shortcut_rate"],
            "cache_hit_rate": way["cache_hit_rate"],
            "avg_graph_build_ms": way["avg_graph_build_ms"],
        }
        rows.append(row)
        print(
            f"[{run_idx}/{len(configs)}] steps={steps} window={window_size} drop={drop_max} "
            f"bias={cycle_bias_max} landmark={row['landmark_stride']} -> "
            f"ppl_gap={row['ppl_gap']:.3f} cycle%={row['cycle_pct']:.2f} tok_ratio={row['tok_s_ratio']:.3f}"
        )

    csv_path = out_dir / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    qualified = [
        r for r in rows if int(r["steps"]) >= 500 and float(r["cycle_pct"]) >= 10.0 and float(r["ppl_gap"]) <= 2.0
    ]
    if qualified:
        best = min(qualified, key=lambda r: (float(r["ppl_gap"]), -float(r["tok_s_ratio"])))
    else:
        best = max(rows, key=lambda r: float(r["cycle_pct"])) if rows else None

    summary_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "steps_grid": args.steps_grid,
            "window_sizes": args.window_sizes,
            "window_drop_max": args.window_drop_max,
            "bias_cycle_max": args.bias_cycle_max,
            "landmark_strides": args.landmark_strides,
            "max_runs": args.max_runs,
        },
        "num_runs": len(rows),
        "best": best,
        "qualified_hits": len(qualified),
        "summary_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
