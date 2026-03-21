#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from mlx.utils import tree_flatten

from bna.tokenizers import CharTokenizer
from bna.compiler import compile_graph_spec
from bna.graph.abi import graph_metrics
from bna.mlx.metrics import edge_utilization_by_type
from bna.mlx.model import GPTConfigMLX, GPTMLX


def _parse_bool(text: str) -> bool:
    val = str(text).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {text!r}")


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_data(
    text: str, val_fraction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, CharTokenizer]:
    tok = CharTokenizer.from_text(text)
    ids = np.asarray(tok.encode(text), dtype=np.int32)
    n_val = max(1, int(len(ids) * val_fraction))
    train = ids[:-n_val]
    val = ids[-n_val:]
    if len(train) < 4:
        raise ValueError("Training split is too small")
    return train, val, tok


def sample_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[mx.array, mx.array]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Data too small for seq_len={seq_len}")
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
    losses: list[float] = []
    for _ in range(eval_batches):
        xb, yb = sample_batch(data, batch_size=batch_size, seq_len=seq_len, rng=rng)
        out = model(xb, targets=yb)
        loss = out["loss"]
        mx.eval(loss)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def linear_schedule(step: int, *, start: float, end: float, steps: int) -> float:
    if steps <= 1:
        return float(end)
    alpha = min(1.0, max(0.0, float(step) / float(steps - 1)))
    return float(start + alpha * (end - start))


def _first_wayfinder_attn(model: GPTMLX):
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if hasattr(attn, "last_profile"):
            return attn
    return None


def _save_checkpoint(model: GPTMLX, opt: optim.AdamW, out_dir: Path, step: int) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.npz"

    model_flat = tree_flatten(model.parameters())
    opt_flat = tree_flatten(opt.state)

    payload: dict[str, mx.array] = {}
    for k, v in model_flat:
        payload[f"model__{k.replace('.', '__')}"] = v
    for k, v in opt_flat:
        payload[f"opt__{k.replace('.', '__')}"] = v

    mx.savez(str(path), **payload)
    return path


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


def train_one(
    *,
    run_name: str,
    attn_mode: str,
    cfg: GPTConfigMLX,
    train_data: np.ndarray,
    val_data: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    eval_batches: int,
    checkpoint_every: int,
    rng: np.random.Generator,
    metrics_path: Path,
    # Schedules / shaping
    window_drop_max: float,
    window_drop_warm_steps: int,
    bias_cycle_max: float,
    bias_window_min: float,
    bias_landmark_max: float,
    bias_warm_steps: int,
    reliance_reg_coeff: float,
    reliance_cycle_min: float,
    reliance_window_max: float,
) -> Dict[str, Any]:
    model = GPTMLX(cfg)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    _reset_peak_memory()

    def loss_fn(m: GPTMLX, xb: mx.array, yb: mx.array, add_reg: bool, step_idx: int) -> mx.array:
        if not add_reg:
            return m(xb, targets=yb)["loss"]

        out = m(xb, targets=yb, return_debug=True)
        loss = out["loss"]
        b0 = out["debug"]["blocks"][0]["attn"]
        w = b0.get("attn_weights")
        et = b0.get("edge_type")
        if w is None or et is None:
            return loss

        masses = _edge_mass_terms(w, et)
        cycle_deficit = mx.maximum(mx.array(0.0), mx.array(reliance_cycle_min) - masses["cycle"])
        window_excess = mx.maximum(mx.array(0.0), masses["window"] - mx.array(reliance_window_max))
        reg = cycle_deficit + window_excess
        return loss + mx.array(reliance_reg_coeff, dtype=loss.dtype) * reg

    value_and_grad = nn.value_and_grad(
        model,
        lambda m, xb, yb, add_reg, step_idx: loss_fn(m, xb, yb, add_reg, step_idx),
    )

    step_records: list[Dict[str, Any]] = []
    train_losses: list[float] = []
    token_rates: list[float] = []
    cache_hits = 0
    cache_events = 0
    graph_build_ms_values: list[float] = []

    for step in range(1, steps + 1):
        xb, yb = sample_batch(train_data, batch_size=batch_size, seq_len=cfg.seq_len, rng=rng)

        if attn_mode.startswith("wayfinder"):
            scheduled_window_drop = linear_schedule(
                step,
                start=0.0,
                end=window_drop_max,
                steps=max(1, window_drop_warm_steps),
            )
            scheduled_bias = {
                "cycle": linear_schedule(step, start=0.0, end=bias_cycle_max, steps=max(1, bias_warm_steps)),
                "window": linear_schedule(step, start=0.0, end=bias_window_min, steps=max(1, bias_warm_steps)),
                "landmark": linear_schedule(step, start=0.0, end=bias_landmark_max, steps=max(1, bias_warm_steps)),
            }
            model.set_wayfinder_runtime_controls(
                window_drop=scheduled_window_drop,
                schedule_bias=scheduled_bias,
            )
        else:
            scheduled_window_drop = 0.0
            scheduled_bias = {"cycle": 0.0, "window": 0.0, "landmark": 0.0}

        t0 = time.perf_counter()
        loss, grads = value_and_grad(
            model,
            xb,
            yb,
            bool(attn_mode.startswith("wayfinder") and reliance_reg_coeff > 0.0),
            step,
        )
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        dt = time.perf_counter() - t0

        loss_v = float(loss.item())
        tok_s = float((batch_size * cfg.seq_len) / max(dt, 1e-12))
        token_rates.append(tok_s)
        train_losses.append(loss_v)

        rec: Dict[str, Any] = {
            "run": run_name,
            "attn": attn_mode,
            "step": step,
            "train_loss": loss_v,
            "train_ppl": float(math.exp(min(loss_v, 20.0))),
            "tokens_per_sec": tok_s,
        }

        if attn_mode.startswith("wayfinder"):
            rec["window_drop_scheduled"] = float(scheduled_window_drop)
            rec["bias_cycle_scheduled"] = float(scheduled_bias["cycle"])
            rec["bias_window_scheduled"] = float(scheduled_bias["window"])
            rec["bias_landmark_scheduled"] = float(scheduled_bias["landmark"])

            attn_mod = _first_wayfinder_attn(model)
            if attn_mod is not None:
                prof = attn_mod.last_profile.to_dict()
                cache_hit = bool(prof.get("cache_hit", False))
                cache_events += 1
                cache_hits += int(cache_hit)
                gb = float(prof.get("graph_build_ms", 0.0))
                graph_build_ms_values.append(gb)
                rec["cache_hit"] = cache_hit
                rec["graph_build_ms"] = gb
                rec["cache_source"] = prof.get("cache_source")

        if step % checkpoint_every == 0 or step == steps:
            ckpt_path = _save_checkpoint(model, opt, metrics_path.parent, step)
            rec["checkpoint"] = str(ckpt_path)

        if step % eval_every == 0 or step == steps:
            val_loss = eval_loss(
                model,
                val_data,
                batch_size=batch_size,
                seq_len=cfg.seq_len,
                eval_batches=eval_batches,
                rng=rng,
            )
            rec["val_loss"] = val_loss
            rec["val_ppl"] = float(math.exp(min(val_loss, 20.0)))

            if attn_mode.startswith("wayfinder"):
                try:
                    xb_probe, _ = sample_batch(train_data, batch_size=1, seq_len=cfg.seq_len, rng=rng)
                    probe_out = model(xb_probe, return_debug=True)
                    mx.eval(probe_out["logits"])
                    b0 = probe_out["debug"]["blocks"][0]["attn"]
                    w_probe = b0.get("attn_weights")
                    et_probe = b0.get("edge_type")
                    if w_probe is not None and et_probe is not None:
                        mx.eval(w_probe, et_probe)
                        util = edge_utilization_by_type(
                            np.asarray(w_probe, dtype=np.float32),
                            np.asarray(et_probe, dtype=np.uint8),
                        )
                        rec["edge_utilization"] = util

                    graph_abi = b0.get("graph_abi")
                    if graph_abi is not None:
                        rec["graph_metrics"] = graph_metrics(graph_abi)
                except Exception:
                    pass

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        step_records.append(rec)

    summary: Dict[str, Any] = {
        "run": run_name,
        "attn": attn_mode,
        "steps": steps,
        "final_train_loss": float(train_losses[-1]),
        "final_train_ppl": float(math.exp(min(train_losses[-1], 20.0))),
        "avg_tokens_per_sec": float(np.mean(token_rates)),
        "last_tokens_per_sec": float(token_rates[-1]),
        "peak_memory_bytes": int(_peak_memory()),
    }

    eval_points = [r for r in step_records if "val_loss" in r]
    if eval_points:
        summary["final_val_loss"] = float(eval_points[-1]["val_loss"])
        summary["final_val_ppl"] = float(eval_points[-1]["val_ppl"])
        summary["best_val_ppl"] = float(min(r["val_ppl"] for r in eval_points))

    if attn_mode.startswith("wayfinder"):
        summary["cache_hit_rate"] = float(cache_hits / max(cache_events, 1))
        summary["avg_graph_build_ms"] = float(
            np.mean(graph_build_ms_values) if graph_build_ms_values else 0.0
        )
        attn_mod = _first_wayfinder_attn(model)
        if attn_mod is not None:
            summary["cache_persistent_bytes"] = int(attn_mod.cache_persistent_bytes())

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Long tiny MLX dense vs Wayfinder experiment")
    p.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--landmark-stride", type=int, default=32)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--wayfinder-attn", type=str, default="wayfinder_sparse", choices=["wayfinder_sparse", "wayfinder_permute"])
    p.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "greedy", "online_insertion"],
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)

    # Window budget / edge-bias schedules.
    p.add_argument("--window-drop-max", type=float, default=0.5)
    p.add_argument("--window-drop-warm-frac", type=float, default=0.6)
    p.add_argument("--bias-cycle-max", type=float, default=0.4)
    p.add_argument("--bias-window-min", type=float, default=-0.1)
    p.add_argument("--bias-landmark-max", type=float, default=0.1)
    p.add_argument("--bias-warm-frac", type=float, default=0.6)

    # Lightweight edge reliance regularizer.
    p.add_argument("--reliance-reg-coeff", type=float, default=0.01)
    p.add_argument("--reliance-cycle-min", type=float, default=0.10)
    p.add_argument("--reliance-window-max", type=float, default=0.85)

    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--retro-backfill-enabled", type=_parse_bool, default=False)
    p.add_argument("--retro-backfill-alpha", type=float, default=0.0)
    p.add_argument("--retro-backfill-training-only", type=_parse_bool, default=True)
    p.add_argument("--retro-backfill-causal-only", type=_parse_bool, default=True)
    args = p.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    text = load_text(args.data)
    train_data, val_data, tok = build_data(text)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir or (Path("runs/mlx") / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    compiled_graph_dir: Optional[str] = None
    if args.graph_spec is not None:
        compiled = compile_graph_spec(
            args.graph_spec,
            T=args.seq_len,
            H=args.heads,
            out_root=args.graph_cache_root,
        )
        compiled_graph_dir = str(compiled["artifact"]["artifact_dir"])

    cfg_base = GPTConfigMLX(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        n_layers=args.layers,
        n_heads=args.heads,
        n_embd=args.embd,
        dropout=0.0,
        strategy=args.strategy,
        window=args.window,
        landmark_stride=None if args.landmark_stride <= 0 else args.landmark_stride,
        num_cycles=args.num_cycles,
        seed=args.seed,
        edge_bias=True,
        window_drop=0.0,
        graph_spec=None if args.graph_spec is None else str(args.graph_spec),
        compiled_graph_dir=compiled_graph_dir,
        retro_backfill_enabled=bool(args.retro_backfill_enabled),
        retro_backfill_alpha=float(args.retro_backfill_alpha),
        retro_backfill_training_only=bool(args.retro_backfill_training_only),
        retro_backfill_causal_only=bool(args.retro_backfill_causal_only),
    )

    config_payload = {
        "timestamp": stamp,
        "data": str(args.data),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
        "tokenizer": "char",
        "args": _jsonable(vars(args)),
        "base_config": asdict(cfg_base),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = np.random.default_rng(args.seed)

    warm_steps_drop = max(1, int(args.steps * args.window_drop_warm_frac))
    warm_steps_bias = max(1, int(args.steps * args.bias_warm_frac))

    dense_cfg = GPTConfigMLX(**{**asdict(cfg_base), "attn": "dense", "edge_bias": False})
    way_cfg = GPTConfigMLX(**{**asdict(cfg_base), "attn": args.wayfinder_attn, "edge_bias": True})

    dense_summary = train_one(
        run_name="dense",
        attn_mode="dense",
        cfg=dense_cfg,
        train_data=train_data,
        val_data=val_data,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        rng=rng,
        metrics_path=metrics_path,
        window_drop_max=0.0,
        window_drop_warm_steps=1,
        bias_cycle_max=0.0,
        bias_window_min=0.0,
        bias_landmark_max=0.0,
        bias_warm_steps=1,
        reliance_reg_coeff=0.0,
        reliance_cycle_min=0.0,
        reliance_window_max=1.0,
    )

    way_summary = train_one(
        run_name="wayfinder",
        attn_mode=args.wayfinder_attn,
        cfg=way_cfg,
        train_data=train_data,
        val_data=val_data,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        rng=rng,
        metrics_path=metrics_path,
        window_drop_max=args.window_drop_max,
        window_drop_warm_steps=warm_steps_drop,
        bias_cycle_max=args.bias_cycle_max,
        bias_window_min=args.bias_window_min,
        bias_landmark_max=args.bias_landmark_max,
        bias_warm_steps=warm_steps_bias,
        reliance_reg_coeff=args.reliance_reg_coeff,
        reliance_cycle_min=args.reliance_cycle_min,
        reliance_window_max=args.reliance_window_max,
    )

    summary = {
        "dense": dense_summary,
        "wayfinder": way_summary,
        "throughput_ratio_wayfinder_over_dense": float(
            way_summary["avg_tokens_per_sec"] / max(dense_summary["avg_tokens_per_sec"], 1e-12)
        ),
        "val_ppl_gap_wayfinder_minus_dense": float(
            way_summary.get("final_val_ppl", float("nan"))
            - dense_summary.get("final_val_ppl", float("nan"))
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Dense final val ppl: {dense_summary.get('final_val_ppl', float('nan')):.3f}")
    print(f"Wayfinder final val ppl: {way_summary.get('final_val_ppl', float('nan')):.3f}")
    print(f"Dense avg tok/s: {dense_summary['avg_tokens_per_sec']:.1f}")
    print(f"Wayfinder avg tok/s: {way_summary['avg_tokens_per_sec']:.1f}")
    print(
        "PPL gap (wayfinder - dense): "
        f"{summary['val_ppl_gap_wayfinder_minus_dense']:.3f}"
    )


if __name__ == "__main__":
    main()
