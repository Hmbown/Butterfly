#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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


def _first_wayfinder_attn(model: GPTMLX):
    for block in model.blocks:
        attn = getattr(block, "attn", None)
        if hasattr(attn, "last_profile"):
            return attn
    return None


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
    rng: np.random.Generator,
    metrics_path: Path,
) -> Dict[str, Any]:
    model = GPTMLX(cfg)
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    _reset_peak_memory()

    def loss_fn(m: GPTMLX, xb: mx.array, yb: mx.array) -> mx.array:
        out = m(xb, targets=yb)
        return out["loss"]

    value_and_grad = nn.value_and_grad(model, loss_fn)

    step_records: list[Dict[str, Any]] = []
    train_losses: list[float] = []
    token_rates: list[float] = []
    cache_hits = 0
    cache_events = 0
    graph_build_ms_values: list[float] = []
    cache_persistent_bytes = 0

    for step in range(1, steps + 1):
        xb, yb = sample_batch(train_data, batch_size=batch_size, seq_len=cfg.seq_len, rng=rng)

        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, xb, yb)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        dt = time.perf_counter() - t0

        loss_v = float(loss.item())
        tok_s = float((batch_size * cfg.seq_len) / max(dt, 1e-12))
        token_rates.append(tok_s)
        train_losses.append(loss_v)

        rec = {
            "run": run_name,
            "attn": attn_mode,
            "step": step,
            "train_loss": loss_v,
            "train_ppl": float(math.exp(min(loss_v, 20.0))),
            "tokens_per_sec": tok_s,
        }

        if attn_mode.startswith("wayfinder"):
            attn_mod = _first_wayfinder_attn(model)
            if attn_mod is not None:
                prof = attn_mod.last_profile.to_dict()
                cache_hit = bool(prof.get("cache_hit", False))
                cache_events += 1
                cache_hits += int(cache_hit)
                graph_build_ms = float(prof.get("graph_build_ms", 0.0))
                graph_build_ms_values.append(graph_build_ms)
                cache_persistent_bytes = int(prof.get("cache_persistent_bytes", 0))
                rec["cache_hit"] = cache_hit
                rec["graph_build_ms"] = graph_build_ms

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

            # Log edge utilization at eval checkpoints for Wayfinder runs
            if attn_mode.startswith("wayfinder"):
                try:
                    xb_probe, _ = sample_batch(
                        train_data, batch_size=1, seq_len=cfg.seq_len, rng=rng
                    )
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
    if attn_mode.startswith("wayfinder"):
        summary["cache_hit_rate"] = float(cache_hits / max(cache_events, 1))
        summary["avg_graph_build_ms"] = float(
            np.mean(graph_build_ms_values) if graph_build_ms_values else 0.0
        )
        summary["cache_persistent_bytes"] = int(cache_persistent_bytes)

    # Graph + utilization metrics for Wayfinder run.
    if attn_mode.startswith("wayfinder"):
        xb, _ = sample_batch(train_data, batch_size=1, seq_len=cfg.seq_len, rng=rng)
        out = model(xb, return_debug=True)
        mx.eval(out["logits"])
        block0 = out["debug"]["blocks"][0]["attn"]

        g_abi = block0.get("graph_abi")
        if g_abi is not None:
            summary["graph_metrics"] = graph_metrics(g_abi)

        w = block0.get("attn_weights")
        et = block0.get("edge_type")
        ni = block0.get("neigh_idx")
        if w is not None and et is not None and ni is not None:
            mx.eval(w, et, ni)
            w_np = np.asarray(w, dtype=np.float32)
            et_np = np.asarray(et, dtype=np.uint8)
            ni_np = np.asarray(ni, dtype=np.int32)
            summary["edge_utilization"] = edge_utilization_by_type(w_np, et_np)
            np.savez(
                metrics_path.parent / f"{run_name}_graph_debug.npz",
                neigh_idx=ni_np,
                edge_type=et_np,
                attn_weights=w_np,
            )

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run a tiny MLX dense vs Wayfinder experiment")
    p.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--landmark-stride", type=int, default=32)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument(
        "--strategy", type=str, default="random", choices=["random", "greedy", "online_insertion"]
    )
    p.add_argument(
        "--wayfinder-attn",
        type=str,
        default="wayfinder_sparse",
        choices=["wayfinder_sparse", "wayfinder_permute"],
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--window-drop", type=float, default=0.0, help="Window-drop regularization prob")
    p.add_argument("--edge-bias", action="store_true", help="Enable learnable edge-type bias")
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

    compiled_graph_dir: str | None = None
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
        window_drop=args.window_drop,
        edge_bias=args.edge_bias,
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

    dense_cfg = GPTConfigMLX(**{**asdict(cfg_base), "attn": "dense"})
    way_cfg = GPTConfigMLX(**{**asdict(cfg_base), "attn": args.wayfinder_attn})

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
        rng=rng,
        metrics_path=metrics_path,
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
        rng=rng,
        metrics_path=metrics_path,
    )

    summary = {
        "dense": dense_summary,
        "wayfinder": way_summary,
        "throughput_ratio_wayfinder_over_dense": float(
            way_summary["avg_tokens_per_sec"] / max(dense_summary["avg_tokens_per_sec"], 1e-12)
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Dense final val ppl: {dense_summary.get('final_val_ppl', float('nan')):.3f}")
    print(f"Wayfinder final val ppl: {way_summary.get('final_val_ppl', float('nan')):.3f}")
    print(f"Dense avg tok/s: {dense_summary['avg_tokens_per_sec']:.1f}")
    print(f"Wayfinder avg tok/s: {way_summary['avg_tokens_per_sec']:.1f}")
    if "graph_metrics" in way_summary:
        gm = way_summary["graph_metrics"]
        print(
            "Graph metrics: "
            f"shortcut_rate={gm['shortcut_rate']:.3f}, "
            f"reachability_proxy={gm['reachability_proxy']:.2f}"
        )


if __name__ == "__main__":
    main()
