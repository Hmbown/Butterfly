#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.cycles import recommended_num_cycles
from bna.integrations.gpt2_mlx import GPT2WayfinderConfig, swap_gpt2_attention_with_wayfinder


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:  # pragma: no cover
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())  # pragma: no cover


def _clear_cache() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:  # pragma: no cover
        mx.metal.clear_cache()


def _load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _build_dataset(
    tokenizer,
    text: str,
    val_fraction: float,
    *,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    ids_raw = np.asarray(tokenizer.encode(text), dtype=np.int32)
    raw_tokens = int(len(ids_raw))
    if raw_tokens < 2:
        raise ValueError("Dataset too small to tokenize.")

    min_fraction = min(val_fraction, 1.0 - val_fraction)
    min_tokens = int(math.ceil(float(seq_len + 1) / max(min_fraction, 1e-6)))
    repeats = 1
    if raw_tokens < min_tokens:
        repeats = int(math.ceil(float(min_tokens) / float(raw_tokens)))
        ids = np.tile(ids_raw, repeats)
    else:
        ids = ids_raw

    n_val = max(1, int(len(ids) * val_fraction))
    if len(ids) <= n_val + 1:
        raise ValueError("Dataset too small for train/val split.")
    train = ids[:-n_val]
    val = ids[-n_val:]
    meta = {"raw_tokens": raw_tokens, "total_tokens": int(len(ids)), "repeats": repeats}
    return train, val, meta


def _sample_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[mx.array, mx.array]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Data too small for seq_len={seq_len}")
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts]).astype(np.int32)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int32)
    return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def _compute_loss(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
    logits = model(xb)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = yb.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))


def _evaluate(
    model: nn.Module,
    val_data: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(max(1, eval_batches)):
        xb, yb = _sample_batch(
            val_data,
            batch_size=batch_size,
            seq_len=seq_len,
            rng=rng,
        )
        loss = _compute_loss(model, xb, yb)
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def _cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return float(base_lr)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _landmark_stride_for_budget(
    *,
    seq_len: int,
    max_degree: int,
    window: int,
    num_cycles: int,
) -> Optional[int]:
    budget = int(max_degree) - int(window) - 1 - 2 * int(num_cycles)
    if budget <= 0:
        return None
    stride = math.ceil(float(seq_len - 1) / float(budget))
    return int(max(1, stride))


def _expected_max_degree(
    *,
    seq_len: int,
    window: int,
    num_cycles: int,
    landmark_stride: Optional[int],
) -> int:
    landmark_edges = 0 if landmark_stride is None else (seq_len - 1) // int(landmark_stride)
    return int(window) + 1 + 2 * int(num_cycles) + int(landmark_edges)


def _make_wayfinder_cfg(
    *,
    mode: str,
    seq_len: int,
    window: int,
    max_degree: int,
    seed: int,
) -> tuple[GPT2WayfinderConfig, Dict[str, Any]]:
    edge_disjoint = True
    if mode == "landmarks":
        num_cycles = 0
        enforce_hamiltonian = False
        landmark_stride = _landmark_stride_for_budget(
            seq_len=seq_len,
            max_degree=max_degree,
            window=window,
            num_cycles=num_cycles,
        )
        cycles_for_degree = num_cycles
    elif mode == "cycle":
        num_cycles = 1
        enforce_hamiltonian = True
        landmark_stride = _landmark_stride_for_budget(
            seq_len=seq_len,
            max_degree=max_degree,
            window=window,
            num_cycles=num_cycles,
        )
        cycles_for_degree = num_cycles
    elif mode == "multicycle":
        num_cycles = "auto"
        enforce_hamiltonian = True
        edge_disjoint = False
        landmark_stride = None
        cycles_for_degree = recommended_num_cycles(seq_len)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    wf_cfg = GPT2WayfinderConfig(
        path="sparse",
        strategy="random",
        window=int(window),
        landmark_stride=landmark_stride,
        num_cycles=num_cycles,
        seed=int(seed),
        edge_disjoint=edge_disjoint,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        enforce_hamiltonian=enforce_hamiltonian,
    )

    expected_degree = _expected_max_degree(
        seq_len=seq_len,
        window=window,
        num_cycles=cycles_for_degree,
        landmark_stride=landmark_stride,
    )
    meta = {
        "mode": mode,
        "target_max_degree": int(max_degree),
        "expected_max_degree": int(expected_degree),
        "resolved_cycles_for_degree": int(cycles_for_degree),
        "landmark_stride": None if landmark_stride is None else int(landmark_stride),
        "enforce_hamiltonian": bool(enforce_hamiltonian),
        "edge_disjoint": bool(edge_disjoint),
    }
    return wf_cfg, meta


def _train_one(
    *,
    run_name: str,
    model_path: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    seq_len: int,
    steps: int,
    batch_size: int,
    eval_every: int,
    eval_batches: int,
    lr: float,
    warmup_steps: int,
    seed: int,
    metrics_path: Path,
    wayfinder_cfg: Optional[GPT2WayfinderConfig],
) -> Dict[str, Any]:
    np.random.seed(seed)
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tokenizer, config = mlx_load(
        model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    replaced_layers = []
    if wayfinder_cfg is not None:
        replaced_layers = swap_gpt2_attention_with_wayfinder(
            model,
            cfg=wayfinder_cfg,
            layer_indices=None,
        )

    model.train()
    opt = optim.AdamW(learning_rate=float(lr), weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, _compute_loss)

    train_losses: list[float] = []
    tokens_per_sec: list[float] = []
    final_val_loss: Optional[float] = None

    for step in range(1, steps + 1):
        xb, yb = _sample_batch(train_data, batch_size=batch_size, seq_len=seq_len, rng=rng)

        opt.learning_rate = _cosine_lr(step, lr, warmup_steps, steps)
        _reset_peak_memory()
        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, xb, yb)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        dt = time.perf_counter() - t0

        loss_v = float(loss.item())
        train_losses.append(loss_v)
        tok_s = float((batch_size * seq_len) / max(dt, 1e-12))
        tokens_per_sec.append(tok_s)

        rec: Dict[str, Any] = {
            "run": run_name,
            "step": int(step),
            "train_loss": loss_v,
            "train_ppl": float(math.exp(min(loss_v, 20.0))),
            "tokens_per_sec": tok_s,
            "learning_rate": float(opt.learning_rate),
            "peak_memory_bytes": int(_peak_memory()),
            "replaced_layers": replaced_layers,
        }

        if step % eval_every == 0 or step == steps:
            val_loss = _evaluate(
                model,
                val_data,
                batch_size=batch_size,
                seq_len=seq_len,
                eval_batches=eval_batches,
                rng=rng,
            )
            rec["val_loss"] = float(val_loss)
            rec["val_ppl"] = float(math.exp(min(val_loss, 20.0)))
            final_val_loss = float(val_loss)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if step % eval_every == 0 or step == steps:
            val_ppl = rec.get("val_ppl", float("nan"))
            print(
                f"[{run_name}] step={step} loss={loss_v:.4f} val_ppl={val_ppl:.3f} "
                f"tok/s={tok_s:.1f}",
                flush=True,
            )

    summary = {
        "run": run_name,
        "steps": int(steps),
        "final_train_loss": float(train_losses[-1]),
        "final_train_ppl": float(math.exp(min(train_losses[-1], 20.0))),
        "avg_tokens_per_sec": float(np.mean(tokens_per_sec)),
        "peak_memory_bytes": int(_peak_memory()),
        "replaced_layers": replaced_layers,
        "final_val_loss": final_val_loss,
        "final_val_ppl": float(math.exp(min(final_val_loss, 20.0))) if final_val_loss else None,
        "model_config": {
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "hidden_size": config.get("hidden_size"),
        },
        "tokenizer_vocab": int(getattr(tokenizer, "vocab_size", 0) or 0),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Finetune GPT-2 with sparse Wayfinder attention (MLX)")
    p.add_argument("--model-path", type=str, default="openai-community/gpt2")
    p.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--max-degree", type=int, default=130)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    text = _load_text(args.data)
    model, tokenizer, _ = mlx_load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    del model
    train_data, val_data, data_meta = _build_dataset(
        tokenizer,
        text,
        args.val_fraction,
        seq_len=args.seq_len,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("results/finetune_sparse_comparison") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    config_payload: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "model_path": args.model_path,
        "data": str(args.data),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
        "raw_tokens": int(data_meta["raw_tokens"]),
        "total_tokens": int(data_meta["total_tokens"]),
        "dataset_repeats": int(data_meta["repeats"]),
        "seq_len": int(args.seq_len),
        "window": int(args.window),
        "target_max_degree": int(args.max_degree),
        "args": _jsonable(vars(args)),
        "runs": [],
    }

    run_specs = [
        ("dense", None, {}),
        ("window_landmarks", "landmarks", {}),
        ("window_cycle", "cycle", {}),
        ("window_multicycle", "multicycle", {}),
    ]

    summaries: Dict[str, Any] = {}
    for run_name, mode, _extra in run_specs:
        if mode is None:
            wf_cfg = None
            meta = {"mode": "dense"}
        else:
            wf_cfg, meta = _make_wayfinder_cfg(
                mode=mode,
                seq_len=args.seq_len,
                window=args.window,
                max_degree=args.max_degree,
                seed=args.seed,
            )
            meta["wayfinder_config"] = dict(wf_cfg.__dict__)
        config_payload["runs"].append({"run": run_name, **meta})

        summary = _train_one(
            run_name=run_name,
            model_path=args.model_path,
            train_data=train_data,
            val_data=val_data,
            seq_len=args.seq_len,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            metrics_path=metrics_path,
            wayfinder_cfg=wf_cfg,
        )
        summaries[run_name] = summary
        _clear_cache()

    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"Results saved to: {out_dir}")
    print("\nFinal comparison:")
    header = f"{'config':<20} {'val_ppl':>10} {'train_loss':>12} {'tok/s':>10}"
    print(header)
    print("-" * len(header))
    for name in ["dense", "window_landmarks", "window_cycle", "window_multicycle"]:
        summ = summaries.get(name, {})
        val_ppl = summ.get("final_val_ppl")
        train_loss = summ.get("final_train_loss")
        tok_s = summ.get("avg_tokens_per_sec")
        print(
            f"{name:<20} "
            f"{(float(val_ppl) if val_ppl is not None else float('nan')):>10.3f} "
            f"{(float(train_loss) if train_loss is not None else float('nan')):>12.4f} "
            f"{(float(tok_s) if tok_s is not None else float('nan')):>10.1f}"
        )


if __name__ == "__main__":
    main()
