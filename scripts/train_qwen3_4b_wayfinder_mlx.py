#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.compiler import compile_graph_spec
from bna.integrations.qwen_mlx import QwenWayfinderConfig, QwenWayfinderAttention, swap_qwen_attention_with_wayfinder


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:  # pragma: no cover
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())  # pragma: no cover


def _linear_schedule(step: int, start: float, end: float, warmup_steps: int) -> float:
    if warmup_steps <= 1:
        return float(end)
    alpha = min(1.0, max(0.0, float(step - 1) / float(warmup_steps - 1)))
    return float(start + alpha * (end - start))


def _load_sequences(bin_path: Path, seq_len: int) -> np.ndarray:
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    n_tokens = bin_path.stat().st_size // np.dtype(np.uint32).itemsize
    if n_tokens % seq_len != 0:
        raise ValueError(
            f"{bin_path} token count {n_tokens} is not divisible by seq_len={seq_len}"
        )
    n_seq = n_tokens // seq_len
    arr = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(n_seq, seq_len))
    return arr


def _sample_batch(
    sequences: np.ndarray,
    *,
    batch_size: int,
    sample_seq_len: int,
    rng: np.random.Generator,
) -> mx.array:
    n_seq, base_len = sequences.shape
    if sample_seq_len > base_len:
        raise ValueError(f"sample_seq_len={sample_seq_len} exceeds base dataset seq_len={base_len}")

    seq_idx = rng.integers(0, n_seq, size=batch_size)
    if sample_seq_len == base_len:
        batch = sequences[seq_idx]
    else:
        max_start = base_len - sample_seq_len
        starts = rng.integers(0, max_start + 1, size=batch_size)
        chunks = []
        for i, s in enumerate(starts):
            chunks.append(sequences[seq_idx[i], s : s + sample_seq_len])
        batch = np.stack(chunks, axis=0)
    return mx.array(batch.astype(np.int32), dtype=mx.int32)


def _compute_loss(model: nn.Module, tokens: mx.array) -> mx.array:
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    y_flat = y.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(logits_flat, y_flat))


def _collect_wayfinder_modules(model: nn.Module) -> List[QwenWayfinderAttention]:
    out: list[QwenWayfinderAttention] = []
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, QwenWayfinderAttention):
            out.append(attn)
    return out


def _set_wayfinder_runtime_controls(
    modules: List[QwenWayfinderAttention],
    *,
    window_drop: float,
    schedule_bias: Dict[str, float],
) -> None:
    for mod in modules:
        mod.set_runtime_controls(window_drop=window_drop, schedule_bias=schedule_bias)


def _save_adapters(model: nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_path), adapter_weights)


def _evaluate(
    model: nn.Module,
    val_sequences: np.ndarray,
    *,
    eval_seq_len: int,
    eval_batches: int,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(max(1, eval_batches)):
        xb = _sample_batch(
            val_sequences,
            batch_size=batch_size,
            sample_seq_len=eval_seq_len,
            rng=rng,
        )
        loss = _compute_loss(model, xb)
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-4B long-context LoRA training with HCSA")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--warmup-seq-len", type=int, default=8192)
    p.add_argument("--warmup-steps", type=int, default=80)
    p.add_argument("--eval-seq-len", type=int, default=32768)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-lr-steps", type=int, default=20)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume-adapter-file", type=Path, default=None)

    p.add_argument("--wayfinder-mode", type=str, default="permute", choices=["none", "sparse", "permute"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--swap-last-n-layers", type=int, default=-1)
    p.add_argument("--graph-spec", type=Path, default=Path("configs/graph_specs/default.wf"))
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))

    p.add_argument("--window-drop-start", type=float, default=0.0)
    p.add_argument("--window-drop-end", type=float, default=0.25)
    p.add_argument("--window-drop-warmup-steps", type=int, default=100)
    p.add_argument("--edge-bias-cycle-start", type=float, default=0.0)
    p.add_argument("--edge-bias-cycle-end", type=float, default=0.4)
    p.add_argument("--edge-bias-window-start", type=float, default=0.0)
    p.add_argument("--edge-bias-window-end", type=float, default=-0.1)
    p.add_argument("--edge-bias-landmark-start", type=float, default=0.0)
    p.add_argument("--edge-bias-landmark-end", type=float, default=0.0)
    p.add_argument("--edge-bias-warmup-steps", type=int, default=100)

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-scale", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--num-lora-layers", type=int, default=8)
    args = p.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    meta_path = args.dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_seq_len = int(meta["seq_len"])

    train_sequences = _load_sequences(args.dataset_dir / "train.bin", base_seq_len)
    val_long_path = args.dataset_dir / "val_long.bin"
    val_path = val_long_path if val_long_path.exists() else (args.dataset_dir / "val.bin")
    val_sequences = _load_sequences(val_path, base_seq_len)

    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    compiled_graph_dir: Optional[str] = None
    if args.wayfinder_mode != "none" and args.graph_spec is not None and args.graph_spec.exists():
        compiled = compile_graph_spec(
            args.graph_spec,
            T=int(args.seq_len),
            H=int(config["num_attention_heads"]),
            out_root=args.graph_cache_root,
        )
        compiled_graph_dir = str(compiled["artifact"]["artifact_dir"])

    replaced_layers: List[int] = []
    if args.wayfinder_mode != "none":
        all_layers = list(range(len(model.layers)))
        if args.swap_last_n_layers > 0:
            layer_indices = all_layers[-int(args.swap_last_n_layers) :]
        else:
            layer_indices = all_layers
        wf_cfg = QwenWayfinderConfig(
            path=args.wayfinder_mode,  # type: ignore[arg-type]
            strategy="random",
            window=int(args.window),
            landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
            num_cycles=int(args.num_cycles),
            seed=int(args.seed),
            edge_bias=True,
            window_drop=float(args.window_drop_start),
            compiled_graph_dir=compiled_graph_dir,
        )
        replaced_layers = swap_qwen_attention_with_wayfinder(
            model,
            cfg=wf_cfg,
            layer_indices=layer_indices,
        )

    model.freeze()
    linear_to_lora_layers(
        model,
        int(args.num_lora_layers),
        {
            "rank": int(args.lora_rank),
            "scale": float(args.lora_scale),
            "dropout": float(args.lora_dropout),
        },
        use_dora=False,
    )

    if args.resume_adapter_file is not None:
        model.load_weights(str(args.resume_adapter_file), strict=False)

    model.train()
    print_trainable_parameters(model)

    opt = optim.AdamW(learning_rate=float(args.lr), weight_decay=0.0)
    value_and_grad = nn.value_and_grad(model, _compute_loss)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("runs/mlx") / f"qwen3_4b_wayfinder_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "adapters"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "args": vars(args),
        "model_path": args.model_path,
        "model_config": {
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "hidden_size": config.get("hidden_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_scaling": config.get("rope_scaling"),
        },
        "dataset_dir": str(args.dataset_dir),
        "dataset_meta": meta,
        "compiled_graph_dir": compiled_graph_dir,
        "replaced_layers": replaced_layers,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    wf_modules = _collect_wayfinder_modules(model)
    cache_hits = 0
    cache_events = 0
    graph_build_vals: list[float] = []

    for step in range(1, int(args.steps) + 1):
        train_seq_len = (
            int(args.warmup_seq_len)
            if step <= int(args.warmup_steps)
            else int(args.seq_len)
        )
        train_seq_len = min(train_seq_len, base_seq_len)

        window_drop = _linear_schedule(
            step,
            float(args.window_drop_start),
            float(args.window_drop_end),
            int(args.window_drop_warmup_steps),
        )
        edge_bias = {
            "cycle": _linear_schedule(
                step,
                float(args.edge_bias_cycle_start),
                float(args.edge_bias_cycle_end),
                int(args.edge_bias_warmup_steps),
            ),
            "window": _linear_schedule(
                step,
                float(args.edge_bias_window_start),
                float(args.edge_bias_window_end),
                int(args.edge_bias_warmup_steps),
            ),
            "landmark": _linear_schedule(
                step,
                float(args.edge_bias_landmark_start),
                float(args.edge_bias_landmark_end),
                int(args.edge_bias_warmup_steps),
            ),
        }
        if wf_modules:
            _set_wayfinder_runtime_controls(
                wf_modules,
                window_drop=window_drop,
                schedule_bias=edge_bias,
            )

        lr_now = _linear_schedule(step, 0.0, float(args.lr), int(args.warmup_lr_steps))
        opt.learning_rate = lr_now

        _reset_peak_memory()
        t0 = time.perf_counter()
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)
        for _ in range(max(1, int(args.grad_accum))):
            xb = _sample_batch(
                train_sequences,
                batch_size=int(args.batch_size),
                sample_seq_len=train_seq_len,
                rng=rng,
            )
            loss, grads = value_and_grad(model, xb)
            accum_loss = accum_loss + loss.astype(mx.float32)
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

        if int(args.grad_accum) > 1:
            accum_grads = tree_map(lambda g: g / float(args.grad_accum), accum_grads)

        opt.update(model, accum_grads)
        mx.eval(accum_loss, model.parameters(), opt.state)
        dt = time.perf_counter() - t0

        loss_v = float((accum_loss / float(max(1, int(args.grad_accum)))).item())
        ppl_v = float(math.exp(min(20.0, loss_v)))
        tok_s = float((int(args.batch_size) * train_seq_len * int(args.grad_accum)) / max(dt, 1e-12))
        peak_mem = _peak_memory()

        edge_util_avg = {"cycle": 0.0, "window": 0.0, "landmark": 0.0, "rewire": 0.0}
        reachability = 0.0
        shortcut = 0.0
        step_graph_build = 0.0
        step_cache_hit = 0.0
        if wf_modules:
            gb_vals: list[float] = []
            hit_vals: list[float] = []
            cyc = win = lmk = rew = 0.0
            for mod in wf_modules:
                prof = mod.last_profile.to_dict()
                gb = float(prof.get("graph_build_ms", 0.0))
                hit = float(1.0 if bool(prof.get("cache_hit", False)) else 0.0)
                gb_vals.append(gb)
                hit_vals.append(hit)
                cyc += float(mod.last_edge_utilization_proxy.get("cycle", 0.0))
                win += float(mod.last_edge_utilization_proxy.get("window", 0.0))
                lmk += float(mod.last_edge_utilization_proxy.get("landmark", 0.0))
                rew += float(mod.last_edge_utilization_proxy.get("rewire", 0.0))

            nmods = float(len(wf_modules))
            step_graph_build = float(np.mean(gb_vals)) if gb_vals else 0.0
            step_cache_hit = float(np.mean(hit_vals)) if hit_vals else 0.0
            edge_util_avg = {
                "cycle": cyc / nmods,
                "window": win / nmods,
                "landmark": lmk / nmods,
                "rewire": rew / nmods,
            }

            cache_events += len(hit_vals)
            cache_hits += int(sum(1 for h in hit_vals if h > 0.5))
            graph_build_vals.extend(gb_vals)
            reachability = float(wf_modules[0].last_graph_metrics.get("reachability_proxy", 0.0))
            shortcut = float(wf_modules[0].last_graph_metrics.get("shortcut_rate", 0.0))

        rec: Dict[str, Any] = {
            "step": int(step),
            "train_seq_len": int(train_seq_len),
            "eval_seq_len": int(args.eval_seq_len),
            "loss": loss_v,
            "ppl": ppl_v,
            "tokens_per_sec": tok_s,
            "peak_memory_bytes": int(peak_mem),
            "learning_rate": float(lr_now),
            "window_drop": float(window_drop),
            "edge_bias": edge_bias,
            "graph_build_ms": float(step_graph_build),
            "cache_hit_rate_step": float(step_cache_hit),
            "edge_utilization_proxy": edge_util_avg,
            "reachability_proxy": float(reachability),
            "shortcut_rate": float(shortcut),
        }

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            eval_seq = min(int(args.eval_seq_len), base_seq_len)
            val_loss = _evaluate(
                model,
                val_sequences,
                eval_seq_len=eval_seq,
                eval_batches=int(args.eval_batches),
                batch_size=1,
                rng=rng,
            )
            rec["val_loss"] = float(val_loss)
            rec["val_ppl"] = float(math.exp(min(20.0, val_loss)))

        if step % int(args.save_every) == 0 or step == int(args.steps):
            step_path = ckpt_dir / f"{step:07d}_adapters.safetensors"
            latest_path = ckpt_dir / "adapters.safetensors"
            _save_adapters(model, step_path)
            _save_adapters(model, latest_path)
            rec["adapter_checkpoint"] = str(step_path)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(
            f"step={step} seq={train_seq_len} loss={loss_v:.4f} "
            f"ppl={ppl_v:.3f} tok/s={tok_s:.1f} cache_hit={step_cache_hit:.3f}"
        )

    summary = {
        "run_dir": str(run_dir),
        "steps": int(args.steps),
        "model_path": args.model_path,
        "dataset_dir": str(args.dataset_dir),
        "train_seq_len_final": int(args.seq_len),
        "eval_seq_len": int(args.eval_seq_len),
        "cache_hit_rate": float(cache_hits / max(cache_events, 1)),
        "avg_graph_build_ms": float(np.mean(graph_build_vals) if graph_build_vals else 0.0),
        "replaced_layers": replaced_layers,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
