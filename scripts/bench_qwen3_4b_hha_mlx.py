#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.compiler import compile_graph_spec
from hcsa.integrations.qwen_mlx import (
    QwenHHAConfig,
    QwenWayfinderAttention,
    extract_qkv_from_qwen_attention,
    swap_qwen_attention_with_wayfinder,
)


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:  # pragma: no cover
        mx.metal.reset_peak_memory()


def _peak_memory() -> int:
    if hasattr(mx, "get_peak_memory"):
        return int(mx.get_peak_memory())
    return int(mx.metal.get_peak_memory())  # pragma: no cover


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _bench(fn, *, warmup: int, iters: int):
    for _ in range(max(1, warmup)):
        y = fn()
        mx.eval(y)

    _reset_peak_memory()
    times: list[float] = []
    outputs: list[mx.array] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - t0)
        outputs.append(y)
    return _median(times), _peak_memory(), outputs[-1]


def _bench_one_seq(
    model,
    *,
    seq_len: int,
    batch: int,
    warmup: int,
    iters: int,
    dtype_name: str,
    hha_cfg: QwenHHAConfig,
    graph_spec: Path | None,
    graph_cache_root: Path,
    full_swap: bool,
) -> Dict[str, Any]:
    dtype = getattr(mx, dtype_name)
    hidden_size = int(model.args.hidden_size)
    x = mx.random.normal((batch, seq_len, hidden_size), dtype=dtype)
    layer0 = model.layers[0]
    base_attn = layer0.self_attn

    if graph_spec is not None:
        compiled = compile_graph_spec(
            graph_spec,
            T=seq_len,
            H=int(base_attn.n_heads),
            out_root=graph_cache_root,
        )
        hha_cfg = QwenHHAConfig(
            **{**hha_cfg.__dict__, "compiled_graph_dir": str(compiled["artifact"]["artifact_dir"])}
        )

    hha_attn = QwenWayfinderAttention(base_attn, hha_cfg)

    def baseline_attn_fn():
        q, k, v = extract_qkv_from_qwen_attention(base_attn, x, cache=None)
        y = scaled_dot_product_attention(q, k, v, cache=None, scale=base_attn.scale, mask="causal")
        return base_attn.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))

    def hha_attn_fn():
        return hha_attn(x, mask="causal", cache=None)

    dense_s, dense_mem, dense_out = _bench(baseline_attn_fn, warmup=warmup, iters=iters)

    first_hha = hha_attn_fn()
    mx.eval(first_hha)
    first_profile = hha_attn.last_profile.to_dict()

    hha_s, hha_mem, hha_out = _bench(hha_attn_fn, warmup=max(0, warmup - 1), iters=iters)

    diff = mx.abs(dense_out.astype(mx.float32) - hha_out.astype(mx.float32))
    mae = mx.mean(diff)
    mx.eval(mae)

    profile = hha_attn.last_profile.to_dict()
    row: Dict[str, Any] = {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "level_a_real_qkv": {
            "sanity_mae": float(mae.item()),
            "baseline_attention": {
                "tokens_per_sec": float((batch * seq_len) / max(dense_s, 1e-12)),
                "latency_ms": float(dense_s * 1000.0),
                "peak_memory_bytes": int(dense_mem),
            },
            "hha_attention": {
                "tokens_per_sec": float((batch * seq_len) / max(hha_s, 1e-12)),
                "latency_ms": float(hha_s * 1000.0),
                "peak_memory_bytes": int(hha_mem),
                "graph_build_ms_first": float(first_profile.get("graph_build_ms", 0.0)),
                "graph_build_ms_cached": float(profile.get("graph_build_ms", 0.0)),
                "cache_hit_rate": float(1.0 if bool(profile.get("cache_hit", False)) else 0.0),
                "cache_source": profile.get("cache_source"),
            },
        },
    }

    def baseline_block_fn():
        return layer0(x, mask="causal", cache=None)

    base_block_s, base_block_mem, _ = _bench(
        baseline_block_fn,
        warmup=max(1, warmup // 2),
        iters=max(1, iters // 2),
    )

    orig_attn = layer0.self_attn
    layer0.self_attn = hha_attn
    try:
        def hha_block_fn():
            return layer0(x, mask="causal", cache=None)

        hha_block_s, hha_block_mem, _ = _bench(
            hha_block_fn,
            warmup=max(1, warmup // 2),
            iters=max(1, iters // 2),
        )
        row["block"] = {
            "baseline": {
                "tokens_per_sec": float((batch * seq_len) / max(base_block_s, 1e-12)),
                "latency_ms": float(base_block_s * 1000.0),
                "peak_memory_bytes": int(base_block_mem),
            },
            "hha": {
                "tokens_per_sec": float((batch * seq_len) / max(hha_block_s, 1e-12)),
                "latency_ms": float(hha_block_s * 1000.0),
                "peak_memory_bytes": int(hha_block_mem),
                "cache_persistent_bytes": int(hha_attn.cache_persistent_bytes()),
                "edge_utilization_proxy": hha_attn.last_edge_utilization_proxy,
                "graph_metrics": hha_attn.last_graph_metrics,
            },
        }
    finally:
        layer0.self_attn = orig_attn

    if full_swap:
        replaced = swap_qwen_attention_with_wayfinder(model, cfg=hha_cfg, layer_indices=None)
        z = mx.random.randint(0, int(model.args.vocab_size), shape=(1, min(256, seq_len)), dtype=mx.int32)
        _reset_peak_memory()
        t0 = time.perf_counter()
        out = model(z)
        mx.eval(out)
        row["level_b_full_swap_smoke"] = {
            "replaced_layers": replaced,
            "tokens_per_sec": float((z.shape[0] * z.shape[1]) / max(time.perf_counter() - t0, 1e-12)),
            "peak_memory_bytes": int(_peak_memory()),
            "seq_len": int(z.shape[1]),
        }

    return row


def _readme(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3-4B HHA MLX Benchmark")
    lines.append("")
    lines.append(f"- created_at: `{payload['created_at']}`")
    lines.append(f"- model_path: `{payload['model_path']}`")
    lines.append("")
    lines.append("| T | dense attn tok/s | hha attn tok/s | dense attn mem | hha attn mem |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in payload["results"]:
        if "error" in row:
            lines.append(f"| {row['seq_len']} | error | error | error | error |")
            continue
        a = row["level_a_real_qkv"]
        lines.append(
            "| {T} | {d:.1f} | {h:.1f} | {dm} | {hm} |".format(
                T=row["seq_len"],
                d=a["baseline_attention"]["tokens_per_sec"],
                h=a["hha_attention"]["tokens_per_sec"],
                dm=a["baseline_attention"]["peak_memory_bytes"],
                hm=a["hha_attention"]["peak_memory_bytes"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-4B HHA MLX benchmark")
    p.add_argument("--model-path", type=str, default="mlx-community/Qwen3-4B-4bit")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--path", type=str, default="permute", choices=["sparse", "permute"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-drop", type=float, default=0.0)
    p.add_argument("--graph-spec", type=Path, default=None)
    p.add_argument("--graph-cache-root", type=Path, default=Path(".cache/wayfinder"))
    p.add_argument("--full-swap", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    model, tokenizer, config = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    hha_cfg = QwenHHAConfig(
        path=args.path,  # type: ignore[arg-type]
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=int(args.num_cycles),
        seed=int(args.seed),
        edge_bias=True,
        window_drop=float(args.window_drop),
        compiled_graph_dir=None,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("benchmarks/mlx/qwen3_4b_hha") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    for T in args.seq_lens:
        try:
            row = _bench_one_seq(
                model,
                seq_len=int(T),
                batch=int(args.batch),
                warmup=int(args.warmup),
                iters=int(args.iters),
                dtype_name=args.dtype,
                hha_cfg=hha_cfg,
                graph_spec=args.graph_spec,
                graph_cache_root=args.graph_cache_root,
                full_swap=bool(args.full_swap),
            )
            rows.append(row)
        except Exception as exc:  # pragma: no cover - runtime dependent
            rows.append(
                {
                    "seq_len": int(T),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "model_path": args.model_path,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "model_config": {
            "model_type": config.get("model_type"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "hidden_size": config.get("hidden_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_scaling": config.get("rope_scaling"),
        },
        "hha_config": hha_cfg.__dict__,
        "results": rows,
    }

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(payload), encoding="utf-8")
    print(f"Wrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
