#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from hcsa.torch.attention_dense import dense_causal_attention
from hcsa.torch.attention_wayfinder_permute import wayfinder_permute_window_attention
from hcsa.torch.bench_utils import sync_device


def _bench(fn, *, warmup: int, iters: int, device: torch.device) -> tuple[float, int]:
    for _ in range(max(1, warmup)):
        y = fn()
        if isinstance(y, tuple):
            y = y[0]
        _ = y
        sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, tuple):
            y = y[0]
        _ = y
        sync_device(device)
        times.append(time.perf_counter() - t0)

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return float(statistics.median(times)), peak


def _find_qkv_projections(model: torch.nn.Module):
    for mod in model.modules():
        if all(hasattr(mod, name) for name in ("q_proj", "k_proj", "v_proj")):
            return mod
    return None


def _load_real_qkv(
    *,
    model_id: str,
    seq_len: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    attn = _find_qkv_projections(model)
    if attn is None:
        raise RuntimeError("Could not find q_proj/k_proj/v_proj in model")

    hidden_size = int(getattr(cfg, "hidden_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = hidden_size // num_heads

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)

    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    meta = {
        "source": "hf_model",
        "model_id": model_id,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }
    return q, k, v, meta


def _load_config_shaped_qkv(
    *,
    model_id: str,
    seq_len: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        hidden_size = int(getattr(cfg, "hidden_size"))
        num_heads = int(getattr(cfg, "num_attention_heads"))
    except Exception:
        # Fallback defaults close to Qwen3-4B shape.
        hidden_size = 2560
        num_heads = 32

    head_dim = hidden_size // num_heads

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    meta = {
        "source": "config_or_synthetic",
        "model_id": model_id,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }
    return q, k, v, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-shaped Torch attention microbench")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--load-model", action="store_true", help="Load the full HF model and extract real QKV projections")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    dtype = getattr(torch, args.dtype)

    if args.load_model:
        q, k, v, qkv_meta = _load_real_qkv(
            model_id=args.model_id,
            seq_len=args.seq_len,
            batch=args.batch,
            device=device,
            dtype=dtype,
        )
    else:
        q, k, v, qkv_meta = _load_config_shaped_qkv(
            model_id=args.model_id,
            seq_len=args.seq_len,
            batch=args.batch,
            device=device,
            dtype=dtype,
        )

    b, h, t, _dh = q.shape

    neigh = torch.zeros((h, t, 1), device=device, dtype=torch.long)
    edge = torch.full((h, t, 1), 2, device=device, dtype=torch.uint8)
    perms = [torch.randperm(t, device=device).tolist() for _ in range(h)]

    dense_s, dense_mem = _bench(
        lambda: dense_causal_attention(q, k, v, return_weights=False),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    perm_s, perm_mem = _bench(
        lambda: wayfinder_permute_window_attention(
            q,
            k,
            v,
            window=args.window,
            neigh_idx=neigh,
            edge_type=edge,
            graph_meta={"cycle_perms": perms},
            return_weights=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    result: Dict[str, Any] = {
        "config": {
            **vars(args),
            "dtype": args.dtype,
            "device": str(device),
            "out": None if args.out is None else str(args.out),
        },
        "qkv_meta": qkv_meta,
        "dense": {
            "tok_s": float((b * t) / max(dense_s, 1e-12)),
            "latency_ms": dense_s * 1000.0,
            "peak_memory_bytes": dense_mem,
        },
        "wayfinder_permute": {
            "tok_s": float((b * t) / max(perm_s, 1e-12)),
            "latency_ms": perm_s * 1000.0,
            "peak_memory_bytes": perm_mem,
            "window": args.window,
        },
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
