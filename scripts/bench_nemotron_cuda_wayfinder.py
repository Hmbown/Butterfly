#!/usr/bin/env python3
"""Benchmark dense vs Wayfinder prefill on Nemotron 3 Super (CUDA).

Measures prefill-only forward pass latency at various sequence lengths,
comparing stock dense attention against Wayfinder permute-window sparse
attention on the attention blocks.

Usage:
    python scripts/bench_nemotron_cuda_wayfinder.py \
        --model-path ~/HF_Models/nvidia/Nemotron-3-Super-49B-v1 \
        --seq-lens 64 128 256 512 1024 2048 \
        --warmup 2 --repeats 5
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _make_dummy_input(tokenizer, seq_len: int, device: torch.device) -> dict:
    """Create a dummy input of the desired length by repeating a seed sentence."""
    seed = "The theory of sparse attention in transformer models is important "
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = (seq_len // len(seed_ids)) + 2
    ids = (seed_ids * repeats)[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids}


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_gb() -> dict[str, float]:
    """Return current/peak/free GPU memory in GB."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
        "total_gb": round(total, 2),
    }


def bench_prefill(
    model: torch.nn.Module,
    inputs: dict,
    *,
    warmup: int = 2,
    repeats: int = 5,
    label: str = "",
) -> Dict[str, Any]:
    """Run prefill-only forward passes and return timing stats."""
    seq_len = inputs["input_ids"].shape[1]

    for _ in range(warmup):
        with torch.inference_mode():
            _sync_cuda()
            _ = model(**inputs, use_cache=False)
            _sync_cuda()

    times_ms: List[float] = []
    for _ in range(repeats):
        _sync_cuda()
        gc.collect()
        torch.cuda.empty_cache()
        _sync_cuda()

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
        _sync_cuda()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    return {
        "label": label,
        "seq_len": seq_len,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": round(median_ms, 2),
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "all_ms": [round(t, 2) for t in times_ms],
        "gpu_mem": _gpu_mem_gb(),
    }


def collect_wayfinder_profiles(model: torch.nn.Module) -> List[Dict[str, Any]]:
    from hcsa.integrations.nemotron_h_torch import iter_nemotron_h_wayfinder_layers
    profiles = []
    for layer in iter_nemotron_h_wayfinder_layers(model):
        p = layer.last_profile
        profiles.append({
            "layer_idx": p.get("layer_idx"),
            "mode": p.get("mode"),
            "reason": p.get("reason"),
            "elapsed_ms": p.get("elapsed_ms"),
            "path": p.get("path"),
            "engine": p.get("engine"),
            "graph_cache_hit": p.get("graph_cache_hit"),
        })
    return profiles


def main() -> None:
    p = argparse.ArgumentParser(description="Bench dense vs Wayfinder prefill on Nemotron 3 Super CUDA")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--seq-lens", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--engine", type=str, default="auto",
                    choices=["auto", "flex", "batched", "legacy"],
                    help="Wayfinder engine: auto, flex, batched, or legacy.")
    p.add_argument("--output", type=str, default=None,
                    help="Path to save results (ndjson). Default: auto-generated.")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hcsa.integrations.nemotron_h_torch import (
        NemotronHWayfinderConfig,
        swap_nemotron_h_attention_with_wayfinder,
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    _log(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = Path(args.model_path).name
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = REPO_ROOT / "benchmarks" / "cuda" / "nemotron_wayfinder"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"EXP-{timestamp}-{model_name}.ndjson"

    results: List[Dict[str, Any]] = []
    experiment_meta = {
        "type": "experiment_meta",
        "timestamp": timestamp,
        "model_path": str(args.model_path),
        "model_name": model_name,
        "seq_lens": args.seq_lens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "window": args.window,
        "landmark_stride": args.landmark_stride,
        "num_cycles": args.num_cycles,
        "engine": args.engine,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    results.append(experiment_meta)
    _log(f"Experiment: {json.dumps(experiment_meta, indent=2)}")

    # ── Phase 1: Dense baseline ──────────────────────────────────────────
    _log("\n═══ Phase 1: Dense baseline ═══")
    _log(f"Loading model (dense)...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model_dense.eval()
    device = next(model_dense.parameters()).device

    for seq_len in args.seq_lens:
        mem = _gpu_mem_gb()
        _log(f"  Dense T={seq_len:,} (free={mem.get('free_gb', '?')}GB)...")
        inputs = _make_dummy_input(tokenizer, seq_len, device)
        try:
            result = bench_prefill(
                model_dense, inputs,
                warmup=args.warmup, repeats=args.repeats, label="dense",
            )
            result["type"] = "bench"
            results.append(result)
            _log(f"    median={result['median_ms']:.1f}ms  "
                 f"mean={result['mean_ms']:.1f}ms  "
                 f"min={result['min_ms']:.1f}ms  "
                 f"mem={result['gpu_mem'].get('allocated_gb', '?')}GB")
        except torch.cuda.OutOfMemoryError as e:
            _log(f"    OOM at T={seq_len:,}: {e}")
            results.append({
                "type": "bench", "label": "dense", "seq_len": seq_len,
                "error": f"OOM: {e}", "gpu_mem": _gpu_mem_gb(),
            })
            torch.cuda.empty_cache()
        except Exception as e:
            _log(f"    FAILED: {e}")
            results.append({
                "type": "bench", "label": "dense", "seq_len": seq_len,
                "error": str(e),
            })

    del model_dense
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: Wayfinder ───────────────────────────────────────────────
    _log("\n═══ Phase 2: Wayfinder (permute, random) ═══")
    _log(f"Loading model (wayfinder)...")
    model_wf = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model_wf.eval()
    device = next(model_wf.parameters()).device

    cfg = NemotronHWayfinderConfig(
        path="permute",
        strategy="random",
        window=args.window,
        landmark_stride=args.landmark_stride if args.landmark_stride > 0 else None,
        num_cycles=args.num_cycles,
        engine=args.engine,
    )
    replaced = swap_nemotron_h_attention_with_wayfinder(model_wf, cfg)
    _log(f"Replaced layers: {replaced}")

    for seq_len in args.seq_lens:
        mem = _gpu_mem_gb()
        _log(f"  Wayfinder T={seq_len:,} (free={mem.get('free_gb', '?')}GB)...")
        inputs = _make_dummy_input(tokenizer, seq_len, device)
        try:
            result = bench_prefill(
                model_wf, inputs,
                warmup=args.warmup, repeats=args.repeats, label="wayfinder",
            )
            result["type"] = "bench"
            result["window"] = args.window
            result["landmark_stride"] = args.landmark_stride
            result["num_cycles"] = args.num_cycles

            profiles = collect_wayfinder_profiles(model_wf)
            result["wayfinder_profiles"] = profiles

            results.append(result)
            _log(f"    median={result['median_ms']:.1f}ms  "
                 f"mean={result['mean_ms']:.1f}ms  "
                 f"min={result['min_ms']:.1f}ms  "
                 f"mem={result['gpu_mem'].get('allocated_gb', '?')}GB")

            wf_active = sum(1 for p in profiles if p["mode"] == "wayfinder")
            _log(f"    wayfinder_active_layers={wf_active}/{len(profiles)}")
        except torch.cuda.OutOfMemoryError as e:
            _log(f"    OOM at T={seq_len:,}: {e}")
            results.append({
                "type": "bench", "label": "wayfinder", "seq_len": seq_len,
                "error": f"OOM: {e}", "gpu_mem": _gpu_mem_gb(),
            })
            torch.cuda.empty_cache()
        except Exception as e:
            _log(f"    FAILED: {e}")
            results.append({
                "type": "bench", "label": "wayfinder", "seq_len": seq_len,
                "error": str(e),
            })

    del model_wf
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    _log(f"\n═══ Results saved to {out_path} ═══")

    # ── Summary table ────────────────────────────────────────────────────
    _log("\n╔══════════════════════════════════════════════════════╗")
    _log("║       Dense vs Wayfinder Prefill (Nemotron)         ║")
    _log("╠══════════╦═══════════════╦═══════════════╦══════════╣")
    _log("║  SeqLen  ║  Dense (ms)   ║ Wayfinder(ms) ║ Speedup  ║")
    _log("╠══════════╬═══════════════╬═══════════════╬══════════╣")

    dense_by_seq = {}
    wf_by_seq = {}
    for r in results:
        if r.get("type") != "bench" or "error" in r:
            continue
        if r["label"] == "dense":
            dense_by_seq[r["seq_len"]] = r["median_ms"]
        elif r["label"] == "wayfinder":
            wf_by_seq[r["seq_len"]] = r["median_ms"]

    for seq_len in args.seq_lens:
        d = dense_by_seq.get(seq_len)
        w = wf_by_seq.get(seq_len)
        d_str = f"{d:>11.1f}" if d else "       N/A"
        w_str = f"{w:>11.1f}" if w else "       N/A"
        if d and w and w > 0:
            speedup = d / w
            s_str = f"{speedup:>6.2f}x"
        else:
            s_str = "   N/A"
        _log(f"║  {seq_len:>6}  ║ {d_str}   ║ {w_str}   ║ {s_str} ║")

    _log("╚══════════╩═══════════════╩═══════════════╩══════════╝")


if __name__ == "__main__":
    main()
