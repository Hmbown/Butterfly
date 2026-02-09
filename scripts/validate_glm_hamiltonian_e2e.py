#!/usr/bin/env python3
"""Quick E2E validation: GLM-4.7-Flash + Actually Hamiltonian features.

Tests that circular windowing, K4 active-row path, and union multigraph
all work with real model weights — not toy configs. Runs in ~30 seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from hcsa.integrations.glm_mlx import (
    GLMWayfinderAttention,
    GLMWayfinderConfig,
    swap_glm_attention_with_wayfinder,
)


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _peak_mb() -> float:
    try:
        return mx.metal.get_peak_memory() / 1e6
    except Exception:
        return mx.get_peak_memory() / 1e6


def _check_output(logits: mx.array, label: str, seq_len: int) -> bool:
    mx.eval(logits)
    ok = True
    if logits.ndim != 3:
        print(f"  FAIL [{label}] expected 3D logits, got {logits.ndim}D")
        ok = False
    if logits.shape[1] != seq_len:
        print(f"  FAIL [{label}] seq dim={logits.shape[1]}, expected {seq_len}")
        ok = False
    if mx.any(mx.isnan(logits)).item():
        print(f"  FAIL [{label}] NaN in logits!")
        ok = False
    if mx.any(mx.isinf(logits)).item():
        print(f"  FAIL [{label}] Inf in logits!")
        ok = False
    # Check logits are reasonable (not all zeros, not absurdly large)
    max_abs = mx.max(mx.abs(logits)).item()
    if max_abs < 1e-6:
        print(f"  FAIL [{label}] logits near-zero (max_abs={max_abs})")
        ok = False
    if max_abs > 1e6:
        print(f"  WARN [{label}] very large logits (max_abs={max_abs})")
    return ok


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Cannot find model layers")


def validate_config(model, cfg: GLMWayfinderConfig, label: str,
                    seq_len: int = 2048, decode_steps: int = 8,
                    chunk_size: int = 512) -> bool:
    """Swap attention, run chunked prefill + decode, check outputs."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  circular={cfg.circular}, multi_cycle_mode={cfg.multi_cycle_mode}")
    print(f"  seq_len={seq_len}, decode_steps={decode_steps}")
    print(f"{'='*60}")

    # Swap
    t0 = time.perf_counter()
    replaced = swap_glm_attention_with_wayfinder(model, cfg=cfg)
    swap_ms = (time.perf_counter() - t0) * 1000
    _log(f"Swapped {len(replaced)} layers in {swap_ms:.0f}ms")

    # Check flags on a sample layer
    layers = _get_layers(model)
    sample = None
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, GLMWayfinderAttention):
            sample = attn
            break
    if sample is None:
        print("  FAIL: no GLMWayfinderAttention found after swap")
        return False
    _log(f"Sample layer: circular={sample.circular}, "
         f"multi_cycle_mode={sample.multi_cycle_mode}")
    if sample.circular != cfg.circular:
        print(f"  FAIL: circular mismatch: {sample.circular} vs {cfg.circular}")
        return False
    if sample.multi_cycle_mode != cfg.multi_cycle_mode:
        print(f"  FAIL: multi_cycle_mode mismatch")
        return False

    # Generate prompt tokens
    rng = np.random.default_rng(42)
    prompt = mx.array(
        rng.integers(100, 50000, size=(1, seq_len)).astype(np.int32)
    )

    # Chunked prefill
    cache = list(make_prompt_cache(model))
    all_ok = True

    _log(f"Running chunked prefill ({seq_len} tokens, chunk={chunk_size})...")
    t0 = time.perf_counter()
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = prompt[:, start:end]
        logits = model(chunk, cache=cache)
        mx.eval(logits)
    prefill_sec = time.perf_counter() - t0
    prefill_tok_s = seq_len / prefill_sec
    _log(f"Prefill: {prefill_sec:.2f}s ({prefill_tok_s:.0f} tok/s)")

    # Check last chunk logits
    if not _check_output(logits, f"{label}/prefill", end - start):
        all_ok = False

    # Profile from sample layer
    if sample is not None:
        prof = sample.last_profile
        _log(f"Attention path: {prof.path}")
        notes = prof.notes if isinstance(prof.notes, dict) else {}
        _log(f"  graph_build_ms={prof.graph_build_ms:.1f}, "
             f"attn_ms={prof.attention_ms:.1f}")
        if "active_query_mode" in notes:
            _log(f"  active_query_mode={notes['active_query_mode']}")
        if "active_dense_triggered" in notes:
            _log(f"  active_dense_triggered={notes['active_dense_triggered']}")

    # Decode
    if decode_steps > 0:
        _log(f"Running decode ({decode_steps} steps)...")
        t0 = time.perf_counter()
        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        for step in range(decode_steps):
            logits = model(next_tok, cache=cache)
            mx.eval(logits)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        decode_sec = time.perf_counter() - t0
        decode_tok_s = decode_steps / decode_sec
        _log(f"Decode: {decode_sec:.2f}s ({decode_tok_s:.1f} tok/s)")

        if not _check_output(logits, f"{label}/decode", 1):
            all_ok = False

    _log(f"Peak memory: {_peak_mb():.0f} MB")

    if all_ok:
        print(f"  PASS: {label}")
    else:
        print(f"  FAIL: {label}")
    return all_ok


def main() -> None:
    print("Loading GLM-4.7-Flash-4bit...")
    model, tokenizer, config = load(
        "mlx-community/GLM-4.7-Flash-4bit",
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    print("Model loaded.\n")

    # Use shorter seq for validation speed
    SEQ = 4096
    CHUNK = 1024
    DECODE = 4

    results = {}

    # Test 1: Baseline (circular=False, average, K4 active path)
    cfg_baseline = GLMWayfinderConfig(
        window=64, circular=False, multi_cycle_mode="average",
        permute_head_chunk_size=2, query_chunk_size=384,
        active_dense_threshold=49152,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
    )
    results["baseline"] = validate_config(
        model, cfg_baseline, "Baseline (linear, average, d=1)",
        seq_len=SEQ, decode_steps=DECODE, chunk_size=CHUNK,
    )

    # Test 2: Circular windowing
    cfg_circular = GLMWayfinderConfig(
        window=64, circular=True, multi_cycle_mode="average",
        permute_head_chunk_size=2, query_chunk_size=384,
        active_dense_threshold=49152,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
    )
    results["circular"] = validate_config(
        model, cfg_circular, "Circular windowing",
        seq_len=SEQ, decode_steps=DECODE, chunk_size=CHUNK,
    )

    # Test 3: Circular + union multigraph (d=2)
    cfg_union = GLMWayfinderConfig(
        window=64, circular=True, multi_cycle_mode="union",
        num_cycles=2,
        permute_head_chunk_size=2, query_chunk_size=384,
        active_dense_threshold=49152,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
    )
    results["circular_union"] = validate_config(
        model, cfg_union, "Circular + Union (d=2)",
        seq_len=SEQ, decode_steps=DECODE, chunk_size=CHUNK,
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll GLM Hamiltonian E2E validations PASSED.")
    else:
        print("\nSome validations FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
