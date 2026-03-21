#!/usr/bin/env python3
"""Profile a single QwenWayfinderAttention layer — graph build + forward.

Loads model, swaps ONE layer, times graph build vs attention separately.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx
from mlx_lm import load

from bna.integrations.qwen_mlx import QwenWayfinderConfig, QwenWayfinderAttention


def main():
    model, _tok, _cfg = load(
        "mlx-community/Qwen3-4B-4bit",
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    cfg = QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=64,
        landmark_stride=64,
        num_cycles=1,
        seed=42,
        edge_bias=True,
    )

    base_attn = model.layers[0].self_attn
    wf_attn = QwenWayfinderAttention(base_attn, cfg)

    hidden_size = int(model.args.hidden_size)
    seq_lens = [512, 1024, 2048, 4096]

    print(f"{'T':>6s}  {'1st call (s)':>12s}  {'graph (ms)':>10s}  {'attn (ms)':>10s}  "
          f"{'2nd call (s)':>12s}  {'graph (ms)':>10s}  {'attn (ms)':>10s}  {'tok/s':>8s}")
    print("-" * 100)

    for T in seq_lens:
        x = mx.random.normal((1, T, hidden_size), dtype=mx.bfloat16)
        mx.eval(x)

        # Clear cache to force rebuild
        from bna.integrations.qwen_mlx import _QWEN_GRAPH_CACHE_STORE
        _QWEN_GRAPH_CACHE_STORE.clear()

        # 1st call — includes graph build
        t0 = time.perf_counter()
        y1 = wf_attn(x, mask=None, cache=None)
        mx.eval(y1)
        t1 = time.perf_counter()
        first_s = t1 - t0
        p1 = wf_attn.last_profile

        # 2nd call — cache hit
        t2 = time.perf_counter()
        y2 = wf_attn(x, mask=None, cache=None)
        mx.eval(y2)
        t3 = time.perf_counter()
        second_s = t3 - t2
        p2 = wf_attn.last_profile

        tok_s = T / second_s

        print(f"{T:>6d}  {first_s:12.4f}  {p1.graph_build_ms:10.1f}  {p1.attention_ms:10.1f}  "
              f"{second_s:12.4f}  {p2.graph_build_ms:10.1f}  {p2.attention_ms:10.1f}  {tok_s:8.0f}")


if __name__ == "__main__":
    main()
