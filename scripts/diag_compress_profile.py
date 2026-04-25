#!/usr/bin/env python3
"""Quick diagnostic: run compressed butterfly 8192 with profiling and print stats."""
import json
import os
import subprocess
import sys

os.environ["BNA_COMPRESS_PROFILE"] = "1"

model = "/Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462"
out_dir = "results/benchmarks/qwen35_0p8b_mlx/compressed_butterfly_8192_profiled"

cmd = [
    sys.executable,
    "scripts/bench_qwen_consumer_mlx.py",
    "--model-path", model,
    "--hf-home", "/Volumes/VIXinSSD/hf_cache",
    "--hf-hub-cache", "/Volumes/VIXinSSD/hf_cache/hub",
    "--hf-offline",
    "--mode", "compressed_butterfly",
    "--block-partner-rule", "causal_shift",
    "--compressed-local-window-tokens", "128",
    "--seq-lens", "8192",
    "--decode-len", "8",
    "--repeats", "1",
    "--chunk-size", "384",
    "--kv-step", "384",
    "--query-chunk-size", "384",
    "--block-size", "128",
    "--butterfly-decode-backend", "stock",
    "--skip-multi-turn",
    "--skip-quality",
    "--out-dir", out_dir,
]

print("Running profiled 8192 chunked prefill...")
subprocess.run(cmd, cwd="/Volumes/VIXinSSD/butterfly", env={**os.environ, "BNA_COMPRESS_PROFILE": "1"})

# Read and display results
with open(f"{out_dir}/results.json") as f:
    data = json.load(f)

row = data["single_turn"][0]
prof = row.get("compress_profile", {})

print("\n=== Compressed Butterfly Profile ===")
print(f"_block_mean_summaries calls: {prof.get('calls', 0)}")
print(f"summary construction ms: {prof.get('summary_ms', 0):.2f}")
print(f"attention compute ms:    {prof.get('attn_ms', 0):.2f}")
total = prof.get('summary_ms', 0) + prof.get('attn_ms', 0)
print(f"ratio summary/total:     {prof.get('summary_ms', 0)/(total+1e-9):.2%}")
print(f"\nUnique shapes/dtypes seen:")
seen = set()
for s in prof.get('shapes', []):
    key = (s['seq_len'], s['block_size'], s['num_blocks'], tuple(s['x_shape']), s['x_dtype'])
    if key not in seen:
        seen.add(key)
        print(f"  seq_len={s['seq_len']} block_size={s['block_size']} num_blocks={s['num_blocks']} x_shape={s['x_shape']} dtype={s['x_dtype']}")

print(f"\nPrefill time: {row['prefill_sec']:.4f}s")
print(f"Peak memory:  {row['peak_memory_bytes']/1e9:.2f}GB")
