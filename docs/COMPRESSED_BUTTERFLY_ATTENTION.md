# Compressed Butterfly Attention

This note turns the DeepSeek V4 attention lesson into a Butterfly-native test target.
It does not copy the DeepSeek architecture. It uses the paper as a systems checklist:
compressed KV layout, exact local detail, long-range sparse routing, and evaluation at
the memory wall.

Primary external references:

- DeepSeek model card and technical report: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>
- vLLM DeepSeek V4 implementation note: <https://vllm.ai/blog/deepseek-v4>

## Target Variant

**Compressed Butterfly Attention** is:

1. Exact sliding-window attention over recent raw tokens.
2. One mean-pooled KV summary per older block.
3. Deterministic `causal_shift` routing over those compressed block summaries.
4. Optional future ablation: content or learned top-k summaries in addition to deterministic partners.

The current implementation tests items 1-3. It deliberately does not claim a learned
DeepSeek-style indexer yet.

## Why This Is The Right Butterfly Object

The previous block-sparse Butterfly path selected blocks, then attended over every raw
token inside those blocks. That was useful for topology validation, but it was not a
serious KV-cache design: long contexts still paid raw-token gather costs for every
selected block.

Compressed Butterfly changes the cache object:

- local raw-token cache: recent exact detail
- compressed block-summary cache: older routed context
- deterministic topology: `causal_shift` staged partners over block summaries

This makes Butterfly a cache layout plus a routing topology, not only a sparse mask.

## Current Semantics

For a query token `t` in block `b`:

- local candidates are raw tokens in `[t - W + 1, t]`
- summary candidates are routed neighbor blocks from the Butterfly row for `b`
- summary candidates must be strictly older than the local window
- each summary key/value is the mean of its block key/value states
- current-block summaries are never used, so the path remains causal

The initial summary pool is mean pooling because it is deterministic and testable.
A trained compressor can replace it later without changing the routing/cache contract.

## Public Test Surface

Use `compressed_butterfly` as the public variant name. `block_sparse` remains a legacy
compatibility alias for the older raw-block path.

MLX:

```bash
./.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
  --model-path /Volumes/VIXinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462 \
  --hf-home /Volumes/VIXinSSD/hf_cache \
  --hf-hub-cache /Volumes/VIXinSSD/hf_cache/hub \
  --hf-offline \
  --mode compressed_butterfly \
  --block-partner-rule causal_shift \
  --compressed-local-window-tokens 128 \
  --seq-lens 4096 8192 \
  --decode-len 8 \
  --repeats 1 \
  --chunk-size 384 \
  --kv-step 384 \
  --query-chunk-size 384 \
  --block-size 128 \
  --butterfly-decode-backend stock \
  --skip-multi-turn \
  --skip-quality \
  --out-dir results/benchmarks/qwen35_0p8b_mlx/compressed_butterfly_smoke
```

CUDA:

```bash
python scripts/bench_qwen35_cuda_wayfinder.py \
  --model-path <QWEN3.5_MODEL_PATH_OR_ID> \
  --path compressed_butterfly \
  --block-partner-rule causal_shift \
  --compressed-local-window-tokens 128 \
  --engine sdpa \
  --forward-target backbone \
  --seq-lens 4096 8192 \
  --repeats 1 \
  --phases butterfly dense \
  --output benchmarks/cuda/qwen35_butterfly/compressed_butterfly_smoke.ndjson
```

The CUDA script name still contains older terminology. The runtime object and public
variant should be reported as Butterfly.

## Evaluation Matrix

Do not claim a performance win from short smoke tests. Compare against stock attention
and record:

- peak KV/cache memory or process peak memory
- prefill time
- TTFT
- decode token/s
- quality or perplexity drift
- long-context retrieval
- largest shared non-OOM sequence length

Recommended first matrix:

| Variant | Local raw window | Long-range route | Summary cache | Router |
| --- | ---: | --- | --- | --- |
| stock | full | full | raw KV | none |
| Butterfly | windowed/permuted | deterministic | raw KV | `causal_shift` or cycle |
| Compressed Butterfly | 128 tokens | deterministic | mean block KV | `causal_shift` |
| Future top-k ablation | 128 tokens | deterministic + top-k | learned/compressed KV | trained or content top-k |

## Claim Boundary

Current MLX status (Qwen 3.5 0.8B 4-bit, decode_len=8, frozen pretrained
checkpoint, deterministic `causal_shift` routing, mean-pool compressor,
SWA window n_win=64, block_size=128, query_chunk_size=64):

| context | e2e (s) | peak (GB) | retained KV (MB) | vs stock e2e | vs stock peak | vs stock KV |
|---|---:|---:|---:|---:|---:|---:|
| 32k stock      | 9.52   | 1.78 | 414  | — | — | — |
| 32k compressed | 10.09  | 0.91 |  16  | 1.06×    | **0.51×** | **0.040×** |
| 64k stock      | 25.10  | 2.64 | 817  | — | — | — |
| 64k compressed | 25.54  | 1.14 |  21  | 1.02×    | **0.43×** | **0.026×** |
| 128k stock     | 80.95  | 4.26 | 1622 | — | — | — |
| 128k compressed| 72.23  | 1.44 |  26  | **0.89×** | **0.34×** | **0.016×** |
| 256k stock     | 277.85 | 7.84 | 3233 | — | — | — |
| 256k compressed| 220.74 | 2.10 |  40  | **0.79×** | **0.27×** | **0.012×** |

Artifacts: `results/benchmarks/qwen35_0p8b_mlx/pA3_*` and `OVERNIGHT_LOG.md`.

What this proves on MLX:

- The DeepSeek V4 systems architecture (two-pool KV layout, multi-stream
  online-softmax attention without union K/V materialization, counter-RoPE
  surface preserved) ports to Apple silicon without a custom Metal kernel.
- Butterfly `causal_shift` deterministic routing as the sparse selector
  (replacing V4's learned top-k indexer) produces measurable wins on a
  **frozen** pretrained checkpoint. No training required.
- The advantage **grows** with context length: at 256k compressed Butterfly
  is 21% faster than stock, uses 27% of stock's peak memory, and retains 81×
  less KV cache.

Still not claimed:

- Quality preservation under sparse routing — only a smoke run exists
  (`scripts/quality_smoke.py`, `OVERNIGHT_LOG.md` Phase 5). At 1k / 4k / 16k
  context, compressed produces coherent text that diverges from stock's
  exact tokens; top-50 candidate overlap is ~40-50%, top-1 differs, greedy
  decode of 32 tokens matches 0/32. This rules out catastrophic breakage
  but is not a quality measurement; perplexity / downstream-task / retrieval
  evaluations remain future work.
- Learned-compressor competitive parity with V4-Pro (mean-pool stays as a
  no-train placeholder; a learned compressor would need training).
- Generalization to other model families beyond Qwen 3.5 (only this
  checkpoint validated).
- CUDA path improvements (this run cut over MLX only; the CUDA mirror tests
  still pass against the prior algorithmic surface).
- Performance beyond 256k (not yet measured).

The legacy items below remain not claimed for the same reasons:

- learned indexer parity with DeepSeek CSA (we deliberately use deterministic
  routing instead, as the unique design point that enables retrofit).
