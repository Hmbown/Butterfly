# Butterfly Compressed Attention

A retrofit-ready, no-train, deterministic-routing variant of DeepSeek-V4-style
compressed attention for long-context inference on **frozen** pretrained
checkpoints. Validated on Apple Silicon (MLX) for the Qwen 3.5 hybrid family.

> **What stands today.** On Qwen 3.5 0.8B 4-bit MLX at 256 k context, Butterfly
> is **21 % faster end-to-end**, uses **27 % of stock peak memory**, and
> retains an **81 × smaller KV cache** than stock attention — on a frozen
> pretrained checkpoint, with no training, no learned indexer, and no custom
> Metal kernel. The win **grows monotonically with context**. The same
> architecture replicates on Qwen 3.5 4B 4-bit MLX (32 layers, 8 dense
> full-attention) with the same trend; on the 4B checkpoint at 1 k context,
> compressed-Butterfly's greedy decode is **bit-for-bit identical** to
> stock's for 32 tokens.

The full positioning, V4 vs Butterfly framing, replication recipe, claim
boundary, and code map live in
**[docs/BUTTERFLY_POSITIONING.md](docs/BUTTERFLY_POSITIONING.md)**.

## Headline scoreboard

![Ladder: e2e × peak × KV vs context for Qwen 3.5 0.8B and 4B 4-bit MLX](docs/assets/butterfly_ladder.png)

Lower is better in all panels. The crossover where compressed Butterfly
beats stock attention on **every** metric — speed, peak memory, retained KV —
sits between 64 k and 128 k of context. By 256 k on the 0.8B checkpoint
the gap is decisive (0.79× e2e, 0.27× peak, 81× smaller KV). The 256 k row
on 4B is harness-blocked by the bench's O(T²) synthetic-prompt builder, not
by anything attention is doing.

## What changes when Butterfly replaces V4's selector

![V4 Lightning Indexer vs Butterfly causal_shift adjacency](docs/assets/butterfly_vs_v4.png)

DeepSeek V4 picks compressed-block partners with a *learned* Lightning
Indexer — three trainable projections plus a ReLU-scored top-k selector
(paper §2.3.1). Butterfly substitutes a *deterministic* `causal_shift`
adjacency — one block partner per query block per layer, set by
`partner = block − 2^stage` — with **zero learned parameters**. That single
substitution is what makes the architecture drop in on a frozen pretrained
checkpoint. Everything else (two-pool fixed-buffer cache, multi-stream
SWA + compressed-block attention, online-softmax merge) is V4-shape,
deliberately.

## Topology

![Butterfly causal_shift adjacency across 5 stages of 32 blocks](docs/assets/butterfly_topology.png)

Each layer ℓ uses one stage; `stage = ℓ mod ⌈log₂ N⌉`. Across the stages,
the union covers every causal predecessor at depth ⌈log₂ N⌉. This is the
classical butterfly-graph mixing property, adapted for a causal language
model.

## Quality smoke

![Quality smoke: top-50 overlap and 32-token greedy parity, 0.8B vs 4B](docs/assets/butterfly_quality_card.png)

The 4B-1k bar is the single most important sanity gate in this work: at
1 k context on a real-size checkpoint, compressed Butterfly produces
**identically the same 32-token greedy continuation** as stock attention,
and the top-1 candidate at end-of-prefill agrees. At 4 k and 16 k the
candidate distributions overlap meaningfully (top-50 32 / 50, 23 / 50)
and KL is bounded, but greedy decode amplifies any single-token
disagreement into a fully different sequence; that is expected for
sparse routing without indexer retraining, not breakage. A real
quality benchmark (perplexity, retrieval) is the next evidence step
([§H Future work](docs/BUTTERFLY_POSITIONING.md#h-future-work)).

## Quick replication

Apple Silicon, MLX. Single Mac. Models are uniform 4-bit MLX checkpoints
from `mlx-community`.

```bash
VENV=/Volumes/VIXinSSD/butterfly/.venv-macos-metal/bin/python   # adjust
HF_CACHE=/Volumes/VIXinSSD/hf_cache                             # adjust

QWEN08B=$HF_CACHE/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/<HASH>
QWEN4B=$HF_CACHE/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<HASH>

# Stock vs Butterfly compressed at 64 k on the 0.8B checkpoint
$VENV scripts/bench_qwen_consumer_mlx.py --model-path "$QWEN08B" \
  --hf-home "$HF_CACHE" --hf-hub-cache "$HF_CACHE/hub" --hf-offline \
  --mode stock --seq-lens 65536 --decode-len 8 --repeats 1 \
  --chunk-size 384 --kv-step 384 \
  --skip-multi-turn --skip-quality --stage-timeout-sec 1200 \
  --out-dir results/qwen08b/stock_64k

$VENV scripts/bench_qwen_consumer_mlx.py --model-path "$QWEN08B" \
  --hf-home "$HF_CACHE" --hf-hub-cache "$HF_CACHE/hub" --hf-offline \
  --mode compressed_butterfly --block-partner-rule causal_shift \
  --compressed-local-window-tokens 64 --query-chunk-size 64 --block-size 128 \
  --butterfly-decode-backend stock \
  --seq-lens 65536 --decode-len 8 --repeats 1 \
  --chunk-size 384 --kv-step 384 \
  --skip-multi-turn --skip-quality --stage-timeout-sec 1200 \
  --out-dir results/qwen08b/comp_64k
```

Read the resulting `results.json`:

```python
import json
d = json.load(open("results/qwen08b/comp_64k/results.json"))["single_turn"][0]
print(d["e2e_sec"],
      d["peak_memory_bytes"] / (1024**3),                 # GB
      d["cache_storage_after_prefill"]["total_bytes"]/1e6) # MB retained KV
```

The full ladder script and quality-smoke commands are in
[BUTTERFLY_POSITIONING §G](docs/BUTTERFLY_POSITIONING.md#g-replicating-these-numbers).

## Repo map

| Path | What it is |
|---|---|
| `bna/topology/butterfly.py` | `causal_shift` partner rule and stage metadata. |
| `bna/mlx/attention.py` | Multi-stream SWA + compressed-block attention, online-softmax merge. |
| `bna/mlx/compressed_cache.py` | Two-pool fixed-buffer KV cache (V4 Fig 6 layout). |
| `bna/integrations/qwen_mlx.py` | Layer swap + dispatch + graph-cache eviction fix. |
| `scripts/bench_qwen_consumer_mlx.py` | Bench harness used for every number in the doc. |
| `scripts/quality_smoke.py` | Top-50 / KL / greedy parity. |
| `scripts/render_butterfly_assets.py` | Re-renders the four PNGs from raw JSON. |
| `docs/BUTTERFLY_POSITIONING.md` | Public-facing positioning doc — start here. |
| `docs/COMPRESSED_BUTTERFLY_ATTENTION.md` | Variant claim-boundary + same numbers. |
| `docs/ARCHITECTURE.md` | Contributor-facing implementation map. |
| `docs/APPLE_SILICON_SETUP.md` | Apple Silicon bootstrap, MLX venv, model catalog. |
| `docs/BUTTERFLY_THEOREMS.md` | Topology proof status (paper-side claims). |
| `results/benchmarks/qwen35_0p8b_mlx/` | Raw JSON + `OVERNIGHT_LOG.md` chronology. |
| `results/benchmarks/qwen35_4b_mlx/` | 4B replication artifacts. |
| `results/quality/qwen35_4b/` | 4B quality-smoke JSON. |

## Next-up tests

The natural extensions, in priority order — see
[§H Future work](docs/BUTTERFLY_POSITIONING.md#h-future-work) for
context:

1. **RULER / NIAH at 32 k–128 k** to turn "smoke" into "evidence".
2. **Qwen 3.6 27B 4-bit on a 36 GB-RAM Mac** — the 27 B regime is exactly
   where the 63 ×–81 × KV reduction starts to matter for whether the model
   fits at all.
3. **Streamed prompt-build** in the bench harness to unblock 4B-256 k and
   either model at 512 k+.
4. **Learned compressor** to replace mean-pool — closes the §E quality gap.
5. **Counter-RoPE** on the compressed-stream output (V4 §2.3.3).
6. **Decode-mode kernel** to extend the wins past prefill.

## Related work

- DeepSeek V4 — [hf.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [BigBird](https://arxiv.org/abs/2007.14062)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [Monarch](https://arxiv.org/abs/2204.00595)
- [FlexPrefill](https://arxiv.org/abs/2502.20766)
- [NSA](https://arxiv.org/abs/2502.11089)
- [MoBA](https://arxiv.org/abs/2502.13189)

## License

MIT
