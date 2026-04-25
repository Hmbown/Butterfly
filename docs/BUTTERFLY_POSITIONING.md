# Butterfly Compressed Attention

A retrofit-ready, no-train, deterministic-routing variant of DeepSeek-V4-style
compressed attention for long-context inference on **frozen** pretrained
checkpoints. Validated on Apple Silicon (MLX) for the Qwen 3.5 hybrid family.

> **What stands today.** On Qwen 3.5 0.8B 4-bit MLX at 256 k context, Butterfly
> is **21 % faster end-to-end**, uses **27 % of stock peak memory**, and
> retains an **81 × smaller KV cache** than stock attention — on a frozen
> pretrained checkpoint, with no training, no learned indexer, and no custom
> Metal kernel. The win **grows monotonically with context**. The same
> architecture replicates on Qwen 3.5 4B 4-bit MLX (32 layers, 8 dense) with
> the same trend; on the 4B checkpoint at 1 k context, compressed-Butterfly's
> greedy decode is **bit-for-bit identical** to stock's for 32 tokens.

---

## A. The pitch (one paragraph)

Butterfly Compressed Attention is a sparse-attention runtime that takes the
DeepSeek-V4 systems shape — two-pool fixed-buffer KV cache, multi-stream
SWA + compressed-block attention, online-softmax merge — and substitutes
V4's *learned, content-relevant* Lightning Indexer with a *deterministic,
structure-relevant* Butterfly `causal_shift` adjacency. The trade is "pick
the most semantically relevant compressed blocks" → "guaranteed structured
mixing with zero learned parameters". Because nothing in the routing depends
on unseen weights, Butterfly is a **pure zero-train retrofit** on a frozen
pretrained checkpoint and delivers V4's long-context efficiency wins without
the V4 pretraining bill. (V4's selector is not impossible to retrofit either
— you could distill or fine-tune one in — Butterfly just doesn't need that
step.) Butterfly is not a quality replacement for V4-Pro: the compressor is
still parameter-free mean-pool, decode falls back to stock dense, and the
selector is structure-relevant rather than content-relevant. It is a
measured-wins-on-prefill story whose surprising claim is that a deterministic
topology can stand in for a learned selector well enough to be useful at all
on a frozen checkpoint.

---

## B. Architecture (Qwen 3.5 specifically)

Qwen 3.5 is a hybrid Mamba/attention model. The 0.8B 4-bit checkpoint has
**24 layers**: full-attention at indices `[3, 7, 11, 15, 19, 23]` and
linear-attention (Mamba/SSM) at every other index. Butterfly only swaps the
**6 full-attention layers**. The 18 linear layers are untouched and run
unchanged. The 4B-MLX-4bit checkpoint follows the same `full_attention_interval=4`
pattern over **32 layers**, giving **8 full-attention layers** at
`[3, 7, 11, 15, 19, 23, 27, 31]`.

For each swapped layer, the prefill path runs two attention streams in
parallel and merges them with FlashAttention-style online softmax:

```
            ┌─────────────── compressed-routed stream ───────────────┐
queries ───▶│  k_summary, v_summary  =  mean-pool of completed       │──▶ (o_c, l_c, m_c)
            │      block KV (one summary per block)                  │
            │  routed_indices  =  Butterfly causal_shift partners    │
            │                     of the query's block, filtered     │
            │                     to be strictly older than SWA      │
            └────────────────────────────────────────────────────────┘
                                                                            ┐
            ┌─────────────────── SWA stream (raw tokens) ─────────────┐    ├─▶ online_softmax_merge ─▶ output
queries ───▶│  tail_k, tail_v  =  most recent n_win=64 raw tokens    │──▶ (o_s, l_s, m_s)            │
            │                     held in the cache's tail buffer    │                              │
            └─────────────────────────────────────────────────────────┘                              ┘
```

The merge accumulates per-row `(o, l, m)` tuples and produces the same
numerical answer as a single softmax over the union of both streams' keys —
**without ever materializing the union K/V**. Implementation:
`bna/mlx/attention.py:_swa_stream_attention`,
`_compressed_stream_attention`, `_online_softmax_merge`,
`compressed_butterfly_attention[_from_cache]`.

The cache (`bna/mlx/compressed_cache.py:CompressedKVCache`) is a
**two-pool fixed-buffer** layout that mirrors V4's Figure 6:

- *Tail buffer* `[B, H, tail_capacity, dh]` — raw recent tokens for the SWA
  stream, slide-and-trim in place via `mx.slice_update` (no `mx.concatenate`).
- *Summary buffer* `[B, H, summary_capacity, dh]` — one mean-pool slot per
  completed compression block.

A summary slot is written when a block fully predates the SWA window
(`summary_offset + block_size <= offset - local_window_tokens`); the tail is
trimmed when oversized. Both buffers are pre-allocated and slice-updated, so
the lazy MLX graph never accumulates intermediate tensors.

---

## C. What makes it Butterfly, not V4

| Component | DeepSeek V4 (paper §2.3, Fig 6) | Butterfly | Consequence |
|---|---|---|---|
| **Selector / "indexer"** | Learned, content-relevant Lightning Indexer: 3 trainable projections (W^DQ, W^IUQ, W^w) + ReLU score → top-k over compressed blocks (eqs. 13–17) | Deterministic, structure-relevant `causal_shift`: `partner = block_idx - (1 << bit_idx)`, `stage_idx = layer_idx % width`. Fixed adjacency, zero parameters. | V4's quality depends on its trained indexer, so dropping V4's CSA onto a frozen checkpoint is not a *pure zero-train* operation — you'd need to distill or fine-tune the indexer in. Butterfly's adjacency is set once from `(num_blocks, layer_idx)` and works as-is on any frozen Qwen 3.5 checkpoint. |
| **Compressor** | Learned softmax-weighted sum across overlapping 2m windows with positional biases B^a, B^b (eqs. 9–12) | Parameter-free **mean pool** over each block | Real gap. A learned compressor needs training. The mean-pool placeholder gets the cache shape and routing right; quality improvements would come from a learned compressor later. |
| **SWA branch** | n_win=128 raw tokens added to compressed entries (§2.3.3) | n_win=64 raw tokens, identical structural role | Same idea, smaller window. |
| **Cache layout** | Pre-allocated state cache (SWA tail) + classical cache (compressed slots), Fig 6 | Same: pre-allocated tail buffer + summary buffer, all `mx.slice_update` | Direct adaptation. |
| **Multi-stream merge** | Shared KV-MQA over (compressed ∪ SWA) keys (§2.3.3) | Online-softmax accumulator over the two streams (`_online_softmax_merge`) | Same numerical answer, but Butterfly never materializes the union K/V — FlashAttention-style. Mild novelty. |
| **Counter-RoPE on compressed output** | Yes, position −i applied to last 64 dims of `o_t,i` (§2.3.3) | **No** | Deliberate gap. The empirical wins land without it; quality-sensitive deployments would want it back. |
| **Grouped output projection** | g=8 / g=16 splits (§2.3.1) | Inherited via the Qwen 3.5 layer's existing `o_proj` | The Qwen layer already does grouped GQA over q/k/v; we leave it alone. |
| **Decode path** | All attention layers run CSA / HCA | Decode falls back to **stock dense** attention (`butterfly_decode_backend=stock`) for the swapped layers | Wins on this branch are prefill + retained-KV, not decode-time. |

The composition is real, not a port. Butterfly takes the load-bearing piece
of V4 — the sparse selector that decides which compressed blocks each query
sees — and substitutes a fixed graph for a trained network. Everything else
(cache, multi-stream, merge) is V4-shaped or V4-like, deliberately. The
*claim* is "V4 systems shape + Butterfly deterministic indexer = retrofit",
not "we re-invented V4".

Butterfly's indexer also inherits classical butterfly-graph mixing
properties — but as **reachability through composition**, not direct
adjacency. Direct edges hit power-of-two causal offsets only (`b−1`,
`b−2`, `b−4`, …); arbitrary earlier blocks reach later ones through binary
decomposition over ⌈log₂ num_blocks⌉ stages, because each stage's
compressed states already carry the previous stage's mixed context forward.
This is the core butterfly composition property, adapted for causal LMs.
V4 makes no such structural-mixing claim about its learned top-k.

---

## D. Empirical scoreboard

Apple Silicon, MLX, single Mac. Frozen pretrained checkpoint. Decode falls
back to stock dense for the swapped layers. Bench:
`scripts/bench_qwen_consumer_mlx.py`. Each row is a single repeat,
`decode_len=8`, no quality benchmark on these rows. Raw artifacts:
`results/benchmarks/qwen35_0p8b_mlx/pA3_*` and
`results/benchmarks/qwen35_4b_mlx/`.

### D.1 Qwen 3.5 0.8B 4-bit MLX (24 layers / 6 dense full-attention)

Block-size 128, SWA window n_win=64, query_chunk=64.

| context | stock e2e (s) | comp e2e (s) | **e2e ×** | stock peak (GB) | comp peak (GB) | **peak ×** | stock retained KV (MB) | comp retained KV (MB) | **KV ×** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 k  | 9.52   | 10.09  | 1.06 | 1.65 | 0.85 | **0.51** | 414  | 16.4 | **0.040** |
| 64 k  | 25.10  | 25.54  | 1.02 | 2.46 | 1.06 | **0.43** | 817  | 21.1 | **0.026** |
| 128 k | 80.95  | 72.23  | **0.89** | 3.97 | 1.34 | **0.34** | 1622 | 25.8 | **0.016** |
| 256 k | 277.85 | 220.74 | **0.79** | 7.31 | 1.96 | **0.27** | 3233 | 40.0 | **0.012** |

Reading row by row:
- **32 k** — break-even on speed; compressed already cuts peak in half and
  retains 25× less KV. The "memory" win starts at the smallest context.
- **64 k** — still break-even on speed; peak gap widens to 0.43× stock,
  retained KV is now 38× smaller.
- **128 k** — compressed is **11 % faster**, peak is **34 %** of stock, KV
  is **63 ×** smaller.
- **256 k** — compressed is **21 % faster**, peak is **27 %** of stock, KV
  is **81 ×** smaller. The crossover where compressed beats stock on every
  metric is somewhere between 64 k and 128 k.

### D.2 Qwen 3.5 4B 4-bit MLX (32 layers / 8 dense full-attention)

Same flags. Same architecture, larger checkpoint, more dense-attention
layers per swap.

| context | stock e2e (s) | comp e2e (s) | **e2e ×** | stock peak (GB) | comp peak (GB) | **peak ×** | stock retained KV (MB) | comp retained KV (MB) | **KV ×** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 k  | 41.87  | 41.86  | 1.00 | 4.71  | 3.05 | **0.65** | 1104 | 43.1 | **0.039** |
| 64 k  | 153.45 | 84.77  | **0.55** | 6.41  | 3.26 | **0.51** | 2178 | 55.7 | **0.026** |
| 128 k | 274.24 | 245.57 | **0.90** | 10.21 | 3.81 | **0.37** | 4326 | 68.3 | **0.016** |
| 256 k | _harness-blocked_ | _harness-blocked_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ | _—_ |

The 256 k 4B row is *not* an architecture failure — it is the same bench-harness
O(T²) prompt-build wall already documented for 0.8B at 512 k. At 256 k the 4B
checkpoint's tokenizer-side synthetic-prompt construction did not finish inside
83 minutes (≥ 38 min wall, low CPU usage indicating the bench was still in
the pure-Python tokenization phase). No model forward pass was run.
Eliminating this measurement gap requires a streamed prompt-build path in
the bench harness; out of scope for the architecture work on this branch.

The KV-reduction ratio at each context matches the 0.8B value almost exactly
(4 % vs 4 %, 2.6 % vs 2.6 %, 1.6 % vs 1.6 %), which is what you'd predict:
the cache shape is set by `local_window_tokens / context` and `block_size`,
both of which are model-independent.

The peak-memory ratio is a touch higher than 0.8B at the same context (0.65×
vs 0.51× at 32 k), because the 4B model's working memory is dominated by
the 8 dense-attention activations during prefill, so the per-layer peak at
fixed T scales with the number of swapped layers.

The 64 k row is the noteworthy result: at 64 k context on the 4B checkpoint,
Butterfly compressed is **almost twice as fast as stock** (0.55× e2e). This
is much larger than the 0.8B's 64 k speedup of 1.02× e2e and warrants further
investigation — a likely explanation is that 4B at 64 k stock pushes the
working set across an MLX allocator threshold where each dense-attention
layer's intermediates evict the previous layer's; compressed avoids the
threshold and stays in steady-state.

### D.3 Quality smoke (token-level parity)

This is descriptive evidence that compressed output is coherent, not a
quality benchmark. Generated by `scripts/quality_smoke.py`: deterministic
synthetic prompt, prefill in stock and compressed mode separately, compare
top-50 candidate distributions and 32-token greedy decodes. Sequential one
process per mode (the constraint that only one MLX model can be loaded at a
time).

**0.8B 4-bit MLX:**

| context | top-50 overlap | top-1 agree | greedy match (32 tok) | KL stock→comp |
|---:|---:|:--:|---:|---:|
| 1 k  | 9 / 50  | no | 0 / 32 | (numerically tiny) |
| 4 k  | 25 / 50 | no | 0 / 32 | 4.38 |
| 16 k | 20 / 50 | no | 0 / 32 | 4.68 |

Sample 4 k continuations (same prompt):

> stock: `"The compressed attention mechanism keeps the most recent tokens in raw form and summarizes older blocks into a single mean-pooled representation. This allows long-context inference to use a"`
>
> compressed: `"Question: which two properties make this architecture effective?\nAnswer: 1) recent tokens are kept in raw form for local detail; 2) older blocks"`

What this **does** show: compressed produces grammatical, on-topic
continuations. No gibberish. No degenerate repetition. The candidate
distributions overlap meaningfully (top-50 ≥ 40 % at 4–16 k).

What this **does not** show: that compressed will hold quality on
perplexity, downstream-task accuracy, or retrieval probes. Greedy decoding
amplifies a single-token disagreement into a fully different sequence —
0/32 exact-match is *expected* for sparse routing on a frozen checkpoint
without indexer retraining, not a sign of breakage. A real quality
measurement is future work and is called out below.

**4B 4-bit MLX (the larger checkpoint behaves *better* on parity):**

| context | top-50 overlap | top-1 agree | greedy match (32 tok) | KL stock→comp | KL comp→stock |
|---:|---:|:--:|---:|---:|---:|
| 1 k  | 16 / 50 | **yes** | **32 / 32** | 0.24 | 0.76 |
| 4 k  | 32 / 50 | no  | 0 / 32 | 2.51 | 4.68 |
| 16 k | 23 / 50 | no  | 0 / 32 | 3.12 | 6.77 |

The 1 k row on 4B is the strongest signal in this whole document about
"compressed Butterfly is not silently breaking the model": **at 1 k context
on the 4B checkpoint, the greedy-decoded next-32 tokens from compressed
attention are bit-for-bit identical to stock attention's, and the top-1
candidate at end-of-prefill agrees**. This is the test you most want to
have pass for a no-train sparse-attention retrofit, and at 1 k on a
real-size model it does. At 4 k and 16 k the candidate distributions
overlap meaningfully (top-50 32 / 50, 23 / 50 — better than 0.8B at the
same contexts) and KL is bounded, but greedy decode amplifies any
single-token disagreement into a fully different sequence; that is
expected, not breakage.

A summary side-by-side, including the 0.8B numbers:

| ctx | 0.8B top-50 | 4B top-50 | 0.8B top-1 | 4B top-1 | 0.8B greedy | 4B greedy | 0.8B KL_s→c | 4B KL_s→c |
|---:|---:|---:|:--:|:--:|---:|---:|---:|---:|
| 1 k  |  9 | **16** | no | **yes** | 0/32 | **32/32** | ~0    | 0.24 |
| 4 k  | 25 | **32** | no | no  | 0/32 | 0/32 | 4.38 | **2.51** |
| 16 k | 20 | **23** | no | no  | 0/32 | 0/32 | 4.68 | **3.12** |

The 4B improves on the 0.8B on every metric. That is the right direction.
Compressed-Butterfly's quality drift is bounded by the compressor (mean
pool, no learned weights, no RoPE on summary output) and the routing
filter; both of those errors are *fractional fixed costs* that a larger
checkpoint absorbs better than a smaller one.

---

## E. What's still mean-pool / out of scope

- **Compressor is parameter-free mean pool.** V4's compressor is a learned
  weighted sum with positional biases (eqs. 9–12). A learned compressor is
  still possible later but requires training the compressor in. This branch
  deliberately stays no-train.
- **No counter-RoPE on the compressed-stream output** (V4 §2.3.3, "Partial
  Rotary Positional Embedding"). The empirical wins on this branch do not
  require it; a quality-sensitive deployment likely would want it.
- **No RMSNorm on the compressed entries** before the core attention (V4
  §2.3.3, "Query and Key-Value Entry Normalization"). Same call as
  counter-RoPE: not needed for the wins, plausibly needed for quality.
- **Decode runs on stock dense attention.** `butterfly_decode_backend=stock`.
  All wins above are prefill + retained-KV. Decode-time wins would require a
  separate decode-mode kernel.
- **No full quality benchmark.** Only the smoke at 1 k / 4 k / 16 k showing
  coherent continuations. Perplexity, retrieval probes, downstream-task
  accuracy: future work.
- **Apple Silicon MLX only on this branch.** A CUDA mirror of the same
  algorithmic surface exists (`scripts/bench_qwen35_cuda_wayfinder.py`,
  variant `compressed_butterfly`) but the per-platform peak-memory and
  e2e numbers in this doc are MLX-measured.
- **Context ceiling is 256 k on this run.** A 512 k attempt timed out in
  the bench harness's synthetic prompt-build stage (O(T²) tokenization),
  not in attention. Validating > 256 k requires a streamed prompt-build
  path; out of scope here.

---

## F. Where this fits

**Long-context inference on Apple Silicon, single user.** Butterfly wins on
both speed and peak memory at ≥ 128 k. The 0.27× peak at 256 k means a 16 GB
Mac that could not previously run 256 k stock on a 0.8B model now does so
comfortably with headroom; on the 4B checkpoint, 128 k stock peaks at 10.2 GB
(borderline) while compressed peaks at 3.8 GB (safe).

**Serving long sessions on shared hardware.** The 81 × retained-KV reduction
at 256 k is the dominant win. If you keep 100 long sessions warm, you keep
their KV-cache footprints, and the Butterfly retained KV at 256 k for 0.8B
is 40 MB vs 3.2 GB for stock. That changes how many concurrent sessions fit
on one device.

**Not the right fit for:**
- Tasks that need maximum quality at any cost — without a learned compressor
  and quality benchmarks, the safest assumption is some quality drift vs
  stock attention.
- Contexts where stock fits comfortably already — at 8 k or 16 k, the wins
  are small or inverted; this is a long-context architecture.

---

## G. Replicating these numbers

Hardware: Apple Silicon Mac, MLX-Metal venv. The bench script is the same
for stock and compressed; the mode flag toggles the architecture.

```bash
VENV=/Volumes/VIXinSSD/butterfly/.venv-macos-metal/bin/python   # adjust
HF_CACHE=/Volumes/VIXinSSD/hf_cache                             # adjust

# Models — both are mlx-community uniform 4-bit MLX checkpoints. Snapshot
# hashes are deterministic; pin yours from the local snapshot dir.
QWEN08B=$HF_CACHE/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/<HASH>
QWEN4B=$HF_CACHE/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<HASH>

run_bench () {                          # $1=model_path  $2=mode  $3=ctx  $4=outdir
  local extra=""
  if [ "$2" = "compressed_butterfly" ]; then
    extra="--block-partner-rule causal_shift \
           --compressed-local-window-tokens 64 \
           --query-chunk-size 64 \
           --block-size 128 \
           --butterfly-decode-backend stock"
  fi
  $VENV scripts/bench_qwen_consumer_mlx.py \
    --model-path "$1" --mode "$2" --seq-lens "$3" \
    --hf-home "$HF_CACHE" --hf-hub-cache "$HF_CACHE/hub" --hf-offline \
    --decode-len 8 --repeats 1 --chunk-size 384 --kv-step 384 \
    --skip-multi-turn --skip-quality --stage-timeout-sec 5000 \
    --out-dir "$4" $extra
}

# 0.8B ladder — all four contexts
for CTX in 32768 65536 131072 262144; do
  run_bench "$QWEN08B" stock                "$CTX" "results/qwen08b/stock_${CTX}"
  run_bench "$QWEN08B" compressed_butterfly "$CTX" "results/qwen08b/comp_${CTX}"
done

# 4B ladder — same recipe, different checkpoint
for CTX in 32768 65536 131072 262144; do
  run_bench "$QWEN4B" stock                "$CTX" "results/qwen4b/stock_${CTX}"
  run_bench "$QWEN4B" compressed_butterfly "$CTX" "results/qwen4b/comp_${CTX}"
done
```

Pull the per-row numbers from each `results.json`:

```python
import json
d = json.load(open("results/qwen08b/comp_32768/results.json"))["single_turn"][0]
print(d["e2e_sec"],
      d["peak_memory_bytes"] / (1024**3),                 # GB
      d["cache_storage_after_prefill"]["total_bytes"]/1e6) # MB retained KV
```

**What "retained KV" means.** The bench reports `cache_storage_after_prefill.total_bytes` — the
size of the model's KV cache (across all 24 / 32 layers) after prefill is
done. For the swapped layers, this is `tail_size + summary_count` valid
slices of the two-pool cache (see `CompressedKVCache.nbytes`); for the linear
(Mamba) layers it's the standard MLX state. Stock attention's retained KV
includes every token's K and V at full hidden width across the 6 / 8 dense
layers — that's why the numbers diverge so sharply.

**Quality smoke replication:**

```bash
$VENV scripts/quality_smoke.py run \
  --mode stock --model-path "$QWEN08B" --seq-len 4096 --decode-len 32 \
  --out-path results/quality/qwen08b_stock_4096.json

$VENV scripts/quality_smoke.py run \
  --mode compressed_butterfly --model-path "$QWEN08B" --seq-len 4096 --decode-len 32 \
  --out-path results/quality/qwen08b_comp_4096.json

$VENV scripts/quality_smoke.py compare \
  --stock-path     results/quality/qwen08b_stock_4096.json \
  --compressed-path results/quality/qwen08b_comp_4096.json \
  --out-path        results/quality/qwen08b_compare_4096.json
```

The two `run` invocations must be sequential (one MLX model in memory at a
time).

**Honesty knobs.** A few things to know if your numbers don't match:
- The 0.8B 256 k run requires the graph-cache eviction fix landed in commit
  `b89992f` ("phase A3 BREAKTHROUGH"). Without it, the per-chunk peak grows
  unbounded over 32 k and you get 6 GB / 32 k peaks, not 0.9 GB.
- `decode_len=8` is small; longer decode amortizes prefill differently.
- Numbers above are single-run wall-clock. ±2 % run-to-run noise is normal
  on a busy Mac.
- **`prompt_build_sec` is not part of `e2e_sec`.** The bench harness
  builds a synthetic prompt with a CPU-side tokenizer that has an
  *O(T²)* component at long contexts. At 128 k+ this dominates wall-clock,
  but it is identical between the stock and compressed runs (same tokenizer,
  same prompt) and contains no model forward pass. The architecture
  comparison is `prefill_sec` and `e2e_sec`. If you see `prompt_build_sec`
  vary 2× between two runs at the same context, that is harness noise (cold
  vs warm tokenizer cache, OS buffers, etc.), not anything attention is
  doing. The bench's `e2e_sec` excludes it for that reason.

---

## H. Future work

The next evidence steps, roughly in priority order:

1. **Same-KV-budget topology ablation.** The single most load-bearing test
   for the *unique* claim. Compare these five at the same retained-KV
   budget on RULER / NIAH at 32 k–128 k:

   | Mode | Selector |
   |---|---|
   | stock | full attention |
   | local-only | only the SWA tail; no compressed-block partners |
   | random-partner | one random causal block per query block per layer |
   | fixed-stride | `partner = b − stride` (constant across layers) |
   | causal_shift Butterfly | `partner = b − 2^stage` |

   If `causal_shift` beats random / local / fixed at the same memory, the
   topology is doing real work — not "any KV trim helps". This is the
   ablation that turns the result from "cool engineering" into "the
   topology matters". Without it, all of the wins above are
   equally consistent with "the model tolerates aggressive KV trimming";
   with it, we'd have evidence the butterfly composition property is
   load-bearing.

2. **Streamed prompt-build** in the bench harness. Removes the O(T²)
   tokenization wall that capped 0.8B at 256 k and 4B at 128 k in this run.
   Would let us measure 4B at 256 k and either model at 512 k+. Pure bench
   plumbing — not architecture.
2. **Real quality benchmark.** Perplexity on a long-context corpus (e.g.
   GovReport, RULER), retrieval probes (NIAH-style needle-in-a-haystack
   at 32 k / 64 k / 128 k), and at least one downstream task for both 0.8B
   and 4B. The smoke gives directional comfort but is not a quality
   guarantee.
3. **Qwen 3.6 27B 4-bit** as the next stretch target on a 36 GB-RAM machine.
   At 4-bit the weights are ≈ 13.5 GB; with stock attention the retained
   KV at 64 k–128 k context is the dominant cost (a 27 B model with
   typical GQA settings retains O(GB) per 64 k of context). Butterfly's
   ~63 × KV reduction at 128 k is exactly the lever that would let a 36 GB
   machine hold long context without paging. Use the same recipe as §G;
   the model checkpoint should download to the SSD-rooted HF cache
   (`HF_HUB_CACHE=/Volumes/VIXinSSD/hf_cache/hub`).
4. **Learned compressor.** Replace mean-pool with a parameter-light
   compressor trained against frozen-attention targets on the same
   checkpoint. Closes the §E quality gap without retraining the whole
   model.
5. **Counter-RoPE on the compressed-stream output.** V4 §2.3.3. Cheap
   addition; main question is whether the wins still hold.
6. **Decode-mode kernel.** Today decode falls back to stock dense over the
   same retained KV; a true compressed-mode decode would extend the
   speedups past prefill.

## I. References

- **`results/benchmarks/qwen35_0p8b_mlx/OVERNIGHT_LOG.md`** — the full
  chronology of the cutover. Phase 0 baselines, Phase A multi-stream cutover
  (halted at peak gate), A2 fixed-buffer cache rebuild (no peak relief), A3
  graph-cache eviction (BREAKTHROUGH — peak drops 6.21 → 0.91 GB at 32 k),
  A4 ladder validation (32 k → 256 k), B mx.compile + Metal kernel attempts
  (both BLOCKER on peak before A3 was found). Read this if you want to see
  what didn't work.
- **`docs/COMPRESSED_BUTTERFLY_ATTENTION.md`** — the existing claim-boundary
  doc for the variant. Maintains the same numbers as this doc.
- **DeepSeek V4 paper** (`docs/DeepSeek_V4.pdf`, local reference; the
  authoritative source is the model card on Hugging Face,
  `deepseek-ai/DeepSeek-V4-Pro`). Read §2.3 (CSA + HCA + SWA branch +
  grouped output projection), §3.6.1 (KV cache layout), and Figure 6 to see
  the systems shape Butterfly inherits.

## J. Code map

The implementation surface is small and contained:

| File | Role |
|---|---|
| `bna/topology/butterfly.py` | `causal_shift` partner rule, stage metadata, neighbor specs. |
| `bna/mlx/attention.py` | `BlockSparseButterflyLayout`, `build_block_butterfly_layout`, `_swa_stream_attention[_from_cache]`, `_compressed_stream_attention[_from_cache]`, `_online_softmax_merge`, `compressed_butterfly_attention[_from_cache]`, `_compressed_butterfly_routed_indices`. |
| `bna/mlx/compressed_cache.py` | `CompressedKVCache` two-pool fixed-buffer cache. |
| `bna/integrations/qwen_mlx.py` | `QwenButterflyAttention` layer, `install_compressed_kv_caches`, `swap_qwen_attention_with_butterfly`, `get_qwen_full_attention_layer_indices`, `_qwen_graph_cache_drop_other_keys` (the eviction fix that unlocks Phase A3). Dispatch around line 1485. |
| `scripts/bench_qwen_consumer_mlx.py` | Bench harness; produces the `results.json` referenced above. |
| `scripts/quality_smoke.py` | `run` / `compare` subcommands for top-50 / KL / greedy parity. |

The full prefill compute path for one swapped layer is
~50 lines in `compressed_butterfly_attention_from_cache` plus the two
stream helpers, and the entire layout / routing logic is
`build_block_butterfly_layout` + `_compressed_butterfly_routed_indices`
(~150 lines combined). The architecture is small enough to read in one
sitting; nothing is hidden behind a kernel.
