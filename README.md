# Wayfinder

Block-structured attention acceleration for long-context inference. Training-free — works on existing dense-attention models at inference time.

## How it works

Dense causal attention computes scores between every token and every earlier token — O(T^2) work per layer. At 262K context that's ~34 billion score computations per layer. Most of that work contributes negligibly to the output: attention distributions in trained models are typically sparse, concentrating on local context and a few distant positions.

Wayfinder exploits this by replacing dense attention with a block-sparse pattern where each block of tokens only attends to a small, fixed set of other blocks. The total work per layer becomes O(T·B) where B is the constant number of blocks in the support — linear in sequence length instead of quadratic. That's where the speedup comes from: as context grows, dense attention's cost grows quadratically while Wayfinder's grows linearly, so the gap widens at longer sequences.

### Block topology

The sequence is partitioned into fixed-size blocks (typically 128 tokens). Each query block attends to:

1. **Self + local predecessors** — the block itself and a fixed number of immediately preceding blocks
2. **Partner blocks** — deterministic long-range blocks from a staged communication schedule
3. **Sink blocks** — a small set of early-sequence blocks (handles the [attention sink](https://arxiv.org/abs/2309.17453) phenomenon)

### Global mixing across layers

Restricting each layer to a few blocks would lose long-range information — except the partner selection changes by layer. The schedule is borrowed from interconnect network theory (the same math behind switch fabrics and butterfly networks): at layer `l`, each block `b` connects to block `b XOR (1 << (l mod log₂ N))` (or a bit-reversal / Benes variant). After O(log N) layers, every block can reach every other block through a chain of these hops.

This is the key property: no single layer has global attention, but the *stack* of layers provides global reachability. Information flows everywhere in O(log N) hops, while each individual layer does only O(T) work.

### Why it's hardware-friendly

The block support is static and known at compile time — no routing decisions, no content-dependent gating, no irregular memory access. Each block's attention is a regular dense matmul against a small contiguous set of K/V blocks, which maps directly to Triton/CUDA block operations and stays on the fast path of the memory hierarchy.

## Results

### Qwen 3.5 9B (CUDA, Triton block-sparse, DGX Spark GB10)

Backbone prefill wall-clock, `block_sparse` path, `engine=triton`, `block_size=128`. Only the 8 full-attention layers are replaced. 8K–131K use BF16 weights (`warmup=1`, `repeats=3`). 262K uses FP8 weight-only quantization via `torchao` (`warmup=1`, `repeats=1`) — BF16 OOMs at this length.

| Context | Dense (ms) | Wayfinder (ms) | Speedup | Peak memory |
|--------:|-----------:|----------------:|--------:|------------:|
| 8,192   | 4,961      | 4,824           | 1.03x   | 18.4 GB     |
| 32,768  | 20,669     | 19,414          | 1.06x   | 23.5 GB     |
| 65,536  | 44,423     | 38,017          | 1.17x   | 30.2 GB     |
| 98,304  | 69,574     | 59,218          | 1.17x   | 37.0 GB     |
| 131,072 | 96,021     | 78,645          | 1.22x   | 43.8 GB     |
| 262,144 | 208,589    | 153,112         | **1.36x** | 64.4 GB   |

Wayfinder prefill throughput stays roughly flat as context grows while dense throughput degrades — the speedup increases with sequence length. Peak memory is matched because only 8 of 32 layers are swapped and the block topology overhead is near zero. At 262K, FP8 weight-only quantization (`--quantize fp8-weight-only`) reduces model memory from ~19 GB to ~11 GB, leaving ~100 GB free on the DGX Spark's 120 GB unified memory.

Qwen 3.5 is a hybrid model — 8 full-attention layers (`Qwen3_5Attention`, GQA) and 24 linear-attention layers (`Qwen3_5GatedDeltaNet`, [Gated DeltaNet](https://arxiv.org/abs/2412.06464)). Wayfinder only modifies the 8 full-attention layers. The linear-attention layers use a completely different computation (linear recurrence with causal conv1d + gating) and are **not** targeted. End-to-end speedup is bounded by Amdahl's law.

**Quality impact** (logit divergence, same model/input, dense vs Wayfinder):

| Context | Top-1 agreement | Relative L2 |
|--------:|----------------:|------------:|
| 4,096   | 99.88%          | 0.291       |
| 16,384  | 94.44%          | 0.340       |

At 4k context, Wayfinder predictions are nearly identical to dense. At 16k, ~5.6% of top-1 token predictions change. Quality degrades with context length — evaluate on your workload before deploying.

### Qwen 3.5 35B A3B FP8 (CUDA, Triton block-sparse, DGX Spark GB10)

Backbone prefill wall-clock on the text path (`forward_target=backbone`). The checkpoint is Qwen's native fine-grained FP8 release, so the working load path is `--quantize none` with BF16 compute. Only the 10 `full_attention` layers are replaced. The 30 `linear_attention` / MoE layers stay stock.

| Context | Dense (ms) | Wayfinder (ms) | Speedup | Peak memory |
|--------:|-----------:|----------------:|--------:|------------:|
| 8,192   | 8,800      | 8,589           | 1.02x   | 35.7 GB     |
| 32,768  | 25,602     | 25,186          | 1.02x   | 40.5 GB     |
| 65,536  | 52,810     | 49,433          | 1.07x   | 46.9 GB     |
| 131,072 | 115,919    | 98,460          | 1.18x   | 59.7 GB     |

Wayfinder-only ceiling probes on the same setup:

| Context | Wayfinder (ms) | Wayfinder tok/s | Peak memory |
|--------:|----------------:|----------------:|------------:|
| 163,840 | 125,504         | 1,306           | 66.1 GB     |
| 196,608 | 144,106         | 1,364           | 72.5 GB     |
| 229,376 | 185,970         | 1,233           | 78.9 GB     |

## Try it

```bash
git clone https://github.com/Hmbown/Wayfinder && cd Wayfinder
pip install -e ".[dev]"

# Benchmark dense vs Wayfinder prefill
python scripts/bench_qwen35_cuda_wayfinder.py \
    --model-path <path-to-Qwen3.5-9B> \
    --path block_sparse \
    --block-size 128 \
    --engine triton \
    --seq-lens 4096 8192 16384 32768

# Serve with OpenAI-compatible API
python scripts/serve_qwen_wayfinder_cuda.py \
    --model-path <path-to-Qwen3.5-9B> \
    --mode wayfinder \
    --port 8012
```

## Limitations

- **Approximation.** Models were trained with dense attention. Wayfinder changes which tokens attend to which at inference time. Evaluate perplexity on your workload. Quality impact is under measurement but not yet validated at scale.
- **Prefill only.** Decode uses dense attention by default — at q_len=1, dense is already fast.
- **Hybrid models: only full-attention layers.** On Qwen 3.5, only the 8 `full_attention` layers are swapped. The 24 `linear_attention` (Gated DeltaNet) layers are a different computation entirely and are not targeted.

## Related work

There are a few ways to deal with the quadratic cost of attention. Wayfinder is **training-free** — it works on existing dense-attention models at inference time, with no retraining. This makes it complementary to training-time methods rather than competing with them.

**Training-free (like Wayfinder):**
- **[FlexPrefill](https://arxiv.org/abs/2502.20766)** (ByteDance, ICLR 2025 Oral) — content-aware: estimates each head's attention distribution and allocates sparsity budgets per head per input.

**Requires training:**
- **[MoDA](https://arxiv.org/abs/2603.15619)** (ByteDance/HUST) — mixture-of-depths attention: each head attends to KV pairs from preceding layers (depth stream) in addition to the current sequence. Orthogonal to sequence-level sparsity — could be combined with Wayfinder.
- **[NSA/DSA](https://arxiv.org/abs/2502.11089)** (DeepSeek, GLM-5) — trained sparse attention with hardware-aligned kernels. Strong results, but requires training with the pattern.
- **[MoBA](https://arxiv.org/abs/2502.13189)** (Kimi) — learned block-level attention gating.

**Different computation:**
- **[MLA](https://arxiv.org/abs/2502.07864)** (DeepSeek, Kimi K2) — compresses the KV cache, reducing memory. Doesn't reduce compute.
- **[Gated DeltaNet](https://arxiv.org/abs/2412.06464)** (Qwen 3.5) — replaces softmax attention with linear recurrence. Different computation entirely.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** — exact dense attention with better memory access. Same work, faster kernels.

## Project structure

```
hcsa/
├── graph/abi.py          # Graph ABI: neigh_idx [T,D] int32, edge_type uint8
├── torch/                # PyTorch/CUDA backend (Triton block-sparse kernels)
├── integrations/         # Model wrappers (Qwen, GLM, Nemotron)
├── topology/             # Block topology construction
└── compiler/             # Graph spec compiler (.wf -> ABI)
```

## License

MIT
