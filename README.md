# HCSA

Sparse attention using Hamiltonian cycles as shortcuts between token positions. Training-free — works on existing dense-attention models at inference time.

## How it works

In dense causal attention, every token attends to every earlier token — O(T^2) total edges. HCSA replaces this with a sparse graph where each token attends to a small neighborhood:

1. **Window** — the `w` most recent tokens (local context)
2. **Cycle shortcuts** — two tokens from a random Hamiltonian cycle
3. **Landmarks** — every `s`-th token (periodic anchors)

The Hamiltonian cycle is the key idea. Take all T token positions and shuffle them into a random loop — a permutation where the last element connects back to the first. Each token gets two "shortcut" neighbors from this loop. Because the cycle visits every position exactly once, any token can reach any other through a chain of hops. Global information flow, without global attention.

Causal masking still applies — tokens only attend to earlier positions. Each token attends to at most `w + 2 + T/s` neighbors instead of all `T-1` predecessors.

![Attention patterns at T=64: dense, sliding window, Longformer-style, BigBird-style, HCSA](docs/assets/attention_comparison_5panel.png)

## Results

Measured on Apple Silicon (M4 Max, 36 GB). Prefill tok/s — decode uses dense attention by default.

### Qwen 3.5 9B Q8 (llama.cpp, Apple Silicon)

| Context | Dense tok/s | HCSA tok/s | Delta |
|--------:|------------:|-----------:|------:|
| 4,096   | 7,038       | 7,038      | 0%    |
| 32,768  | 610         | 610        | 0%    |
| 65,536  | 517         | 631        | +22%  |

Qwen 3.5 is a hybrid model — 8 full attention layers and 24 [Gated DeltaNet](https://arxiv.org/abs/2412.06464) (linear attention) layers. HCSA only modifies the attention layers, so the end-to-end speedup is bounded by Amdahl's law. The sparse gate activates above 32K context, where attention becomes a meaningful fraction of compute.

## Try it

```bash
git clone https://github.com/Hmbown/hcsa && cd hcsa
pip install -e ".[mlx]"

# Chat with HCSA-accelerated GLM-4.7-Flash (downloads automatically)
python3 scripts/chat_glm_wayfinder.py

# Dense baseline for comparison
python3 scripts/chat_glm_wayfinder.py --mode dense
```

## Limitations

- **Approximation.** Models were trained with dense attention. HCSA changes which tokens attend to which at inference time. Evaluate perplexity on your workload.
- **Prefill only.** Decode uses dense attention by default — at q_len=1, dense is already fast.
- **Graph construction on CPU.** At short context the overhead exceeds the savings, which is why the sparse gate only activates at longer sequences.

## Related work

There are a few ways to deal with the quadratic cost of attention. HCSA is training-free — it works on existing models without retraining.

- **[MLA](https://arxiv.org/abs/2502.07864)** (DeepSeek, Kimi K2) — compresses the KV cache, reducing memory. Doesn't reduce compute.
- **[Gated DeltaNet](https://arxiv.org/abs/2412.06464)** (Qwen 3.5) — replaces softmax attention with linear recurrence. Different computation entirely.
- **[NSA/DSA](https://arxiv.org/abs/2502.11089)** (DeepSeek, GLM-5) — trained sparse attention with hardware-aligned kernels. Strong results, but requires training with the pattern.
- **[MoBA](https://arxiv.org/abs/2502.13189)** (Kimi) — learned block-level attention gating. Also requires training.
- **[FlexPrefill](https://arxiv.org/abs/2502.20766)** (ByteDance, ICLR 2025 Oral) — training-free like HCSA, but content-aware: estimates each head's attention distribution and allocates sparsity budgets per head per input.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** — exact dense attention with better memory access. Same work, faster kernels.

## Project structure

```
hcsa/
├── graph/abi.py          # Graph ABI: neigh_idx [T,D] int32, edge_type uint8
├── mlx/                  # MLX backend (fused dispatch, attention, model)
├── torch/                # PyTorch/CUDA backend
├── integrations/         # Model wrappers (GLM, Qwen, GPT-2)
├── topology/             # Graph construction
├── compiler/             # Graph spec compiler (.wf -> ABI)
└── cycles.py             # Hamiltonian cycle generation
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Benchmark evidence: [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md).

## License

MIT
