<div align="center">

# HCSA

**Hamiltonian Cycle Sparse Attention**

Sparse causal attention for language models using Hamiltonian cycles as the attention backbone.
O(T) attention with graph-theoretic connectivity guarantees.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple_Silicon-000000.svg)](https://github.com/ml-explore/mlx)

</div>

---

Standard self-attention scales as O(T^2) in both compute and memory. HCSA replaces it with a sparse graph induced by Hamiltonian cycles, restricting each token's attention to O(w + k) neighbors while guaranteeing every position remains reachable.

## North Star: Kimi K2.5 on One Mac Studio

The goal is **[Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2) at 4-bit on a single Mac Studio M4 Ultra (512 GB)** — a 1.04-trillion-parameter MoE model running at its full 256K context length.

This is a hard problem on two fronts:

1. **The model barely fits.** At 4-bit, all 384 experts total ~520 GB — already over the 512 GB budget before KV cache and activations. MoE helps (only 32B params active per token), so with expert offloading you keep the attention layers + hot experts resident and page cold experts from SSD. But memory headroom is razor-thin.

2. **Dense attention at 256K is a dealbreaker.** Even with MLA compressing KV cache ~10x, dense O(T^2) attention at 256K context takes both too much memory and too much compute for practical generation on consumer hardware. This is where HCSA comes in.

**What HCSA does:** replaces O(T^2) attention with O(T * W) where W=64 — a ~4,000x compute reduction per layer. This makes 256K context practical instead of theoretical, and frees memory headroom that the model weights desperately need.

| Component | Size | Notes |
|---|---|---|
| Model weights (4-bit) | ~520 GB | All 384 experts; exceeds 512 GB alone |
| Active expert weights | ~41 GB | 9 active experts + non-MoE layers |
| KV cache at 256K (MLA) | ~16 GB | MLA compresses ~10x vs standard |
| Attention intermediates (dense) | O(T^2) per layer | The memory/compute wall |
| **Attention intermediates (HCSA)** | **O(T * W) per layer** | **Makes 256K feasible** |

The path: expert offloading handles the weight budget, HCSA handles the attention budget. Apple Silicon unified memory + NVMe makes expert paging viable; HCSA makes long-context generation fast enough to be usable.

## How It Works

A Hamiltonian cycle visits every vertex exactly once and returns to the start. HCSA uses such cycles as an attention backbone:

| Component | What it provides |
|---|---|
| **Cycle neighbors** | Two adjacent nodes in the Hamiltonian cycle (long-range shortcuts) |
| **Local window** | The w preceding tokens (local context, always included) |
| **Landmarks** | Every k-th token (optional global anchors) |
| **Self** | Always included |

Causality is enforced by masking any neighbor j > i. Neighborhood size per token is O(w + k), independent of T.

**The permute-window fast path** reorders Q, K, V into cycle order so cycle-neighbor attention becomes a contiguous local window op — no gather/scatter needed, throughput exceeds dense at long sequences.

## Results

All benchmarks on Apple Silicon (MLX backend). Dense baseline uses fused SDPA.

### GLM-4.7-Flash (MLA + MoE, 4-bit) — Current Focus

This model uses the same MLA architecture as Kimi K2.5 at smaller scale. End-to-end single-turn with 256-token decode:

| Seq Length | E2E Delta | TTFT Delta | Memory Reduction |
|---:|---:|---:|---:|
| 2,048 | +6.5% | **-17%** | 0.4% |
| 8,192 | +20.8% | **-17%** | **5.2%** |
| 32,768 | **-4.8%** | **-96%** | **15.9%** |

At T=32K: 4.8% faster E2E, **25x faster time-to-first-token**, 15.9% less memory (30 GB vs 36 GB).

65K chunked prefill (prefill-only):

| Config | tok/s | Peak Memory |
|---|---:|---:|
| Dense (chunked) | 90.0 | 33.2 GB |
| **HCSA** | **100.1** | **29.6 GB** |

7% faster, **10.8% less memory**. A known dense-fallback bug currently masks the full sparse benefit at 65K — fixing it is the next priority.

### Earlier Milestones

| Model | T | Throughput vs Dense | Memory vs Dense |
|---|---:|---:|---:|
| Tiny synthetic (4h/128d) | 4,096 | **9.2x** | **-90%** |
| GPT-2 (12h/768d, block-level) | 8,192 | **1.38x** | **-5%** |
| Qwen3-1.7B-4bit (GQA) | 8,192 | 0.47x | **-11%** |

Pattern: throughput advantage grows with sequence length; memory wins are consistent. The Qwen throughput gap is a known integration overhead issue, not fundamental.

### Quality

| Setting | Dense ppl | HCSA ppl |
|---|---:|---:|
| TinyShakespeare 200-step + edge-bias | 28.06 | 29.15 |
| TinyShakespeare 1k-step scheduled | 70.20 | **22.92** |
| Tiny-long retro backfill (alpha=0.2) | 91.02 | **79.92** |

Retro backfill improves perplexity 12% at 8% throughput cost. Training-only by default.

### Roadmap

| What | Status |
|---|---|
| GLM-4.7 consumer benchmark at 65K | In progress |
| Fix chunked prefill dense fallback | Diagnosed, next up |
| Quality parity validation | Framework built |
| Kimi K2.5 integration | Target |

Full benchmark artifacts in [`benchmarks/mlx/`](benchmarks/mlx/) and [`notes/LAB_NOTEBOOK.md`](notes/LAB_NOTEBOOK.md).

## Quickstart

```bash
git clone https://github.com/Hmbown/hcsa.git && cd hcsa
pip install -e ".[dev]"

# PyTorch: train dense vs HCSA
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn dense --steps 200
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn hcsa \
  --cycle random --window 32 --landmark-stride 32 --steps 200
```

**MLX backend** (Apple Silicon):

```bash
pip install -e ".[mlx]"

python scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32
```

## Architecture

```
hcsa/
├── model.py                   GPT with configurable attention (dense / hcsa)
├── attention_hcsa.py          Sparse attention via neighbor-index gathering
├── cycles.py                  Cycle construction (random, greedy, online)
├── graph/abi.py               Graph ABI — backend-agnostic neighbor index format
├── topology/core.py           First-class topology runtime
├── mlx/                       MLX backend (Apple Silicon)
│   ├── model.py               GPTMLX with dense / wayfinder_sparse / wayfinder_permute
│   └── attention.py           MLX attention implementations
├── torch/                     PyTorch backend (CUDA + CPU)
│   ├── model.py               GPTTorch with same attention modes
│   └── attention_wayfinder_permute.py  Permute-window fast path
├── compiler/                  Graph compiler (.wf spec → IR → cache artifacts)
└── integrations/
    ├── qwen_mlx.py            Qwen3 attention swap
    └── glm_mlx.py             GLM-4 attention swap (MLA + MoE)
```

**Key design boundary:** the Graph ABI is the bridge between backends. Strategies produce cycles on CPU, the ABI wraps them as NumPy arrays, each backend converts to native tensors. The permute fast path reorders Q/K/V into cycle order for contiguous local-window attention. The sparse gather path serves as correctness reference.

## Related Work

| Method | Approach | Graph Guarantee |
|---|---|---|
| **HCSA** (this work) | Hamiltonian cycle backbone + local window | Every token reachable via cycle |
| NSA (DeepSeek, 2025) | Hardware-aligned sparse with token compression | None |
| Longformer (2020) | Sliding window + global tokens | Global tokens only |
| BigBird (2020) | Window + global + random edges | No cycle guarantee |

## References

- Draganic et al. (2025). Hamilton cycles in pseudorandom graphs. [arXiv:2507.22807](https://arxiv.org/abs/2507.22807)
- Draganic et al. (2024). Hamiltonicity of expanders: optimal bounds and applications. [arXiv:2405.18875](https://arxiv.org/abs/2405.18875)

## Citation

```bibtex
@software{bown2026hcsa,
  author    = {Bown, Hunter},
  title     = {{HCSA}: Hamiltonian Cycle Sparse Attention},
  year      = {2026},
  url       = {https://github.com/Hmbown/hcsa},
  version   = {0.3.0}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
