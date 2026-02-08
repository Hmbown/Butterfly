<div align="center">

# Wayfinder

**Hamiltonian Cycle Sparse Attention (HCSA)**

Sparse causal attention for language models using Hamiltonian cycles as the attention backbone.
O(T) attention with graph-theoretic connectivity backbone.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple_Silicon-000000.svg)](https://github.com/ml-explore/mlx)

</div>

---

Standard self-attention scales as O(T²) in compute. HCSA replaces it with a sparse graph induced by Hamiltonian cycles, restricting each token's attention to O(w) neighbors (cycle + local window) while maintaining a connected backbone over all positions. Optional landmarks add O(T/s) anchors per token for better effective diameter.

<details><summary><b>On reachability vs. effective receptive field</b></summary>

A single Hamiltonian cycle guarantees connectivity but has diameter O(T) — information cannot cross 256K tokens in ~61 layers via cycle edges alone. HCSA addresses this three ways: (1) **landmarks act as aggregation hubs** (every token attends to O(T/stride) landmark positions, providing O(1)-hop shortcuts), (2) **cycle resampling across layers** gives each layer a different long-range shortcut topology, and (3) **multi-head diversity** (each head can use an independent cycle) creates a union-of-cycles with expander-like properties and much smaller effective diameter.
</details>

## North Star: Kimi K2.5 on One Mac Studio

The goal is **[Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5) at 4-bit on a single Mac Studio (M3 Ultra, 512 GB)** — a 1.04-trillion-parameter MoE model running at its full 256K context length.

This is a hard problem on two fronts:

1. **The model barely fits.** At 4-bit, the full 1.04T-parameter model totals ~520 GB (1.04T × 0.5 bytes; actual size higher with group-quantization overhead) — already over the 512 GB budget before KV cache and activations. MoE helps (only 32B params active per token), so with expert offloading you keep the attention layers + hot experts resident and page cold experts from SSD. But memory headroom is razor-thin.

2. **Dense attention at 256K is a dealbreaker.** Even with MLA compressing KV cache ~25× (kv_lora_rank=512 vs 64-head MHA), dense O(T²) attention at 256K context is a **compute wall** — the quadratic cost dominates prefill and per-token decode scales as O(T) in context length. (Modern SDPA/FlashAttention kernels avoid materializing the full T×T matrix at inference, so the memory issue is primarily a **training activation cost**, but the compute is still O(T²) either way.) This is where HCSA comes in.

**What HCSA does:** replaces O(T²) attention compute with O(T·W) where W=64 — a **~4,000× attention compute reduction** per layer at 256K context (T/W ≈ 256,000/64 ≈ 4,000). This makes 256K context practical instead of theoretical, and frees memory headroom that the model weights desperately need.

| Component | Size | Notes |
|---|---|---|
| Model weights (4-bit) | ~520 GB | 1.04T × 0.5 bytes; experts are ~98% of params |
| Active per token (4-bit) | ~16 GB | 32B activated params × 0.5 bytes |
| KV cache at 256K (MLA) | ~16 GB | MLA compresses ~25× vs standard MHA (kv_lora_rank=512) |
| Attention intermediates (dense) | O(T²) per layer | The compute wall |
| **Attention intermediates (HCSA)** | **O(T·W) per layer** | **Makes 256K feasible** |

<details><summary><b>How we got these numbers</b> (from <a href="https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json">config.json</a>)</summary>

**Model weights — ~520 GB at 4-bit**

The [Kimi K2.5 paper](https://arxiv.org/abs/2602.02276) reports 1.04T total parameters. At INT4 (0.5 bytes/param): 1.04T × 0.5 = **520 GB**. This is the theoretical minimum; real quantized formats add per-group scale overhead (e.g., group_size=32 adds ~12.5%, pushing to ~585 GB). Kimi's [native INT4](https://huggingface.co/moonshotai/Kimi-K2-Thinking#4-native-int4-quantization) only quantizes routed experts (keeping attention at BF16), making the native checkpoint even larger (~594 GB). The 520 GB figure represents a uniform-4-bit lower bound.

Where the 1.04T comes from: 60 MoE layers × 384 routed experts × 3 matrices × 7168 × 2048 (SwiGLU) = **1.015T** in experts alone. Add 61 layers of MLA attention (~6.2B), shared experts (~2.6B), embeddings + LM head (~2.3B), dense-layer MLP (~0.4B) → **~1.026T** text params + 400M vision encoder ≈ 1.04T.

**Active per token — ~16 GB**

Kimi K2.5 activates 32B params per token (8 routed + 1 shared expert per layer, plus all attention layers). At 4-bit: 32B × 0.5 = **16 GB**. This is what must be resident in fast memory; cold experts can be paged from SSD.

**KV cache — ~16 GB at 256K context**

MLA stores a compressed latent **c_kv** of dimension kv_lora_rank=512 per token per layer (plus optionally qk_rope_head_dim=64 for RoPE keys, often recomputed instead of cached):

- MLA: 61 layers × 256K tokens × 512 × 2 bytes = **16.0 GB**
- Standard MHA (no MLA): 61 layers × 256K tokens × 64 heads × 112 head_dim × 2 (K+V) × 2 bytes = **~450 GB** ← impossible
- Compression ratio: 450 / 16 ≈ **28×** (we say ~25× conservatively including RoPE overhead)

</details>

The path: expert offloading handles the weight budget, HCSA handles the attention budget. Apple Silicon unified memory + NVMe makes expert paging viable; HCSA makes long-context generation fast enough to be usable.

<details><summary><b>Assumptions to validate</b></summary>

Expert paging depends on **expert locality** — a high cache hit-rate among the 384 experts so that cold-expert loads from SSD are infrequent. If routing is too diffuse (every token activates a different set of experts), random I/O thrash per layer per token will dominate latency. We expect locality to be high based on natural-language token distributions, but this must be measured empirically with real routing traces before the full Kimi K2.5-on-one-box claim is solid.
</details>

## How It Works

A Hamiltonian cycle visits every vertex exactly once and returns to the start. HCSA uses such cycles as an attention backbone:

| Component | What it provides |
|---|---|
| **Cycle neighbors** | Two adjacent nodes in the Hamiltonian cycle (long-range shortcuts) |
| **Local window** | The w preceding tokens (local context, always included) |
| **Landmarks** | Every s-th token (optional global anchors; adds O(T/s) neighbors) |
| **Self** | Always included |

Causality is enforced by masking any neighbor j > i. Without landmarks, neighborhood size is O(w) per token, independent of T. With landmarks at stride s, each token adds O(T/s) neighbors — a tunable compute/diameter trade-off.

**The permute-window fast path** reorders Q, K, V into cycle order so cycle-neighbor attention becomes a contiguous local window op — no gather/scatter needed. On GPT-2 at T≥4K (block-level) and GLM-4.7-Flash at T≥32K (end-to-end), the permute path meets or exceeds dense SDPA throughput.

<details><summary><b>Cycle strategies</b></summary>

| Strategy | Description | Properties |
|---|---|---|
| `random` | Random permutation per head | O(T), good baseline |
| `greedy` | Nearest-neighbor TSP heuristic using routing similarity | O(T²), content-aware |
| `regular_partition` | Cluster-balanced interleaving (k clusters) | O(T), 5–10% faster permute than random (better spatial locality) |
| `edge_disjoint` | Multiple Hamiltonian cycles with no shared edges | Maximizes edge diversity across cycles |
| `covering` | Greedy cycle generation targeting edge-coverage fraction | Monotonically approaches dense attention as cycles increase |
</details>

<details><summary><b>Graph-theoretic analysis toolkit</b></summary>

Diagnostic utilities in `hcsa/graph/analysis.py`, grounded in the Hamiltonian cycle literature:

- **Spectral gap** — eigenvalue gap of the cycle+window adjacency matrix; larger gap → faster information mixing.
- **Expansion proxy** — random-walk mixing time to stationarity; validates expander-like properties.
- **Resilience** — empirical survival rate under random edge drops, aligned with Draganić et al. (2025) Theorem 1.5: cycle+window graphs tolerate ~30% edge dropout while preserving Hamiltonicity (96% survival at drop_rate=0.3, T=128, w=32).
- **Regularity** — ε-regularity of edge distribution across clusters; validates structural uniformity.
- **Edge coverage** — fraction of all possible edges covered by a set of cycles; covering_cycles reaches ≥95% coverage given enough cycles.
</details>

## Results

All benchmarks on Apple Silicon (MLX backend), dated 2026-02-08. Dense baselines use fused SDPA.

<details><summary><b>Methodology note</b></summary>

Results below fall into two categories. **Throughput/memory benchmarks** (GLM, GPT-2, Qwen, Tiny synthetic) measure the HCSA attention mechanism swapped into pretrained models at inference — no retraining. These models were not trained with HCSA, so quality impact of the swap is not yet validated. **Quality (perplexity) results** come from small models trained from scratch with HCSA attention on TinyShakespeare.
</details>

### GLM-4.7-Flash (MLA + MoE, 4-bit) — Attention Swap, Current Focus

Pretrained MoE model ([mlx-community/GLM-4.7-Flash-4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit)) with HCSA swapped into all 47 attention layers at inference. Same MLA architecture as Kimi K2.5 at smaller scale. End-to-end single-turn with 256-token decode:

| Seq Length | E2E Delta | TTFT Delta | Memory Reduction |
|---:|---:|---:|---:|
| 2,048 | +6.5% | **-17%** | 0.4% |
| 8,192 | +20.8% | **-17%** | **5.2%** |
| 32,768 | **-4.8%** | **-96%** | **15.9%** |

At T=32K: 4.8% faster E2E, **25x faster time-to-first-token** (0.22s vs 5.42s), 15.9% less memory (30 GB vs 36 GB).

<details><summary>Source runs</summary>

- Dense: `20260208_glm47_consumer_dense_partial_r03` (no-swap control)
- HCSA: `20260208_glm47_consumer_hcsa_partial_r04` (47 layers swapped, permute path, w=64, th=49152)
</details>

65K chunked prefill (prefill-only, chunk_size=4096):

| Config | tok/s | Peak Memory |
|---|---:|---:|
| Dense (chunked, no-swap) | 90.0 | 33.2 GB |
| **HCSA (th=49152, q=384, h=2)** | **135** | **29.6 GB** |

50% faster prefill, **10.8% less memory**. Dense throughput is median of 4 control runs (range 83–105 tok/s); HCSA is median of 2 repeat runs. A known dense-fallback bug in the hybrid path means some chunks still run dense — the full sparse benefit at 65K is larger than shown.

<details><summary>Source runs</summary>

- Dense controls: `20260208_glm_chunked_dense_control_65k_matched` (83.5 tok/s), `_matched_repeat` (104.6), `phase2_dense_noswap_r01` (96.5), `_r02` (83.4)
- HCSA: `20260208_glm_hybrid_thresh49152_q384_h2_65k` (135.5 tok/s), `_repeat` (135.0)
</details>

### Earlier Milestones — Attention Swap (no retraining)

| Model | T | Throughput vs Dense | Memory vs Dense | Level | Source |
|---|---:|---:|---:|---|---|
| Tiny synthetic (4h/128d) | 4,096 | **3.2x** | **-62%** | attention | `tiny_batched_gate_mlx306_sdpa_dense` |
| GPT-2 (12h/768d) | 8,192 | **1.38x** | **-5%** | block | `northstar_after_v3_stable` |
| Qwen3-1.7B-4bit (GQA) | 8,192 | 0.47x | **-11%** | attention | `qwen_long_cmp_base2` |

All benchmarks use fused SDPA as the dense baseline. "Level" indicates what is timed: `attention` = attention kernel only, `block` = full transformer block including MLP.

Pattern: throughput advantage grows with sequence length; memory wins are consistent. The Qwen throughput gap is a known integration overhead issue, not fundamental — at block-level, Qwen is ~0.68x.

### Quality — Models Trained From Scratch with HCSA

These results are from small models trained from scratch on TinyShakespeare (MLX backend), **not** from swapping attention into pretrained models. Quality parity for the inference-swap approach (GLM, Qwen, GPT-2 above) is not yet validated.

| Setting | Dense ppl | HCSA ppl | Source |
|---|---:|---:|---|
| 200-step + edge-bias | 28.06 | 29.15 | `runs/mlx/20260207_050514` |
| 1k-step scheduled | 70.20 | **22.92** | `runs/mlx/20260207_050440` |

<details><summary>Note on 1k-step result</summary>

Dense overfits severely on this tiny dataset (final val ppl 70.20 vs best 16.25). HCSA's lower final ppl (22.92 vs best 16.89) is partly an implicit regularization effect — sparse attention constrains capacity. This is a real phenomenon but should not be read as "HCSA produces better language models."
</details>

<details><summary><b>Retro backfill</b> (HCSA-without-retro vs HCSA-with-retro)</summary>

| Setting | HCSA ppl (retro off) | HCSA ppl (retro on) | Source |
|---|---:|---:|---|
| Tiny-long 1k-step (alpha=0.2) | 91.02 | **79.92** | `runs/mlx/20260207_retro_{control,treatment}` |

Retro backfill improves perplexity 12% at 8% throughput cost. Training-only by default; disabled at inference for causality safety.
</details>

<details><summary><b>Graph-theory experiments (2026-02-08)</b></summary>

Validating theoretical properties of the cycle-based attention graph:

| Experiment | Result | Source |
|---|---|---|
| **Edge-disjoint cycles (d=2 vs d=1)** | -42% throughput, +0.3% memory — disjoint constraint adds overhead | `tiny_wayfinder/disjoint_d{1,2}` |
| **Covering cycles (d=1→4→8, T=512)** | L2-to-dense: -6.5% (d=4), -9.0% (d=8); cosine sim: +6.2%, +9.9% | `EXP-20260208-HCY-IDEA3-COVERING-RESULT` |
| **Resilience (T=128, w=32)** | 96% survival at 30% edge drop; 0% at 80% — aligned with Draganić Thm 1.5 | `EXP-20260208-HCY-IDEA2-RESILIENCE-RESULT` |
| **Regular partition (reg8/reg16 vs random)** | 5–10% faster permute, flat memory | `tiny_wayfinder/regularity_*` |

Covering cycles monotonically converge toward dense attention as cycle count increases — validates the theoretical motivation for Hamiltonian cycle backbones.
</details>

### Roadmap

| What | Status |
|---|---|
| GLM-4.7 consumer benchmark at 65K | In progress |
| Fix chunked prefill dense fallback | Diagnosed, next up |
| **Quality parity for attention-swap models** | **Not yet tested — next priority after 65K** |
| Quality parity validation (from-scratch) | Framework built |
| Kimi K2.5 integration | Target |

Full benchmark artifacts in [`benchmarks/mlx/`](benchmarks/mlx/) and [`notes/LAB_NOTEBOOK.md`](notes/LAB_NOTEBOOK.md).

### Claim Hygiene

| Status | Claim |
|---|---|
| ✅ Demonstrated | We reduce **attention compute** from O(T²) to O(T·W). |
| ✅ Demonstrated | End-to-end wins on GLM-4.7-Flash (MoE+MLA) at T≥32K on Apple Silicon. |
| ✅ Demonstrated | 3.2× attention throughput on tiny synthetic at T=4K vs fused SDPA baseline. |
| ⚠️ To validate | **Quality parity when swapping HCSA into pretrained models** — all throughput/memory results above are inference-swap only; output quality has not been measured. |
| ⚠️ To validate | Full Kimi K2.5-on-one-box requires (a) expert locality sufficient for SSD paging and (b) sparse pattern with small effective diameter under causality at ~61 layers. |

## Quickstart

```bash
git clone https://github.com/Hmbown/Wayfinder.git && cd Wayfinder
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

<details><summary>Project layout</summary>

```
hcsa/
├── model.py                   GPT with configurable attention (dense / hcsa)
├── attention_hcsa.py          Sparse attention via neighbor-index gathering
├── cycles.py                  Cycle construction (random, greedy, online, edge-disjoint, regular, covering)
├── graph/
│   ├── abi.py                 Graph ABI — backend-agnostic neighbor index format
│   └── analysis.py            Spectral gap, expansion, resilience, regularity, edge coverage
├── topology/core.py           First-class topology runtime
├── mlx/                       MLX backend (Apple Silicon)
│   ├── model.py               GPTMLX with dense / wayfinder_sparse / wayfinder_permute
│   └── attention.py           MLX attention (incl. wayfinder_covering_attention)
├── torch/                     PyTorch backend (CUDA + CPU)
│   ├── model.py               GPTTorch with same attention modes
│   └── attention_wayfinder_permute.py  Permute-window fast path
├── compiler/                  Graph compiler (.wf spec → IR → cache artifacts)
└── integrations/
    ├── qwen_mlx.py            Qwen3 attention swap
    ├── glm_mlx.py             GLM-4 attention swap (MLA + MoE)
    └── gpt2_mlx.py            GPT-2 attention swap
```
</details>

## Related Work

| Method | Approach | Graph Guarantee |
|---|---|---|
| **HCSA** (this work) | Hamiltonian cycle backbone + local window | Connected backbone; effective receptive field grows with depth + resampling |
| Exphormer (Shirzad et al., 2023) | Expander graphs + virtual global nodes for graph transformers | Near-Ramanujan spectral expansion |
| SPLA (Wang et al., 2026) | Block sparse top-k + residual linear attention for long tail | None (block selection is input-dependent) |
| NSA (DeepSeek, 2025) | Hardware-aligned sparse with token compression | None |
| Longformer (2020) | Sliding window + global tokens | Global tokens only |
| BigBird (2020) | Window + global + random edges | No cycle guarantee |

Exphormer is the closest prior work — it uses d>1 independent Hamiltonian cycles per head to create expander graphs with O(log n) mixing time for graph transformers. HCSA adapts this idea to causal LM attention with a permute-window fast path. SPLA tackles a complementary problem: instead of sparsifying the attention graph structure, it keeps block-sparse exact attention for "peaks" and compresses the remaining context via linear attention, avoiding the hard-truncation quality loss of NSA/InfLLM-v2.

## References

- Draganić et al. (2025). Hamilton cycles in pseudorandom graphs: resilience and approximate decompositions. [arXiv:2507.22807](https://arxiv.org/abs/2507.22807)
- Draganić et al. (2024). Hamiltonicity of expanders: optimal bounds and applications. [arXiv:2402.06603](https://arxiv.org/abs/2402.06603)
- Shirzad et al. (2023). Exphormer: Sparse Transformers for Graphs. ICML 2023. [arXiv:2303.06147](https://arxiv.org/abs/2303.06147)
- Wang et al. (2026). SPLA: Block Sparse Plus Linear Attention for Long Context Modeling. [arXiv:2601.22379](https://arxiv.org/abs/2601.22379)

## Citation

```bibtex
@software{bown2026hcsa,
  author    = {Bown, Hunter},
  title     = {{HCSA}: Hamiltonian Cycle Sparse Attention},
  year      = {2026},
  url       = {https://github.com/Hmbown/Wayfinder},
  version   = {0.3.0}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
