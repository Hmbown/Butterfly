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

Standard self-attention computes scores over all T previous tokens per position, requiring O(T^2) work and memory. HCSA replaces this with a sparse graph induced by Hamiltonian cycles over token positions, restricting each token's attention to O(w + k) neighbors while guaranteeing that every position remains reachable.

## Results

Measured on Apple M-series silicon (MLX backend). Dense baseline uses standard causal self-attention.

### Current Qwen3-1.7B-4bit (attention path, isolated process per sequence, retro off)

Source runs:
- `benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_matrix_isolated_base/`
- `benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_long_cmp_base2/`

**Throughput** (tokens/sec)

| Seq Length | Dense | HCSA (permute) | Ratio |
|---:|---:|---:|---:|
| 256 | 151,996 | 77,005 | 0.51x |
| 512 | 167,220 | 115,664 | 0.69x |
| 1,024 | 177,895 | 146,204 | 0.82x |
| 2,048 | 292,217 | 158,745 | 0.54x |
| 4,096 | 241,498 | 80,475 | 0.33x |
| 8,192 | 174,107 | 81,139 | 0.47x |

**Peak memory** (attention path)

| Seq Length | Dense | HCSA (permute) | Reduction |
|---:|---:|---:|---:|
| 256 | 13.9 MB | 23.3 MB | -67.5% |
| 512 | 20.7 MB | 32.6 MB | -57.5% |
| 1,024 | 34.3 MB | 47.4 MB | -38.1% |
| 2,048 | 61.6 MB | 77.0 MB | -25.1% |
| 4,096 | 116.1 MB | 105.6 MB | **+9.1%** |
| 8,192 | 225.2 MB | 201.0 MB | **+10.8%** |

### Retrocausal Backfill Ablation (Qwen, `alpha=0.2`, offsets `[1,2,4]`)

Source runs:
- `benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_matrix_isolated_retro_infer/`
- `benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_long_cmp_retro_infer2/`

Current retro setting is **not** the default because it regresses both throughput and memory in this Qwen setup:
- `T=8192` dense peak: `225,182,248`
- `T=8192` retro peak: `262,084,974` (worse than dense)

Note: benchmarking retro at inference now requires explicit `--retro-allow-inference` (default is training-only safety).

### Historical TinyShakespeare Quality Reference (non-Qwen)

| Setting | Dense val ppl | HCSA val ppl | Gap |
|---|---:|---:|---:|
| 200-step baseline | 28.13 | 31.66 | +3.53 |
| 200-step + edge-bias + window-drop | 28.06 | 29.15 | +1.09 |
| 1k-step scheduled | 70.20 | 22.92 | -47.29 |

These quality numbers are from the original non-Qwen track and remain the reference for cycle-push schedule behavior.

### Tiny Gate (current, MLX 0.30.6, non-Qwen)

Source run:
- `benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306/results.json`

Command:
- `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306`

| Seq Length | Dense tok/s | HCSA (permute) tok/s | Ratio | Dense step bytes | HCSA step bytes | Memory reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 532,143.84 | 2,278,641.85 | **4.282x** | 561,922,092 | 118,458,644 | **78.92%** |
| 4,096 | 261,828.24 | 2,420,297.41 | **9.244x** | 2,203,552,300 | 227,425,708 | **89.68%** |

Historical Tiny targets were `1.19x / 37%` at `T=2048` and `2.45x / 67%` at `T=4096`; current measured run exceeds all four thresholds.

### Tiny-long Retro Backfill A/B (matched settings)

Source runs:
- Control (retro off): `runs/mlx/tiny_long/20260207_retro_control/summary.json`
- Treatment (retro on): `runs/mlx/tiny_long/20260207_retro_treatment/summary.json`

Both runs use:
- `scripts/run_mlx_experiment_tiny_long.py`
- `--wayfinder-attn wayfinder_permute`
- `steps=1000`, `batch_size=8`, `seq_len=128`

| Run | Wayfinder val ppl | Wayfinder avg tok/s | Wayfinder peak memory |
|---|---:|---:|---:|
| Retro off | 91.0188 | 236,987.47 | 120,135,536 |
| Retro on (`alpha=0.2`, training-only, causal-only) | 79.9204 | 218,823.10 | 121,956,224 |

Retro-on vs control (Wayfinder):
- val ppl improves by `11.10` (`-12.19%`, lower is better)
- throughput decreases by `18,164.37 tok/s` (`-7.66%`)
- peak memory increases by `1,820,688` bytes (`+1.52%`)

### GPT-2 North Star (MLX 0.30.6)

Source runs:
- Before: `benchmarks/mlx/gpt2_wayfinder/20260207_180248_northstar_before/`
- After: `benchmarks/mlx/gpt2_wayfinder/20260207_northstar_after_v3_stable/`

Command:
- `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 4 --iters 8 --path permute --window 64 --landmark-stride 64 --seed 42`

**Attention-level throughput** (after lazy eval fix):

| Seq Length | Dense tok/s | HCSA (permute) tok/s | Ratio | C_fit | T* |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1,034,648 | 497,903 | 0.481x | 32.99 | 4,256 |
| 4,096 | 778,769 | 762,035 | **0.979x** | — | — |
| 8,192 | 489,173 | 854,676 | **1.747x** | — | — |

**Block-level throughput** (whole transformer layer, fairer comparison):

| Seq Length | Dense tok/s | HCSA (permute) tok/s | Ratio | Dense peak mem | HCSA peak mem | Mem delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 417,016 | 396,494 | 0.951x | 132.6 MB | 127.0 MB | **-4.2%** |
| 4,096 | 422,338 | 426,723 | **1.010x** | 237.1 MB | 225.9 MB | **-4.8%** |
| 8,192 | 321,248 | 444,108 | **1.382x** | 446.3 MB | 423.5 MB | **-5.1%** |

### GLM-4.7-Flash Long-Context Chunked Prefill (current primary objective)

Source runs:
- Monolithic baseline: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/`
- Chunked dense controls: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_chunked_dense_control_65k_matched/` and `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_chunked_dense_control_65k_matched_repeat/`
- Tuned HCSA long-context runs:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k/`
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k_repeat/`

Settings:
- `seq_len=65536`, `chunk_size=4096`, `decode_len=0`, retro off.
- Tuned reproducible HCSA config: `threshold=49152`, `query_chunk_size=384`, `head_chunk_size=2`.

| Configuration | Prefill sec | Prefill tok/s | Peak memory |
|---|---:|---:|---:|
| Monolithic dense baseline | 126.69 | 517.30 | 44.89 GB |
| Chunked dense control (median of 2) | 705.95 | 94.02 | 33.16 GB |
| HCSA tuned (median of 2) | **484.48** | **135.27** | **29.59 GB** |

HCSA tuned vs chunked dense control median:
- `-221.47s` prefill latency
- `+41.26 tok/s` throughput
- `10.77%` lower peak memory

HCSA tuned vs monolithic baseline:
- still slower in absolute prefill (`+357.79s`) but with `34.09%` lower peak memory

Exploratory note: `threshold=45056` produced one very fast run but failed reproducibility (high variance on repeat), so `49152` is kept as the current reproducible default.

### Fairness Correction

Dense baseline uses MLX fused SDPA (`mx.fast.scaled_dot_product_attention`), which avoids materializing T×T attention weights (memory ~O(T)) but compute remains ~O(T²). HCSA permute-window does O(T·W) compute via chunked local-window attention in permuted space.

At the attention level, HCSA peak memory includes graph cache and permutation artifacts (~40 MB fixed overhead). At the block level (whole transformer layer), this overhead is amortized and HCSA uses 4–5% *less* memory than dense across all tested sequence lengths.

The throughput win at long T is primarily a **compute complexity** advantage: O(T·W) vs O(T²). The memory picture depends on implementation fusion level — fused SDPA can have lower attention-level intermediates than unfused chunked attention, but whole-layer memory favors HCSA.

### Gap to Target

Target direction (from original HCSA results): win throughput by `T>=2048` and achieve strong memory reductions at `T=2048/4096`.

Current status:
- Tiny non-Qwen gate is met/exceeded at both `T=2048` and `T=4096`.
- GPT-2 North Star gate: block-level throughput ≥0.95x at all T, ≥1.0x at T≥4096, memory better at all T.
- Qwen still shows the long-context memory trend, but throughput crossover remains open in current isolated runs.

Benchmark artifacts: [`benchmarks/mlx/`](benchmarks/mlx/) | Training runs: [`runs/`](runs/) | Experiment log: [`notes/LAB_NOTEBOOK.md`](notes/LAB_NOTEBOOK.md)

## How It Works

A Hamiltonian cycle visits every vertex in a graph exactly once and returns to the start. HCSA uses one or more such cycles over token positions as an undirected attention backbone.

Each token attends to a small, structured neighborhood:

| Component | What it provides |
|---|---|
| **Cycle neighbors** | The two adjacent nodes in a Hamiltonian cycle (long-range shortcuts) |
| **Local window** | The w preceding tokens (local context, always included) |
| **Landmarks** | Every k-th token (optional global anchors) |
| **Self** | Always included |

Causality is enforced by masking any neighbor j > i. Total neighborhood size per token is O(w + k), independent of sequence length T.

**The permute-window fast path** reorders Q, K, V into cycle order so that cycle-neighbor attention becomes a contiguous local window operation. This avoids gather/scatter overhead and enables throughput exceeding dense attention at long sequences, since the effective attention window is constant regardless of T.

### Cycle Strategies

| Strategy | Complexity | Description |
|---|---|---|
| `random` | O(T) | Random permutation per head. No data dependence. |
| `greedy` | O(T^2) | Nearest-neighbor heuristic using learned routing similarity |
| `online_insertion` | O(T) per step | Incremental cycle maintenance as tokens arrive |

### Theoretical Foundation

The approach draws on results showing that pseudorandom graphs admit approximate Hamiltonian decompositions (Draganić et al., 2025). This provides a mathematical basis for using Hamiltonian cycles as sparse connectivity structures: they are guaranteed to exist in sufficiently well-connected graphs and can tile the edge set near-optimally.

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

# Scaling benchmark
python scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32 \
  --warmup 2 --iters 4
```

**Qwen3-4B integration** (MLX, long-context):

```bash
# Attention-level benchmark against dense baseline
python scripts/bench_qwen_wayfinder_mlx.py \
  --model-path mlx-community/Qwen3-4B-4bit \
  --seq-lens 2048 8192 32768 \
  --batch 1 --path permute --window 64 --landmark-stride 64

# LoRA fine-tuning with HCSA attention
python scripts/train_qwen3_4b_wayfinder_mlx.py \
  --model-path mlx-community/Qwen3-4B-4bit \
  --dataset-dir datasets/qwen3_4b/ \
  --seq-len 32768 --wayfinder-mode permute \
  --window 64 --landmark-stride 64
```

## Architecture

```
hcsa/
├── model.py                   GPT with configurable attention (dense / hcsa)
├── attention_dense.py         Dense causal attention baseline
├── attention_hcsa.py          Sparse attention via neighbor-index gathering
├── cycles.py                  Cycle construction (random, greedy, online)
├── graph_strategies.py        Strategy protocol and implementations
├── train.py                   Training with cosine annealing + mixed precision
├── generate.py                Text generation from checkpoints
├── graph/
│   └── abi.py                 Graph ABI — backend-agnostic neighbor index format
├── topology/
│   └── core.py                First-class topology runtime (construct/save/load/rewire)
├── mlx/                       MLX backend (Apple Silicon)
│   ├── model.py               GPTMLX with dense / wayfinder_sparse / wayfinder_permute
│   └── attention.py           MLX attention implementations
├── torch/                     PyTorch backend (CUDA + CPU)
│   ├── model.py               GPTTorch with same attention modes
│   ├── attention_wayfinder_sparse.py   Sparse-row gather reference path
│   └── attention_wayfinder_permute.py  Permute-window fast path
├── compiler/                  Graph compiler (.wf spec → IR → cache artifacts)
│   ├── sexp.py                S-expression parser
│   ├── graph_ir.py            Intermediate representation
│   └── passes/                Compiler passes (validate, normalize, lower, emit)
└── integrations/
    └── qwen_mlx.py            Qwen3 attention swap for MLX models

scripts/                       Benchmarks, training, ablation sweeps
tests/                         Test suite (tests/pytorch/, tests/mlx/)
configs/                       Experiment configs and graph specs (.wf files)
docs/                          Technical documentation and reports
```

### Key Design Boundaries

- The **Graph ABI** (`WayfinderGraphABI`) is the bridge between backends. Strategies produce cycles on CPU, the ABI wraps them as NumPy arrays, and each backend converts to its native tensor format.
- The **Topology runtime** (`hcsa.topology.Topology`) is now a first-class graph owner. Attention modules can consume injected topology graphs (`topology_graph=...`) or use internal cache-backed construction.
- The **permute fast path** reorders Q/K/V into cycle order so attention becomes a contiguous local window operation — significantly faster but requires `cycle_perms` in the ABI metadata.
- The **sparse gather path** works with arbitrary neighbor lists and serves as the correctness reference.
- The **graph compiler** takes `.wf` spec files through IR passes to produce cached neighbor-index artifacts, avoiding redundant graph construction at runtime.

## Related Work

| Method | Approach | Graph Guarantee |
|---|---|---|
| **HCSA** (this work) | Hamiltonian cycle backbone + local window | Every token reachable via cycle; O(1) long-range edges per token |
| NSA (DeepSeek, 2025) | Hardware-aligned sparse attention with token compression | None |
| XAttention (2025) | Optimal sparse attention via KV cache optimization | None |
| DHSA (2025) | Dynamic hierarchical sparse attention | None |
| Longformer (2020) | Sliding window + global tokens | Global tokens only |
| BigBird (2020) | Window + global + random edges | Random edges, no cycle guarantee |

The key distinction is that Hamiltonian cycles provide a principled, mathematically grounded sparsity pattern where global reachability is guaranteed by construction rather than achieved heuristically.

## References

- Draganić, N., Kim, J., Lee, H., Munhá Correia, D., Pavez-Signé, M., & Sudakov, B. (2025). Hamilton cycles in pseudorandom graphs. [arXiv:2507.22807](https://arxiv.org/abs/2507.22807)
- Draganić, N., Montgomery, R., Munhá Correia, D., Pokrovskiy, A., & Sudakov, B. (2024). Hamiltonicity of expanders: optimal bounds and applications. [arXiv:2405.18875](https://arxiv.org/abs/2405.18875)

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
