# HCSA: Hamiltonian Cycle Sparse Attention

A research framework for **sparse causal attention** in language models using **Hamiltonian cycles** as the attention backbone. HCSA replaces dense quadratic self-attention with a sparse graph induced by Hamiltonian cycles over token positions, providing O(T) attention with graph-theoretic connectivity guarantees.

## Results

Measured on Apple M-series silicon (MLX backend). Dense baseline uses standard causal self-attention.

### Throughput (tokens/sec, attention path)

| Sequence Length | Dense | HCSA (permute) | Ratio |
|---:|---:|---:|---:|
| 256 | 1,254,902 | 487,648 | 0.39x |
| 512 | 2,131,482 | 526,134 | 0.25x |
| 1,024 | 1,173,527 | 546,820 | 0.47x |
| 2,048 | 480,136 | 570,452 | **1.19x** |
| 4,096 | 241,770 | 592,406 | **2.45x** |

### Peak memory (attention path)

| Sequence Length | Dense | HCSA (permute) | Reduction |
|---:|---:|---:|---:|
| 2,048 | 547.5 MB | 343.2 MB | 37% |
| 4,096 | 2.1 GB | 710.1 MB | 67% |

### Quality (TinyShakespeare, character-level)

| Setting | Dense val ppl | HCSA val ppl | Gap |
|---|---:|---:|---:|
| 200-step baseline | 28.13 | 31.66 | +3.53 |
| 200-step + edge-bias + window-drop | 28.06 | 29.15 | +1.09 |
| 1k-step scheduled | 70.20 | 22.92 | -47.29 |

At 1k steps with scheduled cycle-push training, the perplexity gap closes to < +1.0 by step 100 and becomes negative thereafter. Peak cycle utilization reaches 32.87%, indicating the model learns to route information through Hamiltonian cycle edges.

Full benchmark data: [`results/benchmarks/`](results/benchmarks/) | Training runs: [`runs/`](runs/)

## Background

A Hamiltonian cycle visits every vertex in a graph exactly once and returns to the start. HCSA uses one or more such cycles over token positions as an undirected attention backbone, providing each token with O(1) long-range connections while guaranteeing that every position is reachable.

The theoretical foundation draws on recent results showing that pseudorandom graphs admit approximate Hamiltonian decompositions (Draganić et al., 2025). This provides a mathematical basis for using Hamiltonian cycles as sparse connectivity structures: they are guaranteed to exist in sufficiently well-connected graphs and can tile the edge set near-optimally.

### How It Works

**Standard attention** computes scores over all T previous tokens per position, requiring O(T^2) work.

**HCSA** restricts each token's attention neighborhood to:

1. **Cycle neighbors** — the two adjacent nodes in a Hamiltonian cycle (provides long-range shortcuts)
2. **Local causal window** — the w preceding tokens (provides local context)
3. **Landmarks** (optional) — every k-th token (provides global anchors)
4. **Self** — always included

Causality is enforced by masking any neighbor j > i. The total neighborhood size per token is O(w + k), independent of T.

**The permute-window fast path** reorders Q, K, V into cycle order so that cycle-neighbor attention becomes a contiguous local window operation. This avoids gather/scatter overhead and enables throughput exceeding dense attention at long sequences, since the effective attention window is constant regardless of T.

### Cycle Strategies

| Strategy | Complexity | Description |
|---|---|---|
| `random` | O(T) | Random permutation per head. No data dependence. |
| `greedy` | O(T^2) | Nearest-neighbor heuristic using learned routing similarity s(i,j) = (r_i . r_j) / sqrt(d_r) |
| `online_insertion` | O(T) per step | Incremental cycle maintenance as tokens arrive |

## Quickstart

```bash
git clone https://github.com/Hmbown/hcsa.git && cd hcsa
pip install -e ".[dev]"

# PyTorch: train dense vs HCSA
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn dense --steps 200
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn hcsa --cycle random --window 32 --landmark-stride 32 --steps 200
```

For the MLX backend (Apple Silicon):
```bash
pip install -e ".[mlx]"

# Reproduce scaling benchmark
python scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32 \
  --warmup 2 --iters 4

# Reproduce 1k-step quality run with cycle-push training
python scripts/run_mlx_experiment_tiny_long.py \
  --steps 1000 --lr 1e-4 --eval-every 100 --checkpoint-every 200 \
  --batch-size 8 --seq-len 128 --layers 2 --heads 4 --embd 128 \
  --window 2 --landmark-stride 0 --wayfinder-attn wayfinder_sparse \
  --window-drop-max 0.95 --bias-cycle-max 0.8 --bias-window-min -0.2 \
  --bias-landmark-max 0.0 --reliance-reg-coeff 0.05 \
  --reliance-cycle-min 0.20 --reliance-window-max 0.60
```

## Repository Structure

```
hcsa/                   Core library
  attention_dense.py      Dense causal attention (baseline)
  attention_hcsa.py       Sparse attention via neighbor-index gathering
  cycles.py               Cycle construction (random, greedy, online insertion)
  model.py                GPT architecture with configurable attention
  graph_strategies.py     Strategy protocol and implementations
  train.py                Training with cosine annealing + mixed precision
  generate.py             Text generation from checkpoints
  graph/                  Graph ABI (backend-agnostic neighbor index format)
  mlx/                    MLX backend (Apple Silicon)
  compiler/               Graph compiler (spec -> IR -> cache artifacts)
scripts/                Benchmarks, training scripts, ablations, visualization
tests/                  Unified test suite (tests/pytorch/, tests/mlx/)
configs/                Experiment configurations and graph specs
results/                Benchmark outputs
docs/                   Technical documentation
```

## Related Work

HCSA differs from recent sparse and efficient attention methods in its use of graph-theoretic structure with formal connectivity guarantees:

| Method | Approach | Graph Guarantee |
|---|---|---|
| **HCSA** (this work) | Hamiltonian cycle backbone + local window | Every token reachable via cycle; O(1) long-range edges per token |
| NSA (DeepSeek, 2025) | Hardware-aligned sparse attention with token compression | None |
| XAttention (2025) | Optimal sparse attention via KV cache optimization | None |
| DHSA (2025) | Dynamic hierarchical sparse attention | None |
| SPLA (2025) | Sparse low-rank attention | None |
| PBS-Attn (2025) | Paged block sparse attention | None |
| Longformer (2020) | Sliding window + global tokens | Global tokens only |
| BigBird (2020) | Window + global + random edges | Random edges, no cycle guarantee |

The key distinction is that Hamiltonian cycles provide a principled, mathematically grounded sparsity pattern where global reachability is guaranteed by construction rather than achieved heuristically.

## Theoretical References

- Draganić, N., Kim, J., Lee, H., Munhá Correia, D., Pavez-Signé, M., & Sudakov, B. (2025). Hamilton cycles in pseudorandom graphs: resilience and approximate decompositions. [arXiv:2507.22807](https://arxiv.org/abs/2507.22807)
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
