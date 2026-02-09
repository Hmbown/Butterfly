# Wayfinder (HCSA)

Wayfinder implements **Hamiltonian Cycle Sparse Attention (HCSA)**: causal self-attention where each token attends to a bounded neighborhood defined by a graph over positions `0..T-1` (local window + Hamiltonian cycle + optional landmarks/rewires). The Hamiltonian cycle is a degree-2 backbone; any mixing comes from the union of edges (and optional multiple cycles/rewires) plus composition across layers. Execution targets **PyTorch** and **MLX** via a backend-agnostic Graph ABI.

**Graph ABI** (`hcsa/graph/abi.py`):
- `neigh_idx`: padded `int32` adjacency list; `-1` denotes padding. Shape `[T, D]` or `[H, T, D]`
- `edge_type`: `uint8` edge labels in `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`

## Definition And Cost Model

Let the sequence positions be `0..T-1`. Construct a Hamiltonian cycle `C` over these positions. For token `i`, candidate neighbors are:

1. `W` most recent predecessors (local causal window)
2. the two neighbors of `i` in `C` (cycle backbone)
3. landmark positions at stride `s`

Apply the causal mask `j < i`, yielding a directed sparse neighborhood for token `i`.

**Cost model** (`d` = head/embedding dim, `s` = landmark stride):

Dense causal: per-token fan-in is `T-1`, so total edges are `O(T^2)` and attention work is `O(T^2 d)`.
HCSA: per-token fan-in is bounded by `D <= W + 2 + T/s`; after causal masking on the cycle, typical average is closer to `D ~= W + 1 + T/s`. Total edges are `O(TD)` and attention work is `O(T D d)`.

At `T=4096`, `W=64`, `s=64`: `D=130` vs `4095`, a **31x** reduction in edges per token.

## Install

```bash
git clone <this-repo> && cd <this-repo>
pip install -e ".[dev]"

# Optional
pip install -e ".[mlx]"   # Apple Silicon backend
pip install -e ".[viz]"   # matplotlib/networkx diagnostics
```

## Quickstart

Run tests:

```bash
pytest
```

Tiny train (PyTorch core):

```bash
python3 -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char \
  --attn hcsa --cycle random --window 32 --landmark-stride 32 --steps 200
```

MLX scaling benchmark:

```bash
python3 scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32
```

## Workflow (Steps)

1. Propose: write a hypothesis in `notes/LAB_NOTEBOOK.md` (required for benchmarks).
2. Change: edit code/config (often `configs/graph_specs/*.wf`).
3. Benchmark: run a targeted script under `scripts/` (or `python3 scripts/wayc.py bench` for a tiny MLX microbench).
4. Record: append results to `notes/experiments.ndjson`.
5. Decide: keep/revert/follow-up in `notes/LAB_NOTEBOOK.md`.

Protocol details: see `AGENTS.md`.

## Benchmarks

All numbers: Apple M4 Max, MLX backend, 4-bit weights, `W=64`, chunk=`4096`.

### GLM-4.7-Flash-4bit - full model (47 layers, MoE)

End-to-end chunked prefill through all 47 layers of a production MoE model (9B total, ~4B active). HCSA is **faster than dense at every sequence length** tested.

| T | Dense | HCSA | Speedup | Dense memory | HCSA memory |
|------:|----------:|----------:|--------:|--------:|--------:|
| 8,192 | 254 tok/s | **293 tok/s** | **1.15x** | 20.7 GB | 20.1 GB |
| 16,384 | 365 tok/s | **852 tok/s** | **2.34x** | 22.4 GB | 21.8 GB |
| 32,768 | 227 tok/s | **666 tok/s** | **2.93x** | 26.0 GB | 25.4 GB |

Dense scales O(T^2); HCSA scales O(T*W). The advantage grows with sequence length.

### Qwen3-4B-4bit - isolated attention (single layer)

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 241,600 | 309,100 | **1.28x** |
| 8,192 | 176,000 | 313,600 | **1.78x** |
| 16,384 | 113,700 | 314,800 | **2.77x** |

### Qwen3-4B-4bit - full transformer block (attention + MLP + norms)

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 85,200 | 92,000 | **1.08x** |
| 8,192 | 76,300 | 94,000 | **1.23x** |
| 16,384 | 61,300 | 93,700 | **1.53x** |

### GPT-2 (MLX) - isolated attention

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 4,096 | 790,000 | 1,020,000 | **1.30x** |
| 8,192 | 500,000 | 1,050,000 | **2.12x** |
| 16,384 | 280,000 | 1,090,000 | **3.87x** |

Raw data: `results/`, `benchmarks/`

## How It's Fast

Three optimizations combine to make HCSA sparse attention faster than dense on real models:

1. **Vectorized all-head dispatch** (`hcsa/mlx/fused_attention.py`): Processes all query heads simultaneously via flat-index arithmetic + MLX's fused SDPA kernel, eliminating per-head Python loops and eval barriers. Two code paths: contiguous-window (when graph matches data) and flat-index gather (oversized graphs).

2. **Fast perms-only graph build** (`hcsa/topology/core.py`): `Topology.construct_perms_only()` generates only cycle permutations in O(T) time, skipping the O(T*D) neighbor-index construction that is unused by the permute attention path. **1904x faster** at T=32,768 (24s to 12ms).

3. **Exact graph horizon** (`hcsa/integrations/glm_mlx.py`): Builds graphs at exactly the KV cache size (Tg=Tk) instead of rounding up to a power-of-two horizon. This enables the contiguous-window active-row path where pre-permuted K/V can be sliced without random-access gathers.

## Visuals

Dense attention (left) vs HCSA connectivity (right) at `T=64`, `W=8`, `s=16`:

![Dense vs HCSA attention matrices](docs/assets/attention_comparison.png)

HCSA graph on 32 tokens (circle layout; only causal edges shown):

![HCSA graph circle layout](docs/assets/hcsa_graph_circle.png)

## Project Map

| Path | Purpose |
|---|---|
| `hcsa/attention_hcsa.py`, `hcsa/model.py` | Core sparse attention + reference GPT |
| `hcsa/cycles.py`, `hcsa/graph_strategies.py` | Cycle construction + strategy wrappers |
| `hcsa/graph/abi.py` | Graph ABI (neighbor indices + edge typing) |
| `hcsa/graph/analysis.py` | Empirical diagnostics: spectral gap, random-walk mixing, resilience |
| `hcsa/topology/core.py` | Topology runtime (construct/save/load/rewire/`construct_perms_only`) |
| `hcsa/compiler/` + `configs/graph_specs/*.wf` | Graph-spec compiler + cache artifacts |
| `hcsa/mlx/attention.py` | MLX attention dispatch (dense, sparse-gather, permute-window) |
| `hcsa/mlx/fused_attention.py` | Vectorized all-head fused dispatch (full-prefill + active-row) |
| `hcsa/integrations/` | Model integrations (Qwen3, GLM-4, GPT-2) |
| `hcsa/torch/` | PyTorch/CUDA backend |
| `scripts/` | Benchmarks, training, ablations, visualization |
| `tests/` | 240 tests covering correctness + diagnostics |
| `benchmarks/`, `results/` | Experiment results + benchmark data |

## Research Questions

Can sparse attention learn like a slime mold - start with a structured backbone and discover which edges carry information at every scale?

Practical version:
- Start from an overcomplete candidate graph (window + landmarks + `k` cycles/rewires).
- Learn a degree-budgeted subgraph (per-token/head top-`D` edges) via gated edges or periodic prune/rewire using usage/gradient credit.
- Measure: (1) long-context quality, (2) layer depth needed for near-full-prefix receptive fields, (3) end-to-end throughput and peak memory at fixed `D`.

Related structural question: does multiscale cycle richness of the undirected skeleton predict the depth needed for full-prefix receptive fields under causal composition, beyond spectral gap alone?

## References

Sparse/structured attention and graph-based sparsity:
- BigBird: https://arxiv.org/abs/2007.14062
- Exphormer: https://arxiv.org/abs/2303.06147
- FlashInfer: https://arxiv.org/abs/2501.01005
- Flex Attention: https://arxiv.org/abs/2412.05496

## License

MIT. See `LICENSE`.
