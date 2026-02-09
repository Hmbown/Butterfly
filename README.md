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

All numbers: Apple M-series silicon, MLX backend, 4-bit weights, `W=64`, chunk=`4096`, circular windowing.

### Qwen3-1.7B-4bit - isolated attention (single layer, no MLP)

Attention kernel only (one layer; no MLP), isolating `O(TD)` vs `O(T^2)` scaling with `D << T` in the sparse regime.

| T | Dense tok/s | HCSA tok/s | Speedup |
|------:|----------:|----------:|--------:|
| 8192 | 174,964 | 227,455 | **1.30x** |
| 16384 | 112,437 | 236,100 | **2.10x** |
| 32768 | 62,733 | 236,420 | **3.77x** |

HCSA stays ~flat in this configuration while dense drops sharply with `T`.

### Qwen3-1.7B-4bit - full transformer block (attention + MLP + norms)

One complete block; `O(T)` terms (MLP/norms) dilute attention-only speedups.

| T | Dense tok/s | HCSA tok/s | Speedup | Memory reduction |
|------:|----------:|----------:|--------:|--------:|
| 8192 | 75,857 | 82,949 | 1.09x | 15.2% |
| 16384 | 60,916 | 84,444 | 1.39x | 15.6% |
| 32768 | 42,847 | 85,466 | **2.00x** | 10.1% |

### GLM-4.7-Flash-4bit - full model (47 layers, MoE)

End-to-end chunked prefill through all 47 layers of a production MoE model (9B total, ~4B active).

| T | Dense tok/s | HCSA tok/s | Dense memory | HCSA memory | Memory reduction |
|------:|----------:|----------:|--------:|--------:|--------:|
| 8192 | 254 | 121 | 19.2 GB | 20.0 GB | - |
| 16384 | 360 | 175 | 20.9 GB | 20.4 GB | 2.4% |
| 32768 | 177 | 148 | 24.2 GB | 20.8 GB | **22.9%** |

At these lengths HCSA is slower end-to-end because attention is a smaller fraction of total work; the main win in this table is memory at 32K (22.9% reduction in this run).

Raw data: `benchmarks/`

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
| `hcsa/topology/core.py` | Topology runtime (construct/save/load/rewire) |
| `hcsa/compiler/` + `configs/graph_specs/*.wf` | Graph-spec compiler + cache artifacts |
| `hcsa/mlx/`, `hcsa/torch/` | Backend implementations |
| `scripts/wayc.py` | CLI: validate/compile/bench + discovery setup |
| `tests/` | Correctness + diagnostics coverage |
| `benchmarks/` | Experiment results + benchmark data |

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
