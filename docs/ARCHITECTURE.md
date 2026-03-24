# Architecture

This document is contributor-facing and focuses on how Wayfinder/HCSA is structured and why the current implementation behaves the way it does.

Public naming note: the GitHub landing page now uses `Butterfly` / `BNA` as the main project name. This document keeps the older `Wayfinder` / `HCSA` terminology because much of the implementation and benchmark history still uses it.

## Graph ABI
Source: `bna/graph/abi.py`

Wayfinder uses a backend-agnostic graph ABI with two key tensors:
- `neigh_idx`: padded `int32` adjacency list with `-1` as padding. Shape `[T, D]` or `[H, T, D]`.
- `edge_type`: `uint8` labels in `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`.

This ABI is shared across integration and attention-dispatch code paths so the topology contract stays stable across PyTorch and MLX.

## Definition And Cost Model
Let positions be `0..T-1`. Build a Hamiltonian cycle `C` over these positions.

For token `i`, candidate neighbors are:
1. `W` most recent predecessors (causal window)
2. cycle neighbors of `i` in `C`
3. optional landmarks at stride `s`

Apply causal mask `j < i` to obtain directed sparse neighborhoods.

Cost model (`d` = head/embedding dim):
- Dense causal: per-token fan-in `T-1`, total work `O(T^2 d)`
- HCSA: bounded fan-in `D <= W + 2 + T/s` (often closer to `W + 1 + T/s` after masking), total work `O(T D d)`

Example at `T=4096`, `W=64`, `s=64`:
- HCSA candidate fan-in `D=130`
- Dense fan-in `4095`
- approximately `31x` fewer edges per token

## How It's Fast
Key implementation mechanisms (current tree):

1. Vectorized all-head dispatch (`bna/mlx/fused_attention.py`)
- processes all query heads together with fused SDPA kernels
- avoids per-head Python loops and unnecessary eval barriers
- supports contiguous-window path and gather path depending on graph/data alignment

2. Perms-only graph construction (`bna/topology/core.py`)
- `Topology.construct_perms_only()` builds cycle permutation views without always materializing full neighbor-index graphs
- reduces topology-build overhead on permute-centric paths

3. Exact graph horizon behavior in integrations (`bna/integrations/glm_mlx.py`)
- graph horizons align with KV cache sizing decisions
- helps active-row path selection avoid unnecessary random gather behavior in key cases

## Project Map
| Path | Purpose |
|---|---|
| `bna/attention_hcsa.py`, `bna/model.py` | Core sparse attention + reference GPT |
| `bna/cycles.py`, `bna/graph_strategies.py` | Cycle construction + strategy wrappers |
| `bna/graph/abi.py` | Graph ABI (neighbor indices + edge typing) |
| `bna/graph/analysis.py` | Diagnostics: spectral gap, random-walk mixing, resilience |
| `bna/topology/core.py` | Topology runtime and perms-only construction |
| `bna/compiler/`, `configs/graph_specs/*.wf` | Graph-spec compiler + cache artifacts |
| `bna/mlx/attention.py` | MLX dispatch paths (dense, gather, permute-window) |
| `bna/mlx/fused_attention.py` | Vectorized fused dispatch paths |
| `bna/integrations/` | Model integrations (GLM, Qwen, GPT-2) |
| `bna/torch/` | PyTorch/CUDA backend |
| `scripts/` | Benchmarks, training, ablations, diagnostics |
| `tests/` | Correctness and regression tests |
| `benchmarks/`, `results/` | Measurement artifacts |

## Notes for contributors

- Keep benchmark execution sequential for reproducibility and memory safety.
- Keep public claims tied to measured evidence in `docs/FIRST_RELEASE.md`.
- See `CONTRIBUTING.md` for PR expectations.
