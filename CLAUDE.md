# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HCSA (Hamiltonian Cycle Sparse Attention) — a research framework for **sparse attention in language models using Hamiltonian cycles**. Replaces dense causal self-attention with sparse attention graphs induced by Hamiltonian cycles over token positions.

Two backends:
- **PyTorch** (`hcsa/`) — core attention, cycles, model, training
- **MLX** (`hcsa/mlx/`) — Apple Silicon optimized, uses a unified Graph ABI abstraction

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"          # pytest, ruff, mypy, hypothesis, pytest-benchmark
pip install -e ".[mlx]"         # MLX backend

# Run all tests
pytest

# Run only PyTorch tests
pytest tests/pytorch/

# Run only MLX tests
pytest tests/mlx/

# Run a single test file or test
pytest tests/pytorch/test_cycles.py
pytest tests/pytorch/test_causality.py::test_causal_mask -v

# Lint
ruff check hcsa/ tests/
ruff format --check hcsa/ tests/

# Type check
mypy hcsa/

# Train (PyTorch)
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn dense --steps 5
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char --attn hcsa --cycle greedy --window 64 --landmark-stride 64 --steps 5

# Generate text
python -m hcsa.generate --ckpt runs/<run>/ckpt.pt --prompt "To be" --max-new 50

# Benchmark
python scripts/bench.py --device auto --seq-lens 128 256 512 1024
python scripts/bench_mlx_wayfinder_scale.py   # MLX scaling benchmark
```

## Architecture

### Core Concept: Sparse Attention via Hamiltonian Cycles

Each token `i` attends to a small neighborhood instead of all previous tokens:
1. **Cycle neighbors** — two adjacent nodes in a Hamiltonian cycle (undirected edges, causality enforced by masking `j > i`)
2. **Local causal window** — `{max(0, i-w), ..., i-1}` (always included)
3. **Landmarks** (optional) — every k-th token: `{j | j % stride == 0 and j < i}`
4. **Self** — always included

Implementation builds a padded index tensor `neigh_idx: [T, D]` (with `-1` padding), gathers K/V, computes scores only for gathered entries, masks padding + causal violations, then softmax over the neighbor dimension.

### Cycle Strategies (`hcsa/cycles.py`)

- `random_cycle` — O(T) random permutation per head
- `greedy_cycle` — O(T²) nearest-neighbor TSP-like heuristic using routing similarity `s(i,j) = (r_i · r_j) / sqrt(d_r)`
- `online_insertion_cycle` — O(T) incremental insertion maintaining cycle

### PyTorch Core (`hcsa/`)

- `model.py` — GPT architecture with `GPTConfig`, `Block`, `MLP`; attention type is configurable (`dense` or `hcsa`)
- `attention_dense.py` — dense causal baseline using `F.scaled_dot_product_attention`
- `attention_hcsa.py` — sparse attention with gather-based neighbor index computation
- `graph_strategies.py` — `GraphStrategy` protocol + concrete implementations wrapping cycle functions
- `permute_attention.py` — optimized permutation-based attention path
- `train.py` — CLI entry point with cosine annealing + mixed precision
- `generate.py` — text generation from checkpoints
- `data.py` / `data_mmap.py` — data loading (TinyShakespeare, HuggingFace datasets)
- `tokenizers.py` — character and BPE tokenizers

### MLX Backend (`hcsa/mlx/`)

- `model.py` — `GPTMLX` with three attention modes: `dense`, `wayfinder_sparse` (reference gather path), `wayfinder_permute` (fast contiguous path)
- `attention.py` — `dense_causal_attention()`, `sparse_gather_attention()`, `permute_window_attention()`
- `graph_abi.py` — MLX-specific graph conversions
- `metrics.py` — graph metrics (shortcut_rate, reachability_proxy, edge utilization)

### Graph ABI (`hcsa/graph/`)

- `abi.py` — `WayfinderGraphABI` dataclass: language-agnostic neighbor index + edge type tensors. `EdgeType` enum: `PAD`, `CYCLE`, `WINDOW`, `LANDMARK`, `REWIRE`. Shapes: `neigh_idx [T, D]` (single-head) or `[H, T, D]` (multi-head)

### Compiler (`hcsa/compiler/`)

- `sexp.py` — S-expression parser for `.wf` graph specs
- `graph_ir.py` — Intermediate representation
- `passes/` — Compiler passes (validate, normalize, lower, specialize, cache-key, emit)

### Key Design Boundaries

- The Graph ABI is the bridge between backends — PyTorch strategies produce cycles, ABI wraps them as NumPy arrays, MLX converts to `mx.array`
- The permute fast path (`wayfinder_permute`) reorders Q/K/V into cycle order so attention becomes a contiguous local window operation — significantly faster but requires `cycle_perms` in ABI meta
- The sparse gather path (`wayfinder_sparse`) works with arbitrary neighbor lists and serves as the correctness reference
- Graph construction currently runs on CPU (Python) and is timed separately (`graph_build_ms`)

## Testing Notes

- All tests under `tests/` with subdirectories `tests/pytorch/` and `tests/mlx/`
- Shared conftest at `tests/conftest.py` adds repo root to `sys.path`
- PyTorch conftest sets `torch.set_num_threads(1)` for CPU determinism
- Ruff line-length is 100 characters

## Configuration

- Python >= 3.10, package at repo root under `hcsa/`
- `--device auto` picks `cuda` > `mps` > `cpu`
- Checkpoints saved to `runs/<timestamp>/ckpt.pt` with `config.json`
- Graph specs in `configs/graph_specs/`
