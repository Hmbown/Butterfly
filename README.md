# Wayfinder (HCSA)

**Hamiltonian Cycle Sparse Attention (HCSA)**: sparse causal attention where each token attends to an explicit **graph neighborhood** (Hamiltonian-cycle backbone + local causal window + optional landmarks). Core representation is a backend-agnostic **Graph ABI**; backends include **PyTorch** and **MLX**.

**Graph ABI** (`hcsa/graph/abi.py`):
- `neigh_idx`: padded `int32` neighbor indices (`-1` = PAD), shape `[T,D]` or `[H,T,D]`
- `edge_type`: `uint8` edge labels (`PAD/CYCLE/WINDOW/LANDMARK/REWIRE`)

## Token interactions: dense vs sparse

```mermaid
%%{init: {'flowchart': {'curve': 'basis'}}}%%
flowchart LR
  subgraph D["Dense causal (query t7)"]
    direction LR
    d0((t0))
    d1((t1))
    d2((t2))
    d3((t3))
    d4((t4))
    d5((t5))
    d6((t6))
    d7((t7))

    d7 --> d6
    d7 --> d5
    d7 --> d4
    d7 --> d3
    d7 --> d2
    d7 --> d1
    d7 --> d0
  end

  subgraph S["HCSA neighborhood (W=2 + a few long edges)"]
    direction LR
    s0((t0))
    s1((t1))
    s2((t2))
    s3((t3))
    s4((t4))
    s5((t5))
    s6((t6))
    s7((t7))

    %% local window
    s7 --> s6
    s7 --> s5
    s6 --> s5
    s6 --> s4

    %% sparse long edges (examples)
    s7 -. cycle .-> s2
    s7 -. landmark .-> s4
    s6 -. cycle .-> s1
    s4 -. landmark .-> s0
  end
```

Dense fan-in per query token ≈ **O(T)** ⇒ **O(T²)** edges total.
HCSA fan-in per query token ≈ **W + 2 (cycle) + (T/s) (landmarks)** ⇒ **O(T·W)** (plus landmarks).
Research question: how these low-degree token graphs affect reachability/mixing *in practice* (see `hcsa/graph/analysis.py`).

## Install

```bash
git clone <this-repo> && cd <this-repo>
pip install -e ".[dev]"

# Optional
pip install -e ".[mlx]"   # Apple Silicon backend
pip install -e ".[viz]"   # matplotlib/networkx diagnostics
```

## Run the tests

```bash
pytest
```

## Run something small

Tiny train (PyTorch core):

```bash
python -m hcsa.train --data data/tinyshakespeare.txt --tokenizer char \
  --attn hcsa --cycle random --window 32 --landmark-stride 32 --steps 200
```

MLX scaling benchmark:

```bash
python scripts/bench_mlx_wayfinder_scale.py \
  --seq-lens 256 512 1024 2048 4096 \
  --batch 2 --heads 4 --embd 128 \
  --window 32 --landmark-stride 32
```

## Graph properties → attention patterns

![Attention weights on sparse edges](docs/assets/attention_on_edges_heatmap.png)

![Highway distance histogram](docs/assets/highway_distance_hist.png)

## Where to look (research map)

| Path | Purpose |
|---|---|
| `hcsa/attention_hcsa.py`, `hcsa/model.py` | Core sparse attention + reference GPT |
| `hcsa/cycles.py`, `hcsa/graph_strategies.py` | Cycle construction + strategy wrappers |
| `hcsa/graph/abi.py` | Graph ABI (neighbor indices + edge typing) |
| `hcsa/graph/analysis.py` | Spectral gap / mixing proxy / resilience / regularity / coverage |
| `hcsa/topology/core.py` | Topology runtime (construct/save/load/rewire) |
| `hcsa/compiler/` + `configs/graph_specs/*.wf` | Graph-spec compiler + cache artifacts |
| `hcsa/mlx/`, `hcsa/torch/` | Backend implementations |
| `scripts/wayc.py` | CLI: compile/validate/bench + discovery setup |
| `tests/` | Correctness + diagnostics coverage |
| `benchmarks/`, `notes/LAB_NOTEBOOK.md` | Experiments + results |

## Kernel auto-find (optional; setup here)

This repo provides **discovery target specs + setup scaffolding** (no model load / no inference). Kernel search/export uses `zmlx.discover` from **ZMLX**: https://github.com/Hmbown/ZMLX.

```bash
# List discovery targets (K1–K5)
python scripts/wayc.py discover-targets --targets all

# Generate session stubs + seed kernels (setup-only)
python scripts/wayc.py discover-setup \
  --targets all \
  --zmlx-root /path/to/ZMLX \
  --sessions-root discover_sessions \
  --kernel-out-root hcsa/mlx/kernels/metal \
  --strict
```

See: `docs/discover_setup.md`, `hcsa/discover/targets.py`, `hcsa/discover/session.py`.

## References

- Hamilton cycles in pseudorandom graphs (resilience/decompositions): https://arxiv.org/abs/2507.22807
- Exphormer (Sparse Transformers for Graphs): https://arxiv.org/abs/2303.06147
- TTT-Discover (test-time tuning/training): https://arxiv.org/abs/2601.16175
- ZMLX (`zmlx.discover`): https://github.com/Hmbown/ZMLX

## License

MIT. See [`LICENSE`](LICENSE).
