# Wayfinder (HCSA)

**Hamiltonian Cycle Sparse Attention (HCSA)**: sparse causal attention where each token attends to an explicit **graph neighborhood** (Hamiltonian-cycle backbone + local causal window + optional landmarks). Core representation is a backend-agnostic **Graph ABI**; backends include **PyTorch** and **MLX**.

**Graph ABI** (`hcsa/graph/abi.py`):
- `neigh_idx`: padded `int32` neighbor indices (`-1` = PAD), shape `[T,D]` or `[H,T,D]`
- `edge_type`: `uint8` edge labels (`PAD/CYCLE/WINDOW/LANDMARK/REWIRE`)

## Token interactions: dense vs HCSA

Both panels show which tokens **t7** attends to in a single layer.

```mermaid
%%{init: {'theme': 'neutral', 'flowchart': {'curve': 'basis'}}}%%
flowchart TB
    subgraph D["Dense causal (t7)"]
        d7((t7)):::q
        d7 --> d6((t6))
        d7 --> d5((t5))
        d7 --> d4((t4))
        d7 --> d3((t3))
        d7 --> d2((t2))
        d7 --> d1((t1))
        d7 --> d0((t0))
    end

    subgraph S["HCSA (t7, W=2, stride=4)"]
        s7((t7)):::q
        s7 --> s6((t6)):::w
        s7 --> s5((t5)):::w
        s7 -.-> s3((t3)):::c
        s7 -.-> s1((t1)):::c
        s7 ==> s4((t4)):::l
        s7 ==> s0((t0)):::l
        s7 ~~~ s2((t2)):::skip
    end

    classDef q fill:#e63946,stroke:#fff,color:#fff,stroke-width:3px
    classDef w fill:#457b9d,stroke:#1d3557,color:#fff
    classDef c fill:#e76f51,stroke:#9c4835,color:#fff
    classDef l fill:#2a9d8f,stroke:#1b6b62,color:#fff
    classDef skip fill:#f1faee,stroke:#ccc,color:#bbb,stroke-dasharray:5 5
```

**HCSA edge types** (self-attention always included, omitted from diagram):

| Node color | Arrow | Type | Rule |
|---|---|---|---|
| red | — | query | the token computing attention |
| blue | solid `-->` | window | W nearest causal neighbors |
| orange | dashed `-.->` | cycle | Hamiltonian-cycle neighbor(s) with j < i |
| green | thick `==>` | landmark | every stride-th position with j < i |
| gray dashed | — | *not reached* | outside neighborhood this layer |

Dense: fan-in = **T - 1** per token, **O(T^2)** total edges, **O(T^2 d)** attention cost.
HCSA: fan-in = **W + 2 + T/s** per token (degree D), **O(T D)** total edges, **O(T D d)** attention cost.
At T = 4096, W = 64, stride = 64: D = 130 vs 4095 — a **31x** reduction in edges per token.

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

## What the attention graph looks like

Dense attention (left) vs HCSA connectivity (right) at T=64, W=8, stride=16:

![Dense vs HCSA attention matrices](docs/assets/attention_comparison.png)

HCSA graph on 32 tokens (circle layout — blue = cycle, green = window, red = landmark; only causal edges shown):

![HCSA graph circle layout](docs/assets/hcsa_graph_circle.png)

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
| `benchmarks/` | Experiment results + benchmark data |

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

See: `hcsa/discover/targets.py`, `hcsa/discover/session.py`.

## References

- Hamilton cycles in pseudorandom graphs (resilience/decompositions): https://arxiv.org/abs/2507.22807
- Exphormer (Sparse Transformers for Graphs): https://arxiv.org/abs/2303.06147
- TTT-Discover (test-time tuning/training): https://arxiv.org/abs/2601.16175
- ZMLX (`zmlx.discover`): https://github.com/Hmbown/ZMLX

## License

MIT. See [`LICENSE`](LICENSE).
