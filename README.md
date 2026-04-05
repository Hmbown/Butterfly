# Butterfly

Butterfly Network Attention (`bna`) is a training-free sparse-attention runtime for long-context inference. It replaces quadratic attention layers with a structured sparse pattern based on Hamiltonian cycle permutations, achieving linear scaling in sequence length without retraining.

![Dense causal vs BNA block topology](docs/assets/bna_block_topology.png)

## What this repo contains

- A PyTorch package, `bna`, for sparse-attention research and integration work
- CUDA and MLX benchmark scripts for Qwen, GLM, GPT-2, and related paths
- Measured benchmark artifacts under `benchmarks/`, `results/`, and `notes/`
- Older docs and scripts that still use the legacy names `Wayfinder` and `HCSA`

Public naming note: `Butterfly` / `BNA` is the current public project name. `Wayfinder` / `HCSA` are legacy names still present in deeper docs, scripts, benchmark artifact paths, and archived research material.

## Status

| Tier | What to trust | Evidence |
|---|---|---|
| Validated | GLM-4.7-Flash-4bit on MLX at the public stable profile | [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md) |
| Experimental | Qwen 3.5 CUDA block-sparse path and long-context scaling work | `scripts/bench_qwen35_cuda_wayfinder.py`, `benchmarks/cuda/qwen35_wayfinder/` |
| Experimental | Qwen 3.5 MLX / Apple Silicon path | `scripts/bench_qwen_consumer_mlx.py`, `results/benchmarks/` |
| Research / archive | Older Wayfinder/HCSA docs, prompts, and exploratory runs | `docs/`, `notes/`, `archive/` |

If you are new to the project, start from the validated GLM path first. The Qwen work is promising, but it should still be read as active engineering rather than a locked public release.

## How it works

### The problem BNA solves

Qwen 3.5 is a **hybrid linear + quadratic attention** model. Of its 32 layers, 24 use GatedDeltaNet (linear attention, O(T) per layer) and 8 use full grouped-query attention (O(T^2) per layer) at indices 3, 7, 11, 15, 19, 23, 27, 31. The 8 quadratic layers are the only superlinear component in the entire model. At long context, they dominate compute and memory.

BNA replaces **only those 8 quadratic layers** with a structured sparse pattern. The 24 GatedDeltaNet layers are untouched.

### The permute trick

The core mechanism in three steps:

1. **Permute** — Generate a random Hamiltonian cycle (a permutation that visits every token exactly once). Reorder Q, K, V into cycle order so that cycle-adjacent tokens become memory-adjacent.

2. **Window attend** — Apply a standard sliding window of size 2W+1 (W=64, so 129 tokens) in the permuted space. Because cycle neighbors are now contiguous, this captures both local context and long-range connections in a single cache-friendly pass.

3. **Unpermute** — Reorder the output back to the original token order.

The key insight: a local sliding window over a randomly permuted sequence is effectively a **random global sample** of the original sequence. This converts O(T^2) quadratic attention into O(T x 129) linear attention per layer.

### Coverage guarantee

With window W=129 and 8 BNA layers, the receptive field grows exponentially: 1 layer reaches 129 tokens, 2 layers reach ~16K, 3 layers reach ~2.1M. This far exceeds the model's 262K max context, ensuring full information flow without explicit global tokens or handcrafted patterns.

### Complexity

| Component | Stock Qwen 3.5 | Qwen 3.5 + BNA |
|-----------|----------------|-----------------|
| 24 GatedDeltaNet layers | O(T) each | O(T) each (unchanged) |
| 8 quadratic layers | O(T^2) each | O(T x 129) each |
| Total model scaling | Quadratic tail at long context | **Fully linear in T** |

At T=32K, the 8 stock quadratic layers compute 1.07B attention entries each. BNA reduces that to 4.1M — a **260x reduction** per layer.

For contributor-facing implementation details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/BNA_QWEN35_ARCHITECTURE.md](docs/BNA_QWEN35_ARCHITECTURE.md).

## Measured evidence

### Validated public path: GLM on MLX

The clearest in-repo release evidence today is the GLM-4.7-Flash-4bit stable profile documented in [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md).

At `seq_len=8192` and `decode_len=32` on the validated MLX path:

| Mode | E2E | Prefill | Decode tok/s | Peak memory |
|---|---:|---:|---:|---:|
| Dense | 17.15s | 16.36s | 40.58 | 20.66 GB |
| Butterfly | 10.56s | 9.75s | 39.85 | 20.07 GB |
| Delta vs dense | -38.44% | -40.38% | -1.79% | -2.85% |

That is the safest benchmark slice to cite publicly from this tree today.

### Experimental CUDA path: Qwen 3.5 9B

Triton block-sparse path on Qwen 3.5 9B. BNA replaces 8 of 32 layers (the quadratic Qwen3NextAttention layers). The 24 GatedDeltaNet layers are untouched.

| Context | Stock tok/s | BNA tok/s | Top-1 agreement |
|--------:|------------:|----------:|----------------:|
| 4,096   | —           | —         | 99.88%          |
| 8,192   | 1,651       | 1,698     | —               |
| 16,384  | —           | —         | 94.44%          |
| 32,768  | 1,585       | 1,688     | —               |
| 65,536  | 1,475       | 1,724     | —               |
| 98,304  | 1,413       | 1,660     | —               |
| 131,072 | 1,365       | 1,667     | —               |
| 262,144 | 1,257       | 1,712     | —               |

BNA maintains flat throughput as context grows while stock throughput degrades. At 262K, BNA is 36% faster. This path should still be treated as experimental until quality boundaries are documented as tightly as the GLM release.

### Experimental CUDA path: Qwen 3.5 35B A3B FP8

| Context | Stock tok/s | BNA tok/s |
|--------:|------------:|----------:|
| 8,192   | 931         | 954       |
| 32,768  | 1,280       | 1,301     |
| 65,536  | 1,241       | 1,326     |
| 131,072 | 1,131       | 1,331     |
| 163,840 | —           | 1,306     |
| 196,608 | —           | 1,364     |
| 229,376 | —           | 1,233     |

Stock Qwen 3.5 35B runs out of memory above 131K. BNA continues to 229K.

### Experimental Apple Silicon path: Qwen 3.5 9B on M4 Max

MLX permute-window path with K6 fused Metal kernel, `window=64`. 8 of 32 quadratic layers replaced. Model: `mlx-community/Qwen3.5-9B-MLX-4bit`.

| Context | Stock TTFT | BNA TTFT | Stock tok/s | BNA tok/s | Peak memory |
|--------:|-----------:|---------:|------------:|----------:|------------:|
| 2,048   | 71 ms      | 49 ms    | 62.2        | 62.0      | 7.1 GB      |
| 8,192   | 116 ms     | 86 ms    | 57.2        | 58.8      | 9.9 GB      |
| 32,768  | 100 ms     | 99 ms    | 49.6        | 47.1      | 13.7 GB     |
| 65,536  | 160 ms     | 202 ms   | 41.5        | 39.8      | 18.9 GB     |
| 98,304  | 2.0 s      | 1.2 s    | 17.2        | 22.4      | 24.0 GB     |
| 131,072 | 6.9 s      | 7.5 s    | 7.3         | 6.8       | 29.1 GB     |
| 163,840 | 26.8 s     | 21.5 s   | 2.2         | 2.7       | 34.2 GB     |

### Experimental Apple Silicon path: Qwen 3.5 4B on M4 Max

MLX permute-window path, `window=64`, 8 of 32 quadratic layers replaced. Model: `mlx-community/Qwen3.5-4B-MLX-4bit`.

**Single-turn (BNA only — clean stock comparison at these lengths pending):**

| Context | BNA TTFT | BNA e2e | Decode ITL p95 | Peak memory |
|--------:|---------:|--------:|---------------:|------------:|
| 2,048   | 21 ms    | 3.2 s   | 11 ms          | 4.9 GB      |
| 4,096   | 38 ms    | 5.1 s   | 11 ms          | 6.9 GB      |
| 8,192   | 53 ms    | 9.2 s   | 11 ms          | 7.9 GB      |
| 16,384  | 77 ms    | 18.2 s  | 12 ms          | 9.3 GB      |
| 32,768  | 131 ms   | 39.1 s  | 14 ms          | 12.0 GB     |

TTFT scales linearly: 6.2x increase for 16x more tokens. Decode latency is stable across all context lengths.

**Multi-turn context push (BNA only — cumulative context per turn):**

| Turn | Context | BNA TTFT | Decode ITL p95 |
|-----:|--------:|---------:|---------------:|
| 1    | 13K     | 19 ms    | 12 ms          |
| 2    | 26K     | 36 ms    | 13 ms          |
| 4    | 52K     | 70 ms    | 16 ms          |
| 6    | 79K     | 51 ms    | 18 ms          |
| 7    | 92K     | 62 ms    | 19 ms          |
| 8    | 105K    | 76 ms    | 21 ms          |

BNA processed 105K tokens on a 4B model with sub-100ms TTFT — 2.6x the model's original 40K design point. Push to the full 262K max context is in progress.

These MLX paths use the permute-window mechanism for prefill. Decode currently falls back to stock quadratic attention on the 8 replaced layers (`wayfinder_decode_backend = "dense"`). Sparse decode is implemented but not yet benchmarked.

Top-1 agreement in the Qwen 9B experiments is `99.88%` at 4K and `94.44%` at 16K. Perplexity and downstream evaluation are still in progress, so avoid treating these tables as universal quality-parity claims.

## Quick start

### CUDA (NVIDIA GPU, Linux/WSL2)

```bash
git clone https://github.com/Hmbown/Butterfly.git
cd Butterfly
./scripts/bootstrap_wsl2_ubuntu.sh    # WSL2 Ubuntu
# or: ./scripts/bootstrap_cuda_linux.sh  # native Linux
```

Validated public path:

```bash
./scripts/run_public_stable_profile_glm.sh
```

Experimental Qwen CUDA benchmark:

```bash
source ./scripts/cuda_local_env.sh
./.venv-cuda/bin/python scripts/model_catalog.py download qwen35_2b_hf
./scripts/run_qwen35_3080_profile.sh
```

The recommended 10GB consumer-GPU target is `Qwen/Qwen3.5-2B` with `--quantize bnb-4bit`. The `4B` checkpoint is not the default on `RTX 3080 10GB` because long-context KV/cache headroom gets too tight above the mid-context range.

Use the Linux / WSL2 bootstrap path above for consumer NVIDIA boxes. The repo-level bring-up guide is in [docs/CUDA_3080_SETUP.md](docs/CUDA_3080_SETUP.md).

### MLX (Apple Silicon)

```bash
git clone https://github.com/Hmbown/Butterfly.git
cd Butterfly
pip install -e ".[mlx]"
pip install mlx-lm zmlx
```

Environment check:

```bash
python scripts/env_check_mlx.py
```

Experimental Qwen MLX benchmark:

```bash
python scripts/bench_qwen_consumer_mlx.py \
    --model-path mlx-community/Qwen3.5-9B-MLX-4bit \
    --mode wayfinder \
    --seq-lens 2048 8192 32768 \
    --decode-len 256 \
    --repeats 3 \
    --out-dir results/benchmarks/my_run
```

The `--mode dense` flag runs the stock Qwen 3.5 baseline (hybrid GatedDeltaNet + quadratic) for comparison. `--mode butterfly` replaces the 8 quadratic layers with BNA. Add `--skip-quality` to benchmark only throughput.

### Basic checks

```bash
pytest
ruff check bna tests
```

## Repo map

| Path | What it is |
|---|---|
| `bna/` | Core package and backend integrations |
| `scripts/` | Benchmarks, diagnostics, serving helpers, and figure generation |
| `docs/` | Contributor-facing architecture, release evidence, and research notes |
| `benchmarks/`, `results/` | Raw benchmark outputs and summaries |
| `notes/` | Lab notebook, experiment log, handoff prompts, and planning material |
| `archive/` | Older exploratory code and preserved artifacts |

## Where to read next

- [docs/BNA_QWEN35_ARCHITECTURE.md](docs/BNA_QWEN35_ARCHITECTURE.md): how BNA works on Qwen 3.5's hybrid architecture, with diagrams
- [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md): validated benchmark slice and reproduction commands
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): contributor-facing implementation map
- [docs/APPLE_SILICON_SETUP.md](docs/APPLE_SILICON_SETUP.md): Apple Silicon bootstrap, llama.cpp Metal baseline, model catalog
- [docs/CUDA_3080_SETUP.md](docs/CUDA_3080_SETUP.md): Linux / WSL2 bootstrap and the recommended Qwen 3.5 2B 4-bit consumer-GPU profile
- [CONTRIBUTING.md](CONTRIBUTING.md): expectations for docs, claims, and performance changes

## Related work

- [BigBird](https://arxiv.org/abs/2007.14062)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [Monarch](https://arxiv.org/abs/2204.00595)
- [FlexPrefill](https://arxiv.org/abs/2502.20766)
- [NSA](https://arxiv.org/abs/2502.11089)
- [MoBA](https://arxiv.org/abs/2502.13189)

## License

MIT
