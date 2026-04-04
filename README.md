# Butterfly

Butterfly Network Attention (`bna`) is a training-free sparse-attention runtime for long-context inference. It is aimed at engineers who want measurable speed or memory wins without retraining the model.

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

Dense causal attention does `O(T^2)` work per layer. Butterfly replaces that with a bounded sparse pattern over fixed-size token blocks.

At a high level, each block attends to:

- its local neighborhood
- a small number of deterministic long-range partners
- optional global or anchor-style connections, depending on the backend

The exact sparse pattern differs across code paths. Older Wayfinder/HCSA integrations describe this as `window + cycle + landmarks`; the current Butterfly README uses the simpler butterfly-partner framing. In both cases the goal is the same: keep attention neighborhoods explicit, bounded, and cheap enough to help at long context.

For contributor-facing implementation details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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

The repo also contains experimental CUDA benchmark results for a Triton block-sparse path on Qwen 3.5 9B, where 8 of 32 layers are replaced and the remaining DeltaNet layers stay untouched.

| Context | Dense tok/s | Butterfly tok/s | Top-1 agreement |
|--------:|------------:|----------------:|----------------:|
| 4,096   | —           | —               | 99.88%          |
| 8,192   | 1,651       | 1,698           | —               |
| 16,384  | —           | —               | 94.44%          |
| 32,768  | 1,585       | 1,688           | —               |
| 65,536  | 1,475       | 1,724           | —               |
| 98,304  | 1,413       | 1,660           | —               |
| 131,072 | 1,365       | 1,667           | —               |
| 262,144 | 1,257       | 1,712           | —               |

These numbers suggest flatter throughput than dense attention at long context, but this path should still be treated as experimental until the quality and support boundaries are documented as tightly as the GLM release path.

### Experimental CUDA path: Qwen 3.5 35B A3B FP8

| Context | Dense tok/s | Butterfly tok/s |
|--------:|------------:|----------------:|
| 8,192   | 931         | 954             |
| 32,768  | 1,280       | 1,301           |
| 65,536  | 1,241       | 1,326           |
| 131,072 | 1,131       | 1,331           |
| 163,840 | —           | 1,306           |
| 196,608 | —           | 1,364           |
| 229,376 | —           | 1,233           |

### Experimental Apple Silicon path: Qwen 3.5 9B on M4 Max

MLX permute-window path with K6 fused Metal kernel, `window=64`. 8 of 32 attention layers are replaced. Model: `mlx-community/Qwen3.5-9B-MLX-4bit`.

| Context | Dense TTFT | Butterfly TTFT | Dense tok/s | Butterfly tok/s | Peak memory |
|--------:|-----------:|---------------:|------------:|----------------:|------------:|
| 2,048   | 71 ms      | 49 ms          | 62.2        | 62.0            | 7.1 GB      |
| 8,192   | 116 ms     | 86 ms          | 57.2        | 58.8            | 9.9 GB      |
| 32,768  | 100 ms     | 99 ms          | 49.6        | 47.1            | 13.7 GB     |
| 65,536  | 160 ms     | 202 ms         | 41.5        | 39.8            | 18.9 GB     |
| 98,304  | 2.0 s      | 1.2 s          | 17.2        | 22.4            | 24.0 GB     |
| 131,072 | 6.9 s      | 7.5 s          | 7.3         | 6.8             | 29.1 GB     |
| 163,840 | 26.8 s     | 21.5 s         | 2.2         | 2.7             | 34.2 GB     |

This MLX path uses chunked-gather plus native SDPA for prefill and a fused Metal kernel for decode. It shows wins at short context and again near the memory wall, but it is still an experimental path rather than a validated public release.

Top-1 agreement in the Qwen 9B experiments is `99.88%` at 4K and `94.44%` at 16K. Perplexity and downstream evaluation are still in progress, so avoid treating these tables as universal quality-parity claims.

## Quick start

### CUDA (NVIDIA GPU)

```bash
git clone https://github.com/Hmbown/Butterfly.git
cd Butterfly
pip install -e ".[dev,kernels]"
```

Validated public path:

```bash
./scripts/run_public_stable_profile_glm.sh
```

Experimental Qwen CUDA benchmark:

```bash
python scripts/bench_qwen35_cuda_wayfinder.py \
    --model-path <path-to-Qwen3.5-9B> \
    --path block_sparse \
    --engine triton \
    --block-size 128 \
    --seq-lens 4096 8192 16384 32768
```

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

The `--mode dense` flag runs the stock attention baseline for comparison. Add `--skip-quality` to benchmark only throughput.

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

- [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md): validated benchmark slice and reproduction commands
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): contributor-facing implementation map
- [docs/APPLE_SILICON_SETUP.md](docs/APPLE_SILICON_SETUP.md): Apple Silicon bootstrap, llama.cpp Metal baseline, model catalog
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
