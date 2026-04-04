# Apple Silicon Setup

This repo now includes a Mac-specific bootstrap for the April 3, 2026 target machine:

- macOS 26.1
- Apple M4 Max, 36 GB unified memory
- Xcode + `cmake` + `ninja` already installed
- `/Volumes/VIXinSSD/ZMLX` present
- `/Volumes/VIXinSSD/hf_cache` present

## What is supported now

| Path | Model family | Status | Notes |
| --- | --- | --- | --- |
| MLX + ZMLX custom kernels | Qwen 3.5 | Supported now | Primary Apple Silicon Butterfly path. Use `mlx-community/Qwen3.5-9B-MLX-4bit` first. |
| MLX + ZMLX custom kernels | Gemma 4 | Blocked for now | Local `mlx-lm 0.31.1` does not expose `gemma4` / `gemma4_text`. |
| llama.cpp + Metal | Qwen 3.5 | Supported when a GGUF is available | Use a local GGUF path or a GGUF HF repo with `--hf-repo`. |
| llama.cpp + Metal | Gemma 4 | Immediate baseline path | Preferred Apple Silicon baseline until MLX gains Gemma 4 support. |

## Bootstrap

Run the bootstrap once from the repo root:

```bash
./scripts/bootstrap_macos_metal.sh
```

What it does:

- creates `.venv-macos-metal`
- installs the repo plus MLX-side packages, including `datasets`
- links `/Volumes/VIXinSSD/ZMLX/src` into the venv with a `.pth` file
- builds `llama.cpp` tag `b8656` with Metal under `/Volumes/VIXinSSD/toolchains/llama.cpp-b8656-metal`
- creates `/Volumes/VIXinSSD/toolchains/llama.cpp-metal-current`
- runs smoke checks for `env_check_mlx.py`, `zmlx`, and `llama-cli`

If you only want the resolved environment values:

```bash
./scripts/bootstrap_macos_metal.sh --print-env-only
```

## Shared local paths

Source the shared shell environment before running benchmarks:

```bash
source ./scripts/macos_local_env.sh
```

That standardizes:

- `HF_HOME=/Volumes/VIXinSSD/hf_cache`
- `HF_HUB_CACHE=/Volumes/VIXinSSD/hf_cache/hub`
- `BUTTERFLY_MODELS_ROOT=/Volumes/VIXinSSD/models`
- `BUTTERFLY_ZMLX_ROOT=/Volumes/VIXinSSD/ZMLX`
- `BUTTERFLY_LLAMA_CPP_ROOT=/Volumes/VIXinSSD/toolchains/llama.cpp-b8656-metal`

## Model staging

List the built-in model catalog:

```bash
./.venv-macos-metal/bin/python scripts/model_catalog.py list
```

Stage the main MLX Qwen model into `/Volumes/VIXinSSD/models` as a symlink into the shared HF cache:

```bash
./.venv-macos-metal/bin/python scripts/model_catalog.py download qwen35_9b_mlx_4bit
```

Stage official Gemma 4 checkpoints into the shared HF cache:

```bash
./.venv-macos-metal/bin/python scripts/model_catalog.py download gemma4_31b_it_hf
./.venv-macos-metal/bin/python scripts/model_catalog.py download gemma4_26b_a4b_it_hf
```

For `llama.cpp`, link a local GGUF into the shared models root:

```bash
./.venv-macos-metal/bin/python scripts/model_catalog.py link-local \
  gemma4_31b_gguf /absolute/path/to/gemma4-31b-q4.gguf
```

## MLX / Butterfly benchmark workflow

Environment preflight:

```bash
./.venv-macos-metal/bin/python scripts/env_check_mlx.py
./scripts/verify_install_and_preflight.sh
```

Qwen 3.5 9B Butterfly smoke (uses `--mode wayfinder` internally; `--mode butterfly` also accepted):

```bash
./.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
  --model-path mlx-community/Qwen3.5-9B-MLX-4bit \
  --hf-offline \
  --mode wayfinder \
  --seq-lens 512 \
  --decode-len 16 \
  --repeats 1 \
  --skip-multi-turn \
  --skip-quality \
  --cooldown-sec 0 \
  --out-dir .benchmarks/qwen35_9b_mlx_smoke
```

Recommended reproducible Qwen 3.5 9B run:

```bash
./.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
  --model-path mlx-community/Qwen3.5-9B-MLX-4bit \
  --mode wayfinder \
  --seq-lens 2048 8192 32768 65536 98304 131072 163840 \
  --decode-len 256 \
  --repeats 3 \
  --window 64 \
  --head-chunk-size 2 \
  --query-chunk-size 384 \
  --out-dir results/benchmarks/qwen35_9b_mlx/wayfinder_repro
```

Dense baseline:

```bash
./.venv-macos-metal/bin/python scripts/bench_qwen_consumer_mlx.py \
  --model-path mlx-community/Qwen3.5-9B-MLX-4bit \
  --mode dense \
  --seq-lens 2048 8192 32768 65536 98304 131072 163840 \
  --decode-len 256 \
  --repeats 3 \
  --out-dir results/benchmarks/qwen35_9b_mlx/dense_repro
```

## llama.cpp baseline workflow

The new harness writes `manifest.json`, `speed.json`, `memory.json`, `max_context.json`, and `summary.json`.

Starter configs:

- `configs/benchmarks/qwen35_9b_llama_cpp_metal.json`
- `configs/benchmarks/gemma4_31b_llama_cpp_metal.json`

Example with a local Gemma 4 GGUF:

```bash
./.venv-macos-metal/bin/python scripts/bench_llama_cpp_metal.py \
  --config configs/benchmarks/gemma4_31b_llama_cpp_metal.json \
  --model /Volumes/VIXinSSD/models/gemma4_31b_gguf/model.gguf \
  --out-dir .benchmarks/gemma4_31b_llama_cpp_metal
```

Example with an HF GGUF repo that `llama.cpp` can fetch directly:

```bash
./.venv-macos-metal/bin/python scripts/bench_llama_cpp_metal.py \
  --config configs/benchmarks/qwen35_9b_llama_cpp_metal.json \
  --hf-repo <user>/<gguf-repo>:Q4_K_M \
  --out-dir .benchmarks/qwen35_9b_llama_cpp_metal
```

Notes:

- `speed.json` comes from `llama-bench`
- `memory.json` and `max_context.json` come from `llama-cli` wrapped with `/usr/bin/time -l`
- the memory probe uses a short prompt because KV allocation is driven by `--ctx-size`

## Practical support matrix

Use this repo as follows until the MLX gap closes:

1. Qwen 3.5 on MLX Butterfly is the primary Apple Silicon target.
2. Gemma 4 on MLX should be treated as blocked until `mlx-lm` adds `gemma4`.
3. Gemma 4 on `llama.cpp` Metal is the immediate Apple baseline for memory, max-context, and speed measurements.
