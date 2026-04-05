# CUDA 3080 Setup

This is the recommended Butterfly bring-up for a single `RTX 3080 10GB` machine with about `12 GB` of system RAM.

Assumption:

- Linux or WSL2 Ubuntu
- NVIDIA driver already working
- one CUDA GPU

## WSL2 host setup

If this machine is a Windows box using WSL2, do this first:

1. Install the latest Windows NVIDIA driver with CUDA-on-WSL support.
2. From an elevated Windows terminal, run:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

3. Start Ubuntu and confirm the GPU is visible:

```bash
nvidia-smi
```

Do not install a Linux NVIDIA display driver inside WSL. The Windows driver is the CUDA driver for the WSL guest.

## Recommended model

Use:

- `Qwen/Qwen3.5-2B`

Run it with:

- `--quantize bnb-4bit`

Why this is the default:

- the official Qwen 3.5 `2B` and `4B` checkpoints both expose `262,144` native context
- the `4B` checkpoint has materially larger hidden state and KV footprint, which makes the `10 GB` budget too tight once long-context cache and temporary buffers are included
- the `2B` checkpoint leaves enough headroom to probe the upper context ladder on a 10 GB card without depending on fragile community AWQ installs

Approximate memory floor from config-derived KV math:

| Model | Weight path | 131K KV only | 262K KV only | Practical verdict on 10GB |
| --- | --- | ---: | ---: | --- |
| `Qwen/Qwen3.5-2B` + `bnb-4bit` | runtime 4-bit | ~1.5 GiB | ~3.0 GiB | viable long-context target |
| `Qwen/Qwen3.5-4B` + `bnb-4bit` | runtime 4-bit | ~4.0 GiB | ~8.0 GiB | too tight for upper-range work |

The `4B` model may still be useful for shorter runs, but it should not be the default 3080 experiment target.

## Bootstrap

For WSL2 Ubuntu, run once from the repo root:

```bash
./scripts/bootstrap_wsl2_ubuntu.sh
```

For native Linux, use:

```bash
./scripts/bootstrap_cuda_linux.sh
```

This creates:

- `.venv-cuda`
- repo-local Hugging Face cache under `.cache/butterfly`
- a CUDA-ready Python env with `transformers`, `accelerate`, `bitsandbytes`, `triton`, `fastapi`, and repo extras

The WSL2 wrapper also installs the Ubuntu-side prerequisites (`python3`, `python3-venv`, `python3-pip`, `build-essential`) before delegating to the shared CUDA bootstrap.

If you only want the resolved environment values:

```bash
./scripts/bootstrap_cuda_linux.sh --print-env-only
```

## Shared local paths

Before downloading models or benchmarking:

```bash
source ./scripts/cuda_local_env.sh
```

That standardizes:

- `HF_HOME=$REPO_ROOT/.cache/butterfly/hf_cache`
- `HF_HUB_CACHE=$HF_HOME/hub`
- `BUTTERFLY_MODELS_ROOT=$REPO_ROOT/.cache/butterfly/models`

## Download the model

Recommended explicit download:

```bash
./.venv-cuda/bin/python scripts/model_catalog.py download qwen35_2b_hf
```

That stages the official checkpoint in the shared cache and creates a stable local link under `BUTTERFLY_MODELS_ROOT`.

You can also skip the manual download and let the benchmark pull `Qwen/Qwen3.5-2B` on first run.

## Smoke check

```bash
./.venv-cuda/bin/python scripts/env_check_cuda.py
```

## Full 3080 experiment

Recommended full ladder:

```bash
./scripts/run_qwen35_3080_profile.sh
```

Current defaults in that wrapper:

- model: `Qwen/Qwen3.5-2B`
- quantization: `bnb-4bit`
- dtype: `float16`
- path: `block_sparse`
- engine: `triton`
- forward target: `backbone`
- seq lens: `8192 16384 32768 65536 131072 196608 262144`
- phases: dense + wayfinder
- divergence: skipped for long-context safety

This profile uses the current CUDA Butterfly block-sparse path rather than the older flex-only Wayfinder surface:

- prefill backend: `triton` by default
- unsupported-arch fallback: `sdpa` when Triton is unavailable
- cached decode / cached prefill: exact sparse GQA over the Butterfly block support

Override any default with environment variables. Example:

```bash
MODEL_PATH=Qwen/Qwen3.5-2B \
SEQ_LENS="16384 32768 65536 131072" \
REPEATS=1 \
./scripts/run_qwen35_3080_profile.sh
```

## Notes

- This profile deliberately uses the official `2B` checkpoint plus `bitsandbytes` 4-bit runtime quantization instead of a community AWQ default.
- The reason is operational stability: current Hugging Face docs warn that `autoawq` downgrades `transformers`, which is not a good default for this repo.
- If you later want a higher-quality mid-context CUDA target, open `Qwen/Qwen3.5-4B` behind the same `bnb-4bit` path, but do not treat it as the default 10GB long-context setup.
