# Qwen3.5-35B-A3B Wayfinder Setup (No-Inference)

Date: 2026-02-25

This document prepares a Wayfinder setup for `Qwen3.5-35B-A3B` without running inference or benchmarks.

Status note: this path is experimental. Current work includes custom kernels needed to get HCSA/Wayfinder working correctly on the Qwen 3.5 family. The present goal is to keep prefill speed roughly consistent as context length increases, but that goal is still under investigation and is not a validated claim.

## Facts

1. Local model artifacts exist on VixinSSD:
   - `/Volumes/VixinSSD/models/Qwen3.5-35B-A3B-MLX-4bit` (selected local path, 4-bit MLX safetensors shards)
   - `/Volumes/VixinSSD/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6`
   - `/Volumes/VixinSSD/hf_cache/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec`

2. The selected MLX model config reports a hybrid MoE stack:
   - top-level `model_type=qwen3_5_moe`
   - `text_config.num_hidden_layers=40`
   - `layer_types`: 30 `linear_attention`, 10 `full_attention` (`full_attention_interval=4`)
   - `max_position_embeddings=262144`, `num_experts=256`, `num_experts_per_tok=8`

3. Wayfinder Qwen swap logic only replaces layers exposing `self_attn`:
   - `hcsa/integrations/qwen_mlx.py` (`swap_qwen_attention_with_wayfinder`) skips layers that do not have `self_attn`.
   - On this model family, that means only full-attention layers are eligible for replacement by default.

4. Consumer/benchmark harness already supports the required Wayfinder knobs:
   - `scripts/bench_qwen_consumer_mlx.py` supports `--model-path`, HF cache controls, `--mode`, chunking controls, and decode backend policy (`--debug-wayfinder-decode-backend`).
   - `scripts/bench_qwen_wayfinder_mlx.py` supports `--model-path`, HF cache controls, graph controls, and `--decode-backend`.

5. Local runtime compatibility blocker exists right now:
   - local `mlx-lm` is `0.30.5`
   - `mlx_lm.utils._get_classes` fails on `qwen3_5_moe` in this environment
   - upstream `mlx-lm` release notes indicate `qwen3.5` text-only support was added in `v0.30.7`.

## Assumptions

1. Upgrading `mlx-lm` to `>=0.30.7` will unblock model loading for this setup path without repo-side loader patches.
2. Running Wayfinder only on full-attention layers is acceptable for first-pass parity/perf experiments on this hybrid architecture.
3. Initial queue should stay conservative (`retro_backfill_inference=off`, one process at a time).

## Prepared Setup Artifacts

1. Experiment setup config:
   - `configs/experiments/qwen3_5_35b_a3b_wayfinder_setup.yaml`
2. Queue script:
   - `scripts/queue_qwen35_a3b_wayfinder.sh`
   - default behavior: print commands only (no model execution)
   - execution requires explicit `--execute`

## Prepared Command Queue (Later Execution)

Queue script emits and can execute (with `--execute`) four paired runs:

1. Dense at `T=2048`
2. Wayfinder (dense decode backend) at `T=2048`
3. Dense at `T=8192`
4. Wayfinder (dense decode backend) at `T=8192`

All commands are pinned to:

- local model path: `/Volumes/VixinSSD/models/Qwen3.5-35B-A3B-MLX-4bit`
- HF offline cache:
  - `HF_HOME=/Volumes/VixinSSD/hf_cache`
  - `HF_HUB_CACHE=/Volumes/VixinSSD/hf_cache/hub`
  - `--hf-offline`
- conservative run policy:
  - `--skip-multi-turn --skip-quality`
  - `--stage-timeout-sec 1800 --heartbeat-sec 30`
  - retro/backfill inference disabled by default

## Stop-Gates

1. Do not run queue until `mlx-lm >= 0.30.7`.
2. Any nonzero exit or missing `results.json` is a hard fail.
3. For Wayfinder runs, require `swap.replaced_layers >= 1`.
4. If dense fallback occurs, require informative `dense_fallback_reason_counts`.

## Primary Sources

- Qwen model card (`Qwen/Qwen3.5-35B-A3B`): <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- Transformers model doc (`Qwen3.5-MoE`): <https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5_moe>
- mlx-lm release (`v0.30.7`, qwen3.5 text support note): <https://github.com/ml-explore/mlx-lm/releases/tag/v0.30.7>
- mlx-lm PR (`#869`, add qwen3.5 support): <https://github.com/ml-explore/mlx-lm/pull/869>
