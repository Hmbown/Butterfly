# Qwen3-4B Long-Context MLX Infrastructure Map

## Model Selection

| Candidate | Status | Source |
|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | **Selected** | MLX-native (pre-quantized W4A16) |
| `mlx-community/Qwen3-4B-Instruct-4bit` | Fallback | MLX-native instruct variant |
| `Qwen/Qwen3-4B` | Fallback | HF base (would need conversion) |
| `Qwen/Qwen3-4B-Instruct` | Fallback | HF instruct (would need conversion) |

**Final model**: `mlx-community/Qwen3-4B-4bit`
- Model type: `qwen3`
- Hidden size: 2560
- Attention heads: 32 (Q), 8 (KV) -- GQA with 4:1 ratio
- Max position embeddings: 40960
- Vocab size: 151936
- RoPE scaling: None (native up to 40960)
- Quantization: 4-bit (group size 64, bfloat16 compute)

## Tokenizer

- Source: `mlx-community/Qwen3-4B-4bit` (bundled with model)
- Loaded via `mlx_lm.utils.load_tokenizer` with `trust_remote_code=True`
- EOS token ID: 151645
- Tokenizer hash: `e109e68e41c8588b`

## Context & RoPE

- Native max context: 40960 tokens
- Training target: 32768 tokens (within native range, no RoPE scaling needed)
- Warmup context: 8192 tokens
- Benchmark contexts: 2048, 8192, 32768

## Dataset

- **Primary**: Local codebase at `<repo>/` (self-referential)
- **Fallback logic**: Script tries `hf:<dataset_id>` first, falls back to `local:<path>` on failure
- **Packing**: Fixed-length non-overlapping at 32768 tokens per sequence
- **Split**: 95% train / 5% val (deterministic seed=42)
- **Result**: 232 train sequences + 12 val sequences
- **Total tokens**: ~8M raw tokens
- **Storage**: `datasets/qwen3_4b/local__volumes_vixinssd_wayfinder/{train.bin,val.bin,val_long.bin,meta.json}`

## Training Mode

- **Method**: LoRA adapters via `mlx_lm.tuner.utils.linear_to_lora_layers`
- **LoRA config**: rank=8, scale=16.0, dropout=0.0, last 8 layers
- **Optimizer**: AdamW (lr=1e-5, no weight decay)
- **LR schedule**: Linear warmup over 20 steps
- **Gradient accumulation**: 8 (at 8k) / 32 (at 32k)
- **Batch size**: 1
- **Precision**: bfloat16

## HCSA Application

### Level A (Mandatory) -- Attention Microbenchmark
- Extract real Q/K/V from Qwen3 attention layer using `extract_qkv_from_qwen_attention()`
- Run Wayfinder attention (sparse or permute path) on extracted Q/K/V
- Compare output MAE against dense baseline
- Measure: tok/s, peak memory, graph build time, cache hit rate

### Level B (Optional) -- Full Model Swap
- Replace all `self_attn` modules with `QwenWayfinderAttention` via `swap_qwen_attention_with_wayfinder()`
- Smoke test: short-context forward pass through full model
- Used in training loop for actual loss optimization

### HCSA Settings
- **Path**: `permute` (fast contiguous window path)
- **Strategy**: `random` (deterministic, input-independent -> enables static caching)
- **Window**: 64
- **Landmark stride**: 64
- **Num cycles**: 1
- **Seed**: 42
- **Edge bias**: Enabled
- **Window drop**: Ramped 0.0 -> 0.25 over 100 steps (training regularization)
- **Edge bias schedule**: Cycle 0.0->0.4, Window 0.0->-0.1, Landmark 0.0->0.0

### Graph Spec
- Config file: `configs/graph_specs/qwen3_4b_long.wf`
- Degree: 128
- Backbone: random cycle (k=1, seed=42)
- Local window: 128->32 linear schedule over 2000 steps
- Highways: landmarks stride=128
- Permute window: enabled, window=64

## Artifact Locations

| Artifact | Path |
|---|---|
| Env check | `runs/mlx/qwen_env_check_full.json` |
| Model fetch | `runs/mlx/qwen3_fetch_or_convert_full.json` |
| Baseline bench | `benchmarks/mlx/qwen3_4b_baseline/<timestamp>/` |
| HCSA bench | `benchmarks/mlx/qwen3_4b_wayfinder/<timestamp>/` |
| Dataset | `datasets/qwen3_4b/local__volumes_vixinssd_wayfinder/` |
| Training run | `runs/mlx/qwen3_4b_wayfinder_<timestamp>/` |
| Graph spec | `configs/graph_specs/qwen3_4b_long.wf` |
| Graph cache | `.cache/wayfinder/` |

## Hardware

- **Machine**: Apple M4 Max
- **Architecture**: applegpu_g16s (arm64)
- **Unified memory**: 36 GB
- **Recommended working set**: 28.08 GB
- **MLX version**: 0.30.6
- **mlx_lm version**: 0.30.6
- **Disk free**: ~345 GB (external SSD)
