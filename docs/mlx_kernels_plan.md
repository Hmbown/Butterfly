# MLX Kernel Plan (Apple Silicon)

## Scope
This repo now uses a single Wayfinder Graph ABI and two MLX attention paths:
- `wayfinder_sparse`: general sparse-row gather (correctness path)
- `wayfinder_permute`: permute-to-cycle-order + local window (fast path)

Tonight uses **Option 2** (kernel seam + kernel-friendly MLX pipeline).

## Memory Layout Decisions
- Q/K/V head layout: `[B, H, T, dh]`
- Graph ABI layout: `neigh_idx [H, T, D]`, `edge_type [H, T, D]`
- Sparse gather uses `k_h[:, idx_h]` / `v_h[:, idx_h]` where `idx_h [T, D]`
- Permute fast path uses contiguous cycle-ordered tensors:
  - `q_pi = q[:, perm]`, `k_pi = k[:, perm]`, `v_pi = v[:, perm]`
  - sliding window gather by contiguous offsets in permuted space

## Metal Seam (Exact Future Kernel Signature)

### 1) Sparse-row fused kernel (general path)
Proposed signature:
- Inputs:
  - `q: half/bfloat16 [B, H, T, dh]`
  - `k: half/bfloat16 [B, H, T, dh]`
  - `v: half/bfloat16 [B, H, T, dh]`
  - `neigh_idx: int32 [H, T, D]` (`-1` pad)
  - `edge_type: uint8 [H, T, D]` (optional for stats)
- Output:
  - `out: half/bfloat16 [B, H, T, dh]`
  - optional `attn: float32 [B, H, T, D]` for diagnostics
- Compute contract:
  - dot products in FP32
  - masked softmax in FP32
  - weighted sum accumulation in FP32, cast to output dtype
  - all-masked rows produce zeros

### 2) Permute-window fused kernel (fast path)
Proposed signature:
- Inputs:
  - `q_pi, k_pi, v_pi: half/bfloat16 [B, H, T, dh]`
  - `perm: int32 [H, T]`
  - `window: int32`
  - optional `orig_pos: int32 [H, T]` (or infer from `perm`)
- Output:
  - `out_pi: half/bfloat16 [B, H, T, dh]`
- Compute contract:
  - contiguous local window in permuted space
  - causal check in original index space (`orig_key <= orig_query`)
  - stable masked softmax in FP32

## Fast vs General Path
- Fast path on MLX: `wayfinder_permute`
  - contiguous memory, reduced irregular gather pressure
- General path: `wayfinder_sparse`
  - exact ABI semantics over arbitrary neighbor lists
  - reference path for correctness and regression checks

## Known Constraints
- `D` can vary per head; stacked ABI pads to `D_max`.
- `cycle_perms` must be present in ABI meta for permute fast path.
- `edge_type` currently tracks `PAD/CYCLE/WINDOW/LANDMARK/REWIRE`; rewiring support is plumbing-ready.
- Current graph build runs on CPU (Python + torch strategies) and is timed separately (`graph_build_ms`).

## TODO Markers For Metal Hookup
- Add `hcsa/mlx/kernels/metal/sparse_row_attention.metal`
- Add MLX custom op wrapper under `hcsa/mlx/kernels/metal/__init__.py`
- Swap `sparse_gather_attention()` compute core with custom op call behind `use_metal_kernel` flag.
- Keep ABI unchanged so kernel drop-in is non-breaking.

## Discovery Setup Workflow (No Inference)

Use setup-only commands to scaffold K1-K5 discovery targets before any model runs:

```bash
python3 scripts/wayc.py discover-targets --targets all
python3 scripts/wayc.py discover-setup \
  --targets all \
  --zmlx-root /path/to/ZMLX \
  --sessions-root discover_sessions \
  --kernel-out-root hcsa/mlx/kernels/metal \
  --strict
```

This writes:
- `discover_sessions/manifest.json`
- `discover_sessions/*_session.stub.json`
- `hcsa/mlx/kernels/metal/seeds/*.metal`

No model loading, inference, or attention benchmarking is performed in this phase.
