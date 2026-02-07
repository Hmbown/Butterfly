# HCSA Qwen3-4B MLX Overnight Run — Handoff Prompt

**Copy this entire file as a prompt to Claude Code or another AI coding agent.**

---

You are an autonomous coding agent with terminal + file edit access. Run long jobs to completion.

## REPO
- `/Volumes/VIXinSSD/wayfinder`
- Branch: `codex/qwen3-4b-mlx-overnight-full` (already created, already checked out)
- Must use `python3` not `python`

## IMPORTANT WORKTREE RULES
- Do NOT revert or disturb any pre-existing changes.
- Keep all work MLX-only on Apple Silicon. No CUDA, no Triton, no GPUMode.

## CONTEXT: WHAT'S ALREADY DONE
Everything below is already present and working unless marked with **BLOCKED**.

### Phase 0 (DONE)
- `docs/qwen3_4b_infra_map.md` — infrastructure map with model/dataset/training details
- `runs/mlx/qwen_env_check_full.json` — environment validated
  - Apple M4 Max, 36GB unified memory, MLX 0.30.4, mlx_lm 0.30.5
  - Recommended: batch=1, grad_accum=8 (8k) / 32 (32k), bfloat16

### Phase 1 (DONE)
- Model: `mlx-community/Qwen3-4B-4bit` validated and cached
  - `runs/mlx/qwen3_fetch_or_convert_full.json`
  - 32 Q heads, 8 KV heads (GQA 4:1), hidden=2560, max_pos=40960
- Baseline benchmark: `benchmarks/mlx/qwen3_4b_baseline/20260207_054544/results.json`
  - T=2048: attn 129k tok/s, 94 MB | block 41k tok/s, 240 MB
  - T=8192: attn 78k tok/s, 374 MB | block 34k tok/s, 798 MB
  - T=32768: attn 28k tok/s, 1334 MB | block 19k tok/s, 2870 MB

### Phase 2 (**BLOCKED — THE MAIN ISSUE**)
- HHA benchmark at T=2048 completed (full): `benchmarks/mlx/qwen3_4b_hha/20260207_054622/results.json`
  - But shows HHA is 7100x SLOWER than dense (17.9 tok/s vs 128k)
  - HHA uses 4.2 GB vs dense 120 MB at T=2048
- HHA at T=8192 and T=32768: **COULD NOT COMPLETE — benchmark script times out**
- Quick per-head benchmark (partial output, no JSON written):
  - T=2048 per-head: head 0 = 0.10s, head 1 = 0.01s, per-head peak = 539 MB
  - T=2048 graph cache: 110 MB
  - T=8192 graph build started but was killed

### ROOT CAUSE ANALYSIS — WHY HHA IS SLOW
The `QwenWayfinderAttention.__call__()` in `hcsa/integrations/qwen_mlx.py` has TWO critical performance bugs:

**Bug 1: Per-head Python loop in permute attention**
The old `wayfinder_permute_window_attention()` in `hcsa/mlx/attention.py` processes all 32 heads sequentially in a Python `for` loop (line 355). Each iteration creates large intermediate tensors. The loop prevents MLX from parallelizing across heads.

**Partial fix already applied:** A new `wayfinder_permute_window_attention_batched()` function has been added to `hcsa/mlx/attention.py` (line 390+). It uses `mx.take_along_axis` to process all heads simultaneously with stacked tensors. The `_QwenGraphCache` in `qwen_mlx.py` has been updated to include stacked tensors (`perm_mx_stacked`, `inv_perm_stacked`, `pi_idx_stacked`, `valid_mask_stacked`, `causal_mask_stacked`). But `QwenWayfinderAttention.__call__()` has **NOT yet been updated to call the batched function** — it still calls the old per-head-loop sparse/permute paths.

**Bug 2: `_edge_utilization_proxy` calls `mx.eval()` inside forward path**
The old code called `mx.eval()` 4 times inside a loop (one per edge type). A **partial fix has been applied** — it now vectorizes the computation into one `mx.eval()` call. But this is still a synchronization point inside the forward path.

### WHAT YOU NEED TO DO FOR PHASE 2

1. **Wire up the batched permute path in `QwenWayfinderAttention.__call__()`:**
   In `hcsa/integrations/qwen_mlx.py`, around line 550 where it currently calls `wayfinder_permute_window_attention(...)`, change the `elif self.path == "permute":` branch to call `wayfinder_permute_window_attention_batched(...)` instead, passing the stacked tensors from `graph_cache`:
   ```python
   elif self.path == "permute":
       y_h, _w = wayfinder_permute_window_attention_batched(
           queries, keys, values,
           all_perms=graph_cache.perm_mx_stacked,
           all_inv_perms=graph_cache.inv_perm_stacked,
           all_pi_idx=graph_cache.pi_idx_stacked,
           all_valid=graph_cache.valid_mask_stacked,
           all_causal=graph_cache.causal_mask_stacked,
           edge_type_bias_scalar=etb_scalar,
           window_drop_prob=effective_window_drop if is_training else 0.0,
           training=is_training,
       )
       permute_ms = 0.0
       keep_mask = graph_cache.causal_mask
   ```

2. **Make `_edge_utilization_proxy` optional during timed benchmarks:**
   The function is already partially vectorized. Consider making it lazy or conditional — only compute it every N steps during training, or skip it entirely in benchmark mode. Right now it's called every forward pass unconditionally.

3. **Verify correctness:** After wiring up the batched path, run:
   ```bash
   python3 -m pytest tests/mlx/ -v
   ```
   Then run a small smoke test:
   ```python
   # Quick verification that batched matches per-head
   python3 -c "
   import sys; sys.path.insert(0,'.')
   import mlx.core as mx
   from hcsa.integrations.qwen_mlx import QwenHHAConfig, QwenWayfinderAttention
   from mlx_lm import load
   model, tok, cfg = load('mlx-community/Qwen3-4B-4bit', return_config=True, lazy=True, tokenizer_config={'trust_remote_code': True})
   attn = model.layers[0].self_attn
   hha = QwenWayfinderAttention(attn, QwenHHAConfig(path='permute', window=64, landmark_stride=64, seed=42))
   x = mx.random.normal((1, 512, 2560), dtype=mx.bfloat16)
   y = hha(x, mask='causal', cache=None)
   mx.eval(y)
   print(f'Output shape: {y.shape}, dtype: {y.dtype}')
   print('SUCCESS')
   "
   ```

4. **Re-run benchmarks** once the batched path works:
   ```bash
   # Baseline (already done, but re-run if you want fresh numbers)
   python3 scripts/bench_qwen3_4b_baseline_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 2048 8192 32768 --batch 1 --warmup 1 --iters 3

   # HHA benchmark at all three T values
   python3 scripts/bench_qwen3_4b_hha_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 2048 8192 32768 --batch 1 --warmup 1 --iters 3 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap

   # If 32768 is too slow or OOMs, run separately:
   python3 scripts/bench_qwen3_4b_hha_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 2048 --batch 1 --warmup 1 --iters 3 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap
   python3 scripts/bench_qwen3_4b_hha_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 8192 --batch 1 --warmup 1 --iters 2 --path permute --window 64 --landmark-stride 64 --seed 42
   python3 scripts/bench_qwen3_4b_hha_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 32768 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42
   ```

### Phase 3 (DONE)
- Dataset: `datasets/qwen3_4b/local__volumes_vixinssd_wayfinder/`
  - `{train.bin, val.bin, val_long.bin, meta.json}` all present
  - 232 train sequences + 12 val sequences @ 32768 tokens
  - Seed=42, deterministic packing

### Phase 4 (NOT STARTED — depends on Phase 2 fix)
After the batched permute path is working and benchmarks show reasonable HHA performance:

```bash
# Training: 80 steps at 8k warmup, then steps at 32k
python3 scripts/train_qwen3_4b_wayfinder_mlx.py \
  --model-path mlx-community/Qwen3-4B-4bit \
  --dataset-dir datasets/qwen3_4b/local__volumes_vixinssd_wayfinder \
  --seq-len 32768 \
  --warmup-seq-len 8192 \
  --warmup-steps 80 \
  --steps 200 \
  --batch-size 1 \
  --grad-accum 8 \
  --eval-every 10 \
  --eval-batches 2 \
  --save-every 25 \
  --lr 1e-5 \
  --hha-mode permute \
  --window 64 \
  --landmark-stride 64 \
  --num-cycles 1 \
  --swap-last-n-layers -1 \
  --lora-rank 8 \
  --num-lora-layers 8 \
  --seed 42
```

**IMPORTANT:** The training script also uses `QwenWayfinderAttention.__call__()` for every forward pass on every layer. If a single forward pass through one layer takes 100+ seconds (as with the old per-head loop), training is impossible. The batched path fix from Phase 2 is **mandatory** before training can work.

If training at 32k is too slow or OOMs even with the fix, reduce scope:
- `--swap-last-n-layers 8` (only swap last 8 of 36 layers)
- `--warmup-steps 50 --steps 100` (shorter run)
- `--grad-accum 32` at 32k

### Phase 5 (NOT STARTED — depends on Phases 2+4)
Write the report and README update after benchmarks and training complete.

## KEY FILES TO MODIFY

1. **`hcsa/integrations/qwen_mlx.py`** — Wire batched permute path into `QwenWayfinderAttention.__call__()`. The stacked cache tensors and batched function are ready — just need to be called.

2. **`hcsa/mlx/attention.py`** — The `wayfinder_permute_window_attention_batched()` function is already added. May need debugging if shapes don't match. Also update `WayfinderAttentionMLX.__call__()` (the non-Qwen module) similarly if needed.

## QUALITY GATES
- `pytest tests/mlx/` — all 26 tests must pass
- `pytest tests/pytorch/` — all 45 tests must pass (don't break existing)
- Baseline bench at T={2048,8192,32768} exists and is valid
- HHA bench completes at T={2048,8192,32768} with reasonable performance
- HHA shows memory advantage at T=32768 (sparse T*D << dense T^2)
- Training completes at least 50 steps at 8k + some steps at 32k
- Report at `docs/qwen3_4b_overnight_report.md`
- README updated with Qwen3-4B section

## WHAT "SHOWING THIS OFF PROPERLY" MEANS
The key claim of HCSA/HHA is: sparse attention via Hamiltonian cycles gives O(T*D) memory instead of dense O(T^2), which matters at long contexts. At T=32768 with D~100 and W=129:
- Dense attention matrix: T^2 = 1.07B entries → ~4 GB
- HHA window: T*W = 4.2M entries → ~17 MB per head
- That's a **254x theoretical memory reduction** for the attention computation

The benchmark needs to show this materializes in practice, even if throughput is lower due to the prototype Python-level implementation. The story is: "HHA achieves significant memory reduction at 32k+ context, enabling longer sequences on consumer hardware. Current throughput gap is an implementation artifact of Python-level head processing, not fundamental to the algorithm."

## EXISTING BENCHMARK DATA TO REFERENCE
Baseline at T=32768: attn 28k tok/s, 1334 MB peak
The HHA benchmark needs to show something like: attn X tok/s (slower), Y MB peak (lower than 1334 MB ideally)
Graph build: ~1.6s at T=2048, should be amortized by caching

## EXECUTION NOTES
- Long jobs can run a long time — the 10-minute tool timeout is real. Use `run_in_background` for commands > 5 minutes.
- The model is already cached at `/Volumes/VIXinSSD/hf_cache/` — loads in <1 second.
- If graph build at T=32768 takes too long (>60s), that's expected — report it honestly.
- All artifacts should go in timestamped directories (don't overwrite existing).
- Commit to `codex/qwen3-4b-mlx-overnight-full` branch when done.
