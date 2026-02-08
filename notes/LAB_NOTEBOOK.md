# Bell Labs Notebook

This notebook is the narrative companion to `notes/experiments.ndjson`.
Each experiment is recorded as hypothesis -> intervention -> measurement -> decision.

## 2026-02-07 — Qwen3-1.7B Permute Path (MLX)

### EXP-20260207-133728
- Question: Can chunked permute attention reduce peak memory vs dense at long context?
- Hypothesis: Removing `[T,W,dh]` materialization and deferring KV repeat should drop long-context peak memory.
- Change set: chunked batched permute path + GQA defer + benchmark retention fix.
- Command:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens 2048 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap`
- Key result:
  - `T=8192` attention dense=`1,186,678,824`, wayfinder=`1,156,994,376` (better by `29,684,448`).
- Decision: Keep.

### EXP-20260207-154622
- Question: Can we preserve long-context memory win while improving Wayfinder latency?
- Hypothesis: Preallocated chunk output + tuned chunk policy (`h<=2`, `q<=384`) reduces overhead.
- Change set: `y_pi` preallocation + chunk writes; policy update in Qwen integration.
- Command:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens 2048 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap`
- Key result:
  - `T=8192` attention dense=`1,186,678,824`, wayfinder=`1,162,467,674` (better by `24,211,150`, `+2.04%` reduction).
  - Wayfinder latency improved from `162.41 ms` to `96.28 ms` (`40.7%` faster vs previous Wayfinder).
- Decision: Keep.

### EXP-20260207-qwen-matrix-isolated-base
- Question: What is the clean isolated throughput/memory curve from 256..8192 with retro disabled?
- Hypothesis: Short contexts remain slower, long contexts should show memory reduction.
- Change set: none (measurement-only).
- Commands:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens {256|512|1024|2048|4096} --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --out-dir benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_matrix_isolated_base/T{T}`
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap --out-dir benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_long_cmp_base2`
- Key result:
  - `T=8192` memory reduction = `10.75%` (dense `225,182,248` -> wayfinder `200,971,098`).
  - `T=4096` memory reduction = `9.1%`.
- Decision: Keep as current baseline report.

### EXP-20260207-qwen-matrix-isolated-retro-a02
- Question: Does current retrocausal backfill improve Qwen memory/throughput?
- Hypothesis: Might improve representational flow but likely costs runtime/memory.
- Change set: retro enabled (`alpha=0.2`, offsets=`[1,2,4]`).
- Commands:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens {256|512|1024|2048|4096} --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --retro-backfill --retro-alpha 0.2 --retro-offsets 1 2 4 --retro-allow-inference --out-dir benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_matrix_isolated_retro_infer/T{T}`
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap --retro-backfill --retro-alpha 0.2 --retro-offsets 1 2 4 --retro-allow-inference --out-dir benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_long_cmp_retro_infer2`
- Key result:
  - `T=8192` retro memory is worse than dense (`262,084,974` vs `225,182,248`).
  - Throughput also worse than retro-off baseline.
- Decision: Do not enable retro by default; continue as training-only research path.

## 2026-02-07 — Simplified Hamiltonian-Local Retro Backfill

### EXP-20260207-retro-simple
- Question: Can a simpler Hamiltonian-local retro backfill maintain representational benefit with lower overhead?
- Hypothesis: Replacing the complex multi-offset gated retro with a single `mx.roll()` operation reduces complexity from O(T×k) to O(T), eliminates temporary allocations, and maintains training-time representational flexibility because the Hamiltonian cycle already defines a natural local neighborhood.
- Change set:
  - `hcsa/mlx/attention.py`: Remove `retro_backfill_offsets` parameter
  - `hcsa/mlx/attention.py`: Replace gated multi-offset loop with `mx.roll(y_pi, shift=-1, axis=2) * alpha`
  - `hcsa/mlx/attention.py`: Keep `retro_backfill_training_only=True` as causal-safe default
  - `tests/mlx/test_permute_retro_backfill.py`: Add tests for alpha=0 no-op, causality safety, multi-head perms
- Implementation sketch:
  ```python
  # OLD: O(T * len(offsets)) with learned gates and multiple allocations
  for off in retro_offsets:
      future = mx.concatenate([y_pi[:, :, off:, :], tail], axis=2)
      sim = mx.sum(y_pi * future, axis=-1, keepdims=True) * sim_scale
      gate = mx.sigmoid(sim)
      retro_acc = retro_acc + gate * future * valid
  
  # NEW: O(T) single roll, no gates, minimal memory
  retro_term = mx.roll(y_pi, shift=-1, axis=2)  # each pos gets successor in cycle
  y_pi = y_pi + alpha * retro_term
  ```
- Causal safety:
  - Default: `retro_backfill_enabled=False`
  - Guard: `retro_backfill_training_only=True` means retro only activates when `training=True`
  - Inference cannot accidentally enable retro without explicitly setting both flags
- Expected metrics:
  - Memory: Closer to baseline (no multi-offset materialization, no valid_acc accumulator)
  - Latency: Single roll + add vs loop over offsets
  - Quality: To be measured on original non-Qwen HCSA first (per strategic sequence)
- Decision: Pending test results on original non-Qwen HCSA.
- Next action: Run isolated matrix on original HCSA training to validate PPL impact before Qwen re-benchmark.

## Next Notebook Targets
- Reproduce original non-Qwen HCSA schedule runs and retro ablations before further Qwen retro tuning.
- Track cycle-utilization proxies alongside ppl for schedule decisions.
- Validate simplified retro on original HCSA, then port to Qwen if successful.

### EXP-20260207-TINY-SCALE-REFRESH (planned)
- Question: Does the original tiny MLX path still show long-context memory reduction after current permute and retro safety updates?
- Hypothesis: Tiny permute should preserve strong memory reduction at T>=2048, with retro-disabled baseline unaffected.
- Change set: none (measurement-only).
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 1024 2048 4096 --batch 1 --heads 8 --embd 512 --window 64 --landmark-stride 64 --warmup 1 --iters 1 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_scale_refresh`

### EXP-20260207-GPT2-WAYFINDER-BASE (planned)
- Question: Does GPT-2 show the same long-context memory trend as Qwen when swapped to Wayfinder permute path?
- Hypothesis: GPT-2 should show neutral/negative gain at short T and clear memory reduction at longer T, validating GPT-2 as stepping stone.
- Change set: new `hcsa/integrations/gpt2_mlx.py`, new `scripts/bench_gpt2_wayfinder_mlx.py`.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 1024 2048 4096 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_base`

### EXP-20260207-QWEN-RETRO-CAUSAL (planned)
- Question: Does Hamiltonian-local retro with causal-only guard avoid the prior memory regression on Qwen?
- Hypothesis: With causal-only guard, retro inference overhead should be near baseline, avoiding the large memory blowup seen in unsafe retro.
- Change set: `retro_backfill_causal_only` in batched attention and Qwen integration.
- Command:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-1.7B-4bit --seq-lens 4096 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --full-swap --retro-backfill --retro-alpha 0.2 --retro-allow-inference --retro-causal-only --out-dir benchmarks/mlx/qwen3_1.7b_4bit_wayfinder/20260207_qwen_retro_causal`

### EXP-20260207-TOPOLOGY-FIRST-EXTRACTION
- Question: Can we extract graph construction into a first-class topology runtime without breaking MLX/Torch behavior?
- Hypothesis: Replacing per-module strategy builders with shared `Topology` should preserve outputs/tests and make graph injection possible.
- Change set:
  - `hcsa/topology/core.py` and `hcsa/topology/__init__.py`
  - `hcsa/mlx/attention.py` now uses `Topology` and accepts `topology_graph=` injection
  - `hcsa/torch/attention_wayfinder_sparse.py` now uses `Topology` and accepts `topology_graph=` injection
  - `hcsa/integrations/qwen_mlx.py` graph runtime now uses `Topology`
  - Added `tests/mlx/test_topology_first_class.py`
- Validation commands:
  - `python3 -m pytest tests/mlx/ -v`
  - `python3 -m pytest tests/pytorch -q`
- Key result:
  - MLX tests: `35 passed`
  - PyTorch tests: all passed (warning-only about unknown mark)
- Decision: Keep.

### EXP-20260207-TINY-BASELINE-TOPOLOGY-CHECK
- Question: After topology extraction, do we still match the original tiny benchmark pattern at long contexts?
- Hypothesis: Long-context crossover and memory reduction remain, with some short-context variance.
- Commands:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 4 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_readme_repro_topology`
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_readme_repro_topology_rerun`
- Key result (rerun with higher iters):
  - `T=2048` ratio `1.078x`, memory reduction `35.40%`
  - `T=4096` ratio `2.196x`, memory reduction `66.41%`
- Decision: Keep topology extraction; tiny long-context gains are preserved.

### EXP-20260207-TINY-BASELINE-GATE-REPRO (planned)
- Question: What is the exact current Tiny baseline under the original apples-to-apples gate settings?
- Hypothesis: Current topology-first Tiny path will remain below historical throughput/memory targets at T=2048/4096, validating the need to move the core path to batched permute.
- Change set: none (measurement-only, pre-optimization).
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_baseline_gate_repro`

### EXP-20260207-TINY-BATCHED-PERMUTE-GATE (planned)
- Question: Does routing Tiny `wayfinder_permute` through chunked batched permute restore/beat historical long-context throughput without regressing memory?
- Hypothesis: Batched path removes per-head Python-loop overhead and should increase T=2048/4096 throughput while preserving memory reduction.
- Change set:
  - `hcsa/mlx/attention.py`: `WayfinderAttentionMLX` permute path uses batched kernel
  - `hcsa/mlx/model.py`: add first-class retro config fields and wire to attention
  - tests: add model-path retro routing coverage
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate`

### EXP-20260207-TINY-LONG-RETRO-CONTROL (planned)
- Question: On tiny-long training, what are quality/throughput/memory with retro explicitly disabled?
- Hypothesis: Control run defines baseline quality and speed with strict causal defaults.
- Change set: none (configuration-only run).
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled false --retro-backfill-alpha 0.0 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_control`

### EXP-20260207-TINY-LONG-RETRO-TREATMENT (planned)
- Question: Does training-time retrocausal backfill improve tiny-long validation perplexity at comparable compute?
- Hypothesis: Enabling retro during training with causal guard improves or stabilizes val ppl; any inference/runtime cost should remain opt-in because defaults stay training-only.
- Change set: same code as control; retro toggled on for training.
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled true --retro-backfill-alpha 0.2 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_treatment`

## 2026-02-07 — Tiny Gate + Retro Validation (MLX 0.30.6)

Environment:
- `python3 -m pip show mlx mlx-metal` -> `mlx==0.30.6`, `mlx-metal==0.30.6`

### EXP-20260207-TINY-BASELINE-GATE-REPRO
- Question: What is the exact current Tiny baseline under the original apples-to-apples gate settings?
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_baseline_gate_repro`
- Baseline result (long-context focus):
  - `T=2048`: ratio `1.1825x`, memory reduction `41.57%`
  - `T=4096`: ratio `2.4196x`, memory reduction `68.77%`
- Decision: Keep as pre-optimization baseline.

### EXP-20260207-TINY-BATCHED-PERMUTE-GATE
- Question: Does routing Tiny `wayfinder_permute` through chunked batched permute restore/beat historical long-context throughput without regressing memory?
- Change set:
  - `hcsa/mlx/attention.py`: `WayfinderAttentionMLX` permute path now calls batched kernel
  - `hcsa/mlx/attention.py`: cache now stores stacked `all_perms` / `all_inv_perms`
  - `hcsa/mlx/model.py`: config-level retro controls wired into attention
  - `tests/mlx/test_wayfinder_attention_retro_path.py`: model-path retro coverage
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306`
- Result:
  - `T=2048`: ratio `4.2820x`, memory reduction `78.92%`
  - `T=4096`: ratio `9.2438x`, memory reduction `89.68%`
- Delta vs baseline (`20260207_tiny_baseline_gate_repro`):
  - `T=2048`: ratio `+3.0995x` (`+262.10%`), memory reduction `+37.35 pp`
  - `T=4096`: ratio `+6.8243x` (`+282.05%`), memory reduction `+20.91 pp`
- Delta vs historical targets (`1.19x/37%` at 2048, `2.45x/67%` at 4096):
  - `T=2048`: ratio `+3.0920x` (`+259.83%`), memory reduction `+41.92 pp`
  - `T=4096`: ratio `+6.7938x` (`+277.30%`), memory reduction `+22.68 pp`
- Decision: Keep.

### EXP-20260207-TINY-LONG-RETRO-CONTROL
- Question: On tiny-long training, what are quality/throughput/memory with retro explicitly disabled?
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled false --retro-backfill-alpha 0.0 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_control`
- Wayfinder result:
  - `val ppl=91.0188`
  - `avg tok/s=236,987.47`
  - `peak memory=120,135,536`
- Decision: Keep as control.

### EXP-20260207-TINY-LONG-RETRO-TREATMENT
- Question: Does training-time retrocausal backfill improve tiny-long validation perplexity at comparable compute?
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled true --retro-backfill-alpha 0.2 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_treatment`
- Wayfinder result:
  - `val ppl=79.9204`
  - `avg tok/s=218,823.10`
  - `peak memory=121,956,224`
- Delta vs control:
  - `val ppl`: `-11.0983` (better, `-12.19%`)
  - `avg tok/s`: `-18,164.37` (`-7.66%`)
  - `peak memory`: `+1,820,688` (`+1.52%`)
- Decision: Keep retro as a training-time feature (default remains inference-safe off).

### EXP-20260207-TINY-GATE-SDPA-DENSE (planned)
- Question: After switching dense baseline to fused SDPA/Flash-style kernel, how do Tiny gate metrics change?
- Hypothesis: Throughput ratio and memory reduction percentages will drop relative to naive dense baseline, but long-context throughput should remain >1x.
- Change set:
  - `hcsa/mlx/attention.py`: dense path uses `mx.fast.scaled_dot_product_attention` when weights are not requested
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306_sdpa_dense`

### EXP-20260207-QWEN4B-SDPA-REALITY-SLICE (planned)
- Question: At real model scale (Qwen3-4B), does long-context memory trend still hold under the current SDPA-based dense baseline?
- Hypothesis: Wayfinder should retain memory benefit trend at longer context, though throughput crossover may remain sequence-dependent.
- Change set: measurement-only.
- Command:
  - `python3 scripts/bench_qwen_wayfinder_mlx.py --model-path mlx-community/Qwen3-4B-4bit --seq-lens 2048 4096 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --out-dir benchmarks/mlx/qwen3_4b_wayfinder/20260207_qwen4b_sdpa_reality_slice`

## 2026-02-07 — Baseline Fairness Correction (SDPA Dense) + Ordered Progression

Execution order confirmed:
1. Tiny (complete)
2. GPT-2 (next)
3. Qwen3-1.7B (after GPT-2)
4. Qwen3-4B (deferred until prior two are complete)

### EXP-20260207-TINY-GATE-SDPA-DENSE
- Question: After switching dense baseline to fused SDPA/Flash-style kernel, how do Tiny gate metrics change?
- Change set:
  - `hcsa/mlx/attention.py`: `dense_causal_attention` now uses `mx.fast.scaled_dot_product_attention(..., mask="causal")` when `return_weights=False`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_mlx_wayfinder_scale.py --seq-lens 256 512 1024 2048 4096 --batch 2 --heads 4 --embd 128 --window 32 --landmark-stride 32 --warmup 2 --iters 6 --out-dir benchmarks/mlx/tiny_wayfinder/20260207_tiny_batched_gate_mlx306_sdpa_dense`
- Result:
  - `T=2048`: ratio `1.5720x`, memory reduction `26.53%`
  - `T=4096`: ratio `3.2198x`, memory reduction `62.17%`
- Delta vs historical targets (`1.19x/37%` at 2048, `2.45x/67%` at 4096):
  - `T=2048`: ratio `+0.3820x` (`+32.10%`), memory reduction `-10.47 pp`
  - `T=4096`: ratio `+0.7698x` (`+31.42%`), memory reduction `-4.83 pp`
- Interpretation:
  - Prior memory percentages were inflated by naive dense materialization.
  - Throughput crossover still holds at long context under fair dense SDPA baseline.
- Decision: Keep SDPA dense baseline as default comparison path.

### EXP-20260207-TINY-LONG-RETRO-CONTROL-SDPA-DENSE
- Question: With fair SDPA dense baseline, does tiny-long control remain stable?
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled false --retro-backfill-alpha 0.0 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_control_sdpa_dense`
- Wayfinder result:
  - `val ppl=91.4865`
  - `avg tok/s=208,759.37`
  - `peak memory=120,135,536`
- Decision: Keep as SDPA-aligned control.

### EXP-20260207-TINY-LONG-RETRO-TREATMENT-SDPA-DENSE
- Question: Under the same SDPA-aligned setup, does retro still improve quality?
- Command:
  - `PYTHONPATH=. python3 scripts/run_mlx_experiment_tiny_long.py --wayfinder-attn wayfinder_permute --retro-backfill-enabled true --retro-backfill-alpha 0.2 --retro-backfill-training-only true --retro-backfill-causal-only true --out-dir runs/mlx/tiny_long/20260207_retro_treatment_sdpa_dense`
- Wayfinder result:
  - `val ppl=79.6804`
  - `avg tok/s=194,631.02`
  - `peak memory=121,972,608`
- Delta vs SDPA control:
  - `val ppl`: `-11.8061` (`-12.90%`, better)
  - `avg tok/s`: `-14,128.35` (`-6.77%`)
  - `peak memory`: `+1,837,072` (`+1.53%`)
- Decision: Keep retro as training-only feature; inference default remains retro-off.

### EXP-20260207-QWEN4B-SDPA-REALITY-SLICE
- Decision update: deferred by protocol order.
- Rationale: Do not run Qwen until GPT-2 then Qwen3-1.7B stages are complete with logged results.

## 2026-02-07 — GPT-2 Kernel Push (Pre-Permute Adaptive Path)

### EXP-20260207-GPT2-PREPERM-ADAPTIVE (planned)
- Question: Can we improve GPT-2 Wayfinder throughput (T=2048/4096/8192) without adding model-specific hacks by using a core adaptive pre-permute path in batched permute attention?
- Hypothesis: If we pre-permute Q/K/V once per head-chunk (instead of gather-per-chunk), we reduce gather overhead enough to materially raise throughput, while keeping memory bounded via head chunking.
- Change set:
  - `hcsa/mlx/attention.py`: add adaptive pre-permute mode to `wayfinder_permute_window_attention_batched`
  - `hcsa/integrations/gpt2_mlx.py`: expose pre-permute mode via config and pass through
  - `scripts/bench_gpt2_wayfinder_mlx.py`: add CLI flags for chunk and pre-permute controls
- Baseline artifacts:
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_base_sdpa_steady_v4/results.json`
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_8192_v4/results.json`
- Command (post-change):
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_adaptive_v1`

### EXP-20260207-GPT2-PREPERM-KV-AUTO (planned)
- Question: Can we keep most of the throughput gain while reducing attention-path memory overhead by pre-permuting only K/V (not Q)?
- Hypothesis: KV-only pre-permute should preserve long-context speedup from reduced repeated K/V gathers, while cutting peak memory relative to full QKV pre-permute.
- Change set:
  - `hcsa/mlx/attention.py`: split pre-permute behavior into KV-only vs QKV
  - `hcsa/integrations/gpt2_mlx.py`: allow pre-permute mode values `auto/off/kv/qkv`
  - `scripts/bench_gpt2_wayfinder_mlx.py`: expose `kv` and `qkv` choices
- Baseline artifact:
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_adaptive_v1/results.json`
- Command (post-change):
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_kv_auto_v2`

### EXP-20260207-GPT2-STREAM-ACCUM-INFER (planned)
- Question: Can we reduce GPT-2 Wayfinder peak memory without hardcoded model branches by forcing streaming y accumulation in inference for all permute plans?
- Hypothesis: Eliminating chunk-list accumulation during inference removes avoidable peak buffers and improves memory while preserving current throughput trend.
- Change set:
  - `hcsa/mlx/attention.py`: inference path always uses streaming y accumulation (`y_pi.at[..., s:e].add`) even with pre-permute enabled
- Baseline artifact:
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_kv_auto_v2/results.json`
- Command (post-change):
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_stream_accum_v3`

### EXP-20260207-GPT2-AUTO-PLANNER-BUDGET1 (planned)
- Question: Can a non-hardcoded auto planner choose permute execution mode from math (movement + peak budget) and improve GPT-2 Pareto behavior?
- Hypothesis: With budget tied to dense peak memory (multiplier=1.0), planner should avoid memory-heavy modes at shorter lengths while still selecting faster modes when feasible.
- Change set:
  - `hcsa/mlx/attention.py`: auto planner for pre-permute mode (`off/kv/qkv`) using closed-form movement and peak-byte estimates, optional budget constraint
  - `hcsa/integrations/gpt2_mlx.py`: pass optional memory budget through runtime controls
  - `scripts/bench_gpt2_wayfinder_mlx.py`: set per-seq budget from measured dense memory (`--permute-memory-budget-multiplier`)
- Baseline artifacts:
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_kv_auto_v2/results.json`
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_base_sdpa_steady_v4/results.json`
  - `benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_8192_v4/results.json`
- Command (post-change):
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_budget1_v4`

### EXP-20260207-GPT2-AUTO-PLANNER-BUDGET-SWEEP (planned)
- Question: What memory-budget multiplier yields the best GPT-2 Pareto point (throughput vs memory) under the non-hardcoded planner?
- Hypothesis: Lower multipliers should force memory-safer plans at 2048/4096 with throughput tradeoff; a middle multiplier should improve 4096/8192 throughput while avoiding worst memory regressions.
- Change set: none (measurement sweep on planner controls).
- Command:
  - `for m in 0.50 0.70 0.85 1.00; do PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 1 --iters 3 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier $m --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_sweep_m${m}; done`

- Sweep extension: budget multiplier `0.10` to test whether planner constraints become active.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 1 --iters 3 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 0.10 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_sweep_m0.10`

- Sweep extension 2: budget multipliers `0.20 0.30 0.40` to locate planner phase transition.
- Command:
  - `for m in 0.20 0.30 0.40; do PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 1 --iters 3 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier $m --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_sweep_m${m}; done`

### EXP-20260207-GPT2-AUTO-PLANNER-REFINE-M010-M020 (planned)
- Question: Under stable settings (warmup=2, iters=4), which budget multiplier is the better default: 0.10 (memory-first) or 0.20 (balanced)?
- Hypothesis: `0.20` may retain >1x at 8192 while reducing memory regression vs throughput-only settings; `0.10` should remain memory-safer but slower.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 0.10 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_refine_m010`
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 0.20 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_refine_m020`

- Boundary check: budget multiplier `0.15` (stable settings) to test compromise between `0.10` and `0.20`.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 0.15 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_refine_m015`

### EXP-20260207-GPT2-PREPERM-ADAPTIVE (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_adaptive_v1`
- Result:
  - `T=2048`: attn ratio `0.394x`, attn mem reduction `-48.07%`
  - `T=4096`: attn ratio `0.862x`, attn mem reduction `-43.62%`
  - `T=8192`: attn ratio `1.618x`, attn mem reduction `-46.21%`
- Delta vs baseline (`.../20260207_gpt2_base_sdpa_steady_v4` + `.../20260207_gpt2_8192_v4`):
  - Throughput improved strongly, memory regressed materially.
- Decision: Follow-up (keep idea, not final default).

### EXP-20260207-GPT2-PREPERM-KV-AUTO (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_preperm_kv_auto_v2`
- Result:
  - `T=2048`: attn ratio `0.497x`, attn mem reduction `-48.07%`
  - `T=4096`: attn ratio `0.772x`, attn mem reduction `-43.62%`
  - `T=8192`: attn ratio `1.662x`, attn mem reduction `-46.22%`
- Decision: Follow-up (best raw long-context throughput, memory too costly).

### EXP-20260207-GPT2-STREAM-ACCUM-INFER (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_stream_accum_v3`
- Result:
  - `T=2048`: attn ratio `0.306x`, attn mem reduction `-14.62%`
  - `T=4096`: attn ratio `0.422x`, attn mem reduction `-10.81%`
  - `T=8192`: attn ratio `0.776x`, attn mem reduction `-10.94%`
- Decision: Reject as default (memory improved but throughput collapsed).

### EXP-20260207-GPT2-AUTO-PLANNER-BUDGET1 (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_budget1_v4`
- Result:
  - `T=2048`: attn ratio `0.548x`, attn mem reduction `-48.07%`
  - `T=4096`: attn ratio `0.856x`, attn mem reduction `-40.93%`
  - `T=8192`: attn ratio `1.603x`, attn mem reduction `-46.22%`
- Decision: Keep planner infrastructure; budget=1.0 is throughput-first.

### EXP-20260207-GPT2-AUTO-PLANNER-REFINE-M010-M020 (result)
- `m=0.10` (`.../20260207_gpt2_auto_planner_refine_m010`):
  - `T=2048`: `0.312x`, `-22.53%`
  - `T=4096`: `0.586x`, `-21.67%`
  - `T=8192`: `0.751x`, `-16.00%`
- `m=0.20` (`.../20260207_gpt2_auto_planner_refine_m020`):
  - `T=2048`: `0.464x`, `-24.77%`
  - `T=4096`: `0.877x`, `-43.62%`
  - `T=8192`: `1.068x`, `-22.23%`
- Decision: `m=0.20` is speed-favoring; `m=0.10` is memory-favoring.

### EXP-20260207-GPT2-AUTO-PLANNER-REFINE-M015 (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 0.15 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_gpt2_auto_planner_refine_m015`
- Result:
  - `T=2048`: attn ratio `0.417x`, attn mem reduction `-24.77%`
  - `T=4096`: attn ratio `0.601x`, attn mem reduction `-19.35%`
  - `T=8192`: attn ratio `1.068x`, attn mem reduction `-22.23%`
- Decision: Keep as balanced default candidate for GPT-2 stage (best current compromise).

### EXP-20260207-180248-NORTHSTAR-GPT2-BEFORE
- Question: What is the current GPT-2 Wayfinder/HCSA permute throughput ratio vs fused dense SDPA?
- Hypothesis: Throughput ratio grows with T but remains <1.0x at T=4096 due to per-iteration `mx.eval` sync overhead.
- Change set: reporting-only (`scripts/report_northstar.py`).
- Baseline result (attention level):
  - `T=2048`: ratio `0.704x`
  - `T=4096`: ratio `0.812x`
  - `T=8192`: ratio `1.186x`
  - `C_fit=39.08`, `T*=5042`
- Decision: Baseline locked. Root cause investigation next.

### EXP-20260207-NORTHSTAR-MICRO-PROFILE
- Question: Which term dominates constant C in R(T) ≈ T/(W·C)?
- Hypothesis: Per-query-chunk `mx.eval` GPU sync is >80% of total time.
- Result: **Confirmed.** `mx.eval` sync is 87–89% of total at all T values.
  - `T=2048`: eval_sync=4.6ms / total=5.3ms (87%)
  - `T=4096`: eval_sync=9.3ms / total=10.6ms (88%)
  - `T=8192`: eval_sync=15.9ms / total=17.8ms (89%)
  - Current path: 66 `mx.eval` calls at T=8192; lazy path: 2 calls.
  - Lazy path speedup: 2.67x–4.23x over current path.
- Decision: Root cause confirmed. Fix: eliminate per-query-chunk `mx.eval`.

### EXP-20260207-NORTHSTAR-GPT2-AFTER
- Question: After removing all internal `mx.eval` syncs (keeping 1 per head chunk), does GPT-2 meet the North Star gate?
- Hypothesis: Reducing sync count from ~66 to 2 should push T=4096 above 0.95x and T=8192 well above 1.0x.
- Change set:
  - `hcsa/mlx/attention.py`: remove `inference_fast_path` with per-chunk `mx.eval(y_pi)`
  - `hcsa/mlx/attention.py`: remove prepermute `mx.eval(q_pi_buf/k_pi_buf/v_pi_buf)`
  - `hcsa/mlx/attention.py`: remove retro `mx.eval(y_pi)`
  - All paths now use lazy chunk-list accumulation with single `mx.eval(y_h)` per head chunk.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_gpt2_wayfinder_mlx.py --model-path openai-community/gpt2 --seq-lens 2048 4096 8192 --batch 1 --warmup 4 --iters 8 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --permute-head-chunk-size 8 --permute-query-chunk-size 256 --out-dir benchmarks/mlx/gpt2_wayfinder/20260207_northstar_after_v3_stable`
- Result (attention level):
  - `T=2048`: ratio `0.481x` (dense baseline 23% faster in this run; see block level)
  - `T=4096`: ratio `0.979x` 
  - `T=8192`: ratio `1.747x` 
  - `C_fit=32.99`, `T*=4256` (before: C=39.08, T*=5042)
- Result (block level — fairer whole-layer comparison):
  - `T=2048`: ratio `0.951x`, mem reduction `4.2%` (WF uses *less* memory)
  - `T=4096`: ratio `1.010x`, mem reduction `4.8%`
  - `T=8192`: ratio `1.382x`, mem reduction `5.1%`
- Memory fairness note:
  - Attention-only WF mem is ~77% higher than dense (graph cache + permutation artifacts).
  - Block-level WF mem is 4–5% *lower* because graph overhead is amortized and WF avoids T×T intermediates.
  - Dense uses fused SDPA (~O(T) memory); WF uses O(T·W) chunked attention.
- Decision: **Keep.** All block-level gates pass. Attention-level T=4096 and T=8192 pass.
- Next action: Update README.md with fairness correction, then prepare commit.

### EXP-20260207-185738-GLM-INTEGRATION-SMOKE (planned)
- Question: Does a new GLM-4.7-Flash MLA integration (`hcsa/integrations/glm_mlx.py`) run correctly with Wayfinder swap while preserving retro-off inference defaults?
- Hypothesis: Reusing the Qwen graph runtime and padding MLA value latent dim to q/k dim only inside Wayfinder kernels will produce valid forwards, pass MLX tests, and allow a GLM benchmark smoke run without touching MoE routing.
- Change set:
  - `hcsa/integrations/glm_mlx.py` (new)
  - `hcsa/integrations/__init__.py` (exports)
  - `scripts/bench_glm_wayfinder_mlx.py` (new)
- Baseline: `benchmarks/mlx/gpt2_wayfinder/20260207_northstar_after_v3_stable/results.json` (current non-Qwen reference stage)
- Command:
  - `PYTHONPATH=. python3 -m pytest tests/mlx/ -x -q`
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 2048 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_integration_smoke`
- Controls:
  - Retro disabled by default (`--retro-backfill` omitted)
  - Batch=1, window=64, landmark_stride=64, path=permute
  - No MoE gating modifications

### EXP-20260207-190227-GLM-LONG-SMOKE (planned)
- Question: On GLM-4.7-Flash MLA, does Wayfinder permute sustain throughput > dense at long context (T=8192) with retro disabled?
- Hypothesis: At T=8192, Wayfinder attention and block throughput should exceed dense baseline (>1.0x) with reduced peak memory because compute scales with local window instead of full T².
- Change set:
  - `hcsa/integrations/glm_mlx.py`
  - `scripts/bench_glm_wayfinder_mlx.py`
  - `hcsa/mlx/attention.py` (`v.itemsize` planner bugfix)
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_integration_smoke/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 4096 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_long_smoke`
- Controls:
  - Retro disabled by default (`--retro-backfill` omitted)
  - Batch=1, window=64, landmark_stride=64, same model/checkpoint

### EXP-20260207-185738-GLM-INTEGRATION-SMOKE (result)
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_integration_smoke/results.json` (dense baseline metrics in-row)
- Command:
  - `PYTHONPATH=. python3 -m pytest tests/mlx/ -x -q`
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 2048 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_integration_smoke`
- Validation:
  - `tests/mlx`: `38 passed`
- Result (T=2048):
  - Attention throughput: absolute `75,061.04 tok/s`; delta vs dense `+13,751.40 tok/s`; percentage delta `+22.43%`
  - Attention peak memory: absolute `209,706,682 B`; delta vs dense `-216,525,682 B`; percentage delta `-50.80%`
  - Attention memory reduction sign convention: `100 * (1 - wayfinder/dense) = 50.80%`
  - Block throughput: absolute `37,291.45 tok/s`; delta vs dense `+4,334.74 tok/s`; percentage delta `+13.15%`
  - Block peak memory: absolute `377,596,464 B`; delta vs dense `-227,295,244 B`; percentage delta `-37.58%`
  - Block memory reduction sign convention: `100 * (1 - wayfinder/dense) = 37.58%`
  - Sanity MAE: `0.003798`
- Decision: Keep.
- Next action: Extend to long-context checkpoints (4096, 8192+) and verify acceptance criteria.

### EXP-20260207-190227-GLM-LONG-SMOKE (result)
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_long_smoke/results.json` (dense baseline metrics in-row)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 4096 8192 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_long_smoke`
- Result (T=4096):
  - Attention throughput: absolute `83,344.28 tok/s`; delta vs dense `+59,466.79 tok/s`; percentage delta `+249.05%`
  - Attention peak memory: absolute `407,434,558 B`; delta vs dense `-776,672,494 B`; percentage delta `-65.59%`
  - Attention memory reduction sign convention: `100 * (1 - wayfinder/dense) = 65.59%`
  - Block throughput: absolute `38,374.73 tok/s`; delta vs dense `+14,785.72 tok/s`; percentage delta `+62.68%`
  - Block peak memory: absolute `707,504,688 B`; delta vs dense `-749,240,324 B`; percentage delta `-51.43%`
  - Block memory reduction sign convention: `100 * (1 - wayfinder/dense) = 51.43%`
  - Sanity MAE: `0.004029`
- Result (T=8192):
  - Attention throughput: absolute `87,598.27 tok/s`; delta vs dense `+68,243.02 tok/s`; percentage delta `+352.58%`
  - Attention peak memory: absolute `831,760,944 B`; delta vs dense `-2,924,019,708 B`; percentage delta `-77.85%`
  - Attention memory reduction sign convention: `100 * (1 - wayfinder/dense) = 77.85%`
  - Block throughput: absolute `38,596.26 tok/s`; delta vs dense `+23,018.00 tok/s`; percentage delta `+147.76%`
  - Block peak memory: absolute `1,368,631,860 B`; delta vs dense `-2,572,156,930 B`; percentage delta `-65.27%`
  - Block memory reduction sign convention: `100 * (1 - wayfinder/dense) = 65.27%`
  - Sanity MAE: `0.003984`
- Decision: Keep.
- Next action: Run full sequence set (2048/4096/8192/16384/32768) with stable warmup/iters and update README measured GLM integration results.

### EXP-20260207-190427-GLM-FULL-SWAP-SMOKE (planned)
- Question: Can all GLM transformer `self_attn` layers be swapped to Wayfinder attention without model-forward breakage?
- Hypothesis: `swap_glm_attention_with_wayfinder()` should replace all attention layers and a short-token full forward should succeed with retro disabled.
- Change set:
  - `hcsa/integrations/glm_mlx.py`
  - `scripts/bench_glm_wayfinder_mlx.py`
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_integration_smoke/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 1024 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --full-swap --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_full_swap_smoke`
- Controls:
  - Retro disabled by default
  - Full swap enabled

### EXP-20260207-190617-GLM-STABLE-MEDIAN (planned)
- Question: Do GLM Wayfinder gains persist under stricter benchmarking (median of multiple iterations)?
- Hypothesis: Compared with dense baseline, Wayfinder remains >1.0x at long context (4096, 8192) with material memory reduction under the same runtime controls.
- Change set:
  - No new code; measurement-only stability run on current integration.
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_long_smoke/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_stable_median`
- Controls:
  - Retro disabled (no `--retro-backfill`)
  - Batch=1, window=64, landmark_stride=64, same model/checkpoint

### EXP-20260207-190427-GLM-FULL-SWAP-SMOKE (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 1024 --batch 1 --warmup 1 --iters 1 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --full-swap --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_full_swap_smoke`
- Result:
  - Full swap replaced layers: `47`
  - Full-model smoke (`seq_len=256`): `11.80 tok/s`, peak memory `17,016,685,356 B`
  - Layer-level `T=1024` block ratio: `0.872x`; block memory reduction: `24.94%`
- Decision: Keep (functional full-swap success).
- Next action: evaluate full-model throughput at larger token counts separately from layer-level kernels.

### EXP-20260207-190617-GLM-STABLE-MEDIAN (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_wayfinder_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 2048 4096 8192 --batch 1 --warmup 2 --iters 4 --path permute --window 64 --landmark-stride 64 --seed 42 --permute-prepermute-mode auto --permute-memory-budget-multiplier 1.0 --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_stable_median`
- Baseline path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_long_smoke/results.json`
- Result (T=2048):
  - Attention throughput: absolute `85,844.00 tok/s`; delta vs dense `+14,392.32`; percentage delta `+20.14%`
  - Attention memory: absolute `209,706,682 B`; delta vs dense `-216,525,682 B`; percentage delta `-50.80%`
  - Memory reduction formula: `100 * (1 - wayfinder/dense) = 50.80%`
  - Block throughput: absolute `39,499.70 tok/s`; delta vs dense `+3,116.90`; percentage delta `+8.57%`
  - Block memory: absolute `377,596,464 B`; delta vs dense `-227,295,244 B`; percentage delta `-37.58%`
  - Block memory reduction formula: `100 * (1 - wayfinder/dense) = 37.58%`
- Result (T=4096):
  - Attention throughput: absolute `87,843.27 tok/s`; delta vs dense `+43,791.28`; percentage delta `+99.41%`
  - Attention memory: absolute `443,487,550 B`; delta vs dense `-776,672,494 B`; percentage delta `-63.65%`
  - Memory reduction formula: `100 * (1 - wayfinder/dense) = 63.65%`
  - Block throughput: absolute `40,480.41 tok/s`; delta vs dense `+14,078.97`; percentage delta `+53.33%`
  - Block memory: absolute `708,160,048 B`; delta vs dense `-749,240,324 B`; percentage delta `-51.41%`
  - Block memory reduction formula: `100 * (1 - wayfinder/dense) = 51.41%`
- Result (T=8192):
  - Attention throughput: absolute `94,958.59 tok/s`; delta vs dense `+70,452.17`; percentage delta `+287.48%`
  - Attention memory: absolute `832,416,304 B`; delta vs dense `-2,924,019,708 B`; percentage delta `-77.84%`
  - Memory reduction formula: `100 * (1 - wayfinder/dense) = 77.84%`
  - Block throughput: absolute `40,887.71 tok/s`; delta vs dense `+23,319.77`; percentage delta `+132.74%`
  - Block memory: absolute `1,369,287,220 B`; delta vs dense `-2,572,156,930 B`; percentage delta `-65.26%`
  - Block memory reduction formula: `100 * (1 - wayfinder/dense) = 65.26%`
- Decision: Keep.
- Next action: run full-seq sweep (16384/32768) with stable iters and publish measured GLM section.

### EXP-20260207-190908-GLM-MAXCTX-PROBE (planned)
- Question: Can GLM Wayfinder run up to the model max context (`202,752`) on this machine, and what are latency/memory trends as T scales?
- Hypothesis: Wayfinder layer-level prefill will scale beyond 8192 and may reach 202,752 with significant runtime but without dense-baseline feasibility; peak memory should grow sub-quadratically relative to dense attention.
- Change set:
  - Measurement only (no code edits).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_glm_stable_median/results.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... progressive layer-level GLMWayfinderAttention probe for T in [16384, 32768, 65536, 98304, 131072, 163840, 202752] ... PY`
- Controls:
  - Retro disabled
  - Batch=1, path=permute, window=64, landmark_stride=64
  - Single layer (`model.layers[0].self_attn`) to isolate attention path

### EXP-20260207-190908-GLM-MAXCTX-PROBE (result)
- Command:
  - Progressive probe with rungs `16384 -> 32768 -> 65536 -> 98304 -> 131072 -> 163840 -> 202752` under per-rung timeouts.
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_191444_maxctx_wayfinder_layer_probe/summary.json`
- Result:
  - Success through `T=131072`
  - `T=163840` failed with allocator error:
    - `RuntimeError: [metal::malloc] Attempting to allocate 34419507200 bytes which is greater than the maximum allowed buffer size of 22613000192 bytes.`
  - Best successful rung metrics (`T=131072`):
    - `sec=885.48`, `tok/s=148.02`, `peak_memory_bytes=27,773,636,722`
- Decision: Follow-up.
- Next action: retry high-context probe with reduced landmark memory footprint (`landmark_stride=None`) to continue toward `202752`.

### EXP-20260207-200906-GLM-MAXCTX-NOLANDMARK (planned)
- Question: Can we reach `202,752` context if landmark edges are disabled to reduce graph memory pressure?
- Hypothesis: `landmark_stride=None` will reduce graph-ABI memory enough to pass beyond `163,840` and potentially hit `202,752` on the same hardware.
- Change set:
  - Measurement only (runtime flag change: `landmark_stride=None`).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_191444_maxctx_wayfinder_layer_probe/summary.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... progressive no-landmark probe for T in [163840, 202752], timeout 90m/120m ... PY`
- Controls:
  - Retro disabled
  - Batch=1, path=permute, window=64
  - Single layer (`model.layers[0].self_attn`)
  - Only landmark stride changes (`None`)

### EXP-20260207-201256-GLM-FULLMODEL-PREFILL-KV-CHECKPOINT (planned)
- Question: At full-model scale, does `prefill + 1 token` show KV-cache-dominated growth versus `prefill-only`?
- Hypothesis: Wayfinder controls attention workspace, but `prefill+1` peak memory will exceed `prefill-only` and the gap should increase with larger prefill lengths due to KV retention.
- Change set:
  - Measurement only (no code edits).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_200939_maxctx_nolandmark_probe/summary.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... full-model Wayfinder swap, scenario A prefill-only vs scenario B prefill+1 at T=[32768,65536] ... PY`
- Controls:
  - Retro disabled
  - Full attention swap across all layers
  - `landmark_stride=None` (to avoid graph-ABI allocator wall)
  - Same random-token input per scenario per T

### EXP-20260207-202313-GLM-NOLANDMARK-DELTA131072 (planned)
- Question: At fixed `T=131072`, how much does `landmark_stride=None` change runtime and peak memory vs default `landmark_stride=64`?
- Hypothesis: Disabling landmark edges will materially reduce graph build cost and peak memory at identical sequence length.
- Change set:
  - Measurement only.
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_191444_maxctx_wayfinder_layer_probe/summary.json` (default stride=64, T=131072)
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... single-layer Wayfinder probe T=131072, landmark_stride=None ... PY`
- Controls:
  - Retro disabled
  - Batch=1, path=permute, window=64, same model and attention layer

### EXP-20260207-200906-GLM-MAXCTX-NOLANDMARK (result)
- Command:
  - No-landmark probe at `T=163840`, `T=202752` with long timeouts.
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_200939_maxctx_nolandmark_probe/summary.json`
- Baseline path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_191444_maxctx_wayfinder_layer_probe/summary.json`
- Result:
  - `T=163840`: success, `sec=70.44`, `tok/s=2325.80`, `peak_memory_bytes=15,667,546,668`
  - `T=202752`: success, `sec=88.06`, `tok/s=2302.46`, `peak_memory_bytes=19,386,010,156`
- Baseline comparison discipline:
  - At `T=163840`, baseline run failed (allocator limit), so absolute delta/% vs baseline throughput/memory are not defined.
  - At `T=202752`, baseline has no successful run, so absolute delta/% are not defined.
- Decision: Keep.
- Next action: run fixed-`T` comparison (`T=131072`) to quantify delta vs default stride.

### EXP-20260207-202313-GLM-NOLANDMARK-DELTA131072 (result)
- Command:
  - Single-layer probe `T=131072` with `landmark_stride=None`.
- Baseline path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_191444_maxctx_wayfinder_layer_probe/summary.json` (`landmark_stride=64` at `T=131072`)
- Result (`landmark_stride=None` at `T=131072`):
  - Absolute runtime: `55.05 s`
  - Absolute throughput: `2381.11 tok/s`
  - Absolute peak memory: `12,536,515,116 B`
- Delta vs baseline (`landmark_stride=64`, `T=131072`):
  - Runtime delta: `-830.43 s` (`-93.78%`)
  - Throughput delta: `+2233.09 tok/s` (`+1508.60%`)
  - Peak memory delta: `-15,237,121,606 B` (`-54.86%`)
  - Memory reduction sign convention: `100 * (1 - wayfinder_nolandmark/wayfinder_default) = 54.86%`
- Decision: Keep for max-context probing mode.
- Next action: gate this as a benchmark/runtime option and characterize quality impact of removing landmarks.

### EXP-20260207-201256-GLM-FULLMODEL-PREFILL-KV-CHECKPOINT (result)
- Command:
  - Full-model Wayfinder swap (`47` layers), scenario A `prefill-only` vs scenario B `prefill+1` at `T=32768` and `T=65536`.
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Result (`T=32768`):
  - Prefill-only: `61.29 s`, `534.66 tok/s`, peak `28,928,288,404 B`
  - Prefill+1: total `106.03 s` (prefill `85.62 s` + decode1 `20.41 s`), peak `29,053,543,348 B`
  - Delta (`prefill+1` vs `prefill-only`):
    - Peak memory: `+125,254,944 B` (`+0.433%`)
- Result (`T=65536`):
  - Prefill-only: `126.69 s`, `517.30 tok/s`, peak `44,890,891,940 B`
  - Prefill+1: total `227.30 s` (prefill `140.66 s` + decode1 `86.64 s`), peak `44,890,891,940 B`
  - Delta (`prefill+1` vs `prefill-only`):
    - Peak memory: `0 B` (`0.0%`)
- Additional scaling check (`prefill-only`, `65536` vs `32768`):
  - Peak memory delta: `+15,962,603,536 B` (`+55.18%`)
  - Throughput delta: `-17.36 tok/s` (`-3.25%`)
- Decision: Follow-up.
- Next action: isolate KV contribution with cache instrumentation (cache tensor bytes per layer) and compare with/without KV quantization/paging strategies.

### EXP-20260207-230429-GLM-FULLMODEL-KVQ4-CHECKPOINT (planned)
- Question: Does quantized KV cache (4-bit) reduce full-model peak memory for prefill-only and prefill+1 scenarios at long context?
- Hypothesis: KV quantization (`bits=4`, `group_size=64`) should reduce peak memory versus non-quantized checkpoint at `T=32768/65536`, with some throughput penalty.
- Change set:
  - Measurement only (runtime cache mode change).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... full-model prefill-only vs prefill+1 with make_prompt_cache(...)->to_quantized(bits=4,group_size=64) at T=[32768,65536] ... PY`
- Controls:
  - Retro disabled
  - Full attention swap across all layers
  - `landmark_stride=None`
  - Same random-token generation procedure as baseline checkpoint

### EXP-20260207-230429-GLM-FULLMODEL-KVQ4-CHECKPOINT (result)
- Command:
  - Full-model Wayfinder swap with quantized prompt cache (`to_quantized(bits=4, group_size=64)`).
- Result path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_230511_fullmodel_prefill_kvq4_checkpoint/results.json`
- Result:
  - Failed at first rung (`T=32768`) with:
    - `TypeError: tuple indices must be integers or slices, not tuple`
  - Root cause:
    - `QuantizedKVCache.update_and_fetch()` returns quantized tuple trees (packed weights/scales/biases) while GLM MLA extraction path slices dense key tensors (`keys[..., :-qk_rope_head_dim]`).
- Decision: Follow-up (integration gap).
- Next action: test rotating/paged KV strategy (`max_kv_size`) which remains dense-array-compatible with current MLA extraction path.

### EXP-20260207-230611-GLM-FULLMODEL-ROTATINGKV-CHECKPOINT (planned)
- Question: Can rotating KV cache (`max_kv_size`) cap peak memory during full-model `prefill+1` while preserving operational correctness?
- Hypothesis: Setting `max_kv_size=8192` will reduce `prefill+1` peak memory versus unbounded KV baseline at `T=32768/65536` with some quality/context tradeoff.
- Change set:
  - Measurement only (cache mode: `make_prompt_cache(model, max_kv_size=8192)`).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... full-model prefill-only vs prefill+1 with rotating KV max_kv_size=8192 at T=[32768,65536] ... PY`
- Controls:
  - Retro disabled
  - Full attention swap across all layers
  - `landmark_stride=None`
  - Same random-token generation procedure as baseline checkpoint

### EXP-20260207-231759-GLM-DECODE64-KV-COMPARISON (planned)
- Question: Under full-model Wayfinder at `T=32768`, how does KV policy affect memory/time for `prefill + 64 decode tokens`?
- Hypothesis: Rotating KV (`max_kv_size=8192`) should lower peak memory versus unbounded KV when decode length extends beyond a single token; throughput may drop due to cache management overhead.
- Change set:
  - Measurement only.
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 - <<'PY' ... full-model compare cache_mode in {normal,rotating_8192} for prefill + decode64 at T=32768 ... PY`
- Controls:
  - Retro disabled
  - Full attention swap across all layers
  - `landmark_stride=None`
  - Same model and random-token input generation pattern

### EXP-20260207-233428-GLM-CHUNKED-PREFILL-SWEEP (planned)
- Question: Does full-model GLM Wayfinder with chunked prefill reduce long-context peak memory (activation wall) versus the monolithic prefill baseline while keeping usable throughput?
- Hypothesis: Chunking prefill into `4096/8192/16384` with persistent prompt cache will materially reduce peak memory at fixed `T` (`32768/65536/131072`), and at least one `T>=65536` run will complete with lower peak memory than baseline.
- Change set:
  - `scripts/bench_glm_chunked_prefill_mlx.py` (new full-model chunked prefill harness).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 65536 131072 --chunk-sizes 4096 8192 16384 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_233428_chunked_prefill_fullmatrix`
- Controls:
  - Retro/backfill disabled (`retro_backfill_enabled=false`, `alpha=0.0`, training-only guard on)
  - Full attention swap on all GLM layers
  - Same model, path, window, and baseline-comparison discipline
  - Random token prefill inputs with fixed seed (`42`)

### EXP-20260208-000124-GLM-CHUNK-SWEEP-32768 (planned)
- Question: At fixed `T=32768`, which chunk size (`4096/8192/16384`) gives the best memory-throughput tradeoff for full-model GLM chunked prefill?
- Hypothesis: Larger chunks (`8192` or `16384`) should improve latency while retaining meaningful peak-memory reduction versus monolithic baseline.
- Change set:
  - `scripts/bench_glm_chunked_prefill_mlx.py` cumulative prefill/decode execution path.
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 8192 16384 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_000124_chunk_sweep_32768`
- Controls:
  - Retro/backfill disabled
  - Full attention swap across all layers
  - Batch=1, seed=42, same model and inference path

### EXP-20260208-001331-GLM-LENGTH-SWEEP-CHUNK4096 (planned)
- Question: With chunk size fixed at `4096` (best memory point at `T=32768`), does chunked prefill preserve memory gains at longer contexts (`65536`, `131072`)?
- Hypothesis: `chunk_size=4096` will keep peak memory materially below monolithic baseline at `T=65536`, and at least one high-context run (`T=131072`) will complete without causality breakage.
- Change set:
  - Measurement only (reuse `scripts/bench_glm_chunked_prefill_mlx.py`).
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 65536 131072 --chunk-sizes 4096 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096`
- Controls:
  - Retro/backfill disabled
  - Full attention swap across all layers
  - Batch=1, seed=42, same model/inference path

### EXP-20260207-233428-GLM-CHUNKED-PREFILL-SWEEP (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 65536 131072 --chunk-sizes 4096 8192 16384 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_233428_chunked_prefill_fullmatrix`
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_233428_chunked_prefill_fullmatrix/results.json`
- Result:
  - Interrupted during `T=65536, chunk=4096, prefill_only` due prohibitive runtime for full Cartesian sweep in-session.
  - Completed checkpoint before interruption: `T=32768, chunk=4096`.
- Decision: Follow-up (split into targeted sweeps).
- Next action: run chunk-size sweep at fixed `T=32768` and separate long-context length sweep with selected chunk size.

### EXP-20260208-000124-GLM-CHUNK-SWEEP-32768 (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 8192 16384 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_000124_chunk_sweep_32768`
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_000124_chunk_sweep_32768/results.json`
- Baseline path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Result (`T=32768`):
  - `chunk=4096`:
    - Prefill-only absolute: `163.51 s`, `200.40 tok/s`, `26,021,806,692 B`
    - Delta vs baseline prefill-only: latency `+102.22 s` (`+166.79%`), tok/s `-334.26` (`-62.52%`), peak memory `-2,906,481,712 B` (`-10.05%`)
    - Memory reduction sign convention: `100 * (1 - wayfinder/baseline) = 10.05%`
    - Prefill+1 absolute: total `165.16 s`, peak `26,021,806,692 B`
    - Prefill+1 delta vs baseline: total `+59.13 s` (`+55.77%`), prefill tok/s `-182.29` (`-47.63%`), peak memory reduction `10.43%`
  - `chunk=8192`:
    - Prefill-only absolute: `231.86 s`, `141.32 tok/s`, `30,798,216,294 B`
    - Delta vs baseline prefill-only: latency `+278.32%`, tok/s `-73.57%`, peak memory `+6.46%` (worse)
  - `chunk=16384`:
    - Prefill-only absolute: `212.84 s`, `153.95 tok/s`, `41,954,965,882 B`
    - Delta vs baseline prefill-only: latency `+247.29%`, tok/s `-71.21%`, peak memory `+45.03%` (worse)
- Decision: Keep `chunk=4096` as memory-optimal candidate for long-context sweep.
- Next action: run length sweep (`32768/65536/131072`) at `chunk=4096`.

### EXP-20260208-001331-GLM-LENGTH-SWEEP-CHUNK4096 (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 65536 131072 --chunk-sizes 4096 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096`
- Results path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096/results.json`
- Progress path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096/progress.json`
- Baseline path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- Completed rows:
  - `T=32768`:
    - Prefill-only absolute: `163.33 s`, `200.62 tok/s`, `26,021,806,692 B`
    - Delta vs baseline prefill-only: latency `+102.04 s` (`+166.50%`), tok/s `-334.04` (`-62.48%`), peak memory reduction `10.05%`
    - Prefill+1 absolute: total `164.97 s`, peak `26,021,806,692 B`
    - Prefill+1 delta vs baseline: total `+58.94 s` (`+55.58%`), prefill tok/s `-182.07` (`-47.58%`), peak memory reduction `10.43%`
    - Prefill+64 absolute: total `165.53 s`, peak `26,021,806,692 B`
  - `T=65536`:
    - Prefill-only absolute: `876.31 s`, `74.79 tok/s`, `33,164,840,500 B`
    - Delta vs baseline prefill-only: latency `+749.62 s` (`+591.70%`), tok/s `-442.51` (`-85.54%`), peak memory `-11,726,051,440 B` (`-26.12%`)
    - Memory reduction sign convention: `100 * (1 - wayfinder/baseline) = 26.12%`
    - Prefill+1 absolute: total `903.32 s`, peak `33,164,840,500 B`
    - Prefill+1 delta vs baseline: total `+676.01 s` (`+297.41%`), prefill tok/s `-391.12` (`-83.95%`), peak memory reduction `26.12%`
    - Prefill+64 absolute: total `904.31 s`, peak `33,164,840,500 B`
- `T=131072` status:
  - Started (`prefill_only`) but interrupted due runtime before completion.
- Decision: Follow-up.
- Next action: run dedicated `T=131072` checkpoint (possibly with larger chunk or prefill-only first) to complete high-context acceptance.

## 2026-02-07 — Chunked Prefill Latency Regression Root-Cause Diagnosis

Key observation from prior sweep: at T=65536 chunk=4096, chunked prefill achieved 26.12% memory reduction but +591.70% latency regression (876s vs 127s baseline). The regression comes from 16 sequential forward passes through 47 layers. Each pass carries fixed overhead (HCSA pattern construction, MLX eval barriers, Python dispatch). The baseline doesn't chunk — comparing monolithic prefill (one call, GPU stays hot) against 16 sequential sparse calls with Python round-trips.

### EXP-20260207-DIAG01-CHUNKED-DENSE-BASELINE (planned)
- Question: How much of the 6x chunked prefill latency regression is from chunking overhead itself vs HCSA-specific overhead?
- Hypothesis: Running the same chunked prefill loop with stock GLM attention (no Wayfinder swap) will isolate the chunking-inherent cost. If stock chunked prefill is also much slower than monolithic, the bottleneck is MLX chunking overhead, not HCSA graph construction.
- Change set:
  - `scripts/bench_glm_chunked_prefill_mlx.py`: added `--no-swap` flag, true autoregressive decode (64x1-token), `--kv-step` preallocation control.
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json`
- HCSA chunked reference path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 1 64 --cache-modes normal --no-swap --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_diag01_chunked_dense_baseline`
- Controls:
  - `--no-swap`: stock GLM attention, no HCSA graph construction
  - Same model, chunk size, seq len, seed, batch=1
  - Per-chunk timing already captured in chunk_reports

### EXP-20260207-DIAG02-CHUNK-LATENCY-PROFILE (result — ROOT CAUSE FOUND)
- Question: Do later chunks get progressively slower (growing cache penalty) or is per-chunk overhead flat?
- **ROOT CAUSE**: Line 259 of `hcsa/integrations/glm_mlx.py`:
  ```python
  if queries.shape[2] != keys.shape[2]:
      out = self._dense_fallback(queries, keys, values, mask, cache)
  ```
  During chunked prefill, `extract_qkv_from_glm_attention` merges the chunk K into the KV cache via `cache.update_and_fetch()`, returning full cached K (size `cache_offset + chunk_size`). Q remains at `chunk_size`. After chunk 0:
  - **Chunk 0**: Q=4096, K=4096 → HCSA sparse path (correct, 4.29s)
  - **Chunks 1-15**: Q=4096, K>4096 → **DENSE MLA FALLBACK** (O(chunk×cache))
- The 24x per-chunk slowdown and 6.9x total regression is entirely explained by 15 of 16 chunks falling back to O(T²) dense attention.
- The HCSA permute-window kernel (`wayfinder_permute_window_attention_batched`) requires `Q_len == K_len == T` because it permutes Q/K/V into cycle order over the same T positions. It has no codepath for Q_len < K_len (prefix-extended chunked prefill).
- Decision: **Critical fix needed** — implement Q_len < K_len support in HCSA permute-window path.
- Next action: Design and implement sparse chunked prefill that scatters Q into permuted space, runs local-window attention only at active positions, and gathers output back. Expected complexity: O(C·W) per chunk, matching monolithic HCSA for those positions.

### CONCLUSION: GLM Chunked Prefill Latency Regression (2026-02-08)

**Root Cause**: The entire 6.9x latency regression is caused by a single code path bug — `hcsa/integrations/glm_mlx.py:259` falls back to O(T²) dense MLA attention whenever `Q_len != K_len`, which is the case for all chunks after the first during chunked prefill.

**Impact**:
- Memory reduction (26.12% at T=65536) IS REAL — chunking works for memory
- Latency regression (+591.70%) is NOT inherent to HCSA — it's from the dense fallback
- If HCSA sparse attention is used for all chunks, per-chunk cost should be O(C·W) = constant, not growing with cache size
- Expected fix: chunked prefill latency should be within 2x of monolithic (not 6.9x)

**What remains for production-grade 200k prefill+generation**:
1. Implement Q_len < K_len support in the HCSA permute-window kernel
2. This requires: scatter Q into permuted space, run local-window attention at active positions only, gather output back
3. Graph/permutation should be pre-built once for target T and reused across chunks
4. After fix, re-benchmark T=65536/131072/202752 to validate

**Full-model superiority demonstrated**: Layer-level results are strong (3.5x throughput, 78% memory reduction at T=8192). The chunked prefill memory reduction is real (26% at T=65536). The latency regression is a fixable integration bug, not a fundamental limitation.

### EXP-20260207-DIAG03-HCSA-CHUNKED-4096-8192-FOCUSED (planned — deferred)
- Question: After diagnosing root cause, does chunk=8192 offer a better latency/memory Pareto point at T=65536 for HCSA?
- Hypothesis: Halving chunk count (8 vs 16) should roughly halve the fixed-overhead component, with memory tradeoff dependent on per-chunk activation footprint.
- Change set: measurement-only (reuse patched script).
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 8192 --decode-lens 0 1 64 --cache-modes normal --path permute --window 64 --landmark-stride 0 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_diag03_focused_chunk_sweep`

## 2026-02-08 — GLM Active-Query Permute Fix Validation

### EXP-20260208-GLM-ACTIVE-PERMUTE-FIX (planned)
- Question: Does an active-query Hamiltonian permute path (Q_len < K_len) eliminate dense fallback in chunked prefill and recover positive latency while preserving memory reduction?
- Hypothesis: Routing chunked prefill through active-query windowed attention will remove the O(chunk*cache) dense path for chunks >0, making per-chunk time flatter and improving prefill latency materially versus `EXP-20260208-001331-GLM-LENGTH-SWEEP-CHUNK4096-RESULT`, while keeping peak memory at or below prior chunked levels.
- Change set:
  - `hcsa/mlx/attention.py`: add `wayfinder_permute_window_attention_active_batched(...)` for active query rows
  - `hcsa/integrations/glm_mlx.py`: use active-query permute path when `self.path=="permute"` and `Q_len < K_len` instead of dense fallback
- Baseline run path:
  - `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_001331_length_sweep_chunk4096/results.json`
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 1 --cache-modes normal --path permute --window 64 --landmark-stride 0 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_active_permute_fix`
- Controls:
  - Retro/backfill disabled for inference
  - Same model/path/window/chunk as prior chunked baseline reference
  - Baseline comparison retained via `--baseline-path`

### EXP-20260208-GLM-HYBRID-THRESHOLD-SWEEP (planned)
- Question: With active-row matmul + adaptive graph reuse in place, what dense/permute crossover threshold minimizes 32k chunked prefill latency while preserving memory reduction?
- Hypothesis: A hybrid gate (`dense` for early low-K chunks, `permute` for later high-K chunks) will beat always-permute latency at 32k while keeping a meaningful memory reduction; expected best threshold in the `8k..24k` range.
- Change set:
  - `hcsa/mlx/attention.py`: active-row kernel uses batched `mx.matmul` for score/value aggregation
  - `hcsa/integrations/glm_mlx.py`: configurable `active_dense_threshold` gate for active mode
  - `scripts/bench_glm_chunked_prefill_mlx.py`: per-chunk profile sample includes mode/length fields
- Baseline run paths:
  - Chunked always-permute (adaptive graph): `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_active_permute_adaptive_graph_matmul_32k/results.json`
  - Chunked stock dense: `benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_chunked_dense_control_32k_adaptive_cmp/results.json`
- Command:
  - `for t in 0 4096 8192 16384 24576 32768; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_matmul_32k; done`
- Controls:
  - Same model/seq/chunk/window/batch/seed, retro off
  - Same chunking loop across all thresholds
  - Compare absolute + delta + percentage vs named baselines

### EXP-20260208-GLM-HYBRID-THRESHOLD-SWEEP (result)
- Question: With active-row matmul + adaptive graph reuse in place, what dense/permute crossover threshold minimizes 32k chunked prefill latency while preserving memory reduction?
- Commands:
  - `for t in 0 4096 8192 16384 32768; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_matmul_32k_v3; done`
- Key results (`T=32768`, `chunk=4096`, prefill-only):
  - `threshold=0` (always Hamiltonian active-permute): `267.87 s`, `122.33 tok/s`, peak `22,314,797,052 B`, memory reduction vs monolithic `22.86%`, chunk-path mix `permute=8, dense=0`.
  - `threshold=4096`: `245.80 s`, `133.31 tok/s`, peak `22,310,307,836 B`, memory reduction `22.88%`, chunk-path mix `permute=7, dense=1`.
  - `threshold=8192`: `228.55 s`, `143.38 tok/s`, peak `22,310,307,836 B`, memory reduction `22.88%`, chunk-path mix `permute=6, dense=2`.
  - `threshold=16384`: `197.32 s`, `166.07 tok/s`, peak `22,446,619,216 B`, memory reduction `22.41%`, chunk-path mix `permute=4, dense=4`.
  - `threshold=32768` (always dense in active mode): `164.15 s`, `199.62 tok/s`, peak `26,018,070,576 B`, memory reduction `10.06%`, chunk-path mix `permute=0, dense=8`.
- Comparison discipline:
  - Absolute metric + delta + percent were computed against monolithic baseline via script output.
  - Memory reduction sign convention preserved: `100 * (1 - wayfinder / baseline)`.
- Interpretation:
  - `attention_ms` for Hamiltonian chunks correlates with current `k_len/seq_len` (chunks 1..7 correlation with `seq_len` is strongly positive), while `graph_seq_len` was fixed for reused-graph runs.
  - Graph construction amortization is working (`cache_hit=true` after first active chunk, near-zero `graph_build_ms`), but remaining gap is active attention kernel constant factor.
  - Hybrid gating provides a controllable latency/memory Pareto frontier at 32k.
- Decision: Keep hybrid gate + matmul optimization.
- Next action:
  - Use `active_dense_threshold=16384` as a working default for follow-up (retains ~22% memory reduction with major latency improvement vs always-permute).
  - Re-run at `T=65536` with thresholds `{16384, 32768}` to locate long-context crossover where Hamiltonian path becomes net-latency favorable.

### EXP-20260208-GLM-HYBRID-THRESHOLD-TRIANGULATION-65536 (planned)
- Question: At `T=65536`, where is the practical crossover between dense and Hamiltonian active-permute, and does adding a `49152` threshold point triangulate that transition?
- Hypothesis: At 65k, mixed thresholds should reveal a later-chunk crossover where Hamiltonian chunks become competitive/faster while preserving a stronger memory reduction than always-dense chunked control.
- Change set:
  - Measurement-only run using existing hybrid gate + active-row matmul + adaptive graph reuse.
- Command:
  - `for t in 16384 32768 49152; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_matmul_65k; done`
- Controls:
  - Same model/chunk/window/batch/seed, retro off.
  - Compare against:
    - always-permute 32k/65k reference runs,
    - chunked dense control (threshold high / `--no-swap` reference),
    - monolithic baseline via `--baseline-path`.

### EXP-20260208-GLM-CHUNK-CROSSOVER-EXTRACTION-32K (result)
- Question: From existing 32k sweep data, at what `k_len` does Hamiltonian chunk time approach dense chunk time, and does `attention_ms` track `k_len` or `graph_seq_len`?
- Hypothesis: `attention_ms` in active-permute tracks live cache length (`k_len`) and not `graph_seq_len`; chunk wall time crossover should appear near the tail chunk.
- Change set:
  - Analysis-only (no code changes).
- Command:
  - `python3 - <<'PY' ... compare chunk_reports from threshold=0 vs threshold=32768, compute per-chunk delta and correlations ... PY`
- Controls:
  - Same model/sequence/chunk (`T=32768`, `chunk=4096`) across both runs.
  - Same benchmark harness and fixed adaptive graph reuse config (`graph_seq_len=32768`).
- Metrics:
  - Per-chunk `permute_sec - dense_sec` by `k_len`:
    - `4096`: `+33.37 s`
    - `8192`: `+18.87 s`
    - `12288`: `+15.75 s`
    - `16384`: `+13.81 s`
    - `20480`: `+10.43 s`
    - `24576`: `+7.61 s`
    - `28672`: `+3.71 s`
    - `32768`: `+0.18 s`
  - Correlations:
    - `corr(k_len, permute_attention_ms) = 0.9671`
    - `corr(k_len, dense_chunk_sec) = 0.7469`
    - `permute graph_seq_len unique = [32768]`
    - `cache_hit` pattern for permute run: `[False, True, True, True, True, True, True, True]`
- Interpretation:
  - Graph-build amortization is working (single miss then stable hits).
  - Remaining active-permute cost scales with live cache/query workload, not graph rebuild.
  - At `T=32768`, dense remains faster across all chunks, but gap collapses at the tail (`k_len=32768`), consistent with an approaching crossover regime.
- Decision: Keep hybrid strategy; keep pressure on reducing active-kernel constant factor.
- Next action:
  - Complete `T=65536` triangulation (`16384/32768/49152`) and extract the first chunk index where `permute_sec <= dense_sec`.

### EXP-20260208-GLM-HYBRID-THRESHOLD-TRIANGULATION-65536 (status)
- Status: Completed (this block preserves launch metadata).
- Start time (UTC): `2026-02-08T04:56:08Z`
- Command:
  - `for t in 16384 32768 49152; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_matmul_65k; done`
- Immediate objective:
  - Quantify latency/memory tradeoff at 65k and identify the practical dense↔Hamiltonian crossover zone for a production default threshold.

### EXP-20260208-GLM-HYBRID-THRESHOLD-TRIANGULATION-65536 (result)
- Question: At `T=65536`, where is the practical dense↔Hamiltonian crossover, and what threshold gives the best measured latency/memory tradeoff?
- Commands:
  - `for t in 16384 32768 49152; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_matmul_65k; done`
- Baseline (`T=65536`, monolithic stock from `baseline-path`):
  - `126.689 s`, `517.296 tok/s`, peak `44,890,891,940 B`.
- Key results (`T=65536`, `chunk=4096`, prefill-only):
  - `threshold=16384`: `959.900 s`, `68.274 tok/s`, peak `24,143,352,764 B`, delta vs baseline: `+833.211 s` (`+657.68%`), `-449.023 tok/s` (`-86.80%`), peak memory `-20,747,539,176 B` (`-46.22%`). Path mix: `permute=12`, `dense=4`.
  - `threshold=32768`: `923.290 s`, `70.981 tok/s`, peak `26,018,201,648 B`, delta vs baseline: `+796.600 s` (`+628.78%`), `-446.315 tok/s` (`-86.28%`), peak memory `-18,872,690,292 B` (`-42.04%`). Path mix: `permute=8`, `dense=8`.
  - `threshold=49152`: `890.700 s`, `73.578 tok/s`, peak `29,589,653,008 B`, delta vs baseline: `+764.010 s` (`+603.06%`), `-443.718 tok/s` (`-85.78%`), peak memory `-15,301,238,932 B` (`-34.09%`). Path mix: `permute=4`, `dense=12`.
- Comparison discipline:
  - Absolute metrics and deltas are listed against the named baseline run.
  - Memory reduction sign convention preserved: `100 * (1 - wayfinder / baseline)`.
- Additional crossover signal (tentative, cross-run pairing):
  - Comparing same `k_len` chunks where threshold policy differs between `32768` and `49152`:
    - `k_len=36864`: permute slower by `+24.25 s`
    - `k_len=40960`: permute faster by `-27.39 s`
    - `k_len=45056`: permute faster by `-15.13 s`
    - `k_len=49152`: permute faster by `-7.65 s`
  - This suggests a potential crossover band around `k_len ~= 40k` (needs repeat verification due run-to-run variance).
- Interpretation:
  - At 65k, raising threshold (more dense early chunks) monotonically improves latency but sacrifices memory savings.
  - Even the fastest tested point (`49152`) is still far from monolithic baseline latency.
  - Current bottleneck remains active-kernel constant factor and/or high-K chunk compute, not graph build amortization.
- Decision: Keep direction (hybrid + Hamiltonian) but prioritize kernel-level constant-factor reduction over additional threshold-only tuning.
- Next action:
  - Add a matched `--no-swap` chunked dense control at `T=65536` (same chunking) to isolate chunking/KV growth cost from Wayfinder-specific overhead.
  - Implement active-path vectorization (remove Python per-head loop/eval barriers), then rerun `16384/32768/49152`.

### EXP-20260208-GLM-ACTIVE-HEAD-CHUNK-VECTORIZATION (planned)
- Question: Does vectorizing active-query attention across head chunks reduce the constant-factor latency gap without breaking chunked prefill correctness?
- Hypothesis: Removing per-head Python loops in `wayfinder_permute_window_attention_active_batched` will materially reduce chunk wall time at the same memory profile, especially for mixed thresholds.
- Change set:
  - `hcsa/mlx/attention.py`: vectorize active-row path over `hc` heads per chunk; keep profiling semantics unchanged.
- Command (smoke):
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 8192 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 0 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_active_vec_heads_smoke_8k`
- Controls:
  - Retro/backfill disabled; same model/chunk/window/kv-step as prior runs.
- Success criteria:
  - Run completes without shape/causality errors.
  - Prefill latency is not worse than pre-vectorization smoke at comparable settings.

### EXP-20260208-GLM-ACTIVE-HEAD-CHUNK-VECTORIZATION (result: smoke)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 8192 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 0 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_active_vec_heads_smoke_8k_fix1`
- Metrics (`prefill-only`):
  - `sec=111.190`, `tok/s=73.676`, `peak=22,175,554,348 B`, `ok=true`.
  - chunk timing: `chunk0=71.881s`, `chunk1=39.284s`.
- Decision: Proceed to 32k mixed-threshold rerun for apples-to-apples impact check.
- Next action:
  - Run `T=32768`, `chunk=4096`, `active_dense_threshold=16384` with same controls as prior sweep and compare against `20260208_glm_hybrid_thresh16384_matmul_32k_v3`.

### EXP-20260208-GLM-CHUNKED-DENSE-CONTROL-65K-MATCHED (result)
- Question: How much latency at `T=65536` is from chunking/KV growth itself, independent of Wayfinder/Hamiltonian path?
- Commands:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --no-swap --path permute --window 64 --landmark-stride 0 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_chunked_dense_control_65k_matched`
  - Repeat command with out-dir `.../20260208_glm_chunked_dense_control_65k_matched_repeat`.
- Baseline (`T=65536`, monolithic stock): `126.689 s`, `517.296 tok/s`, peak `44,890,891,940 B`.
- Results (`prefill-only`):
  - Run A: `785.103 s`, `83.474 tok/s`, peak `33,161,104,180 B`.
    - Delta vs baseline: `+658.413 s` (`+519.71%`), `-433.822 tok/s` (`-83.86%`), peak memory `-11,729,787,760 B` (`-26.13%`).
  - Run B: `626.799 s`, `104.557 tok/s`, peak `33,161,104,180 B`.
    - Delta vs baseline: `+500.109 s` (`+394.75%`), `-412.740 tok/s` (`-79.79%`), peak memory `-11,729,787,760 B` (`-26.13%`).
- Variance summary:
  - Median prefill sec: `705.951 s` (std `79.152 s`).
  - Median tok/s: `94.016` (std `10.541`).
- Interpretation:
  - Chunking/KV-growth overhead at 65k is itself severe and high variance on this setup.
  - Wayfinder runs (`890.7..959.9 s`) are slower than chunked dense control, but the control confirms the majority of the absolute gap vs monolithic baseline is not Wayfinder-only.
- Decision: Keep as fairness control baseline for all future 65k decisions.
- Next action:
  - Use median/replicated reporting for 65k comparisons; avoid single-run conclusions.

### EXP-20260208-GLM-ACTIVE-HEAD-CHUNK-VECTORIZATION-32K (result: failed; reverted)
- Question: Does active-path head-chunk vectorization improve 32k mixed-threshold latency?
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 16384 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh16384_vec_heads_32k`
- Comparison vs pre-vectorization reference (`20260208_glm_hybrid_thresh16384_matmul_32k_v3`):
  - New: `943.046 s`, `34.747 tok/s`, peak `23,531,751,416 B`
  - Old: `197.318 s`, `166.067 tok/s`, peak `22,446,619,216 B`
  - Delta: `+745.728 s`, `-131.320 tok/s`, peak `+1,085,132,200 B`
- Interpretation:
  - Vectorized head-chunk implementation introduced a major performance regression.
  - Likely cause: high-cost gather/broadcast pattern in vectorized path outweighs Python-loop savings.
- Decision: Revert vectorized active-path loop (done) and continue with safer optimizations.
- Next action:
  - Target lower-risk improvements: adaptive query-chunk tuning and conditional fused/local path experiments, then re-benchmark 32k before returning to 65k.

### EXP-20260208-GLM-QUERY-CHUNK-SWEEP-32K (planned)
- Question: Can tuning `query_chunk_size` recover latency at 32k mixed threshold without changing kernel semantics?
- Hypothesis: Larger query chunks (`256` or `384`) may reduce Python/dispatch overhead enough to improve total prefill at acceptable memory cost.
- Change set:
  - `scripts/bench_glm_chunked_prefill_mlx.py`: add `--query-chunk-size` CLI control and pass through to `GLMWayfinderConfig.query_chunk_size`.
- Command:
  - `for q in 256 384; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 16384 --query-chunk-size $q --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh16384_qchunk${q}_32k; done`
- Controls:
  - Same model/seq/chunk/threshold/window/kv-step/retro-off as the 32k reference run.
- Success criteria:
  - Beat or match `197.318 s` from `20260208_glm_hybrid_thresh16384_matmul_32k_v3` while preserving comparable memory.

### EXP-20260208-GLM-QUERY-CHUNK-SWEEP-32K-Q512 (planned)
- Question: Does increasing query chunk size to `512` improve 32k mixed-threshold latency beyond the `q=384` result?
- Hypothesis: `q=512` may further reduce dispatch overhead, but could start increasing kernel memory pressure; outcome uncertain.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 16384 --query-chunk-size 512 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh16384_qchunk512_32k`
- Controls:
  - Same model/chunk/threshold/window/kv-step/retro-off, compare against `q=384` and prior 32k reference.

### EXP-20260208-GLM-HEAD-CHUNK-TUNING-32K (planned)
- Question: Can increasing `permute_head_chunk_size` reduce active-path overhead at 32k mixed threshold?
- Hypothesis: Increasing head chunk size from `2` to `4` lowers synchronization/dispatch overhead and improves runtime, at modest memory risk.
- Change set:
  - `scripts/bench_glm_chunked_prefill_mlx.py`: add `--head-chunk-size` CLI control and wire to `GLMWayfinderConfig.permute_head_chunk_size`.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 16384 --query-chunk-size 384 --head-chunk-size 4 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh16384_q384_h4_32k`
- Controls:
  - Compare against `q=384,h=2` run: `20260208_glm_hybrid_thresh16384_qchunk384_32k`.

### EXP-20260208-GLM-QUERY-CHUNK-SWEEP-32K (result)
- Question: Can query chunk tuning improve 32k mixed-threshold runtime?
- Commands:
  - `q=256` run: `.../20260208_glm_hybrid_thresh16384_qchunk256_32k`
  - `q=384` run: `.../20260208_glm_hybrid_thresh16384_qchunk384_32k`
  - `q=512` run: `.../20260208_glm_hybrid_thresh16384_qchunk512_32k`
- Reference (`q=192,h=2`): `197.318 s`, `166.067 tok/s`, peak `22,446,619,216 B`.
- Results:
  - `q=256,h=2`: `195.345 s`, `167.745 tok/s`, peak `22,446,619,216 B`.
    - Delta vs reference: `-1.973 s`, `+1.678 tok/s`.
  - `q=384,h=2`: `193.848 s`, `169.039 tok/s`, peak `22,446,619,216 B`.
    - Delta vs reference: `-3.470 s`, `+2.973 tok/s`.
  - `q=512,h=2`: `193.906 s`, `168.989 tok/s`, peak `22,446,619,216 B`.
    - Delta vs reference: `-3.412 s`, `+2.922 tok/s`.
- Interpretation:
  - Query-chunk tuning gives a small but repeatable latency gain (~1.7-1.8%) with no memory penalty.
  - Best point observed: `q=384` (very close to `q=512`).
- Decision: Keep tuned query chunk (`q=384`) for follow-on runs.
- Next action:
  - Validate 65k impact using best tuned setting (`q=384`) at latency-favoring threshold (`49152`).

### EXP-20260208-GLM-HEAD-CHUNK-TUNING-32K (result)
- Question: Does increasing head chunk size from `2` to `4` help at 32k mixed threshold?
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 32768 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 16384 --query-chunk-size 384 --head-chunk-size 4 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh16384_q384_h4_32k`
- Result:
  - `q=384,h=4`: `193.977 s`, `168.928 tok/s`, peak `22,446,619,216 B`.
  - Compared to `q=384,h=2`: slightly slower (`+0.129 s`, `-0.111 tok/s`).
- Decision: Keep `head_chunk_size=2`.
- Next action:
  - Carry forward `q=384,h=2` as current tuned configuration.

### EXP-20260208-GLM-65K-TUNED-Q384-TH49152 (planned)
- Question: Does the 32k-tuned setting (`q=384,h=2`) improve 65k latency at the best prior threshold (`49152`)?
- Hypothesis: Query-chunk tuning should carry over partially and reduce total runtime versus prior `threshold=49152` result (`890.700 s`).
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 49152 --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k`
- Controls:
  - Compare against prior `threshold=49152` run: `20260208_glm_hybrid_thresh49152_matmul_65k`.

### EXP-20260208-GLM-65K-TUNED-Q384-TH49152-REPEAT (planned)
- Question: Is the large 65k tuned gain (`483.5s`) reproducible under the same controls?
- Hypothesis: A repeat run should stay in the same performance regime and remain faster than chunked dense controls.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 49152 --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k_repeat`

### EXP-20260208-GLM-65K-TUNED-Q384-TH49152 (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 49152 --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k`
- Metrics (`prefill-only`):
  - `sec=483.501`, `tok/s=135.545`, peak `29,589,653,008 B`.
- Delta vs prior `threshold=49152` run (`20260208_glm_hybrid_thresh49152_matmul_65k`):
  - `-407.199 s`, `+61.967 tok/s`, same peak memory.
- Delta vs chunked dense controls:
  - vs run A (`785.103 s`): `-301.602 s`, `+52.070 tok/s`.
  - vs run B (`626.799 s`): `-143.298 s`, `+30.988 tok/s`.
- Interpretation:
  - Tuned query-chunk configuration creates a clear positive benchmark versus chunked dense control at 65k while preserving strong memory advantage.

### EXP-20260208-GLM-65K-TUNED-Q384-TH49152-REPEAT (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 49152 --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh49152_q384_h2_65k_repeat`
- Metrics (`prefill-only`):
  - `sec=485.454`, `tok/s=134.999`, peak `29,589,653,008 B`.
- Pair summary (A/B tuned runs):
  - Median: `484.477 s`, `135.272 tok/s`, peak `29,589,653,008 B`.
  - Stddev: `0.977 s`, `0.273 tok/s`.
- Comparison discipline:
  - vs prior `threshold=49152` (`890.700 s`, `73.578 tok/s`): median delta `-406.223 s`, `+61.694 tok/s`.
  - vs chunked dense control median (`705.951 s`, `94.016 tok/s`, peak `33,161,104,180 B`): median delta `-221.474 s`, `+41.257 tok/s`, and `10.77%` lower peak memory.
  - vs monolithic baseline (`126.689 s`, `517.296 tok/s`): still slower in absolute latency (`+357.788 s`) but with `34.09%` peak-memory reduction.
- Decision: Keep tuned config as current best-known for 65k chunked regime.
- Next action:
  - Promote tuned params (`query_chunk_size=384`, `head_chunk_size=2`, `threshold=49152`) as current benchmark default for long-context chunked runs.
  - Re-run 65k dense control once under same machine state window for a tighter paired comparison, then attempt threshold re-tuning around this tuned query chunk.

### EXP-20260208-GLM-65K-THRESHOLD-RETUNE-Q384H2 (planned)
- Question: Around the tuned setting (`q=384,h=2`), does threshold `45056` or `53248` outperform `49152` at 65k?
- Hypothesis: A nearby threshold may further improve latency while maintaining a memory advantage over chunked dense control.
- Command:
  - `for t in 45056 53248; do PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold $t --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh${t}_q384_h2_65k; done`
- Controls:
  - Compare against current best: `20260208_glm_hybrid_thresh49152_q384_h2_65k(_repeat)`.

### EXP-20260208-GLM-65K-BEST-TH45056-REPEAT (planned)
- Question: Is the new best setting (`threshold=45056`, `q=384`, `h=2`) reproducible?
- Hypothesis: Repeat remains near the observed `464.7s` regime and keeps advantage over `49152` tuned median.
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py --model-path mlx-community/GLM-4.7-Flash-4bit --seq-lens 65536 --chunk-sizes 4096 --decode-lens 0 --cache-modes normal --path permute --window 64 --landmark-stride 0 --active-dense-threshold 45056 --query-chunk-size 384 --head-chunk-size 2 --kv-step 4096 --baseline-path benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260207_201347_fullmodel_prefill_kv_checkpoint/results.json --out-dir benchmarks/mlx/glm_4_7_flash_4bit_wayfinder/20260208_glm_hybrid_thresh45056_q384_h2_65k_repeat`

### EXP-20260208-GLM-65K-THRESHOLD-RETUNE-Q384H2 (result)
- Command:
  - `for t in 45056 53248; do ... --active-dense-threshold $t --query-chunk-size 384 --head-chunk-size 2 ... ; done`
- Results:
  - `threshold=45056`: `464.723 s`, `141.022 tok/s`, peak `28,696,790,168 B`.
    - vs tuned `49152` run A (`483.501 s`, `135.545 tok/s`): `-18.778 s`, `+5.477 tok/s`, lower peak by `892,862,840 B`.
    - path mix: `permute=5`, `dense=11`.
  - `threshold=53248`: `638.096 s`, `102.706 tok/s`, peak `30,482,515,848 B`.
    - vs tuned `49152` run A: `+154.596 s`, `-32.839 tok/s`, higher peak by `892,862,840 B`.
    - path mix: `permute=3`, `dense=13`.
- Interpretation:
  - `45056` looked promising on single run, `53248` is clearly worse.
  - Need a repeat for `45056` before promotion because this regime has known high variance.
- Decision: Run repeat at `45056` before changing default.

### EXP-20260208-GLM-65K-BEST-TH45056-REPEAT (result)
- Command:
  - `PYTHONPATH=. python3 scripts/bench_glm_chunked_prefill_mlx.py ... --active-dense-threshold 45056 --query-chunk-size 384 --head-chunk-size 2 ... --out-dir .../20260208_glm_hybrid_thresh45056_q384_h2_65k_repeat`
- Metrics (`prefill-only`):
  - Repeat run: `658.550 s`, `99.515 tok/s`, peak `28,696,790,168 B`.
- Pair summary (`45056`):
  - Runs: `464.723 s` and `658.550 s`.
  - Median: `561.637 s`, `120.269 tok/s`.
  - Stddev: `96.914 s`, `20.753 tok/s`.
- Stability comparison:
  - Tuned `49152` pair median/std: `484.477 s ± 0.977 s`, `135.272 tok/s ± 0.273`.
  - `45056` pair shows high variance and lower median performance than stable `49152`.
- Decision: Keep `threshold=49152` as reproducible long-context default.
- Next action:
  - Use `threshold=49152`, `query_chunk_size=384`, `head_chunk_size=2` as the promoted benchmark config in README/roadmap.
