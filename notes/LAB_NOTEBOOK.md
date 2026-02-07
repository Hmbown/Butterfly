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
