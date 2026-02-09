# E2E Verdict: HCSA Validation (Tiny -> Qwen3-1.7B -> GLM-4.7-Flash)

- Output root: `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216`
- Target superiority: **PASS**
- General superiority: **PASS**
- Confidence: **medium**

## 1) Idea Checks (1..5)
| Idea | Pass/Fail | Evidence Path | Directional Logic Check |
|---|---|---|---|
| Idea 1: Edge-disjoint cycles (d=1 vs d=2) | PASS | `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/idea1_disjoint_d1/results.json` | More cycles increased compute and reduced tok/s (T=4096 delta=-48.65%), matching expected overhead direction. |
| Idea 2: Resilience under edge drop | PASS | `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/tiny_wayfinder/idea_metrics_e2e.json` | Survival is high at drop_rate=0.3 (0.96) and collapses at 0.8 (0.00), matching theory. |
| Idea 3: Covering cycles convergence | PASS | `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/tiny_wayfinder/idea_metrics_e2e.json` | As d increases (1->4->8), coverage rises and L2-to-dense falls while cosine rises at T=256 and T=512. |
| Idea 4: Spectral/expansion diagnostics | PASS | `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/tiny_wayfinder/idea_metrics_e2e.json` | Random cycle expansion ratio (1.1251) exceeds identity (0.9939), and proxy mixing is faster. |
| Idea 5: Regular partition locality | FAIL | `/Volumes/VIXinSSD/wayfinder/benchmarks/mlx/e2e_validation_20260208_192216/tiebreak_idea5_reg16/results.json` | Direction is unstable: old run reg16@4096 +12.57%, current -9.65%, tie-break -1.44% vs random. |

## 2) Tiny Table
| Scenario | Dense tok/s | HCSA tok/s | Delta tok/s | % Delta tok/s | Dense mem | HCSA mem | Memory reduction % | Dense ppl | HCSA ppl | Joint utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_short | 417161.20 | 303755.41 | -113405.79 | -27.19% | 95191432 | 97337116 | -2.2541% | 55.0104 | 34.4972 | 0.7121 |
| tiny_long | 431693.67 | 232426.26 | -199267.41 | -46.16% | 95191432 | 120135536 | -26.2041% | 822.3014 | 90.0523 | 0.4266 |

## 3) Qwen Table (Block-level, steady-state)
| Seq | Dense tok/s | HCSA tok/s | Delta tok/s | % Delta tok/s | Dense mem | HCSA mem | Memory reduction % | Joint utility | sanity_mae | first-call build ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 82290.03 | 76625.67 | -5664.36 | -6.88% | 192422456 | 171450928 | 10.8987% | 1.0451 | 1.150623 | 806.63 |
| 8192 | 69279.01 | 77308.85 | 8029.84 | 11.59% | 1625016380 | 1541130292 | 5.1622% | 1.1766 | 1.228521 | 4858.61 |
| 32768 | 39325.68 | 81658.84 | 42333.15 | 107.65% | 3459500088 | 3258173488 | 5.8195% | 2.2048 | 1.311835 | 45685.67 |

## 4) GLM Tables
### 4A) Attention/Block Swap (steady-state)
| Seq | Status | Block joint utility | Note |
|---:|---|---:|---|
| 2048 | OK | 1.8617 | sanity_mae=0.003714, first-call=841.27 ms |
| 8192 | OK | 2.7699 | sanity_mae=0.004119, first-call=4048.77 ms |
| 32768 | ERROR | n/a | RuntimeError: [metal::malloc] Attempting to allocate 42949672960 bytes which is greater than the maximum allowed buffer size of 22613000192 bytes. |

### 4B/4C) Consumer Dense vs HCSA (E2E)
| Seq | Dense e2e sec | HCSA e2e sec | % Delta e2e sec | Dense ttft | HCSA ttft | % Delta ttft | Dense peak mem | HCSA peak mem | Memory reduction % | Joint utility |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 6.6615 | 7.2996 | 9.58% | 0.1070 | 0.1180 | 10.28% | 18284344600 | 18284344788 | -0.000001% | 0.9126 |
| 8192 | 191.9119 | 23.1856 | -87.92% | 99.6097 | 0.1922 | -99.81% | 20660500140 | 20660500328 | -0.000001% | 8.2772 |
| 32768 | 410.4896 | 276.4979 | -32.64% | 22.9608 | 13.8446 | -39.70% | 26017775484 | 26017775672 | -0.000001% | 1.4846 |
- Quality: dense=3/6 (0.500), hcsa=3/6 (0.500)

### 4D/4E) Chunked Prefill vs Dense (current run baseline)
| Config | Seq | Dense prefill tok/s | Candidate prefill tok/s | % Delta tok/s | Dense mem | Candidate mem | Memory reduction % | Joint utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thr16384 | 32768 | 48.1277 | 150.5934 | 212.90% | 26018037620 | 22446619216 | 13.7267% | 3.6269 |
| thr16384 | 65536 | 76.3449 | 113.8367 | 49.11% | 33161071420 | 24155149252 | 27.1581% | 2.0470 |
| thr49152 | 32768 | 48.1277 | 188.0168 | 290.66% | 26018037620 | 26018070576 | -0.0001% | 3.9066 |
| thr49152 | 65536 | 76.3449 | 128.1612 | 67.87% | 33161071420 | 29589653016 | 10.7699% | 1.8813 |
- Average joint utility: thr16384=2.8370, thr49152=2.8940; selected `thr49152`

## 5) Explicit Verdicts
- Target superiority (GLM long-context): **PASS**
- General superiority (across families): **PASS**
- Overall: HCSA is superior in long-context GLM/Qwen inference settings tested here, but not universally superior in Tiny training throughput/memory.
- Confidence: **medium**
  - Most long-context GLM/Qwen scenarios have joint_utility > 1.
  - One GLM attention-swap 32768 dense baseline failed due hardware buffer cap, so that slice uses consumer/chunked evidence instead.
  - Idea-5 regular partition showed run-state instability after tie-break.

## 6) Blockers
- Tiny long training run had joint_utility < 1.0 (speed-memory disadvantage despite quality gate pass).
- GLM attention-swap microbenchmark at seq_len=32768 could not run dense baseline on this 36GB machine due Metal max-buffer limit.
- Regular-partition throughput sign is unstable across repeats (tie-break resolved as non-robust at T=4096).
- Chunked dense control throughput drifted strongly vs historical baseline, indicating environment/state sensitivity for absolute tok/s.

## 7) GLM -> Kimi K2.5 Hypothesis (Falsifiable)
- Hypothesis: If Kimi K2.5 4-bit on 512GB Mac Studio remains attention-significant in 256K prefill and avoids dense fallback, GLM-observed long-context gains suggest meaningful net speedups with modest peak-memory relief.
- Projected 256K Kimi range: speedup 1.25x to 2.3x; peak-memory reduction 5% to 20%.
- Assumptions:
  - attention remains a major fraction of prefill wall-time after MoE routing and expert paging costs
  - no pathological dense fallback in chunked path at target context
  - expert paging locality is sufficient to avoid I/O-dominated stalls
- Falsifiers:
  - Reject high-gain hypothesis if 256K Kimi proxy speedup is <1.15x under matched dense-vs-HCSA settings.
  - Reject memory hypothesis if peak-memory reduction is <5% with swap fully active and dense fallback disabled.
  - Reject transfer if expert paging/I-O dominates >30% wall-time, masking attention savings.

## 8) Next 3 Actions (Ranked)
1. Promote threshold=49152 as default long-context inference setting; add auto-threshold fallback to 16384 for 65K-heavy workloads if per-run joint_utility drops.
2. Stabilize regular-partition strategy (seed/cache/state controls) before enabling it by default; keep random as default meanwhile.
3. Run a Kimi proxy experiment at >=128K with explicit dense-fallback/off and expert-paging telemetry to validate the scaling hypothesis bounds.
