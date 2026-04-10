# Butterfly Paper Completeness Checklist

This document assesses the readiness of the current research artifacts for a full research paper on the Butterfly topology. Each section is marked as ✅ (ready), ⚠️ (partial), or ❌ (missing).

---

## Paper Sections

### 1. Introduction

| Subsection | Status | Notes |
|---|---|---|
| Motivation (sparse attention, long-context) | ✅ | Can be drafted from existing docs |
| Problem statement (block-sparse causal communication) | ✅ | Clear from `BUTTERFLY_THEOREMS.md` |
| High-level contributions | ✅ | Support sufficiency, staging advantage |
| Limitations | ✅ | Documented in non-claims section |

---

### 2. Background / Related Work

| Subsection | Status | Notes |
|---|---|---|
| Sparse attention mechanisms (Longformer, BigBird, etc.) | ❌ | Need literature review section |
| Classical Butterfly/Beneš networks | ❌ | Need explicit mapping to switching network literature |
| Staged communication networks | ❌ | Need positioning relative to other staged topologies |
| Block-sparse attention in transformers | ⚠️ | Partially covered in docs but not synthesized |

---

### 3. Problem Formulation

| Subsection | Status | Notes |
|---|---|---|
| Formal definition of block-sparse causal topology | ✅ | `BUTTERFLY_THEOREMS.md` has clear definitions |
| Causal-prefix support definition | ✅ | Clearly defined |
| Boolean reachability vs. learned mixing | ✅ | Distinction made explicit |
| Degree budget constraints | ✅ | Proved by construction |

---

### 4. Butterfly Construction

| Subsection | Status | Notes |
|---|---|---|
| Partner rules (xor, bit_reversal, benes) | ✅ | API in `bna/topology/butterfly.py` |
| Stage schedule definition | ✅ | Clear definition |
| Block layout algorithm | ✅ | `build_block_butterfly_layout` in code |
| Causality filtering | ✅ | Implemented and tested |

---

### 5. Main Propositions

| Proposition | Status | Evidence |
|---|---|---|
| Support sufficiency at ⌈log₂N⌉ | ⚠️ | Empirical 12/12 (powers of 2), no general proof |
| Staging advantage over frozen/local | ⚠️ | Empirical 12/12 (powers of 2), no asymptotic analysis |
| Degree budget bounded | ✅ | Proof by construction |

---

### 6. Experiments

| Experiment | Status | Details |
|---|---|---|
| Support coverage vs depth (powers of 2) | ✅ | `butterfly_validity/summary.json` |
| Support AUC comparison (staged vs frozen vs local) | ✅ | `butterfly_staging_validity/summary.json` |
| Partner rule ablation (xor, bit_reversal, benes) | ✅ | 3 rules × 4 sizes = 12 cases |
| Block count ablation (32, 64, 128, 256) | ✅ | 4 sizes × 3 rules = 12 cases |
| Non-power-of-2 block counts | ❌ | Missing (33, 65, 100, 129, 200, 255, 257) |
| Weighted mixing diagnostics | ⚠️ | Mixed results, not robust enough for claims |
| Degree budget verification | ✅ | Test in `test_butterfly_topology.py` |
| Reachability diameter measurement | ✅ | Test in `test_butterfly_topology.py` |

---

### 7. Ablations and Controls

| Ablation/Control | Status | Details |
|---|---|---|
| Local-only baseline | ✅ | Tested and worse |
| Frozen long-range controls | ✅ | All frozen stages tested, all worse |
| Random predecessor control | ✅ | Tested, mixed results |
| Partner count variation | ⚠️ | Tested for 1 partner, not for 2+ systematically |
| Local window size variation | ⚠️ | Tested for 1 block, not for other sizes |
| Sink count variation | ⚠️ | Tested for 1 sink, not for other counts |

---

### 8. Failure Modes

| Failure Mode | Status | Notes |
|---|---|---|
| Mixing diagnostics not robust | ✅ | Explicitly documented as non-claim |
| Non-power-of-2 behavior untested | ✅ | Documented as limitation |
| Model-level quality not evaluated | ✅ | Documented as separate work |
| MLX path not equivalent | ✅ | Documented as different topology |

---

### 9. Model-Level Evidence

| Evidence Type | Status | Notes |
|---|---|---|
| Perplexity vs full attention | ❌ | Not evaluated |
| Needle-in-haystack retrieval | ❌ | Not evaluated |
| RULER benchmarks | ❌ | Not evaluated |
| Chat quality regression set | ❌ | Not evaluated |
| CUDA Qwen 3.5 inference | ⚠️ | Benchmarks exist but not part of validated public default |
| MLX Qwen 3.5 inference | ⚠️ | Benchmarks exist but topology not characterized |

---

### 10. Theoretical Analysis

| Analysis | Status | Notes |
|---|---|---|
| Formal proof of support sufficiency | ❌ | Only empirical evidence |
| Asymptotic diameter bounds | ❌ | Not characterized |
| AUC scaling analysis | ❌ | Not characterized |
| Relation to switching network theory | ❌ | Not written |
| Information-theoretic analysis | ❌ | Not attempted |

---

### 11. Implementation

| Component | Status | Notes |
|---|---|---|
| Public API (`bna/topology/butterfly.py`) | ✅ | Clean, documented |
| Validation framework (`bna/topology/validation.py`) | ✅ | Comprehensive |
| Test coverage | ✅ | `test_butterfly_topology.py`, `test_butterfly_staging_validity.py`, `test_butterfly_operator_mixing.py` |
| CUDA integration | ✅ | `qwen_torch.py` with `path="block_sparse"` |
| MLX integration | ⚠️ | Uses permute-window, not block-sparse Butterfly |

---

### 12. Reproducibility

| Aspect | Status | Notes |
|---|---|---|
| Experiment scripts | ✅ | `experiment_butterfly_validity.py`, `experiment_butterfly_staging_validity.py` |
| Result artifacts | ✅ | JSON summaries in `results/proof/` |
| Benchmark CLI | ✅ | Documented in `WAYFINDER_BLOCK_SPARSE.md` |
| Model checkpoints | ⚠️ | Paths are local, not public |
| Environment setup | ⚠️ | Partial (Apple Silicon setup doc exists) |

---

## Priority Gaps for Publication

### High Priority (Required for a complete paper)

1. **Formal proof or stronger empirical evidence** for support sufficiency
   - Add non-power-of-2 block counts to the empirical matrix
   - Consider a general proof or at least a conjecture with strong empirical backing
   - **Repo task:** Add test with N ∈ {33, 65, 100, 129, 200, 255, 257} (Step 5 of plan)

2. **Literature positioning**
   - Write background section mapping to classical Butterfly/Beneš networks
   - Position relative to other sparse attention mechanisms
   - **Repo task:** Create literature review section in paper outline

3. **Model-level evidence**
   - At least one model-level evaluation (perplexity or retrieval)
   - Minimal honest claim: "topology enables sparse attention without catastrophic degradation"
   - **Repo task:** Build eval script skeleton (can run later on GPU host)

### Medium Priority (Strengthens the paper)

4. **Asymptotic analysis**
   - Diameter bounds as function of N
   - AUC scaling rate
   - **Repo task:** Add theoretical analysis section to paper outline

5. **MLX topology characterization**
   - Measure support coverage of MLX permute-window graphs
   - Determine if MLX achieves similar expansion to block-sparse Butterfly
   - **Repo task:** Build MLX topology bridge script (Step 6 of plan)

6. **Expanded ablations**
   - Partner count variation (2, 3, 4 partners)
   - Local window size variation (2, 4, 8 blocks)
   - Sink count variation (0, 2, 4 sinks)
   - **Repo task:** Extend experiment scripts with additional configs

### Low Priority (Nice to have)

7. **Information-theoretic analysis**
   - Capacity of the staged communication network
   - Relation to mixing time in Markov chains

8. **Additional partner rules**
   - Explore other deterministic partner schedules
   - Randomized partner selection with analysis

---

## Summary

| Category | Ready | Partial | Missing |
|---|---|---|---|
| Problem formulation | ✅ | | |
| Construction | ✅ | | |
| Main propositions (empirical) | | ⚠️ | |
| Main propositions (formal) | | | ❌ |
| Experiments (core) | ✅ | | |
| Experiments (extended) | | ⚠️ | |
| Ablations | | ⚠️ | |
| Controls | ✅ | | |
| Failure modes | ✅ | | |
| Model-level evidence | | | ❌ |
| Theoretical analysis | | | ❌ |
| Literature positioning | | | ❌ |
| Implementation | ✅ | ⚠️ | |
| Reproducibility | | ⚠️ |

**Overall assessment:** The core empirical story is strong (12/12 support sufficiency, 12/12 staging advantage). The paper is missing the theoretical formalization (general proof, asymptotics), literature positioning, and model-level evidence that would make it a complete research contribution.
