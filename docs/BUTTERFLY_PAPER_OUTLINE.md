# Butterfly Paper Outline

Section-by-section outline for a research paper on the Butterfly block-sparse topology, with repo task pointers for each gap.

---

## Title

**Butterfly Block-Sparse Attention: Staged Topology for Long-Context Transformers**

---

## Abstract

- Motivation: Long-context transformers require efficient sparse attention
- Challenge: Designing sparse patterns that preserve global communication
- Contribution: Butterfly topology — staged block-sparse schedule with O(log N) depth for full causal-prefix support
- Key results: Empirical support sufficiency (12/12 cases), staging advantage over frozen/local baselines (12/12 cases)
- Limitations: Empirical (not formal proof), topology-only (not model-level quality)

---

## 1. Introduction

### 1.1 Motivation
- Long-context models (128K+ tokens) demand efficient attention
- Full attention is O(N²) infeasible
- Sparse attention must balance efficiency with global communication

### 1.2 Problem Statement
- Design a block-sparse causal topology that:
  - Has bounded per-block degree
  - Achieves full causal-prefix support in O(log N) layers
  - Is deterministic and compile-time predictable

### 1.3 Contributions
1. **Butterfly construction:** Staged partner schedule with three partner rules (xor, bit_reversal, benes)
2. **Support sufficiency:** Empirical demonstration of full support at depth ⌈log₂N⌉ (12/12 cases)
3. **Staging advantage:** Staged schedule outperforms frozen-stage and local-only controls (12/12 cases)
4. **Validation framework:** Public implementation with comprehensive tests

### 1.4 Limitations
- Empirical evidence (powers of 2 only, not a general theorem)
- Topology-only validation (no model-level quality claims)
- MLX integration uses different topology (permute-window vs block-sparse)

---

## 2. Background and Related Work

### 2.1 Sparse Attention Mechanisms
- **Longformer:** Sliding window + global tokens
- **BigBird:** Random + sliding + global
- **Sparse Transformer:** Fixed sparse patterns
- **Routing mechanisms:** Routing-based dynamic attention

**Repo task:** Add literature review with citations. Map to classical sparse attention taxonomy.

### 2.2 Classical Butterfly/Beneš Networks
- **Butterfly networks:** O(log N) depth non-blocking switching
- **Beneš networks:** Rearrangeably non-blocking with 2log₂N - 2 stages
- **Connection to our work:** Staged partner schedule is analogous to switching network stages

**Repo task:** Write explicit mapping between Butterfly topology stages and classical network literature. Cite original Beneš (1965) and related work.

### 2.3 Staged Communication in Transformers
- **Layer-wise attention patterns:** Some work varies patterns by layer
- **Our contribution:** Deterministic stage schedule with provable support expansion

**Repo task:** Survey staged attention approaches and position Butterfly within this space.

---

## 3. Problem Formulation

### 3.1 Block-Sparse Causal Topology
- Sequence partitioned into N blocks of fixed size
- Query block attends to: self, local window, sink blocks, partner blocks
- Causality constraint: only attend to blocks ≤ query position

### 3.2 Causal-Prefix Support
- Definition: block i can reach {0, ..., i} after L layers
- Support coverage: fraction of causal prefix reachable
- Full support: coverage = 1.0

### 3.3 Boolean Reachability vs. Learned Mixing
- **Boolean reachability:** Structural connectivity (can information flow?)
- **Learned mixing:** Weighted attention scores under specific model
- **Our focus:** Boolean reachability as a structural prerequisite

### 3.4 Degree Budget
- Upper bound: 1 + local_window + partner_count + sink_count
- Standard config: ≤ 4 neighbors per block

---

## 4. Butterfly Construction

### 4.1 Partner Rules
- **xor:** partner = block XOR (1 << stage_bit)
- **bit_reversal:** partner = bit-reversed index at current stage
- **benes:** forward half matches XOR, backward half complementary

### 4.2 Stage Schedule
- xor/bit_reversal: stage = layer_idx mod ⌈log₂N⌉
- benes: stage = layer_idx mod (2⌈log₂N⌉ - 2)

### 4.3 Block Layout Algorithm
- Pseudocode for `build_block_butterfly_layout`
- Causality filtering for early blocks
- Sink block inclusion

### 4.4 Correctness Properties
- Self-inclusion (every block attends to itself)
- Causality (no future blocks in support)
- Bounded degree (respect degree budget)

---

## 5. Main Propositions

### 5.1 Support Sufficiency
**Proposition 1:** For Butterfly topology with partner rules ∈ {xor, bit_reversal, benes}, the composed boolean reachability operator achieves full last-row causal-prefix support at depth L = ⌈log₂N⌉ for all tested N ∈ {32, 64, 128, 256}.

**Status:** Empirical (12/12 cases).

**Limitations:** Only powers of 2 tested. No formal proof for general N.

**Repo task:** Add non-power-of-2 block counts (33, 65, 100, 129, 200, 255, 257) to empirical matrix. See Step 5 of plan.

### 5.2 Staging Advantage
**Proposition 2:** The staged Butterfly schedule strictly dominates both local-only and best frozen-stage controls on support AUC across all tested (partner_rule, N) pairs. Full support by depth ⌈log₂N⌉ is achieved for Butterfly but not for controls.

**Status:** Empirical (12/12 cases).

**Limitations:** No asymptotic AUC analysis.

**Repo task:** Add asymptotic AUC scaling analysis if possible, or clearly state as empirical observation.

### 5.3 Degree Budget
**Proposition 3:** The Butterfly topology maintains a bounded per-block degree ≤ 1 + local_window_blocks + partner_count + sink_count.

**Status:** Proof by construction.

---

## 6. Experiments

### 6.1 Experimental Setup
- **Metrics:** Support coverage, support AUC, reachability diameter
- **Topologies:** Butterfly (3 rules), local-only, frozen-stage controls, random predecessor
- **Block counts:** 32, 64, 128, 256
- **Depth:** Up to 2⌈log₂N⌉ layers

### 6.2 Support Coverage Results
- **Table 1:** Support coverage at depth ⌈log₂N⌉
  - Butterfly: 1.0 (12/12)
  - Local-only: < 1.0 (0/12)
  - Random: mixed (3/12)
- **Figure 1:** Support curves over depth

**Repo data:** `results/proof/butterfly_validity/summary.json`

### 6.3 Staging Advantage Results
- **Table 2:** Support AUC comparison
  - Butterfly wins vs local: 12/12
  - Butterfly wins vs best frozen: 12/12
- **Figure 2:** Support AUC bar chart

**Repo data:** `results/proof/butterfly_staging_validity/summary.json`

### 6.4 Partner Rule Ablation
- All three rules (xor, bit_reversal, benes) achieve full support
- Benes has more stages (2⌈log₂N⌉ - 2 vs ⌈log₂N⌉)

### 6.5 Block Count Ablation
- Full support holds across 32, 64, 128, 256 blocks
- Trend suggests scaling to larger N

**Repo task:** Add non-power-of-2 block counts (33, 65, 100, 129, 200, 255, 257). See Step 5 of plan.

### 6.6 Weighted Mixing Diagnostics (Non-Claims)
- Total variation to uniform: 47/60 wins
- Entropy: 33/60 wins
- Effective support: 24/60 wins
- **Conclusion:** Not robust enough for claims — included as exploratory analysis only

**Repo data:** `results/proof/butterfly_staging_validity/summary.json` weighted metrics

---

## 7. Ablations and Controls

### 7.1 Local-Only Baseline
- Same degree budget as Butterfly
- No long-range partners
- Fails to achieve full support

### 7.2 Frozen-Stage Controls
- Fix long-range partners to a single stage
- Test all possible frozen stages
- All fail to achieve full support by depth ⌈log₂N⌉

### 7.3 Random Predecessor Control
- Random long-range connections
- Mixed performance (3/12 full support)
- Less predictable than deterministic Butterfly

### 7.4 Parameter Sensitivity (Future Work)
- Partner count variation (2, 3, 4 partners)
- Local window size variation (2, 4, 8 blocks)
- Sink count variation (0, 2, 4 sinks)

**Repo task:** Extend experiment scripts with additional parameter sweeps.

---

## 8. Failure Modes and Limitations

### 8.1 Mixing Diagnostics Not Robust
- Weighted metrics depend on weighting model
- No single model produces consistent wins
- Not suitable as a general mixing guarantee

### 8.2 Non-Power-of-2 Behavior Untested
- Current results limited to powers of 2
- Need extended empirical study or formal proof

### 8.3 Model-Level Quality Not Evaluated
- Topology validation is structural, not behavioral
- Perplexity, retrieval, chat quality not assessed
- Separate evaluation required

### 8.4 MLX Path Not Equivalent
- MLX integration uses permute-window attention
- Different from block-sparse Butterfly topology
- Need separate characterization

**Repo task:** Build MLX topology bridge script (Step 6 of plan) to characterize MLX permute-window properties.

---

## 9. Implementation

### 9.1 Public API
- `bna/topology/butterfly.py`: Butterfly construction functions
- `bna/topology/validation.py`: Validation framework

### 9.2 Test Coverage
- `test_butterfly_topology.py`: Causality, reachability, degree, stage correctness
- `test_butterfly_staging_validity.py`: Staging advantage regression checks
- `test_butterfly_operator_mixing.py`: Mixing diagnostics (secondary)

### 9.3 CUDA Integration
- `bna/integrations/qwen_torch.py`: Qwen 3.5 CUDA with `path="block_sparse"`
- Uses staged Butterfly topology

### 9.4 MLX Integration
- `bna/integrations/qwen_mlx.py`: Qwen 3.5 MLX with permute-window attention
- **Note:** Different topology — not block-sparse Butterfly

**Repo task:** Characterize MLX topology properties (Step 6 of plan).

---

## 10. Reproducibility

### 10.1 Experiment Scripts
- `scripts/experiment_butterfly_validity.py`: Support coverage experiments
- `scripts/experiment_butterfly_staging_validity.py`: Staging advantage experiments

### 10.2 Result Artifacts
- `results/proof/butterfly_validity/summary.json`
- `results/proof/butterfly_staging_validity/summary.json`

### 10.3 Environment
- Python 3.12+, PyTorch, NumPy
- See `docs/APPLE_SILICON_SETUP.md` for MLX setup

### 10.4 Model Checkpoints
- Qwen 3.5 models (local paths, not public)
- Future: Add public checkpoint references

---

## 11. Theoretical Analysis (Future Work)

### 11.1 Formal Proof
- Conjecture: For all N, Butterfly achieves full support at depth ⌈log₂N⌉
- Approach: Induction on bit positions

### 11.2 Asymptotic Bounds
- **Diameter:** Upper bound on reachability diameter
- **AUC scaling:** Growth rate of support AUC vs N

### 11.3 Information-Theoretic Analysis
- Capacity of staged communication network
- Relation to mixing time in Markov chains

**Repo task:** Add theoretical analysis if formal proof is achievable, or clearly state as open problem.

---

## 12. Model-Level Evaluation (Future Work)

### 12.1 Perplexity
- Long-context perplexity vs full attention
- Qwen 3.5 on standard benchmarks

### 12.2 Retrieval
- Needle-in-haystack accuracy
- RULER-style benchmarks

### 12.3 Chat Quality
- Small regression set at 4K, 16K, 64K, 128K
- Human evaluation or automated metrics

**Repo task:** Build eval script skeleton (can run later on GPU host).

---

## 13. Conclusion

### 13.1 Summary
- Butterfly topology achieves O(log N) depth for full causal-prefix support
- Staged schedule outperforms frozen and local controls
- Empirical validation on powers of 2 (12/12 cases)

### 13.2 Limitations
- Empirical (not formal proof)
- Topology-only (not model-level)
- MLX path not equivalent

### 13.3 Future Directions
- Formal proof for general N
- Model-level evaluation
- MLX topology characterization
- Extended ablations

---

## Appendix A: Pseudocode

### A.1 Butterfly Block Layout
- Detailed pseudocode for `build_block_butterfly_layout`
- Partner computation for each rule

### A.2 Support Operator Composition
- Pseudocode for boolean reachability composition
- Support coverage measurement

---

## Appendix B: Additional Results

### B.1 Full Support Curves
- Detailed support curves for all configurations
- Per-partner-rule breakdown

### B.2 Weighted Metrics Tables
- Complete weighted diagnostic results
- Per-weighting-model breakdown

---

## Repo Task Summary

| Task | Priority | Status |
|---|---|---|
| Add non-power-of-2 block counts to empirical matrix | High | Step 5 of plan (pending) |
| Build MLX topology bridge script | High | Step 6 of plan (pending) |
| Write literature review section | Medium | Not started |
| Add asymptotic analysis | Medium | Not started |
| Build model-level eval script skeleton | Medium | Not started |
| Extend parameter ablations | Low | Not started |
