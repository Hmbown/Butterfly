# Butterfly Topology: Research Status and Paper-Safe Claims

This document restates the current Butterfly topology/proof story in theorem/lemma style, with explicit status for each claim (proved, empirically supported, or missing). It is intended as a reference for drafting a research paper.

---

## Definitions

### Butterfly Block Topology

Given a sequence partitioned into `N` blocks of fixed size, the Butterfly topology defines for each query block a support set consisting of:

1. **Self:** the query block itself
2. **Local window:** `w` immediately preceding blocks
3. **Sink blocks:** a fixed set of early blocks (typically `[0, 1, ..., k-1]`)
4. **Partner blocks:** one or more deterministic long-range blocks determined by a stage-dependent partner rule

The partner rules are:
- **xor:** partner = `block_idx XOR (1 << stage_bit)`
- **bit_reversal:** partner = bit-reversed index at the current stage
- **benes:** forward half matches XOR, backward half uses a complementary pattern

The **stage schedule** cycles through stages over layers:
- For `xor` and `bit_reversal`: `stage = layer_idx mod ceil(log2(N))`
- For `benes`: `stage = layer_idx mod (2 * ceil(log2(N)) - 2)`

### Causal-Prefix Support

For a block at position `i`, the causal prefix is the set of blocks `{0, 1, ..., i}`. A topology achieves **causal-prefix support** for block `i` after `L` layers if the composed boolean reachability operator (union of supports across layers) can reach every block in `{0, ..., i}`.

**Support coverage** for the last block (position `N-1`) is the fraction of the full causal prefix reachable after `L` layers. Full support coverage = 1.0 means the last block can reach all causally-prior blocks.

### Staged Support Expansion

A topology exhibits **staged support expansion** if the support coverage curve over depth (layers) grows faster than controls that fix the long-range partners to a single stage (frozen) or use only local connections.

**Support AUC** (area under the support curve) is a scalar summary: higher AUC means faster expansion.

### Boolean Reachability vs. Learned Mixing

The validation framework distinguishes:
- **Boolean reachability:** structural connectivity (can information flow from block A to block B?)
- **Learned mixing:** weighted attention scores under a specific weighting model (how much information flows?)

The proof surface focuses on boolean reachability. Weighted mixing metrics (entropy, total variation to uniform, effective support, spectral diagnostics) are secondary surrogates and are **not** robust across different weighting models.

---

## Paper-Safe Claims

### Proposition 1 (Support Sufficiency)

**Claim:** For the staged Butterfly topology with partner rules ∈ {xor, bit_reversal, benes}, the composed boolean reachability operator achieves full last-row causal-prefix support at depth `L = ceil(log2(N))` for all tested `N ∈ {32, 64, 128, 256}`.

**Status:** Empirically supported (12/12 cases).

**Evidence:** `results/proof/butterfly_validity/summary.json` shows `support_coverage_last_row == 1.0` for Butterfly in all 12 cases (3 rules × 4 block counts). Local-only achieves 0/12; random predecessor achieves 3/12.

**Limitations:**
- Tested only on powers of 2 (32, 64, 128, 256)
- No formal proof for general N
- Non-power-of-2 behavior is not fully characterized

---

### Proposition 2 (Staging Advantage)

**Claim:** The staged Butterfly schedule strictly dominates both local-only and best frozen-stage controls on support AUC across all tested (partner_rule, N) pairs. Full support by depth `ceil(log2(N))` is achieved for Butterfly but not for controls.

**Status:** Empirically supported (12/12 cases).

**Evidence:** `results/proof/butterfly_staging_validity/summary.json` shows:
- `support_auc_wins == 12` (Butterfly beats local and all frozen variants)
- `support_full_by_width == 12` for Butterfly, `0` for local, `0` for frozen

**Limitations:**
- Tested only on powers of 2
- Comparison is against specific frozen-stage baselines (not all possible static schedules)
- No asymptotic analysis of AUC growth rate

---

### Proposition 3 (Degree Budget)

**Claim:** The Butterfly topology maintains a bounded per-block degree: at most `1 + local_window_blocks + partner_count + sink_count`. For the standard configuration (`local_window_blocks=1`, `partner_count=1`, `sink_count=1`), this is at most 4.

**Status:** Proved by construction.

**Evidence:** The topology construction in `bna/topology/butterfly.py` explicitly adds only these neighbor categories. Causality filtering may reduce degree for early blocks but never exceeds the budget.

---

### Non-Claims

#### Mixing/Concentration Guarantees

**Claim:** Butterfly topology achieves better mixing (lower total variation to uniform, higher entropy, higher effective support) than local-only baselines.

**Status:** Not robust enough for a paper claim.

**Evidence:** Weighted surrogate diagnostics in `results/proof/butterfly_staging_validity/summary.json` are mixed:
- TV wins: 47/60 cases
- Entropy wins: 33/60 cases
- Effective support wins: 24/60 cases
- Max mass wins: 17/60 cases
- Spectral wins: 8/60 cases

The results depend on the weighting model (uniform, local_biased, partner_biased, sink_biased, dirichlet_random). No single weighting model produces consistent wins across all metrics.

**Conclusion:** Do not claim a general mixing guarantee. These are secondary surrogates that are sensitive to weighting assumptions.

---

#### Model-Level Quality

**Claim:** Butterfly topology improves task performance (perplexity, retrieval, chat quality) compared to full attention or local-only baselines.

**Status:** Not established in this proof surface.

**Evidence:** The validation framework operates on boolean reachability and surrogate weighted operators, not on actual model inference or task evaluation.

**Conclusion:** The topology story is structural, not behavioral. Model-level claims require separate evaluation (perplexity, needle-in-haystack, RULER, etc.).

---

#### MLX Path Topology Equivalence

**Claim:** The MLX integration (`bna/integrations/qwen_mlx.py`) uses the same staged block-sparse Butterfly topology.

**Status:** False.

**Evidence:** The MLX path uses Hamiltonian permute-window attention, not the block-sparse Butterfly topology from `bna/topology/butterfly.py`. The block-sparse Butterfly path exists only in the CUDA integration (`bna/integrations/qwen_torch.py` with `path="block_sparse"`).

**Conclusion:** The topology proof results do not directly apply to the MLX permute-window path. A separate characterization of MLX topology properties is needed.

---

## Missing for a Full Paper

### Formal Proof

- **General theorem:** Prove that for all N (not just powers of 2), Butterfly achieves full support at depth `ceil(log2(N))`.
- **Non-power-of-2 analysis:** Full matrix of results for N ∈ {33, 65, 100, 129, 200, 255, 257} across all partner rules.

### Asymptotic Characterization

- **Diameter bounds:** Tight upper/lower bounds on reachability diameter as a function of N.
- **AUC scaling:** Asymptotic growth rate of support AUC for Butterfly vs controls.

### Literature Positioning

- **Relation to classical Butterfly/Beneš networks:** Explicitly map the staged partner schedule to classical non-blocking switching network literature.
- **Relation to sparse attention:** Position the block-sparse approach relative to other sparse attention mechanisms (longformer, bigbird, etc.).

### Model-Level Evidence

- **Perplexity:** Long-context perplexity vs full attention for Qwen 3.5.
- **Retrieval:** Needle-in-haystack style retrieval accuracy.
- **RULER:** RULER-style long-context benchmarks if available.
- **Chat quality:** Small regression set at 4K, 16K, 64K, 128K.

### MLX Path Characterization

- **MLX topology properties:** Measure support coverage, reachability diameter, and staging properties of the MLX permute-window graphs using the same validation framework.
- **Equivalence or difference:** Determine whether MLX permute-window achieves similar support expansion to block-sparse Butterfly.

---

## Summary

| Claim | Status | Evidence |
|---|---|---|
| Full support at ⌈log₂N⌉ (powers of 2) | ✅ Empirical | 12/12 cases |
| Staging beats frozen/local (powers of 2) | ✅ Empirical | 12/12 cases |
| Bounded degree budget | ✅ Proof | By construction |
| Mixing guarantees robust | ❌ Non-claim | Mixed 33-47/60 |
| Model-level quality | ❌ Missing | Not evaluated |
| MLX path equivalence | ❌ False | Different topology |
| General theorem for all N | ⚠️ Missing | Only powers of 2 tested |
| Asymptotic analysis | ⚠️ Missing | Not characterized |
| Literature positioning | ❌ Missing | Not written |
