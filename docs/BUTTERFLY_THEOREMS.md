# Butterfly Topology: Research Status and Paper-Safe Claims

This document is the current Butterfly topology/proof story and claim-safety
ledger for paper drafting.

## Definitions

### Butterfly Block Topology

Given a sequence partitioned into `N` fixed-size blocks, each query block attends
to:

1. itself
2. a bounded local predecessor window
3. optional early sink blocks
4. one or more deterministic stage partners

All partner proposals are causally filtered. The public implementation is
`bna.topology.butterfly`, with CUDA and MLX block-sparse integration surfaces.

Current partner rules:

- `xor`: `partner = block_idx XOR (1 << stage_bit)`
- `bit_reversal`: bit-reverse, flip the stage bit, reverse back
- `benes`: forward/backward staged bit schedule over XOR-style partners
- `causal_shift`: `partner = block_idx - (1 << stage_bit)`

`causal_shift` is the proof-clean rule. The other three are legacy experimental
rules retained for compatibility with existing artifacts.

### Causal-Prefix Support

For block `i`, the causal prefix is `{0, 1, ..., i}`. A composed topology reaches
causal-prefix support for `i` after `L` layers when boolean reachability can
reach every block in that prefix.

### Boolean Reachability vs. Learned Mixing

The validation framework separates:

- boolean reachability: structural connectivity
- learned mixing: actual attention weights chosen by a model

The proof surface is about reachability. Weighted spread, entropy, total
variation, and spectrum metrics are retained as diagnostics only.

## Paper-Safe Claims

### Proposition 1: Bounded Degree

**Claim:** Butterfly maintains per-block degree at most
`1 + local_window_blocks + partner_count + sink_count`.

**Status:** Proved by construction.

**Evidence:** `bna/topology/butterfly.py` only adds those neighbor categories;
causal filtering can remove neighbors but cannot add new ones.

### Proposition 2: `causal_shift` Prefix Theorem

**Claim:** With `partner_count >= 1`, self edges, and stage bits
`0..ceil(log2 N)-1`, the `causal_shift` rule reaches every block's full causal
prefix after exactly `ceil(log2 N)` layers.

**Status:** Proved here and regression-tested.

**Proof:** Let `w = ceil(log2 N)`. For any target block `j <= i`, set
`d = i - j`. Since `0 <= d <= i <= N - 1 < 2^w`, `d` has a binary expansion
using only stage bits `0..w-1`. At stage `k`, take the `causal_shift` edge
`x -> x - 2^k` if bit `k` of `d` is set; otherwise take the self edge. The path
never moves below `j`, so every chosen shift is causal and nonnegative. After
all `w` stages, the path has subtracted exactly `d` and lands at `j`.

**Evidence:** Focused tests cover `N = 16, 32, 33, 64, 65, 100, 129, 200, 255,
257`.

### Proposition 3: Last-Row Support Sufficiency

**Claim:** The staged Butterfly topology reaches full last-row causal-prefix
support by `L = ceil(log2 N)` across the currently generated matrix.

**Status:** Empirically supported.

**Evidence:** `results/proof/butterfly_validity/summary.json` reports full
last-row support in 20/20 cases: partner rules `xor`, `bit_reversal`, `benes`,
and `causal_shift` over block counts `16, 32, 33, 64, 128`. Local-only reaches
0/20 at the same horizon; the random predecessor control reaches 8/20.

**Limitations:** This proposition is empirical for the legacy rules. Only
`causal_shift` currently has the all-row log-depth proof.

### Proposition 4: Staging Advantage Over Simple Controls

**Claim:** The staged schedule expands last-row support faster than local-only
and best frozen-stage controls in the current matrix.

**Status:** Empirically supported with one caveat.

**Evidence:** `results/proof/butterfly_staging_validity/summary.json` reports
support-AUC wins in 19/20 cases and full last-row support by `ceil(log2 N)` in
20/20 cases, versus 0/20 for local-only and 0/20 for the reference frozen
control at that horizon.

**Caveat:** The one support-AUC miss is `causal_shift` at `N = 16`, where a
frozen stage has slightly higher AUC. This does not break the prefix theorem,
but it means the paper claim should not say "strictly dominates every frozen
control in all cases."

## Non-Claims

### General Mixing Guarantee

**Status:** Not established.

Weighted diagnostics remain mixed. In the current staged-validity artifact,
across 100 weighted comparisons, Butterfly beats both controls on entropy AUC
30 times, effective-support AUC 21 times, TV-to-uniform AUC 40 times, max-mass
AUC 14 times, and effective-rank-ratio AUC 26 times.

Conclusion: do not claim a robust near-uniform mixing or conditioning theorem.

### Model-Level Quality

**Status:** Not established by the topology proof surface.

The topology experiments do not evaluate perplexity, retrieval, chat quality, or
dense-attention equivalence. Those require separate model-level evaluation.

### MLX Path Equivalence

**Status:** Split by path.

- `path="block_sparse"` in `bna/integrations/qwen_mlx.py` uses the staged
  block-sparse Butterfly topology and is covered by this structural proof.
- `path="permute"` is the legacy MLX permute-window compatibility route and is
  not covered by the block-sparse topology theorem.

## Missing For A Full Paper

- Literature positioning against sparse attention and classical staged networks.
- End-to-end model-quality evidence: perplexity, retrieval, RULER-style checks,
  and chat regressions.
- Backend-specific performance gates for CUDA and MLX, especially for
  `causal_shift`.
- A tighter characterization of the legacy `xor`, `bit_reversal`, and `benes`
  rules under causal filtering.
- Optimality or lower-bound analysis against broader sparse schedules.

## Summary

| Claim | Status | Evidence |
|---|---|---|
| Bounded degree | Proved | By construction |
| `causal_shift` all-row prefix at `ceil(log2 N)` | Proved | This file, tests |
| Legacy-rule last-row support at `ceil(log2 N)` | Empirical | 20/20 generated cases including `causal_shift` |
| Staging beats local/best frozen on support AUC | Empirical with caveat | 19/20 generated cases |
| Robust mixing guarantee | Non-claim | Mixed weighted diagnostics |
| Model-level quality | Missing | Not evaluated here |
| MLX block-sparse topology equivalence | True for `path="block_sparse"` | Shared topology builder |
