# Butterfly Paper Completeness Checklist

This checklist reflects the current block-sparse Butterfly method after adding
the proof-clean `causal_shift` partner rule. The canonical math and theorem
status live in `docs/BUTTERFLY_THEOREMS.md`.

## What Is Ready

| Area | Status | Notes |
|---|---|---|
| Block-sparse topology definition | Ready | self + local + sink + staged partner blocks |
| Bounded degree argument | Ready | proof by construction |
| `causal_shift` all-row prefix theorem | Ready | binary-decomposition proof at `ceil(log2 N)` |
| CPU structural experiments | Ready | regenerated `results/proof/*/summary.*` |
| CUDA block-sparse integration | Ready for experiments | exposed through `path="block_sparse"` |
| MLX block-sparse integration | Ready for experiments | exposed through `path="block_sparse"` |
| Legacy-rule compatibility | Ready | `xor`, `bit_reversal`, `benes` remain supported |

## What Is Partial

| Area | Status | Gap |
|---|---|---|
| Legacy-rule theory | Partial | causal filtering weakens the clean classical butterfly/Benes theorem |
| Staging advantage | Partial | support-AUC beats controls in 19/20 generated cases, not all 20 |
| Non-power-of-two coverage | Partial | `causal_shift` is proved; legacy rules remain empirical |
| Weighted mixing diagnostics | Partial | useful for debugging, not robust enough for claims |
| MLX performance | Partial | block-sparse path exists, but proof-clean `causal_shift` needs fresh benchmarks |
| Quality evaluation | Partial | benchmark artifacts exist, but not enough for broad quality claims |

## What Is Missing

| Area | Status | Needed |
|---|---|---|
| Model-level quality | Missing | perplexity, retrieval, RULER-style checks, chat regressions |
| Backend performance claim for `causal_shift` | Missing | CUDA and MLX block-sparse timing vs `xor` and stock attention |
| Literature positioning | Missing | sparse attention, butterfly networks, Benes networks, staged communication |
| Optimality analysis | Missing | lower bounds or comparisons against broader sparse schedules |
| Public release gate | Missing | quality + latency + memory evidence for a single recommended model/profile |

## Current Honest Paper Shape

The legitimate research contribution is:

1. A bounded-degree block-sparse causal topology.
2. A proof-clean `causal_shift` rule that gives full all-row causal-prefix
   reachability in `ceil(log2 N)` staged layers.
3. Empirical evidence that the legacy staged rules also give strong last-row
   support and that staging usually beats local-only/frozen controls.
4. A clear non-claim boundary: topology reachability is not model quality,
   dense-attention equivalence, or a robust mixing theorem.

## Highest-Value Next Tasks

1. Benchmark `path="block_sparse" --block-partner-rule causal_shift` on MLX
   against stock attention and the current `xor` default.
2. Run a small model-quality gate for stock vs block-sparse on the same model:
   perplexity first, then retrieval/RULER if practical.
3. Write the related-work section only after the claim is frozen around the
   prefix-scan theorem, not around the older causal-filtered XOR intuition.
4. Decide whether `causal_shift` should become the default after fresh quality
   and performance measurements.
