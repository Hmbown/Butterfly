# Butterfly Paper Outline

Section-by-section outline for a paper built around the current, claim-safe
Butterfly block-sparse topology.

## Working Title

**Butterfly Block-Sparse Attention: A Causal Prefix-Scan Topology for
Long-Context Transformers**

## Abstract

- Problem: long-context attention needs sparse patterns with explicit causal
  communication guarantees.
- Construction: bounded-degree block support with local, sink, and staged
  partner edges.
- Main theorem: the `causal_shift` partner rule reaches every block's full
  causal prefix in `ceil(log2 N)` layers.
- Evidence: generated support artifacts show last-row support in 20/20 cases and
  staged-vs-control support-AUC wins in 19/20 cases.
- Boundary: topology reachability only; model quality and backend performance
  are separate measurements.

## 1. Introduction

- Full attention is expensive at long context.
- Sparse attention needs more than a cheap mask; it needs a communication story.
- Present Butterfly as a deterministic block-sparse staged topology.
- State the narrow contribution: a causal prefix-scan construction and a
  validation harness, not a universal quality guarantee.

## 2. Background

- Sparse attention: sliding-window, global-token, random, and block-sparse
  methods.
- Classical staged networks: butterfly and Benes-style communication.
- Difference from the repo's method: the transformer setting is causal, so
  non-causal partner networks need causality-aware treatment.

## 3. Construction

- Sequence partitioned into `N` fixed-size blocks.
- Per-block support: self, local predecessors, optional sinks, stage partners.
- Degree bound: `1 + local_window_blocks + sink_count + partner_count`.
- Partner rules:
  - `causal_shift`: proof-clean rule, `i - 2^k`
  - `xor`, `bit_reversal`, `benes`: legacy empirical rules

## 4. Main Theorem

**Theorem:** With `causal_shift`, `partner_count >= 1`, and self edges, every
block reaches its full causal prefix after `ceil(log2 N)` staged layers.

Proof idea:

- Any causal distance `d = i - j` is less than `2^ceil(log2 N)`.
- Write `d` in binary.
- At each stage bit, take the shift edge if that bit is set; otherwise take the
  self edge.
- The path subtracts exactly `d` and lands at `j`.

## 5. Empirical Structural Evidence

Use generated artifacts:

- `results/proof/butterfly_validity/summary.json`
- `results/proof/butterfly_staging_validity/summary.json`

Current result bullets:

- full last-row support at `ceil(log2 N)`: 20/20 cases
- local-only full support at the same horizon: 0/20
- random predecessor full support at the same horizon: 8/20
- staged support-AUC beats local-only and best frozen-stage controls: 19/20
- stronger weighted mixing diagnostics are mixed and should stay non-claims

## 6. Implementation

- Public topology API: `bna/topology/butterfly.py`
- Validation framework: `bna/topology/validation.py`
- CUDA integration: `bna/integrations/qwen_torch.py`
- MLX integration: `bna/integrations/qwen_mlx.py` with `path="block_sparse"`
- Legacy MLX permute-window path remains a compatibility route, not the theorem
  target.

## 7. Model-Level Evaluation

Required before making a model-quality claim:

- perplexity vs stock attention
- long-context retrieval or needle-style checks
- RULER-style checks if practical
- small chat regression set at multiple context lengths
- latency and memory measurements for `causal_shift`, `xor`, and stock

## 8. Limitations

- The theorem is structural reachability, not learned attention behavior.
- It does not prove dense-attention equivalence.
- It does not prove robust weighted mixing.
- It does not prove optimality against arbitrary sparse schedules.
- Current public performance artifacts mostly predate the proof-clean
  `causal_shift` rule.

## 9. Recommended Paper Claim

Paper-safe claim:

> Butterfly is a bounded-degree causal block-sparse topology. With the
> `causal_shift` partner rule, it provably gives every block access to its full
> causal prefix after `ceil(log2 N)` staged layers. Existing CPU structural
> experiments validate the implementation and show that staging usually expands
> support faster than local-only or frozen long-range controls.

Avoid claiming:

- general quality preservation
- general mixing superiority
- universal latency wins
- theorem-level guarantees for causal-filtered `xor`, `bit_reversal`, or
  `benes`
