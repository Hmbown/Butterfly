# Research

This document captures open research questions and hypothesis framing. It is intentionally exploratory and does not define public product defaults.

Naming note: use `Butterfly` / `BNA` for current public-facing discussion. Legacy terms such as `Wayfinder` and `HCSA` may still appear in artifact names and older experiments.

## Research Questions
Primary question:
- Can sparse attention learn like a slime mold: start from structured connectivity and discover which edges carry information at scale?

Practical version:
1. Start from an overcomplete candidate graph (window + landmarks + cycles/rewires).
2. Learn or adapt a degree-budgeted subgraph (for example, per-token/head top-`D` edges).
3. Measure three outcomes:
- long-context quality retention
- depth required for near-full-prefix receptive fields under causal composition
- end-to-end throughput and peak memory at fixed sparsity budget

Related structural question:
- Does multiscale cycle richness of the undirected skeleton predict depth needed for full-prefix receptive fields beyond spectral-gap-only explanations?

## Experimental Framing
Current working assumptions (subject to data):
- Topology quality and dispatch quality are coupled; a theoretically sparse graph can still underperform if runtime routing is fallback-heavy.
- Decode-path behavior at long boundaries can dominate end-to-end outcomes even when prefill improves.
- Model-specific integration behavior (GLM vs Qwen/Nanbeige) must be reported separately; transfer assumptions are unsafe without matched controls.

What counts as evidence:
- paired dense vs Butterfly runs under matched controls
- complete fallback diagnostics and stop-gate compliance
- explicit deltas (absolute and percent) versus named baselines

What is not accepted as evidence:
- unpaired runs
- partial artifacts with missing `single_turn` rows
- aspirational claims without reproducible command+artifact linkage

## Current Program Direction
- Treat GLM-4.7 stable path as validated default for public use.
- Continue Nanbeige/Qwen as experimental slices until boundary regressions and fallback pressure are reduced.
- Keep release claims conservative and tied to documented artifacts.

Detailed release measurements and reproduction commands:
- `docs/FIRST_RELEASE.md`

## References
Sparse/structured attention and graph sparsity references:

- BigBird: <https://arxiv.org/abs/2007.14062>
- Exphormer: <https://arxiv.org/abs/2303.06147>
- FlashInfer: <https://arxiv.org/abs/2501.01005>
- Flex Attention: <https://arxiv.org/abs/2412.05496>
