# Glossary

## Butterfly / BNA
Current public project/runtime identity used in this repository for the sparse-attention runtime.

## Wayfinder
Legacy project/runtime identity still present in implementation details, benchmark artifact paths, and older docs.

## HCSA
Legacy mechanism name: Hamiltonian Cycle Sparse Attention. In this repo it refers to bounded-neighborhood sparse attention built from window edges, permutation-cycle edges, and optional landmarks.

## Hamiltonian cycle (repo definition)
A permutation-induced cycle over token indices. Each token receives two cycle neighbors before causal masking. This is not a metric-space TSP claim.

## Graph ABI
Backend-agnostic adjacency contract:
- `neigh_idx`: `int32`, padded with `-1`, shape `[T,D]` or `[H,T,D]`
- `edge_type`: `uint8` enum `{PAD, CYCLE, WINDOW, LANDMARK, REWIRE}`

## Stable public profile
Validated default path for current release:
- script: `./scripts/run_public_stable_profile_glm.sh`
- model: `mlx-community/GLM-4.7-Flash-4bit`
- operating point: `T=8192`, `decode_len=32`

## Support matrix tiers
- `Validated`: recommended default for public usage
- `Experimental`: opt-in only, not default
- `Known regression`: explicitly non-default

## Prefill
The initial forward pass over prompt tokens (`q_len > 2` in practical flow), where Butterfly currently targets most acceleration.

## Decode
Autoregressive generation steps (`q_len <= 2` in this release posture), routed to stock quadratic attention by default for stability.

## Stock (mode)
The native model attention configuration with no BNA modifications. For Qwen 3.5, this is the hybrid GatedDeltaNet + quadratic architecture (NOT dense attention — only 8 of 32 layers are quadratic). For GLM and GPT-2, stock IS dense attention. In CLI args, `--mode stock` and `--mode dense` are aliases. Internal code uses `"dense"` as the enum value for backward compatibility with existing results.

## Dense fallback
Runtime behavior where Butterfly routes a step to stock quadratic SDPA due to configured thresholds or decode policy.

## Memory reduction convention
Release convention for memory reporting:
- `reduction % = 100 * (1 - butterfly / stock)`
- Positive values mean Butterfly uses less memory.

## Retro/backfill
Research path for retrocausal-style augmentation. For inference, default posture remains off unless explicitly enabled for experiments.
