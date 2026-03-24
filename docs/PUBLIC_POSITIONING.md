# Public Positioning (Inference Engineers)

Naming note: this repo now presents `Butterfly` / `BNA` as the public name on GitHub, while older internal docs still use `Wayfinder` / `HCSA`.

## Audience
Primary audience: inference engineers who need measurable long-context speedups with explicit stability boundaries.

## Core promise
Butterfly is a training-free sparse-attention runtime for long-context inference. It targets prefill acceleration by replacing dense attention neighborhoods with a bounded graph (`window + Hamiltonian cycle + landmarks`).

## What to lead with
1. A validated performance path exists today.
2. The mechanism is explicit and reproducible.
3. Stability boundaries are clear (validated vs experimental).

## Message pillars

### 1) Prefill acceleration with measured evidence
- Validated default: GLM-4.7 stable profile at `T=8192`, `decode_len=32`.
- Prefill delta: `-40.38%`, e2e delta: `-38.44%`.
- Evidence source: `docs/FIRST_RELEASE.md` and `benchmarks/mlx/first_release/EXP-20260218T151213Z-STABLE-PROFILE/stable_profile_summary.json`.

### 2) Graph-structured runtime, not retraining
- Butterfly neighborhood is bounded and explicit:
  - local causal window
  - permutation-induced Hamiltonian cycle neighbors
  - optional landmark edges
- Mechanism references:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/VISUAL_STORYBOARD.md`

### 3) Safety-first release posture
- Public default is narrow and auditable.
- Experimental and non-default slices remain clearly marked.
- Sources:
  - `README.md` support matrix
  - `docs/FIRST_RELEASE.md`
  - `docs/RELEASE_GATE.md`

### 4) Qwen 3.5 status
- Qwen 3.5 work is experimental.
- Current work includes custom kernels needed to get Butterfly working correctly on that model family.
- Current objective: keep prefill speed roughly consistent as context length increases.
- Treat that objective as an engineering target, not as a validated public claim.

## Explicit non-claims
Do not claim:
- universal quality parity vs dense across all workloads
- universal speedups on decode paths
- validated status for Qwen/Nanbeige paths
- validated Qwen 3.5 scaling behavior before the kernel work is benchmarked and reproduced

## Launch narrative order
1. Problem: dense prefill cost grows rapidly at long context.
2. Mechanism: bounded graph neighborhood via window + cycle + landmarks.
3. Proof: validated GLM stable-profile metrics.
4. Boundaries: dense-first decode policy and experimental tiers.
5. Action: reproducible command and artifact paths.

## Recommended headline and subhead
- Headline: `Butterfly: graph-sparse prefill acceleration for long-context inference`
- Subhead: `Training-free sparse-attention runtime with validated GLM-4.7 evidence, explicit release boundaries, and experimental Qwen 3.5 kernel work.`

## CTA for technical launch
- Verify environment: `./scripts/verify_install_and_preflight.sh --run-id EXP-YYYYMMDDTHHMMSSZ-VERIFY-INSTALL --out-dir benchmarks/mlx/preflight`
- Run validated default: `./scripts/run_public_stable_profile_glm.sh`
- Inspect evidence: `docs/FIRST_RELEASE.md`
