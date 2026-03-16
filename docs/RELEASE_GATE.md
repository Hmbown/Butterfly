# Public release gate

This gate defines when Wayfinder is ready for broader public promotion.

## Soft launch gate (engineering audience)
All items must be true:

1. Stable-profile reproducibility
- `./scripts/run_public_stable_profile_glm.sh` completes successfully.
- Outputs include:
  - `dense/results.json`
  - `wayfinder/results.json`
  - `stable_profile_summary.json`
  - `stable_profile_summary.md`

2. Evidence integrity
- Public performance claims map to `docs/FIRST_RELEASE.md`.
- Each claim has a concrete artifact path.

3. Scope discipline
- Support matrix marks only GLM stable profile as the validated default.
- Qwen/Nanbeige remain explicitly experimental or known-regression.
- Qwen 3.5 kernel work may be discussed as active engineering work, but not as a validated release claim.
- Any statement about keeping Qwen 3.5 prefill speed roughly consistent as context grows must be framed as an experimental target, not as a verified result.

4. Visual/document integrity
- Visual assets render from documented commands in `docs/VISUAL_STORYBOARD.md`.
- README and docs links are valid.

5. Security and contribution policy visible
- `SECURITY.md` exists and is linked from README.
- `CONTRIBUTING.md` exists and is linked from README.

## Broad Advertising Gate
Soft launch gate plus all items below:

1. Stability confirmation
- Latest strict path-audit run has informative fallback reason labeling (no `unspecified`).
- No unresolved OOM issues on validated scope.

2. Messaging consistency
- Headline claims match latest validated evidence.
- No unsupported claims for experimental tiers.

3. Operational readiness
- Troubleshooting section includes failure handling and artifact expectations.
- Repro commands are copy-paste safe.

## Current Default Gate Sources
- Stable evidence: `docs/FIRST_RELEASE.md`
- Architecture constraints: `docs/ARCHITECTURE.md`
- Positioning contract: `docs/PUBLIC_POSITIONING.md`
- Visual narrative: `docs/VISUAL_STORYBOARD.md`
