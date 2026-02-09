# Discover Setup (No Inference)

This document defines a setup-only workflow for fused-kernel discovery.
It prepares target metadata, readiness checks, and session stubs without
loading models or running attention/inference benchmarks.

## Scope

- Includes: K1-K5 target registry, environment/path validation, session scaffolding.
- Excludes: model loading, inference, attention benchmarking, and any LLM kernel search execution.

## Target IDs

- `k1`: `hcsa_permute_window_fused` (P0)
- `k2`: `hcsa_sparse_gather_fused` (P2)
- `k3`: `hcsa_graph_construct` (P1)
- `k4`: `hcsa_active_row_fused` (P1)
- `k5`: `hcsa_wayfinder_ttt_fused` (P2)

## Commands

List discovery targets:

```bash
python3 scripts/wayc.py discover-targets --targets all
```

Run setup scaffold in dry-run mode:

```bash
python3 scripts/wayc.py discover-setup \
  --targets all \
  --zmlx-root /path/to/ZMLX \
  --sessions-root discover_sessions \
  --kernel-out-root hcsa/mlx/kernels/metal \
  --dry-run
```

Write session stubs and seed kernels:

```bash
python3 scripts/wayc.py discover-setup \
  --targets all \
  --zmlx-root /path/to/ZMLX \
  --sessions-root discover_sessions \
  --kernel-out-root hcsa/mlx/kernels/metal \
  --strict
```

Read setup manifest:

```bash
python3 scripts/wayc.py discover-status --manifest discover_sessions/manifest.json
```

## Outputs

- `discover_sessions/manifest.json`
- `discover_sessions/*_session.stub.json` (one stub per selected target)
- `hcsa/mlx/kernels/metal/seeds/*.metal` (seed placeholders)

## Safety Defaults

- Retro/backfill defaults remain inference-safe (`retro_backfill_enabled=False`).
- Setup commands are metadata-only and do not execute model paths.
