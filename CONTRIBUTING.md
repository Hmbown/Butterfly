# Contributing

Thanks for your interest in BNA. This repo mixes stable benchmark evidence, active runtime development, and a large amount of research history, so keep public-facing changes simple and evidence-backed.

## Before you change anything

Read the docs that define the current public posture:

- [README.md](README.md)
- [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [AGENTS.md](AGENTS.md)

## Local setup

```bash
pip install -e ".[dev]"
pytest
ruff check bna tests
```

If you are working on visuals or notebooks, install the extra dependencies you need instead of assuming they are present.

## Public claims

- Treat the GLM stable profile in [docs/FIRST_RELEASE.md](docs/FIRST_RELEASE.md) as the strongest validated public benchmark slice in this repo.
- Treat Qwen CUDA and MLX work as experimental unless you are also updating their support boundaries and evidence trail.
- Do not merge new performance claims without artifact paths and exact commands.

## Experiment discipline

For benchmark or ablation work, follow the lab-notebook policy in [AGENTS.md](AGENTS.md):

- add a hypothesis before a run
- record results after a run
- update both `notes/LAB_NOTEBOOK.md` and `notes/experiments.ndjson`
- compare against a named baseline with absolute and delta metrics

## Pull requests

Keep pull requests narrow.

- Separate docs-only cleanup from runtime changes when possible.
- Explain what changed, why, and what evidence supports it.
- Link benchmark artifacts for any performance, quality, or memory claim.
- Call out whether the change affects validated, experimental, or archival parts of the repo.

## Naming

The repo still contains older `Wayfinder` / `HCSA` terminology. For public-facing copy, prefer `Butterfly` / `BNA` unless you are editing historical material that must keep the older name for traceability.
