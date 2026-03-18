# Contributing

Thanks for your interest in HCSA.

## Before opening a PR

1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for project structure and design decisions.
2. Run tests: `pytest`
3. Run lint: `ruff check hcsa/ tests/`

## Pull request expectations

- Explain what changed and why.
- Link benchmark artifacts for any performance claims.
- Keep scope tight — separate doc changes from runtime changes when possible.

## Benchmarks

If your change affects attention performance, run the relevant benchmark and include before/after numbers:

```bash
# MLX
python3 scripts/bench_glm_consumer_mlx.py --mode wayfinder --seq-lens 8192 --decode-len 32 --repeats 1 --skip-multi-turn --skip-quality

# PyTorch
python3 scripts/bench.py --device auto --seq-lens 128 256 512 1024
```

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md).
