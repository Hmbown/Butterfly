# Window-Drop Resilience (Theorem 1.5 Alignment)

This note documents why HCSA window-drop regularization can be safe in practice.

## Theory Link

Draganić et al. (2025), Theorem 1.5: if a spanning subgraph preserves
minimum degree above roughly half of the original degree, Hamiltonicity is
resilient to random edge deletions.

In HCSA terms:
- Base graph: cycle edges + local window edges.
- Perturbation: drop non-cycle window edges during training.
- Safety rule: keep self and cycle-neighbor edges; monitor residual minimum
  degree against a half-degree threshold.

## Runtime Policy

- `window_drop` is a training-time regularizer.
- Permute kernels preserve offsets `{0, -1, +1}` in cycle order.
- Resilience checks are empirical and optional via `hcsa.graph.analysis.check_resilience`.

## Empirical Check (T=128, window=32)

Using a fixed random cycle permutation:

- `drop_rate=0.3`: survival rate `0.96`, min-degree mean `20.08`,
  threshold `17.0`.
- `drop_rate=0.8`: survival rate `0.00`, min-degree mean `3.12`,
  threshold `17.0`.

Interpretation:
- Moderate drop behaves as a safe regularizer under this topology.
- Aggressive drop destroys the required degree margin and should be avoided.

## Repro Command

```bash
python3 -c "from hcsa.graph.analysis import check_resilience; import numpy as np; p=np.random.default_rng(42).permutation(128); print(check_resilience(p, window=32, drop_rate=0.3, num_trials=100)); print(check_resilience(p, window=32, drop_rate=0.8, num_trials=100))"
```
