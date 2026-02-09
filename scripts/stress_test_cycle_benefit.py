#!/usr/bin/env python3
"""Stress test: does adding Hamiltonian cycles help causally or only undirectedly?

Compares window_only vs window + k random cycles at multiple sequence lengths.
Computes both undirected metrics (spectral gap, Fiedler, diameter) and
causal metrics (L-hop coverage, causal diameter).

Usage:
    python3 scripts/stress_test_cycle_benefit.py --seq-lens 128 256 512 1024
    python3 scripts/stress_test_cycle_benefit.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hcsa.graph.expander import (  # noqa: E402
    causal_diameter,
    causal_reachability,
    effective_diameter,
    spectral_gap,
)


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

def build_window_only(T: int, w: int) -> np.ndarray:
    """Window-only neigh_idx [T, w]."""
    ni = np.full((T, w), -1, dtype=np.int32)
    for i in range(T):
        col = 0
        for j in range(max(0, i - w), i):
            ni[i, col] = j
            col += 1
    return ni


def build_window_plus_cycles(
    T: int, w: int, k: int, rng: np.random.Generator,
) -> np.ndarray:
    """Window + k random Hamiltonian cycles → neigh_idx."""
    adj: list[set[int]] = [set() for _ in range(T)]

    # window edges
    for i in range(T):
        for j in range(max(0, i - w), i):
            adj[i].add(j)
            adj[j].add(i)

    # k random cycles
    for _ in range(k):
        perm = rng.permutation(T)
        for idx in range(T):
            u = int(perm[idx])
            v = int(perm[(idx + 1) % T])
            adj[u].add(v)
            adj[v].add(u)

    # remove self-loops
    for i in range(T):
        adj[i].discard(i)

    max_d = max(len(s) for s in adj) if adj else 1
    ni = np.full((T, max_d), -1, dtype=np.int32)
    for i, s in enumerate(adj):
        nbrs = sorted(s)
        ni[i, : len(nbrs)] = nbrs
    return ni


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def compute_metrics(
    ni: np.ndarray, max_layers: int, rng: np.random.Generator,
) -> dict:
    """Compute both undirected and causal metrics."""
    T = ni.shape[0]

    # Undirected
    gap_info = spectral_gap(ni)
    diam = effective_diameter(ni, num_samples=min(50, T), rng=rng)

    # Causal
    cr = causal_reachability(ni, max_layers=max_layers)
    cd = causal_diameter(ni, num_samples=min(50, T), rng=rng)

    return {
        "spectral_gap": gap_info["spectral_gap"],
        "undir_diameter": diam["max_distance"],
        "causal_diameter": cd["causal_diameter"],
        "causal_unreachable_frac": cd["unreachable_frac"],
        "coverage_by_layer": cr["mean_coverage"],
        "p95_layers": cr["p95_layers_to_full"],
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stress test: cycle benefit under causal masking"
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--num-cycles", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument(
        "--output", type=str,
        default="results/cycle_benefit_stress_test.ndjson",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.seq_lens = [128, 256]
        args.num_cycles = [0, 1, 2]
        args.max_layers = 6

    rng = np.random.default_rng(args.seed)
    results: list[dict] = []

    print(f"{'Config':<30} {'T':>6} {'SpGap':>7} {'UDiam':>6} "
          f"{'CDiam':>6} {'Cov@5':>7} {'p95L':>5}")
    print("-" * 75)

    for T in args.seq_lens:
        for k in args.num_cycles:
            label = f"window_w{args.window}" if k == 0 else (
                f"w{args.window}+{k}cyc"
            )

            t0 = time.time()
            if k == 0:
                ni = build_window_only(T, args.window)
            else:
                ni = build_window_plus_cycles(T, args.window, k, rng)

            metrics = compute_metrics(ni, args.max_layers, rng)
            elapsed = time.time() - t0

            cov5 = (
                metrics["coverage_by_layer"][4]
                if len(metrics["coverage_by_layer"]) > 4
                else metrics["coverage_by_layer"][-1]
            )

            record = {
                "T": T,
                "window": args.window,
                "num_cycles": k,
                "config": label,
                "elapsed_s": round(elapsed, 3),
                **metrics,
            }
            results.append(record)

            p95_str = (
                str(metrics["p95_layers"])
                if metrics["p95_layers"] > 0
                else ">L"
            )
            print(
                f"{label:<30} {T:>6} {metrics['spectral_gap']:>7.4f} "
                f"{metrics['undir_diameter']:>6} "
                f"{metrics['causal_diameter']:>6} "
                f"{cov5:>7.3f} {p95_str:>5}"
            )

    # Summary
    print("\n" + "=" * 75)
    print("KEY QUESTION: Does +1 cycle help causally, not just undirectedly?")
    print("=" * 75)

    for T in args.seq_lens:
        base = [r for r in results if r["T"] == T and r["num_cycles"] == 0]
        plus1 = [r for r in results if r["T"] == T and r["num_cycles"] == 1]
        if base and plus1:
            b, p = base[0], plus1[0]
            layer_idx = min(4, len(b["coverage_by_layer"]) - 1)
            b_cov = b["coverage_by_layer"][layer_idx]
            p_cov = p["coverage_by_layer"][layer_idx]
            lift = p_cov - b_cov
            print(
                f"  T={T:>5}: coverage@L=5 "
                f"{b_cov:.3f} → {p_cov:.3f} "
                f"(+{lift:.3f})"
            )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for r in results:
            # convert numpy arrays to lists for JSON
            row = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    row[k] = v.tolist()
                elif isinstance(v, list) and v and isinstance(v[0], np.floating):
                    row[k] = [float(x) for x in v]
                else:
                    row[k] = v
            f.write(json.dumps(row) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
