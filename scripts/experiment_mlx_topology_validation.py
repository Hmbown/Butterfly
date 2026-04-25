#!/usr/bin/env python3
"""MLX topology validation: characterize permute-window graph properties.

This script extracts block-level reachability from the MLX permute-window attention
graphs and measures support coverage, reachability diameter, and staging properties
using the same validation framework as the block-sparse Butterfly topology experiments.

Goal: Answer whether the MLX permute-window path achieves similar support expansion
to the block-sparse Butterfly topology.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.graph.abi import WayfinderGraphABI
from bna.mlx.attention import build_block_butterfly_layout
from bna.topology import Topology


@dataclass
class MLXTopologyConfig:
    """Configuration for MLX topology validation."""
    topology: str = "permute"
    n_heads: int = 1
    window: int = 64
    landmark_stride: int | None = 64
    strategy: str = "random"
    num_cycles: int = 1
    edge_disjoint: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    block_local_window_blocks: int = 1
    block_partner_count: int = 1
    block_sink_blocks: int = 1
    block_partner_rule: str = "xor"


@dataclass
class BlockLevelMetrics:
    """Metrics computed from block-level reachability."""
    num_blocks: int
    block_size: int
    seq_len: int
    support_coverage_last_row: float
    support_coverage_mean: float
    support_coverage_min: float
    reachability_diameter: int
    degree_mean: float
    degree_max: int
    degree_min: int


def _token_to_block_neighbors(
    neigh_idx: np.ndarray,
    block_size: int,
    seq_len: int,
) -> list[list[int]]:
    """Convert token-level neighbor indices to block-level neighbors.

    For each query block, collect all unique blocks that any token in the block
    attends to. This gives a block-level reachability graph.

    Args:
        neigh_idx: Token-level neighbor indices [T, D] or [H, T, D]
        block_size: Size of each block in tokens
        seq_len: Total sequence length in tokens

    Returns:
        Block-level neighbor lists [num_blocks, ...]
    """
    if neigh_idx.ndim == 3:
        # Multi-head: average across heads (use head 0 for now)
        neigh_idx = neigh_idx[0]

    num_blocks = (seq_len + block_size - 1) // block_size

    block_neighbors: list[set[int]] = [set() for _ in range(num_blocks)]

    for token_idx in range(seq_len):
        block_idx = token_idx // block_size
        neighbors = neigh_idx[token_idx]
        # Filter valid neighbors (>= 0 and < seq_len)
        valid_neighbors = neighbors[(neighbors >= 0) & (neighbors < seq_len)]
        for neighbor_token in valid_neighbors:
            neighbor_block = int(neighbor_token) // block_size
            block_neighbors[block_idx].add(neighbor_block)

    # Convert sets to sorted lists
    return [sorted(neighbors) for neighbors in block_neighbors]


def _compute_reachability(
    block_neighbors: list[list[int]],
    num_blocks: int,
    num_layers: int,
) -> dict[int, set[int]]:
    """Compute reachable set for each block after num_layers.

    Simulates information flow: at each layer, each block receives info from
    all neighbors in that layer's topology, plus what those neighbors could
    already reach.

    Args:
        block_neighbors: Block-level neighbor lists [num_blocks, ...]
        num_blocks: Total number of blocks
        num_layers: Number of layers to simulate

    Returns:
        Dictionary mapping block_idx -> set of reachable blocks
    """
    reachable: dict[int, set[int]] = {i: {i} for i in range(num_blocks)}

    for _ in range(num_layers):
        new_reachable: dict[int, set[int]] = {
            i: set(reachable[i]) for i in range(num_blocks)
        }
        for block_idx in range(num_blocks):
            neighbors = block_neighbors[block_idx]
            for neighbor in neighbors:
                if neighbor < 0 or neighbor >= num_blocks:
                    continue
                new_reachable[block_idx] |= reachable[neighbor]
        reachable = new_reachable

    return reachable


def _measure_block_metrics(
    block_neighbors: list[list[int]],
    num_blocks: int,
    num_layers: int,
) -> BlockLevelMetrics:
    """Measure block-level reachability metrics."""
    # Compute reachability after num_layers
    reachable = _compute_reachability(block_neighbors, num_blocks, num_layers)

    # Support coverage
    coverage_values: list[float] = []
    for block_idx in range(num_blocks):
        expected_prefix = set(range(block_idx + 1))
        coverage = len(reachable[block_idx] & expected_prefix) / len(expected_prefix)
        coverage_values.append(coverage)

    support_coverage_last_row = coverage_values[-1]
    support_coverage_mean = np.mean(coverage_values).item()
    support_coverage_min = np.min(coverage_values).item()

    # Reachability diameter (fewest layers for last block to reach full prefix)
    diameter = None
    for layers in range(1, num_layers + 1):
        reachable_at_l = _compute_reachability(block_neighbors, num_blocks, layers)
        last_block = num_blocks - 1
        if reachable_at_l[last_block] == set(range(num_blocks)):
            diameter = layers
            break

    # Degree statistics
    degrees = [len(neighbors) for neighbors in block_neighbors]
    degree_mean = float(np.mean(degrees))
    degree_max = int(np.max(degrees))
    degree_min = int(np.min(degrees))

    return BlockLevelMetrics(
        num_blocks=num_blocks,
        block_size=0,  # Will be filled by caller
        seq_len=0,  # Will be filled by caller
        support_coverage_last_row=support_coverage_last_row,
        support_coverage_mean=support_coverage_mean,
        support_coverage_min=support_coverage_min,
        reachability_diameter=diameter if diameter is not None else num_layers,
        degree_mean=degree_mean,
        degree_max=degree_max,
        degree_min=degree_min,
    )


def _measure_block_metrics_from_schedule(
    schedule: list[list[list[int]]],
    num_blocks: int,
) -> BlockLevelMetrics:
    reachable: dict[int, set[int]] = {i: {i} for i in range(num_blocks)}
    for block_neighbors in schedule:
        new_reachable = {i: set(reachable[i]) for i in range(num_blocks)}
        for block_idx in range(num_blocks):
            for neighbor in block_neighbors[block_idx]:
                if 0 <= neighbor < num_blocks:
                    new_reachable[block_idx] |= reachable[neighbor]
        reachable = new_reachable

    coverage_values: list[float] = []
    for block_idx in range(num_blocks):
        expected_prefix = set(range(block_idx + 1))
        coverage = len(reachable[block_idx] & expected_prefix) / len(expected_prefix)
        coverage_values.append(coverage)

    diameter = None
    for layers in range(1, len(schedule) + 1):
        partial_reachable: dict[int, set[int]] = {i: {i} for i in range(num_blocks)}
        for block_neighbors in schedule[:layers]:
            next_reachable = {i: set(partial_reachable[i]) for i in range(num_blocks)}
            for block_idx in range(num_blocks):
                for neighbor in block_neighbors[block_idx]:
                    if 0 <= neighbor < num_blocks:
                        next_reachable[block_idx] |= partial_reachable[neighbor]
            partial_reachable = next_reachable
        last_block = num_blocks - 1
        if partial_reachable[last_block] == set(range(num_blocks)):
            diameter = layers
            break

    degrees = [len(neighbors) for neighbors in schedule[0]] if schedule else [0]
    return BlockLevelMetrics(
        num_blocks=num_blocks,
        block_size=0,
        seq_len=0,
        support_coverage_last_row=float(coverage_values[-1]),
        support_coverage_mean=float(np.mean(coverage_values)),
        support_coverage_min=float(np.min(coverage_values)),
        reachability_diameter=diameter if diameter is not None else len(schedule),
        degree_mean=float(np.mean(degrees)),
        degree_max=int(np.max(degrees)),
        degree_min=int(np.min(degrees)),
    )


def _build_mlx_graph(
    seq_len: int,
    cfg: MLXTopologyConfig,
) -> WayfinderGraphABI:
    """Build an MLX-style permute-window graph.

    This mimics the graph construction used in bna/integrations/qwen_mlx.py.
    """
    topology = Topology(
        n_heads=cfg.n_heads,
        strategy=cfg.strategy,
        num_cycles=cfg.num_cycles,
        edge_disjoint=cfg.edge_disjoint,
        regular_num_clusters=cfg.regular_num_clusters,
        seed=cfg.seed,
        window=cfg.window,
        landmark_stride=cfg.landmark_stride,
        enforce_hamiltonian=True,
    )

    return topology.construct({"T": int(seq_len), "include_self": True}).abi


def _build_block_sparse_neighbors(
    seq_len: int,
    block_size: int,
    cfg: MLXTopologyConfig,
    *,
    layer_idx: int,
) -> list[list[int]]:
    layout = build_block_butterfly_layout(
        seq_len=seq_len,
        block_size=block_size,
        num_key_value_heads=1,
        num_key_value_groups=1,
        layer_idx=layer_idx,
        local_window_blocks=cfg.block_local_window_blocks,
        sink_count=cfg.block_sink_blocks,
        partner_count=cfg.block_partner_count,
        partner_rule=cfg.block_partner_rule,
    )
    rows: list[list[int]] = []
    for block_idx in range(layout.num_blocks):
        row = np.asarray(layout.block_neighbors[0, block_idx], dtype=np.int32)
        rows.append(row[row >= 0].tolist())
    return rows


def run_one(
    seq_len: int,
    block_size: int,
    cfg: MLXTopologyConfig,
    num_layers: int,
) -> dict[str, Any]:
    """Run validation for a single configuration."""
    num_blocks = (seq_len + block_size - 1) // block_size
    if cfg.topology == "block_sparse":
        schedule = [
            _build_block_sparse_neighbors(
                seq_len,
                block_size,
                cfg,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_layers)
        ]
        metrics = _measure_block_metrics_from_schedule(schedule, num_blocks)
    else:
        graph_abi = _build_mlx_graph(seq_len, cfg)
        block_neighbors = _token_to_block_neighbors(
            graph_abi.neigh_idx, block_size, seq_len
        )
        metrics = _measure_block_metrics(block_neighbors, num_blocks, num_layers)
    metrics.block_size = block_size
    metrics.seq_len = seq_len

    return {
        "seq_len": seq_len,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "config": {
            "topology": cfg.topology,
            "n_heads": cfg.n_heads,
            "window": cfg.window,
            "landmark_stride": cfg.landmark_stride,
            "strategy": cfg.strategy,
            "num_cycles": cfg.num_cycles,
            "edge_disjoint": cfg.edge_disjoint,
            "regular_num_clusters": cfg.regular_num_clusters,
            "seed": cfg.seed,
            "block_local_window_blocks": cfg.block_local_window_blocks,
            "block_partner_count": cfg.block_partner_count,
            "block_sink_blocks": cfg.block_sink_blocks,
            "block_partner_rule": cfg.block_partner_rule,
        },
        "num_layers": num_layers,
        "metrics": {
            "support_coverage_last_row": metrics.support_coverage_last_row,
            "support_coverage_mean": metrics.support_coverage_mean,
            "support_coverage_min": metrics.support_coverage_min,
            "reachability_diameter": metrics.reachability_diameter,
            "degree_mean": metrics.degree_mean,
            "degree_max": metrics.degree_max,
            "degree_min": metrics.degree_min,
        },
    }


def main() -> None:
    """Run MLX topology validation across multiple configurations."""
    output_dir = Path("results/proof/mlx_topology_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test configurations
    seq_lens = [2048, 4096, 8192]  # Typical MLX sequence lengths
    block_sizes = [64, 128]
    strategies = ["random", "regular_partition"]
    topologies = ["permute", "block_sparse"]

    results = []

    for seq_len in seq_lens:
        for block_size in block_sizes:
            for strategy in strategies:
                for topology in topologies:
                    cfg = MLXTopologyConfig(
                        topology=topology,
                        n_heads=1,
                        window=block_size,  # Match block size
                        landmark_stride=block_size,
                        strategy=strategy,
                        num_cycles=1,
                        edge_disjoint=True,
                        seed=0,
                        block_local_window_blocks=1,
                        block_partner_count=1,
                        block_sink_blocks=1,
                        block_partner_rule="xor",
                    )

                    num_blocks = (seq_len + block_size - 1) // block_size
                    # Use 2 * ceil(log2(num_blocks)) layers (same as Butterfly experiments)
                    num_layers = 2 * int(math.ceil(math.log2(num_blocks)))

                    try:
                        result = run_one(seq_len, block_size, cfg, num_layers)
                        results.append(result)
                        print(
                            f"topology={topology}, seq_len={seq_len}, block_size={block_size}, "
                            f"strategy={strategy}: support_last_row={result['metrics']['support_coverage_last_row']:.3f}"
                        )
                    except Exception as e:
                        print(
                            f"Error for topology={topology}, seq_len={seq_len}, "
                            f"block_size={block_size}, strategy={strategy}: {e}"
                        )

    # Save results
    output_file = output_dir / "summary.json"
    with open(output_file, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary statistics
    full_support_count = sum(
        1 for r in results if r["metrics"]["support_coverage_last_row"] >= 0.999
    )
    print(f"\nFull support coverage (>= 0.999): {full_support_count}/{len(results)}")


if __name__ == "__main__":
    main()
