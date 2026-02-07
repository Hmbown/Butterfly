from __future__ import annotations

from hcsa.compiler.graph_ir import GraphIR


def validate_pass(ir: GraphIR) -> GraphIR:
    if ir.degree <= 0:
        raise ValueError(f"degree must be > 0, got {ir.degree}")
    if ir.num_cycles <= 0:
        raise ValueError(f"num_cycles must be > 0, got {ir.num_cycles}")
    if ir.window_size < 0:
        raise ValueError(f"window_size must be >= 0, got {ir.window_size}")
    if ir.landmark_stride is not None and ir.landmark_stride <= 0:
        raise ValueError(f"landmark_stride must be > 0 or None, got {ir.landmark_stride}")
    if ir.permute_window_size < 0:
        raise ValueError(
            f"permute_window_size must be >= 0, got {ir.permute_window_size}"
        )

    valid_strategies = {"random", "greedy", "online_insertion"}
    if ir.strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {sorted(valid_strategies)}, got {ir.strategy}")

    return ir
