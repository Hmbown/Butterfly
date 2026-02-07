from __future__ import annotations

from typing import Dict

import numpy as np

from hcsa.graph.abi import EdgeType


def edge_utilization_by_type(
    attn_weights: np.ndarray,
    edge_type: np.ndarray,
) -> Dict[str, float]:
    """Aggregate attention mass by edge type.

    attn_weights: [B,H,T,D]
    edge_type:    [H,T,D]
    """
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be [B,H,T,D], got {attn_weights.shape}")
    if edge_type.ndim != 3:
        raise ValueError(f"edge_type must be [H,T,D], got {edge_type.shape}")

    if attn_weights.shape[1:] != edge_type.shape:
        raise ValueError(
            f"Shape mismatch: weights {attn_weights.shape} vs edge_type {edge_type.shape}"
        )

    total = float(attn_weights.sum())
    if total <= 0:
        return {k: 0.0 for k in ["cycle", "window", "landmark", "rewire"]}

    out: Dict[str, float] = {}
    mapping = {
        "cycle": int(EdgeType.CYCLE),
        "window": int(EdgeType.WINDOW),
        "landmark": int(EdgeType.LANDMARK),
        "rewire": int(EdgeType.REWIRE),
    }
    for name, code in mapping.items():
        m = (edge_type == code)[None, ...]
        mass = float((attn_weights * m).sum())
        out[name] = mass / total
    return out


def largest_intermediate_bytes(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    dh: int,
    path: str,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """Best-effort tensor size proxy for benchmark reporting."""
    if path == "dense":
        scores = B * H * T * T * 4  # fp32 logits
        weights = B * H * T * T * 4
        return {
            "scores_fp32": int(scores),
            "weights_fp32": int(weights),
            "largest": int(max(scores, weights)),
        }

    gather = B * H * T * D * dh * dtype_bytes
    scores = B * H * T * D * 4
    weights = B * H * T * D * 4
    return {
        "kv_gather": int(gather),
        "scores_fp32": int(scores),
        "weights_fp32": int(weights),
        "largest": int(max(gather, scores, weights)),
    }
