from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

import mlx.core as mx

from hcsa.graph.abi import WayfinderGraphABI, validate_graph_abi


@dataclass(frozen=True)
class MLXGraphABI:
    """MLX view of the Wayfinder graph ABI."""

    neigh_idx: mx.array  # [H, T, D] int32
    edge_type: mx.array  # [H, T, D] uint8
    meta: Dict[str, Any]


def _ensure_3d(arr: np.ndarray, heads: int | None) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,D] or [H,T,D], got {arr.shape}")
    if heads is None:
        heads = 1
    out = np.broadcast_to(arr[None, :, :], (heads, arr.shape[0], arr.shape[1])).copy()
    return out


def to_mlx_graph_abi(
    abi: WayfinderGraphABI,
    *,
    heads: int | None = None,
    validate: bool = True,
) -> MLXGraphABI:
    if validate:
        validate_graph_abi(abi)

    neigh = np.asarray(abi.neigh_idx)
    edge = np.asarray(abi.edge_type)

    neigh = _ensure_3d(neigh, heads)
    edge = _ensure_3d(edge, heads)

    if neigh.dtype != np.int32:
        neigh = neigh.astype(np.int32)
    if edge.dtype != np.uint8:
        edge = edge.astype(np.uint8)

    return MLXGraphABI(
        neigh_idx=mx.array(neigh, dtype=mx.int32),
        edge_type=mx.array(edge, dtype=mx.uint8),
        meta=dict(abi.meta),
    )


def safe_neighbor_idx(neigh_idx: mx.array, seq_len: int) -> mx.array:
    return mx.clip(neigh_idx, 0, seq_len - 1)


def causal_neighbor_mask(neigh_idx: mx.array, seq_len: int) -> mx.array:
    valid = neigh_idx >= 0
    safe = safe_neighbor_idx(neigh_idx, seq_len)
    i_idx = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len, 1)
    return valid & (safe <= i_idx)
