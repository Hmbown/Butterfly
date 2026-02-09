"""Metal kernel seam for future fused Wayfinder ops.

Current runtime uses MLX Python primitives. This module defines the explicit
signatures expected for a future custom-op drop-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx


_KERNEL_DIR = Path(__file__).resolve().parent
_ACTIVE_ROW_DISCOVERED = _KERNEL_DIR / "hcsa_active_row_fused_discovered.metal"
_PERMUTE_WINDOW_DISCOVERED = _KERNEL_DIR / "hcsa_permute_window_fused_discovered.metal"


def has_discovered_active_row_kernel() -> bool:
    """True when a post-search K4 artifact has been exported."""
    return _ACTIVE_ROW_DISCOVERED.exists()


def has_discovered_permute_window_kernel() -> bool:
    """True when a post-search K1 artifact has been exported."""
    return _PERMUTE_WINDOW_DISCOVERED.exists()


def sparse_row_attention_fused(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    neigh_idx: mx.array,
    *,
    edge_type: Optional[mx.array] = None,
) -> mx.array:
    """Future Metal fused op seam.

    Expected shapes:
    - q,k,v: [B,H,T,dh]
    - neigh_idx: [H,T,D] int32 with -1 padding
    - edge_type: [H,T,D] uint8 (optional)
    - return: [B,H,T,dh]
    """
    raise NotImplementedError("Metal fused kernel not wired yet; use MLX fallback path")
