"""Metal kernel seam for future fused Wayfinder ops.

Current runtime uses MLX Python primitives. This module defines the explicit
signatures expected for a future custom-op drop-in.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


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
