"""Metal kernel seam for future fused Wayfinder ops.

Current runtime uses MLX Python primitives. This module defines the explicit
signatures expected for a future custom-op drop-in.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Optional

import mlx.core as mx


_KERNEL_DIR = Path(__file__).resolve().parent
_ACTIVE_ROW_DISCOVERED = _KERNEL_DIR / "hcsa_active_row_fused_discovered.metal"
_PERMUTE_WINDOW_DISCOVERED = _KERNEL_DIR / "hcsa_permute_window_fused_discovered.metal"
_FUSED_ATTENTION_DISCOVERED = _KERNEL_DIR / "hcsa_fused_attention_discovered.metal"


def has_discovered_active_row_kernel() -> bool:
    """True when a post-search K4 artifact has been exported."""
    return _ACTIVE_ROW_DISCOVERED.exists()


def has_discovered_permute_window_kernel() -> bool:
    """True when a post-search K1 artifact has been exported."""
    return _PERMUTE_WINDOW_DISCOVERED.exists()


def has_discovered_fused_attention_kernel() -> bool:
    """True when a post-search K6 artifact has been exported."""
    return _FUSED_ATTENTION_DISCOVERED.exists()


def has_fused_dispatch() -> bool:
    """True when the Python-level fused all-head dispatch is available.

    Always returns True — the fused dispatch is implemented in pure Python/MLX
    (no Metal kernel required). A future K6 Metal artifact would replace the
    Python path but the eligibility gate is independent of the Metal file.
    """
    return True


@cache
def fused_attention_kernel():
    """Load the discovered K6 fused-attention Metal kernel."""
    if not has_discovered_fused_attention_kernel():
        raise FileNotFoundError(
            "K6 fused-attention kernel not found. Expected: "
            f"{_FUSED_ATTENTION_DISCOVERED}"
        )
    try:
        from zmlx.metal import kernel as metal_kernel
        from zmlx.msl import DEFAULT_HEADER
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "zmlx is required to load the discovered fused-attention kernel. "
            "Install ZMLX or set PYTHONPATH to the ZMLX source."
        ) from exc

    # Helper to store float→half/bfloat/float via template deduction
    _K6_HEADER_EXTRA = """
template <typename T>
inline void kk_store(device T* ptr, uint idx, float val) {
    ptr[idx] = T(val);
}
"""
    source = _FUSED_ATTENTION_DISCOVERED.read_text()
    return metal_kernel(
        name="kk_discovered_hcsa_fused_attention",
        input_names=[
            "q",
            "k",
            "v",
            "all_perms",
            "all_inv_perms",
            "query_positions",
            "window",
        ],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER + _K6_HEADER_EXTRA,
        cache=True,
    )


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
