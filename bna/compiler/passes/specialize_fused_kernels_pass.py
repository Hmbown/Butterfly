"""Compiler pass: determine fused kernel eligibility from GraphIR properties.

Emits a ``kernel_specializations`` dict that downstream passes (e.g.
``emit_cache_artifact_pass``) can persist in ``meta.json``.
"""
from __future__ import annotations

from typing import Any, Dict

from bna.compiler.graph_ir import GraphIR
from bna.mlx.kernels.metal import (
    has_discovered_active_row_kernel,
    has_discovered_permute_window_kernel,
)


def _has_discovered_fused_attention_kernel() -> bool:
    """K6 fused-attention discovery artifact gate (placeholder)."""
    try:
        from pathlib import Path

        _k6_path = (
            Path(__file__).resolve().parents[2]
            / "mlx"
            / "kernels"
            / "metal"
            / "hcsa_fused_attention_discovered.metal"
        )
        return _k6_path.exists()
    except Exception:
        return False


def specialize_fused_kernels_pass(
    ir: GraphIR,
    *,
    permute_payload: Dict[str, Any] | None = None,
    circular: bool = False,
    multi_cycle_mode: str = "average",
    retro_backfill_enabled: bool = False,
    edge_type_bias_active: bool = False,
) -> Dict[str, Any]:
    """Determine which fused dispatch paths are eligible.

    Args:
        ir: The compiled GraphIR.
        permute_payload: output of ``specialize_perm_window_pass`` (optional).
        circular: whether circular windowing is active.
        multi_cycle_mode: "average" or "union".
        retro_backfill_enabled: whether retro-backfill is active.
        edge_type_bias_active: whether non-zero edge-type bias is configured.

    Returns:
        Dict with ``fused_all_head_dispatch``, ``discovered_kernels``, and
        metadata fields for ``meta.json`` emission.
    """
    fused_eligible = (
        ir.permute_window_enabled
        and ir.num_cycles == 1
        and not circular
        and multi_cycle_mode != "union"
        and not retro_backfill_enabled
        and not edge_type_bias_active
    )

    return {
        "fused_all_head_dispatch": bool(fused_eligible),
        "window": int(ir.permute_window_size),
        "circular": bool(circular),
        "multi_cycle_mode": str(multi_cycle_mode),
        "retro_backfill_enabled": bool(retro_backfill_enabled),
        "edge_type_bias_active": bool(edge_type_bias_active),
        "discovered_kernels": {
            "k1_permute_window": has_discovered_permute_window_kernel(),
            "k4_active_row": has_discovered_active_row_kernel(),
            "k6_fused_attention": _has_discovered_fused_attention_kernel(),
        },
    }
