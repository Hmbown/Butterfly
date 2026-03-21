from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from bna.compiler.graph_ir import GraphIR
from bna.compiler.passes import (
    cache_key_pass,
    emit_cache_artifact_pass,
    lower_to_neighborindex_pass,
    normalize_pass,
    specialize_perm_window_pass,
    validate_pass,
)
from bna.compiler.sexp import load_graph_ir, parse_graph_ir, parse_sexp


def compile_graph_spec(
    spec_path: str | Path,
    *,
    T: int,
    H: int,
    dtype: str = "float16",
    out_root: str | Path = ".cache/wayfinder",
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    ir = validate_pass(normalize_pass(load_graph_ir(spec_path)))
    lowered = lower_to_neighborindex_pass(ir, T=T, H=H)
    perm = specialize_perm_window_pass(abi=lowered["abi"], window=ir.permute_window_size)
    key = cache_key_pass(ir, T=T, H=H, dtype=dtype)

    meta = {
        "ir": ir.to_dict(),
        "cache_payload": key["payload"],
        "cycle_perms": lowered["abi"].meta.get("cycle_perms", []),
        "graph_metrics": lowered["graph_metrics"],
    }
    emitted = emit_cache_artifact_pass(
        neigh_idx=lowered["neigh_idx"],
        edge_type=lowered["edge_type"],
        permute_payload=perm,
        meta=meta,
        cache_hash=key["hash"],
        out_root=out_root,
        out_dir=out_dir,
    )

    return {
        "ir": ir,
        "cache_hash": key["hash"],
        "lowered": lowered,
        "permute": perm,
        "artifact": emitted,
    }


__all__ = [
    "GraphIR",
    "compile_graph_spec",
    "load_graph_ir",
    "parse_graph_ir",
    "parse_sexp",
]
