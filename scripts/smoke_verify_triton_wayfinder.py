#!/usr/bin/env python3
"""Smoke verification: confirm the Triton fused kernel is active for HCSA sparse attention.

Builds synthetic Q/K/V with Qwen 3.5-9B geometry, constructs an HCSA graph via Topology,
dispatches through sparse_row_attention_gqa_chunked, and asserts the triton_fused backend
is selected. Compares output against the manual_matmul reference path.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.topology import Topology
from bna.torch.attention_wayfinder_sparse import sparse_row_attention_gqa_chunked


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    # Qwen 3.5-9B geometry
    B, Hkv, groups, T, dh = 1, 4, 4, 256, 256
    Hq = Hkv * groups  # 16

    # --- Build HCSA graph via Topology ---
    topo = Topology(
        n_heads=Hkv,
        strategy="random",
        window=64,
        landmark_stride=64,
        seed=42,
    )
    graph = topo.construct(T)
    abi = graph.abi
    neigh_idx = torch.from_numpy(abi.neigh_idx).to(device=device, dtype=torch.long)
    edge_type = torch.from_numpy(abi.edge_type).to(device=device, dtype=torch.uint8)

    D = abi.max_degree
    print(f"Graph: T={T}, Hkv={Hkv}, D={D} (density={D / T:.1%})")

    # Edge type composition
    et_flat = abi.edge_type.flatten()
    counts = Counter(int(v) for v in et_flat)
    total_nonpad = sum(c for k, c in counts.items() if k != 0)
    print("Edge composition:", {k: c for k, c in sorted(counts.items()) if k != 0},
          f"(total non-pad: {total_nonpad})")

    # --- Build synthetic Q/K/V ---
    torch.manual_seed(42)
    q = torch.randn(B, Hq, T, dh, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, dh, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, dh, device=device, dtype=torch.bfloat16)

    # --- Triton fused path (return_weights=False) ---
    profile_fused: dict = {}
    out_fused, w_fused = sparse_row_attention_gqa_chunked(
        q, k, v,
        neigh_idx=neigh_idx,
        edge_type=edge_type,
        num_key_value_groups=groups,
        return_weights=False,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_profile=profile_fused,
    )

    backend = profile_fused.get("sparse_contraction_backend", "UNKNOWN")
    print(f"Backend: {backend}")

    if backend != "triton_fused":
        print(f"FAIL: expected triton_fused, got {backend}")
        sys.exit(1)

    assert w_fused is None, "return_weights=False should yield None weights"

    # --- Reference path (return_weights=True forces manual_matmul) ---
    profile_ref: dict = {}
    out_ref, w_ref = sparse_row_attention_gqa_chunked(
        q, k, v,
        neigh_idx=neigh_idx,
        edge_type=edge_type,
        num_key_value_groups=groups,
        return_weights=True,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_profile=profile_ref,
    )

    ref_backend = profile_ref.get("sparse_contraction_backend", "UNKNOWN")
    print(f"Reference backend: {ref_backend}")

    # --- Compare outputs ---
    max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
    print(f"Max abs diff (fused vs ref): {max_diff:.6f}")

    if max_diff > 0.05:
        print(f"FAIL: max diff {max_diff:.6f} exceeds tolerance 0.05")
        sys.exit(1)

    print(f"\nPASS — triton_fused confirmed, output matches reference (max_diff={max_diff:.6f})")


if __name__ == "__main__":
    main()
