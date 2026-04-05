from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bna.graph.abi import WayfinderGraphABI
from bna.topology import Topology, TopologyGraph
from bna.torch.attention_wayfinder_permute import wayfinder_permute_window_attention, recover_cycle_perms
from bna.torch.triton_fused_sparse_attn import (
    TRITON_AVAILABLE,
    triton_fused_sparse_gqa_attention,
)
from bna.torch.bench_utils import (
    AttentionProfile,
    causal_neighbor_mask,
    load_compiled_graph_abi,
    normalize_graph_tensors,
    now_ms,
    safe_neighbor_idx,
    stable_masked_softmax,
    tensor_nbytes,
)


_GRAPH_CACHE_STORE: Dict[int, "_GraphCache"] = {}


@dataclass(frozen=True)
class _GraphCache:
    abi: WayfinderGraphABI
    neigh_idx: torch.Tensor  # [H,T,D] long
    edge_type: torch.Tensor  # [H,T,D] uint8
    safe_idx: torch.Tensor  # [H,T,D]
    causal_mask: torch.Tensor  # [H,T,D]
    perm: list[torch.Tensor]  # H arrays, each [T]
    inv_perm: list[torch.Tensor]  # H arrays, each [T]
    pi_idx_clamped: list[torch.Tensor]  # H arrays, each [T,W]
    valid_mask: list[torch.Tensor]  # H arrays, each [T,W]
    causal_masks: list[torch.Tensor]  # H arrays, each [T,W]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: str | None = None
    persistent_bytes: int = 0


def _normalize_graph_aux_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    out = torch.as_tensor(tensor, device=device, dtype=dtype)
    if out.ndim == 3:
        if out.shape[0] != n_heads or out.shape[1] != seq_len:
            raise ValueError(f"Expected {name} [H,T,D]=[{n_heads},{seq_len},D], got {tuple(out.shape)}")
        out = out.unsqueeze(0)
    elif out.ndim == 4:
        if out.shape[0] not in (1, batch_size) or out.shape[1] != n_heads or out.shape[2] != seq_len:
            raise ValueError(
                f"Expected {name} [B,H,T,D]=[{batch_size},{n_heads},{seq_len},D], got {tuple(out.shape)}"
            )
    else:
        raise ValueError(f"Expected {name} ndim 3 or 4, got {out.ndim}")

    if out.shape[0] == 1 and batch_size > 1:
        out = out.expand(batch_size, -1, -1, -1)
    return out.contiguous()


def _estimate_sparse_gqa_chunk_temp_mib(
    *,
    query_chunk_size: int,
    kv_head_chunk_size: int,
    num_key_value_groups: int,
    degree: int,
    degree_chunk_size: int,
    head_dim: int,
    kv_element_size: int,
    return_weights: bool,
) -> float:
    degree_eff = max(1, min(int(degree), int(degree_chunk_size)))
    # Peak live bytes only include one gathered neighborhood block at a time.
    # The streamed no-weights path keeps a float32 output accumulator plus
    # per-row max/sum state instead of materializing the full D-wide weights.
    gather_bytes = (
        int(kv_head_chunk_size)
        * int(query_chunk_size)
        * degree_eff
        * int(head_dim)
        * int(kv_element_size)
    )
    score_bytes = (
        int(kv_head_chunk_size)
        * int(num_key_value_groups)
        * int(query_chunk_size)
        * degree_eff
        * 4
    )
    if return_weights:
        weight_bytes = score_bytes
        return float(gather_bytes + score_bytes + weight_bytes) / float(1024 ** 2)
    accum_bytes = (
        int(kv_head_chunk_size)
        * int(num_key_value_groups)
        * int(query_chunk_size)
        * int(head_dim)
        * 4
    )
    normalizer_bytes = (
        int(kv_head_chunk_size)
        * int(num_key_value_groups)
        * int(query_chunk_size)
        * 4
        * 2  # row_max + row_sum in float32
    )
    return float(gather_bytes + score_bytes + accum_bytes + normalizer_bytes) / float(1024 ** 2)


def _resolve_sparse_gqa_chunking(
    *,
    seq_len: int,
    degree: int,
    hkv: int,
    groups: int,
    head_dim: int,
    kv_element_size: int,
    query_chunk_size: int,
    kv_head_chunk_size: int,
    degree_chunk_size: int,
    chunk_temp_budget_mib: float,
    return_weights: bool,
) -> dict[str, Any]:
    manual_query = int(query_chunk_size) > 0
    manual_head = int(kv_head_chunk_size) > 0
    if manual_query and manual_head:
        chunk_mode = "manual"
    elif manual_query:
        chunk_mode = "manual_query_auto_head"
    elif manual_head:
        chunk_mode = "auto_query_manual_head"
    else:
        chunk_mode = "auto"

    if manual_query:
        query_candidates = [min(int(seq_len), int(query_chunk_size))]
    else:
        query_candidates = []
        for candidate in (
            int(seq_len),
            4096,
            3072,
            2048,
            1536,
            1280,
            1024,
            768,
            640,
            512,
            384,
            320,
            256,
            192,
            160,
            128,
            96,
            64,
            48,
            32,
            16,
            8,
            4,
            2,
            1,
        ):
            if candidate <= 0:
                continue
            query_candidates.append(min(int(seq_len), int(candidate)))
        query_candidates = sorted(set(query_candidates), reverse=True)

    if manual_head:
        head_candidates = [min(int(hkv), int(kv_head_chunk_size))]
    else:
        head_candidates = list(range(int(hkv), 0, -1))

    if return_weights or int(degree_chunk_size) <= 0:
        degree_candidates = [max(1, int(degree))]
    else:
        degree_candidates = [max(1, min(int(degree), int(degree_chunk_size)))]

    best_fit: Optional[dict[str, Any]] = None
    best_any: Optional[dict[str, Any]] = None
    budget_exceeded = False

    for head_chunk_eff in head_candidates:
        for query_chunk_eff in query_candidates:
            for degree_chunk_eff in degree_candidates:
                estimated_temp_mib = _estimate_sparse_gqa_chunk_temp_mib(
                    query_chunk_size=int(query_chunk_eff),
                    kv_head_chunk_size=int(head_chunk_eff),
                    num_key_value_groups=int(groups),
                    degree=int(degree),
                    degree_chunk_size=int(degree_chunk_eff),
                    head_dim=int(head_dim),
                    kv_element_size=int(kv_element_size),
                    return_weights=bool(return_weights),
                )
                num_query_chunks = max(1, math.ceil(int(seq_len) / int(query_chunk_eff)))
                num_head_blocks = max(1, math.ceil(int(hkv) / int(head_chunk_eff)))
                num_degree_blocks = max(1, math.ceil(int(degree) / int(degree_chunk_eff)))
                tile_count = int(num_query_chunks * num_head_blocks * num_degree_blocks)
                candidate_meta = {
                    "query_chunk_eff": int(query_chunk_eff),
                    "head_chunk_eff": int(head_chunk_eff),
                    "degree_chunk_eff": int(degree_chunk_eff),
                    "estimated_temp_mib": float(estimated_temp_mib),
                    "num_query_chunks": int(num_query_chunks),
                    "num_head_blocks": int(num_head_blocks),
                    "num_degree_blocks": int(num_degree_blocks),
                    "tile_count": tile_count,
                    # Long-context cost is still led by serial query chunks; the
                    # full tiling product matters, but only after that.
                    "sort_key": (
                        int(num_query_chunks),
                        tile_count,
                        int(num_head_blocks),
                        int(num_degree_blocks),
                        -int(query_chunk_eff),
                        -int(head_chunk_eff),
                        -int(degree_chunk_eff),
                        float(estimated_temp_mib),
                    ),
                }
                if best_any is None or candidate_meta["sort_key"] < best_any["sort_key"]:
                    best_any = candidate_meta
                if float(estimated_temp_mib) <= float(chunk_temp_budget_mib):
                    if best_fit is None or candidate_meta["sort_key"] < best_fit["sort_key"]:
                        best_fit = candidate_meta

    selected = best_fit
    if selected is None:
        budget_exceeded = True
        if best_any is None:
            raise RuntimeError("Failed to resolve sparse GQA chunking candidates")
        selected = best_any

    query_chunk_eff = int(selected["query_chunk_eff"])
    head_chunk_eff = int(selected["head_chunk_eff"])
    degree_chunk_eff = int(selected["degree_chunk_eff"])
    estimated_temp_mib = float(selected["estimated_temp_mib"])
    num_query_chunks = int(selected["num_query_chunks"])
    num_head_blocks = int(selected["num_head_blocks"])
    num_degree_blocks = int(selected["num_degree_blocks"])
    return {
        "sparse_chunk_mode": chunk_mode,
        "sparse_query_chunk_size": int(query_chunk_eff),
        "sparse_kv_head_chunk_size": int(head_chunk_eff),
        "sparse_degree_chunk_size": int(degree_chunk_eff),
        "sparse_num_query_chunks": int(num_query_chunks),
        "sparse_num_head_blocks": int(num_head_blocks),
        "sparse_num_degree_blocks": int(num_degree_blocks),
        "sparse_streamed_degree": bool((not return_weights) and int(degree_chunk_eff) < int(degree)),
        "sparse_chunk_budget_exceeded": bool(budget_exceeded),
        "sparse_estimated_temp_mib": round(float(estimated_temp_mib), 2),
    }


def _gather_kv_head_block_chunk(
    head_states: torch.Tensor,
    idx_chunk: torch.Tensor,
) -> torch.Tensor:
    """Gather a block of KV heads for a [chunk_len, degree] neighborhood once."""
    b, heads, seq_len, dh = head_states.shape
    chunk_len = int(idx_chunk.shape[2])
    degree = int(idx_chunk.shape[3])
    if chunk_len == 0 or degree == 0:
        return head_states.new_empty((b, heads, chunk_len, degree, dh))

    source = head_states.reshape(b * heads, seq_len, dh)
    flat_idx = idx_chunk.reshape(b * heads, chunk_len * degree)
    gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, dh)
    gathered = torch.gather(source, dim=1, index=gather_idx)
    return gathered.reshape(b, heads, chunk_len, degree, dh)


def _resolve_sparse_gqa_contraction_backend(
    *,
    return_weights: bool,
    degree_block_size: int,
    degree: int,
    device: torch.device,
    backend_override: Optional[str] = None,
    allow_triton_auto: bool = True,
) -> str:
    if backend_override is None or backend_override == "auto":
        if allow_triton_auto and not return_weights and TRITON_AVAILABLE and device.type == "cuda":
            return "triton_fused"
        if not return_weights and degree_block_size < degree:
            return "streamed_online_softmax"
        if not return_weights:
            return "sdpa"
        return "manual_matmul"

    backend = str(backend_override)
    if backend not in {"triton_fused", "streamed_online_softmax", "sdpa", "manual_matmul"}:
        raise ValueError(f"Unsupported sparse contraction backend: {backend!r}")
    if return_weights and backend != "manual_matmul":
        raise ValueError(
            f"sparse contraction backend {backend!r} does not support return_weights=True"
        )
    if backend == "triton_fused" and (not TRITON_AVAILABLE or device.type != "cuda"):
        raise ValueError("sparse contraction backend 'triton_fused' requires CUDA + Triton")
    return backend


def _streaming_sparse_gqa_v_chunk(
    q_chunk: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    idx_chunk: torch.Tensor,
    *,
    mask_chunk: torch.Tensor,
    bias_chunk: torch.Tensor | None,
    degree_chunk_size: int,
    scale: float,
) -> torch.Tensor:
    b, heads, groups, chunk_len, dh = q_chunk.shape
    degree = int(idx_chunk.shape[-1])
    if degree == 0:
        return torch.zeros((b, heads, groups, chunk_len, dh), device=q_chunk.device, dtype=v_block.dtype)

    degree_chunk_size = max(1, min(int(degree_chunk_size), degree))
    neg_inf = torch.tensor(-1e30, device=q_chunk.device, dtype=torch.float32)
    q_scaled = q_chunk * scale
    acc = torch.zeros((b, heads, groups, chunk_len, dh), device=q_chunk.device, dtype=torch.float32)
    row_max = torch.full((b, heads, groups, chunk_len), neg_inf, device=q_chunk.device, dtype=torch.float32)
    row_sum = torch.zeros((b, heads, groups, chunk_len), device=q_chunk.device, dtype=torch.float32)

    for degree_start in range(0, degree, degree_chunk_size):
        degree_stop = min(degree, degree_start + degree_chunk_size)
        idx_degree = idx_chunk[..., degree_start:degree_stop]
        mask_degree = mask_chunk[..., degree_start:degree_stop]

        k_g = _gather_kv_head_block_chunk(k_block, idx_degree)
        scores = torch.matmul(
            q_scaled.unsqueeze(-2),
            k_g.unsqueeze(2).transpose(-1, -2),
        ).squeeze(-2).float()
        del k_g

        if bias_chunk is not None:
            scores = scores + bias_chunk[:, :, None, :, degree_start:degree_stop].float()

        masked_scores = torch.where(mask_degree, scores, neg_inf)
        del scores

        block_max = masked_scores.max(dim=-1).values
        new_row_max = torch.maximum(row_max, block_max)
        prev_scale = torch.exp(row_max - new_row_max)
        exp_scores = torch.exp(masked_scores - new_row_max.unsqueeze(-1))
        exp_scores = torch.where(mask_degree, exp_scores, torch.zeros_like(exp_scores))
        del masked_scores

        v_g = _gather_kv_head_block_chunk(v_block, idx_degree)
        acc = acc * prev_scale.unsqueeze(-1) + torch.matmul(
            exp_scores.unsqueeze(-2),
            v_g.unsqueeze(2).float(),
        ).squeeze(-2)
        row_sum = row_sum * prev_scale + exp_scores.sum(dim=-1)
        row_max = new_row_max
        del v_g
        del exp_scores
        del block_max
        del new_row_max
        del prev_scale

    inv_row_sum = torch.where(row_sum > 0.0, row_sum.reciprocal(), torch.zeros_like(row_sum))
    return (acc * inv_row_sum.unsqueeze(-1)).to(dtype=v_block.dtype)


def _sdpa_sparse_gqa_v_chunk(
    q_chunk: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    idx_chunk: torch.Tensor,
    *,
    mask_chunk: torch.Tensor,
    bias_chunk: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    b, heads, groups, chunk_len, dh = q_chunk.shape
    degree = int(idx_chunk.shape[-1])
    if degree == 0:
        return torch.zeros((b, heads, groups, chunk_len, dh), device=q_chunk.device, dtype=v_block.dtype)

    k_g = _gather_kv_head_block_chunk(k_block, idx_chunk)
    v_g = _gather_kv_head_block_chunk(v_block, idx_chunk)

    num_rows = b * heads * chunk_len
    q_sdpa = q_chunk.permute(0, 1, 3, 2, 4).reshape(num_rows, groups, 1, dh)
    k_sdpa = k_g.reshape(num_rows, 1, degree, dh)
    v_sdpa = v_g.reshape(num_rows, 1, degree, dh)
    mask_sdpa = mask_chunk.permute(0, 1, 3, 2, 4).reshape(num_rows, 1, 1, degree)
    valid_rows = mask_sdpa.any(dim=-1, keepdim=True)

    attn_mask: torch.Tensor
    if bias_chunk is None:
        attn_mask = mask_sdpa
    else:
        bias_sdpa = bias_chunk.reshape(num_rows, 1, 1, degree).float()
        attn_mask = bias_sdpa.masked_fill(~mask_sdpa, torch.finfo(torch.float32).min)

    out_sdpa = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )
    out_sdpa = torch.where(valid_rows, out_sdpa, torch.zeros_like(out_sdpa))
    return out_sdpa.reshape(b, heads, chunk_len, groups, dh).permute(0, 1, 3, 2, 4).contiguous()


def build_sparse_edge_bias_tensor(
    edge_type: torch.Tensor,
    *,
    edge_type_bias: torch.Tensor | None = None,
    edge_type_bias_offset: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if edge_type_bias is None and edge_type_bias_offset is None:
        return None

    target_device = edge_type.device if device is None else device
    bias_vec = torch.zeros((4,), device=target_device, dtype=torch.float32)
    if edge_type_bias is not None:
        bias_vec = bias_vec + edge_type_bias.to(device=target_device, dtype=torch.float32)
    if edge_type_bias_offset is not None:
        bias_vec = bias_vec + edge_type_bias_offset.to(device=target_device, dtype=torch.float32)
    full_bias = torch.cat([torch.zeros((1,), device=target_device, dtype=torch.float32), bias_vec])
    return full_bias[edge_type.to(device=target_device).long()]


def sparse_row_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neigh_idx: torch.Tensor,
    edge_type: torch.Tensor | None = None,
    return_weights: bool = False,
    precomputed_safe_idx: torch.Tensor | None = None,
    precomputed_causal_mask: torch.Tensor | None = None,
    edge_type_bias: torch.Tensor | None = None,
    edge_type_bias_offset: torch.Tensor | None = None,
    window_drop_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse-row gather attention reference path using [B,H,T,D] neighbor index.

    Causality is enforced in original token order: neighbor j is valid iff j <= i.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")

    b, h, t, dh = q.shape

    neigh, et = normalize_graph_tensors(
        neigh_idx,
        edge_type,
        batch_size=b,
        n_heads=h,
        seq_len=t,
        device=q.device,
    )
    d = int(neigh.shape[-1])

    if d == 0:
        out = torch.zeros_like(q)
        if return_weights:
            return out, torch.zeros((b, h, t, 0), dtype=torch.float32, device=q.device)
        return out, None

    if precomputed_safe_idx is not None:
        safe_idx = precomputed_safe_idx
        if safe_idx.ndim == 3:
            safe_idx = safe_idx.unsqueeze(0).expand(b, -1, -1, -1)
    else:
        safe_idx = safe_neighbor_idx(neigh, t)

    if precomputed_causal_mask is not None:
        mask = precomputed_causal_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
    else:
        mask = causal_neighbor_mask(neigh, t)

    if window_drop_mask is not None:
        wd = window_drop_mask
        if wd.ndim == 3:
            wd = wd.unsqueeze(0).expand(b, -1, -1, -1)
        mask = mask & wd

    gather_idx = safe_idx.unsqueeze(-1).expand(b, h, t, d, dh)

    k_exp = k.unsqueeze(3).expand(b, h, t, d, dh)
    v_exp = v.unsqueeze(3).expand(b, h, t, d, dh)

    k_g = torch.gather(k_exp, dim=2, index=gather_idx)
    v_g = torch.gather(v_exp, dim=2, index=gather_idx)

    scores = torch.sum(q.unsqueeze(3).float() * k_g.float(), dim=-1) / math.sqrt(float(dh))

    bias_htd = build_sparse_edge_bias_tensor(
        et,
        edge_type_bias=edge_type_bias,
        edge_type_bias_offset=edge_type_bias_offset,
        device=q.device,
    )

    if bias_htd is not None:
        scores = scores + bias_htd.float()

    weights = stable_masked_softmax(scores.float(), mask, dim=-1)
    out = torch.sum(weights.unsqueeze(-1) * v_g.float(), dim=3).to(dtype=v.dtype)

    if return_weights:
        return out, weights
    return out, None


def sparse_row_attention_gqa_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    neigh_idx: torch.Tensor,
    num_key_value_groups: int,
    edge_type: torch.Tensor | None = None,
    return_weights: bool = False,
    precomputed_safe_idx: torch.Tensor | None = None,
    precomputed_causal_mask: torch.Tensor | None = None,
    edge_type_bias: torch.Tensor | None = None,
    edge_type_bias_offset: torch.Tensor | None = None,
    window_drop_mask: torch.Tensor | None = None,
    query_chunk_size: int = 0,
    kv_head_chunk_size: int = 0,
    degree_chunk_size: int = 0,
    chunk_temp_budget_mib: float = 160.0,
    chunk_profile: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse-row attention for grouped-query attention without repeating KV heads.

    The graph is defined on KV heads. Query heads within the same KV group reuse
    the same sparse neighborhood, and query positions are processed in chunks to
    bound the size of temporary gather tensors.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")

    b, hq, t, dh = q.shape
    bk, hkv, tk, dh_k = k.shape
    bv, hkv_v, tv, dh_v = v.shape
    groups = int(num_key_value_groups)
    chunk_size = int(query_chunk_size)
    head_chunk_size = int(kv_head_chunk_size)
    degree_block_size = int(degree_chunk_size)

    if bk != b or bv != b or tk != t or tv != t or dh_k != dh or dh_v != dh:
        raise ValueError("q/k/v must agree on batch, seq_len, and head_dim")
    if hkv != hkv_v:
        raise ValueError("k and v must have the same number of KV heads")
    if groups <= 0:
        raise ValueError("num_key_value_groups must be > 0")
    if chunk_size < 0:
        raise ValueError("query_chunk_size must be >= 0")
    if head_chunk_size < 0:
        raise ValueError("kv_head_chunk_size must be >= 0")
    if degree_block_size < 0:
        raise ValueError("degree_chunk_size must be >= 0")
    if float(chunk_temp_budget_mib) <= 0.0:
        raise ValueError("chunk_temp_budget_mib must be > 0")
    if hq != hkv * groups:
        raise ValueError(
            f"Expected hq == hkv * num_key_value_groups, got {hq} vs {hkv} * {groups}"
        )

    neigh, et = normalize_graph_tensors(
        neigh_idx,
        edge_type,
        batch_size=b,
        n_heads=hkv,
        seq_len=t,
        device=q.device,
    )
    d = int(neigh.shape[-1])

    chunk_meta = _resolve_sparse_gqa_chunking(
        seq_len=t,
        degree=d,
        hkv=hkv,
        groups=groups,
        head_dim=dh,
        kv_element_size=int(k.element_size()),
        query_chunk_size=chunk_size,
        kv_head_chunk_size=head_chunk_size,
        degree_chunk_size=degree_block_size,
        chunk_temp_budget_mib=float(chunk_temp_budget_mib),
        return_weights=bool(return_weights),
    )
    chunk_size = int(chunk_meta["sparse_query_chunk_size"])
    head_chunk_size = int(chunk_meta["sparse_kv_head_chunk_size"])
    degree_block_size = int(chunk_meta["sparse_degree_chunk_size"])
    contraction_backend = _resolve_sparse_gqa_contraction_backend(
        return_weights=bool(return_weights),
        degree_block_size=degree_block_size,
        degree=d,
        device=q.device,
        allow_triton_auto=True,
    )
    if chunk_profile is not None:
        chunk_profile.update(chunk_meta)
        chunk_profile["sparse_contraction_backend"] = contraction_backend

    if d == 0:
        out = torch.zeros_like(q)
        if return_weights:
            return out, torch.zeros((b, hq, t, 0), dtype=torch.float32, device=q.device)
        return out, None

    if precomputed_safe_idx is not None:
        safe_idx = _normalize_graph_aux_tensor(
            precomputed_safe_idx,
            batch_size=b,
            n_heads=hkv,
            seq_len=t,
            device=q.device,
            dtype=torch.long,
            name="precomputed_safe_idx",
        )
    else:
        safe_idx = safe_neighbor_idx(neigh, t)

    if precomputed_causal_mask is not None:
        mask = _normalize_graph_aux_tensor(
            precomputed_causal_mask,
            batch_size=b,
            n_heads=hkv,
            seq_len=t,
            device=q.device,
            dtype=torch.bool,
            name="precomputed_causal_mask",
        )
    else:
        mask = causal_neighbor_mask(neigh, t)

    if window_drop_mask is not None:
        wd = _normalize_graph_aux_tensor(
            window_drop_mask,
            batch_size=b,
            n_heads=hkv,
            seq_len=t,
            device=q.device,
            dtype=torch.bool,
            name="window_drop_mask",
        )
        mask = mask & wd

    bias_htd = build_sparse_edge_bias_tensor(
        et,
        edge_type_bias=edge_type_bias,
        edge_type_bias_offset=edge_type_bias_offset,
        device=q.device,
    )

    scale = 1.0 / math.sqrt(float(dh))

    # ── Triton fused path: single kernel launch, no gathered K/V ──
    if contraction_backend == "triton_fused":
        out = triton_fused_sparse_gqa_attention(
            q, k, v, safe_idx, mask,
            num_key_value_groups=groups,
            scale=scale,
            bias=bias_htd,
        )
        return out, None

    out = torch.empty((b, hq, t, dh), device=q.device, dtype=v.dtype)
    weights_out = (
        torch.empty((b, hq, t, d), device=q.device, dtype=torch.float32)
        if return_weights
        else None
    )
    q_grouped = q.reshape(b, hkv, groups, t, dh)
    out_grouped = out.reshape(b, hkv, groups, t, dh)
    weights_grouped = (
        weights_out.reshape(b, hkv, groups, t, d)
        if weights_out is not None
        else None
    )

    for head_start in range(0, hkv, head_chunk_size):
        head_stop = min(hkv, head_start + head_chunk_size)
        q_block = q_grouped[:, head_start:head_stop]
        k_block = k[:, head_start:head_stop]
        v_block = v[:, head_start:head_stop]
        safe_block = safe_idx[:, head_start:head_stop]
        mask_block = mask[:, head_start:head_stop]
        bias_block = None if bias_htd is None else bias_htd[:, head_start:head_stop]

        for start in range(0, t, chunk_size):
            stop = min(t, start + chunk_size)
            q_chunk = q_block[:, :, :, start:stop, :]
            idx_chunk = safe_block[:, :, start:stop, :]
            mask_chunk = mask_block[:, :, None, start:stop, :]
            if contraction_backend == "streamed_online_softmax":
                out_grouped[:, head_start:head_stop, :, start:stop, :] = _streaming_sparse_gqa_v_chunk(
                    q_chunk,
                    k_block,
                    v_block,
                    idx_chunk,
                    mask_chunk=mask_chunk,
                    bias_chunk=bias_block[:, :, start:stop, :] if bias_block is not None else None,
                    degree_chunk_size=degree_block_size,
                    scale=scale,
                )
                continue

            if contraction_backend == "sdpa":
                out_grouped[:, head_start:head_stop, :, start:stop, :] = _sdpa_sparse_gqa_v_chunk(
                    q_chunk,
                    k_block,
                    v_block,
                    idx_chunk,
                    mask_chunk=mask_chunk,
                    bias_chunk=bias_block[:, :, start:stop, :] if bias_block is not None else None,
                    scale=scale,
                )
                continue

            k_g = _gather_kv_head_block_chunk(k_block, idx_chunk)
            scores = torch.matmul(
                (q_chunk * scale).unsqueeze(-2),
                k_g.unsqueeze(2).transpose(-1, -2),
            ).squeeze(-2).float()
            del k_g

            if bias_block is not None:
                scores = scores + bias_block[:, :, None, start:stop, :].float()

            weights = stable_masked_softmax(scores, mask_chunk, dim=-1)
            del scores

            v_g = _gather_kv_head_block_chunk(v_block, idx_chunk)
            out_grouped[:, head_start:head_stop, :, start:stop, :] = torch.matmul(
                weights.to(dtype=v.dtype).unsqueeze(-2),
                v_g.unsqueeze(2),
            ).squeeze(-2)
            del v_g
            if weights_grouped is not None:
                weights_grouped[:, head_start:head_stop, :, start:stop, :] = weights
            del weights

    if return_weights:
        return out, weights_out
    return out, None


def sparse_row_attention_gqa_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    safe_idx: torch.Tensor,
    causal_mask: torch.Tensor,
    num_key_value_groups: int,
    edge_type: torch.Tensor | None = None,
    return_weights: bool = False,
    edge_type_bias: torch.Tensor | None = None,
    edge_type_bias_offset: torch.Tensor | None = None,
    query_chunk_size: int = 0,
    kv_head_chunk_size: int = 0,
    degree_chunk_size: int = 0,
    chunk_temp_budget_mib: float = 160.0,
    contraction_backend_override: Optional[str] = None,
    chunk_profile: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse-row GQA with precomputed rectangular indices.

    Unlike ``sparse_row_attention_gqa_chunked``, this function supports
    ``q_len != kv_len`` as long as the caller supplies exact absolute token
    indices for each query position in ``safe_idx`` together with a causal mask.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")

    b, hq, tq, dh = q.shape
    bk, hkv, tk, dh_k = k.shape
    bv, hkv_v, tv, dh_v = v.shape
    groups = int(num_key_value_groups)
    chunk_size = int(query_chunk_size)
    head_chunk_size = int(kv_head_chunk_size)
    degree_block_size = int(degree_chunk_size)

    if bk != b or bv != b or tk != tv or dh_k != dh or dh_v != dh:
        raise ValueError("q/k/v must agree on batch, KV seq_len, and head_dim")
    if hkv != hkv_v:
        raise ValueError("k and v must have the same number of KV heads")
    if groups <= 0:
        raise ValueError("num_key_value_groups must be > 0")
    if chunk_size < 0:
        raise ValueError("query_chunk_size must be >= 0")
    if head_chunk_size < 0:
        raise ValueError("kv_head_chunk_size must be >= 0")
    if degree_block_size < 0:
        raise ValueError("degree_chunk_size must be >= 0")
    if float(chunk_temp_budget_mib) <= 0.0:
        raise ValueError("chunk_temp_budget_mib must be > 0")
    if hq != hkv * groups:
        raise ValueError(
            f"Expected hq == hkv * num_key_value_groups, got {hq} vs {hkv} * {groups}"
        )

    safe_idx_norm = _normalize_graph_aux_tensor(
        safe_idx,
        batch_size=b,
        n_heads=hkv,
        seq_len=tq,
        device=q.device,
        dtype=torch.long,
        name="safe_idx",
    )
    mask_norm = _normalize_graph_aux_tensor(
        causal_mask,
        batch_size=b,
        n_heads=hkv,
        seq_len=tq,
        device=q.device,
        dtype=torch.bool,
        name="causal_mask",
    )
    edge_type_norm = None
    if edge_type is not None:
        edge_type_norm = _normalize_graph_aux_tensor(
            edge_type,
            batch_size=b,
            n_heads=hkv,
            seq_len=tq,
            device=q.device,
            dtype=torch.uint8,
            name="edge_type",
        )

    d = int(safe_idx_norm.shape[-1])
    chunk_meta = _resolve_sparse_gqa_chunking(
        seq_len=tq,
        degree=d,
        hkv=hkv,
        groups=groups,
        head_dim=dh,
        kv_element_size=int(k.element_size()),
        query_chunk_size=chunk_size,
        kv_head_chunk_size=head_chunk_size,
        degree_chunk_size=degree_block_size,
        chunk_temp_budget_mib=float(chunk_temp_budget_mib),
        return_weights=bool(return_weights),
    )
    chunk_size = int(chunk_meta["sparse_query_chunk_size"])
    head_chunk_size = int(chunk_meta["sparse_kv_head_chunk_size"])
    degree_block_size = int(chunk_meta["sparse_degree_chunk_size"])
    contraction_backend = _resolve_sparse_gqa_contraction_backend(
        return_weights=bool(return_weights),
        degree_block_size=degree_block_size,
        degree=d,
        device=q.device,
        backend_override=contraction_backend_override,
        allow_triton_auto=False,
    )
    if chunk_profile is not None:
        chunk_profile.update(chunk_meta)
        chunk_profile["sparse_contraction_backend_requested"] = (
            "auto" if contraction_backend_override is None else str(contraction_backend_override)
        )
        chunk_profile["sparse_contraction_backend"] = contraction_backend

    if d == 0:
        out = torch.zeros_like(q)
        if return_weights:
            return out, torch.zeros((b, hq, tq, 0), dtype=torch.float32, device=q.device)
        return out, None

    bias_htd = (
        build_sparse_edge_bias_tensor(
            edge_type_norm,
            edge_type_bias=edge_type_bias,
            edge_type_bias_offset=edge_type_bias_offset,
            device=q.device,
        )
        if edge_type_norm is not None
        else None
    )
    scale = 1.0 / math.sqrt(float(dh))

    if contraction_backend == "triton_fused":
        out = triton_fused_sparse_gqa_attention(
            q,
            k,
            v,
            safe_idx_norm,
            mask_norm,
            num_key_value_groups=groups,
            scale=scale,
            bias=bias_htd,
        )
        return out, None

    out = torch.empty((b, hq, tq, dh), device=q.device, dtype=v.dtype)
    weights_out = (
        torch.empty((b, hq, tq, d), device=q.device, dtype=torch.float32)
        if return_weights
        else None
    )
    q_grouped = q.reshape(b, hkv, groups, tq, dh)
    out_grouped = out.reshape(b, hkv, groups, tq, dh)
    weights_grouped = (
        weights_out.reshape(b, hkv, groups, tq, d)
        if weights_out is not None
        else None
    )

    for head_start in range(0, hkv, head_chunk_size):
        head_stop = min(hkv, head_start + head_chunk_size)
        q_block = q_grouped[:, head_start:head_stop]
        k_block = k[:, head_start:head_stop]
        v_block = v[:, head_start:head_stop]
        idx_block = safe_idx_norm[:, head_start:head_stop]
        mask_block = mask_norm[:, head_start:head_stop]
        bias_block = None if bias_htd is None else bias_htd[:, head_start:head_stop]

        for start in range(0, tq, chunk_size):
            stop = min(tq, start + chunk_size)
            q_chunk = q_block[:, :, :, start:stop, :]
            idx_chunk = idx_block[:, :, start:stop, :]
            mask_chunk = mask_block[:, :, None, start:stop, :]
            if contraction_backend == "streamed_online_softmax":
                out_grouped[:, head_start:head_stop, :, start:stop, :] = _streaming_sparse_gqa_v_chunk(
                    q_chunk,
                    k_block,
                    v_block,
                    idx_chunk,
                    mask_chunk=mask_chunk,
                    bias_chunk=bias_block[:, :, start:stop, :] if bias_block is not None else None,
                    degree_chunk_size=degree_block_size,
                    scale=scale,
                )
                continue

            if contraction_backend == "sdpa":
                out_grouped[:, head_start:head_stop, :, start:stop, :] = _sdpa_sparse_gqa_v_chunk(
                    q_chunk,
                    k_block,
                    v_block,
                    idx_chunk,
                    mask_chunk=mask_chunk,
                    bias_chunk=bias_block[:, :, start:stop, :] if bias_block is not None else None,
                    scale=scale,
                )
                continue

            k_g = _gather_kv_head_block_chunk(k_block, idx_chunk)
            scores = torch.matmul(
                (q_chunk * scale).unsqueeze(-2),
                k_g.unsqueeze(2).transpose(-1, -2),
            ).squeeze(-2).float()
            del k_g

            if bias_block is not None:
                scores = scores + bias_block[:, :, None, start:stop, :].float()

            weights = stable_masked_softmax(scores, mask_chunk, dim=-1)
            del scores

            v_g = _gather_kv_head_block_chunk(v_block, idx_chunk)
            out_grouped[:, head_start:head_stop, :, start:stop, :] = torch.matmul(
                weights.to(dtype=v.dtype).unsqueeze(-2),
                v_g.unsqueeze(2),
            ).squeeze(-2)
            del v_g
            if weights_grouped is not None:
                weights_grouped[:, head_start:head_stop, :, start:stop, :] = weights
            del weights

    if return_weights:
        return out, weights_out
    return out, None


class WayfinderAttentionTorch(nn.Module):
    """PyTorch Butterfly attention with shared graph ABI and cache semantics."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        *,
        routing_dim: Optional[int] = None,
        dropout: float = 0.0,
        window: int = 64,
        landmark_stride: Optional[int] = 64,
        strategy: Literal["random", "greedy", "online_insertion"] = "random",
        num_cycles: int = 1,
        seed: int = 0,
        path: Literal["sparse", "permute"] = "sparse",
        edge_bias: bool = False,
        window_drop: float = 0.0,
        compiled_graph_dir: Optional[str] = None,
    ):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = self.n_embd // self.n_heads
        self.routing_dim = int(routing_dim or self.head_dim)

        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.strategy = strategy
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self.path = path
        self.window_drop_prob = float(window_drop)
        self.compiled_graph_dir = compiled_graph_dir

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wr = nn.Linear(self.n_embd, self.n_heads * self.routing_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if edge_bias:
            self.edge_type_bias = nn.Parameter(torch.zeros((4,), dtype=torch.float32))
        else:
            self.register_parameter("edge_type_bias", None)

        self.topology = Topology(
            n_heads=self.n_heads,
            strategy=self.strategy,
            num_cycles=self.num_cycles,
            seed=self.seed,
            window=self.window,
            landmark_stride=self.landmark_stride,
            enforce_hamiltonian=True,
        )

        self.last_profile = AttentionProfile(path=path)
        self.last_graph_abi: Optional[WayfinderGraphABI] = None
        self._runtime_window_drop_override: Optional[float] = None
        self._runtime_schedule_bias_vec = torch.zeros((4,), dtype=torch.float32)

    def set_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        self._runtime_window_drop_override = (
            None if window_drop is None else float(min(1.0, max(0.0, window_drop)))
        )

        bias = torch.zeros((4,), dtype=torch.float32)
        if schedule_bias:
            mapping = {"cycle": 0, "window": 1, "landmark": 2, "rewire": 3}
            for k, v in schedule_bias.items():
                idx = mapping.get(str(k).lower())
                if idx is not None:
                    bias[idx] = float(v)
        self._runtime_schedule_bias_vec = bias

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec = torch.zeros((4,), dtype=torch.float32)

    def cache_persistent_bytes(self) -> int:
        cache = _GRAPH_CACHE_STORE.get(id(self))
        return int(cache.persistent_bytes) if cache is not None else 0

    @property
    def _cache_mode(self) -> str:
        return self.topology.cache_mode

    def _cache_key_from_t(self, t: int, device: torch.device) -> tuple:
        return (
            int(t),
            self.strategy,
            self.num_cycles,
            self.window,
            self.landmark_stride,
            self.seed,
            self.path,
            str(Path(self.compiled_graph_dir).resolve()) if self.compiled_graph_dir else None,
            str(device),
        )

    def _build_cache(
        self,
        abi: WayfinderGraphABI,
        *,
        t: int,
        device: torch.device,
        cache_key: tuple,
        source: str,
        artifact_dir: str | None = None,
    ) -> _GraphCache:
        neigh = torch.as_tensor(abi.neigh_idx, dtype=torch.long, device=device)
        edge = torch.as_tensor(abi.edge_type, dtype=torch.uint8, device=device)

        safe = safe_neighbor_idx(neigh, t)
        causal = causal_neighbor_mask(neigh, t)

        perms = recover_cycle_perms(neigh, edge, meta=abi.meta)
        w = 2 * self.window + 1
        offsets = torch.arange(-self.window, self.window + 1, device=device, dtype=torch.long)

        perm_list: list[torch.Tensor] = []
        inv_list: list[torch.Tensor] = []
        pi_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        causal_list: list[torch.Tensor] = []

        for head in range(self.n_heads):
            perm_h = perms[head] if head < len(perms) else None
            if perm_h is None:
                raise ValueError(f"Could not recover cycle permutation for head {head}")

            perm_t = torch.as_tensor(perm_h, dtype=torch.long, device=device)
            inv_t = torch.argsort(perm_t)

            pi_idx = torch.arange(t, device=device, dtype=torch.long).view(t, 1) + offsets.view(1, w)
            valid = (pi_idx >= 0) & (pi_idx < t)
            pi_clamped = pi_idx.clamp(0, t - 1)

            neigh_orig = perm_t[pi_clamped]
            query_orig = perm_t.view(t, 1)
            causal_h = neigh_orig <= query_orig

            perm_list.append(perm_t)
            inv_list.append(inv_t)
            pi_list.append(pi_clamped)
            valid_list.append(valid)
            causal_list.append(causal_h)

        persistent_bytes = tensor_nbytes(neigh) + tensor_nbytes(edge) + tensor_nbytes(safe) + tensor_nbytes(causal)
        for arr in perm_list + inv_list + pi_list + valid_list + causal_list:
            persistent_bytes += tensor_nbytes(arr)

        return _GraphCache(
            abi=abi,
            neigh_idx=neigh,
            edge_type=edge,
            safe_idx=safe,
            causal_mask=causal,
            perm=perm_list,
            inv_perm=inv_list,
            pi_idx_clamped=pi_list,
            valid_mask=valid_list,
            causal_masks=causal_list,
            cache_key=cache_key,
            source=source,
            artifact_dir=artifact_dir,
            persistent_bytes=int(persistent_bytes),
        )

    def _build_graph_abi(self, x: torch.Tensor) -> WayfinderGraphABI:
        _b, t, _c = x.shape

        routing_by_head: list[torch.Tensor] | None = None
        if self.strategy in {"greedy", "online_insertion"}:
            r_tensor = self.Wr(x[0]).reshape(t, self.n_heads, self.routing_dim).permute(1, 0, 2)
            r_tensor = r_tensor.detach().float().cpu()
            routing_by_head = [r_tensor[h] for h in range(self.n_heads)]

        topo_graph = self.topology.construct(
            {"T": int(t), "include_self": True},
            routing_by_head=routing_by_head,
        )
        abi = topo_graph.abi
        self.last_graph_abi = abi
        return abi

    def _load_compiled_cache(self, *, t: int, device: torch.device, cache_key: tuple) -> _GraphCache | None:
        if not self.compiled_graph_dir:
            return None

        abi = load_compiled_graph_abi(self.compiled_graph_dir, n_heads=self.n_heads, seq_len=t)
        if abi is None:
            return None

        self.last_graph_abi = abi
        return self._build_cache(
            abi,
            t=t,
            device=device,
            cache_key=cache_key,
            source="compiled",
            artifact_dir=str(Path(self.compiled_graph_dir).resolve()),
        )

    def _get_or_build_cache(self, x: torch.Tensor) -> tuple[_GraphCache, bool]:
        _b, t, _c = x.shape
        cache_key = self._cache_key_from_t(int(t), x.device)

        existing = _GRAPH_CACHE_STORE.get(id(self))
        if self._cache_mode == "static" and existing is not None and existing.cache_key == cache_key:
            return existing, True

        compiled_cache = self._load_compiled_cache(t=int(t), device=x.device, cache_key=cache_key)
        if compiled_cache is not None:
            if self._cache_mode == "static":
                _GRAPH_CACHE_STORE[id(self)] = compiled_cache
            return compiled_cache, False

        abi = self._build_graph_abi(x)
        cache = self._build_cache(
            abi,
            t=int(t),
            device=x.device,
            cache_key=cache_key,
            source="runtime",
        )
        if self._cache_mode == "static":
            _GRAPH_CACHE_STORE[id(self)] = cache
        return cache, False

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_debug: bool = False,
        topology_graph: Optional[TopologyGraph] = None,
    ):
        t_total0 = now_ms()
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        t_graph0 = now_ms()
        if topology_graph is not None:
            cache = self._build_cache(
                topology_graph.abi,
                t=int(t),
                device=x.device,
                cache_key=("injected", int(t), self.path, str(x.device)),
                source=topology_graph.source,
                artifact_dir=topology_graph.artifact_dir,
            )
            cache_hit = False
        else:
            cache, cache_hit = self._get_or_build_cache(x)
        graph_ms = now_ms() - t_graph0

        is_training = self.training
        effective_window_drop = (
            self.window_drop_prob
            if self._runtime_window_drop_override is None
            else self._runtime_window_drop_override
        )

        schedule_bias = self._runtime_schedule_bias_vec.to(device=x.device, dtype=torch.float32)
        if float(schedule_bias.abs().sum().item()) == 0.0:
            schedule_bias = None

        wd_mask: Optional[torch.Tensor] = None
        if is_training and effective_window_drop > 0.0 and self.path == "sparse":
            et = cache.edge_type
            is_window = et == 2
            i_idx = torch.arange(t, device=x.device).view(1, t, 1)
            is_self = cache.safe_idx == i_idx
            droppable = is_window & (~is_self)
            drop_rand = torch.rand(et.shape, device=x.device) < float(effective_window_drop)
            wd_mask = ~(droppable & drop_rand)

        edge_bias_scalar: float | None = None
        if self.edge_type_bias is not None and self.path == "permute":
            edge_bias_scalar = float(self.edge_type_bias[0].item())
        if schedule_bias is not None and self.path == "permute":
            cycle_bias = float(schedule_bias[0].item())
            edge_bias_scalar = cycle_bias if edge_bias_scalar is None else edge_bias_scalar + cycle_bias

        t_attn0 = now_ms()
        permute_ms = 0.0
        if self.path == "sparse":
            y_h, w = sparse_row_attention(
                q,
                k,
                v,
                neigh_idx=cache.neigh_idx,
                edge_type=cache.edge_type,
                return_weights=return_debug,
                precomputed_safe_idx=cache.safe_idx,
                precomputed_causal_mask=cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                edge_type_bias_offset=schedule_bias,
                window_drop_mask=wd_mask,
            )
        elif self.path == "permute":
            y_h, w, permute_ms, _attn_inner = wayfinder_permute_window_attention(
                q,
                k,
                v,
                window=self.window,
                neigh_idx=cache.neigh_idx,
                edge_type=cache.edge_type,
                graph_meta=cache.abi.meta,
                return_weights=return_debug,
                cache=cache,
                edge_type_bias_scalar=edge_bias_scalar,
                window_drop_prob=effective_window_drop if is_training else 0.0,
                training=is_training,
            )
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = now_ms() - t_attn0

        y = y_h.transpose(1, 2).contiguous().view(b, t, c)
        y = self.out(y)
        y = self.dropout(y)

        total_ms = now_ms() - t_total0
        self.last_profile = AttentionProfile(
            graph_build_ms=graph_ms,
            permute_ms=permute_ms,
            attention_ms=attn_ms,
            total_ms=total_ms,
            path=self.path,
            notes={
                "seq_len": int(t),
                "max_degree": int(cache.neigh_idx.shape[-1]),
                "cache_hit": bool(cache_hit),
                "cache_mode": self._cache_mode,
                "cache_source": cache.source,
                "cache_persistent_bytes": int(cache.persistent_bytes),
                "window_drop_effective": float(effective_window_drop),
            },
        )

        if return_debug:
            debug = {
                "graph_abi": self.last_graph_abi,
                "profile": self.last_profile.to_dict(),
                "attn_weights": w,
                "edge_type": cache.edge_type,
                "neigh_idx": cache.neigh_idx,
            }
            return y, debug

        return y
