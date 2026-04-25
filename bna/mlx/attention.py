from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

import mlx.core as mx
import mlx.nn as nn

from bna.graph.abi import EdgeType, WayfinderGraphABI, validate_graph_abi
from bna.graph.analysis import expansion_proxy, spectral_gap
from bna.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)
from bna.mlx.fused_attention import (
    _fused_active_dispatch_eligible,
    _fused_dispatch_eligible,
)
from bna.topology import Topology, TopologyGraph
from bna.topology.butterfly import (
    ButterflyLayoutMetadata,
    build_butterfly_neighbor_role_row,
    butterfly_layout_metadata,
)


NEG_INF = mx.array(-1e30, dtype=mx.float32)
EPS = mx.array(1e-9, dtype=mx.float32)

# Module-level store for graph caches — keeps mx.arrays out of nn.Module
# parameter trees so optimizers don't try to update them.
_GRAPH_CACHE_STORE: Dict[int, "_GraphCache"] = {}

# Lightweight instrumentation for compressed-butterfly diagnosis.
# Set BNA_COMPRESS_PROFILE=1 to collect counters.
_COMPRESS_PROFILE = os.environ.get("BNA_COMPRESS_PROFILE", "0") == "1"
_COMPRESS_FORCE_MANUAL = os.environ.get("BNA_COMPRESS_FORCE_MANUAL", "0") == "1"
_compress_stats: Dict[str, Any] = {"calls": 0, "summary_ms": 0.0, "attn_ms": 0.0, "shapes": []}


def _compress_profile_reset() -> None:
    _compress_stats["calls"] = 0
    _compress_stats["summary_ms"] = 0.0
    _compress_stats["attn_ms"] = 0.0
    _compress_stats["shapes"] = []


def _compress_profile_dump() -> Dict[str, Any]:
    return dict(_compress_stats)


def _maybe_profile_block_mean_summaries(x: mx.array, *, seq_len: int, block_size: int, num_blocks: int) -> mx.array:
    if not _COMPRESS_PROFILE:
        return _block_mean_summaries_impl(x, seq_len=seq_len, block_size=block_size, num_blocks=num_blocks)
    t0 = time.perf_counter()
    out = _block_mean_summaries_impl(x, seq_len=seq_len, block_size=block_size, num_blocks=num_blocks)
    mx.eval(out)
    elapsed = (time.perf_counter() - t0) * 1000.0
    _compress_stats["calls"] += 1
    _compress_stats["summary_ms"] += elapsed
    _compress_stats["shapes"].append({
        "seq_len": int(seq_len),
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
    })
    return out


@dataclass(frozen=True)
class _GraphCache:
    """Precomputed graph artifacts for cache-hit fast path."""

    mlx_graph: MLXGraphABI
    numpy_abi: WayfinderGraphABI
    # sparse path artifacts
    safe_idx: mx.array  # [H, T, D]
    causal_mask: mx.array  # [H, T, D]
    # permute path artifacts
    perm_mx: List[mx.array]  # H arrays, each [T]
    inv_perm: List[mx.array]  # H arrays, each [T]
    all_perms: mx.array  # [H, T] or [H, d, T]
    all_inv_perms: mx.array  # [H, T] or [H, d, T]
    pi_idx_clamped: List[mx.array]  # H arrays, each [T, W]
    valid_mask: List[mx.array]  # H arrays, each [T, W]
    causal_masks: List[mx.array]  # H arrays, each [T, W]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: str | None = None
    persistent_bytes: int = 0


@dataclass(frozen=True)
class BlockSparseButterflyLayout:
    """Static block-sparse Butterfly neighborhood for MLX attention kernels."""

    seq_len: int
    block_size: int
    num_blocks: int
    block_neighbors: mx.array  # [Hq, N, K]
    block_mask: mx.array  # [Hq, N, N]
    block_token_idx: mx.array  # [N, K * block_size]
    block_causal_mask: mx.array  # [N, block_size, K * block_size]
    metadata: ButterflyLayoutMetadata
    topology_name: str = "butterfly"

    @property
    def sink_blocks(self) -> tuple[int, ...]:
        return tuple(int(block_idx) for block_idx in self.metadata.sink_blocks)

    @property
    def stage_idx(self) -> int:
        return int(self.metadata.stage_idx)

    @property
    def stage_count(self) -> int:
        return int(self.metadata.stage_count)

    @property
    def local_window_blocks(self) -> int:
        return int(self.metadata.local_window_blocks)

    @property
    def partner_rule(self) -> str:
        return str(self.metadata.partner_rule)

    @property
    def partner_count(self) -> int:
        return int(self.metadata.partner_count)


@dataclass
class AttentionProfile:
    graph_build_ms: float = 0.0
    permute_ms: float = 0.0
    attention_ms: float = 0.0
    total_ms: float = 0.0
    path: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "graph_build_ms": float(self.graph_build_ms),
            "permute_ms": float(self.permute_ms),
            "attention_ms": float(self.attention_ms),
            "total_ms": float(self.total_ms),
            "path": self.path,
        }
        out.update(self.notes)
        return out


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _mx_nbytes(arr: mx.array) -> int:
    mx.eval(arr)
    return int(np.asarray(arr).nbytes)


def _schedule_bias_to_vec(schedule_bias: Optional[Dict[str, float]]) -> np.ndarray:
    vec = np.zeros((4,), dtype=np.float32)
    if schedule_bias is None:
        return vec

    mapping = {"cycle": 0, "window": 1, "landmark": 2, "rewire": 3}
    for k, v in schedule_bias.items():
        idx = mapping.get(str(k).lower())
        if idx is None:
            continue
        vec[idx] = float(v)
    return vec


def _neighbor_roles_to_edge_type(roles: tuple[str, ...]) -> int:
    if "partner" in roles:
        return int(EdgeType.REWIRE)
    if "sink" in roles:
        return int(EdgeType.LANDMARK)
    return int(EdgeType.WINDOW)


def build_block_butterfly_layout(
    *,
    seq_len: int,
    block_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    layer_idx: int,
    local_window_blocks: int = 4,
    sink_count: int = 1,
    partner_count: int = 1,
    partner_rule: str = "xor",
) -> BlockSparseButterflyLayout:
    """Build a staged Butterfly block topology for MLX block-sparse attention."""
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    if int(num_key_value_heads) <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if int(num_key_value_groups) <= 0:
        raise ValueError("num_key_value_groups must be positive")
    if int(local_window_blocks) < 0:
        raise ValueError("local_window_blocks must be >= 0")
    if int(sink_count) < 0:
        raise ValueError("sink_count must be >= 0")
    if int(partner_count) < 0:
        raise ValueError("partner_count must be >= 0")

    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    metadata = butterfly_layout_metadata(
        num_blocks=int(num_blocks),
        layer_idx=int(layer_idx),
        partner_rule=str(partner_rule),
        partner_count=int(partner_count),
        sink_count=int(sink_count),
        local_window_blocks=int(local_window_blocks),
    )

    row_specs_by_block = [
        build_butterfly_neighbor_role_row(block_idx=block_idx, metadata=metadata)
        for block_idx in range(int(num_blocks))
    ]
    max_neighbors = max(1, max(len(row_specs) for row_specs in row_specs_by_block))
    num_query_heads = int(num_key_value_heads) * int(num_key_value_groups)

    block_neighbors = np.full(
        (int(num_query_heads), int(num_blocks), int(max_neighbors)),
        -1,
        dtype=np.int32,
    )
    block_mask = np.zeros(
        (int(num_query_heads), int(num_blocks), int(num_blocks)),
        dtype=np.bool_,
    )

    key_tokens_per_block = int(max_neighbors) * int(block_size)
    safe_fill = max(int(seq_len) - 1, 0)
    block_token_idx = np.full(
        (int(num_blocks), int(key_tokens_per_block)),
        safe_fill,
        dtype=np.int32,
    )
    block_causal_mask = np.zeros(
        (int(num_blocks), int(block_size), int(key_tokens_per_block)),
        dtype=np.bool_,
    )

    for block_idx, row_specs in enumerate(row_specs_by_block):
        neighbor_ids = [int(spec.neighbor) for spec in row_specs]
        if neighbor_ids:
            block_neighbors[:, block_idx, : len(neighbor_ids)] = np.asarray(
                neighbor_ids,
                dtype=np.int32,
            )
            block_mask[:, block_idx, np.asarray(neighbor_ids, dtype=np.int32)] = True

        q_start = int(block_idx) * int(block_size)
        q_end = min(int(seq_len), q_start + int(block_size))
        q_count = max(0, q_end - q_start)
        if q_count <= 0:
            continue
        q_positions = np.arange(q_start, q_end, dtype=np.int32)

        for spec_idx, spec in enumerate(row_specs):
            tok_start = int(spec.neighbor) * int(block_size)
            tok_end = min(int(seq_len), tok_start + int(block_size))
            tok_count = max(0, tok_end - tok_start)
            if tok_count <= 0:
                continue
            base = int(spec_idx) * int(block_size)
            token_positions = np.arange(tok_start, tok_end, dtype=np.int32)
            block_token_idx[block_idx, base : base + tok_count] = token_positions
            block_causal_mask[block_idx, :q_count, base : base + tok_count] = (
                token_positions[None, :] <= q_positions[:, None]
            )

    return BlockSparseButterflyLayout(
        seq_len=int(seq_len),
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        block_neighbors=mx.array(block_neighbors, dtype=mx.int32),
        block_mask=mx.array(block_mask, dtype=mx.bool_),
        block_token_idx=mx.array(block_token_idx, dtype=mx.int32),
        block_causal_mask=mx.array(block_causal_mask, dtype=mx.bool_),
        metadata=metadata,
    )


def _swa_stream_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    n_win: int,
    scale: float,
    q_chunk: int = 256,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Sliding-window attention stream returning `(o, l, m)` tuples for online merge.

    Each query position `t` attends to keys at positions
    `[max(0, t - n_win + 1), t]` (causal + window). The result is computed in
    chunks of `q_chunk` queries to bound peak memory; `mx.eval` is called per
    chunk so the lazy graph does not retain across chunks.

    Returns:
      o: [B, H, T, dh] in q's dtype.
      l: [B, H, T, 1] in float32.  Sum_j exp(score_j - m).
      m: [B, H, T, 1] in float32.  Per-row score max.
    """
    if q.shape[2] == 0:
        zero = mx.zeros_like(q)
        ld = mx.zeros((*q.shape[:3], 1), dtype=mx.float32)
        md = mx.full((*q.shape[:3], 1), -1e30, dtype=mx.float32)
        return zero, ld, md
    B, H, T, dh = q.shape
    n_win = max(1, int(n_win))
    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    m_chunks: list[mx.array] = []
    out_dtype = q.dtype
    for start in range(0, T, q_chunk):
        end = min(start + q_chunk, T)
        q_c = q[:, :, start:end, :]
        k_lo = max(0, start - n_win + 1)
        k_hi = end
        k_local = k[:, :, k_lo:k_hi, :]
        v_local = v[:, :, k_lo:k_hi, :]

        q_pos = mx.arange(start, end, dtype=mx.int32).reshape(end - start, 1)
        k_pos = mx.arange(k_lo, k_hi, dtype=mx.int32).reshape(1, k_hi - k_lo)
        mask_2d = (k_pos <= q_pos) & (k_pos > q_pos - n_win)  # [c, K_local]
        mask = mask_2d.reshape(1, 1, end - start, k_hi - k_lo)

        scores = mx.matmul(
            q_c.astype(mx.float32),
            k_local.transpose(0, 1, 3, 2).astype(mx.float32),
        ) * scale
        scores = mx.where(mask, scores, NEG_INF)
        m_c = mx.max(scores, axis=-1, keepdims=True)
        e = mx.exp(scores - m_c)
        e = mx.where(mask, e, mx.zeros_like(e))
        l_c = mx.sum(e, axis=-1, keepdims=True)
        o_c = mx.matmul(e, v_local.astype(mx.float32)) / mx.maximum(l_c, EPS)
        mx.eval(o_c, l_c, m_c)
        o_chunks.append(o_c)
        l_chunks.append(l_c)
        m_chunks.append(m_c)

    if len(o_chunks) == 1:
        o, l, m = o_chunks[0], l_chunks[0], m_chunks[0]
    else:
        o = mx.concatenate(o_chunks, axis=2)
        l = mx.concatenate(l_chunks, axis=2)
        m = mx.concatenate(m_chunks, axis=2)
    return o.astype(out_dtype), l, m


def _compressed_stream_attention(
    q: mx.array,
    k_summary: mx.array,
    v_summary: mx.array,
    *,
    block_size: int,
    scale: float,
    routed_indices: mx.array | None = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Compressed-block attention stream returning `(o, l, m)` for online merge.

    Two operational modes:
      - `routed_indices is None` (HCA-style): every query at position `t` in
        block `b` attends to all compressed-block summaries `k_summary[0..b-1]`
        (strict causal). Query tokens in block 0 receive no contribution and
        their `(o, l, m)` slots are zeroed / set to `m = -inf` so they merge
        as no-ops with another stream.
      - `routed_indices is not None` (Butterfly-routed CSA-style): per-query-block
        gather of routed neighbor compressed blocks. Shape
        `routed_indices: [H, num_blocks, K_neighbors]`, dtype int32, with `-1`
        indicating an invalid slot. Per query-block we attend only to the
        valid routed neighbors.

    Inputs:
      q:          [B, H, T, dh]
      k_summary:  [B, H, num_blocks, dh]
      v_summary:  [B, H, num_blocks, dh]

    Outputs:
      o: [B, H, T, dh] in q's dtype.
      l: [B, H, T, 1] in float32. (0 on invalid query rows.)
      m: [B, H, T, 1] in float32. (-1e30 on invalid query rows.)
    """
    B, H, T, dh = q.shape
    _, _, num_blocks, _ = k_summary.shape
    out_dtype = q.dtype
    NEG_INF_F32 = mx.array(-1e30, dtype=mx.float32)
    EPS_F32 = mx.array(1e-9, dtype=mx.float32)

    if routed_indices is None:
        # HCA-style: dense over all causally-prior compressed blocks.
        scores = mx.matmul(
            q.astype(mx.float32),
            k_summary.transpose(0, 1, 3, 2).astype(mx.float32),
        ) * scale  # [B, H, T, num_blocks]

        q_pos = mx.arange(T, dtype=mx.int32)
        q_block = q_pos // int(block_size)  # [T]
        k_block = mx.arange(num_blocks, dtype=mx.int32)  # [num_blocks]
        valid = k_block.reshape(1, num_blocks) < q_block.reshape(T, 1)  # [T, num_blocks]
        valid_mask = valid.reshape(1, 1, T, num_blocks)

        scores = mx.where(valid_mask, scores, NEG_INF_F32)
        m = mx.max(scores, axis=-1, keepdims=True)
        # Zero out exp where no valid keys: detect rows that are entirely invalid.
        any_valid = mx.any(valid, axis=-1, keepdims=True).reshape(1, 1, T, 1)
        e = mx.exp(scores - m)
        e = mx.where(valid_mask, e, mx.zeros_like(e))
        l = mx.sum(e, axis=-1, keepdims=True)
        o = mx.matmul(e, v_summary.astype(mx.float32)) / mx.maximum(l, EPS_F32)
        # Force degenerate (m=-inf, l=0, o=0) on rows with no valid keys.
        zero_o = mx.zeros_like(o)
        zero_l = mx.zeros_like(l)
        neg_m = mx.full(m.shape, -1e30, dtype=mx.float32)
        o = mx.where(any_valid, o, zero_o)
        l = mx.where(any_valid, l, zero_l)
        m = mx.where(any_valid, m, neg_m)
        mx.eval(o, l, m)
        return o.astype(out_dtype), l, m

    # Routed mode: per-query-block gather + SDPA, looped with mx.eval flush.
    # routed_indices: [H, num_blocks, K_neighbors], int32, -1 = invalid.
    K_neighbors = int(routed_indices.shape[-1])
    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    m_chunks: list[mx.array] = []

    # Pre-pad k_summary, v_summary with a sentinel slot at index num_blocks so
    # we can replace -1 with num_blocks and gather safely. The sentinel's mask
    # bit is False so it never contributes.
    pad_zero = mx.zeros((B, H, 1, dh), dtype=k_summary.dtype)
    k_padded = mx.concatenate([k_summary, pad_zero], axis=2)
    v_padded = mx.concatenate([v_summary, pad_zero], axis=2)

    num_q_blocks = (T + block_size - 1) // block_size
    for q_block in range(num_q_blocks):
        q_start = q_block * block_size
        q_end = min(T, q_start + block_size)
        q_c = q[:, :, q_start:q_end, :]  # [B, H, c, dh]
        c = q_end - q_start

        # Per-head routed indices for this query block: [H, K_neighbors]
        idx_h = routed_indices[:, q_block, :]  # [H, K_neighbors], int32
        idx_clamped = mx.where(idx_h >= 0, idx_h, mx.array(num_blocks, dtype=mx.int32))
        valid_neighbor = idx_h >= 0  # [H, K_neighbors]
        # Causal: drop neighbors at index >= q_block.
        causal_neighbor = idx_h < q_block
        valid_neighbor = valid_neighbor & causal_neighbor

        # Gather K, V for each head: result [B, H, K_neighbors, dh].
        # Per-head gather because indices vary per head.
        head_k_chunks = []
        head_v_chunks = []
        for h in range(H):
            head_k_chunks.append(mx.take(k_padded[:, h : h + 1, :, :], idx_clamped[h], axis=2))
            head_v_chunks.append(mx.take(v_padded[:, h : h + 1, :, :], idx_clamped[h], axis=2))
        k_routed = mx.concatenate(head_k_chunks, axis=1)  # [B, H, K_neighbors, dh]
        v_routed = mx.concatenate(head_v_chunks, axis=1)
        valid_mask = valid_neighbor.reshape(1, H, 1, K_neighbors)

        if not bool(mx.any(valid_neighbor)):
            zero_o_c = mx.zeros((B, H, c, dh), dtype=mx.float32)
            zero_l_c = mx.zeros((B, H, c, 1), dtype=mx.float32)
            neg_m_c = mx.full((B, H, c, 1), -1e30, dtype=mx.float32)
            mx.eval(zero_o_c, zero_l_c, neg_m_c)
            o_chunks.append(zero_o_c)
            l_chunks.append(zero_l_c)
            m_chunks.append(neg_m_c)
            continue

        scores = mx.matmul(
            q_c.astype(mx.float32),
            k_routed.transpose(0, 1, 3, 2).astype(mx.float32),
        ) * scale  # [B, H, c, K_neighbors]
        scores = mx.where(valid_mask, scores, NEG_INF_F32)
        m_c = mx.max(scores, axis=-1, keepdims=True)
        e = mx.exp(scores - m_c)
        e = mx.where(valid_mask, e, mx.zeros_like(e))
        l_c = mx.sum(e, axis=-1, keepdims=True)
        o_c = mx.matmul(e, v_routed.astype(mx.float32)) / mx.maximum(l_c, EPS_F32)
        # Zero out rows that ended up with no valid neighbors at all (per-head).
        any_valid_h = mx.any(valid_mask, axis=-1, keepdims=True)
        zero_o_c = mx.zeros_like(o_c)
        zero_l_c = mx.zeros_like(l_c)
        neg_m_c = mx.full(m_c.shape, -1e30, dtype=mx.float32)
        o_c = mx.where(any_valid_h, o_c, zero_o_c)
        l_c = mx.where(any_valid_h, l_c, zero_l_c)
        m_c = mx.where(any_valid_h, m_c, neg_m_c)
        mx.eval(o_c, l_c, m_c)
        o_chunks.append(o_c)
        l_chunks.append(l_c)
        m_chunks.append(m_c)

    o = mx.concatenate(o_chunks, axis=2) if len(o_chunks) > 1 else o_chunks[0]
    l = mx.concatenate(l_chunks, axis=2) if len(l_chunks) > 1 else l_chunks[0]
    m = mx.concatenate(m_chunks, axis=2) if len(m_chunks) > 1 else m_chunks[0]
    return o.astype(out_dtype), l, m


def _online_softmax_merge(
    a: Tuple[mx.array, mx.array, mx.array],
    b: Tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Merge two attention-stream outputs by online softmax (m, l, o) accumulators.

    Each input is `(o, l, m)` where:
        o: per-stream output already divided by `l`, shape `[B, H, Tq, dh]`.
        l: `sum_j exp(score_j - m)`, shape `[B, H, Tq, 1]`.
        m: per-row score max, shape `[B, H, Tq, 1]`.

    Returns the output of a softmax computed over the union of the two streams'
    keys, without ever materializing the union K/V tensor. Mathematically:

        m_total = max(m_a, m_b)
        l_total = exp(m_a - m_total) * l_a + exp(m_b - m_total) * l_b
        o_total = (exp(m_a - m_total) * l_a * o_a
                 + exp(m_b - m_total) * l_b * o_b) / l_total

    A stream that is empty for a given query row should pass `m = -inf` and
    `l = 0`; the merge then degenerates to the other stream's contribution.
    """
    o_a, l_a, m_a = a
    o_b, l_b, m_b = b
    m_total = mx.maximum(m_a, m_b)
    coeff_a = mx.exp(m_a - m_total)
    coeff_b = mx.exp(m_b - m_total)
    l_total = coeff_a * l_a + coeff_b * l_b
    out = (coeff_a * l_a * o_a + coeff_b * l_b * o_b) / mx.maximum(l_total, EPS)
    return out


def stable_masked_softmax(
    scores_f32: mx.array,
    mask: mx.array,
    axis: int = -1,
    *,
    preserve_dtype: bool = False,
) -> mx.array:
    """Numerically stable masked softmax that returns zeros on all-masked rows."""
    if preserve_dtype:
        dtype = scores_f32.dtype
        # float16 cannot represent -1e30; use a finite floor to avoid -inf rows.
        neg_val = -1e4 if dtype == mx.float16 else -1e30
        neg_inf = mx.array(neg_val, dtype=dtype)
        eps = mx.array(1e-6 if dtype == mx.float16 else 1e-9, dtype=dtype)
    else:
        neg_inf = NEG_INF
        eps = EPS

    masked = mx.where(mask, scores_f32, neg_inf)
    row_max = mx.max(masked, axis=axis, keepdims=True)
    expv = mx.exp(masked - row_max)
    expv = mx.where(mask, expv, mx.zeros_like(expv))
    denom = mx.sum(expv, axis=axis, keepdims=True)
    return expv / mx.maximum(denom, eps)


def dense_causal_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    return_weights: bool = False,
) -> Tuple[mx.array, mx.array | None]:
    """Dense causal attention for q/k/v of shape [B,H,T,dh]."""
    B, H, T, dh = q.shape

    # Use MLX fused SDPA for production baseline comparisons.
    # Keep explicit weights path for debug/introspection callers.
    if (
        not return_weights
        and hasattr(mx, "fast")
        and hasattr(mx.fast, "scaled_dot_product_attention")
    ):
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(dh),
            mask="causal",
        ).astype(v.dtype)
        return y, None

    scores = mx.matmul(q.astype(mx.float32), k.transpose(0, 1, 3, 2).astype(mx.float32))
    scores = scores / math.sqrt(dh)

    i = mx.arange(T, dtype=mx.int32).reshape(T, 1)
    j = mx.arange(T, dtype=mx.int32).reshape(1, T)
    causal2d = j <= i
    mask = mx.broadcast_to(causal2d[None, None, :, :], (B, H, T, T))

    w = stable_masked_softmax(scores, mask, axis=-1)
    y = mx.matmul(w, v.astype(mx.float32))
    y = y.astype(v.dtype)

    if return_weights:
        return y, w
    return y, None


def build_union_multigraph_index(
    all_perms: mx.array,
    all_inv_perms: mx.array,
    *,
    window: int,
    circular: bool = False,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Build union neighbor index from multiple cycle permutations.

    For each query position i (original space), computes the set of keys
    reachable through any of the d cycles' windows, with multiplicity.

    Args:
        all_perms: [H, d, T] cycle permutations
        all_inv_perms: [H, d, T] inverse permutations
        window: half-window size
        circular: use modular wrap-around

    Returns:
        union_neigh_idx: [H, T, D_union] neighbor indices (-1 = pad)
        multiplicity: [H, T, D_union] edge multiplicity per neighbor
        valid_mask: [H, T, D_union] boolean mask
    """
    H, d, T = all_perms.shape

    # Work in numpy for the graph construction (runs on CPU)
    perms_np = np.array(all_perms, dtype=np.int64)
    inv_perms_np = np.array(all_inv_perms, dtype=np.int64)

    # For each head, build union neighbor set
    all_rows: list[list[list[int]]] = []  # [H][T][neighbors]
    all_mults: list[list[list[int]]] = []  # [H][T][multiplicities]
    max_deg = 0

    for h in range(H):
        head_rows: list[list[int]] = []
        head_mults: list[list[int]] = []
        for i in range(T):
            neigh_count: dict[int, int] = {}
            for c in range(d):
                rank_i = int(inv_perms_np[h, c, i])
                for off in range(-window, window + 1):
                    rank_j = rank_i + off
                    if circular:
                        rank_j = rank_j % T
                    elif rank_j < 0 or rank_j >= T:
                        continue
                    j = int(perms_np[h, c, rank_j])
                    if j > i:
                        continue  # causal: only attend to j <= i
                    neigh_count[j] = neigh_count.get(j, 0) + 1
            neighbors = sorted(neigh_count.keys())
            mults = [neigh_count[n] for n in neighbors]
            head_rows.append(neighbors)
            head_mults.append(mults)
            max_deg = max(max_deg, len(neighbors))
        all_rows.append(head_rows)
        all_mults.append(head_mults)

    if max_deg == 0:
        max_deg = 1

    # Pad to uniform degree
    neigh_idx = np.full((H, T, max_deg), -1, dtype=np.int32)
    mult = np.zeros((H, T, max_deg), dtype=np.int32)
    valid = np.zeros((H, T, max_deg), dtype=np.bool_)

    for h in range(H):
        for i in range(T):
            row = all_rows[h][i]
            m = all_mults[h][i]
            n = len(row)
            if n > 0:
                neigh_idx[h, i, :n] = row
                mult[h, i, :n] = m
                valid[h, i, :n] = True

    return (
        mx.array(neigh_idx),
        mx.array(mult),
        mx.array(valid),
    )


def _union_multigraph_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    union_neigh_idx: mx.array,
    multiplicity: mx.array,
    valid_mask: mx.array,
    multiplicity_bias_scale: float = 1.0,
) -> mx.array:
    """Single-pass attention over union multigraph with multiplicity bias.

    Args:
        q: [B, H, T, dh]
        k, v: [B, Hkv, T, dh]
        union_neigh_idx: [H, T, D] neighbor indices
        multiplicity: [H, T, D] edge multiplicities
        valid_mask: [H, T, D] boolean
        multiplicity_bias_scale: scale for log(multiplicity) bias

    Returns:
        y: [B, H, T, dh]
    """
    B, Hq, T, dh = q.shape
    Hkv = k.shape[1]
    gqa_ratio = Hq // Hkv

    ys: list[mx.array] = []
    for h in range(Hq):
        kv_h = h // gqa_ratio
        q_h = q[:, h]  # [B, T, dh]
        k_h = k[:, kv_h]  # [B, T, dh]
        v_h = v[:, kv_h]  # [B, T, dh]

        idx_h = mx.clip(union_neigh_idx[h], 0, T - 1)  # [T, D], safe gather
        mask_h = valid_mask[h]  # [T, D]

        k_g = k_h[:, idx_h]  # [B, T, D, dh]
        v_g = v_h[:, idx_h]  # [B, T, D, dh]

        scores = mx.sum(
            q_h[:, :, None, :].astype(mx.float32) * k_g.astype(mx.float32),
            axis=-1,
        ) / math.sqrt(dh)  # [B, T, D]

        # Add multiplicity bias: log(m) scaled
        mult_h = multiplicity[h].astype(mx.float32)  # [T, D]
        mult_bias = mx.log(mx.maximum(mult_h, mx.array(1.0))) * multiplicity_bias_scale
        scores = scores + mult_bias[None, :, :]

        w = stable_masked_softmax(scores, mask_h[None, :, :], axis=-1)
        y_h = mx.sum(w[:, :, :, None] * v_g.astype(mx.float32), axis=2)
        ys.append(y_h.astype(v.dtype))
        # Sync per head to avoid accumulating huge lazy gather graphs in memory
        mx.eval(ys[-1])

    return mx.stack(ys, axis=1)


def _sparse_gather_attention_vectorized(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    s_idx: mx.array,
    mask: mx.array,
    *,
    chunk_size: int = 256,
) -> mx.array:
    """Vectorized sparse-neighbor attention — all heads in one pass per chunk.

    Uses batched matmul for dot products instead of per-head Python loop.
    Correct butterfly topology (CYCLE + WINDOW + LANDMARK edges via neigh_idx).
    """
    B, H, T, dh = q.shape
    Hkv = k.shape[1]
    D = int(s_idx.shape[-1])
    scale = 1.0 / math.sqrt(dh)

    # GQA: expand K/V to match query heads
    if Hkv < H:
        reps = H // Hkv
        k = mx.repeat(k, repeats=reps, axis=1)
        v = mx.repeat(v, repeats=reps, axis=1)

    BH = B * H
    k_flat = k.reshape(BH, T, dh)
    v_flat = v.reshape(BH, T, dh)

    y_chunks: list[mx.array] = []
    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)
        Qblk = e - s

        q_blk = q[:, :, s:e, :]  # [B, H, Qblk, dh]
        idx_blk = s_idx[:, s:e, :]  # [H, Qblk, D]
        mask_blk = mask[:, s:e, :]  # [H, Qblk, D]

        # Gather K/V for all heads simultaneously: [BH, Qblk*D, dh]
        idx_flat = mx.broadcast_to(idx_blk[None], (B, H, Qblk, D)).reshape(BH, Qblk * D)
        idx_exp = mx.broadcast_to(idx_flat[:, :, None], (BH, Qblk * D, dh))

        k_g = mx.take_along_axis(k_flat, idx_exp, axis=1).reshape(B, H, Qblk, D, dh)
        v_g = mx.take_along_axis(v_flat, idx_exp, axis=1).reshape(B, H, Qblk, D, dh)

        # Batched matmul: [B,H,Qblk,1,dh] @ [B,H,Qblk,dh,D] → [B,H,Qblk,1,D]
        scores = mx.matmul(
            q_blk[:, :, :, None, :].astype(mx.float32),
            k_g.astype(mx.float32).transpose(0, 1, 2, 4, 3),
        ).squeeze(3) * scale  # [B, H, Qblk, D]

        # Masked softmax over D
        w = stable_masked_softmax(scores, mask_blk[None, :, :, :], axis=-1)

        # Weighted V: [B,H,Qblk,D,1] * [B,H,Qblk,D,dh] → sum over D
        y_blk = mx.sum(w[:, :, :, :, None] * v_g.astype(mx.float32), axis=3).astype(v.dtype)
        y_chunks.append(y_blk)

    return mx.concatenate(y_chunks, axis=2) if len(y_chunks) > 1 else y_chunks[0]


def _sparse_gather_attention_vectorized_active(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    s_idx: mx.array,
    mask: mx.array,
    *,
    chunk_size: int = 256,
) -> mx.array:
    """Vectorized sparse attention for active rows where Tq <= Tk."""
    B, H, Tq, dh = q.shape
    Hkv = k.shape[1]
    Tk = int(k.shape[2])
    D = int(s_idx.shape[-1])
    scale = 1.0 / math.sqrt(dh)

    if Hkv < H:
        reps = H // Hkv
        k = mx.repeat(k, repeats=reps, axis=1)
        v = mx.repeat(v, repeats=reps, axis=1)

    BH = B * H
    k_flat = k.reshape(BH, Tk, dh)
    v_flat = v.reshape(BH, Tk, dh)

    y_chunks: list[mx.array] = []
    for s in range(0, Tq, chunk_size):
        e = min(Tq, s + chunk_size)
        q_blk_len = e - s

        q_blk = q[:, :, s:e, :]
        idx_blk = s_idx[:, s:e, :]
        mask_blk = mask[:, s:e, :]

        idx_flat = mx.broadcast_to(idx_blk[None], (B, H, q_blk_len, D)).reshape(BH, q_blk_len * D)
        idx_exp = mx.broadcast_to(idx_flat[:, :, None], (BH, q_blk_len * D, dh))

        k_g = mx.take_along_axis(k_flat, idx_exp, axis=1).reshape(B, H, q_blk_len, D, dh)
        v_g = mx.take_along_axis(v_flat, idx_exp, axis=1).reshape(B, H, q_blk_len, D, dh)

        scores = mx.matmul(
            q_blk[:, :, :, None, :].astype(mx.float32),
            k_g.astype(mx.float32).transpose(0, 1, 2, 4, 3),
        ).squeeze(3) * scale

        w = stable_masked_softmax(scores, mask_blk[None, :, :, :], axis=-1)
        y_blk = mx.sum(w[:, :, :, :, None] * v_g.astype(mx.float32), axis=3).astype(v.dtype)
        y_chunks.append(y_blk)

    return mx.concatenate(y_chunks, axis=2) if len(y_chunks) > 1 else y_chunks[0]


def _sparse_gather_attention_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    neigh_idx_safe: mx.array,
    causal_mask: mx.array,
) -> mx.array:
    """K7 Metal kernel path for sparse-neighbor attention.

    All heads fused in a single Metal dispatch — no per-head Python loop.
    """
    from bna.mlx.kernels.metal import sparse_neighbor_attention_kernel

    kernel = sparse_neighbor_attention_kernel()
    B, Hq, Tq, dh = q.shape
    Hkv = k.shape[1]

    # Build GQA head map: query head h → KV head h // (Hq // Hkv)
    gqa_rep = Hq // Hkv
    hkv_map = mx.array([h // gqa_rep for h in range(Hq)], dtype=mx.int32)

    # Ensure mask is uint8 for Metal
    mask_u8 = causal_mask.astype(mx.uint8) if causal_mask.dtype != mx.uint8 else causal_mask

    _grid = (dh, B * Hq * Tq, 1)
    _tg = (min(dh, 256), 1, 1)
    y = kernel(
        q, k, v,
        neigh_idx_safe.astype(mx.int32),
        mask_u8,
        hkv_map,
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=_grid,
        threadgroup=_tg,
    )[0]
    return y


def sparse_gather_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    graph: MLXGraphABI,
    *,
    return_weights: bool = False,
    precomputed_safe_idx: Optional[mx.array] = None,
    precomputed_causal_mask: Optional[mx.array] = None,
    edge_type_bias: Optional[mx.array] = None,
    edge_type_bias_offset: Optional[mx.array] = None,
    window_drop_mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array | None]:
    """General sparse-row gather path using Wayfinder graph ABI.

    Optional precomputed_safe_idx / precomputed_causal_mask skip recomputation.
    edge_type_bias: [4] learnable bias for [CYCLE, WINDOW, LANDMARK, REWIRE].
    window_drop_mask: [H, T, D] bool — False means drop that edge.
    """
    B, H, T, dh = q.shape
    D = int(graph.neigh_idx.shape[-1])

    if D == 0:
        out = mx.zeros_like(q)
        if return_weights:
            return out, mx.zeros((B, H, T, 0), dtype=mx.float32)
        return out, None

    if precomputed_safe_idx is not None:
        s_idx = precomputed_safe_idx
    else:
        s_idx = safe_neighbor_idx(graph.neigh_idx, T)
    if precomputed_causal_mask is not None:
        mask = precomputed_causal_mask
    else:
        mask = causal_neighbor_mask(graph.neigh_idx, T)

    # Apply window-drop: zero out dropped WINDOW edges in mask
    if window_drop_mask is not None:
        mask = mask & window_drop_mask

    # --- Fast paths: no per-head loop, no edge bias ---
    if (
        not return_weights
        and edge_type_bias is None
        and edge_type_bias_offset is None
    ):
        Tq = q.shape[2]
        from bna.mlx.kernels.metal import has_sparse_neighbor_kernel

        if Tq <= 32 and has_sparse_neighbor_kernel():
            # Decode (few queries): K7 Metal kernel — fused, zero overhead
            try:
                y = _sparse_gather_attention_metal(q, k, v, s_idx, mask)
            except Exception:
                y = _sparse_gather_attention_vectorized(q, k, v, s_idx, mask)
        else:
            # Prefill (many queries): vectorized gather + batched matmul
            y = _sparse_gather_attention_vectorized(q, k, v, s_idx, mask)
        return y, None

    # Precompute per-edge bias tensor [H, T, D] if edge_type_bias provided
    bias_htd: Optional[mx.array] = None
    if edge_type_bias is not None or edge_type_bias_offset is not None:
        # edge_type_bias is [4] for CYCLE=1, WINDOW=2, LANDMARK=3, REWIRE=4
        # Map edge codes -> bias. PAD (0) gets 0 bias.
        bias_vec = mx.zeros((4,), dtype=mx.float32)
        if edge_type_bias is not None:
            bias_vec = bias_vec + edge_type_bias.astype(mx.float32)
        if edge_type_bias_offset is not None:
            bias_vec = bias_vec + edge_type_bias_offset.astype(mx.float32)
        full_bias = mx.concatenate([mx.zeros((1,), dtype=mx.float32), bias_vec])  # [5]
        et = graph.edge_type.astype(mx.int32)  # [H, T, D]
        bias_htd = full_bias[et]  # [H, T, D]

    ys: list[mx.array] = []
    ws: list[mx.array] = []

    for h in range(H):
        q_h = q[:, h]  # [B,T,dh]
        k_h = k[:, h]
        v_h = v[:, h]
        idx_h = s_idx[h]  # [T,D]
        mask_h = mask[h]  # [T,D]

        k_g = k_h[:, idx_h]  # [B,T,D,dh]
        v_g = v_h[:, idx_h]

        scores = mx.sum(
            q_h[:, :, None, :].astype(mx.float32) * k_g.astype(mx.float32),
            axis=-1,
        ) / math.sqrt(dh)

        if bias_htd is not None:
            scores = scores + bias_htd[h][None, :, :]  # broadcast [1,T,D]

        w_h = stable_masked_softmax(scores, mask_h[None, :, :], axis=-1)  # [B,T,D]
        y_h = mx.sum(w_h[:, :, :, None] * v_g.astype(mx.float32), axis=2)
        ys.append(y_h.astype(v.dtype))
        if return_weights:
            ws.append(w_h)

    y = mx.stack(ys, axis=1)  # [B,H,T,dh]
    if return_weights:
        return y, mx.stack(ws, axis=1)
    return y, None


def sparse_gather_attention_active(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    graph: MLXGraphABI,
    *,
    query_positions: mx.array,
    return_weights: bool = False,
    precomputed_safe_idx: Optional[mx.array] = None,
    precomputed_causal_mask: Optional[mx.array] = None,
    edge_type_bias: Optional[mx.array] = None,
    edge_type_bias_offset: Optional[mx.array] = None,
    window_drop_mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Sparse gather for active query rows where Tq <= Tk.

    q: [B, H, Tq, dh] contains only active query rows.
    k/v: [B, H, Tk, dh] contains full currently available KV prefix.
    query_positions: [Tq] original token positions for each query row in q.
    """
    B, H, Tq, dh = q.shape
    Bk, Hk, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B:
        raise ValueError("q, k, v batch size mismatch")
    if Hk != H or Hv != H:
        raise ValueError("q, k, v must share head count for sparse active gather")
    if dhk != dh or dhv != dh:
        raise ValueError("q, k, v head_dim mismatch")
    if Tv != Tk:
        raise ValueError("k and v sequence length mismatch")
    if tuple(query_positions.shape) != (Tq,):
        raise ValueError(f"query_positions must be shape ({Tq},), got {query_positions.shape}")

    Tg = int(graph.neigh_idx.shape[1])
    D = int(graph.neigh_idx.shape[-1])
    if Tg < Tk:
        raise ValueError(f"graph sequence length Tg={Tg} must be >= Tk={Tk}")

    if D == 0:
        out = mx.zeros((B, H, Tq, dh), dtype=q.dtype)
        if return_weights:
            return out, mx.zeros((B, H, Tq, 0), dtype=mx.float32)
        return out, None

    q_pos = query_positions.astype(mx.int32)
    if Tq > 0:
        q_min = int(mx.min(q_pos).item())
        q_max = int(mx.max(q_pos).item())
        if q_min < 0 or q_max >= Tg:
            raise ValueError(
                f"query_positions must be in [0, {Tg}), got min={q_min}, max={q_max}"
            )

    if precomputed_safe_idx is not None:
        s_idx_full = precomputed_safe_idx
    else:
        s_idx_full = safe_neighbor_idx(graph.neigh_idx, Tg)
    if precomputed_causal_mask is not None:
        mask_full = precomputed_causal_mask
    else:
        mask_full = causal_neighbor_mask(graph.neigh_idx, Tg)

    s_idx = mx.take(s_idx_full, q_pos, axis=1)  # [H, Tq, D]
    mask = mx.take(mask_full, q_pos, axis=1)  # [H, Tq, D]

    if window_drop_mask is not None:
        mask = mask & mx.take(window_drop_mask, q_pos, axis=1)

    available = s_idx < Tk
    mask = mask & available
    s_idx = mx.clip(s_idx, 0, max(Tk - 1, 0))

    bias_htd: Optional[mx.array] = None
    if edge_type_bias is not None or edge_type_bias_offset is not None:
        bias_vec = mx.zeros((4,), dtype=mx.float32)
        if edge_type_bias is not None:
            bias_vec = bias_vec + edge_type_bias.astype(mx.float32)
        if edge_type_bias_offset is not None:
            bias_vec = bias_vec + edge_type_bias_offset.astype(mx.float32)
        full_bias = mx.concatenate([mx.zeros((1,), dtype=mx.float32), bias_vec])  # [5]
        et_rows = mx.take(graph.edge_type.astype(mx.int32), q_pos, axis=1)  # [H, Tq, D]
        bias_htd = full_bias[et_rows]

    ys: list[mx.array] = []
    ws: list[mx.array] = []

    for h in range(H):
        q_h = q[:, h]  # [B, Tq, dh]
        k_h = k[:, h]  # [B, Tk, dh]
        v_h = v[:, h]  # [B, Tk, dh]
        idx_h = s_idx[h]  # [Tq, D]
        mask_h = mask[h]  # [Tq, D]

        k_g = k_h[:, idx_h]  # [B, Tq, D, dh]
        v_g = v_h[:, idx_h]

        scores = mx.sum(
            q_h[:, :, None, :].astype(mx.float32) * k_g.astype(mx.float32),
            axis=-1,
        ) / math.sqrt(dh)

        if bias_htd is not None:
            scores = scores + bias_htd[h][None, :, :]

        w_h = stable_masked_softmax(scores, mask_h[None, :, :], axis=-1)
        y_h = mx.sum(w_h[:, :, :, None] * v_g.astype(mx.float32), axis=2)
        ys.append(y_h.astype(v.dtype))
        if return_weights:
            ws.append(w_h)

    y = mx.stack(ys, axis=1)  # [B, H, Tq, dh]
    if return_weights:
        return y, mx.stack(ws, axis=1)
    return y, None


def _build_block_sparse_active_indices(
    layout: BlockSparseButterflyLayout,
    *,
    kv_len: int,
    query_positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """Expand block neighborhoods only for the active query rows."""
    if int(kv_len) <= 0:
        raise ValueError("kv_len must be positive")

    q_pos_np = np.asarray(query_positions, dtype=np.int32)
    if q_pos_np.ndim != 1:
        raise ValueError(f"query_positions must be 1D, got {q_pos_np.shape}")
    if q_pos_np.size == 0:
        shape = (int(layout.block_neighbors.shape[0]), 0, int(layout.block_token_idx.shape[-1]))
        return (
            mx.zeros(shape, dtype=mx.int32),
            mx.zeros(shape, dtype=mx.bool_),
        )

    block_size = int(layout.block_size)
    num_blocks = int(layout.num_blocks)
    block_idx = np.clip(q_pos_np // max(1, block_size), 0, max(0, num_blocks - 1))

    block_rows = np.asarray(mx.take(layout.block_neighbors[0], mx.array(block_idx, dtype=mx.int32), axis=0))
    valid_blocks = block_rows >= 0
    safe_blocks = np.clip(block_rows, 0, max(0, num_blocks - 1))

    offsets = np.arange(block_size, dtype=np.int32)
    token_idx_raw = (
        safe_blocks[:, :, None] * int(block_size) + offsets[None, None, :]
    ).reshape(int(q_pos_np.shape[0]), -1)
    valid_tokens = np.repeat(valid_blocks[:, :, None], int(block_size), axis=2).reshape(
        int(q_pos_np.shape[0]),
        -1,
    )

    causal_mask = valid_tokens & (token_idx_raw <= q_pos_np[:, None]) & (token_idx_raw < int(kv_len))
    safe_idx = np.clip(token_idx_raw, 0, int(max(0, kv_len - 1)))

    heads = int(layout.block_neighbors.shape[0])
    safe_idx_h = np.broadcast_to(safe_idx[None, :, :], (heads, *safe_idx.shape)).copy()
    causal_mask_h = np.broadcast_to(causal_mask[None, :, :], (heads, *causal_mask.shape)).copy()
    return mx.array(safe_idx_h, dtype=mx.int32), mx.array(causal_mask_h, dtype=mx.bool_)


def _block_sparse_attention_manual(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    mask: mx.array,
    scale: float,
) -> mx.array:
    scores = mx.matmul(
        q.astype(mx.float32),
        k.transpose(0, 1, 3, 2).astype(mx.float32),
    ) * float(scale)
    weights = stable_masked_softmax(scores, mask, axis=-1, preserve_dtype=False)
    return mx.matmul(weights, v.astype(mx.float32)).astype(v.dtype)


def _compressed_butterfly_block_indices(
    layout: BlockSparseButterflyLayout,
    *,
    kv_len: int,
    local_window_tokens: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build token-local and summary-block indices for compressed Butterfly."""
    if int(kv_len) <= 0:
        raise ValueError("kv_len must be positive")
    if int(local_window_tokens) <= 0:
        raise ValueError("local_window_tokens must be positive")

    num_blocks = int(layout.num_blocks)
    block_size = int(layout.block_size)
    raw_slots = int(local_window_tokens) + int(block_size) - 1
    summary_slots = int(layout.block_neighbors.shape[-1])

    raw_idx = np.zeros((num_blocks, raw_slots), dtype=np.int32)
    raw_mask = np.zeros((num_blocks, block_size, raw_slots), dtype=np.bool_)
    summary_idx = np.zeros((num_blocks, summary_slots), dtype=np.int32)
    summary_mask = np.zeros((num_blocks, block_size, summary_slots), dtype=np.bool_)

    neighbors = np.asarray(layout.block_neighbors[0], dtype=np.int32)
    raw_offsets = np.arange(raw_slots, dtype=np.int32)

    for block_idx in range(num_blocks):
        q_start = int(block_idx) * block_size
        q_positions = q_start + np.arange(block_size, dtype=np.int32)
        q_valid = q_positions < int(kv_len)
        local_start = max(0, q_start - int(local_window_tokens) + 1)

        raw_positions = local_start + raw_offsets
        raw_valid = raw_positions < int(kv_len)
        raw_idx[block_idx] = np.clip(raw_positions, 0, max(0, int(kv_len) - 1))
        raw_mask[block_idx] = (
            q_valid[:, None]
            & raw_valid[None, :]
            & (raw_positions[None, :] <= q_positions[:, None])
            & (raw_positions[None, :] >= (q_positions[:, None] - int(local_window_tokens) + 1))
        )

        cursor = 0
        for neighbor in neighbors[block_idx].tolist():
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor >= block_idx:
                continue
            # Exact local tokens already cover this region; route only older summaries.
            neighbor_end = (neighbor + 1) * block_size
            if neighbor_end > local_start:
                continue
            summary_idx[block_idx, cursor] = neighbor
            summary_mask[block_idx, :, cursor] = q_valid
            cursor += 1
            if cursor >= summary_slots:
                break

    return (
        mx.array(raw_idx, dtype=mx.int32),
        mx.array(raw_mask, dtype=mx.bool_),
        mx.array(summary_idx, dtype=mx.int32),
        mx.array(summary_mask, dtype=mx.bool_),
    )


def _compressed_butterfly_active_indices(
    layout: BlockSparseButterflyLayout,
    *,
    kv_len: int,
    query_positions: mx.array,
    local_window_tokens: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build token-local and summary-block indices for active compressed rows."""
    if int(kv_len) <= 0:
        raise ValueError("kv_len must be positive")
    if int(local_window_tokens) <= 0:
        raise ValueError("local_window_tokens must be positive")

    q_pos_np = np.asarray(query_positions, dtype=np.int32)
    if q_pos_np.ndim != 1:
        raise ValueError(f"query_positions must be 1D, got {q_pos_np.shape}")

    Tq = int(q_pos_np.size)
    raw_slots = int(local_window_tokens)
    summary_slots = int(layout.block_neighbors.shape[-1])
    block_size = int(layout.block_size)
    num_blocks = int(layout.num_blocks)
    neighbors = np.asarray(layout.block_neighbors[0], dtype=np.int32)

    raw_idx = np.zeros((Tq, raw_slots), dtype=np.int32)
    raw_mask = np.zeros((Tq, raw_slots), dtype=np.bool_)
    summary_idx = np.zeros((Tq, summary_slots), dtype=np.int32)
    summary_mask = np.zeros((Tq, summary_slots), dtype=np.bool_)

    offsets = np.arange(raw_slots, dtype=np.int32)
    for row_idx, q_pos_t in enumerate(q_pos_np.tolist()):
        q_pos = int(q_pos_t)
        block_idx = min(max(0, q_pos // max(1, block_size)), max(0, num_blocks - 1))
        local_start = max(0, q_pos - int(local_window_tokens) + 1)
        raw_positions = local_start + offsets
        raw_valid = (raw_positions <= q_pos) & (raw_positions < int(kv_len))
        raw_idx[row_idx] = np.clip(raw_positions, 0, max(0, int(kv_len) - 1))
        raw_mask[row_idx] = raw_valid

        cursor = 0
        for neighbor in neighbors[block_idx].tolist():
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor >= block_idx:
                continue
            neighbor_end = (neighbor + 1) * block_size
            if neighbor_end > local_start:
                continue
            summary_idx[row_idx, cursor] = neighbor
            summary_mask[row_idx, cursor] = True
            cursor += 1
            if cursor >= summary_slots:
                break

    return (
        mx.array(raw_idx, dtype=mx.int32),
        mx.array(raw_mask, dtype=mx.bool_),
        mx.array(summary_idx, dtype=mx.int32),
        mx.array(summary_mask, dtype=mx.bool_),
    )


def _compressed_butterfly_routed_indices(
    layout: BlockSparseButterflyLayout,
    *,
    kv_len: int,
    local_window_tokens: int,
) -> mx.array:
    """Build `[H, num_blocks, summary_slots]` routed-block indices for the
    compressed-stream attention.

    Each `(h, b, slot)` entry is a compressed-block index that query block `b`
    on head `h` should attend to, or `-1` for an invalid slot. Filters routing
    to only neighbors whose entire block ends before the local-window start so
    raw-window tokens are not double-counted between the SWA and compressed
    streams. Mirrors the eligibility logic in
    `_compressed_butterfly_block_indices` (lines ~1230-1259).
    """
    num_blocks = int(layout.num_blocks)
    block_size = int(layout.block_size)
    summary_slots = int(layout.block_neighbors.shape[-1])
    H = int(layout.block_neighbors.shape[0])

    routed = np.full((H, num_blocks, summary_slots), -1, dtype=np.int32)
    neighbors_all = np.asarray(layout.block_neighbors, dtype=np.int32)

    for b in range(num_blocks):
        q_start = b * block_size
        local_start = max(0, q_start - int(local_window_tokens) + 1)
        for h in range(H):
            cursor = 0
            for neighbor in neighbors_all[h, b].tolist():
                neighbor_i = int(neighbor)
                if neighbor_i < 0 or neighbor_i >= b:
                    continue
                neighbor_end = (neighbor_i + 1) * block_size
                if neighbor_end > local_start:
                    continue
                # Also require the neighbor block is entirely within the cache.
                if neighbor_end > int(kv_len):
                    continue
                routed[h, b, cursor] = neighbor_i
                cursor += 1
                if cursor >= summary_slots:
                    break
    return mx.array(routed, dtype=mx.int32)


def _block_mean_summaries_impl(x: mx.array, *, seq_len: int, block_size: int, num_blocks: int) -> mx.array:
    """Mean-pool token states into one summary vector per block."""
    B, H, T, dh = x.shape
    if int(T) != int(seq_len):
        raise ValueError(f"summary seq_len mismatch: x has T={T}, expected {seq_len}")
    pad_len = int(num_blocks) * int(block_size) - int(seq_len)
    if pad_len > 0:
        x_work = mx.pad(x, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
    else:
        x_work = x

    x_blocks = x_work.reshape(B, H, int(num_blocks), int(block_size), dh)
    token_pos = mx.arange(int(num_blocks) * int(block_size), dtype=mx.int32).reshape(
        int(num_blocks),
        int(block_size),
    )
    valid = token_pos < int(seq_len)
    valid_f = valid.astype(mx.float32)
    denom = mx.maximum(mx.sum(valid_f, axis=1, keepdims=True), mx.array(1.0, dtype=mx.float32))
    summed = mx.sum(x_blocks.astype(mx.float32) * valid_f[None, None, :, :, None], axis=3)
    return (summed / denom[None, None, :, :]).astype(x.dtype)


def _block_mean_summaries(x: mx.array, *, seq_len: int, block_size: int, num_blocks: int) -> mx.array:
    """Mean-pool token states into one summary vector per block (instrumented)."""
    return _maybe_profile_block_mean_summaries(x, seq_len=seq_len, block_size=block_size, num_blocks=num_blocks)


def compressed_butterfly_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    local_window_tokens: int,
    return_weights: bool = False,
    block_chunk_size: int = 0,
    scale: Optional[float] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Compressed Butterfly prefill via two streams + online softmax merge.

    Stream 1: sliding-window attention over recent raw tokens
    (`_swa_stream_attention`, window = `local_window_tokens`).
    Stream 2: Butterfly-routed compressed-block attention
    (`_compressed_stream_attention` with `routed_indices` derived from
    `layout.block_neighbors` and filtered against the SWA window so the
    same token never contributes through both streams).

    The two streams are merged by online-softmax accumulators, producing the
    same numerical result as a single softmax over the union of their keys but
    without ever materializing that union in framework-level memory. The
    `block_chunk_size` argument is accepted for backward-compat with callers
    but is no longer used (chunking now lives inside the per-stream helpers).
    """
    del block_chunk_size  # legacy knob; chunking is per-stream now.
    if return_weights:
        raise NotImplementedError("return_weights is not supported for compressed Butterfly MLX attention")

    B, H, T, dh = q.shape
    Bk, Hk, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Hk != H or Hv != H:
        raise ValueError("q, k, v must share batch/head dimensions for compressed Butterfly")
    if Tk != T or Tv != T:
        raise ValueError("Compressed Butterfly prefill requires k/v length to match q length")
    if dhk != dh or dhv != dh:
        raise ValueError("q, k, v head_dim mismatch")
    if int(layout.seq_len) != int(T):
        raise ValueError(f"layout.seq_len={layout.seq_len} must match q length {T}")

    num_blocks = int(layout.num_blocks)
    block_size = int(layout.block_size)
    q_scale = float(scale if scale is not None else (dh ** -0.5))
    n_win = int(local_window_tokens)

    if int(layout.block_neighbors.shape[0]) != H:
        raise ValueError(
            f"layout.block_neighbors head dim {layout.block_neighbors.shape[0]} must match q heads {H}"
        )

    routed_indices = _compressed_butterfly_routed_indices(
        layout, kv_len=int(T), local_window_tokens=n_win,
    )

    k_summary = _block_mean_summaries(k, seq_len=int(T), block_size=block_size, num_blocks=num_blocks)
    v_summary = _block_mean_summaries(v, seq_len=int(T), block_size=block_size, num_blocks=num_blocks)

    swa_o, swa_l, swa_m = _swa_stream_attention(q, k, v, n_win=n_win, scale=q_scale)
    cmp_o, cmp_l, cmp_m = _compressed_stream_attention(
        q, k_summary, v_summary,
        block_size=block_size, scale=q_scale,
        routed_indices=routed_indices,
    )

    out = _online_softmax_merge((swa_o, swa_l, swa_m), (cmp_o, cmp_l, cmp_m))
    return out.astype(v.dtype), None


def compressed_butterfly_attention_active(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    query_positions: mx.array,
    local_window_tokens: int,
    return_weights: bool = False,
    scale: Optional[float] = None,
    precomputed_k_summary: Optional[mx.array] = None,
    precomputed_v_summary: Optional[mx.array] = None,
    query_chunk_size: int = 0,
) -> Tuple[mx.array, mx.array | None]:
    """Compressed Butterfly attention for active decode rows.

    Args:
        precomputed_k_summary: Optional [B, H, num_blocks, dh] from a cached summary path.
        precomputed_v_summary: Optional [B, H, num_blocks, dh] from a cached summary path.
    """
    B, H, Tq, dh = q.shape
    Bk, Hk, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Hk != H or Hv != H:
        raise ValueError("q, k, v must share batch/head dimensions for compressed Butterfly active attention")
    if Tv != Tk:
        raise ValueError("k and v sequence length mismatch")
    if dhk != dh or dhv != dh:
        raise ValueError("q, k, v head_dim mismatch")

    q_chunk = int(query_chunk_size)
    if q_chunk > 0 and q_chunk < Tq:
        out_chunks: list[mx.array] = []
        weight_chunks: list[mx.array] = []
        for start in range(0, Tq, q_chunk):
            end = min(start + q_chunk, Tq)
            y_i, w_i = compressed_butterfly_attention_active(
                q[:, :, start:end, :],
                k,
                v,
                layout=layout,
                query_positions=query_positions[start:end],
                local_window_tokens=local_window_tokens,
                return_weights=return_weights,
                scale=scale,
                precomputed_k_summary=precomputed_k_summary,
                precomputed_v_summary=precomputed_v_summary,
                query_chunk_size=0,
            )
            if w_i is None:
                mx.eval(y_i)
            else:
                mx.eval(y_i, w_i)
                weight_chunks.append(w_i)
            out_chunks.append(y_i)
        out = mx.concatenate(out_chunks, axis=2)
        if return_weights:
            return out, mx.concatenate(weight_chunks, axis=2)
        return out, None

    raw_idx, raw_mask, summary_idx, summary_mask = _compressed_butterfly_active_indices(
        layout,
        kv_len=int(Tk),
        query_positions=query_positions,
        local_window_tokens=int(local_window_tokens),
    )

    num_blocks = int(layout.num_blocks)
    block_size = int(layout.block_size)
    if precomputed_k_summary is not None and precomputed_v_summary is not None:
        k_summary = precomputed_k_summary
        v_summary = precomputed_v_summary
        if int(k_summary.shape[2]) != num_blocks or int(v_summary.shape[2]) != num_blocks:
            raise ValueError(
                f"precomputed summary block count mismatch: "
                f"expected {num_blocks}, got k={k_summary.shape[2]}, v={v_summary.shape[2]}"
            )
    else:
        k_summary = _block_mean_summaries(k, seq_len=int(Tk), block_size=block_size, num_blocks=num_blocks)
        v_summary = _block_mean_summaries(v, seq_len=int(Tk), block_size=block_size, num_blocks=num_blocks)

    raw_k = mx.take(k, raw_idx, axis=2)
    raw_v = mx.take(v, raw_idx, axis=2)
    summary_k = mx.take(k_summary, summary_idx, axis=2)
    summary_v = mx.take(v_summary, summary_idx, axis=2)
    k_g = mx.concatenate([raw_k, summary_k], axis=3)
    v_g = mx.concatenate([raw_v, summary_v], axis=3)
    mask = mx.concatenate([raw_mask, summary_mask], axis=1)

    q_scale = float(scale if scale is not None else (dh ** -0.5))
    if _COMPRESS_PROFILE:
        t_attn0 = time.perf_counter()
    if (
        not _COMPRESS_FORCE_MANUAL
        and not return_weights
        and hasattr(mx, "fast")
        and hasattr(mx.fast, "scaled_dot_product_attention")
    ):
        q_r = q.transpose(0, 2, 1, 3).reshape(B * Tq, H, 1, dh)
        k_r = k_g.transpose(0, 2, 1, 3, 4).reshape(B * Tq, H, int(k_g.shape[3]), dh)
        v_r = v_g.transpose(0, 2, 1, 3, 4).reshape(B * Tq, H, int(v_g.shape[3]), dh)
        mask_r = mx.broadcast_to(
            mask[None, :, None, :],
            (B, Tq, H, int(mask.shape[1])),
        ).reshape(B * Tq, H, 1, int(mask.shape[1]))
        out = mx.fast.scaled_dot_product_attention(
            q_r,
            k_r,
            v_r,
            scale=q_scale,
            mask=mask_r,
        ).reshape(B, Tq, H, dh).transpose(0, 2, 1, 3).astype(v.dtype)
        if _COMPRESS_PROFILE:
            mx.eval(out)
            _compress_stats["attn_ms"] += (time.perf_counter() - t_attn0) * 1000.0
        return out, None
    scores = mx.sum(
        q[:, :, :, None, :].astype(mx.float32) * k_g.astype(mx.float32),
        axis=-1,
    ) * q_scale
    w = stable_masked_softmax(scores, mask[None, None, :, :], axis=-1)
    out = mx.sum(w[:, :, :, :, None] * v_g.astype(mx.float32), axis=3).astype(v.dtype)
    if _COMPRESS_PROFILE:
        mx.eval(out)
        _compress_stats["attn_ms"] += (time.perf_counter() - t_attn0) * 1000.0
    if return_weights:
        return out, w
    return out, None


def _compressed_butterfly_active_indices_from_tail(
    layout: BlockSparseButterflyLayout,
    *,
    kv_len: int,
    tail_start: int,
    num_summary_blocks: int,
    tail_len: int,
    query_positions: mx.array,
    local_window_tokens: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if int(kv_len) <= 0:
        raise ValueError("kv_len must be positive")
    q_pos_np = np.asarray(query_positions, dtype=np.int32)
    if q_pos_np.ndim != 1:
        raise ValueError(f"query_positions must be 1D, got {q_pos_np.shape}")

    Tq = int(q_pos_np.size)
    raw_slots = int(local_window_tokens)
    summary_slots = int(layout.block_neighbors.shape[-1])
    block_size = int(layout.block_size)
    num_blocks = int(layout.num_blocks)
    neighbors = np.asarray(layout.block_neighbors[0], dtype=np.int32)
    tail_start_i = int(tail_start)
    tail_end = tail_start_i + int(tail_len)

    raw_idx = np.zeros((Tq, raw_slots), dtype=np.int32)
    raw_mask = np.zeros((Tq, raw_slots), dtype=np.bool_)
    summary_idx = np.zeros((Tq, summary_slots), dtype=np.int32)
    summary_mask = np.zeros((Tq, summary_slots), dtype=np.bool_)
    offsets = np.arange(raw_slots, dtype=np.int32)

    for row_idx, q_pos_t in enumerate(q_pos_np.tolist()):
        q_pos = int(q_pos_t)
        block_idx = min(max(0, q_pos // max(1, block_size)), max(0, num_blocks - 1))
        local_start = max(0, q_pos - int(local_window_tokens) + 1)
        raw_positions = local_start + offsets
        raw_valid = (
            (raw_positions <= q_pos)
            & (raw_positions < int(kv_len))
            & (raw_positions >= tail_start_i)
            & (raw_positions < tail_end)
        )
        raw_idx[row_idx] = np.clip(raw_positions - tail_start_i, 0, max(0, int(tail_len) - 1))
        raw_mask[row_idx] = raw_valid

        cursor = 0
        complete_summary_blocks = int(num_summary_blocks)
        for neighbor in neighbors[block_idx].tolist():
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor >= block_idx or neighbor >= complete_summary_blocks:
                continue
            neighbor_end = (neighbor + 1) * block_size
            if neighbor_end > local_start:
                continue
            summary_idx[row_idx, cursor] = neighbor
            summary_mask[row_idx, cursor] = True
            cursor += 1
            if cursor >= summary_slots:
                break

    return (
        mx.array(raw_idx, dtype=mx.int32),
        mx.array(raw_mask, dtype=mx.bool_),
        mx.array(summary_idx, dtype=mx.int32),
        mx.array(summary_mask, dtype=mx.bool_),
    )


def _swa_stream_from_cache(
    q: mx.array,
    tail_k: mx.array,
    tail_v: mx.array,
    *,
    query_positions: mx.array,
    tail_start: int,
    n_win: int,
    scale: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    """SWA stream over the cached tail keys/values, returning `(o, l, m)`.

    Each query at absolute position `p = query_positions[qi]` attends to the
    subset of `tail_k` whose absolute positions lie in `[p - n_win + 1, p]`.
    `tail_k[i]` has absolute position `tail_start + i`.
    """
    B, H, Tq, dh = q.shape
    n_tail = int(tail_k.shape[2])

    q_pos = mx.reshape(query_positions, (Tq, 1)).astype(mx.int32)
    k_abs = mx.reshape(
        tail_start + mx.arange(n_tail, dtype=mx.int32),
        (1, n_tail),
    )
    mask_2d = (k_abs <= q_pos) & (k_abs > q_pos - int(n_win))  # [Tq, n_tail]
    mask = mx.reshape(mask_2d, (1, 1, Tq, n_tail))

    scores = mx.matmul(
        q.astype(mx.float32),
        tail_k.transpose(0, 1, 3, 2).astype(mx.float32),
    ) * scale  # [B, H, Tq, n_tail]
    scores = mx.where(mask, scores, NEG_INF)
    m = mx.max(scores, axis=-1, keepdims=True)
    e = mx.exp(scores - m)
    e = mx.where(mask, e, mx.zeros_like(e))
    l = mx.sum(e, axis=-1, keepdims=True)
    any_valid = mx.any(mask_2d, axis=-1, keepdims=True).reshape(1, 1, Tq, 1)
    o = mx.matmul(e, tail_v.astype(mx.float32)) / mx.maximum(l, EPS)
    o = mx.where(any_valid, o, mx.zeros_like(o))
    l = mx.where(any_valid, l, mx.zeros_like(l))
    m = mx.where(any_valid, m, mx.full(m.shape, -1e30, dtype=mx.float32))
    mx.eval(o, l, m)
    return o.astype(q.dtype), l, m


def _compressed_stream_from_cache(
    q: mx.array,
    k_summary: mx.array,
    v_summary: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    query_positions: mx.array,
    local_window_tokens: int,
    scale: float,
    query_chunk_size: int = 0,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Routed-compressed-block stream returning `(o, l, m)` for `Tq` active queries.

    Per query position `p`: build the set of compressed-block indices to attend
    to (Butterfly-routed neighbors of block `p // block_size`, filtered against
    the SWA window so the streams do not double-count). Gather + SDPA, with
    `query_chunk_size` controlling the per-chunk graph flush.
    """
    B, H, Tq, dh = q.shape
    num_summary_blocks = int(k_summary.shape[2])
    block_size = int(layout.block_size)
    summary_slots = int(layout.block_neighbors.shape[-1])

    # Build [Tq, summary_slots] routed-block indices (per-query, head-shared).
    q_pos_np = np.asarray(query_positions, dtype=np.int32)
    if q_pos_np.ndim != 1:
        raise ValueError(f"query_positions must be 1D, got {q_pos_np.shape}")
    routed_np = np.full((int(Tq), summary_slots), -1, dtype=np.int32)
    neighbors_h0 = np.asarray(layout.block_neighbors[0], dtype=np.int32)
    n_win = int(local_window_tokens)
    for qi, p_t in enumerate(q_pos_np.tolist()):
        p = int(p_t)
        b = p // block_size
        local_start = max(0, p - n_win + 1)
        cursor = 0
        if b < 0 or b >= int(neighbors_h0.shape[0]):
            continue
        for neighbor in neighbors_h0[b].tolist():
            n_i = int(neighbor)
            if n_i < 0 or n_i >= b:
                continue
            n_end = (n_i + 1) * block_size
            if n_end > local_start:
                continue
            if n_i >= num_summary_blocks:
                continue
            routed_np[qi, cursor] = n_i
            cursor += 1
            if cursor >= summary_slots:
                break
    routed = mx.array(routed_np, dtype=mx.int32)
    valid_q = (routed_np >= 0)  # [Tq, summary_slots]

    q_chunk = int(query_chunk_size)
    if q_chunk <= 0 or q_chunk >= Tq:
        q_chunk = int(Tq)

    # Pad k_summary with sentinel slot at index num_summary_blocks.
    pad_zero_k = mx.zeros((B, H, 1, dh), dtype=k_summary.dtype)
    k_padded = mx.concatenate([k_summary, pad_zero_k], axis=2)
    v_padded = mx.concatenate([v_summary, pad_zero_k], axis=2)

    o_chunks: list[mx.array] = []
    l_chunks: list[mx.array] = []
    m_chunks: list[mx.array] = []
    for start in range(0, int(Tq), q_chunk):
        end = min(int(Tq), start + q_chunk)
        q_c = q[:, :, start:end, :]
        c = end - start
        idx_c = mx.array(
            np.where(routed_np[start:end] >= 0, routed_np[start:end], num_summary_blocks).astype(np.int32),
            dtype=mx.int32,
        )  # [c, summary_slots]
        valid_c = mx.array(valid_q[start:end], dtype=mx.bool_)  # [c, summary_slots]

        k_g = mx.take(k_padded, idx_c, axis=2)  # [B, H, c, summary_slots, dh]
        v_g = mx.take(v_padded, idx_c, axis=2)  # [B, H, c, summary_slots, dh]
        mask_c = valid_c.reshape(1, 1, c, summary_slots)

        scores = mx.sum(q_c[:, :, :, None, :].astype(mx.float32) * k_g.astype(mx.float32), axis=-1) * scale
        scores = mx.where(mask_c, scores, NEG_INF)
        m_c = mx.max(scores, axis=-1, keepdims=True)
        e = mx.exp(scores - m_c)
        e = mx.where(mask_c, e, mx.zeros_like(e))
        l_c = mx.sum(e, axis=-1, keepdims=True)
        o_c = mx.sum(e[:, :, :, :, None] * v_g.astype(mx.float32), axis=3) / mx.maximum(l_c, EPS)
        any_valid = mx.any(mask_c, axis=-1, keepdims=True)
        o_c = mx.where(any_valid, o_c, mx.zeros_like(o_c))
        l_c = mx.where(any_valid, l_c, mx.zeros_like(l_c))
        m_c = mx.where(any_valid, m_c, mx.full(m_c.shape, -1e30, dtype=mx.float32))
        mx.eval(o_c, l_c, m_c)
        o_chunks.append(o_c)
        l_chunks.append(l_c)
        m_chunks.append(m_c)

    if len(o_chunks) == 1:
        o, l, m = o_chunks[0], l_chunks[0], m_chunks[0]
    else:
        o = mx.concatenate(o_chunks, axis=2)
        l = mx.concatenate(l_chunks, axis=2)
        m = mx.concatenate(m_chunks, axis=2)
    return o.astype(q.dtype), l, m


def compressed_butterfly_attention_from_cache(
    q: mx.array,
    tail_k: mx.array,
    tail_v: mx.array,
    k_summary: mx.array,
    v_summary: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    query_positions: mx.array,
    local_window_tokens: int,
    tail_start: int,
    kv_len: int,
    return_weights: bool = False,
    scale: Optional[float] = None,
    query_chunk_size: int = 0,
) -> Tuple[mx.array, mx.array | None]:
    """Cache-aware compressed Butterfly attention via two streams + online merge.

    Replaces the prior gather-raw + gather-summary + concatenate + single SDPA
    pattern. Per active query:
      Stream 1 (SWA): attention over the cached tail (recent raw tokens).
      Stream 2 (compressed routed): attention over Butterfly-routed compressed
        block summaries strictly older than the SWA window.
    Streams are merged by online softmax `(m, l, o)` accumulators so the union
    K/V tensor is never materialized.

    `query_chunk_size > 0` controls per-chunk graph evaluation in the
    compressed stream (the SWA stream's keys/values are already small —
    bounded by `tail_k.shape[2]` — so it does not need chunking).
    """
    if return_weights:
        raise NotImplementedError("return_weights is not supported for compressed cache attention")
    B, H, Tq, dh = q.shape
    if int(tail_k.shape[0]) != B or int(tail_v.shape[0]) != B:
        raise ValueError("q and cache batch dimensions must match")
    if int(tail_k.shape[1]) != H or int(tail_v.shape[1]) != H:
        raise ValueError("q and cache head dimensions must match")
    if int(tail_k.shape[-1]) != dh or int(tail_v.shape[-1]) != dh:
        raise ValueError("q and cache head_dim mismatch")

    q_scale = float(scale if scale is not None else (dh ** -0.5))
    n_win = int(local_window_tokens)

    swa_o, swa_l, swa_m = _swa_stream_from_cache(
        q, tail_k, tail_v,
        query_positions=query_positions,
        tail_start=int(tail_start),
        n_win=n_win,
        scale=q_scale,
    )
    cmp_o, cmp_l, cmp_m = _compressed_stream_from_cache(
        q, k_summary, v_summary,
        layout=layout,
        query_positions=query_positions,
        local_window_tokens=n_win,
        scale=q_scale,
        query_chunk_size=int(query_chunk_size),
    )

    out = _online_softmax_merge((swa_o, swa_l, swa_m), (cmp_o, cmp_l, cmp_m))
    return out.astype(tail_v.dtype), None


def block_sparse_butterfly_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    return_weights: bool = False,
    block_chunk_size: int = 0,
    scale: Optional[float] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Block-sparse Butterfly attention for full-prefix prefill."""
    if return_weights:
        raise NotImplementedError("return_weights is not supported for block-sparse MLX attention")

    B, H, T, dh = q.shape
    Bk, Hk, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Hk != H or Hv != H:
        raise ValueError("q, k, v must share batch/head dimensions for block-sparse attention")
    if Tk != T or Tv != T:
        raise ValueError("Prefill block-sparse attention requires k/v length to match q length")
    if dhk != dh or dhv != dh:
        raise ValueError("q, k, v head_dim mismatch")
    if int(layout.seq_len) != int(T):
        raise ValueError(f"layout.seq_len={layout.seq_len} must match q length {T}")

    num_blocks = int(layout.num_blocks)
    block_size = int(layout.block_size)
    key_tokens_per_block = int(layout.block_token_idx.shape[-1])
    if num_blocks * block_size < T:
        raise ValueError("layout does not cover the requested sequence length")

    q_scale = float(scale if scale is not None else (dh ** -0.5))
    pad_len = int(num_blocks * block_size - T)
    if pad_len > 0:
        q_work = mx.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
    else:
        q_work = q

    q_blocks = q_work.reshape(B, H, num_blocks, block_size, dh)
    chunk_size = int(block_chunk_size)
    if chunk_size <= 0:
        chunk_size = min(int(num_blocks), 4)
    chunk_size = max(1, min(int(num_blocks), int(chunk_size)))

    out_chunks: list[mx.array] = []
    for start in range(0, num_blocks, chunk_size):
        end = min(start + chunk_size, num_blocks)
        n_chunk = int(end - start)

        q_chunk = q_blocks[:, :, start:end, :, :]
        q_chunk = q_chunk.transpose(0, 2, 1, 3, 4).reshape(B * n_chunk, H, block_size, dh)

        idx_chunk = mx.take(layout.block_token_idx, mx.arange(start, end, dtype=mx.int32), axis=0)
        k_chunk = mx.take(k, idx_chunk, axis=2)
        v_chunk = mx.take(v, idx_chunk, axis=2)
        k_chunk = k_chunk.transpose(0, 2, 1, 3, 4).reshape(B * n_chunk, H, key_tokens_per_block, dh)
        v_chunk = v_chunk.transpose(0, 2, 1, 3, 4).reshape(B * n_chunk, H, key_tokens_per_block, dh)

        mask_chunk = mx.take(layout.block_causal_mask, mx.arange(start, end, dtype=mx.int32), axis=0)
        mask_chunk = mx.broadcast_to(
            mask_chunk[None, :, None, :, :],
            (B, n_chunk, H, block_size, key_tokens_per_block),
        ).reshape(B * n_chunk, H, block_size, key_tokens_per_block)

        if hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention"):
            y_chunk = mx.fast.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                scale=q_scale,
                mask=mask_chunk,
            ).astype(v.dtype)
        else:
            y_chunk = _block_sparse_attention_manual(
                q_chunk,
                k_chunk,
                v_chunk,
                mask=mask_chunk,
                scale=q_scale,
            )

        out_chunks.append(y_chunk.reshape(B, n_chunk, H, block_size, dh))

    out = mx.concatenate(out_chunks, axis=1).transpose(0, 2, 1, 3, 4)
    out = out.reshape(B, H, num_blocks * block_size, dh)
    if pad_len > 0:
        out = out[:, :, :T, :]
    return out.astype(v.dtype), None


def block_sparse_butterfly_attention_active(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    layout: BlockSparseButterflyLayout,
    query_positions: mx.array,
    return_weights: bool = False,
    scale: Optional[float] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Block-sparse Butterfly attention for active decode rows."""
    B, H, Tq, dh = q.shape
    Bk, Hk, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Hk != H or Hv != H:
        raise ValueError("q, k, v must share batch/head dimensions for block-sparse active attention")
    if Tv != Tk:
        raise ValueError("k and v sequence length mismatch")
    if dhk != dh or dhv != dh:
        raise ValueError("q, k, v head_dim mismatch")

    safe_idx, causal_mask = _build_block_sparse_active_indices(
        layout,
        kv_len=int(Tk),
        query_positions=query_positions,
    )

    if (
        not return_weights
        and Tq <= 32
    ):
        from bna.mlx.kernels.metal import has_sparse_neighbor_kernel

        if has_sparse_neighbor_kernel():
            try:
                return _sparse_gather_attention_metal(q, k, v, safe_idx, causal_mask), None
            except Exception:
                pass

    if not return_weights:
        return _sparse_gather_attention_vectorized_active(q, k, v, safe_idx, causal_mask), None

    del scale  # Same scaling behavior as sparse gather path.
    ys: list[mx.array] = []
    ws: list[mx.array] = []
    q_scale = 1.0 / math.sqrt(dh)
    for h in range(H):
        q_h = q[:, h]
        k_h = k[:, h]
        v_h = v[:, h]
        idx_h = safe_idx[h]
        mask_h = causal_mask[h]

        k_g = k_h[:, idx_h]
        v_g = v_h[:, idx_h]
        scores = mx.sum(
            q_h[:, :, None, :].astype(mx.float32) * k_g.astype(mx.float32),
            axis=-1,
        ) * q_scale
        w_h = stable_masked_softmax(scores, mask_h[None, :, :], axis=-1)
        y_h = mx.sum(w_h[:, :, :, None] * v_g.astype(mx.float32), axis=2)
        ys.append(y_h.astype(v.dtype))
        ws.append(w_h)

    return mx.stack(ys, axis=1), mx.stack(ws, axis=1)


def permute_cycle_window_attention_single(
    q_h: mx.array,
    k_h: mx.array,
    v_h: mx.array,
    *,
    perm: np.ndarray | List[int],
    window: int,
    return_weights: bool = False,
    pre_perm_mx: Optional[mx.array] = None,
    pre_inv_perm: Optional[mx.array] = None,
    pre_pi_idx_clamped: Optional[mx.array] = None,
    pre_valid_mask: Optional[mx.array] = None,
    pre_causal_mask: Optional[mx.array] = None,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
    circular: bool = False,
) -> Tuple[mx.array, mx.array | None, float, float]:
    """Single-head permute-to-cycle-order local-window attention.

    q_h/k_h/v_h: [B,T,dh]
    Optional pre_* params skip recomputation of permute artifacts.
    edge_type_bias_scalar: approximate cycle bias for permute path (offset ±1 = cycle).
    window_drop_prob: fraction of non-cycle, non-self window offsets to drop during training.
    circular: use modular wrap-around instead of linear clamping for cycle boundaries.
    """
    _B, T, dh = q_h.shape

    t0 = _now_ms()
    if pre_perm_mx is not None:
        perm_mx = pre_perm_mx
        inv_perm = pre_inv_perm
        pi_idx_clamped = pre_pi_idx_clamped
        valid = pre_valid_mask
        permute_ms = 0.0
    else:
        perm_arr = np.asarray(perm, dtype=np.int32)
        if perm_arr.shape != (T,):
            raise ValueError(f"perm must be shape ({T},), got {perm_arr.shape}")
        perm_mx = mx.array(perm_arr, dtype=mx.int32)
        inv_perm = mx.argsort(perm_mx)

        W = 2 * window + 1
        offsets = mx.arange(-window, window + 1, dtype=mx.int32)
        pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
        if circular:
            pi_idx_clamped = pi_idx % T
            valid = mx.ones((T, W), dtype=mx.bool_)
        else:
            valid = (pi_idx >= 0) & (pi_idx < T)
            pi_idx_clamped = mx.clip(pi_idx, 0, T - 1)
        permute_ms = _now_ms() - t0

    q_pi = q_h[:, perm_mx]
    k_pi = k_h[:, perm_mx]
    v_pi = v_h[:, perm_mx]

    t1 = _now_ms()
    k_win = k_pi[:, pi_idx_clamped]  # [B,T,W,dh]
    v_win = v_pi[:, pi_idx_clamped]

    scores = mx.sum(
        q_pi[:, :, None, :].astype(mx.float32) * k_win.astype(mx.float32),
        axis=-1,
    ) / math.sqrt(dh)

    # Approximate edge-type bias: offset ±1 in permuted space = cycle neighbor
    if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
        W_actual = pi_idx_clamped.shape[-1]
        center = W_actual // 2
        bias_vec = mx.zeros((W_actual,), dtype=mx.float32)
        # offset -1 and +1 from center are cycle neighbors
        if center > 0:
            bias_vec = bias_vec.at[center - 1].add(edge_type_bias_scalar)
        if center + 1 < W_actual:
            bias_vec = bias_vec.at[center + 1].add(edge_type_bias_scalar)
        scores = scores + bias_vec.reshape(1, 1, W_actual)

    if pre_causal_mask is not None:
        mask = valid & pre_causal_mask
    else:
        orig_idx = perm_mx
        neigh_orig = orig_idx[pi_idx_clamped]
        query_orig = orig_idx.reshape(T, 1)
        causal = neigh_orig <= query_orig
        mask = valid & causal

    # Window-drop regularization in permute path
    if training and window_drop_prob > 0.0:
        W_actual = pi_idx_clamped.shape[-1]
        center = W_actual // 2
        # Create mask that preserves self (center) and cycle neighbors (center±1)
        preserve = mx.zeros((T, W_actual), dtype=mx.bool_)
        preserve = preserve.at[:, center].add(True)
        if center > 0:
            preserve = preserve.at[:, center - 1].add(True)
        if center + 1 < W_actual:
            preserve = preserve.at[:, center + 1].add(True)
        drop_rand = mx.random.uniform(shape=(T, W_actual)) < window_drop_prob
        drop = drop_rand & (~preserve)
        mask = mask & (~drop)

    w = stable_masked_softmax(scores, mask[None, :, :], axis=-1)
    y_pi = mx.sum(w[:, :, :, None] * v_win.astype(mx.float32), axis=2)
    y_h = y_pi[:, inv_perm].astype(v_h.dtype)
    attn_ms = _now_ms() - t1

    if return_weights:
        return y_h, w, permute_ms, attn_ms
    return y_h, None, permute_ms, attn_ms


def butterfly_permute_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    graph: MLXGraphABI,
    *,
    window: int,
    return_weights: bool = False,
    cache: Optional["_GraphCache"] = None,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
) -> Tuple[mx.array, mx.array | None, float, float]:
    """Fast path: permute to cycle order then contiguous local attention."""
    _B, H, T, _dh = q.shape

    cycle_perms = graph.meta.get("cycle_perms")
    all_cycle_perms = graph.meta.get("all_cycle_perms")
    if not isinstance(cycle_perms, list) or len(cycle_perms) < H:
        raise ValueError("graph.meta['cycle_perms'] is required for permute path")

    ys: list[mx.array] = []
    ws: list[mx.array] = []
    permute_ms = 0.0
    attention_ms = 0.0

    for h in range(H):
        perms_h: list[Any] = []
        if (
            isinstance(all_cycle_perms, list)
            and h < len(all_cycle_perms)
            and isinstance(all_cycle_perms[h], list)
            and len(all_cycle_perms[h]) > 0
        ):
            perms_h = list(all_cycle_perms[h])
        else:
            perm = cycle_perms[h]
            if perm is None:
                raise ValueError(f"Missing cycle permutation for head {h}")
            perms_h = [perm]

        y_passes: list[mx.array] = []
        w_passes: list[mx.array] = []
        for perm in perms_h:
            kwargs: Dict[str, Any] = {
                "perm": perm,
                "window": window,
                "return_weights": return_weights,
                "edge_type_bias_scalar": edge_type_bias_scalar,
                "window_drop_prob": window_drop_prob,
                "training": training,
            }
            if cache is not None and len(perms_h) == 1:
                kwargs["pre_perm_mx"] = cache.perm_mx[h]
                kwargs["pre_inv_perm"] = cache.inv_perm[h]
                kwargs["pre_pi_idx_clamped"] = cache.pi_idx_clamped[h]
                kwargs["pre_valid_mask"] = cache.valid_mask[h]
                kwargs["pre_causal_mask"] = cache.causal_masks[h]

            y_h, w_h, p_ms, a_ms = permute_cycle_window_attention_single(
                q[:, h], k[:, h], v[:, h], **kwargs
            )
            y_passes.append(y_h.astype(mx.float32))
            permute_ms += p_ms
            attention_ms += a_ms
            if return_weights and w_h is not None:
                w_passes.append(w_h.astype(mx.float32))

        y_head = y_passes[0] if len(y_passes) == 1 else mx.mean(mx.stack(y_passes, axis=0), axis=0)
        ys.append(y_head.astype(v.dtype))
        if return_weights:
            if len(w_passes) == 1:
                ws.append(w_passes[0].astype(mx.float32))
            elif len(w_passes) > 1:
                ws.append(mx.mean(mx.stack(w_passes, axis=0), axis=0).astype(mx.float32))

    y = mx.stack(ys, axis=1)  # [B,H,T,dh]
    if return_weights:
        return y, mx.stack(ws, axis=1), permute_ms, attention_ms
    return y, None, permute_ms, attention_ms


def butterfly_permute_window_attention_batched(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    all_pi_idx: Optional[mx.array] = None,
    all_valid: Optional[mx.array] = None,
    all_causal: Optional[mx.array] = None,
    window: int = 64,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
    head_chunk_size: Optional[int] = None,
    query_chunk_size: int = 256,
    prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto",
    memory_budget_bytes: Optional[int] = None,
    retro_backfill_enabled: bool = False,
    retro_backfill_alpha: float = 0.0,
    retro_backfill_training_only: bool = True,
    retro_backfill_causal_only: bool = True,
    log_progress: bool = False,
    circular: bool = False,
    multi_cycle_mode: Literal["average", "union"] = "average",
    use_fused_dispatch: bool = True,
    scale: Optional[float] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Chunked permute-window attention across heads and query positions.

    Args:
        q: [B, Hq, T, dh]
        k, v: [B, Hkv, T, dh] (Hkv may be < Hq for GQA)
        all_perms: [Hq, T] int32 — cycle permutations per query head
        all_inv_perms: [Hq, T] int32 — inverse permutations per query head
        all_pi_idx/all_valid/all_causal: deprecated compatibility args; ignored
        window: half-window size (W = 2*window + 1)
        edge_type_bias_scalar: cycle-neighbor bias (offsets ±1 from center)
        window_drop_prob: fraction of non-cycle window edges to drop during training
        training: whether in training mode
        head_chunk_size: number of heads per chunk (None => all heads)
        query_chunk_size: number of permuted query positions per chunk
        prepermute_mode: pre-permute behavior per head chunk:
            `off` = no pre-permute, `kv` = pre-permute K/V only,
            `qkv`/`on` = pre-permute Q/K/V, `auto` = inference-time KV pre-permute
            when query chunking is active
        memory_budget_bytes: optional peak-memory budget for planner in `auto` mode
        retro_backfill_enabled: enable optional future->past backfill in cycle order
        retro_backfill_alpha: residual scale for retrocausal contribution
        retro_backfill_training_only: apply retro backfill only when training=True
        retro_backfill_causal_only: enforce original-index causality for backfill edges
        log_progress: emit chunk-level progress prints
        circular: use modular wrap-around instead of linear clamping for cycle boundaries
        multi_cycle_mode: "average" runs d independent passes and averages (default),
            "union" builds a single union multigraph with multiplicity bias
        use_fused_dispatch: attempt all-head fused path when eligible (default True)
    Returns:
        y: [B, H, T, dh]
        weights: None (not supported in batched path)
    """
    B, Hq, T, dh = q.shape
    Bk, Hkv, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Tk != T or Tv != T:
        raise ValueError("k/v must share batch and sequence dims with q")
    if dhk != dh:
        raise ValueError("k head_dim must match q head_dim")
    if Hv != Hkv:
        raise ValueError("k and v must have same head count")
    if all_perms.ndim == 3:
        if all_inv_perms.ndim != 3:
            raise ValueError(
                "all_inv_perms must be 3D when all_perms is 3D, got "
                f"{all_inv_perms.shape}"
            )
        if all_perms.shape[0] != Hq or all_perms.shape[2] != T:
            raise ValueError(
                f"all_perms must be shape ({Hq}, d, {T}), got {all_perms.shape}"
            )
        if all_inv_perms.shape != all_perms.shape:
            raise ValueError(
                f"all_inv_perms must match all_perms shape {all_perms.shape}, "
                f"got {all_inv_perms.shape}"
            )

        if multi_cycle_mode == "union":
            union_idx, mult, vmask = build_union_multigraph_index(
                all_perms, all_inv_perms, window=window, circular=circular,
            )
            return _union_multigraph_attention(
                q, k, v,
                union_neigh_idx=union_idx,
                multiplicity=mult,
                valid_mask=vmask,
            ), None

        # Default: average mode
        d = int(all_perms.shape[1])
        ys: list[mx.array] = []
        for c in range(d):
            y_c, _ = butterfly_permute_window_attention_batched(
                q,
                k,
                v,
                all_perms=all_perms[:, c, :],
                all_inv_perms=all_inv_perms[:, c, :],
                all_pi_idx=all_pi_idx,
                all_valid=all_valid,
                all_causal=all_causal,
                window=window,
                edge_type_bias_scalar=edge_type_bias_scalar,
                window_drop_prob=window_drop_prob,
                training=training,
                head_chunk_size=head_chunk_size,
                query_chunk_size=query_chunk_size,
                prepermute_mode=prepermute_mode,
                memory_budget_bytes=memory_budget_bytes,
                retro_backfill_enabled=retro_backfill_enabled,
                retro_backfill_alpha=retro_backfill_alpha,
                retro_backfill_training_only=retro_backfill_training_only,
                retro_backfill_causal_only=retro_backfill_causal_only,
                log_progress=log_progress,
                circular=circular,
                multi_cycle_mode=multi_cycle_mode,
            )
            ys.append(y_c.astype(mx.float32))
        if len(ys) == 1:
            return ys[0].astype(v.dtype), None
        return mx.mean(mx.stack(ys, axis=0), axis=0).astype(v.dtype), None

    if all_perms.shape != (Hq, T):
        raise ValueError(f"all_perms must be shape ({Hq}, {T}) or [H,d,T], got {all_perms.shape}")
    if all_inv_perms.shape != (Hq, T):
        raise ValueError(
            f"all_inv_perms must be shape ({Hq}, {T}) or [H,d,T], got {all_inv_perms.shape}"
        )
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv} for GQA")

    if all_pi_idx is not None:
        W_in = int(all_pi_idx.shape[-1])
        if W_in > 0:
            window = (W_in - 1) // 2
    window = int(max(0, window))

    # --- Fused all-head dispatch: single lazy graph, no per-head eval barriers ---
    if _fused_dispatch_eligible(
        all_perms=all_perms,
        edge_type_bias_scalar=edge_type_bias_scalar,
        window_drop_prob=window_drop_prob,
        training=training,
        retro_backfill_enabled=retro_backfill_enabled,
        circular=circular,
        multi_cycle_mode=multi_cycle_mode,
        use_fused_dispatch=use_fused_dispatch,
    ):
        # --- Chunked-gather + SDPA: per-chunk gathers from original Q/K/V ---
        # Avoids full-T permutation copies while using MLX's fast SDPA.
        from bna.mlx.fused_attention import (
            butterfly_fused_permute_window_attention_chunked_gather,
        )
        y = butterfly_fused_permute_window_attention_chunked_gather(
            q, k, v,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            window=window,
            query_chunk_size=query_chunk_size,
            scale=scale,
        )
        return y, None

    h_chunk = Hq if head_chunk_size is None else int(max(1, head_chunk_size))
    h_chunk = min(h_chunk, Hq)
    q_chunk = int(max(1, query_chunk_size))
    q_chunk = min(q_chunk, T)
    num_h_chunks = (Hq + h_chunk - 1) // h_chunk

    # Causal-safe retro: disabled by default, training-only unless explicitly allowed
    retro_active = (
        bool(retro_backfill_enabled)
        and float(retro_backfill_alpha) != 0.0
        and (training or (not retro_backfill_training_only))
    )

    kv_repeat = Hq // Hkv
    q_to_kv_head = np.arange(Hq, dtype=np.int32) // kv_repeat
    scale = scale if scale is not None else 1.0 / math.sqrt(dh)

    y_chunks: list[mx.array] = []
    for chunk_idx, h0 in enumerate(range(0, Hq, h_chunk), start=1):
        h1 = min(h0 + h_chunk, Hq)
        hc = h1 - h0
        if log_progress:
            print(
                f"      permute_batched: head_chunk {chunk_idx}/{num_h_chunks} heads[{h0}:{h1}] start",
                flush=True,
            )

        q_c = q[:, h0:h1, :, :]
        kv_heads = q_to_kv_head[h0:h1]
        if hc == 1:
            kv_h = int(kv_heads[0])
            k_c = k[:, kv_h : kv_h + 1, :, :]
            v_c = v[:, kv_h : kv_h + 1, :, :]
        elif np.all(kv_heads == kv_heads[0]):
            # Keep shared-KV groups as broadcasted views to avoid materializing
            # full [B, hc, T, dh] copies before query chunking.
            kv_h = int(kv_heads[0])
            k_base = k[:, kv_h : kv_h + 1, :, :]
            v_base = v[:, kv_h : kv_h + 1, :, :]
            k_c = mx.broadcast_to(k_base, (B, hc, T, dh))
            v_c = mx.broadcast_to(v_base, (B, hc, T, dh))
        else:
            kv_idx = mx.array(kv_heads, dtype=mx.int32)
            k_c = mx.take(k, kv_idx, axis=1)
            v_c = mx.take(v, kv_idx, axis=1)

        perms_c = all_perms[h0:h1, :]
        inv_c = all_inv_perms[h0:h1, :]
        q_chunk_count = (T + q_chunk - 1) // q_chunk
        prepermute_mode_l = str(prepermute_mode).lower()
        if prepermute_mode_l in {"on", "always", "true", "qkv"}:
            prepermute_q = True
            prepermute_kv = True
        elif prepermute_mode_l in {"kv"}:
            prepermute_q = False
            prepermute_kv = True
        elif prepermute_mode_l in {"off", "never", "false"}:
            prepermute_q = False
            prepermute_kv = False
        else:
            kblk_worst = min(T, q_chunk + 2 * window)
            qblk_worst = q_chunk
            elem_bytes = int(v.itemsize)
            chunk_elems = (qblk_worst + 2 * kblk_worst) * dh + (qblk_worst * kblk_worst)

            # Candidate plans evaluated with simple cost model:
            # minimize moved elements, optionally under explicit peak-memory budget.
            candidates = [
                ("off", False, False),
                ("kv", False, True),
                ("qkv", True, True),
            ]
            scored: list[tuple[float, float, str, bool, bool]] = []
            for name, cand_q, cand_kv in candidates:
                moved_elems = float(
                    T + (2 * T if cand_kv else (2 * q_chunk_count * kblk_worst))
                )
                extra_elems = float((T * dh if cand_q else 0) + (2 * T * dh if cand_kv else 0))
                peak_bytes = float((chunk_elems + extra_elems) * B * hc * elem_bytes)
                scored.append((moved_elems, peak_bytes, name, cand_q, cand_kv))

            chosen: tuple[float, float, str, bool, bool]
            if memory_budget_bytes is not None:
                feasible = [s for s in scored if s[1] <= float(memory_budget_bytes)]
                if feasible:
                    chosen = min(feasible, key=lambda s: (s[0], s[1]))
                else:
                    chosen = min(scored, key=lambda s: (s[1], s[0]))
            else:
                chosen = min(scored, key=lambda s: (s[0], s[1]))

            _moved, _peak, chosen_name, prepermute_q, prepermute_kv = chosen
            if log_progress:
                budget_str = (
                    "none" if memory_budget_bytes is None else f"{int(memory_budget_bytes)}"
                )
                print(
                    f"      permute_batched: planner mode={chosen_name} "
                    f"budget_bytes={budget_str} moved={_moved:.0f} peak_bytes={_peak:.0f}",
                    flush=True,
                )

        q_pi_buf: Optional[mx.array] = None
        k_pi_buf: Optional[mx.array] = None
        v_pi_buf: Optional[mx.array] = None
        if prepermute_q or prepermute_kv:
            perm_gidx = mx.broadcast_to(perms_c[None, :, :, None], (B, hc, T, 1))
            if prepermute_q:
                q_pi_buf = mx.take_along_axis(q_c, perm_gidx, axis=2)
                q_c = None
            if prepermute_kv:
                k_pi_buf = mx.take_along_axis(k_c, perm_gidx, axis=2)
                v_pi_buf = mx.take_along_axis(v_c, perm_gidx, axis=2)
                k_c = None
                v_c = None

        y_pi_chunks: list[mx.array] = []
        for q_chunk_idx, s in enumerate(range(0, T, q_chunk), start=1):
            e = min(T, s + q_chunk)
            ks = max(0, s - window)
            ke = min(T, e + window)
            if circular:
                k_range_raw = mx.arange(
                    s - window, e + window, dtype=mx.int32,
                )
                k_range = k_range_raw % T
                Kblk = int(k_range.shape[0])
            else:
                Kblk = ke - ks

            if log_progress:
                k_desc = (
                    f"k_circ[{Kblk}]" if circular
                    else f"k[{ks}:{ke}]"
                )
                print(
                    f"        permute_batched: heads[{h0}:{h1}] "
                    f"q_chunk {q_chunk_idx}/{q_chunk_count} "
                    f"q[{s}:{e}] {k_desc}",
                    flush=True,
                )

            q_idx = perms_c[:, s:e]  # [hc, Qblk]
            if circular:
                k_idx = mx.take(perms_c, k_range, axis=1)
            else:
                k_idx = perms_c[:, ks:ke]  # [hc, Kblk]
            if prepermute_q:
                q_blk = q_pi_buf[:, :, s:e, :]  # type: ignore[index]
            else:
                q_gidx = mx.broadcast_to(
                    q_idx[None, :, :, None], (B, hc, e - s, 1),
                )
                q_blk = mx.take_along_axis(q_c, q_gidx, axis=2)
            if prepermute_kv:
                if circular:
                    k_blk = mx.take(
                        k_pi_buf, k_range, axis=2,  # type: ignore
                    )
                    v_blk = mx.take(
                        v_pi_buf, k_range, axis=2,  # type: ignore
                    )
                else:
                    k_blk = k_pi_buf[:, :, ks:ke, :]  # type: ignore
                    v_blk = v_pi_buf[:, :, ks:ke, :]  # type: ignore
            else:
                k_gidx = mx.broadcast_to(
                    k_idx[None, :, :, None], (B, hc, Kblk, 1),
                )
                k_blk = mx.take_along_axis(k_c, k_gidx, axis=2)
                v_blk = mx.take_along_axis(v_c, k_gidx, axis=2)

            q_pos = mx.arange(s, e, dtype=mx.int32)
            q_pos = q_pos.reshape(1, e - s, 1)
            if circular:
                k_pos = k_range_raw.reshape(1, 1, Kblk)
            else:
                k_pos = mx.arange(ks, ke, dtype=mx.int32)
                k_pos = k_pos.reshape(1, 1, Kblk)
            rel = k_pos - q_pos  # [1, Qblk, Kblk]

            in_window = (rel >= -window) & (rel <= window)
            orig_q = q_idx
            orig_k = k_idx
            causal = orig_k[:, None, :] <= orig_q[:, :, None]
            mask_eff = in_window & causal
            use_fused_local_sdpa = (
                hasattr(mx, "fast")
                and hasattr(mx.fast, "scaled_dot_product_attention")
                and (edge_type_bias_scalar is None
                     or edge_type_bias_scalar == 0.0)
                and (window_drop_prob <= 0.0 or (not training))
            )
            if training and window_drop_prob > 0.0:
                preserve = (rel == 0) | (rel == -1) | (rel == 1)
                drop_rand = (
                    mx.random.uniform(shape=(hc, e - s, Kblk))
                    < window_drop_prob
                )
                drop = drop_rand & in_window & (~preserve)
                mask_eff = mask_eff & (~drop)

            if use_fused_local_sdpa:
                y_blk = mx.fast.scaled_dot_product_attention(
                    q_blk,
                    k_blk,
                    v_blk,
                    scale=scale,
                    mask=mask_eff[None, :, :, :],
                ).astype(v.dtype)
            else:
                scores = mx.matmul(
                    q_blk,
                    k_blk.transpose(0, 1, 3, 2),
                ) * scale

                if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
                    cycle_nb = (rel == -1) | (rel == 1)
                    scores = scores + cycle_nb.astype(mx.float32) * float(edge_type_bias_scalar)

                w = stable_masked_softmax(
                    scores,
                    mask_eff[None, :, :, :],
                    axis=-1,
                    preserve_dtype=True,
                )
                y_blk = mx.matmul(w, v_blk).astype(v.dtype)
                scores = None
                w = None
            y_pi_chunks.append(y_blk)
            q_blk = None
            k_blk = None
            v_blk = None
            y_blk = None

        if not y_pi_chunks:  # pragma: no cover
            y_pi = mx.zeros((B, hc, T, dh), dtype=v.dtype)
        elif len(y_pi_chunks) == 1:
            y_pi = y_pi_chunks[0]
        else:
            y_pi = mx.concatenate(y_pi_chunks, axis=2)

        if retro_active and T > 1:
            # Simplified Hamiltonian-local retro backfill:
            # In permuted cycle order, each position receives residual from successor (+1).
            # This is O(T) and keeps extra memory bounded.
            alpha = mx.array(retro_backfill_alpha, dtype=v.dtype)
            tail = mx.zeros((B, hc, 1, dh), dtype=v.dtype)
            retro_term = mx.concatenate([y_pi[:, :, 1:, :], tail], axis=2)
            valid = mx.concatenate(
                [
                    mx.ones((1, 1, T - 1, 1), dtype=v.dtype),
                    mx.zeros((1, 1, 1, 1), dtype=v.dtype),
                ],
                axis=2,
            )

            if retro_backfill_causal_only:
                # Allow backfill only when successor is not in original future.
                causal_ok = perms_c[:, 1:] <= perms_c[:, :-1]  # [hc, T-1]
                causal_ok = mx.concatenate(
                    [
                        causal_ok,
                        mx.zeros((hc, 1), dtype=mx.bool_),
                    ],
                    axis=1,
                )  # [hc, T]
                valid = valid * causal_ok[None, :, :, None].astype(v.dtype)

            y_pi = y_pi + alpha * retro_term * valid

        inv_idx = inv_c[None, :, :, None]
        inv_idx = mx.broadcast_to(inv_idx, (B, hc, T, 1))
        y_h = mx.take_along_axis(y_pi, inv_idx, axis=2).astype(v.dtype)
        y_chunks.append(y_h)
        mx.eval(y_h)
        if log_progress:
            print(
                f"      permute_batched: head_chunk {chunk_idx}/{num_h_chunks} heads[{h0}:{h1}] done",
                flush=True,
            )

    if not y_chunks:  # pragma: no cover
        return mx.zeros((B, Hq, T, dh), dtype=v.dtype), None
    if len(y_chunks) == 1:
        return y_chunks[0], None
    return mx.concatenate(y_chunks, axis=1), None


def butterfly_covering_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    window: int,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
    head_chunk_size: Optional[int] = None,
    query_chunk_size: int = 256,
    prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto",
    memory_budget_bytes: Optional[int] = None,
    circular: bool = False,
    multi_cycle_mode: Literal["average", "union"] = "average",
) -> Tuple[mx.array, mx.array | None]:
    """Covering-mode attention by averaging one permute-window pass per cycle."""
    return butterfly_permute_window_attention_batched(
        q,
        k,
        v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        window=window,
        edge_type_bias_scalar=edge_type_bias_scalar,
        window_drop_prob=window_drop_prob,
        training=training,
        head_chunk_size=head_chunk_size,
        query_chunk_size=query_chunk_size,
        prepermute_mode=prepermute_mode,
        memory_budget_bytes=memory_budget_bytes,
        circular=circular,
        multi_cycle_mode=multi_cycle_mode,
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
        log_progress=False,
    )


def butterfly_permute_window_attention_active_batched(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    query_positions: mx.array,
    window: int = 64,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
    head_chunk_size: Optional[int] = None,
    query_chunk_size: int = 256,
    prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto",
    memory_budget_bytes: Optional[int] = None,
    log_progress: bool = False,
    circular: bool = False,
    multi_cycle_mode: Literal["average", "union"] = "average",
    use_fused_dispatch: bool = False,
    prefer_gather_for_small_tq: bool = False,
    scale: Optional[float] = None,
) -> Tuple[mx.array, mx.array | None]:
    """Permute-window attention for active query rows (Q_len <= K_len).

    q: [B, Hq, Tq, dh] for active queries only
    k, v: [B, Hkv, Tk, dh] for currently available cache prefix
    all_perms/all_inv_perms: [Hq, Tg] with Tg >= Tk (adaptive graph horizon)
    query_positions: [Tq] original token positions for each query row in q
    circular: use modular wrap-around instead of linear clamping.
    multi_cycle_mode: "average" or "union" for multi-cycle handling.
    use_fused_dispatch: attempt all-head fused path when eligible (default False;
        the active-row fused path currently adds graph compilation overhead that
        exceeds the eval barrier savings)
    prefer_gather_for_small_tq: when True, fused active dispatch may skip
        the Tg==Tk full-prefill route for tiny query blocks and directly use
        flat-index gather (decode-oriented path).
    """
    B, Hq, Tq, dh = q.shape
    Bk, Hkv, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Tk != Tv:
        raise ValueError("k/v must share batch and sequence dims")
    if dhk != dh:
        raise ValueError("k head_dim must match q head_dim")
    if Hv != Hkv:
        raise ValueError("k and v must have same head count")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv} for GQA")
    if all_perms.ndim == 3:
        if all_inv_perms.ndim != 3:
            raise ValueError(
                "all_inv_perms must be 3D when all_perms is 3D, got "
                f"{all_inv_perms.shape}"
            )
        if all_perms.shape[0] != Hq:
            raise ValueError(f"all_perms first dim must be {Hq}, got {all_perms.shape}")
        if all_inv_perms.shape != all_perms.shape:
            raise ValueError(
                f"all_inv_perms must match all_perms shape {all_perms.shape}, "
                f"got {all_inv_perms.shape}"
            )

        if multi_cycle_mode == "union":
            Tg = int(all_perms.shape[2])
            union_idx, mult, vmask = build_union_multigraph_index(
                all_perms, all_inv_perms, window=window, circular=circular,
            )
            # For active queries: expand k/v to full Tg, run union attention,
            # then select only the query_positions rows from the output.
            # Pad k/v to Tg if needed.
            if Tk < Tg:
                pad_shape = (B, Hkv, Tg - Tk, dh)
                k_full = mx.concatenate([k, mx.zeros(pad_shape, dtype=k.dtype)], axis=2)
                v_full = mx.concatenate([v, mx.zeros(pad_shape, dtype=v.dtype)], axis=2)
            else:
                k_full = k[:, :, :Tg, :]
                v_full = v[:, :, :Tg, :]
            # Build full Q at Tg (zeros for non-query positions)
            q_full = mx.zeros((B, Hq, Tg, dh), dtype=q.dtype)
            qp_np = np.array(query_positions, dtype=np.int64)
            for qi in range(Tq):
                pos = int(qp_np[qi])
                if 0 <= pos < Tg:
                    q_full[:, :, pos, :] = q[:, :, qi, :]
            y_full = _union_multigraph_attention(
                q_full, k_full, v_full,
                union_neigh_idx=union_idx,
                multiplicity=mult,
                valid_mask=vmask,
            )
            # Extract query rows
            y_rows = []
            for qi in range(Tq):
                pos = int(qp_np[qi])
                y_rows.append(y_full[:, :, pos, :])
            return mx.stack(y_rows, axis=2), None

        # Default: average mode
        d = int(all_perms.shape[1])
        ys: list[mx.array] = []
        for c in range(d):
            y_c, _ = butterfly_permute_window_attention_active_batched(
                q,
                k,
                v,
                all_perms=all_perms[:, c, :],
                all_inv_perms=all_inv_perms[:, c, :],
                query_positions=query_positions,
                window=window,
                edge_type_bias_scalar=edge_type_bias_scalar,
                window_drop_prob=window_drop_prob,
                training=training,
                head_chunk_size=head_chunk_size,
                query_chunk_size=query_chunk_size,
                prepermute_mode=prepermute_mode,
                memory_budget_bytes=memory_budget_bytes,
                log_progress=log_progress,
                circular=circular,
                multi_cycle_mode=multi_cycle_mode,
                use_fused_dispatch=use_fused_dispatch,
                prefer_gather_for_small_tq=prefer_gather_for_small_tq,
                scale=scale,
            )
            ys.append(y_c.astype(mx.float32))
        if len(ys) == 1:
            return ys[0].astype(v.dtype), None
        return (
            mx.mean(mx.stack(ys, axis=0), axis=0).astype(v.dtype),
            None,
        )

    if all_perms.shape[0] != Hq:
        raise ValueError(
            f"all_perms first dim must be {Hq}, got {all_perms.shape}"
        )
    del memory_budget_bytes  # Not used in the 2-D path.
    Tg = int(all_perms.shape[1])
    if all_inv_perms.shape != (Hq, Tg):
        raise ValueError(
            f"all_inv_perms must be shape ({Hq}, {Tg}) or [H,d,T], got {all_inv_perms.shape}"
        )
    if Tg < Tk:
        raise ValueError(f"graph sequence length Tg={Tg} must be >= Tk={Tk}")
    if tuple(query_positions.shape) != (Tq,):
        raise ValueError(f"query_positions must be shape ({Tq},), got {query_positions.shape}")

    # --- Fused all-head dispatch for active-row path ---
    # Uses a more permissive eligibility check than the full-prefill path
    # because the vectorized active-row function handles circular + edge bias.
    if _fused_active_dispatch_eligible(
        all_perms=all_perms,
        window_drop_prob=window_drop_prob,
        training=training,
        multi_cycle_mode=multi_cycle_mode,
        use_fused_dispatch=use_fused_dispatch,
    ):
        from bna.mlx.kernels.metal import has_discovered_fused_attention_kernel

        if has_discovered_fused_attention_kernel():
            from bna.mlx.fused_attention import (
                butterfly_fused_permute_window_attention_active_metal,
            )
            try:
                y = butterfly_fused_permute_window_attention_active_metal(
                    q, k, v,
                    all_perms=all_perms,
                    all_inv_perms=all_inv_perms,
                    query_positions=query_positions,
                    window=int(max(0, window)),
                    query_chunk_size=query_chunk_size,
                    scale=scale,
                )
                return y, None
            except Exception:
                pass

        from bna.mlx.fused_attention import (
            butterfly_fused_permute_window_attention_active,
        )
        y = butterfly_fused_permute_window_attention_active(
            q, k, v,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            query_positions=query_positions,
            window=int(max(0, window)),
            query_chunk_size=query_chunk_size,
            circular=circular,
            edge_type_bias_scalar=edge_type_bias_scalar,
            prefer_gather_for_small_tq=prefer_gather_for_small_tq,
            scale=scale,
        )
        return y, None

    q_pos_all = query_positions.astype(mx.int32)
    window = int(max(0, window))
    q_chunk = int(max(1, min(int(query_chunk_size), Tq)))

    h_chunk = Hq if head_chunk_size is None else int(max(1, head_chunk_size))
    h_chunk = min(h_chunk, Hq)
    num_h_chunks = (Hq + h_chunk - 1) // h_chunk

    prepermute_mode_l = str(prepermute_mode).lower()
    if prepermute_mode_l in {"on", "always", "true", "qkv", "kv"}:
        prepermute_kv = True
    else:
        # "auto" defaults to false for active-row mode to keep work O(Tq*W).
        prepermute_kv = False
    # Cannot prepermute K/V by full graph order when Tg > Tk.
    if Tg != Tk:
        prepermute_kv = False

    kv_repeat = Hq // Hkv
    q_to_kv_head = np.arange(Hq, dtype=np.int32) // kv_repeat
    offsets = mx.arange(-window, window + 1, dtype=mx.int32).reshape(1, 2 * window + 1)
    scale = scale if scale is not None else 1.0 / math.sqrt(dh)

    y_head_chunks: list[mx.array] = []
    for chunk_idx, h0 in enumerate(range(0, Hq, h_chunk), start=1):
        h1 = min(h0 + h_chunk, Hq)
        if log_progress:
            print(
                f"      permute_active: head_chunk {chunk_idx}/{num_h_chunks} heads[{h0}:{h1}] start",
                flush=True,
            )

        y_heads_local: list[mx.array] = []
        for h in range(h0, h1):
            q_h = q[:, h, :, :]  # [B, Tq, dh]
            kv_h = int(q_to_kv_head[h])
            k_h = k[:, kv_h, :, :]  # [B, Tk, dh]
            v_h = v[:, kv_h, :, :]  # [B, Tk, dh]

            perm_h = all_perms[h, :].astype(mx.int32)  # [Tg]
            inv_h = all_inv_perms[h, :].astype(mx.int32)  # [Tg]

            if prepermute_kv:
                k_src = mx.take(k_h, perm_h, axis=1)  # [B, Tk, dh]
                v_src = mx.take(v_h, perm_h, axis=1)  # [B, Tk, dh]
            else:
                k_src = k_h
                v_src = v_h

            y_q_chunks: list[mx.array] = []
            q_chunk_count = (Tq + q_chunk - 1) // q_chunk
            for q_chunk_idx, s in enumerate(range(0, Tq, q_chunk), start=1):
                e = min(Tq, s + q_chunk)
                q_blk = q_h[:, s:e, :]  # [B, Qblk, dh]
                q_pos = q_pos_all[s:e]  # [Qblk] original positions

                q_rank = mx.take(inv_h, q_pos, axis=0)  # [Qblk], in [0, Tg)
                k_rank = q_rank.reshape(-1, 1) + offsets  # [Qblk, W]
                if circular:
                    k_rank_clipped = (k_rank % Tg).astype(mx.int32)
                    valid = mx.ones(k_rank.shape, dtype=mx.bool_)
                else:
                    valid = (k_rank >= 0) & (k_rank < Tg)
                    k_rank_clipped = mx.clip(
                        k_rank, 0, Tg - 1,
                    ).astype(mx.int32)

                # Convert local cycle-window ranks back to original token indices.
                k_orig = mx.take(perm_h, k_rank_clipped, axis=0).astype(mx.int32)  # [Qblk, W]
                available = k_orig < Tk
                causal = k_orig <= q_pos.reshape(-1, 1)
                mask_eff = valid & available & causal

                gather_idx = k_rank_clipped if prepermute_kv else mx.clip(k_orig, 0, Tk - 1)
                k_blk = mx.take(k_src, gather_idx, axis=1)  # [B, Qblk, W, dh]
                v_blk = mx.take(v_src, gather_idx, axis=1)  # [B, Qblk, W, dh]

                if training and window_drop_prob > 0.0:
                    # Same resilience-preserving rule as batched permute path:
                    # keep self and cycle-adjacent offsets, drop only others.
                    preserve = (
                        (k_rank == q_rank.reshape(-1, 1))
                        | (k_rank == (q_rank.reshape(-1, 1) - 1))
                        | (k_rank == (q_rank.reshape(-1, 1) + 1))
                    )
                    drop_rand = mx.random.uniform(shape=mask_eff.shape) < window_drop_prob
                    drop = drop_rand & valid & (~preserve)
                    mask_eff = mask_eff & (~drop)

                # Batched matmul is substantially faster than explicit elementwise
                # multiply + reduction for MLA head dimensions.
                scores = (
                    mx.matmul(
                        q_blk.astype(mx.float32).reshape(B * (e - s), 1, dh),
                        k_blk.astype(mx.float32).reshape(B * (e - s), k_blk.shape[2], dh).transpose(0, 2, 1),
                    ).reshape(B, e - s, k_blk.shape[2])
                    * scale
                )
                if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
                    cycle_nb = valid & (
                        (k_rank == (q_rank.reshape(-1, 1) - 1))
                        | (k_rank == (q_rank.reshape(-1, 1) + 1))
                    )
                    scores = scores + cycle_nb.astype(mx.float32)[None, :, :] * float(
                        edge_type_bias_scalar
                    )
                w = stable_masked_softmax(
                    scores,
                    mask_eff[None, :, :],
                    axis=-1,
                    preserve_dtype=True,
                )
                y_blk = mx.matmul(
                    w.reshape(B * (e - s), 1, w.shape[-1]).astype(mx.float32),
                    v_blk.astype(mx.float32).reshape(B * (e - s), v_blk.shape[2], dh),
                ).reshape(B, e - s, dh).astype(v.dtype)

                y_q_chunks.append(y_blk)
                if log_progress:
                    print(
                        f"        permute_active: head={h} q_chunk {q_chunk_idx}/{q_chunk_count} q[{s}:{e}]",
                        flush=True,
                    )

            if not y_q_chunks:  # pragma: no cover
                y_h = mx.zeros((B, Tq, dh), dtype=v.dtype)
            elif len(y_q_chunks) == 1:
                y_h = y_q_chunks[0]
            else:
                y_h = mx.concatenate(y_q_chunks, axis=1)
            y_heads_local.append(y_h)

            # Drop references eagerly between heads.
            q_h = None
            k_h = None
            v_h = None
            k_src = None
            v_src = None
            perm_h = None
            inv_h = None

        if len(y_heads_local) == 1:
            y_local = y_heads_local[0][:, None, :, :]
        else:
            y_local = mx.stack(y_heads_local, axis=1)
        y_head_chunks.append(y_local)
        # Skip eval for small Tq (decode) to allow cross-layer graph fusion.
        # The eval bounds memory during large-Tq multi-chunk active prefill.
        if Tq > q_chunk:
            mx.eval(y_local)
        if log_progress:
            print(
                f"      permute_active: head_chunk {chunk_idx}/{num_h_chunks} heads[{h0}:{h1}] done",
                flush=True,
            )

    if not y_head_chunks:  # pragma: no cover
        return mx.zeros((B, Hq, Tq, dh), dtype=v.dtype), None
    if len(y_head_chunks) == 1:
        return y_head_chunks[0], None
    return mx.concatenate(y_head_chunks, axis=1), None


class ButterflyAttentionMLX(nn.Module):
    """Butterfly sparse attention in MLX with ABI-driven graph construction."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        *,
        routing_dim: Optional[int] = None,
        dropout: float = 0.0,
        window: int = 64,
        landmark_stride: Optional[int] = 64,
        strategy: Literal["random", "greedy", "online_insertion", "regular_partition"] = "random",
        num_cycles: int | str = 1,
        edge_disjoint: bool = True,
        regular_num_clusters: int = 8,
        seed: int = 0,
        path: Literal["sparse", "permute"] = "sparse",
        edge_bias: bool = False,
        window_drop: float = 0.0,
        compiled_graph_dir: Optional[str] = None,
        retro_backfill_enabled: bool = False,
        retro_backfill_alpha: float = 0.0,
        retro_backfill_training_only: bool = True,
        retro_backfill_causal_only: bool = True,
        permute_head_chunk_size: Optional[int] = None,
        permute_query_chunk_size: int = 256,
        circular: bool = False,
        multi_cycle_mode: Literal["average", "union"] = "average",
        verify_spectral_gap: bool = False,
        spectral_gap_threshold: float = 4.0,
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
        self._num_cycles_raw = num_cycles
        self.num_cycles = 1 if num_cycles == "auto" else int(num_cycles)
        self.edge_disjoint = bool(edge_disjoint)
        self.regular_num_clusters = int(max(1, regular_num_clusters))
        self.seed = int(seed)
        self.path = path
        self.window_drop_prob = float(window_drop)
        self.compiled_graph_dir = compiled_graph_dir
        self.retro_backfill_enabled = bool(retro_backfill_enabled)
        self.retro_backfill_alpha = float(retro_backfill_alpha)
        self.retro_backfill_training_only = bool(retro_backfill_training_only)
        self.retro_backfill_causal_only = bool(retro_backfill_causal_only)
        self.circular = bool(circular)
        self.multi_cycle_mode = str(multi_cycle_mode)
        self.verify_spectral_gap = bool(verify_spectral_gap)
        self.spectral_gap_threshold = float(max(0.0, spectral_gap_threshold))
        self.permute_head_chunk_size = (
            None if permute_head_chunk_size is None else int(max(1, permute_head_chunk_size))
        )
        self.permute_query_chunk_size = int(max(1, permute_query_chunk_size))

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.Wr = nn.Linear(self.n_embd, self.n_heads * self.routing_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if edge_bias:
            # 4 scalars for [CYCLE, WINDOW, LANDMARK, REWIRE]
            self.edge_type_bias = mx.zeros((4,))
        else:
            self.edge_type_bias = None

        self.topology = Topology(
            n_heads=self.n_heads,
            strategy=self.strategy,
            num_cycles=self.num_cycles,
            edge_disjoint=self.edge_disjoint,
            regular_num_clusters=self.regular_num_clusters,
            seed=self.seed,
            window=self.window,
            landmark_stride=self.landmark_stride,
            enforce_hamiltonian=True,
        )

        # Debug state — intentionally NOT stored as mx.array attributes
        # to avoid polluting the nn.Module parameter tree.
        self.last_profile: AttentionProfile = AttentionProfile(path=path)
        self.last_graph_abi: Optional[WayfinderGraphABI] = None
        self._last_attn_weights_np: Optional[np.ndarray] = None
        self._runtime_window_drop_override: Optional[float] = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)

    def set_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set per-step controls from training schedules."""
        self._runtime_window_drop_override = (
            None if window_drop is None else float(min(1.0, max(0.0, window_drop)))
        )
        self._runtime_schedule_bias_vec = _schedule_bias_to_vec(schedule_bias)

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)

    def cache_persistent_bytes(self) -> int:
        cache = _GRAPH_CACHE_STORE.get(id(self))
        return int(cache.persistent_bytes) if cache is not None else 0

    @property
    def _cache_mode(self) -> str:
        """'static' for input-independent strategies, 'dynamic' otherwise."""
        return self.topology.cache_mode

    def _resolve_and_sync_num_cycles(self, T: int) -> None:
        """Resolve 'auto' num_cycles at graph-construction time and rebuild topology."""
        if self._num_cycles_raw != "auto":
            return
        from bna.cycles import recommended_num_cycles

        resolved = recommended_num_cycles(T)
        if resolved != self.num_cycles:
            self.num_cycles = resolved
            self.topology = Topology(
                n_heads=self.n_heads,
                strategy=self.strategy,
                num_cycles=resolved,
                edge_disjoint=self.edge_disjoint,
                regular_num_clusters=self.regular_num_clusters,
                seed=self.seed,
                window=self.window,
                landmark_stride=self.landmark_stride,
                enforce_hamiltonian=True,
            )

    def _cache_key_from_T(self, T: int) -> tuple:
        return (
            T,
            self.strategy,
            self.num_cycles,
            self.edge_disjoint,
            self.regular_num_clusters,
            self.window,
            self.landmark_stride,
            self.seed,
            self.path,
            str(Path(self.compiled_graph_dir).resolve()) if self.compiled_graph_dir else None,
        )

    def _build_cache(
        self,
        mlx_graph: MLXGraphABI,
        numpy_abi: WayfinderGraphABI,
        T: int,
        *,
        cache_key: tuple,
        source: str = "runtime",
        artifact_dir: str | None = None,
    ) -> _GraphCache:
        """Precompute all reusable artifacts from graph."""
        s_idx = safe_neighbor_idx(mlx_graph.neigh_idx, T)
        c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, T)

        # Permute-path artifacts
        perm_mx_list: List[mx.array] = []
        inv_perm_list: List[mx.array] = []
        pi_idx_clamped_list: List[mx.array] = []
        valid_mask_list: List[mx.array] = []
        causal_masks_list: List[mx.array] = []

        cycle_perms = mlx_graph.meta.get("cycle_perms", [])
        all_cycle_perms = mlx_graph.meta.get("all_cycle_perms", [])
        W = 2 * self.window + 1
        offsets = mx.arange(-self.window, self.window + 1, dtype=mx.int32)
        per_head_perms: list[list[mx.array]] = []
        per_head_invs: list[list[mx.array]] = []
        max_d = 1

        for h in range(self.n_heads):
            perms_h: list[mx.array] = []
            invs_h: list[mx.array] = []

            perms_src = None
            if (
                isinstance(all_cycle_perms, list)
                and h < len(all_cycle_perms)
                and isinstance(all_cycle_perms[h], list)
                and len(all_cycle_perms[h]) > 0
            ):
                perms_src = all_cycle_perms[h]
            elif (
                isinstance(cycle_perms, list)
                and h < len(cycle_perms)
                and cycle_perms[h] is not None
            ):
                perms_src = [cycle_perms[h]]

            if perms_src is None:
                p_mx = mx.zeros((T,), dtype=mx.int32)
                ip = mx.zeros((T,), dtype=mx.int32)
                perms_h.append(p_mx)
                invs_h.append(ip)
            else:
                for perm in perms_src:
                    perm_arr = np.asarray(perm, dtype=np.int32)
                    p_mx = mx.array(perm_arr, dtype=mx.int32)
                    perms_h.append(p_mx)
                    invs_h.append(mx.argsort(p_mx))

            p_mx = perms_h[0]
            ip = invs_h[0]
            pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
            if self.circular:
                pi_clamped = pi_idx % T
                valid = mx.ones((T, W), dtype=mx.bool_)
            else:
                valid = (pi_idx >= 0) & (pi_idx < T)
                pi_clamped = mx.clip(pi_idx, 0, T - 1)
            # Causal mask in permuted space
            orig_idx = p_mx
            neigh_orig = orig_idx[pi_clamped]
            query_orig = orig_idx.reshape(T, 1)
            causal_h = neigh_orig <= query_orig

            max_d = max(max_d, len(perms_h))
            per_head_perms.append(perms_h)
            per_head_invs.append(invs_h)
            perm_mx_list.append(p_mx)
            inv_perm_list.append(ip)
            pi_idx_clamped_list.append(pi_clamped)
            valid_mask_list.append(valid)
            causal_masks_list.append(causal_h)

        persistent_bytes = _mx_nbytes(mlx_graph.neigh_idx) + _mx_nbytes(mlx_graph.edge_type)
        persistent_bytes += _mx_nbytes(s_idx) + _mx_nbytes(c_mask)
        for arr in perm_mx_list + inv_perm_list + pi_idx_clamped_list + valid_mask_list + causal_masks_list:
            persistent_bytes += _mx_nbytes(arr)
        if max_d == 1:
            all_perms = mx.stack([p[0] for p in per_head_perms], axis=0)
            all_inv_perms = mx.stack([p[0] for p in per_head_invs], axis=0)
        else:
            perm_heads: list[mx.array] = []
            inv_heads: list[mx.array] = []
            for perms_h, invs_h in zip(per_head_perms, per_head_invs):
                while len(perms_h) < max_d:
                    perms_h.append(perms_h[0])
                    invs_h.append(invs_h[0])
                perm_heads.append(mx.stack(perms_h, axis=0))
                inv_heads.append(mx.stack(invs_h, axis=0))
            all_perms = mx.stack(perm_heads, axis=0)
            all_inv_perms = mx.stack(inv_heads, axis=0)
        persistent_bytes += _mx_nbytes(all_perms) + _mx_nbytes(all_inv_perms)

        return _GraphCache(
            mlx_graph=mlx_graph,
            numpy_abi=numpy_abi,
            safe_idx=s_idx,
            causal_mask=c_mask,
            perm_mx=perm_mx_list,
            inv_perm=inv_perm_list,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            pi_idx_clamped=pi_idx_clamped_list,
            valid_mask=valid_mask_list,
            causal_masks=causal_masks_list,
            cache_key=cache_key,
            source=source,
            artifact_dir=artifact_dir,
            persistent_bytes=int(persistent_bytes),
        )

    def _load_compiled_cache(self, T: int, cache_key: tuple) -> _GraphCache | None:
        if not self.compiled_graph_dir:
            return None
        art_dir = Path(self.compiled_graph_dir)
        ni_path = art_dir / "neighborindex.npz"
        meta_path = art_dir / "meta.json"
        if not ni_path.exists():
            return None

        payload = np.load(ni_path)
        neigh_idx = np.asarray(payload["neigh_idx"], dtype=np.int32)
        edge_type = np.asarray(payload["edge_type"], dtype=np.uint8)
        if neigh_idx.shape != edge_type.shape:
            return None
        if neigh_idx.ndim == 2:
            neigh_idx = np.broadcast_to(neigh_idx[None, :, :], (self.n_heads, *neigh_idx.shape))
            edge_type = np.broadcast_to(edge_type[None, :, :], (self.n_heads, *edge_type.shape))
        if neigh_idx.ndim != 3:
            return None
        if int(neigh_idx.shape[0]) != self.n_heads or int(neigh_idx.shape[1]) != T:
            return None

        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        meta.setdefault("cycle_perms", [])
        meta.setdefault("all_cycle_perms", [])
        meta.setdefault("max_degree", int(neigh_idx.shape[-1]))
        meta.setdefault("seq_len", int(T))
        meta.setdefault("n_heads", int(self.n_heads))

        numpy_abi = WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)
        validate_graph_abi(
            numpy_abi,
            expect_heads=self.n_heads,
            expect_tokens=T,
            enforce_hamiltonian=True,
        )
        self.last_graph_abi = numpy_abi
        mlx_graph = to_mlx_graph_abi(numpy_abi, heads=self.n_heads, validate=False)
        return self._build_cache(
            mlx_graph,
            numpy_abi,
            T,
            cache_key=cache_key,
            source="compiled",
            artifact_dir=str(art_dir),
        )

    def _get_or_build_cache(self, x: mx.array) -> tuple[_GraphCache, bool]:
        """Return cache and whether this call was served by cache hit."""
        _B, T, _C = x.shape
        cache_key = self._cache_key_from_T(int(T))

        existing = _GRAPH_CACHE_STORE.get(id(self))
        if (
            self._cache_mode == "static"
            and existing is not None
            and existing.cache_key == cache_key
        ):
            return existing, True

        compiled_cache = self._load_compiled_cache(int(T), cache_key)
        if compiled_cache is not None:
            if self._cache_mode == "static":
                _GRAPH_CACHE_STORE[id(self)] = compiled_cache
            return compiled_cache, False

        mlx_graph, numpy_abi = self._build_graph_abi(x)
        cache = self._build_cache(
            mlx_graph,
            numpy_abi,
            int(T),
            cache_key=cache_key,
            source="runtime",
        )
        if self._cache_mode == "static":
            _GRAPH_CACHE_STORE[id(self)] = cache
        return cache, False

    def _build_graph_abi(self, x: mx.array) -> Tuple[MLXGraphABI, WayfinderGraphABI]:
        """Build graph ABI on CPU from routing embeddings for this forward."""
        _B, T, _C = x.shape
        self._resolve_and_sync_num_cycles(int(T))

        routing_by_head: list[torch.Tensor] | None = None
        if self.strategy in {"greedy", "online_insertion"}:
            r = self.Wr(x[0]).reshape(T, self.n_heads, self.routing_dim).transpose(1, 0, 2)
            mx.eval(r)
            r_np = np.asarray(r, dtype=np.float32)
            routing_by_head = [torch.from_numpy(r_np[h]) for h in range(self.n_heads)]

        topo_graph = self.topology.construct(
            {"T": int(T), "include_self": True},
            routing_by_head=routing_by_head,
        )
        abi = topo_graph.abi
        if self.verify_spectral_gap:
            perm = None
            all_cycle_perms = abi.meta.get("all_cycle_perms")
            if isinstance(all_cycle_perms, list) and all_cycle_perms:
                first_head = all_cycle_perms[0]
                if isinstance(first_head, list) and first_head:
                    first_cycle = first_head[0]
                    if first_cycle is not None:
                        perm = first_cycle
            cycle_perms = abi.meta.get("cycle_perms")
            if perm is None and isinstance(cycle_perms, list) and cycle_perms:
                if cycle_perms[0] is not None:
                    perm = cycle_perms[0]

            if perm is not None:
                perm_np = np.asarray(perm, dtype=np.int64)
                if int(T) <= 4096:
                    gap_info = spectral_gap(
                        perm_np,
                        include_window=True,
                        window=self.window,
                        expander_threshold=self.spectral_gap_threshold,
                    )
                else:
                    gap_info = expansion_proxy(
                        perm_np,
                        window=self.window,
                        num_walks=512,
                        walk_len=max(20, int(np.ceil(2.0 * np.log2(max(2, int(T)))))),
                    )
                    gap_info["expander_threshold"] = self.spectral_gap_threshold
                    gap_info["is_good_expander"] = bool(gap_info.get("is_fast_mixer", False))
                abi.meta["spectral_verification"] = gap_info
                if not bool(gap_info.get("is_good_expander", False)):
                    warnings.warn(
                        "Cycle expansion check failed: "
                        f"{gap_info}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        self.last_graph_abi = abi
        return to_mlx_graph_abi(abi, heads=self.n_heads, validate=False), abi

    def __call__(
        self,
        x: mx.array,
        *,
        return_debug: bool = False,
        topology_graph: Optional[TopologyGraph] = None,
    ) -> mx.array | tuple[mx.array, Dict[str, Any]]:
        t_total0 = _now_ms()
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        t_graph0 = _now_ms()
        if topology_graph is not None:
            cache = self._build_cache(
                to_mlx_graph_abi(topology_graph.abi, heads=self.n_heads, validate=False),
                topology_graph.abi,
                int(T),
                cache_key=("injected", int(T), self.path),
                source=topology_graph.source,
                artifact_dir=topology_graph.artifact_dir,
            )
            cache_hit = False
        else:
            cache, cache_hit = self._get_or_build_cache(x)
        graph = cache.mlx_graph
        graph_ms = _now_ms() - t_graph0

        is_training = self.training
        effective_window_drop = (
            self.window_drop_prob
            if self._runtime_window_drop_override is None
            else self._runtime_window_drop_override
        )
        scheduled_edge_bias = (
            mx.array(self._runtime_schedule_bias_vec)
            if float(np.abs(self._runtime_schedule_bias_vec).sum()) > 0.0
            else None
        )

        # Build window-drop mask for sparse path (training only)
        wd_mask: Optional[mx.array] = None
        if is_training and effective_window_drop > 0.0 and self.path == "sparse":
            et = graph.edge_type.astype(mx.int32)  # [H, T, D]
            is_window = et == int(2)  # EdgeType.WINDOW
            # Also check that it's not a self-edge
            s_idx = cache.safe_idx  # [H, T, D]
            i_idx = mx.arange(T, dtype=mx.int32).reshape(1, T, 1)
            is_self = s_idx == i_idx
            droppable = is_window & (~is_self)
            drop_rand = mx.random.uniform(shape=et.shape) < effective_window_drop
            wd_mask = ~(droppable & drop_rand)  # True = keep

        # Edge-type bias scalar for permute path
        etb_scalar: Optional[float] = None
        if self.edge_type_bias is not None and self.path == "permute":
            mx.eval(self.edge_type_bias)
            etb_scalar = float(self.edge_type_bias[0].item())
        if scheduled_edge_bias is not None and self.path == "permute":
            mx.eval(scheduled_edge_bias)
            cycle_bias = float(scheduled_edge_bias[0].item())
            etb_scalar = cycle_bias if etb_scalar is None else (etb_scalar + cycle_bias)

        t_attn0 = _now_ms()
        permute_ms = 0.0
        if self.path == "sparse":
            y_h, w = sparse_gather_attention(
                q,
                k,
                v,
                graph,
                return_weights=return_debug,
                precomputed_safe_idx=cache.safe_idx,
                precomputed_causal_mask=cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                edge_type_bias_offset=scheduled_edge_bias,
                window_drop_mask=wd_mask,
            )
        elif self.path == "permute":
            t_perm0 = _now_ms()
            y_h, w = butterfly_permute_window_attention_batched(
                q,
                k,
                v,
                all_perms=cache.all_perms,
                all_inv_perms=cache.all_inv_perms,
                window=self.window,
                edge_type_bias_scalar=etb_scalar,
                window_drop_prob=effective_window_drop if is_training else 0.0,
                training=is_training,
                head_chunk_size=self.permute_head_chunk_size,
                query_chunk_size=self.permute_query_chunk_size,
                retro_backfill_enabled=self.retro_backfill_enabled,
                retro_backfill_alpha=self.retro_backfill_alpha,
                retro_backfill_training_only=self.retro_backfill_training_only,
                retro_backfill_causal_only=self.retro_backfill_causal_only,
                circular=self.circular,
                multi_cycle_mode=self.multi_cycle_mode,
            )
            permute_ms = _now_ms() - t_perm0
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = _now_ms() - t_attn0

        y = y_h.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.out(y)
        y = self.dropout(y)

        total_ms = _now_ms() - t_total0
        self.last_profile = AttentionProfile(
            graph_build_ms=graph_ms,
            permute_ms=permute_ms,
            attention_ms=attn_ms,
            total_ms=total_ms,
            path=self.path,
            notes={
                "seq_len": int(T),
                "max_degree": int(graph.neigh_idx.shape[-1]),
                "cache_hit": bool(cache_hit),
                "cache_mode": self._cache_mode,
                "cache_source": cache.source,
                "cache_persistent_bytes": int(cache.persistent_bytes),
                "window_drop_effective": float(effective_window_drop),
            },
        )
        if return_debug:
            debug: Dict[str, Any] = {
                "graph_abi": self.last_graph_abi,
                "profile": self.last_profile.to_dict(),
                "attn_weights": w,
                "edge_type": graph.edge_type,
                "neigh_idx": graph.neigh_idx,
            }
            return y, debug

        return y


WayfinderAttentionMLX = ButterflyAttentionMLX
wayfinder_permute_window_attention = butterfly_permute_window_attention
wayfinder_permute_window_attention_batched = butterfly_permute_window_attention_batched
wayfinder_covering_attention = butterfly_covering_attention
wayfinder_permute_window_attention_active_batched = (
    butterfly_permute_window_attention_active_batched
)

__all__ = [
    "BlockSparseButterflyLayout",
    "ButterflyAttentionMLX",
    "WayfinderAttentionMLX",
    "build_block_butterfly_layout",
    "block_sparse_butterfly_attention",
    "block_sparse_butterfly_attention_active",
    "compressed_butterfly_attention",
    "compressed_butterfly_attention_active",
    "butterfly_permute_window_attention",
    "butterfly_permute_window_attention_batched",
    "butterfly_covering_attention",
    "butterfly_permute_window_attention_active_batched",
    "wayfinder_permute_window_attention",
    "wayfinder_permute_window_attention_batched",
    "wayfinder_covering_attention",
    "wayfinder_permute_window_attention_active_batched",
    "dense_causal_attention",
    "_compress_profile_reset",
    "_compress_profile_dump",
]
