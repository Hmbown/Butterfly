from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

import mlx.core as mx
import mlx.nn as nn

from hcsa.graph.abi import WayfinderGraphABI, validate_graph_abi
from hcsa.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)
from hcsa.topology import Topology, TopologyGraph


NEG_INF = mx.array(-1e30, dtype=mx.float32)
EPS = mx.array(1e-9, dtype=mx.float32)

# Module-level store for graph caches — keeps mx.arrays out of nn.Module
# parameter trees so optimizers don't try to update them.
_GRAPH_CACHE_STORE: Dict[int, "_GraphCache"] = {}


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
    all_perms: mx.array  # [H, T]
    all_inv_perms: mx.array  # [H, T]
    pi_idx_clamped: List[mx.array]  # H arrays, each [T, W]
    valid_mask: List[mx.array]  # H arrays, each [T, W]
    causal_masks: List[mx.array]  # H arrays, each [T, W]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: str | None = None
    persistent_bytes: int = 0


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
) -> Tuple[mx.array, mx.array | None, float, float]:
    """Single-head permute-to-cycle-order local-window attention.

    q_h/k_h/v_h: [B,T,dh]
    Optional pre_* params skip recomputation of permute artifacts.
    edge_type_bias_scalar: approximate cycle bias for permute path (offset ±1 = cycle).
    window_drop_prob: fraction of non-cycle, non-self window offsets to drop during training.
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


def wayfinder_permute_window_attention(
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
    if not isinstance(cycle_perms, list) or len(cycle_perms) < H:
        raise ValueError("graph.meta['cycle_perms'] is required for permute path")

    ys: list[mx.array] = []
    ws: list[mx.array] = []
    permute_ms = 0.0
    attention_ms = 0.0

    for h in range(H):
        perm = cycle_perms[h]
        if perm is None:
            raise ValueError(f"Missing cycle permutation for head {h}")

        kwargs: Dict[str, Any] = {
            "perm": perm,
            "window": window,
            "return_weights": return_weights,
            "edge_type_bias_scalar": edge_type_bias_scalar,
            "window_drop_prob": window_drop_prob,
            "training": training,
        }
        if cache is not None:
            kwargs["pre_perm_mx"] = cache.perm_mx[h]
            kwargs["pre_inv_perm"] = cache.inv_perm[h]
            kwargs["pre_pi_idx_clamped"] = cache.pi_idx_clamped[h]
            kwargs["pre_valid_mask"] = cache.valid_mask[h]
            kwargs["pre_causal_mask"] = cache.causal_masks[h]

        y_h, w_h, p_ms, a_ms = permute_cycle_window_attention_single(
            q[:, h], k[:, h], v[:, h], **kwargs
        )
        ys.append(y_h)
        permute_ms += p_ms
        attention_ms += a_ms
        if return_weights and w_h is not None:
            ws.append(w_h)

    y = mx.stack(ys, axis=1)  # [B,H,T,dh]
    if return_weights:
        return y, mx.stack(ws, axis=1), permute_ms, attention_ms
    return y, None, permute_ms, attention_ms


def wayfinder_permute_window_attention_batched(
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
        retro_backfill_training_only: apply retro backfill only when training=True (causal-safe default)
        retro_backfill_causal_only: enforce original-index causality for backfill edges
        log_progress: emit chunk-level progress prints
    Returns:
        y: [B, H, T, dh]
        weights: None (not supported in batched path)
    """
    B, Hq, T, dh = q.shape
    Bk, Hkv, Tk, dhk = k.shape
    Bv, Hv, Tv, dhv = v.shape
    if Bk != B or Bv != B or Tk != T or Tv != T:
        raise ValueError("k/v must share batch and sequence dims with q")
    if dhk != dh or dhv != dh:
        raise ValueError("q/k/v head_dim must match")
    if Hv != Hkv:
        raise ValueError("k and v must have same head count")
    if all_perms.shape != (Hq, T):
        raise ValueError(f"all_perms must be shape ({Hq}, {T}), got {all_perms.shape}")
    if all_inv_perms.shape != (Hq, T):
        raise ValueError(f"all_inv_perms must be shape ({Hq}, {T}), got {all_inv_perms.shape}")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv} for GQA")

    if all_pi_idx is not None:
        W_in = int(all_pi_idx.shape[-1])
        if W_in > 0:
            window = (W_in - 1) // 2
    window = int(max(0, window))

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
    scale = 1.0 / math.sqrt(dh)

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
            elem_bytes = int(np.dtype(np.asarray(mx.array(0, dtype=v.dtype)).dtype).itemsize)
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

            if log_progress:
                print(
                    f"        permute_batched: heads[{h0}:{h1}] q_chunk {q_chunk_idx}/{q_chunk_count} "
                    f"q[{s}:{e}] k[{ks}:{ke}]",
                    flush=True,
                )

            q_idx = perms_c[:, s:e]  # [hc, Qblk]
            k_idx = perms_c[:, ks:ke]  # [hc, Kblk]
            if prepermute_q:
                q_blk = q_pi_buf[:, :, s:e, :]  # type: ignore[index]
            else:
                q_gidx = mx.broadcast_to(q_idx[None, :, :, None], (B, hc, e - s, 1))
                q_blk = mx.take_along_axis(q_c, q_gidx, axis=2)
            if prepermute_kv:
                k_blk = k_pi_buf[:, :, ks:ke, :]  # type: ignore[index]
                v_blk = v_pi_buf[:, :, ks:ke, :]  # type: ignore[index]
            else:
                k_gidx = mx.broadcast_to(k_idx[None, :, :, None], (B, hc, ke - ks, 1))
                k_blk = mx.take_along_axis(k_c, k_gidx, axis=2)
                v_blk = mx.take_along_axis(v_c, k_gidx, axis=2)

            q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, e - s, 1)
            k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, ke - ks)
            rel = k_pos - q_pos  # [1, Qblk, Kblk]

            in_window = (rel >= -window) & (rel <= window)
            orig_q = q_idx
            orig_k = k_idx
            causal = orig_k[:, None, :] <= orig_q[:, :, None]
            mask_eff = in_window & causal
            use_fused_local_sdpa = (
                hasattr(mx, "fast")
                and hasattr(mx.fast, "scaled_dot_product_attention")
                and (edge_type_bias_scalar is None or edge_type_bias_scalar == 0.0)
                and (window_drop_prob <= 0.0 or (not training))
            )
            if training and window_drop_prob > 0.0:
                preserve = (rel == 0) | (rel == -1) | (rel == 1)
                drop_rand = mx.random.uniform(shape=(hc, e - s, ke - ks)) < window_drop_prob
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


class WayfinderAttentionMLX(nn.Module):
    """Wayfinder sparse attention in MLX with ABI-driven graph construction."""

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
        retro_backfill_enabled: bool = False,
        retro_backfill_alpha: float = 0.0,
        retro_backfill_training_only: bool = True,
        retro_backfill_causal_only: bool = True,
        permute_head_chunk_size: Optional[int] = None,
        permute_query_chunk_size: int = 256,
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
        self.retro_backfill_enabled = bool(retro_backfill_enabled)
        self.retro_backfill_alpha = float(retro_backfill_alpha)
        self.retro_backfill_training_only = bool(retro_backfill_training_only)
        self.retro_backfill_causal_only = bool(retro_backfill_causal_only)
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

    def _cache_key_from_T(self, T: int) -> tuple:
        return (
            T,
            self.strategy,
            self.num_cycles,
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
        W = 2 * self.window + 1
        offsets = mx.arange(-self.window, self.window + 1, dtype=mx.int32)

        for h in range(self.n_heads):
            perm = None
            if (
                isinstance(cycle_perms, list)
                and h < len(cycle_perms)
                and cycle_perms[h] is not None
            ):
                perm = cycle_perms[h]
            if perm is not None:
                perm_arr = np.asarray(perm, dtype=np.int32)
                p_mx = mx.array(perm_arr, dtype=mx.int32)
                ip = mx.argsort(p_mx)
                pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
                valid = (pi_idx >= 0) & (pi_idx < T)
                pi_clamped = mx.clip(pi_idx, 0, T - 1)
                # Causal mask in permuted space
                orig_idx = p_mx
                neigh_orig = orig_idx[pi_clamped]
                query_orig = orig_idx.reshape(T, 1)
                causal_h = neigh_orig <= query_orig
            else:
                p_mx = mx.zeros((T,), dtype=mx.int32)
                ip = mx.zeros((T,), dtype=mx.int32)
                pi_clamped = mx.zeros((T, W), dtype=mx.int32)
                valid = mx.zeros((T, W), dtype=mx.bool_)
                causal_h = mx.zeros((T, W), dtype=mx.bool_)

            perm_mx_list.append(p_mx)
            inv_perm_list.append(ip)
            pi_idx_clamped_list.append(pi_clamped)
            valid_mask_list.append(valid)
            causal_masks_list.append(causal_h)

        persistent_bytes = _mx_nbytes(mlx_graph.neigh_idx) + _mx_nbytes(mlx_graph.edge_type)
        persistent_bytes += _mx_nbytes(s_idx) + _mx_nbytes(c_mask)
        for arr in perm_mx_list + inv_perm_list + pi_idx_clamped_list + valid_mask_list + causal_masks_list:
            persistent_bytes += _mx_nbytes(arr)
        all_perms = mx.stack(perm_mx_list, axis=0)
        all_inv_perms = mx.stack(inv_perm_list, axis=0)
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
            y_h, w = wayfinder_permute_window_attention_batched(
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
