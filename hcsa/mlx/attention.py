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

from hcsa.graph_strategies import build_strategy
from hcsa.graph.abi import WayfinderGraphABI, stack_head_abis, validate_graph_abi
from hcsa.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)


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


def stable_masked_softmax(scores_f32: mx.array, mask: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable masked softmax that returns zeros on all-masked rows."""
    masked = mx.where(mask, scores_f32, NEG_INF)
    row_max = mx.max(masked, axis=axis, keepdims=True)
    expv = mx.exp(masked - row_max)
    expv = mx.where(mask, expv, mx.zeros_like(expv))
    denom = mx.sum(expv, axis=axis, keepdims=True)
    return expv / mx.maximum(denom, EPS)


def dense_causal_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    return_weights: bool = False,
) -> Tuple[mx.array, mx.array | None]:
    """Dense causal attention for q/k/v of shape [B,H,T,dh]."""
    B, H, T, dh = q.shape

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
    all_pi_idx: mx.array,
    all_valid: mx.array,
    all_causal: mx.array,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
) -> Tuple[mx.array, mx.array | None]:
    """Vectorized permute-window attention across all heads simultaneously.

    Replaces the per-head Python loop with stacked tensor ops.

    Args:
        q, k, v: [B, H, T, dh]
        all_perms: [H, T] int32 — cycle permutations per head
        all_inv_perms: [H, T] int32 — inverse permutations per head
        all_pi_idx: [H, T, W] int32 — clamped window indices in permuted space
        all_valid: [H, T, W] bool — valid window mask
        all_causal: [H, T, W] bool — causal mask in permuted space
        edge_type_bias_scalar: cycle-neighbor bias (offsets ±1 from center)
        window_drop_prob: fraction of non-cycle window edges to drop during training
        training: whether in training mode
    Returns:
        y: [B, H, T, dh]
        weights: None (not supported in batched path)
    """
    B, H, T, dh = q.shape
    W = all_pi_idx.shape[-1]

    # Permute Q/K/V into cycle order: [B, H, T, dh] -> gather along T axis
    # perm_idx: [H, T] -> broadcast to [B, H, T, dh] for take_along_axis on axis=2
    perm_idx = all_perms[None, :, :, None]  # [1, H, T, 1]
    perm_idx = mx.broadcast_to(perm_idx, (B, H, T, dh))

    q_pi = mx.take_along_axis(q, perm_idx, axis=2)  # [B, H, T, dh]
    k_pi = mx.take_along_axis(k, perm_idx, axis=2)
    v_pi = mx.take_along_axis(v, perm_idx, axis=2)

    # Gather window neighbors in permuted space: [H, T, W] -> [B, H, T, W, dh]
    pi_expanded = all_pi_idx[None, :, :, :, None]  # [1, H, T, W, 1]
    pi_expanded = mx.broadcast_to(pi_expanded, (B, H, T, W, dh))

    # Reshape k_pi/v_pi to [B, H, T, 1, dh] then gather along T via expanded indices
    k_pi_exp = k_pi[:, :, :, None, :]  # [B, H, T, 1, dh]
    k_pi_exp = mx.broadcast_to(k_pi_exp, (B, H, T, W, dh))
    # Use advanced indexing: for each (b,h,t,w), pick k_pi[b,h, pi_idx[h,t,w], :]
    # Flatten T and W dims for take_along_axis on axis=2
    pi_flat = all_pi_idx[None, :, :, :]  # [1, H, T, W]
    pi_flat = mx.broadcast_to(pi_flat, (B, H, T, W))
    pi_flat = pi_flat.reshape(B, H, T * W, 1)  # [B, H, T*W, 1]
    pi_flat = mx.broadcast_to(pi_flat, (B, H, T * W, dh))

    k_win = mx.take_along_axis(k_pi, pi_flat, axis=2).reshape(B, H, T, W, dh)
    v_win = mx.take_along_axis(v_pi, pi_flat, axis=2).reshape(B, H, T, W, dh)

    # Compute attention scores: [B, H, T, W]
    scores = mx.sum(
        q_pi[:, :, :, None, :].astype(mx.float32) * k_win.astype(mx.float32),
        axis=-1,
    ) / math.sqrt(dh)

    # Edge-type bias: offset ±1 from center = cycle neighbors
    if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
        center = W // 2
        bias_vec = mx.zeros((W,), dtype=mx.float32)
        if center > 0:
            bias_vec = bias_vec.at[center - 1].add(edge_type_bias_scalar)
        if center + 1 < W:
            bias_vec = bias_vec.at[center + 1].add(edge_type_bias_scalar)
        scores = scores + bias_vec.reshape(1, 1, 1, W)

    # Combined mask: valid & causal — [H, T, W] broadcast to [B, H, T, W]
    mask = all_valid & all_causal  # [H, T, W]

    # Window-drop regularization
    if training and window_drop_prob > 0.0:
        center = W // 2
        preserve = mx.zeros((T, W), dtype=mx.bool_)
        preserve = preserve.at[:, center].add(True)
        if center > 0:
            preserve = preserve.at[:, center - 1].add(True)
        if center + 1 < W:
            preserve = preserve.at[:, center + 1].add(True)
        drop_rand = mx.random.uniform(shape=(H, T, W)) < window_drop_prob
        drop = drop_rand & (~preserve[None, :, :])
        mask = mask & (~drop)

    # Softmax over window dim
    w = stable_masked_softmax(scores, mask[None, :, :, :], axis=-1)  # [B, H, T, W]

    # Weighted sum: [B, H, T, W, 1] * [B, H, T, W, dh] -> sum over W -> [B, H, T, dh]
    y_pi = mx.sum(w[:, :, :, :, None] * v_win.astype(mx.float32), axis=3)

    # Inverse permute back to original order
    inv_idx = all_inv_perms[None, :, :, None]  # [1, H, T, 1]
    inv_idx = mx.broadcast_to(inv_idx, (B, H, T, dh))
    y = mx.take_along_axis(y_pi, inv_idx, axis=2).astype(v.dtype)

    return y, None


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
            # 4 scalars for [CYCLE, WINDOW, LANDMARK, REWIRE]
            self.edge_type_bias = mx.zeros((4,))
        else:
            self.edge_type_bias = None

        self._strategies = [self._make_strategy(h) for h in range(self.n_heads)]

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

    def _make_strategy(self, head_idx: int):
        if self.strategy == "random":
            return build_strategy(
                "random",
                num_cycles=self.num_cycles,
                seed=self.seed + 7919 * head_idx,
            )
        if self.strategy == "greedy":
            return build_strategy("greedy", num_cycles=self.num_cycles)
        if self.strategy == "online_insertion":
            return build_strategy("online_insertion", seed=self.seed + 7919 * head_idx)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    @property
    def _cache_mode(self) -> str:
        """'static' for input-independent strategies, 'dynamic' otherwise."""
        return "static" if self.strategy == "random" else "dynamic"

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

        return _GraphCache(
            mlx_graph=mlx_graph,
            numpy_abi=numpy_abi,
            safe_idx=s_idx,
            causal_mask=c_mask,
            perm_mx=perm_mx_list,
            inv_perm=inv_perm_list,
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

        r_np: np.ndarray | None = None
        if self.strategy in {"greedy", "online_insertion"}:
            r = self.Wr(x[0]).reshape(T, self.n_heads, self.routing_dim).transpose(1, 0, 2)
            mx.eval(r)
            r_np = np.asarray(r, dtype=np.float32)

        head_abis: list[WayfinderGraphABI] = []
        for h in range(self.n_heads):
            r_h_torch = None
            if r_np is not None:
                r_h_torch = torch.from_numpy(r_np[h])

            abi_h = self._strategies[h].build(
                T=T,
                r=r_h_torch,
                head_idx=h,
                window=self.window,
                landmark_stride=self.landmark_stride,
                include_self=True,
            )
            head_abis.append(abi_h)

        abi = stack_head_abis(head_abis)
        validate_graph_abi(
            abi, expect_heads=self.n_heads, expect_tokens=T, enforce_hamiltonian=True
        )
        self.last_graph_abi = abi
        return to_mlx_graph_abi(abi, heads=self.n_heads, validate=False), abi

    def __call__(
        self, x: mx.array, *, return_debug: bool = False
    ) -> mx.array | tuple[mx.array, Dict[str, Any]]:
        t_total0 = _now_ms()
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        t_graph0 = _now_ms()
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
            y_h, w, permute_ms, _attn_inner = wayfinder_permute_window_attention(
                q,
                k,
                v,
                graph,
                window=self.window,
                return_weights=return_debug,
                cache=cache,
                edge_type_bias_scalar=etb_scalar,
                window_drop_prob=effective_window_drop if is_training else 0.0,
                training=is_training,
            )
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
