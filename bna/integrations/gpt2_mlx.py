from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from bna.graph.abi import WayfinderGraphABI, graph_metrics
from bna.mlx.attention import (
    AttentionProfile,
    butterfly_permute_window_attention_batched,
    sparse_gather_attention,
)
from bna.mlx.graph_abi import causal_neighbor_mask

from .qwen_mlx import _QwenGraphRuntime, _edge_utilization_proxy


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class GPT2ButterflyConfig:
    path: Literal["sparse", "permute"] = "permute"
    strategy: Literal["random", "greedy", "online_insertion", "regular_partition"] = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int | str = 1
    edge_disjoint: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    edge_bias: bool = True
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None
    permute_head_chunk_size: int = 8
    query_chunk_size: int = 256
    permute_prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto"
    permute_memory_budget_bytes: Optional[int] = None
    permute_log_chunks: bool = False
    compute_edge_utilization_proxy: bool = True
    compute_graph_metrics: bool = True
    retro_backfill_enabled: bool = False
    retro_backfill_alpha: float = 0.0
    retro_backfill_training_only: bool = True
    retro_backfill_causal_only: bool = True
    circular: bool = False
    multi_cycle_mode: str = "average"
    verify_spectral_gap: bool = False
    spectral_gap_threshold: float = 4.0
    use_fused_dispatch: bool = True
    enforce_hamiltonian: bool = True


def extract_qkv_from_gpt2_attention(
    attn: nn.Module,
    x: mx.array,
    *,
    cache: Optional[Any] = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Extract GPT-2 Q/K/V in [B, H, T, dh]."""
    B, L, _D = x.shape
    if hasattr(attn, "n_head"):
        n_head = int(attn.n_head)
    elif hasattr(attn, "n_heads"):
        n_head = int(attn.n_heads)
    else:
        raise AttributeError("GPT-2 attention module must provide n_head or n_heads")
    qkv = attn.c_attn(x)
    queries, keys, values = mx.split(qkv, 3, axis=-1)
    queries = queries.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    return queries, keys, values


class GPT2ButterflyAttention(nn.Module):
    """GPT-2 attention module with Butterfly sparse/permute backends."""

    def __init__(self, base_attn: nn.Module, cfg: GPT2ButterflyConfig):
        super().__init__()

        self.n_head = int(base_attn.n_head)
        self.n_embd = int(base_attn.n_embd)
        self.n_heads = int(base_attn.n_head)
        self.n_kv_heads = int(base_attn.n_head)
        self.scale = float(base_attn.scale)
        self.head_dim = int(base_attn.n_embd // base_attn.n_head)

        self.c_attn = base_attn.c_attn
        self.c_proj = base_attn.c_proj

        self.path = cfg.path
        self.permute_head_chunk_size = int(max(1, cfg.permute_head_chunk_size))
        self.query_chunk_size = int(max(1, cfg.query_chunk_size))
        self.permute_prepermute_mode = str(cfg.permute_prepermute_mode).lower()
        self.permute_memory_budget_bytes = (
            None
            if cfg.permute_memory_budget_bytes is None
            else int(max(0, cfg.permute_memory_budget_bytes))
        )
        self.permute_log_chunks = bool(cfg.permute_log_chunks)
        self.retro_backfill_enabled = bool(cfg.retro_backfill_enabled)
        self.retro_backfill_alpha = float(cfg.retro_backfill_alpha)
        self.retro_backfill_training_only = bool(cfg.retro_backfill_training_only)
        self.retro_backfill_causal_only = bool(cfg.retro_backfill_causal_only)
        self.circular = bool(cfg.circular)
        self.multi_cycle_mode = str(cfg.multi_cycle_mode)
        self.use_fused_dispatch = bool(cfg.use_fused_dispatch)
        self.window_drop_prob = float(max(0.0, min(1.0, cfg.window_drop)))
        self.edge_type_bias = mx.zeros((4,)) if cfg.edge_bias else None
        self.graph_runtime = _QwenGraphRuntime(
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            window=cfg.window,
            landmark_stride=cfg.landmark_stride,
            strategy=cfg.strategy,
            num_cycles=cfg.num_cycles,
            edge_disjoint=cfg.edge_disjoint,
            regular_num_clusters=cfg.regular_num_clusters,
            seed=cfg.seed,
            path=cfg.path,
            compiled_graph_dir=cfg.compiled_graph_dir,
            verify_spectral_gap=cfg.verify_spectral_gap,
            spectral_gap_threshold=cfg.spectral_gap_threshold,
            store_numpy_abi=bool(cfg.compute_graph_metrics),
            store_graph_tensors=bool(
                cfg.path == "sparse"
                or cfg.compute_edge_utilization_proxy
                or cfg.compute_graph_metrics
            ),
            enforce_hamiltonian=cfg.enforce_hamiltonian,
        )

        self._runtime_window_drop_override: Optional[float] = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)
        self._runtime_memory_budget_bytes: Optional[int] = None

        self.last_profile: AttentionProfile = AttentionProfile(path=cfg.path)
        self.last_graph_abi: Optional[WayfinderGraphABI] = None
        self.last_graph_metrics: Dict[str, Any] = {}
        self.last_edge_utilization_proxy: Dict[str, float] = {
            "cycle": 0.0,
            "window": 0.0,
            "landmark": 0.0,
            "rewire": 0.0,
        }
        self.compute_edge_utilization_proxy: bool = bool(cfg.compute_edge_utilization_proxy)
        self.compute_graph_metrics: bool = bool(cfg.compute_graph_metrics)

    def set_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
        memory_budget_bytes: Optional[int] = None,
    ) -> None:
        self._runtime_window_drop_override = (
            None if window_drop is None else float(min(1.0, max(0.0, window_drop)))
        )
        vec = np.zeros((4,), dtype=np.float32)
        if schedule_bias is not None:
            mapping = {"cycle": 0, "window": 1, "landmark": 2, "rewire": 3}
            for k, v in schedule_bias.items():
                idx = mapping.get(str(k).lower())
                if idx is not None:
                    vec[idx] = float(v)
        self._runtime_schedule_bias_vec = vec
        self._runtime_memory_budget_bytes = (
            None if memory_budget_bytes is None else int(max(0, memory_budget_bytes))
        )

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)
        self._runtime_memory_budget_bytes = None

    def cache_persistent_bytes(self) -> int:
        # Runtime cache is owned by _QwenGraphRuntime's shared store and keyed by module id.
        # Read from profile notes after a forward if needed.
        return int(self.last_profile.notes.get("cache_persistent_bytes", 0))

    def _dense_fallback(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        y = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self.c_proj(y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        t_total0 = _now_ms()
        queries, keys, values = extract_qkv_from_gpt2_attention(self, x, cache=cache)

        # During incremental decode, Q length != K length. Keep dense path for correctness.
        if queries.shape[2] != keys.shape[2]:
            out = self._dense_fallback(queries, keys, values, mask, cache)
            self.last_profile = AttentionProfile(
                graph_build_ms=0.0,
                permute_ms=0.0,
                attention_ms=0.0,
                total_ms=_now_ms() - t_total0,
                path=f"{self.path}_dense_fallback",
                notes={"cache_hit": True, "cache_source": "dense_fallback"},
            )
            return out

        T = int(keys.shape[2])
        t_graph0 = _now_ms()
        graph_cache, cache_hit = self.graph_runtime.get_or_build_cache(id(self), T)
        graph_ms = _now_ms() - t_graph0

        if graph_cache.numpy_abi is not None and self.last_graph_abi is not graph_cache.numpy_abi:
            self.last_graph_abi = graph_cache.numpy_abi
            if self.compute_graph_metrics:
                self.last_graph_metrics = graph_metrics(graph_cache.numpy_abi)

        is_training = bool(self.training)
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

        wd_mask: Optional[mx.array] = None
        if is_training and effective_window_drop > 0.0 and self.path == "sparse":
            et = graph_cache.mlx_graph.edge_type.astype(mx.int32)
            is_window = et == int(2)
            s_idx = graph_cache.safe_idx
            i_idx = mx.arange(T, dtype=mx.int32).reshape(1, T, 1)
            is_self = s_idx == i_idx
            droppable = is_window & (~is_self)
            drop_rand = mx.random.uniform(shape=et.shape) < effective_window_drop
            wd_mask = ~(droppable & drop_rand)

        etb_scalar: Optional[float] = None
        if self.edge_type_bias is not None and self.path == "permute":
            mx.eval(self.edge_type_bias)
            etb_scalar = float(self.edge_type_bias[0].item())
        if scheduled_edge_bias is not None and self.path == "permute":
            mx.eval(scheduled_edge_bias)
            cycle_bias = float(scheduled_edge_bias[0].item())
            etb_scalar = cycle_bias if etb_scalar is None else (etb_scalar + cycle_bias)

        t_attn0 = _now_ms()
        if self.path == "sparse":
            y_h, _w = sparse_gather_attention(
                queries,
                keys,
                values,
                graph_cache.mlx_graph,
                return_weights=False,
                precomputed_safe_idx=graph_cache.safe_idx,
                precomputed_causal_mask=graph_cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                edge_type_bias_offset=scheduled_edge_bias,
                window_drop_mask=wd_mask,
            )
            keep_mask = graph_cache.causal_mask if wd_mask is None else (graph_cache.causal_mask & wd_mask)
        elif self.path == "permute":
            h_chunk_eff = int(max(1, min(self.permute_head_chunk_size, self.n_heads)))
            q_chunk_eff = int(max(1, min(self.query_chunk_size, T)))
            memory_budget_bytes = (
                self._runtime_memory_budget_bytes
                if self._runtime_memory_budget_bytes is not None
                else self.permute_memory_budget_bytes
            )
            y_h, _w = butterfly_permute_window_attention_batched(
                queries,
                keys,
                values,
                all_perms=graph_cache.perm_mx_stacked,
                all_inv_perms=graph_cache.inv_perm_stacked,
                window=self.graph_runtime.window,
                edge_type_bias_scalar=etb_scalar,
                window_drop_prob=effective_window_drop if is_training else 0.0,
                training=is_training,
                head_chunk_size=h_chunk_eff,
                query_chunk_size=q_chunk_eff,
                prepermute_mode=self.permute_prepermute_mode,  # type: ignore[arg-type]
                memory_budget_bytes=memory_budget_bytes,
                circular=self.circular,
                multi_cycle_mode=self.multi_cycle_mode,
                retro_backfill_enabled=self.retro_backfill_enabled,
                retro_backfill_alpha=self.retro_backfill_alpha,
                retro_backfill_training_only=self.retro_backfill_training_only,
                retro_backfill_causal_only=self.retro_backfill_causal_only,
                log_progress=self.permute_log_chunks,
                use_fused_dispatch=self.use_fused_dispatch,
                scale=self.scale,
            )
            if self.compute_edge_utilization_proxy:
                keep_mask = causal_neighbor_mask(graph_cache.mlx_graph.neigh_idx, T)
            else:
                keep_mask = graph_cache.causal_mask
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = _now_ms() - t_attn0

        if self.compute_edge_utilization_proxy:
            self.last_edge_utilization_proxy = _edge_utilization_proxy(
                graph_cache.mlx_graph.edge_type,
                keep_mask,
            )

        out = self.c_proj(
            y_h.transpose(0, 2, 1, 3).reshape(y_h.shape[0], y_h.shape[2], -1)
        )
        total_ms = _now_ms() - t_total0
        self.last_profile = AttentionProfile(
            graph_build_ms=float(graph_ms),
            permute_ms=0.0,
            attention_ms=float(attn_ms),
            total_ms=float(total_ms),
            path=self.path,
            notes={
                "seq_len": int(T),
                "max_degree": int(graph_cache.mlx_graph.neigh_idx.shape[-1]),
                "cache_hit": bool(cache_hit),
                "cache_mode": self.graph_runtime.cache_mode,
                "cache_source": graph_cache.source,
                "cache_persistent_bytes": int(graph_cache.persistent_bytes),
                "window_drop_effective": float(effective_window_drop),
            },
        )

        return out


def swap_gpt2_attention_with_butterfly(
    model: nn.Module,
    *,
    cfg: GPT2ButterflyConfig,
    layer_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Replace GPT-2 attention blocks with HCSA-backed attention modules."""
    replaced: list[int] = []
    if not hasattr(model, "layers"):
        raise ValueError("Model has no .layers attribute; expected a mlx_lm GPT-2 model.")

    selected = None if layer_indices is None else set(layer_indices)
    for i, layer in enumerate(model.layers):
        if selected is not None and i not in selected:
            continue
        base_attn = getattr(layer, "attn", None)
        if base_attn is None:
            continue
        layer.attn = GPT2ButterflyAttention(base_attn, cfg)
        replaced.append(i)
    return replaced


GPT2WayfinderConfig = GPT2ButterflyConfig
GPT2WayfinderAttention = GPT2ButterflyAttention
swap_gpt2_attention_with_wayfinder = swap_gpt2_attention_with_butterfly

__all__ = [
    "GPT2ButterflyConfig",
    "GPT2ButterflyAttention",
    "GPT2WayfinderConfig",
    "GPT2WayfinderAttention",
    "extract_qkv_from_gpt2_attention",
    "swap_gpt2_attention_with_butterfly",
    "swap_gpt2_attention_with_wayfinder",
]
