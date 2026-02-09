from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from hcsa.graph.abi import WayfinderGraphABI, graph_metrics
from hcsa.mlx.attention import (
    AttentionProfile,
    sparse_gather_attention,
    wayfinder_permute_window_attention_active_batched,
    wayfinder_permute_window_attention_batched,
)
from hcsa.mlx.graph_abi import causal_neighbor_mask
from hcsa.mlx.kernels.metal import (
    has_discovered_active_row_kernel,
    has_discovered_permute_window_kernel,
)

from .qwen_mlx import (
    _QWEN_GRAPH_CACHE_STORE,
    _QwenGraphRuntime,
    _edge_utilization_proxy,
    _repeat_kv_to_q_heads,
    _schedule_bias_to_vec,
)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _pad_value_dim(values: mx.array, target_dim: int) -> mx.array:
    """Pad MLA latent values to match q/k dim expected by Wayfinder kernels."""
    dv = int(values.shape[-1])
    if dv == target_dim:
        return values
    if dv > target_dim:
        raise ValueError(f"value dim {dv} cannot exceed target q/k dim {target_dim}")
    pad = target_dim - dv
    zeros = mx.zeros((*values.shape[:-1], pad), dtype=values.dtype)
    return mx.concatenate([values, zeros], axis=-1)


@dataclass
class GLMWayfinderConfig:
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
    permute_head_chunk_size: int = 2
    query_chunk_size: int = 192
    permute_prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto"
    permute_log_chunks: bool = False
    compute_edge_utilization_proxy: bool = True
    compute_graph_metrics: bool = True
    retro_backfill_enabled: bool = False
    retro_backfill_alpha: float = 0.0
    retro_backfill_training_only: bool = True
    retro_backfill_causal_only: bool = True
    permute_memory_budget_bytes: Optional[int] = None
    active_dense_threshold: Optional[int] = None
    use_discovered_active_row_kernel: bool = True
    circular: bool = False
    multi_cycle_mode: str = "average"
    verify_spectral_gap: bool = False
    spectral_gap_threshold: float = 4.0


def extract_qkv_from_glm_attention(
    attn: nn.Module,
    x: mx.array,
    *,
    cache: Optional[Any] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Extract GLM-4.7-Flash MLA tensors after RoPE.

    Returns:
        queries: [B, Hq, Tq, Dqk]
        keys:    [B, Hkv=1, Tk, Dqk]
        values:  [B, Hkv=1, Tk, Dv_latent]
    """
    B, L, _D = x.shape

    if getattr(attn, "q_lora_rank", None) is None:
        q = attn.q_proj(x)
    else:
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x)))

    q = q.reshape(B, L, attn.num_heads, attn.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [attn.qk_nope_head_dim], axis=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [attn.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = attn.kv_a_layernorm(compressed_kv)

    offset = cache.offset if cache is not None else 0
    q_pe = attn.rope(q_pe, offset)
    k_pe = attn.rope(k_pe, offset)

    kv_latent = mx.expand_dims(kv_latent, axis=1)
    q_nope = attn.embed_q(q_nope)
    keys = mx.concatenate([kv_latent, k_pe], axis=-1)

    if cache is not None:
        keys, _ = cache.update_and_fetch(keys, mx.zeros((B, 1, L, 0)))

    values = keys[..., : -attn.qk_rope_head_dim]
    queries = mx.concatenate([q_nope, q_pe], axis=-1)
    return queries, keys, values


class GLMWayfinderAttention(nn.Module):
    """GLM MLA attention module with HCSA sparse/permute backend."""

    def __init__(self, base_attn: nn.Module, cfg: GLMWayfinderConfig):
        super().__init__()

        self.num_heads = int(base_attn.num_heads)
        self.n_heads = self.num_heads
        self.n_kv_heads = 1  # MLA compressed KV cache is shared (MQA-style).
        self.scale = float(base_attn.scale)
        self.qk_rope_head_dim = int(base_attn.qk_rope_head_dim)
        self.kv_lora_rank = int(base_attn.kv_lora_rank)
        self.qk_nope_head_dim = int(base_attn.qk_nope_head_dim)
        self.q_head_dim = int(base_attn.q_head_dim)
        self.qk_dim = int(self.kv_lora_rank + self.qk_rope_head_dim)
        self.value_dim = int(self.kv_lora_rank)

        self.q_lora_rank = getattr(base_attn, "q_lora_rank", None)
        if self.q_lora_rank is None:
            self.q_proj = base_attn.q_proj
        else:
            self.q_a_proj = base_attn.q_a_proj
            self.q_a_layernorm = base_attn.q_a_layernorm
            self.q_b_proj = base_attn.q_b_proj

        self.kv_a_proj_with_mqa = base_attn.kv_a_proj_with_mqa
        self.kv_a_layernorm = base_attn.kv_a_layernorm
        self.embed_q = base_attn.embed_q
        self.unembed_out = base_attn.unembed_out
        self.o_proj = base_attn.o_proj
        self.rope = base_attn.rope

        self.path = cfg.path
        self.permute_head_chunk_size = int(max(1, cfg.permute_head_chunk_size))
        self.query_chunk_size = int(max(1, cfg.query_chunk_size))
        self.permute_prepermute_mode = str(cfg.permute_prepermute_mode).lower()
        self.permute_log_chunks = bool(cfg.permute_log_chunks)
        self.permute_memory_budget_bytes = (
            None
            if cfg.permute_memory_budget_bytes is None
            else int(max(0, cfg.permute_memory_budget_bytes))
        )
        self.active_dense_threshold = (
            None
            if cfg.active_dense_threshold is None
            else int(max(0, cfg.active_dense_threshold))
        )
        self.use_discovered_active_row_kernel = bool(cfg.use_discovered_active_row_kernel)
        self.retro_backfill_enabled = bool(cfg.retro_backfill_enabled)
        self.retro_backfill_alpha = float(cfg.retro_backfill_alpha)
        self.retro_backfill_training_only = bool(cfg.retro_backfill_training_only)
        self.retro_backfill_causal_only = bool(cfg.retro_backfill_causal_only)
        self.circular = bool(cfg.circular)
        self.multi_cycle_mode = str(cfg.multi_cycle_mode)
        self.window_drop_prob = float(max(0.0, min(1.0, cfg.window_drop)))
        self.edge_type_bias = mx.zeros((4,)) if cfg.edge_bias else None
        self.graph_runtime = _QwenGraphRuntime(
            n_heads=self.n_heads,
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
        )

        self._runtime_window_drop_override: Optional[float] = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)
        self._runtime_memory_budget_bytes: Optional[int] = None
        self._active_graph_seq_len: int = 0

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

    def _effective_permute_chunking(self, T: int) -> tuple[int, int]:
        h_chunk = int(max(1, self.permute_head_chunk_size))
        q_chunk = int(max(1, self.query_chunk_size))
        if T >= 8192:
            h_chunk = min(h_chunk, 2)
            q_chunk = min(q_chunk, 192)
        elif T >= 4096:
            h_chunk = min(h_chunk, 2)
            q_chunk = min(q_chunk, 256)
        return h_chunk, q_chunk

    def _adaptive_graph_seq_len(self, *, k_len: int, q_len: int, cache: Optional[Any]) -> int:
        target = int(k_len)
        max_size_raw = None if cache is None else getattr(cache, "max_size", None)
        max_size = None
        if max_size_raw is not None:
            try:
                max_size = int(max_size_raw)
            except Exception:
                max_size = None
        if max_size is not None and max_size > 0:
            target = max(target, max_size)
        else:
            # Adaptive horizon: amortize graph builds without assuming fixed final length.
            step = max(4096, int(max(1, q_len)) * 8)
            target = ((target + step - 1) // step) * step

        if self._active_graph_seq_len >= target:
            return int(self._active_graph_seq_len)
        self._active_graph_seq_len = int(target)
        return int(self._active_graph_seq_len)

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
        self._runtime_schedule_bias_vec = _schedule_bias_to_vec(schedule_bias)
        self._runtime_memory_budget_bytes = (
            None if memory_budget_bytes is None else int(max(0, memory_budget_bytes))
        )

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)
        self._runtime_memory_budget_bytes = None

    def cache_persistent_bytes(self) -> int:
        cache = _QWEN_GRAPH_CACHE_STORE.get(id(self))
        return int(cache.persistent_bytes) if cache is not None else 0

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
        y = self.unembed_out(y)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        t_total0 = _now_ms()
        queries, keys, values = extract_qkv_from_glm_attention(self, x, cache=cache)
        q_len = int(queries.shape[2])
        k_len = int(keys.shape[2])
        active_mode = self.path == "permute" and cache is not None and q_len < k_len
        discovered_permute_available = (
            self.path == "permute" and has_discovered_permute_window_kernel()
        )
        discovered_active_available = (
            active_mode
            and self.use_discovered_active_row_kernel
            and has_discovered_active_row_kernel()
        )
        force_dense_active = (
            active_mode
            and self.active_dense_threshold is not None
            and k_len <= self.active_dense_threshold
            and (not discovered_active_available)
        )
        use_active_permute = active_mode and not force_dense_active

        # During incremental decode, Q length != K length. Keep dense path for correctness.
        if force_dense_active or (q_len != k_len and not use_active_permute):
            t_attn0 = _now_ms()
            out = self._dense_fallback(queries, keys, values, mask, cache)
            attn_ms = _now_ms() - t_attn0
            self.last_profile = AttentionProfile(
                graph_build_ms=0.0,
                permute_ms=0.0,
                attention_ms=float(attn_ms),
                total_ms=_now_ms() - t_total0,
                path=f"{self.path}_dense_fallback",
                notes={
                    "cache_hit": True,
                    "cache_source": "dense_fallback",
                    "k_len": int(k_len),
                    "active_dense_threshold": self.active_dense_threshold,
                    "active_dense_triggered": bool(force_dense_active),
                    "discovered_permute_available": bool(discovered_permute_available),
                },
            )
            return out

        T = k_len
        graph_T = (
            self._adaptive_graph_seq_len(k_len=k_len, q_len=q_len, cache=cache)
            if use_active_permute
            else T
        )
        t_graph0 = _now_ms()
        if self.permute_log_chunks:
            print(
                f"    glm_wayfinder: requesting graph cache for T={graph_T} (seq_len={T})",
                flush=True,
            )
        graph_cache, cache_hit = self.graph_runtime.get_or_build_cache(id(self), graph_T)
        graph_ms = _now_ms() - t_graph0
        if self.permute_log_chunks:
            print(
                "    glm_wayfinder: graph cache ready "
                f"(hit={bool(cache_hit)}, source={graph_cache.source}, {graph_ms:.1f} ms)",
                flush=True,
            )

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

        values_wf = _pad_value_dim(values, target_dim=int(queries.shape[-1]))

        t_attn0 = _now_ms()
        if self.path == "sparse":
            keys_q = _repeat_kv_to_q_heads(keys, self.n_heads)
            values_q = _repeat_kv_to_q_heads(values_wf, self.n_heads)
            y_h, _w = sparse_gather_attention(
                queries,
                keys_q,
                values_q,
                graph_cache.mlx_graph,
                return_weights=False,
                precomputed_safe_idx=graph_cache.safe_idx,
                precomputed_causal_mask=graph_cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                edge_type_bias_offset=scheduled_edge_bias,
                window_drop_mask=wd_mask,
            )
            keep_mask = graph_cache.causal_mask if wd_mask is None else (graph_cache.causal_mask & wd_mask)
            h_chunk_eff = None
            q_chunk_eff = None
        elif self.path == "permute":
            h_chunk_eff, q_chunk_eff = self._effective_permute_chunking(T)
            memory_budget_bytes = (
                self._runtime_memory_budget_bytes
                if self._runtime_memory_budget_bytes is not None
                else self.permute_memory_budget_bytes
            )
            if use_active_permute:
                active_start = T - q_len
                active_positions = mx.arange(active_start, T, dtype=mx.int32)
                try:
                    y_h, _w = wayfinder_permute_window_attention_active_batched(
                        queries,
                        keys,
                        values_wf,
                        all_perms=graph_cache.perm_mx_stacked,
                        all_inv_perms=graph_cache.inv_perm_stacked,
                        query_positions=active_positions,
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
                        log_progress=self.permute_log_chunks,
                    )
                except Exception as exc:
                    out = self._dense_fallback(queries, keys, values, mask, cache)
                    attn_ms = _now_ms() - t_attn0
                    self.last_profile = AttentionProfile(
                        graph_build_ms=float(graph_ms),
                        permute_ms=0.0,
                        attention_ms=float(attn_ms),
                        total_ms=float(_now_ms() - t_total0),
                        path="permute_active_error_dense_fallback",
                        notes={
                            "seq_len": int(T),
                            "q_len": int(q_len),
                            "graph_seq_len": int(graph_T),
                            "cache_hit": bool(cache_hit),
                            "active_query_mode": True,
                            "active_dense_threshold": self.active_dense_threshold,
                            "active_dense_triggered": bool(force_dense_active),
                            "discovered_active_available": bool(discovered_active_available),
                            "discovered_permute_available": bool(discovered_permute_available),
                            "fallback_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    return out
            else:
                y_h, _w = wayfinder_permute_window_attention_batched(
                    queries,
                    keys,
                    values_wf,
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
                )
            if self.compute_edge_utilization_proxy:
                neigh_idx = graph_cache.mlx_graph.neigh_idx
                if int(neigh_idx.shape[1]) != T:
                    neigh_idx = neigh_idx[:, :T, :]
                keep_mask = causal_neighbor_mask(neigh_idx, T)
            else:
                keep_mask = graph_cache.causal_mask
                if int(keep_mask.shape[1]) != T:
                    keep_mask = keep_mask[:, :T, :]
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = _now_ms() - t_attn0

        if self.compute_edge_utilization_proxy:
            edge_type = graph_cache.mlx_graph.edge_type
            if int(edge_type.shape[1]) != int(keep_mask.shape[1]):
                edge_type = edge_type[:, : int(keep_mask.shape[1]), :]
            self.last_edge_utilization_proxy = _edge_utilization_proxy(
                edge_type,
                keep_mask,
            )

        # Wayfinder kernels use padded value dim for MLA compatibility. Slice back
        # to kv_lora_rank before GLM's unembed_out projection.
        y_latent = y_h[..., : self.value_dim]
        y_proj = self.unembed_out(y_latent)
        out = self.o_proj(
            y_proj.transpose(0, 2, 1, 3).reshape(y_proj.shape[0], y_proj.shape[2], -1)
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
                "graph_seq_len": int(graph_T),
                "q_len": int(q_len),
                "max_degree": int(graph_cache.mlx_graph.neigh_idx.shape[-1]),
                "cache_hit": bool(cache_hit),
                "cache_mode": self.graph_runtime.cache_mode,
                "cache_source": graph_cache.source,
                "cache_persistent_bytes": int(graph_cache.persistent_bytes),
                "window_drop_effective": float(effective_window_drop),
                "permute_head_chunk_effective": int(h_chunk_eff) if self.path == "permute" else None,
                "permute_query_chunk_effective": int(q_chunk_eff) if self.path == "permute" else None,
                "active_query_mode": bool(use_active_permute),
                "active_dense_threshold": self.active_dense_threshold,
                "active_dense_triggered": bool(force_dense_active),
                "discovered_active_available": bool(discovered_active_available),
                "discovered_permute_available": bool(discovered_permute_available),
                "adaptive_graph_reuse": bool(use_active_permute and graph_T != T),
                "mla_qk_dim": int(self.qk_dim),
                "mla_value_dim": int(self.value_dim),
            },
        )
        return out


def _iter_model_layers(model: nn.Module) -> Sequence[nn.Module]:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Model has no .layers or .model.layers attribute; expected a mlx_lm GLM model.")


def swap_glm_attention_with_wayfinder(
    model: nn.Module,
    *,
    cfg: GLMWayfinderConfig,
    layer_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Replace GLM attention blocks with HCSA-backed attention modules."""
    replaced: list[int] = []
    selected = None if layer_indices is None else set(layer_indices)

    for i, layer in enumerate(_iter_model_layers(model)):
        if selected is not None and i not in selected:
            continue
        base_attn = getattr(layer, "self_attn", None)
        if base_attn is None:
            continue
        layer.self_attn = GLMWayfinderAttention(base_attn, cfg)
        replaced.append(i)
    return replaced
