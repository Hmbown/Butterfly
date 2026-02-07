from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from hcsa.graph.abi import WayfinderGraphABI, graph_metrics, stack_head_abis, validate_graph_abi
from hcsa.graph_strategies import build_strategy
from hcsa.mlx.attention import (
    AttentionProfile,
    sparse_gather_attention,
    wayfinder_permute_window_attention_batched,
)
from hcsa.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)


_NEG_EPS = mx.array(1e-9, dtype=mx.float32)
_QWEN_GRAPH_CACHE_STORE: Dict[int, "_QwenGraphCache"] = {}


@dataclass
class QwenHHAConfig:
    path: Literal["sparse", "permute"] = "permute"
    strategy: Literal["random", "greedy", "online_insertion"] = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int = 1
    seed: int = 0
    edge_bias: bool = True
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None


@dataclass(frozen=True)
class _QwenGraphCache:
    mlx_graph: MLXGraphABI
    numpy_abi: WayfinderGraphABI
    safe_idx: mx.array
    causal_mask: mx.array
    perm_mx: List[mx.array]
    inv_perm: List[mx.array]
    pi_idx_clamped: List[mx.array]
    valid_mask: List[mx.array]
    causal_masks: List[mx.array]
    # Stacked tensors for vectorized batched permute path
    perm_mx_stacked: mx.array  # [H, T]
    inv_perm_stacked: mx.array  # [H, T]
    pi_idx_stacked: mx.array  # [H, T, W]
    valid_mask_stacked: mx.array  # [H, T, W]
    causal_mask_stacked: mx.array  # [H, T, W]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: str | None = None
    persistent_bytes: int = 0


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
        if idx is not None:
            vec[idx] = float(v)
    return vec


def extract_qkv_from_qwen_attention(
    attn: nn.Module,
    x: mx.array,
    *,
    cache: Optional[Any] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Extract Q/K/V tensors from a Qwen attention module after RoPE.

    Returns:
        queries: [B, Hq, Tq, Dh]
        keys:    [B, Hk, Tk, Dh]
        values:  [B, Hk, Tk, Dh]
    """
    B, L, _D = x.shape
    queries = attn.q_proj(x)
    keys = attn.k_proj(x)
    values = attn.v_proj(x)

    queries = attn.q_norm(queries.reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = attn.k_norm(keys.reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    return queries, keys, values


class _QwenGraphRuntime:
    """Minimal graph/cache runtime for Qwen attention swap."""

    def __init__(
        self,
        *,
        n_heads: int,
        window: int,
        landmark_stride: Optional[int],
        strategy: str,
        num_cycles: int,
        seed: int,
        path: str,
        compiled_graph_dir: Optional[str],
    ):
        self.n_heads = int(n_heads)
        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.strategy = strategy
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self.path = str(path)
        self.compiled_graph_dir = compiled_graph_dir
        self._strategies = [self._make_strategy(h) for h in range(self.n_heads)]

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
    def cache_mode(self) -> str:
        return "static" if self.strategy == "random" else "dynamic"

    def cache_key(self, T: int) -> tuple:
        return (
            int(T),
            self.strategy,
            self.num_cycles,
            self.window,
            self.landmark_stride,
            self.seed,
            self.path,
            str(Path(self.compiled_graph_dir).resolve()) if self.compiled_graph_dir else None,
        )

    def _build_graph_abi(self, T: int) -> Tuple[MLXGraphABI, WayfinderGraphABI]:
        if self.strategy != "random":
            raise ValueError(
                "Qwen full-swap currently supports strategy='random' only for deterministic "
                "input-independent caching."
            )

        head_abis: list[WayfinderGraphABI] = []
        for h in range(self.n_heads):
            abi_h = self._strategies[h].build(
                T=T,
                r=None,
                head_idx=h,
                window=self.window,
                landmark_stride=self.landmark_stride,
                include_self=True,
            )
            head_abis.append(abi_h)

        abi = stack_head_abis(head_abis)
        validate_graph_abi(
            abi,
            expect_heads=self.n_heads,
            expect_tokens=T,
            enforce_hamiltonian=True,
        )
        mlx_graph = to_mlx_graph_abi(abi, heads=self.n_heads, validate=False)
        return mlx_graph, abi

    def _build_cache(
        self,
        mlx_graph: MLXGraphABI,
        numpy_abi: WayfinderGraphABI,
        T: int,
        *,
        cache_key: tuple,
        source: str = "runtime",
        artifact_dir: str | None = None,
    ) -> _QwenGraphCache:
        s_idx = safe_neighbor_idx(mlx_graph.neigh_idx, T)
        c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, T)

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

            if perm is None:
                p_mx = mx.zeros((T,), dtype=mx.int32)
                ip = mx.zeros((T,), dtype=mx.int32)
                pi_clamped = mx.zeros((T, W), dtype=mx.int32)
                valid = mx.zeros((T, W), dtype=mx.bool_)
                causal_h = mx.zeros((T, W), dtype=mx.bool_)
            else:
                perm_arr = np.asarray(perm, dtype=np.int32)
                p_mx = mx.array(perm_arr, dtype=mx.int32)
                ip = mx.argsort(p_mx)
                pi_idx = mx.arange(T, dtype=mx.int32).reshape(T, 1) + offsets.reshape(1, W)
                valid = (pi_idx >= 0) & (pi_idx < T)
                pi_clamped = mx.clip(pi_idx, 0, T - 1)
                orig_idx = p_mx
                neigh_orig = orig_idx[pi_clamped]
                query_orig = orig_idx.reshape(T, 1)
                causal_h = neigh_orig <= query_orig

            perm_mx_list.append(p_mx)
            inv_perm_list.append(ip)
            pi_idx_clamped_list.append(pi_clamped)
            valid_mask_list.append(valid)
            causal_masks_list.append(causal_h)

        # Stack per-head lists into [H, ...] tensors for vectorized path
        perm_stacked = mx.stack(perm_mx_list, axis=0)  # [H, T]
        inv_stacked = mx.stack(inv_perm_list, axis=0)  # [H, T]
        pi_stacked = mx.stack(pi_idx_clamped_list, axis=0)  # [H, T, W]
        valid_stacked = mx.stack(valid_mask_list, axis=0)  # [H, T, W]
        causal_stacked = mx.stack(causal_masks_list, axis=0)  # [H, T, W]

        persistent_bytes = _mx_nbytes(mlx_graph.neigh_idx) + _mx_nbytes(mlx_graph.edge_type)
        persistent_bytes += _mx_nbytes(s_idx) + _mx_nbytes(c_mask)
        persistent_bytes += _mx_nbytes(perm_stacked) + _mx_nbytes(inv_stacked)
        persistent_bytes += _mx_nbytes(pi_stacked) + _mx_nbytes(valid_stacked)
        persistent_bytes += _mx_nbytes(causal_stacked)

        return _QwenGraphCache(
            mlx_graph=mlx_graph,
            numpy_abi=numpy_abi,
            safe_idx=s_idx,
            causal_mask=c_mask,
            perm_mx=perm_mx_list,
            inv_perm=inv_perm_list,
            pi_idx_clamped=pi_idx_clamped_list,
            valid_mask=valid_mask_list,
            causal_masks=causal_masks_list,
            perm_mx_stacked=perm_stacked,
            inv_perm_stacked=inv_stacked,
            pi_idx_stacked=pi_stacked,
            valid_mask_stacked=valid_stacked,
            causal_mask_stacked=causal_stacked,
            cache_key=cache_key,
            source=source,
            artifact_dir=artifact_dir,
            persistent_bytes=int(persistent_bytes),
        )

    def _load_compiled_cache(self, T: int, cache_key: tuple) -> _QwenGraphCache | None:
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
        if int(neigh_idx.shape[0]) != self.n_heads or int(neigh_idx.shape[1]) != int(T):
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
        mlx_graph = to_mlx_graph_abi(numpy_abi, heads=self.n_heads, validate=False)
        return self._build_cache(
            mlx_graph,
            numpy_abi,
            T,
            cache_key=cache_key,
            source="compiled",
            artifact_dir=str(art_dir),
        )

    def get_or_build_cache(self, owner_id: int, T: int) -> tuple[_QwenGraphCache, bool]:
        key = self.cache_key(T)
        existing = _QWEN_GRAPH_CACHE_STORE.get(owner_id)
        if self.cache_mode == "static" and existing is not None and existing.cache_key == key:
            return existing, True

        compiled = self._load_compiled_cache(T, key)
        if compiled is not None:
            if self.cache_mode == "static":
                _QWEN_GRAPH_CACHE_STORE[owner_id] = compiled
            return compiled, False

        mlx_graph, numpy_abi = self._build_graph_abi(T)
        built = self._build_cache(mlx_graph, numpy_abi, T, cache_key=key, source="runtime")
        if self.cache_mode == "static":
            _QWEN_GRAPH_CACHE_STORE[owner_id] = built
        return built, False


def _repeat_kv_to_q_heads(x: mx.array, n_q_heads: int) -> mx.array:
    B, n_kv, T, dh = x.shape
    if n_kv == n_q_heads:
        return x
    if n_q_heads % n_kv != 0:
        raise ValueError(f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv}")
    repeats = n_q_heads // n_kv
    x = mx.broadcast_to(x[:, :, None, :, :], (B, n_kv, repeats, T, dh))
    return x.reshape(B, n_q_heads, T, dh)


def _edge_utilization_proxy(
    edge_type: mx.array,
    keep_mask: mx.array,
) -> Dict[str, float]:
    """Edge-use proxy from available causal edges (not weighted by attention)."""
    keep = keep_mask.astype(mx.float32)
    total = mx.maximum(mx.sum(keep), _NEG_EPS)

    # Vectorized: compute all 4 fractions in one eval
    codes = mx.array([1, 2, 3, 4], dtype=mx.int32)  # CYCLE, WINDOW, LANDMARK, REWIRE
    # edge_type [...] == codes[i] for each code — broadcast [*shape, 1] vs [4]
    et_flat = edge_type.reshape(-1)[:, None]  # [N, 1]
    keep_flat = keep.reshape(-1)[:, None]  # [N, 1]
    matches = (et_flat == codes[None, :]).astype(mx.float32)  # [N, 4]
    fracs = mx.sum(keep_flat * matches, axis=0) / total  # [4]
    mx.eval(fracs)

    return {
        "cycle": float(fracs[0].item()),
        "window": float(fracs[1].item()),
        "landmark": float(fracs[2].item()),
        "rewire": float(fracs[3].item()),
    }


class QwenWayfinderAttention(nn.Module):
    """Qwen attention module with HCSA sparse/permute backend."""

    def __init__(self, base_attn: nn.Module, cfg: QwenHHAConfig):
        super().__init__()

        self.n_heads = int(base_attn.n_heads)
        self.n_kv_heads = int(base_attn.n_kv_heads)
        self.scale = float(base_attn.scale)
        self.head_dim = int(round(self.scale ** -2))

        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.q_norm = base_attn.q_norm
        self.k_norm = base_attn.k_norm
        self.rope = base_attn.rope

        self.path = cfg.path
        self.window_drop_prob = float(max(0.0, min(1.0, cfg.window_drop)))
        self.edge_type_bias = mx.zeros((4,)) if cfg.edge_bias else None
        self.graph_runtime = _QwenGraphRuntime(
            n_heads=self.n_heads,
            window=cfg.window,
            landmark_stride=cfg.landmark_stride,
            strategy=cfg.strategy,
            num_cycles=cfg.num_cycles,
            seed=cfg.seed,
            path=cfg.path,
            compiled_graph_dir=cfg.compiled_graph_dir,
        )

        self._runtime_window_drop_override: Optional[float] = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)

        self.last_profile: AttentionProfile = AttentionProfile(path=cfg.path)
        self.last_graph_abi: Optional[WayfinderGraphABI] = None
        self.last_graph_metrics: Dict[str, Any] = {}
        self.last_edge_utilization_proxy: Dict[str, float] = {
            "cycle": 0.0,
            "window": 0.0,
            "landmark": 0.0,
            "rewire": 0.0,
        }

    def set_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        self._runtime_window_drop_override = (
            None if window_drop is None else float(min(1.0, max(0.0, window_drop)))
        )
        self._runtime_schedule_bias_vec = _schedule_bias_to_vec(schedule_bias)

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec = np.zeros((4,), dtype=np.float32)

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
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        t_total0 = _now_ms()
        queries, keys, values = extract_qkv_from_qwen_attention(self, x, cache=cache)

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

        keys = _repeat_kv_to_q_heads(keys, self.n_heads)
        values = _repeat_kv_to_q_heads(values, self.n_heads)

        T = int(keys.shape[2])
        t_graph0 = _now_ms()
        graph_cache, cache_hit = self.graph_runtime.get_or_build_cache(id(self), T)
        graph_ms = _now_ms() - t_graph0
        self.last_graph_abi = graph_cache.numpy_abi
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
        permute_ms = 0.0
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
            y_h, _w = wayfinder_permute_window_attention_batched(
                queries,
                keys,
                values,
                all_perms=graph_cache.perm_mx_stacked,
                all_inv_perms=graph_cache.inv_perm_stacked,
                all_pi_idx=graph_cache.pi_idx_stacked,
                all_valid=graph_cache.valid_mask_stacked,
                all_causal=graph_cache.causal_mask_stacked,
                edge_type_bias_scalar=etb_scalar,
                window_drop_prob=effective_window_drop if is_training else 0.0,
                training=is_training,
            )
            keep_mask = graph_cache.causal_mask
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = _now_ms() - t_attn0

        self.last_edge_utilization_proxy = _edge_utilization_proxy(
            graph_cache.mlx_graph.edge_type,
            keep_mask,
        )

        out = self.o_proj(
            y_h.transpose(0, 2, 1, 3).reshape(y_h.shape[0], y_h.shape[2], -1)
        )
        total_ms = _now_ms() - t_total0
        self.last_profile = AttentionProfile(
            graph_build_ms=float(graph_ms),
            permute_ms=float(permute_ms),
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


def swap_qwen_attention_with_wayfinder(
    model: nn.Module,
    *,
    cfg: QwenHHAConfig,
    layer_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Replace Qwen attention blocks with HCSA-backed attention modules.

    Returns:
        List of layer indices that were replaced.
    """
    replaced: list[int] = []
    if not hasattr(model, "layers"):
        raise ValueError("Model has no .layers attribute; expected a mlx_lm Qwen model.")

    for i, layer in enumerate(model.layers):
        if layer_indices is not None and i not in set(layer_indices):
            continue
        base_attn = getattr(layer, "self_attn", None)
        if base_attn is None:
            continue
        layer.self_attn = QwenWayfinderAttention(base_attn, cfg)
        replaced.append(i)
    return replaced

