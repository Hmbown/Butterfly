from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from bna.graph.abi import WayfinderGraphABI, graph_metrics, validate_graph_abi
from bna.graph.analysis import expansion_proxy, spectral_gap
from bna.mlx.attention import (
    AttentionProfile,
    sparse_gather_attention_active,
    sparse_gather_attention,
    stable_masked_softmax,
    wayfinder_permute_window_attention_active_batched,
    wayfinder_permute_window_attention_batched,
)
from bna.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)
from bna.topology import Topology


_NEG_EPS = mx.array(1e-9, dtype=mx.float32)
_QWEN_GRAPH_CACHE_STORE: Dict[int, "_QwenGraphCache"] = {}
_QWEN_GRAPH_CACHE_BY_KEY: Dict[tuple, "_QwenGraphCache"] = {}
EXPECTED_QWEN35_FULL_ATTENTION_LAYERS: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31)


@dataclass
class QwenWayfinderConfig:
    path: Literal["sparse", "permute"] = "permute"
    strategy: Literal["random", "greedy", "online_insertion", "regular_partition"] = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int | str = 1
    edge_disjoint: bool = True
    enforce_hamiltonian: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    edge_bias: bool = True
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None
    permute_head_chunk_size: int = 8
    query_chunk_size: int = 256
    permute_stream_o_proj: bool = False
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
    active_dense_threshold: Optional[int | str] = None
    wayfinder_decode_backend: Literal["active_permute", "dense"] = "active_permute"


@dataclass(frozen=True)
class _QwenGraphCache:
    mlx_graph: MLXGraphABI
    numpy_abi: Optional[WayfinderGraphABI]
    safe_idx: mx.array
    causal_mask: mx.array
    perm_mx: List[mx.array]
    inv_perm: List[mx.array]
    # Stacked tensors for vectorized batched permute path
    perm_mx_stacked: mx.array  # [H, T] or [H, d, T]
    inv_perm_stacked: mx.array  # [H, T] or [H, d, T]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: str | None = None
    persistent_bytes: int = 0


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _mx_nbytes(arr: mx.array) -> int:
    mx.eval(arr)
    # Avoid np.asarray(...) here: converting large MLX tensors to NumPy can
    # transiently duplicate graph/cache storage in host memory.
    if hasattr(arr, "nbytes"):
        return int(arr.nbytes)
    itemsize = getattr(arr, "itemsize", None)
    if itemsize is not None:
        elems = int(np.prod([int(d) for d in arr.shape]))
        return int(elems * int(itemsize))
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


def _attn_get_first(attn: nn.Module, names: Sequence[str]) -> Any:
    for name in names:
        value = getattr(attn, name, None)
        if value is not None:
            return value
    return None


def _attn_num_heads(attn: nn.Module) -> int:
    value = _attn_get_first(attn, ("n_heads", "num_attention_heads", "num_heads"))
    if value is None:
        raise AttributeError("Unable to resolve attention query head count.")
    return int(value)


def _attn_num_kv_heads(attn: nn.Module) -> int:
    value = _attn_get_first(attn, ("n_kv_heads", "num_key_value_heads", "num_kv_heads"))
    if value is None:
        raise AttributeError("Unable to resolve attention KV head count.")
    return int(value)


def _attn_scale(attn: nn.Module, *, head_dim: Optional[int] = None) -> float:
    value = _attn_get_first(attn, ("scale", "scaling"))
    if value is not None:
        return float(value)
    if head_dim is not None:
        return float(head_dim ** -0.5)
    raise AttributeError("Unable to resolve attention scaling factor.")


def _attn_head_dim(attn: nn.Module, *, scale: Optional[float] = None) -> int:
    value = getattr(attn, "head_dim", None)
    if value is not None:
        return int(value)
    resolved_scale = _attn_scale(attn) if scale is None else float(scale)
    return int(round(resolved_scale ** -2))


def _attn_rope(attn: nn.Module) -> nn.Module:
    rope = _attn_get_first(attn, ("rope", "rotary_emb"))
    if rope is None:
        raise AttributeError("Unable to resolve attention RoPE module.")
    return rope


def _apply_rope_qk(
    rope: nn.Module,
    queries: mx.array,
    keys: mx.array,
    *,
    cache: Optional[Any],
) -> tuple[mx.array, mx.array]:
    if cache is not None:
        offset = int(getattr(cache, "offset", 0))
        try:
            return rope(queries, offset=offset), rope(keys, offset=offset)
        except TypeError:
            pass
    else:
        try:
            return rope(queries), rope(keys)
        except TypeError:
            pass

    q_len = int(queries.shape[2])
    start = int(getattr(cache, "offset", 0)) if cache is not None else 0
    positions = (mx.arange(q_len, dtype=mx.int32) + start)[None, :]
    try:
        return rope(positions, queries, keys)
    except TypeError:
        return rope(positions, queries), rope(positions, keys)


def extract_qkv_from_qwen_attention(
    attn: nn.Module,
    x: mx.array,
    *,
    cache: Optional[Any] = None,
    return_gate: bool = False,
) -> Tuple[mx.array, mx.array, mx.array] | Tuple[mx.array, mx.array, mx.array, Optional[mx.array]]:
    """Extract Q/K/V tensors from a Qwen attention module after RoPE.

    Returns:
        queries: [B, Hq, Tq, Dh]
        keys:    [B, Hk, Tk, Dh]
        values:  [B, Hk, Tk, Dh]
    """
    B, L, _D = x.shape
    n_heads = _attn_num_heads(attn)
    n_kv_heads = _attn_num_kv_heads(attn)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)

    head_dim = _attn_head_dim(attn, scale=_attn_get_first(attn, ("scale", "scaling")))
    gate: Optional[mx.array] = None

    if hasattr(attn, "qkv_proj") and not all(
        hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")
    ):
        qkv = attn.qkv_proj(x)
        k_dim = int(n_kv_heads * head_dim)
        v_dim = int(n_kv_heads * head_dim)
        q_dim = int(qkv.shape[-1]) - k_dim - v_dim
        if q_dim <= 0:
            raise ValueError(f"Unable to split qkv projection with shape {qkv.shape}.")
        q_part, k_part, v_part = mx.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        queries = q_part.reshape(B, L, n_heads, -1)
        keys = k_part.reshape(B, L, n_kv_heads, -1)
        values = v_part.reshape(B, L, n_kv_heads, -1)
    else:
        q_proj_out = attn.q_proj(x)
        queries = q_proj_out.reshape(B, L, n_heads, -1)
        keys = attn.k_proj(x).reshape(B, L, n_kv_heads, -1)
        values = attn.v_proj(x).reshape(B, L, n_kv_heads, -1)

    q_dim = int(queries.shape[-1])
    norm_dim = None
    if q_norm is not None and hasattr(q_norm, "weight"):
        try:
            norm_dim = int(q_norm.weight.shape[0])
        except Exception:
            norm_dim = None
    expected_q_dim = norm_dim if norm_dim is not None else head_dim
    if q_dim == 2 * int(expected_q_dim):
        queries, gate_heads = mx.split(queries, 2, axis=-1)
        gate = gate_heads.reshape(B, L, -1)

    if q_norm is not None:
        queries = q_norm(queries)
    if k_norm is not None:
        keys = k_norm(keys)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    rope = _attn_rope(attn)
    queries, keys = _apply_rope_qk(rope, queries, keys, cache=cache)

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    if return_gate:
        return queries, keys, values, gate
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
        num_cycles: int | str,
        edge_disjoint: bool,
        regular_num_clusters: int,
        seed: int,
        path: str,
        compiled_graph_dir: Optional[str],
        verify_spectral_gap: bool = False,
        spectral_gap_threshold: float = 4.0,
        store_numpy_abi: bool,
        store_graph_tensors: bool,
        enforce_hamiltonian: bool = True,
    ):
        self.n_heads = int(n_heads)
        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.strategy = strategy
        self._num_cycles_raw = num_cycles
        self.num_cycles = 1 if num_cycles == "auto" else int(num_cycles)
        self.edge_disjoint = bool(edge_disjoint)
        self.regular_num_clusters = int(max(1, regular_num_clusters))
        self.seed = int(seed)
        self.path = str(path)
        self.compiled_graph_dir = compiled_graph_dir
        self.verify_spectral_gap = bool(verify_spectral_gap)
        self.spectral_gap_threshold = float(max(0.0, spectral_gap_threshold))
        self.store_numpy_abi = bool(store_numpy_abi)
        self.store_graph_tensors = bool(store_graph_tensors)
        self.enforce_hamiltonian = bool(enforce_hamiltonian)
        self.topology = Topology(
            n_heads=self.n_heads,
            strategy=self.strategy,
            num_cycles=self.num_cycles,
            edge_disjoint=self.edge_disjoint,
            regular_num_clusters=self.regular_num_clusters,
            seed=self.seed,
            window=self.window,
            landmark_stride=self.landmark_stride,
            enforce_hamiltonian=self.enforce_hamiltonian,
        )

    @property
    def cache_mode(self) -> str:
        return self.topology.cache_mode

    def _resolve_and_sync_num_cycles(self, T: int) -> None:
        """Resolve 'auto' num_cycles at graph-construction time."""
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
                enforce_hamiltonian=self.enforce_hamiltonian,
            )

    def cache_key(self, T: int) -> tuple:
        return (
            int(self.n_heads),
            int(T),
            self.strategy,
            self.num_cycles,
            self.edge_disjoint,
            self.enforce_hamiltonian,
            self.regular_num_clusters,
            self.window,
            self.landmark_stride,
            self.seed,
            self.path,
            str(Path(self.compiled_graph_dir).resolve()) if self.compiled_graph_dir else None,
        )

    def _build_graph_abi(self, T: int) -> Tuple[MLXGraphABI, WayfinderGraphABI]:
        self._resolve_and_sync_num_cycles(int(T))
        if self.strategy not in {"random", "regular_partition"}:
            raise ValueError(
                "Qwen full-swap currently supports input-independent strategies "
                "('random' or 'regular_partition') for deterministic caching."
            )

        # Fast path: permute path without diagnostics only needs permutations,
        # skip the expensive O(T*D) ABI construction.
        if (
            self.path == "permute"
            and not self.store_graph_tensors
            and not self.store_numpy_abi
            and not self.verify_spectral_gap
        ):
            graph = self.topology.construct_perms_only(int(T))
            abi = graph.abi
            mlx_graph = to_mlx_graph_abi(abi, heads=self.n_heads, validate=False)
            return mlx_graph, abi

        abi = self.topology.construct({"T": int(T), "include_self": True}).abi
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
        if self.path == "sparse":
            s_idx = safe_neighbor_idx(mlx_graph.neigh_idx, T)
            c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, T)
        elif self.store_graph_tensors:
            # Permute path with diagnostics: precompute causal mask once
            # so _edge_utilization_proxy doesn't recompute every forward.
            s_idx = mx.zeros((self.n_heads, T, 0), dtype=mx.int32)
            c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, T)
        else:
            # Permute path without diagnostics: zero-sized placeholders.
            s_idx = mx.zeros((self.n_heads, T, 0), dtype=mx.int32)
            c_mask = mx.zeros((self.n_heads, T, 0), dtype=mx.bool_)

        if self.store_graph_tensors:
            cache_graph = mlx_graph
        else:
            cache_graph = MLXGraphABI(
                neigh_idx=mx.zeros((self.n_heads, T, 0), dtype=mx.int32),
                edge_type=mx.zeros((self.n_heads, T, 0), dtype=mx.uint8),
                meta=dict(mlx_graph.meta),
            )

        perm_mx_list: List[mx.array] = []
        inv_perm_list: List[mx.array] = []

        cycle_perms = mlx_graph.meta.get("cycle_perms", [])
        all_cycle_perms = mlx_graph.meta.get("all_cycle_perms", [])
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
                p_mx = mx.arange(T, dtype=mx.int32)
                perms_h.append(p_mx)
                invs_h.append(p_mx)
            else:
                for perm in perms_src:
                    perm_arr = np.asarray(perm, dtype=np.int32)
                    p_mx = mx.array(perm_arr, dtype=mx.int32)
                    perms_h.append(p_mx)
                    invs_h.append(mx.argsort(p_mx))

            max_d = max(max_d, len(perms_h))
            per_head_perms.append(perms_h)
            per_head_invs.append(invs_h)
            perm_mx_list.append(perms_h[0])
            inv_perm_list.append(invs_h[0])

        if max_d == 1:
            perm_stacked = mx.stack([p[0] for p in per_head_perms], axis=0)  # [H, T]
            inv_stacked = mx.stack([p[0] for p in per_head_invs], axis=0)  # [H, T]
        else:
            perm_heads: list[mx.array] = []
            inv_heads: list[mx.array] = []
            for perms_h, invs_h in zip(per_head_perms, per_head_invs):
                while len(perms_h) < max_d:
                    perms_h.append(perms_h[0])
                    invs_h.append(invs_h[0])
                perm_heads.append(mx.stack(perms_h, axis=0))
                inv_heads.append(mx.stack(invs_h, axis=0))
            perm_stacked = mx.stack(perm_heads, axis=0)  # [H, d, T]
            inv_stacked = mx.stack(inv_heads, axis=0)  # [H, d, T]

        persistent_bytes = _mx_nbytes(cache_graph.neigh_idx) + _mx_nbytes(cache_graph.edge_type)
        persistent_bytes += _mx_nbytes(s_idx) + _mx_nbytes(c_mask)
        persistent_bytes += _mx_nbytes(perm_stacked) + _mx_nbytes(inv_stacked)

        return _QwenGraphCache(
            mlx_graph=cache_graph,
            numpy_abi=numpy_abi if self.store_numpy_abi else None,
            safe_idx=s_idx,
            causal_mask=c_mask,
            perm_mx=perm_mx_list,
            inv_perm=inv_perm_list,
            perm_mx_stacked=perm_stacked,
            inv_perm_stacked=inv_stacked,
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
        meta.setdefault("all_cycle_perms", [])
        meta.setdefault("max_degree", int(neigh_idx.shape[-1]))
        meta.setdefault("seq_len", int(T))
        meta.setdefault("n_heads", int(self.n_heads))

        numpy_abi = WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)
        validate_graph_abi(
            numpy_abi,
            expect_heads=self.n_heads,
            expect_tokens=T,
            enforce_hamiltonian=self.enforce_hamiltonian,
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
        if self.cache_mode == "static":
            shared = _QWEN_GRAPH_CACHE_BY_KEY.get(key)
            if shared is not None:
                _QWEN_GRAPH_CACHE_STORE[owner_id] = shared
                return shared, True

        compiled = self._load_compiled_cache(T, key)
        if compiled is not None:
            if self.cache_mode == "static":
                _QWEN_GRAPH_CACHE_STORE[owner_id] = compiled
                _QWEN_GRAPH_CACHE_BY_KEY[key] = compiled
            return compiled, False

        mlx_graph, numpy_abi = self._build_graph_abi(T)
        built = self._build_cache(mlx_graph, numpy_abi, T, cache_key=key, source="runtime")
        if self.cache_mode == "static":
            _QWEN_GRAPH_CACHE_STORE[owner_id] = built
            _QWEN_GRAPH_CACHE_BY_KEY[key] = built
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

    def __init__(self, base_attn: nn.Module, cfg: QwenWayfinderConfig):
        super().__init__()

        self.n_heads = _attn_num_heads(base_attn)
        self.n_kv_heads = _attn_num_kv_heads(base_attn)
        self.scale = _attn_scale(base_attn)
        self.head_dim = _attn_head_dim(base_attn, scale=self.scale)

        if hasattr(base_attn, "q_proj"):
            self.q_proj = base_attn.q_proj
        if hasattr(base_attn, "k_proj"):
            self.k_proj = base_attn.k_proj
        if hasattr(base_attn, "v_proj"):
            self.v_proj = base_attn.v_proj
        if hasattr(base_attn, "qkv_proj"):
            self.qkv_proj = base_attn.qkv_proj
        self.o_proj = base_attn.o_proj
        self.q_norm = getattr(base_attn, "q_norm", None) or nn.Identity()
        self.k_norm = getattr(base_attn, "k_norm", None) or nn.Identity()
        self.rope = _attn_rope(base_attn)

        self.path = cfg.path
        self.permute_head_chunk_size = int(max(1, cfg.permute_head_chunk_size))
        self.query_chunk_size = int(max(1, cfg.query_chunk_size))
        self.permute_stream_o_proj = bool(cfg.permute_stream_o_proj)
        self.permute_log_chunks = bool(cfg.permute_log_chunks)
        self.retro_backfill_enabled = bool(cfg.retro_backfill_enabled)
        self.retro_backfill_alpha = float(cfg.retro_backfill_alpha)
        self.retro_backfill_training_only = bool(cfg.retro_backfill_training_only)
        self.retro_backfill_causal_only = bool(cfg.retro_backfill_causal_only)
        self.circular = bool(cfg.circular)
        self.multi_cycle_mode = str(cfg.multi_cycle_mode)
        self.use_fused_dispatch = bool(cfg.use_fused_dispatch)
        self.wayfinder_decode_backend = str(cfg.wayfinder_decode_backend)
        if cfg.active_dense_threshold == "auto":
            self.active_dense_threshold = (2 * cfg.window + 1) ** 2
        elif cfg.active_dense_threshold is None:
            self.active_dense_threshold = None
        else:
            self.active_dense_threshold = int(max(0, cfg.active_dense_threshold))
        self.window_drop_prob = float(max(0.0, min(1.0, cfg.window_drop)))
        self.edge_type_bias = mx.zeros((4,)) if cfg.edge_bias else None
        self.graph_runtime = _QwenGraphRuntime(
            n_heads=self.n_heads,
            window=cfg.window,
            landmark_stride=cfg.landmark_stride,
            strategy=cfg.strategy,
            num_cycles=cfg.num_cycles,
            edge_disjoint=cfg.edge_disjoint,
            enforce_hamiltonian=cfg.enforce_hamiltonian,
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
        self._active_graph_seq_len: int = 0
        # Fast graph build: perms-only path avoids O(T*D) ABI construction.
        # When available, use exact k_len so Tg == Tk and active-row decode
        # can keep contiguous local windows over valid tokens.
        self._fast_graph_build = bool(
            self.path == "permute"
            and not cfg.compute_graph_metrics
            and not cfg.compute_edge_utilization_proxy
            and self.path != "sparse"
            and not cfg.verify_spectral_gap
        )

        self.last_profile: AttentionProfile = AttentionProfile(path=cfg.path)
        self.last_graph_abi: Optional[WayfinderGraphABI] = None
        self.last_graph_metrics: Dict[str, Any] = {}
        self.last_edge_utilization_proxy: Dict[str, float] = {
            "cycle": 0.0,
            "window": 0.0,
            "landmark": 0.0,
            "rewire": 0.0,
        }
        self._o_proj_chunk_cache: Dict[tuple[int, int], nn.Module] = {}
        # Timed benchmarks can disable expensive runtime instrumentation.
        self.compute_edge_utilization_proxy: bool = bool(cfg.compute_edge_utilization_proxy)
        self.compute_graph_metrics: bool = bool(cfg.compute_graph_metrics)

    def _effective_permute_chunking(self, T: int) -> tuple[int, int]:
        """Token-aware chunking policy for permute path memory control."""
        h_chunk = int(max(1, self.permute_head_chunk_size))
        q_chunk = int(max(1, self.query_chunk_size))
        if T >= 8192:
            h_chunk = min(h_chunk, 2)
            q_chunk = min(q_chunk, 384)
        elif T >= 4096:
            h_chunk = min(h_chunk, 2)
            q_chunk = min(q_chunk, 384)
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
        elif self._fast_graph_build:
            # Fast perms-only build is cheap; match exact data horizon.
            return int(target)
        elif q_len <= 2:
            # Decode horizon bucketing: avoid rebuilding a new graph per token.
            step = 256
            target = ((target + step - 1) // step) * step
        else:
            # For larger active blocks, amortize graph builds.
            step = max(4096, int(max(1, q_len)) * 8)
            target = ((target + step - 1) // step) * step

        if self._active_graph_seq_len >= target:
            return int(self._active_graph_seq_len)
        self._active_graph_seq_len = int(target)
        return int(self._active_graph_seq_len)

    def _o_proj_part(self, x_btF: mx.array, f0: int, f1: int) -> mx.array:
        """Apply output projection to a feature slice [f0:f1] and return [B,T,D]."""
        if (
            isinstance(self.o_proj, nn.QuantizedLinear)
            and self.permute_stream_o_proj
        ):
            key = (int(f0), int(f1))
            layer = self._o_proj_chunk_cache.get(key)
            if layer is None:
                params = self.o_proj.parameters()
                bits = int(self.o_proj.bits)
                group_size = int(self.o_proj.group_size)
                packed = 32 // bits
                if (
                    f0 % packed != 0
                    or f1 % packed != 0
                    or f0 % group_size != 0
                    or f1 % group_size != 0
                ):
                    raise ValueError(
                        f"o_proj slice [{f0}:{f1}] must align to packed/group boundaries"
                    )
                w = params["weight"][:, f0 // packed : f1 // packed]
                s = params["scales"][:, f0 // group_size : f1 // group_size]
                b = params["biases"][:, f0 // group_size : f1 // group_size]
                layer = nn.QuantizedLinear(
                    input_dims=f1 - f0,
                    output_dims=int(w.shape[0]),
                    bias=False,
                    group_size=group_size,
                    bits=bits,
                    mode=self.o_proj.mode,
                )
                layer.update({"weight": w, "scales": s, "biases": b})
                self._o_proj_chunk_cache[key] = layer
            return layer(x_btF)

        # Fallback for non-quantized linear projections.
        if not hasattr(self.o_proj, "weight"):
            raise ValueError("Unsupported o_proj type for streamed projection")
        w = self.o_proj.weight[:, f0:f1]
        return mx.matmul(x_btF, w.transpose(1, 0))

    def _permute_attention_project_streamed(
        self,
        *,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        gate: Optional[mx.array],
        perms: mx.array,
        window: int,
        q_chunk: int,
        edge_type_bias_scalar: Optional[float],
        window_drop_prob: float,
        training: bool,
        log_progress: bool,
    ) -> mx.array:
        """Stream permute attention per head/chunk directly into projected hidden output."""
        B, _Hq, T, dh = queries.shape
        if isinstance(self.o_proj, nn.QuantizedLinear):
            out_dim = int(self.o_proj.parameters()["weight"].shape[0])
        else:
            out_dim = int(self.o_proj.weight.shape[0])

        out = mx.zeros((B, T, out_dim), dtype=queries.dtype)
        kv_repeat = self.n_heads // self.n_kv_heads
        scale = float(self.scale)

        for h in range(self.n_heads):
            kv_h = h // kv_repeat
            perm_h = perms[h : h + 1, :]  # [1, T]
            q_h = queries[:, h : h + 1, :, :]
            k_h = keys[:, kv_h : kv_h + 1, :, :]
            v_h = values[:, kv_h : kv_h + 1, :, :]
            f0 = h * self.head_dim
            f1 = f0 + self.head_dim

            q_chunk_count = (T + q_chunk - 1) // q_chunk
            for q_chunk_idx, s in enumerate(range(0, T, q_chunk), start=1):
                e = min(T, s + q_chunk)
                ks = max(0, s - window)
                ke = min(T, e + window)
                if log_progress:
                    print(
                        f"      permute_stream: head {h + 1}/{self.n_heads} "
                        f"q_chunk {q_chunk_idx}/{q_chunk_count} q[{s}:{e}] k[{ks}:{ke}]",
                        flush=True,
                    )

                q_idx = perm_h[:, s:e]  # [1, Q]
                k_idx = perm_h[:, ks:ke]  # [1, K]
                q_gidx = mx.broadcast_to(q_idx[None, :, :, None], (B, 1, e - s, 1))
                k_gidx = mx.broadcast_to(k_idx[None, :, :, None], (B, 1, ke - ks, 1))

                q_blk = mx.take_along_axis(q_h, q_gidx, axis=2)
                k_blk = mx.take_along_axis(k_h, k_gidx, axis=2)
                v_blk = mx.take_along_axis(v_h, k_gidx, axis=2)

                q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, e - s, 1)
                k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, ke - ks)
                rel = k_pos - q_pos  # [1, Q, K]
                in_window = (rel >= -window) & (rel <= window)
                causal = k_idx[:, None, :] <= q_idx[:, :, None]
                mask_eff = in_window & causal

                scores = mx.matmul(q_blk, k_blk.transpose(0, 1, 3, 2)) * scale

                if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
                    cycle_nb = ((rel == -1) | (rel == 1)).astype(scores.dtype)
                    bias = mx.array(edge_type_bias_scalar, dtype=scores.dtype)
                    scores = scores + cycle_nb * bias

                if training and window_drop_prob > 0.0:
                    preserve = (rel == 0) | (rel == -1) | (rel == 1)
                    drop_rand = mx.random.uniform(shape=(1, e - s, ke - ks)) < window_drop_prob
                    drop = drop_rand & in_window & (~preserve)
                    mask_eff = mask_eff & (~drop)

                w = stable_masked_softmax(
                    scores,
                    mask_eff[None, :, :, :],
                    axis=-1,
                    preserve_dtype=True,
                )
                y_blk = mx.matmul(w, v_blk).astype(values.dtype)  # [B, 1, Q, dh]
                if gate is not None:
                    g_h = gate[:, :, f0:f1]
                    g_idx = mx.broadcast_to(q_idx[0][None, :, None], (B, e - s, dh))
                    g_blk = mx.take_along_axis(g_h, g_idx, axis=1)
                    y_blk = y_blk * mx.sigmoid(g_blk[:, None, :, :]).astype(y_blk.dtype)
                part = self._o_proj_part(y_blk.reshape(B, e - s, dh), f0, f1)  # [B, Q, D]
                out = out.at[:, q_idx[0], :].add(part)
                mx.eval(out)

        if getattr(self.o_proj, "bias", None) is not None:
            out = out + self.o_proj.bias
        return out

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
        gate: Optional[mx.array],
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
        y_bt = y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1)
        if gate is not None:
            y_bt = y_bt * mx.sigmoid(gate[:, -int(y_bt.shape[1]) :, :]).astype(y_bt.dtype)
        return self.o_proj(y_bt)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        t_total0 = _now_ms()
        queries, keys, values, gate = extract_qkv_from_qwen_attention(
            self,
            x,
            cache=cache,
            return_gate=True,
        )
        q_len = int(queries.shape[2])
        k_len = int(keys.shape[2])
        sparse_active_mode = self.path == "sparse" and cache is not None and q_len < k_len
        active_mode = self.path == "permute" and cache is not None and q_len < k_len
        force_dense_large_active = active_mode and q_len > int(max(1, self.query_chunk_size))
        force_dense_active = (
            active_mode
            and self.active_dense_threshold is not None
            and k_len <= self.active_dense_threshold
            and q_len > 2
        )
        force_dense_wayfinder_decode = (
            active_mode
            and q_len <= 2
            and self.path == "permute"
            and self.wayfinder_decode_backend == "dense"
        )
        use_active_permute = active_mode and not (force_dense_active or force_dense_large_active)
        use_active_permute = use_active_permute and not force_dense_wayfinder_decode
        sparse_active_positions: Optional[mx.array] = None

        # During incremental decode, Q length != K length. Keep dense path for correctness.
        if force_dense_active or force_dense_large_active or (
            force_dense_wayfinder_decode
            or (q_len != k_len and not (use_active_permute or sparse_active_mode))
        ):
            fallback_reason = "active_dense_threshold" if force_dense_active else (
                "active_large_q"
                if force_dense_large_active
                else ("wayfinder_decode_dense" if force_dense_wayfinder_decode else "q_len_mismatch")
            )
            fallback_graph_seq_len = (
                self._adaptive_graph_seq_len(k_len=k_len, q_len=q_len, cache=cache)
                if active_mode
                else int(k_len)
            )
            t_attn0 = _now_ms()
            out = self._dense_fallback(queries, keys, values, gate, mask, cache)
            attn_ms = _now_ms() - t_attn0
            self.last_profile = AttentionProfile(
                graph_build_ms=0.0,
                permute_ms=0.0,
                attention_ms=float(attn_ms),
                total_ms=_now_ms() - t_total0,
                path=f"{self.path}_dense_fallback",
                notes={
                    "seq_len": int(k_len),
                    "graph_seq_len": int(fallback_graph_seq_len),
                    "cache_hit": True,
                    "cache_source": "dense_fallback",
                    "q_len": int(q_len),
                    "active_query_mode": bool(active_mode or sparse_active_mode),
                    "dense_fallback_reason": str(fallback_reason),
                    "active_dense_threshold": self.active_dense_threshold,
                    "active_dense_triggered": bool(force_dense_active),
                    "active_large_q_dense_triggered": bool(force_dense_large_active),
                    "wayfinder_decode_dense_triggered": bool(force_dense_wayfinder_decode),
                    "wayfinder_decode_backend": self.wayfinder_decode_backend,
                    "adaptive_graph_reuse": bool(int(fallback_graph_seq_len) != int(k_len)),
                    "sparse_active_mode": bool(sparse_active_mode),
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
                f"    qwen_wayfinder: requesting graph cache for T={graph_T} (seq_len={T})",
                flush=True,
            )
        graph_cache, cache_hit = self.graph_runtime.get_or_build_cache(id(self), graph_T)
        graph_ms = _now_ms() - t_graph0
        if self.permute_log_chunks:
            print(
                "    qwen_wayfinder: graph cache ready "
                f"(hit={bool(cache_hit)}, source={graph_cache.source}, {graph_ms:.1f} ms)",
                flush=True,
            )
        # Graph metrics are static for a cached ABI; avoid recomputing each forward.
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
        permute_ms = 0.0
        out_stream: Optional[mx.array] = None
        if self.path == "sparse":
            keys_q = _repeat_kv_to_q_heads(keys, self.n_heads)
            values_q = _repeat_kv_to_q_heads(values, self.n_heads)
            if sparse_active_mode:
                sparse_active_positions = mx.arange(T - q_len, T, dtype=mx.int32)
                y_h, _w = sparse_gather_attention_active(
                    queries,
                    keys_q,
                    values_q,
                    graph_cache.mlx_graph,
                    query_positions=sparse_active_positions,
                    return_weights=False,
                    precomputed_safe_idx=graph_cache.safe_idx,
                    precomputed_causal_mask=graph_cache.causal_mask,
                    edge_type_bias=self.edge_type_bias,
                    edge_type_bias_offset=scheduled_edge_bias,
                    window_drop_mask=wd_mask,
                )
                keep_mask_full = (
                    graph_cache.causal_mask
                    if wd_mask is None
                    else (graph_cache.causal_mask & wd_mask)
                )
                keep_mask = mx.take(keep_mask_full, sparse_active_positions, axis=1)
            else:
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
                keep_mask = (
                    graph_cache.causal_mask
                    if wd_mask is None
                    else (graph_cache.causal_mask & wd_mask)
                )
        elif self.path == "permute":
            h_chunk_eff, q_chunk_eff = self._effective_permute_chunking(T)
            streamable = self.permute_stream_o_proj and h_chunk_eff == 1 and not use_active_permute
            if use_active_permute:
                active_start = T - q_len
                active_positions = mx.arange(active_start, T, dtype=mx.int32)
                try:
                    y_h, _w = wayfinder_permute_window_attention_active_batched(
                        queries,
                        keys,
                        values,
                        all_perms=graph_cache.perm_mx_stacked,
                        all_inv_perms=graph_cache.inv_perm_stacked,
                        query_positions=active_positions,
                        window=self.graph_runtime.window,
                        edge_type_bias_scalar=etb_scalar,
                        window_drop_prob=effective_window_drop if is_training else 0.0,
                        training=is_training,
                        head_chunk_size=h_chunk_eff,
                        query_chunk_size=q_chunk_eff,
                        circular=self.circular,
                        multi_cycle_mode=self.multi_cycle_mode,
                        log_progress=self.permute_log_chunks,
                        use_fused_dispatch=self.use_fused_dispatch,
                        prefer_gather_for_small_tq=bool(q_len == 1),
                        scale=self.scale,
                    )
                except Exception as exc:
                    out = self._dense_fallback(queries, keys, values, gate, mask, cache)
                    attn_ms = _now_ms() - t_attn0
                    self.last_profile = AttentionProfile(
                        graph_build_ms=float(graph_ms),
                        permute_ms=0.0,
                        attention_ms=float(attn_ms),
                        total_ms=float(_now_ms() - t_total0),
                        path=f"{self.path}_dense_fallback",
                        notes={
                            "seq_len": int(T),
                            "graph_seq_len": int(graph_T),
                            "cache_hit": bool(cache_hit),
                            "cache_source": "dense_fallback",
                            "q_len": int(q_len),
                            "active_query_mode": True,
                            "dense_fallback_reason": "active_runtime_error",
                            "active_dense_triggered": False,
                            "active_large_q_dense_triggered": bool(force_dense_large_active),
                            "wayfinder_decode_dense_triggered": bool(force_dense_wayfinder_decode),
                            "wayfinder_decode_backend": self.wayfinder_decode_backend,
                            "adaptive_graph_reuse": bool(graph_T != T),
                            "sparse_active_mode": False,
                            "fallback_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    return out
            elif streamable:
                out_stream = self._permute_attention_project_streamed(
                    queries=queries,
                    keys=keys,
                    values=values,
                    gate=gate,
                    perms=graph_cache.perm_mx_stacked,
                    window=self.graph_runtime.window,
                    q_chunk=q_chunk_eff,
                    edge_type_bias_scalar=etb_scalar,
                    window_drop_prob=effective_window_drop if is_training else 0.0,
                    training=is_training,
                    log_progress=self.permute_log_chunks,
                )
            else:
                y_h, _w = wayfinder_permute_window_attention_batched(
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
            keep_mask = graph_cache.causal_mask
            if int(keep_mask.shape[1]) != T:
                keep_mask = keep_mask[:, :T, :]
        else:
            raise ValueError(f"Unknown path: {self.path}")
        attn_ms = _now_ms() - t_attn0

        if self.compute_edge_utilization_proxy:
            edge_type = graph_cache.mlx_graph.edge_type
            if sparse_active_positions is not None:
                edge_type = mx.take(edge_type, sparse_active_positions, axis=1)
            elif int(edge_type.shape[1]) != int(keep_mask.shape[1]):
                edge_type = edge_type[:, : int(keep_mask.shape[1]), :]
            self.last_edge_utilization_proxy = _edge_utilization_proxy(
                edge_type,
                keep_mask,
            )

        if out_stream is not None:
            out = out_stream
        else:
            y_bt = y_h.transpose(0, 2, 1, 3).reshape(y_h.shape[0], y_h.shape[2], -1)
            if gate is not None:
                y_bt = y_bt * mx.sigmoid(gate[:, -int(y_bt.shape[1]) :, :]).astype(y_bt.dtype)
            out = self.o_proj(y_bt)
        total_ms = _now_ms() - t_total0
        self.last_profile = AttentionProfile(
            graph_build_ms=float(graph_ms),
            permute_ms=float(permute_ms),
            attention_ms=float(attn_ms),
            total_ms=float(total_ms),
            path=self.path,
            notes={
                "seq_len": int(T),
                "graph_seq_len": int(graph_T),
                "max_degree": int(graph_cache.mlx_graph.neigh_idx.shape[-1]),
                "cache_hit": bool(cache_hit),
                "cache_mode": self.graph_runtime.cache_mode,
                "cache_source": graph_cache.source,
                "cache_persistent_bytes": int(graph_cache.persistent_bytes),
                "window_drop_effective": float(effective_window_drop),
                "permute_head_chunk_effective": int(h_chunk_eff) if self.path == "permute" else None,
                "permute_query_chunk_effective": int(q_chunk_eff) if self.path == "permute" else None,
                "q_len": int(q_len),
                "active_query_mode": bool(use_active_permute or sparse_active_mode),
                "dense_fallback_reason": None,
                "active_dense_triggered": False,
                "active_large_q_dense_triggered": bool(force_dense_large_active),
                "wayfinder_decode_dense_triggered": bool(force_dense_wayfinder_decode),
                "wayfinder_decode_backend": self.wayfinder_decode_backend,
                "adaptive_graph_reuse": bool(use_active_permute and graph_T != T),
                "sparse_active_mode": bool(sparse_active_mode),
            },
        )
        return out


def swap_qwen_attention_with_wayfinder(
    model: nn.Module,
    *,
    cfg: QwenWayfinderConfig,
    layer_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Replace selected Qwen attention blocks with Butterfly-backed attention modules.

    Returns:
        List of layer indices that were replaced.
    """
    replaced: list[int] = []
    if not hasattr(model, "layers"):
        raise ValueError("Model has no .layers attribute; expected a mlx_lm Qwen model.")
    selected_indices = None if layer_indices is None else {int(idx) for idx in layer_indices}

    for i, layer in enumerate(model.layers):
        if selected_indices is not None and i not in selected_indices:
            continue
        base_attn = getattr(layer, "self_attn", None)
        if base_attn is None:
            continue
        layer.self_attn = QwenWayfinderAttention(base_attn, cfg)
        replaced.append(i)
    return replaced


def get_qwen_full_attention_layer_indices(model: nn.Module) -> List[int]:
    if not hasattr(model, "layers"):
        raise ValueError("Model has no .layers attribute; expected a mlx_lm Qwen model.")
    indices: List[int] = []
    for idx, layer in enumerate(model.layers):
        if getattr(layer, "is_linear", None) is False:
            indices.append(int(idx))
            continue
        if getattr(layer, "self_attn", None) is not None:
            indices.append(int(idx))
    return indices


def validate_qwen35_full_attention_layers(
    model: nn.Module,
    *,
    allow_mismatch: bool = False,
) -> List[int]:
    discovered = get_qwen_full_attention_layer_indices(model)
    expected = list(EXPECTED_QWEN35_FULL_ATTENTION_LAYERS)
    if tuple(discovered) != EXPECTED_QWEN35_FULL_ATTENTION_LAYERS:
        message = (
            "Unexpected Qwen 3.5 full-attention layer layout: "
            f"expected={expected} discovered={discovered}"
        )
        if not allow_mismatch:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return discovered


def iter_qwen_wayfinder_layers(model: nn.Module) -> Iterable[QwenWayfinderAttention]:
    if not hasattr(model, "layers"):
        raise ValueError("Model has no .layers attribute; expected a mlx_lm Qwen model.")
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, QwenWayfinderAttention):
            yield attn
