from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from hcsa.graph.abi import WayfinderGraphABI, stack_head_abis, validate_graph_abi
from hcsa.graph_strategies import build_strategy
from hcsa.torch.attention_hha_permute import hha_permute_window_attention, recover_cycle_perms
from hcsa.torch.bench_utils import (
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

    bias_htd: torch.Tensor | None = None
    if edge_type_bias is not None or edge_type_bias_offset is not None:
        bias_vec = torch.zeros((4,), device=q.device, dtype=torch.float32)
        if edge_type_bias is not None:
            bias_vec = bias_vec + edge_type_bias.to(device=q.device, dtype=torch.float32)
        if edge_type_bias_offset is not None:
            bias_vec = bias_vec + edge_type_bias_offset.to(device=q.device, dtype=torch.float32)
        full_bias = torch.cat([torch.zeros((1,), device=q.device, dtype=torch.float32), bias_vec])
        bias_htd = full_bias[et.long()]

    if bias_htd is not None:
        scores = scores + bias_htd.float()

    weights = stable_masked_softmax(scores.float(), mask, dim=-1)
    out = torch.sum(weights.unsqueeze(-1) * v_g.float(), dim=3).to(dtype=v.dtype)

    if return_weights:
        return out, weights
    return out, None


class WayfinderAttentionTorch(nn.Module):
    """PyTorch Wayfinder/HHA attention with shared graph ABI and cache semantics."""

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

        self._strategies = [self._make_strategy(h) for h in range(self.n_heads)]

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
        return "static" if self.strategy == "random" else "dynamic"

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

        r_tensor: torch.Tensor | None = None
        if self.strategy in {"greedy", "online_insertion"}:
            r_tensor = self.Wr(x[0]).reshape(t, self.n_heads, self.routing_dim).permute(1, 0, 2)
            r_tensor = r_tensor.detach().float().cpu()

        head_abis: list[WayfinderGraphABI] = []
        for h in range(self.n_heads):
            r_h = None if r_tensor is None else r_tensor[h]
            abi_h = self._strategies[h].build(
                T=t,
                r=r_h,
                head_idx=h,
                window=self.window,
                landmark_stride=self.landmark_stride,
                include_self=True,
            )
            head_abis.append(abi_h)

        abi = stack_head_abis(head_abis)
        validate_graph_abi(abi, expect_heads=self.n_heads, expect_tokens=t, enforce_hamiltonian=True)
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

    def forward(self, x: torch.Tensor, *, return_debug: bool = False):
        t_total0 = now_ms()
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        t_graph0 = now_ms()
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
            y_h, w, permute_ms, _attn_inner = hha_permute_window_attention(
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
