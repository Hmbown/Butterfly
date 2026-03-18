from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import torch
import torch.nn as nn

from hcsa.graph.abi import graph_metrics
from hcsa.topology import Topology, TopologyGraph
from hcsa.torch.attention_wayfinder_permute import (
    FLEX_ATTENTION_AVAILABLE,
    build_flex_block_mask,
    recover_cycle_perms,
    wayfinder_flex_attention,
    wayfinder_permute_window_attention,
    wayfinder_permute_window_attention_batched,
)
from hcsa.torch.attention_wayfinder_sparse import sparse_row_attention

WayfinderPath = Literal["permute", "sparse"]
WayfinderStrategy = Literal["random", "regular_partition", "greedy", "online_insertion"]

_DYNAMIC_STRATEGIES = {"greedy", "online_insertion"}
_EDGE_TYPE_IDS = {"cycle": 0, "window": 1, "landmark": 2, "rewire": 3}


@dataclass
class NemotronHWayfinderConfig:
    """Configuration for the Nemotron-H / Nemotron 3 Super Wayfinder CUDA overlay.

    This adapter is intentionally conservative:

    * It only replaces attention blocks.
    * It is inference-first and falls back to the stock attention path for
      training, decode, cached prefill, padding-heavy masks, and attention
      introspection requests.
    """

    path: WayfinderPath = "permute"
    strategy: WayfinderStrategy = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int = 1
    edge_disjoint: bool = True
    enforce_hamiltonian: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    routing_dim: Optional[int] = None
    edge_bias: bool = False
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None
    engine: str = "auto"  # "auto" | "flex" | "batched" | "legacy"
    dense_fallback_q_len: int = 1
    fallback_on_output_attentions: bool = True
    fallback_on_mask: bool = True
    compute_graph_metrics: bool = False

    def __post_init__(self) -> None:
        if self.path not in {"permute", "sparse"}:
            raise ValueError(f"Unsupported Wayfinder path: {self.path!r}")
        if self.strategy not in {"random", "regular_partition", "greedy", "online_insertion"}:
            raise ValueError(f"Unsupported Wayfinder strategy: {self.strategy!r}")
        if int(self.window) < 0:
            raise ValueError("window must be >= 0")
        if int(self.num_cycles) <= 0:
            raise ValueError("num_cycles must be > 0")
        if int(self.regular_num_clusters) <= 0:
            raise ValueError("regular_num_clusters must be > 0")
        if int(self.dense_fallback_q_len) < 0:
            raise ValueError("dense_fallback_q_len must be >= 0")
        if not (0.0 <= float(self.window_drop) <= 1.0):
            raise ValueError("window_drop must be in [0, 1]")


@dataclass
class _NemotronGraphCache:
    graph: TopologyGraph
    neigh_idx: torch.Tensor
    edge_type: torch.Tensor
    safe_idx: Optional[torch.Tensor] = None
    causal_mask: Optional[torch.Tensor] = None
    perm: list[torch.Tensor] = field(default_factory=list)
    inv_perm: list[torch.Tensor] = field(default_factory=list)
    pi_idx_clamped: list[torch.Tensor] = field(default_factory=list)
    valid_mask: list[torch.Tensor] = field(default_factory=list)
    causal_masks: list[torch.Tensor] = field(default_factory=list)
    # Stacked tensors for batched/flex engines
    perm_stacked: Optional[torch.Tensor] = None       # [H_q, T]
    inv_perm_stacked: Optional[torch.Tensor] = None    # [H_q, T]
    kv_perm_stacked: Optional[torch.Tensor] = None     # [H_kv, T]
    kv_inv_perm_stacked: Optional[torch.Tensor] = None # [H_kv, T]
    pi_idx_stacked: Optional[torch.Tensor] = None      # [H_q, T, 2W+1]
    valid_stacked: Optional[torch.Tensor] = None        # [H_q, T, 2W+1]
    causal_stacked: Optional[torch.Tensor] = None       # [H_q, T, 2W+1]
    flex_block_mask: Any = None
    metrics: Optional[Dict[str, Any]] = None
    cache_key: Optional[tuple[Any, ...]] = None


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads to query-head count."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _safe_neighbor_idx(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return neigh_idx.clamp(min=0, max=seq_len - 1)


def _causal_neighbor_mask(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    query_positions = torch.arange(seq_len, device=neigh_idx.device, dtype=torch.long).view(1, seq_len, 1)
    return (neigh_idx >= 0) & (neigh_idx < seq_len) & (neigh_idx <= query_positions)


def _schedule_bias_to_vec(
    schedule_bias: Optional[Dict[str, float]],
    *,
    device: torch.device,
) -> torch.Tensor:
    vec = torch.zeros((4,), device=device, dtype=torch.float32)
    if not schedule_bias:
        return vec
    for name, value in schedule_bias.items():
        idx = _EDGE_TYPE_IDS.get(str(name).lower())
        if idx is not None:
            vec[idx] = float(value)
    return vec


def _has_required_nemotron_attention_attrs(module: nn.Module) -> bool:
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "hidden_size",
        "num_heads",
        "num_key_value_heads",
        "num_key_value_groups",
        "head_dim",
    )
    return all(hasattr(module, name) for name in required)


def _get_decoder_with_layers(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise ValueError("Could not find a decoder with a `.layers` attribute on the supplied model.")


def _cache_seq_len(past_key_values: Optional[Any], layer_idx: Optional[int]) -> int:
    if past_key_values is None or layer_idx is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if getter is None:
        return 0
    try:
        return int(getter(layer_idx))
    except TypeError:
        try:
            return int(getter())
        except Exception:
            return 0
    except Exception:
        return 0


def _is_standard_unpadded_causal_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    q_len: int,
    k_len: int,
) -> bool:
    """Accept None, all-ones 2D masks, or the stock 4D causal mask without padding."""
    if attention_mask is None:
        return True

    if attention_mask.dim() == 2:
        return bool(torch.all(attention_mask == 1))

    if attention_mask.dim() != 4:
        return False

    if int(attention_mask.shape[-2]) != int(q_len) or int(attention_mask.shape[-1]) != int(k_len):
        return False
    if int(attention_mask.shape[1]) != 1:
        return False
    if q_len != k_len:
        return False
    if not attention_mask.is_floating_point():
        return False

    min_dtype = torch.finfo(attention_mask.dtype).min
    # Sampling-based check: verify structure without materializing T×T tensor.
    m = attention_mask
    if q_len > 1:
        if float(m[0, 0, 0, 1]) != min_dtype:
            return False
        if float(m[0, 0, 1, 0]) != 0.0:
            return False
        if float(m[0, 0, q_len - 1, 0]) != 0.0:
            return False
        if float(m[0, 0, 0, q_len - 1]) != min_dtype:
            return False
    diag = torch.diagonal(m[0, 0])
    if not torch.all(diag == 0):
        return False
    return True


def extract_qkv_from_nemotron_h_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    *,
    past_key_values: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Q/K/V tensors using the Nemotron-H attention projection layout."""
    batch_size, query_len, _ = hidden_states.size()
    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(batch_size, query_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, query_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, query_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    if past_key_values is not None:
        layer_idx = getattr(attn, "layer_idx", None)
        if layer_idx is None:
            raise ValueError("Attention module does not have `layer_idx`; cannot update cache safely.")
        key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

    return query_states, key_states, value_states


class NemotronHWayfinderAttention(nn.Module):
    """Wrap a stock Nemotron-H attention mixer with Wayfinder prefill attention."""

    def __init__(self, fallback_attention: nn.Module, cfg: NemotronHWayfinderConfig):
        super().__init__()
        if not _has_required_nemotron_attention_attrs(fallback_attention):
            raise TypeError(
                "fallback_attention does not look like a Nemotron-H attention module. "
                "Expected q_proj/k_proj/v_proj/o_proj, hidden_size, num_heads, "
                "num_key_value_heads, num_key_value_groups, and head_dim."
            )

        self.fallback = fallback_attention
        self.cfg = copy.deepcopy(cfg)

        self.config = getattr(fallback_attention, "config", None)
        self.layer_idx = getattr(fallback_attention, "layer_idx", None)
        self.hidden_size = int(getattr(fallback_attention, "hidden_size"))
        self.num_heads = int(getattr(fallback_attention, "num_heads"))
        self.num_key_value_heads = int(getattr(fallback_attention, "num_key_value_heads"))
        self.num_key_value_groups = int(getattr(fallback_attention, "num_key_value_groups"))
        self.head_dim = int(getattr(fallback_attention, "head_dim"))
        self.scaling = float(getattr(fallback_attention, "scaling", self.head_dim**-0.5))
        self.attention_dropout = float(getattr(fallback_attention, "attention_dropout", 0.0))

        proj_ref = self.fallback.q_proj.weight
        proj_device = proj_ref.device
        proj_dtype = proj_ref.dtype

        self.routing_dim = int(self.cfg.routing_dim or self.head_dim)
        if self.cfg.strategy in _DYNAMIC_STRATEGIES:
            self.routing_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.routing_dim,
                bias=False,
                device=proj_device,
                dtype=proj_dtype,
            )
        else:
            self.routing_proj = None

        if self.cfg.edge_bias:
            self.edge_type_bias = nn.Parameter(torch.zeros((4,), device=proj_device, dtype=torch.float32))
        else:
            self.register_parameter("edge_type_bias", None)

        # Generate one permutation per KV head; shared by all query heads in GQA group.
        self.topology = Topology(
            n_heads=self.num_key_value_heads,
            strategy=self.cfg.strategy,
            num_cycles=self.cfg.num_cycles,
            edge_disjoint=self.cfg.edge_disjoint,
            regular_num_clusters=self.cfg.regular_num_clusters,
            seed=self.cfg.seed,
            window=self.cfg.window,
            landmark_stride=self.cfg.landmark_stride,
            enforce_hamiltonian=self.cfg.enforce_hamiltonian,
        )

        self._graph_cache: dict[tuple[Any, ...], _NemotronGraphCache] = {}
        self._runtime_window_drop_override: Optional[float] = None
        self.register_buffer(
            "_runtime_schedule_bias_vec",
            torch.zeros((4,), device=proj_device, dtype=torch.float32),
            persistent=False,
        )
        self.last_profile: Dict[str, Any] = {}

    @property
    def cache_mode(self) -> str:
        return self.topology.cache_mode

    def _resolve_engine(self) -> str:
        engine = self.cfg.engine
        if engine == "auto":
            return "flex" if FLEX_ATTENTION_AVAILABLE else "batched"
        return engine

    def set_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        self._runtime_window_drop_override = (
            None if window_drop is None else float(min(1.0, max(0.0, float(window_drop))))
        )
        self._runtime_schedule_bias_vec = _schedule_bias_to_vec(
            schedule_bias,
            device=self._runtime_schedule_bias_vec.device,
        )

    def clear_runtime_controls(self) -> None:
        self._runtime_window_drop_override = None
        self._runtime_schedule_bias_vec.zero_()

    def _cache_key(self, seq_len: int, device: torch.device) -> tuple[Any, ...]:
        compiled = None
        if self.cfg.compiled_graph_dir:
            compiled = str(Path(self.cfg.compiled_graph_dir).expanduser().resolve())
        return (
            int(seq_len),
            self.cfg.path,
            self.cfg.strategy,
            int(self.cfg.num_cycles),
            bool(self.cfg.edge_disjoint),
            int(self.cfg.regular_num_clusters),
            int(self.cfg.window),
            None if self.cfg.landmark_stride is None else int(self.cfg.landmark_stride),
            int(self.cfg.seed),
            compiled,
            str(device),
        )

    def _routing_by_head(self, hidden_states: torch.Tensor) -> Optional[list[torch.Tensor]]:
        if self.routing_proj is None:
            return None
        # Wayfinder's torch path uses the first sample to drive per-head dynamic routing.
        routing = self.routing_proj(hidden_states[0])
        routing = routing.reshape(hidden_states.shape[1], self.num_key_value_heads, self.routing_dim).permute(1, 0, 2)
        routing = routing.detach().float().cpu()
        return [routing[h] for h in range(self.num_key_value_heads)]

    def _build_graph(self, hidden_states: torch.Tensor, seq_len: int) -> TopologyGraph:
        if self.cfg.compiled_graph_dir:
            compiled_path = Path(self.cfg.compiled_graph_dir).expanduser()
            if not compiled_path.exists():
                raise FileNotFoundError(f"Compiled graph path does not exist: {compiled_path}")
            return self.topology.load(
                compiled_path,
                expect_heads=self.num_key_value_heads,
                expect_tokens=seq_len,
            )

        if self.cfg.path == "permute" and self.cache_mode == "static":
            return self.topology.construct_perms_only(int(seq_len))

        routing_by_head = None
        if self.cfg.strategy in _DYNAMIC_STRATEGIES:
            routing_by_head = self._routing_by_head(hidden_states)

        return self.topology.construct(
            {"T": int(seq_len), "include_self": True},
            routing_by_head=routing_by_head,
            include_self=True,
        )

    def _extract_cycle_perms(
        self,
        neigh_idx: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        meta: Optional[Dict[str, Any]],
        seq_len: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        perms_raw = recover_cycle_perms(neigh_idx, edge_type, meta=meta)
        perm_list: list[torch.Tensor] = []
        inv_list: list[torch.Tensor] = []
        pi_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        causal_list: list[torch.Tensor] = []

        offsets = torch.arange(
            -int(self.cfg.window),
            int(self.cfg.window) + 1,
            device=device,
            dtype=torch.long,
        )
        base_idx = torch.arange(seq_len, device=device, dtype=torch.long).view(seq_len, 1)

        for head, perm_h in enumerate(perms_raw):
            if perm_h is None:
                raise ValueError(f"Missing cycle permutation for head {head}")

            perm_t = torch.as_tensor(perm_h, dtype=torch.long, device=device)
            if perm_t.numel() != seq_len:
                raise ValueError(
                    f"Recovered permutation for head {head} has length {perm_t.numel()}, expected {seq_len}"
                )

            inv_perm = torch.empty_like(perm_t)
            inv_perm[perm_t] = torch.arange(seq_len, device=device, dtype=torch.long)

            pi_idx = base_idx + offsets.view(1, -1)
            valid = (pi_idx >= 0) & (pi_idx < seq_len)
            pi_idx_clamped = pi_idx.clamp(0, seq_len - 1)

            neigh_orig = perm_t[pi_idx_clamped]
            query_orig = perm_t.view(seq_len, 1)
            causal_h = neigh_orig <= query_orig

            perm_list.append(perm_t)
            inv_list.append(inv_perm)
            pi_list.append(pi_idx_clamped)
            valid_list.append(valid)
            causal_list.append(causal_h)

        return perm_list, inv_list, pi_list, valid_list, causal_list

    def _build_cache(self, hidden_states: torch.Tensor, seq_len: int) -> _NemotronGraphCache:
        graph = self._build_graph(hidden_states, seq_len)
        device = hidden_states.device
        abi = graph.abi

        # neigh_idx/edge_type from topology are [H_kv, T, D].
        neigh_idx_kv = torch.as_tensor(abi.neigh_idx, dtype=torch.long, device=device)
        edge_type_kv = torch.as_tensor(abi.edge_type, dtype=torch.uint8, device=device)

        # Extract cycle perms from H_kv tensors BEFORE replication.
        kv_perm_list: list[torch.Tensor] = []
        kv_inv_list: list[torch.Tensor] = []
        perm_list: list[torch.Tensor] = []
        inv_list: list[torch.Tensor] = []
        pi_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        causal_list: list[torch.Tensor] = []
        if self.cfg.path == "permute":
            kv_perm_list, kv_inv_list, _pil, _vl, _cl = self._extract_cycle_perms(
                neigh_idx_kv,
                edge_type_kv,
                meta=getattr(abi, "meta", None),
                seq_len=seq_len,
                device=device,
            )
            g = self.num_key_value_groups
            perm_list = [p for p in kv_perm_list for _ in range(g)]
            inv_list = [p for p in kv_inv_list for _ in range(g)]
            pi_list = [p for p in _pil for _ in range(g)]
            valid_list = [p for p in _vl for _ in range(g)]
            causal_list = [p for p in _cl for _ in range(g)]

        # Replicate [H_kv, T, D] → [H_q, T, D] so sparse/legacy paths
        # get neigh_idx matching the GQA-expanded Q/K/V head count.
        if neigh_idx_kv.ndim == 3 and self.num_key_value_groups > 1:
            neigh_idx = neigh_idx_kv.repeat_interleave(self.num_key_value_groups, dim=0)
            edge_type = edge_type_kv.repeat_interleave(self.num_key_value_groups, dim=0)
        else:
            neigh_idx = neigh_idx_kv
            edge_type = edge_type_kv

        safe_idx: Optional[torch.Tensor] = None
        causal_mask: Optional[torch.Tensor] = None
        if self.cfg.path == "sparse":
            safe_idx = _safe_neighbor_idx(neigh_idx, seq_len)
            causal_mask = _causal_neighbor_mask(neigh_idx, seq_len)

        metrics = None
        if self.cfg.compute_graph_metrics:
            try:
                metrics = graph_metrics(abi)
            except Exception:
                metrics = None

        # Build stacked tensors for batched/flex engines
        perm_stacked = inv_perm_stacked = None
        kv_perm_stacked = kv_inv_perm_stacked = None
        pi_idx_stacked = valid_stacked = causal_stacked = None
        flex_block_mask_val = None

        if self.cfg.path == "permute" and perm_list:
            kv_perm_stacked = torch.stack(kv_perm_list)     # [H_kv, T]
            kv_inv_perm_stacked = torch.stack(kv_inv_list)  # [H_kv, T]
            perm_stacked = torch.stack(perm_list)            # [H_q, T]
            inv_perm_stacked = torch.stack(inv_list)         # [H_q, T]
            if pi_list:
                pi_idx_stacked = torch.stack(pi_list)        # [H_q, T, 2W+1]
                valid_stacked = torch.stack(valid_list)       # [H_q, T, 2W+1]
                causal_stacked = torch.stack(causal_list)     # [H_q, T, 2W+1]

            engine = self._resolve_engine()
            if engine == "flex" and FLEX_ATTENTION_AVAILABLE:
                try:
                    flex_block_mask_val = build_flex_block_mask(
                        perm_stacked, window=int(self.cfg.window),
                        B=1, device=device,
                    )
                except Exception:
                    pass

        return _NemotronGraphCache(
            graph=graph,
            neigh_idx=neigh_idx,
            edge_type=edge_type,
            safe_idx=safe_idx,
            causal_mask=causal_mask,
            perm=perm_list,
            inv_perm=inv_list,
            pi_idx_clamped=pi_list,
            valid_mask=valid_list,
            causal_masks=causal_list,
            perm_stacked=perm_stacked,
            inv_perm_stacked=inv_perm_stacked,
            kv_perm_stacked=kv_perm_stacked,
            kv_inv_perm_stacked=kv_inv_perm_stacked,
            pi_idx_stacked=pi_idx_stacked,
            valid_stacked=valid_stacked,
            causal_stacked=causal_stacked,
            flex_block_mask=flex_block_mask_val,
            metrics=metrics,
            cache_key=self._cache_key(seq_len, device),
        )

    def _get_or_build_cache(self, hidden_states: torch.Tensor, seq_len: int) -> tuple[_NemotronGraphCache, bool]:
        cache_key = self._cache_key(seq_len, hidden_states.device)
        if self.cache_mode == "static":
            cached = self._graph_cache.get(cache_key)
            if cached is not None:
                return cached, True

        cache = self._build_cache(hidden_states, seq_len)
        if self.cache_mode == "static":
            self._graph_cache[cache_key] = cache
        return cache, False

    def _effective_window_drop(self) -> float:
        if self._runtime_window_drop_override is not None:
            return float(self._runtime_window_drop_override)
        return float(self.cfg.window_drop)

    def _schedule_bias(self, device: torch.device) -> Optional[torch.Tensor]:
        schedule_bias = self._runtime_schedule_bias_vec.to(device=device, dtype=torch.float32)
        if float(schedule_bias.abs().sum().item()) == 0.0:
            return None
        return schedule_bias

    def _window_drop_mask(self, cache: _NemotronGraphCache, seq_len: int) -> Optional[torch.Tensor]:
        effective_window_drop = self._effective_window_drop()
        if not self.training or effective_window_drop <= 0.0 or self.cfg.path != "sparse":
            return None
        if cache.safe_idx is None:
            return None
        et = cache.edge_type
        if et.numel() == 0:
            return None
        is_window = et == 2
        query_idx = torch.arange(seq_len, device=et.device, dtype=torch.long).view(1, seq_len, 1)
        is_self = cache.safe_idx == query_idx
        droppable = is_window & (~is_self)
        drop_rand = torch.rand(et.shape, device=et.device) < float(effective_window_drop)
        return ~(droppable & drop_rand)

    def _fallback_reason(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        k_len_after: int,
        output_attentions: bool,
    ) -> Optional[str]:
        # Wayfinder is intended as an inference-time prefill optimization.
        if self.training:
            return "training"
        if self.layer_idx is None:
            return "missing_layer_idx"
        if self.cfg.fallback_on_output_attentions and output_attentions:
            return "output_attentions"
        if q_len <= int(self.cfg.dense_fallback_q_len):
            return "short_query"
        if k_len_after != q_len:
            return "cached_kv"
        if self.cfg.fallback_on_mask and not _is_standard_unpadded_causal_mask(
            attention_mask,
            q_len=q_len,
            k_len=k_len_after,
        ):
            return "attention_mask"
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        q_len = int(hidden_states.shape[1])
        cache_len = _cache_seq_len(past_key_values, self.layer_idx)
        k_len_after = int(q_len + cache_len)
        output_attentions = bool(kwargs.get("output_attentions", False))
        fallback_reason = self._fallback_reason(
            attention_mask=attention_mask,
            q_len=q_len,
            k_len_after=k_len_after,
            output_attentions=output_attentions,
        )

        if fallback_reason is not None:
            self.last_profile = {
                "mode": "fallback",
                "reason": fallback_reason,
                "layer_idx": self.layer_idx,
                "seq_len": q_len,
                "kv_len": k_len_after,
                "path": self.cfg.path,
                "strategy": self.cfg.strategy,
            }
            return self.fallback(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        t0 = time.perf_counter()
        query_states, key_states, value_states = extract_qkv_from_nemotron_h_attention(
            self.fallback,
            hidden_states,
            past_key_values=past_key_values,
        )

        engine = self._resolve_engine()

        # Only skip GQA expansion for the permute+flex path, which handles
        # GQA natively via enable_gqa=True. All other paths (sparse, batched,
        # legacy) require expanded K/V with H_q heads.
        use_native_gqa = (self.cfg.path == "permute" and engine == "flex")
        if not use_native_gqa:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        cache, cache_hit = self._get_or_build_cache(hidden_states, q_len)
        schedule_bias = self._schedule_bias(hidden_states.device)
        graph_source = cache.graph.source

        if self.cfg.path == "sparse":
            wayfinder_out, _ = sparse_row_attention(
                query_states,
                key_states,
                value_states,
                neigh_idx=cache.neigh_idx,
                edge_type=cache.edge_type,
                return_weights=False,
                precomputed_safe_idx=cache.safe_idx,
                precomputed_causal_mask=cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                edge_type_bias_offset=schedule_bias,
                window_drop_mask=self._window_drop_mask(cache, q_len),
            )
        elif self.cfg.path == "permute":
            if engine == "flex" and cache.perm_stacked is not None:
                wayfinder_out, new_bm = wayfinder_flex_attention(
                    query_states, key_states, value_states,
                    window=int(self.cfg.window),
                    perm=cache.perm_stacked,
                    inv_perm=cache.inv_perm_stacked,
                    kv_perm=cache.kv_perm_stacked,
                    block_mask=cache.flex_block_mask,
                )
                if cache.flex_block_mask is None:
                    cache.flex_block_mask = new_bm

            elif engine == "batched" and cache.perm_stacked is not None:
                wayfinder_out = wayfinder_permute_window_attention_batched(
                    query_states, key_states, value_states,
                    window=int(self.cfg.window),
                    perm=cache.perm_stacked,
                    inv_perm=cache.inv_perm_stacked,
                    pi_idx_clamped=cache.pi_idx_stacked,
                    valid_mask=cache.valid_stacked,
                    causal_mask=cache.causal_stacked,
                )

            else:
                # Legacy per-head loop
                edge_bias_scalar: Optional[float] = None
                if self.edge_type_bias is not None:
                    edge_bias_scalar = float(self.edge_type_bias[0].item())
                if schedule_bias is not None:
                    cycle_bias = float(schedule_bias[0].item())
                    edge_bias_scalar = (
                        cycle_bias if edge_bias_scalar is None
                        else edge_bias_scalar + cycle_bias
                    )

                wayfinder_out, _weights, _permute_ms, _attn_ms = wayfinder_permute_window_attention(
                    query_states,
                    key_states,
                    value_states,
                    window=int(self.cfg.window),
                    neigh_idx=cache.neigh_idx,
                    edge_type=cache.edge_type,
                    graph_meta=getattr(cache.graph.abi, "meta", None),
                    return_weights=False,
                    cache=cache,
                    edge_type_bias_scalar=edge_bias_scalar,
                    window_drop_prob=self._effective_window_drop() if self.training else 0.0,
                    training=self.training,
                )
        else:
            raise ValueError(f"Unknown Wayfinder path: {self.cfg.path!r}")

        batch_size = hidden_states.shape[0]
        attn_output = wayfinder_out.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        attn_output = self.fallback.o_proj(attn_output)

        self.last_profile = {
            "mode": "wayfinder",
            "reason": None,
            "layer_idx": self.layer_idx,
            "seq_len": q_len,
            "kv_len": k_len_after,
            "path": self.cfg.path,
            "engine": self._resolve_engine(),
            "strategy": self.cfg.strategy,
            "graph_source": graph_source,
            "graph_cache_hit": bool(cache_hit),
            "graph_metrics": cache.metrics,
            "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return attn_output, None, past_key_values


def iter_nemotron_h_wayfinder_layers(model: nn.Module) -> Iterable[NemotronHWayfinderAttention]:
    decoder = _get_decoder_with_layers(model)
    for layer in decoder.layers:
        mixer = getattr(layer, "mixer", None)
        if isinstance(mixer, NemotronHWayfinderAttention):
            yield mixer


def set_nemotron_h_wayfinder_runtime_controls(
    model: nn.Module,
    *,
    window_drop: Optional[float] = None,
    schedule_bias: Optional[Dict[str, float]] = None,
) -> int:
    count = 0
    for mixer in iter_nemotron_h_wayfinder_layers(model):
        mixer.set_runtime_controls(window_drop=window_drop, schedule_bias=schedule_bias)
        count += 1
    return count


def clear_nemotron_h_wayfinder_runtime_controls(model: nn.Module) -> int:
    count = 0
    for mixer in iter_nemotron_h_wayfinder_layers(model):
        mixer.clear_runtime_controls()
        count += 1
    return count


def swap_nemotron_h_attention_with_wayfinder(
    model: nn.Module,
    cfg: Optional[NemotronHWayfinderConfig] = None,
    *,
    layer_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Replace Nemotron-H attention mixers with Wayfinder wrappers in-place.

    Parameters
    ----------
    model:
        A loaded Hugging Face Nemotron-H / Nemotron 3 Super model.
    cfg:
        Wayfinder adapter configuration. Defaults to ``NemotronHWayfinderConfig()``.
    layer_indices:
        Optional subset of decoder layer indices to swap.

    Returns
    -------
    list[int]
        The layer indices that were replaced.
    """
    config = copy.deepcopy(cfg) if cfg is not None else NemotronHWayfinderConfig()
    decoder = _get_decoder_with_layers(model)
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)

    replaced: list[int] = []
    for idx, layer in enumerate(decoder.layers):
        if selected is not None and idx not in selected:
            continue
        if getattr(layer, "block_type", None) != "attention":
            continue
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        if isinstance(mixer, NemotronHWayfinderAttention):
            replaced.append(idx)
            continue
        if not _has_required_nemotron_attention_attrs(mixer):
            continue

        if getattr(mixer, "layer_idx", None) is None:
            setattr(mixer, "layer_idx", idx)

        wrapped = NemotronHWayfinderAttention(mixer, config)
        wrapped.train(mixer.training)
        layer.mixer = wrapped
        replaced.append(idx)

    return replaced


__all__ = [
    "NemotronHWayfinderConfig",
    "NemotronHWayfinderAttention",
    "extract_qkv_from_nemotron_h_attention",
    "iter_nemotron_h_wayfinder_layers",
    "set_nemotron_h_wayfinder_runtime_controls",
    "clear_nemotron_h_wayfinder_runtime_controls",
    "swap_nemotron_h_attention_with_wayfinder",
]
