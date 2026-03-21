"""Wayfinder CUDA overlay for Qwen3.5 full-attention layers.

Mirrors the Nemotron-H integration pattern: wraps stock ``Qwen3_5Attention``
modules with Wayfinder attention while leaving linear-attention and MLP layers
untouched.

Conservative by design:
  * Inference-first — falls back to stock attention during training.
  * Mask-aware — nonstandard masks fall back.
  * Static sparse support — connectivity is precomputed per layer/sequence shape.
"""

from __future__ import annotations

import copy
import importlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import torch
import torch.nn as nn

from bna.graph.abi import graph_metrics
from bna.topology import Topology, TopologyGraph
from bna.torch.attention_wayfinder_permute import (
    BlockHamiltonianLayout,
    FLEX_ATTENTION_AVAILABLE,
    _wayfinder_stage_meta as _wayfinder_stage_meta_standalone,
    build_block_wayfinder_layout,
    build_block_hamiltonian_mask,
    build_flex_block_mask,
    recover_cycle_perms,
    wayfinder_block_sparse_attention,
    wayfinder_block_sparse_sdpa_attention,
    wayfinder_flex_attention,
    wayfinder_permute_window_attention,
    wayfinder_permute_window_attention_batched,
)
from bna.torch.attention_wayfinder_sparse import (
    build_sparse_edge_bias_tensor,
    sparse_row_attention_gqa_chunked,
    sparse_row_attention_gqa_precomputed,
)
from bna.torch.bench_utils import _repeat_kv

try:
    from bna.torch.triton_block_sparse_attn import (
        triton_block_sparse_attention,
        TRITON_AVAILABLE as TRITON_BLOCK_SPARSE_AVAILABLE,
    )
except ImportError:
    triton_block_sparse_attention = None  # type: ignore[assignment]
    TRITON_BLOCK_SPARSE_AVAILABLE = False

WayfinderPath = Literal["permute", "sparse", "block_sparse"]
WayfinderStrategy = Literal["random", "regular_partition", "greedy", "online_insertion"]
WayfinderSparseComputeDtype = Literal["auto", "model", "float32"]
WayfinderSparsePrecomputedBackend = Literal[
    "auto",
    "triton_fused",
    "sdpa",
    "streamed_online_softmax",
    "manual_matmul",
]
WayfinderBlockPartnerRule = Literal["xor", "bit_reversal", "benes"]

_DYNAMIC_STRATEGIES = {"greedy", "online_insertion"}
_EDGE_TYPE_IDS = {"cycle": 0, "window": 1, "landmark": 2, "rewire": 3}

_GPTQ_MOE_PATCHED = False


def patch_qwen_moe_for_gptq() -> None:
    """Monkey-patch ``Qwen3_5MoeExperts`` for GPTQ checkpoint compatibility.

    Transformers 5.x stores MoE expert weights as fused 3D ``nn.Parameter``
    tensors (``gate_up_proj [num_experts, 2*intermediate, hidden]``).  GPTQ
    checkpoints created with older transformers store per-expert
    ``nn.Linear``-style keys (``experts.{i}.gate_proj.qweight``).  GPTQ works
    by replacing ``nn.Linear`` → ``QuantLinear``, which cannot target a raw
    ``nn.Parameter``.

    This patch replaces the ``Qwen3_5MoeExperts`` class with a version that
    creates per-expert ``nn.Linear`` modules, so GPTQ weights load naturally.

    **Must be called before** ``AutoModelForCausalLM.from_pretrained()``.
    """
    global _GPTQ_MOE_PATCHED
    if _GPTQ_MOE_PATCHED:
        return

    try:
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen_mod
    except ImportError:
        raise ImportError("transformers does not have qwen3_5_moe model support")

    original_cls = qwen_mod.Qwen3_5MoeExperts
    ACT2FN = qwen_mod.ACT2FN

    class _Qwen3_5MoeExpertsGPTQ(nn.Module):
        """Per-expert ``nn.Linear`` version of ``Qwen3_5MoeExperts``.

        This creates individual ``gate_proj``, ``up_proj``, ``down_proj``
        modules per expert (stored in a flat namespace ``self.{i}``), matching
        the GPTQ checkpoint key structure::

            model.layers.L.mlp.experts.{i}.gate_proj.qweight
            model.layers.L.mlp.experts.{i}.up_proj.qweight
            model.layers.L.mlp.experts.{i}.down_proj.qweight
        """

        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = ACT2FN[config.hidden_act]
            for i in range(self.num_experts):
                expert = nn.ModuleDict({
                    "gate_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "up_proj": nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False),
                    "down_proj": nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False),
                })
                # Register as self.{i} so state_dict keys become experts.{i}.gate_proj.weight etc.
                self.add_module(str(i), expert)

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx >= self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                expert = getattr(self, str(expert_idx.item()))
                gate = expert["gate_proj"](current_state)
                up = expert["up_proj"](current_state)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = expert["down_proj"](current_hidden_states)
                current_hidden_states = (
                    current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
                )
            return final_hidden_states

    # Patch the class in the module so from_pretrained uses it
    qwen_mod.Qwen3_5MoeExperts = _Qwen3_5MoeExpertsGPTQ
    _GPTQ_MOE_PATCHED = True


def _parse_cuda_arch_token(token: str) -> Optional[tuple[int, int]]:
    if "_" not in token:
        return None
    prefix, digits = token.split("_", 1)
    if prefix not in {"sm", "compute"} or not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def get_cuda_arch_support_diagnostics(
    device: Optional[torch.device | int] = None,
) -> Dict[str, Any]:
    arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    supported_caps = sorted(
        {parsed for token in arch_list if (parsed := _parse_cuda_arch_token(token)) is not None}
    )
    capability = None
    if torch.cuda.is_available():
        if isinstance(device, torch.device):
            device_index = (
                torch.cuda.current_device() if device.index is None else int(device.index)
            )
        elif device is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = int(device)
        raw_cap = torch.cuda.get_device_capability(device_index)
        capability = (int(raw_cap[0]), int(raw_cap[1]))
    return {
        "capability": capability,
        "supported_arch_list": arch_list,
        "supported_capabilities": supported_caps,
        "exact_match": capability in supported_caps if capability is not None else False,
    }


def is_flex_attention_supported_on_device(
    device: Optional[torch.device | int] = None,
) -> bool:
    if not FLEX_ATTENTION_AVAILABLE or not torch.cuda.is_available():
        return False
    return bool(get_cuda_arch_support_diagnostics(device)["exact_match"])


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_bits(dtype: torch.dtype) -> Optional[int]:
    try:
        return int(torch.finfo(dtype).bits)
    except TypeError:
        return None


def _resolve_sparse_compute_dtype(
    mode: WayfinderSparseComputeDtype,
    *,
    hidden_dtype: torch.dtype,
    query_dtype: torch.dtype,
) -> torch.dtype:
    if mode == "float32":
        return torch.float32
    if mode == "model":
        return hidden_dtype

    hidden_bits = _dtype_bits(hidden_dtype)
    query_bits = _dtype_bits(query_dtype)
    if hidden_bits is None or query_bits is None:
        return query_dtype
    if hidden_bits < query_bits:
        return hidden_dtype
    return query_dtype


@dataclass
class QwenCUDAWayfinderConfig:
    """Configuration for Qwen3.5 Wayfinder CUDA overlay."""

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
    max_graph_cache_entries: int = 1
    sparse_query_chunk_size: int = 0
    sparse_kv_head_chunk_size: int = 0
    sparse_degree_chunk_size: int = 0
    sparse_chunk_temp_budget_mib: float = 160.0
    sparse_compute_dtype: WayfinderSparseComputeDtype = "auto"
    sparse_precomputed_backend: WayfinderSparsePrecomputedBackend = "auto"
    sparse_trace_dir: Optional[str] = None
    sparse_trace_max_per_layer: int = 0
    sparse_trace_layer_indices: Optional[tuple[int, ...]] = None
    block_size: int = 128
    block_local_window_blocks: int = 1
    block_partner_count: int = 1
    block_sink_blocks: int = 1
    block_partner_rule: WayfinderBlockPartnerRule = "xor"
    block_chunk_size: int = 0  # SDPA block chunk size (0 = all at once)

    def __post_init__(self) -> None:
        if self.path not in {"permute", "sparse", "block_sparse"}:
            raise ValueError(f"Unsupported Wayfinder path: {self.path!r}")
        if self.strategy not in {"random", "regular_partition", "greedy", "online_insertion"}:
            raise ValueError(f"Unsupported Wayfinder strategy: {self.strategy!r}")
        if self.path == "block_sparse" and self.strategy in _DYNAMIC_STRATEGIES:
            raise ValueError(
                "block_sparse currently supports only static strategies "
                "('random', 'regular_partition')"
            )
        if int(self.window) < 0:
            raise ValueError("window must be >= 0")
        if int(self.num_cycles) <= 0:
            raise ValueError("num_cycles must be > 0")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be > 0")
        if int(self.block_local_window_blocks) < 0:
            raise ValueError("block_local_window_blocks must be >= 0")
        if int(self.block_partner_count) < 0:
            raise ValueError("block_partner_count must be >= 0")
        if int(self.block_sink_blocks) < 0:
            raise ValueError("block_sink_blocks must be >= 0")
        if self.block_partner_rule not in {"xor", "bit_reversal", "benes"}:
            raise ValueError(f"Unsupported block_partner_rule: {self.block_partner_rule!r}")
        if int(self.dense_fallback_q_len) < 0:
            raise ValueError("dense_fallback_q_len must be >= 0")
        if int(self.max_graph_cache_entries) <= 0:
            raise ValueError("max_graph_cache_entries must be > 0")
        if int(self.sparse_query_chunk_size) < 0:
            raise ValueError("sparse_query_chunk_size must be >= 0")
        if int(self.sparse_kv_head_chunk_size) < 0:
            raise ValueError("sparse_kv_head_chunk_size must be >= 0")
        if int(self.sparse_degree_chunk_size) < 0:
            raise ValueError("sparse_degree_chunk_size must be >= 0")
        if float(self.sparse_chunk_temp_budget_mib) <= 0.0:
            raise ValueError("sparse_chunk_temp_budget_mib must be > 0")
        if self.sparse_compute_dtype not in {"auto", "model", "float32"}:
            raise ValueError(f"Unsupported sparse_compute_dtype: {self.sparse_compute_dtype!r}")
        if self.sparse_precomputed_backend not in {
            "auto",
            "triton_fused",
            "sdpa",
            "streamed_online_softmax",
            "manual_matmul",
        }:
            raise ValueError(
                f"Unsupported sparse_precomputed_backend: {self.sparse_precomputed_backend!r}"
            )
        if int(self.sparse_trace_max_per_layer) < 0:
            raise ValueError("sparse_trace_max_per_layer must be >= 0")
        if self.sparse_trace_layer_indices is not None:
            self.sparse_trace_layer_indices = tuple(
                sorted({int(layer_idx) for layer_idx in self.sparse_trace_layer_indices})
            )


def _cpu_trace_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone()


@dataclass
class _QwenGraphCache:
    graph: Optional[TopologyGraph]
    neigh_idx: Optional[torch.Tensor]
    edge_type: Optional[torch.Tensor]
    safe_idx: Optional[torch.Tensor] = None
    causal_mask: Optional[torch.Tensor] = None
    perm: list[torch.Tensor] = field(default_factory=list)
    inv_perm: list[torch.Tensor] = field(default_factory=list)
    pi_idx_clamped: list[torch.Tensor] = field(default_factory=list)
    valid_mask: list[torch.Tensor] = field(default_factory=list)
    causal_masks: list[torch.Tensor] = field(default_factory=list)
    # Stacked tensors for batched/flex engines
    perm_stacked: Optional[torch.Tensor] = None  # [H_q, T]
    inv_perm_stacked: Optional[torch.Tensor] = None  # [H_q, T]
    kv_perm_stacked: Optional[torch.Tensor] = None  # [H_kv, T]
    kv_inv_perm_stacked: Optional[torch.Tensor] = None  # [H_kv, T]
    pi_idx_stacked: Optional[torch.Tensor] = None  # [H, T, 2W+1]
    valid_stacked: Optional[torch.Tensor] = None  # [H, T, 2W+1]
    causal_stacked: Optional[torch.Tensor] = None  # [H, T, 2W+1]
    flex_block_mask: Any = None
    block_layout: Optional[BlockHamiltonianLayout] = None
    metrics: Optional[Dict[str, Any]] = None
    cache_key: Optional[tuple[Any, ...]] = None


_QWEN_SHARED_STATIC_GRAPH_CACHE: OrderedDict[tuple[Any, ...], _QwenGraphCache] = OrderedDict()


def clear_shared_qwen_wayfinder_graph_cache() -> int:
    cleared = len(_QWEN_SHARED_STATIC_GRAPH_CACHE)
    _QWEN_SHARED_STATIC_GRAPH_CACHE.clear()
    return cleared


def _safe_neighbor_idx(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    return neigh_idx.clamp(min=0, max=seq_len - 1)


def _causal_neighbor_mask(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    query_positions = torch.arange(seq_len, device=neigh_idx.device, dtype=torch.long).view(
        1, seq_len, 1
    )
    return (neigh_idx >= 0) & (neigh_idx < seq_len) & (neigh_idx <= query_positions)


def _has_required_qwen_attention_attrs(module: nn.Module) -> bool:
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim")
    return all(hasattr(module, name) for name in required)


_QWEN_ROTARY_MODULES_BY_MODEL_TYPE = {
    "qwen3_5": ("transformers.models.qwen3_5.modeling_qwen3_5",),
    "qwen3_5_text": ("transformers.models.qwen3_5.modeling_qwen3_5",),
    "qwen3_5_moe": ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",),
    "qwen3_5_moe_text": ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",),
}


@lru_cache(maxsize=None)
def _load_qwen_apply_rotary_pos_emb(module_name: str):
    module = importlib.import_module(module_name)
    fn = getattr(module, "apply_rotary_pos_emb", None)
    if fn is None:
        raise AttributeError(f"`apply_rotary_pos_emb` not found in {module_name}")
    return fn


def _resolve_qwen_apply_rotary_pos_emb(attn: nn.Module):
    candidates: list[str] = []
    class_module = getattr(attn.__class__, "__module__", None)
    if class_module:
        candidates.append(str(class_module))

    config = getattr(attn, "config", None)
    model_type = getattr(config, "model_type", None)
    candidates.extend(_QWEN_ROTARY_MODULES_BY_MODEL_TYPE.get(str(model_type), ()))
    candidates.extend(
        (
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
            "transformers.models.qwen3_5.modeling_qwen3_5",
        )
    )

    seen: set[str] = set()
    errors: list[str] = []
    for module_name in candidates:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            return _load_qwen_apply_rotary_pos_emb(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    joined = "; ".join(errors) if errors else "no candidates tried"
    raise ImportError(
        "Could not resolve a Qwen rotary helper for attention module "
        f"{attn.__class__.__module__}.{attn.__class__.__name__}. Tried: {joined}"
    )


def _get_decoder_with_layers(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        decoder = model.model.language_model
        if hasattr(decoder, "layers"):
            return decoder
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "layers"):
        return model
    raise ValueError("Could not find a decoder with `.layers` on the supplied model.")


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


def _is_standard_causal_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    q_len: int,
    k_len: int,
) -> bool:
    if attention_mask is None:
        return True
    if attention_mask.dim() == 2:
        return bool(torch.all(attention_mask == 1))
    if attention_mask.dim() != 4:
        return False
    if int(attention_mask.shape[-2]) != q_len or int(attention_mask.shape[-1]) != k_len:
        return False
    if int(attention_mask.shape[1]) != 1:
        return False
    if q_len != k_len:
        return False
    if not attention_mask.is_floating_point():
        return False
    min_dtype = torch.finfo(attention_mask.dtype).min
    # Sampling-based check: verify structure without materializing T×T tensor.
    # A standard causal mask has 0 on/below the diagonal and min_dtype above.
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
    # Check diagonal is zero
    diag = torch.diagonal(m[0, 0])
    if not torch.all(diag == 0):
        return False
    return True


def extract_qkv_from_qwen_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    past_key_values: Optional[Any] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract Q/K/V and output gate from a Qwen3.5 attention module.

    Returns (query_states, key_states, value_states, gate) where gate is the
    sigmoid gate tensor from the gated attention mechanism.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    apply_rotary_pos_emb = _resolve_qwen_apply_rotary_pos_emb(attn)

    query_states, gate = torch.chunk(
        attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        layer_idx = getattr(attn, "layer_idx", None)
        if layer_idx is None:
            raise ValueError("Attention module missing `layer_idx`; cannot update cache.")
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, layer_idx, cache_kwargs
        )

    return query_states, key_states, value_states, gate


class QwenCUDAWayfinderAttention(nn.Module):
    """Wrap a stock Qwen3.5 full-attention layer with Wayfinder prefill attention."""

    def __init__(self, fallback_attention: nn.Module, cfg: QwenCUDAWayfinderConfig):
        super().__init__()
        if not _has_required_qwen_attention_attrs(fallback_attention):
            raise TypeError(
                "fallback_attention does not look like a Qwen3.5 attention module. "
                "Expected q_proj/k_proj/v_proj/o_proj and head_dim."
            )

        self.fallback = fallback_attention
        self.cfg = copy.deepcopy(cfg)

        self.config = getattr(fallback_attention, "config", None)
        self.layer_idx = getattr(fallback_attention, "layer_idx", None)
        self.head_dim = int(getattr(fallback_attention, "head_dim"))
        self.num_key_value_groups = int(getattr(fallback_attention, "num_key_value_groups", 1))
        self.scaling = float(getattr(fallback_attention, "scaling", self.head_dim**-0.5))
        self.attention_dropout = float(getattr(fallback_attention, "attention_dropout", 0.0))
        self.is_causal = getattr(fallback_attention, "is_causal", True)

        # Infer head counts from projection shapes
        q_proj = fallback_attention.q_proj
        k_proj = fallback_attention.k_proj
        # q_proj outputs 2x (query + gate), so divide by 2 then by head_dim
        self.num_heads = q_proj.out_features // (self.head_dim * 2)
        self.num_key_value_heads = k_proj.out_features // self.head_dim
        self.hidden_size = q_proj.in_features

        proj_ref = q_proj.weight
        proj_device = proj_ref.device
        proj_dtype = proj_ref.dtype
        # FP8-quantized weights (e.g. via torchao) report float8 dtypes that cannot
        # be used to initialise new parameters.  Fall back to bfloat16.
        if proj_dtype.is_floating_point and proj_dtype.itemsize < 2:
            proj_dtype = torch.bfloat16

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
            self.edge_type_bias = nn.Parameter(
                torch.zeros((4,), device=proj_device, dtype=torch.float32)
            )
        else:
            self.register_parameter("edge_type_bias", None)

        # Generate one permutation per KV head (not per query head).
        # Each KV-head perm is shared by all query heads in its GQA group,
        # ensuring K/V consistency when flex_attention handles GQA natively.
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

        self._graph_cache: OrderedDict[tuple[Any, ...], _QwenGraphCache] = OrderedDict()
        self._last_graph_cache: Optional[_QwenGraphCache] = None
        self.last_profile: Dict[str, Any] = {}
        self._last_profile_cuda_events: Dict[str, tuple[Any, Any]] = {}
        self._runtime_schedule_bias_vec = torch.zeros((4,), dtype=torch.float32)
        self._sparse_trace_dump_count = 0
        self._last_sparse_trace_path: Optional[str] = None
        self._last_sparse_trace_error: Optional[str] = None

    @property
    def cache_mode(self) -> str:
        return self.topology.cache_mode

    def _uses_wayfinder_block_sparse(self) -> bool:
        return self.cfg.path == "block_sparse"

    def _wayfinder_stage_meta(self, seq_len: int) -> tuple[Optional[int], Optional[int]]:
        if not self._uses_wayfinder_block_sparse():
            return None, None
        num_blocks = (int(seq_len) + int(self.cfg.block_size) - 1) // int(self.cfg.block_size)
        layer_idx = 0 if self.layer_idx is None else int(self.layer_idx)
        stage_idx, stage_count = _wayfinder_stage_meta_standalone(
            num_blocks=num_blocks,
            layer_idx=layer_idx,
            partner_rule=self.cfg.block_partner_rule,
        )
        return stage_idx, stage_count

    def _supports_cached_block_sparse(self, q_len: int, k_len_after: int) -> bool:
        return self._uses_wayfinder_block_sparse() and int(k_len_after) > int(q_len)

    def _resolve_query_positions(
        self,
        *,
        q_len: int,
        cache_len: int,
        cache_position: Optional[torch.LongTensor],
        device: torch.device,
    ) -> torch.Tensor:
        if cache_position is not None:
            return cache_position.to(device=device, dtype=torch.long).view(-1)
        start = int(cache_len)
        return torch.arange(start, start + int(q_len), device=device, dtype=torch.long)

    def _kv_block_neighbors(self, layout: BlockHamiltonianLayout) -> torch.Tensor:
        if int(self.num_key_value_groups) <= 1:
            return layout.block_neighbors
        return layout.block_neighbors[:: int(self.num_key_value_groups)].contiguous()

    def _build_wayfinder_block_sparse_indices(
        self,
        *,
        layout: BlockHamiltonianLayout,
        kv_len: int,
        query_positions: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_block_neighbors = self._kv_block_neighbors(layout)
        hkv = int(kv_block_neighbors.shape[0])
        tq = int(query_positions.numel())
        max_block_degree = int(kv_block_neighbors.shape[-1])
        max_token_degree = max(1, max_block_degree * int(layout.block_size))
        safe_idx = torch.zeros((hkv, tq, max_token_degree), device=device, dtype=torch.long)
        causal_mask = torch.zeros((hkv, tq, max_token_degree), device=device, dtype=torch.bool)

        query_positions_cpu = query_positions.detach().to(device="cpu", dtype=torch.long)
        for head_idx in range(hkv):
            head_rows = kv_block_neighbors[head_idx]
            for q_offset, abs_pos_t in enumerate(query_positions_cpu):
                abs_pos = int(abs_pos_t.item())
                block_idx = min(int(layout.num_blocks) - 1, abs_pos // int(layout.block_size))
                cursor = 0
                row = head_rows[block_idx]
                for block_t in row[row >= 0]:
                    block = int(block_t.item())
                    start = int(block) * int(layout.block_size)
                    end = min(int(kv_len), start + int(layout.block_size))
                    if start > abs_pos:
                        continue
                    if block == block_idx:
                        end = min(end, abs_pos + 1)
                    if end <= start:
                        continue
                    count = end - start
                    safe_idx[head_idx, q_offset, cursor : cursor + count] = torch.arange(
                        start,
                        end,
                        device=device,
                        dtype=torch.long,
                    )
                    causal_mask[head_idx, q_offset, cursor : cursor + count] = True
                    cursor += count
        return safe_idx, causal_mask

    def _block_sparse_graph_source(self, cache: _QwenGraphCache) -> str:
        if cache.graph is not None:
            return cache.graph.source
        if cache.block_layout is not None:
            return f"runtime_block_{cache.block_layout.topology_name}"
        return "runtime_block_sparse"

    def _resolve_engine(self) -> str:
        engine = self.cfg.engine
        if self.cfg.path == "block_sparse":
            if engine == "triton":
                return "triton"
            if engine == "sdpa":
                return "sdpa"
            if engine == "flex":
                return "flex"
            # auto: prefer triton > flex > sdpa (triton requires CUDA device)
            device = self.fallback.q_proj.weight.device
            if TRITON_BLOCK_SPARSE_AVAILABLE and device.type == "cuda":
                return "triton"
            if is_flex_attention_supported_on_device(self.fallback.q_proj.weight.device):
                return "flex"
            return "sdpa"
        if engine == "auto":
            if is_flex_attention_supported_on_device(self.fallback.q_proj.weight.device):
                return "flex"
            return "batched"
        return engine

    def _cache_key(self, seq_len: int, device: torch.device) -> tuple[Any, ...]:
        compiled = None
        if self.cfg.compiled_graph_dir:
            compiled = str(Path(self.cfg.compiled_graph_dir).expanduser().resolve())
        wayfinder_stage_idx, wayfinder_stage_count = self._wayfinder_stage_meta(seq_len)
        if self.cfg.path == "block_sparse":
            # Block-sparse path uses only Wayfinder topology controls;
            # the old hamiltonian-specific fields (strategy, num_cycles, etc.)
            # are irrelevant and omitted from the key.
            strategy = None
            num_cycles = None
            edge_disjoint = None
            regular_num_clusters = None
            window = None
            landmark_stride = None
            seed = None
        else:
            strategy = self.cfg.strategy
            num_cycles = int(self.cfg.num_cycles)
            edge_disjoint = bool(self.cfg.edge_disjoint)
            regular_num_clusters = int(self.cfg.regular_num_clusters)
            window = int(self.cfg.window)
            landmark_stride = (
                None if self.cfg.landmark_stride is None else int(self.cfg.landmark_stride)
            )
            seed = int(self.cfg.seed)
        return (
            int(self.num_heads),
            int(self.num_key_value_heads),
            int(self.num_key_value_groups),
            int(self.head_dim),
            int(seq_len),
            self.cfg.path,
            strategy,
            num_cycles,
            edge_disjoint,
            regular_num_clusters,
            window,
            landmark_stride,
            int(self.cfg.block_size),
            int(self.cfg.block_local_window_blocks),
            int(self.cfg.block_partner_count),
            int(self.cfg.block_sink_blocks),
            self.cfg.block_partner_rule,
            wayfinder_stage_idx,
            wayfinder_stage_count,
            seed,
            self.cfg.engine,
            compiled,
            str(device),
        )

    def _remember_local_cache(
        self,
        cache_key: tuple[Any, ...],
        cache: _QwenGraphCache,
    ) -> None:
        self._graph_cache[cache_key] = cache
        self._graph_cache.move_to_end(cache_key)
        while len(self._graph_cache) > int(self.cfg.max_graph_cache_entries):
            self._graph_cache.popitem(last=False)

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
        if self.cfg.strategy in _DYNAMIC_STRATEGIES and self.routing_proj is not None:
            routing = self.routing_proj(hidden_states[0])
            routing = routing.reshape(
                hidden_states.shape[1], self.num_key_value_heads, self.routing_dim
            ).permute(1, 0, 2)
            routing = routing.detach().float().cpu()
            routing_by_head = [routing[h] for h in range(self.num_key_value_heads)]
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
        need_window_tensors: bool,
    ) -> tuple[list[torch.Tensor], ...]:
        perms_raw = recover_cycle_perms(neigh_idx, edge_type, meta=meta)
        perm_list: list[torch.Tensor] = []
        inv_list: list[torch.Tensor] = []
        pi_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        causal_list: list[torch.Tensor] = []

        offsets = base_idx = None
        if need_window_tensors:
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
                    f"Permutation for head {head} has length {perm_t.numel()}, expected {seq_len}"
                )
            inv_perm = torch.empty_like(perm_t)
            inv_perm[perm_t] = torch.arange(seq_len, device=device, dtype=torch.long)

            perm_list.append(perm_t)
            inv_list.append(inv_perm)
            if need_window_tensors:
                pi_idx = base_idx + offsets.view(1, -1)
                valid = (pi_idx >= 0) & (pi_idx < seq_len)
                pi_idx_clamped = pi_idx.clamp(0, seq_len - 1)
                neigh_orig = perm_t[pi_idx_clamped]
                query_orig = perm_t.view(seq_len, 1)
                causal_h = neigh_orig <= query_orig
                pi_list.append(pi_idx_clamped)
                valid_list.append(valid)
                causal_list.append(causal_h)

        return perm_list, inv_list, pi_list, valid_list, causal_list

    def _build_cache(self, hidden_states: torch.Tensor, seq_len: int) -> _QwenGraphCache:
        device = hidden_states.device
        engine = self._resolve_engine()
        graph: Optional[TopologyGraph] = None
        block_layout: Optional[BlockHamiltonianLayout] = None

        neigh_idx: Optional[torch.Tensor] = None
        edge_type: Optional[torch.Tensor] = None
        kv_perm_list: list[torch.Tensor] = []
        kv_inv_list: list[torch.Tensor] = []
        perm_list = inv_list = pi_list = valid_list = causal_list = []
        if self.cfg.path == "block_sparse":
            block_layout = build_block_wayfinder_layout(
                seq_len=int(seq_len),
                block_size=int(self.cfg.block_size),
                num_key_value_heads=int(self.num_key_value_heads),
                num_key_value_groups=int(self.num_key_value_groups),
                layer_idx=0 if self.layer_idx is None else int(self.layer_idx),
                local_window_blocks=int(self.cfg.block_local_window_blocks),
                sink_count=int(self.cfg.block_sink_blocks),
                partner_count=int(self.cfg.block_partner_count),
                partner_rule=self.cfg.block_partner_rule,
                device=device,
            )
        else:
            graph = self._build_graph(hidden_states, seq_len)
            abi = graph.abi
        if self.cfg.path == "permute" and graph is not None:
            neigh_idx_kv = torch.as_tensor(abi.neigh_idx, dtype=torch.long, device=device)
            edge_type_kv = torch.as_tensor(abi.edge_type, dtype=torch.uint8, device=device)
            kv_perm_list, kv_inv_list, _pil, _vl, _cl = self._extract_cycle_perms(
                neigh_idx_kv,
                edge_type_kv,
                meta=getattr(abi, "meta", None),
                seq_len=seq_len,
                device=device,
                need_window_tensors=(engine == "batched"),
            )
            # Replicate each KV-head entry for its query-head group
            g = self.num_key_value_groups
            perm_list = [p for p in kv_perm_list for _ in range(g)]
            inv_list = [p for p in kv_inv_list for _ in range(g)]
            pi_list = [p for p in _pil for _ in range(g)]
            valid_list = [p for p in _vl for _ in range(g)]
            causal_list = [p for p in _cl for _ in range(g)]
        elif self.cfg.path == "sparse" and graph is not None:
            neigh_idx_kv = torch.as_tensor(abi.neigh_idx, dtype=torch.long, device=device)
            edge_type_kv = torch.as_tensor(abi.edge_type, dtype=torch.uint8, device=device)
            neigh_idx = neigh_idx_kv
            edge_type = edge_type_kv

        safe_idx = causal_mask = None
        if self.cfg.path == "sparse" and neigh_idx is not None:
            safe_idx = _safe_neighbor_idx(neigh_idx, seq_len)
            causal_mask = _causal_neighbor_mask(neigh_idx, seq_len)

        # Build stacked tensors for batched/flex engines
        perm_stacked = inv_perm_stacked = None
        kv_perm_stacked = kv_inv_perm_stacked = None
        pi_idx_stacked = valid_stacked = causal_stacked = None
        flex_block_mask_val = None

        if self.cfg.path == "permute" and perm_list:
            # kv_perm_list is H_kv; perm_list is already H_q (replicated)
            kv_perm_stacked = torch.stack(kv_perm_list)  # [H_kv, T]
            kv_inv_perm_stacked = torch.stack(kv_inv_list)  # [H_kv, T]
            perm_stacked = torch.stack(perm_list)  # [H_q, T]
            inv_perm_stacked = torch.stack(inv_list)  # [H_q, T]
            if pi_list:
                pi_idx_stacked = torch.stack(pi_list)  # [H_q, T, 2W+1]
                valid_stacked = torch.stack(valid_list)  # [H_q, T, 2W+1]
                causal_stacked = torch.stack(causal_list)  # [H_q, T, 2W+1]

            if engine == "flex" and FLEX_ATTENTION_AVAILABLE:
                try:
                    flex_block_mask_val = build_flex_block_mask(
                        perm_stacked,
                        window=int(self.cfg.window),
                        B=1,
                        device=device,
                    )
                except Exception:
                    pass
        elif self.cfg.path == "block_sparse" and block_layout is not None:
            if engine not in {"sdpa", "triton"} and FLEX_ATTENTION_AVAILABLE:
                try:
                    flex_block_mask_val = build_block_hamiltonian_mask(
                        block_layout,
                        device=device,
                    )
                except Exception:
                    pass

        return _QwenGraphCache(
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
            block_layout=block_layout,
            metrics=None,
            cache_key=self._cache_key(seq_len, device),
        )

    def _get_or_build_cache(
        self, hidden_states: torch.Tensor, seq_len: int
    ) -> tuple[_QwenGraphCache, bool]:
        cache_key = self._cache_key(seq_len, hidden_states.device)
        if self.cache_mode == "static":
            cached = self._graph_cache.get(cache_key)
            if cached is not None:
                self._graph_cache.move_to_end(cache_key)
                return cached, True
            shared = _QWEN_SHARED_STATIC_GRAPH_CACHE.get(cache_key)
            if shared is not None:
                _QWEN_SHARED_STATIC_GRAPH_CACHE.move_to_end(cache_key)
                self._remember_local_cache(cache_key, shared)
                return shared, True
        cache = self._build_cache(hidden_states, seq_len)
        if self.cache_mode == "static":
            _QWEN_SHARED_STATIC_GRAPH_CACHE[cache_key] = cache
            _QWEN_SHARED_STATIC_GRAPH_CACHE.move_to_end(cache_key)
            self._remember_local_cache(cache_key, cache)
        return cache, False

    def ensure_last_graph_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.compute_graph_metrics:
            return None
        cache = self._last_graph_cache
        if cache is None:
            return None
        if cache.graph is None:
            return None
        if cache.metrics is None:
            try:
                cache.metrics = graph_metrics(cache.graph.abi)
            except Exception:
                cache.metrics = None
        if self.last_profile.get("mode") == "wayfinder":
            self.last_profile["graph_metrics"] = cache.metrics
        return cache.metrics

    def snapshot_last_profile(self, *, sync: bool = True) -> Dict[str, Any]:
        profile = dict(self.last_profile)
        if sync and self._last_profile_cuda_events and torch.cuda.is_available():
            torch.cuda.synchronize()
        for metric_name, event_pair in self._last_profile_cuda_events.items():
            try:
                start_event, end_event = event_pair
                profile[metric_name] = float(start_event.elapsed_time(end_event))
            except Exception:
                profile[metric_name] = None
        if profile.get("sparse_contraction_cuda_ms") is not None:
            profile["attn_kernel_ms"] = float(profile["sparse_contraction_cuda_ms"])
        if profile.get("block_sparse_cuda_ms") is not None:
            profile["attn_kernel_ms"] = float(profile["block_sparse_cuda_ms"])
        return profile

    def _should_dump_sparse_trace(self) -> bool:
        if self.cfg.path != "sparse":
            return False
        if not self.cfg.sparse_trace_dir:
            return False
        if int(self.cfg.sparse_trace_max_per_layer) <= 0:
            return False
        if self._sparse_trace_dump_count >= int(self.cfg.sparse_trace_max_per_layer):
            return False
        if self.cfg.sparse_trace_layer_indices is not None:
            if (
                self.layer_idx is None
                or int(self.layer_idx) not in self.cfg.sparse_trace_layer_indices
            ):
                return False
        return True

    def _dump_sparse_trace(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache: _QwenGraphCache,
        cache_hit: bool,
        scale: float,
        bias: torch.Tensor | None,
    ) -> Optional[str]:
        if not self._should_dump_sparse_trace():
            return self._last_sparse_trace_path
        if cache.safe_idx is None or cache.causal_mask is None:
            return self._last_sparse_trace_path

        try:
            batch_size = int(query_states.shape[0])
            safe_idx = cache.safe_idx
            if safe_idx.ndim == 3:
                safe_idx = safe_idx.unsqueeze(0).expand(batch_size, -1, -1, -1)

            causal_mask = cache.causal_mask
            if causal_mask.ndim == 3:
                causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

            edge_type = cache.edge_type
            if edge_type is not None and edge_type.ndim == 3:
                edge_type = edge_type.unsqueeze(0).expand(batch_size, -1, -1, -1)

            trace_dir = Path(self.cfg.sparse_trace_dir).expanduser().resolve()
            trace_dir.mkdir(parents=True, exist_ok=True)
            layer_name = "unknown" if self.layer_idx is None else f"{int(self.layer_idx):02d}"
            trace_name = (
                f"qwen_sparse_trace_layer{layer_name}_"
                f"t{int(query_states.shape[2])}_"
                f"d{int(safe_idx.shape[-1])}_"
                f"n{self._sparse_trace_dump_count:02d}_"
                f"{time.time_ns()}.pt"
            )
            trace_path = trace_dir / trace_name
            payload = {
                "format_version": 1,
                "source": "wayfinder_qwen_torch",
                "layer_idx": self.layer_idx,
                "seq_len": int(query_states.shape[2]),
                "degree": int(safe_idx.shape[-1]),
                "path": self.cfg.path,
                "strategy": self.cfg.strategy,
                "window": int(self.cfg.window),
                "landmark_stride": (
                    None if self.cfg.landmark_stride is None else int(self.cfg.landmark_stride)
                ),
                "num_cycles": int(self.cfg.num_cycles),
                "graph_source": cache.graph.source,
                "graph_cache_hit": bool(cache_hit),
                "num_key_value_groups": int(self.num_key_value_groups),
                "scale": float(scale),
                "query_dtype": _dtype_name(query_states.dtype),
                "key_dtype": _dtype_name(key_states.dtype),
                "value_dtype": _dtype_name(value_states.dtype),
                "q": _cpu_trace_tensor(query_states),
                "k": _cpu_trace_tensor(key_states),
                "v": _cpu_trace_tensor(value_states),
                "safe_idx": _cpu_trace_tensor(safe_idx),
                "causal_mask": _cpu_trace_tensor(causal_mask),
                "bias": None if bias is None else _cpu_trace_tensor(bias),
                "edge_type": None if edge_type is None else _cpu_trace_tensor(edge_type),
            }
            torch.save(payload, trace_path)
            self._sparse_trace_dump_count += 1
            self._last_sparse_trace_path = str(trace_path)
            self._last_sparse_trace_error = None
        except Exception as exc:
            self._last_sparse_trace_error = str(exc)
        return self._last_sparse_trace_path

    def _fallback_reason(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        k_len_after: int,
        output_attentions: bool,
    ) -> Optional[str]:
        if self.training:
            return "training"
        cached_block_sparse = self._supports_cached_block_sparse(q_len, k_len_after)
        resolved = self._resolve_engine()
        if (
            self.cfg.path == "block_sparse"
            and resolved not in {"flex", "sdpa", "triton"}
            and not cached_block_sparse
        ):
            return "block_sparse_no_engine"
        if self.layer_idx is None:
            return "missing_layer_idx"
        if self.cfg.fallback_on_output_attentions and output_attentions:
            return "output_attentions"
        if q_len <= int(self.cfg.dense_fallback_q_len) and not cached_block_sparse:
            return "short_query"
        if k_len_after != q_len and not cached_block_sparse:
            return "cached_kv"
        if self.cfg.fallback_on_mask and not _is_standard_causal_mask(
            attention_mask,
            q_len=q_len,
            k_len=k_len_after,
        ):
            return "attention_mask"
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            self._last_graph_cache = None
            self._last_profile_cuda_events = {}
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
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        t0 = time.perf_counter()

        query_states, key_states, value_states, gate = extract_qkv_from_qwen_attention(
            self.fallback,
            hidden_states,
            position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        engine = self._resolve_engine()
        cached_block_sparse = self._supports_cached_block_sparse(q_len, k_len_after)

        # flex_attention handles GQA natively; the gather-based sparse path
        # stays on KV heads and permute batched/legacy still expand to query heads.
        use_native_gqa = (
            self.cfg.path == "permute" and engine == "flex"
        ) or self.cfg.path == "block_sparse"
        if self.cfg.path == "permute" and not use_native_gqa:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        t_graph = time.perf_counter()
        cache_seq_len = int(k_len_after) if cached_block_sparse else int(q_len)
        cache, cache_hit = self._get_or_build_cache(hidden_states, cache_seq_len)
        self._last_graph_cache = cache
        graph_build_ms = float((time.perf_counter() - t_graph) * 1000.0)

        t_attn = time.perf_counter()
        sparse_cuda_events: tuple[Any, Any] | None = None
        block_sparse_cuda_events: tuple[Any, Any] | None = None
        if self.cfg.path == "sparse":
            sparse_profile: Dict[str, Any] = {}
            sparse_profile["sparse_query_input_dtype"] = _dtype_name(query_states.dtype)
            sparse_profile["sparse_key_input_dtype"] = _dtype_name(key_states.dtype)
            sparse_profile["sparse_value_input_dtype"] = _dtype_name(value_states.dtype)
            sparse_dtype = _resolve_sparse_compute_dtype(
                self.cfg.sparse_compute_dtype,
                hidden_dtype=hidden_states.dtype,
                query_dtype=query_states.dtype,
            )
            query_states = query_states.to(dtype=sparse_dtype)
            key_states = key_states.to(dtype=sparse_dtype)
            value_states = value_states.to(dtype=sparse_dtype)
            sparse_profile["sparse_compute_dtype"] = _dtype_name(sparse_dtype)
            trace_bias = (
                build_sparse_edge_bias_tensor(
                    cache.edge_type.unsqueeze(0).expand(query_states.shape[0], -1, -1, -1)
                    if cache.edge_type is not None and cache.edge_type.ndim == 3
                    else cache.edge_type,
                    edge_type_bias=self.edge_type_bias,
                    edge_type_bias_offset=None,
                    device=query_states.device,
                )
                if cache.edge_type is not None
                else None
            )
            dumped_trace_path = self._dump_sparse_trace(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                cache=cache,
                cache_hit=cache_hit,
                scale=self.scaling,
                bias=trace_bias,
            )
            if dumped_trace_path is not None:
                sparse_profile["sparse_trace_path"] = dumped_trace_path
            if self._last_sparse_trace_error is not None:
                sparse_profile["sparse_trace_error"] = self._last_sparse_trace_error
            if query_states.is_cuda:
                sparse_start_event = torch.cuda.Event(enable_timing=True)
                sparse_end_event = torch.cuda.Event(enable_timing=True)
                sparse_start_event.record()
            wayfinder_out, _ = sparse_row_attention_gqa_chunked(
                query_states,
                key_states,
                value_states,
                neigh_idx=cache.neigh_idx,
                num_key_value_groups=self.num_key_value_groups,
                edge_type=cache.edge_type,
                return_weights=False,
                precomputed_safe_idx=cache.safe_idx,
                precomputed_causal_mask=cache.causal_mask,
                edge_type_bias=self.edge_type_bias,
                query_chunk_size=int(self.cfg.sparse_query_chunk_size),
                kv_head_chunk_size=int(self.cfg.sparse_kv_head_chunk_size),
                degree_chunk_size=int(self.cfg.sparse_degree_chunk_size),
                chunk_temp_budget_mib=float(self.cfg.sparse_chunk_temp_budget_mib),
                chunk_profile=sparse_profile,
            )
            if query_states.is_cuda:
                sparse_end_event.record()
                sparse_cuda_events = (sparse_start_event, sparse_end_event)
        elif self.cfg.path == "block_sparse":
            block_profile = {
                "block_sparse_backend": "flex_attention",
                "block_sparse_topology": "wayfinder",
                "block_sparse_block_size": int(self.cfg.block_size),
                "block_sparse_num_blocks": (
                    None if cache.block_layout is None else int(cache.block_layout.num_blocks)
                ),
                "block_sparse_neighbor_blocks": (
                    None
                    if cache.block_layout is None
                    else int(cache.block_layout.block_neighbors.shape[-1])
                ),
                "block_sparse_landmark_blocks": (
                    []
                    if cache.block_layout is None
                    else [int(block_idx) for block_idx in cache.block_layout.landmark_blocks]
                ),
                "block_sparse_sink_blocks": (
                    []
                    if cache.block_layout is None
                    else [int(block_idx) for block_idx in cache.block_layout.sink_blocks]
                ),
                "block_sparse_num_cycles": int(self.cfg.num_cycles),
                "block_sparse_stage": (
                    None if cache.block_layout is None else int(cache.block_layout.stage_idx)
                ),
                "block_sparse_stage_count": (
                    None if cache.block_layout is None else int(cache.block_layout.stage_count)
                ),
                "block_local_window_blocks": int(self.cfg.block_local_window_blocks),
                "block_partner_count": int(self.cfg.block_partner_count),
                "block_partner_rule": self.cfg.block_partner_rule,
            }
            if cached_block_sparse:
                if cache.block_layout is None:
                    raise RuntimeError("Cached Wayfinder block_sparse path requires a block layout")
                sparse_dtype = _resolve_sparse_compute_dtype(
                    self.cfg.sparse_compute_dtype,
                    hidden_dtype=hidden_states.dtype,
                    query_dtype=query_states.dtype,
                )
                query_states = query_states.to(dtype=sparse_dtype)
                key_states = key_states.to(dtype=sparse_dtype)
                value_states = value_states.to(dtype=sparse_dtype)
                query_positions = self._resolve_query_positions(
                    q_len=q_len,
                    cache_len=cache_len,
                    cache_position=cache_position,
                    device=query_states.device,
                )
                safe_idx, causal_mask = self._build_wayfinder_block_sparse_indices(
                    layout=cache.block_layout,
                    kv_len=k_len_after,
                    query_positions=query_positions,
                    device=query_states.device,
                )
                block_profile["block_sparse_backend"] = "sparse_gqa_precomputed"
                block_profile["sparse_compute_dtype"] = _dtype_name(sparse_dtype)
                wayfinder_out, _ = sparse_row_attention_gqa_precomputed(
                    query_states,
                    key_states,
                    value_states,
                    safe_idx=safe_idx,
                    causal_mask=causal_mask,
                    num_key_value_groups=self.num_key_value_groups,
                    return_weights=False,
                    query_chunk_size=int(self.cfg.sparse_query_chunk_size),
                    kv_head_chunk_size=int(self.cfg.sparse_kv_head_chunk_size),
                    degree_chunk_size=int(self.cfg.sparse_degree_chunk_size),
                    chunk_temp_budget_mib=float(self.cfg.sparse_chunk_temp_budget_mib),
                    contraction_backend_override=self.cfg.sparse_precomputed_backend,
                    chunk_profile=block_profile,
                )
            elif engine == "sdpa":
                if cache.block_layout is None:
                    raise RuntimeError("block_sparse sdpa path requires a block layout")
                if query_states.is_cuda:
                    block_sparse_start_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_end_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_start_event.record()
                block_profile["block_sparse_backend"] = "sdpa"
                block_profile["block_chunk_size"] = int(self.cfg.block_chunk_size)
                wayfinder_out = wayfinder_block_sparse_sdpa_attention(
                    query_states,
                    key_states,
                    value_states,
                    layout=cache.block_layout,
                    block_chunk_size=int(self.cfg.block_chunk_size),
                )
                if query_states.is_cuda:
                    block_sparse_end_event.record()
                    block_sparse_cuda_events = (block_sparse_start_event, block_sparse_end_event)
            elif engine == "triton":
                if cache.block_layout is None:
                    raise RuntimeError("block_sparse triton path requires a block layout")
                if query_states.is_cuda:
                    block_sparse_start_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_end_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_start_event.record()
                block_profile["block_sparse_backend"] = "triton"
                wayfinder_out = triton_block_sparse_attention(
                    query_states,
                    key_states,
                    value_states,
                    block_neighbors=cache.block_layout.block_neighbors[
                        : self.num_key_value_heads
                    ].to(device=query_states.device),
                    block_size=int(self.cfg.block_size),
                    num_key_value_groups=self.num_key_value_groups,
                )
                if query_states.is_cuda:
                    block_sparse_end_event.record()
                    block_sparse_cuda_events = (block_sparse_start_event, block_sparse_end_event)
            else:
                if cache.flex_block_mask is None:
                    raise RuntimeError("block_sparse flex path requires a compiled flex block mask")
                if query_states.is_cuda:
                    block_sparse_start_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_end_event = torch.cuda.Event(enable_timing=True)
                    block_sparse_start_event.record()
                wayfinder_out, new_bm = wayfinder_block_sparse_attention(
                    query_states,
                    key_states,
                    value_states,
                    block_mask=cache.flex_block_mask,
                )
                if cache.flex_block_mask is None:
                    cache.flex_block_mask = new_bm
                if query_states.is_cuda:
                    block_sparse_end_event.record()
                    block_sparse_cuda_events = (block_sparse_start_event, block_sparse_end_event)
        elif self.cfg.path == "permute":
            if engine == "flex" and cache.perm_stacked is not None:
                wayfinder_out, new_bm = wayfinder_flex_attention(
                    query_states,
                    key_states,
                    value_states,
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
                    query_states,
                    key_states,
                    value_states,
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

                wayfinder_out, _w, _p_ms, _a_ms = wayfinder_permute_window_attention(
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
                    window_drop_prob=0.0,
                    training=False,
                )
        else:
            raise ValueError(f"Unknown Wayfinder path: {self.cfg.path!r}")

        input_shape = hidden_states.shape[:-1]
        attn_output = wayfinder_out.transpose(1, 2).contiguous().view(*input_shape, -1)

        # Apply Qwen3.5 output gate
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.fallback.o_proj(attn_output)

        attn_kernel_ms_host = float((time.perf_counter() - t_attn) * 1000.0)

        self.last_profile = {
            "mode": "wayfinder",
            "reason": None,
            "layer_idx": self.layer_idx,
            "seq_len": q_len,
            "kv_len": k_len_after,
            "path": self.cfg.path,
            "engine": engine,
            "strategy": self.cfg.strategy,
            "graph_source": self._block_sparse_graph_source(cache),
            "graph_cache_hit": bool(cache_hit),
            "graph_metrics": cache.metrics,
            "graph_build_ms": graph_build_ms,
            "attn_kernel_ms": attn_kernel_ms_host,
            "attn_kernel_ms_host": attn_kernel_ms_host,
            "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        self._last_profile_cuda_events = {}
        if self.cfg.path == "sparse":
            if sparse_cuda_events is not None:
                self._last_profile_cuda_events["sparse_contraction_cuda_ms"] = sparse_cuda_events
            self.last_profile.update(sparse_profile)
        elif self.cfg.path == "block_sparse":
            if block_sparse_cuda_events is not None:
                self._last_profile_cuda_events["block_sparse_cuda_ms"] = block_sparse_cuda_events
            self.last_profile.update(block_profile)
        return attn_output, None


def restore_qwen_dense_attention(model: nn.Module) -> list[int]:
    """Remove Wayfinder wrappers and restore stock Qwen attention in-place.

    Returns the layer indices that were restored.
    """
    clear_shared_qwen_wayfinder_graph_cache()
    decoder = _get_decoder_with_layers(model)
    restored: list[int] = []
    for idx, layer in enumerate(decoder.layers):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, QwenCUDAWayfinderAttention):
            graph_cache = getattr(attn, "_graph_cache", None)
            if isinstance(graph_cache, dict):
                graph_cache.clear()
            attn._last_graph_cache = None
            layer.self_attn = attn.fallback
            restored.append(idx)
    return restored


def iter_qwen_wayfinder_layers(model: nn.Module) -> Iterable[QwenCUDAWayfinderAttention]:
    decoder = _get_decoder_with_layers(model)
    for layer in decoder.layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, QwenCUDAWayfinderAttention):
            yield attn


def swap_qwen_attention_with_wayfinder_cuda(
    model: nn.Module,
    cfg: Optional[QwenCUDAWayfinderConfig] = None,
    *,
    layer_indices: Optional[Sequence[int]] = None,
) -> list[int]:
    """Replace Qwen3.5 full-attention layers with Wayfinder wrappers in-place.

    Only layers with ``layer_type == "full_attention"`` are swapped.

    Returns the layer indices that were replaced.
    """
    config = copy.deepcopy(cfg) if cfg is not None else QwenCUDAWayfinderConfig()
    decoder = _get_decoder_with_layers(model)
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)

    replaced: list[int] = []
    for idx, layer in enumerate(decoder.layers):
        if selected is not None and idx not in selected:
            continue
        if getattr(layer, "layer_type", None) != "full_attention":
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if isinstance(attn, QwenCUDAWayfinderAttention):
            replaced.append(idx)
            continue
        if not _has_required_qwen_attention_attrs(attn):
            continue

        if getattr(attn, "layer_idx", None) is None:
            setattr(attn, "layer_idx", idx)

        wrapped = QwenCUDAWayfinderAttention(attn, config)
        wrapped.train(attn.training)
        layer.self_attn = wrapped
        replaced.append(idx)

    return replaced


__all__ = [
    "QwenCUDAWayfinderConfig",
    "QwenCUDAWayfinderAttention",
    "clear_shared_qwen_wayfinder_graph_cache",
    "extract_qkv_from_qwen_attention",
    "iter_qwen_wayfinder_layers",
    "restore_qwen_dense_attention",
    "swap_qwen_attention_with_wayfinder_cuda",
]
