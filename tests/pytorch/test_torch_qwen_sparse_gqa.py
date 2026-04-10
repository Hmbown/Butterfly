from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bna.torch.attention_wayfinder_sparse import (
    _gather_kv_head_block_chunk,
    _resolve_sparse_gqa_chunking,
    sparse_row_attention,
    sparse_row_attention_gqa_chunked,
    sparse_row_attention_gqa_precomputed,
)
from bna.torch.attention_wayfinder_permute import (
    build_block_wayfinder_layout,
)
from bna.torch.bench_utils import _repeat_kv, stable_masked_softmax


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(module_name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeQwenAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        layer_idx: int,
        num_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 6,
    ) -> None:
        super().__init__()
        hidden_size = num_heads * head_dim
        self.config = types.SimpleNamespace()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.scaling = float(head_dim ** -0.5)
        self.attention_dropout = 0.0
        self.is_causal = True
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim * 2,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.o_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.q_norm = _IdentityNorm()
        self.k_norm = _IdentityNorm()


def _fake_extract_qkv(
    attn: _FakeQwenAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    past_key_values=None,
    cache_position=None,
):
    del position_embeddings, past_key_values, cache_position
    b, t, c = hidden_states.shape
    hq = attn.q_proj.out_features // (attn.head_dim * 2)
    hkv = attn.k_proj.out_features // attn.head_dim
    q = hidden_states.view(b, t, hq, attn.head_dim).transpose(1, 2).contiguous()
    kv_base = hidden_states[:, :, : hkv * attn.head_dim]
    k = kv_base.view(b, t, hkv, attn.head_dim).transpose(1, 2).contiguous()
    v = (kv_base + 0.25).view(b, t, hkv, attn.head_dim).transpose(1, 2).contiguous()
    gate = torch.zeros((b, t, c), device=hidden_states.device, dtype=hidden_states.dtype)
    return q, k, v, gate


def _make_sparse_graph(
    *,
    device: torch.device,
    num_heads: int,
    seq_len: int,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    neigh = torch.full((num_heads, seq_len, degree), -1, device=device, dtype=torch.long)
    edge = torch.zeros((num_heads, seq_len, degree), device=device, dtype=torch.uint8)
    for h in range(num_heads):
        for i in range(seq_len):
            cursor = 0
            for j in range(max(0, i - 2), i + 1):
                if cursor >= degree:
                    break
                neigh[h, i, cursor] = j
                edge[h, i, cursor] = (cursor % 3) + 1
                cursor += 1
            if i + 1 < seq_len and cursor < degree:
                neigh[h, i, cursor] = i + 1
                edge[h, i, cursor] = 1
    return neigh, edge


def _make_wrapped_layers(
    qwen_torch,
    *,
    device: torch.device,
    cfg,
) -> tuple[object, object]:
    layer0 = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=0),
        cfg,
    )
    layer1 = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=1),
        cfg,
    )
    layer0.eval()
    layer1.eval()
    return layer0, layer1


def test_qwen_swap_and_qkv_extract_support_qwen35_moe_attention() -> None:
    import bna.integrations.qwen_torch as qwen_torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    model = Qwen3_5MoeForCausalLM(
        Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts_per_tok=2,
            num_experts=4,
            layer_types=["full_attention"],
        )
    )
    cfg = qwen_torch.QwenCUDAWayfinderConfig(path="sparse", strategy="random")

    replaced = qwen_torch.swap_qwen_attention_with_wayfinder_cuda(model, cfg)

    assert replaced == [0]
    wrapped = model.model.layers[0].self_attn
    assert isinstance(wrapped, qwen_torch.QwenCUDAWayfinderAttention)
    assert wrapped.num_key_value_heads == 2
    assert wrapped.num_key_value_groups == 2

    hidden_states = torch.randn(1, 5, wrapped.hidden_size, dtype=torch.float32)
    position_embeddings = (
        torch.ones(1, 5, wrapped.head_dim, dtype=torch.float32),
        torch.zeros(1, 5, wrapped.head_dim, dtype=torch.float32),
    )
    q, k, v, gate = qwen_torch.extract_qkv_from_qwen_attention(
        wrapped.fallback,
        hidden_states,
        position_embeddings,
    )

    assert q.shape == (1, 4, 5, 16)
    assert k.shape == (1, 2, 5, 16)
    assert v.shape == (1, 2, 5, 16)
    assert gate.shape == (1, 5, 64)


def test_sparse_gqa_chunked_matches_repeated_kv_reference(device: torch.device) -> None:
    b, hkv, groups, t, dh = 2, 2, 3, 7, 4
    hq = hkv * groups
    degree = 4

    q = torch.randn(b, hq, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    neigh_kv, edge_kv = _make_sparse_graph(
        device=device,
        num_heads=hkv,
        seq_len=t,
        degree=degree,
    )
    bias = torch.tensor([0.1, -0.2, 0.05, 0.0], device=device, dtype=torch.float32)
    bias_offset = torch.tensor([-0.05, 0.1, 0.0, 0.0], device=device, dtype=torch.float32)

    out_gqa, weights_gqa = sparse_row_attention_gqa_chunked(
        q,
        k,
        v,
        neigh_idx=neigh_kv,
        edge_type=edge_kv,
        num_key_value_groups=groups,
        return_weights=True,
        edge_type_bias=bias,
        edge_type_bias_offset=bias_offset,
        query_chunk_size=3,
        kv_head_chunk_size=1,
    )

    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    neigh_rep = neigh_kv.repeat_interleave(groups, dim=0)
    edge_rep = edge_kv.repeat_interleave(groups, dim=0)
    out_ref, weights_ref = sparse_row_attention(
        q,
        k_rep,
        v_rep,
        neigh_idx=neigh_rep,
        edge_type=edge_rep,
        return_weights=True,
        edge_type_bias=bias,
        edge_type_bias_offset=bias_offset,
    )

    assert torch.allclose(out_gqa, out_ref, atol=1e-5, rtol=1e-5)
    assert weights_gqa is not None and weights_ref is not None
    assert torch.allclose(weights_gqa, weights_ref, atol=1e-5, rtol=1e-5)


def test_sparse_gqa_chunked_auto_chunking_matches_repeated_kv_reference(
    device: torch.device,
) -> None:
    b, hkv, groups, t, dh = 1, 2, 2, 9, 4
    hq = hkv * groups
    degree = 5

    q = torch.randn(b, hq, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    neigh_kv, edge_kv = _make_sparse_graph(
        device=device,
        num_heads=hkv,
        seq_len=t,
        degree=degree,
    )
    chunk_profile = {}

    out_gqa, weights_gqa = sparse_row_attention_gqa_chunked(
        q,
        k,
        v,
        neigh_idx=neigh_kv,
        edge_type=edge_kv,
        num_key_value_groups=groups,
        return_weights=True,
        chunk_temp_budget_mib=2.0,
        chunk_profile=chunk_profile,
    )

    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    neigh_rep = neigh_kv.repeat_interleave(groups, dim=0)
    edge_rep = edge_kv.repeat_interleave(groups, dim=0)
    out_ref, weights_ref = sparse_row_attention(
        q,
        k_rep,
        v_rep,
        neigh_idx=neigh_rep,
        edge_type=edge_rep,
        return_weights=True,
    )

    assert torch.allclose(out_gqa, out_ref, atol=1e-5, rtol=1e-5)
    assert weights_gqa is not None and weights_ref is not None
    assert torch.allclose(weights_gqa, weights_ref, atol=1e-5, rtol=1e-5)
    assert chunk_profile["sparse_chunk_mode"] == "auto"
    assert chunk_profile["sparse_query_chunk_size"] == t
    assert chunk_profile["sparse_kv_head_chunk_size"] == hkv
    assert chunk_profile["sparse_num_query_chunks"] == 1
    assert chunk_profile["sparse_num_head_blocks"] == 1
    assert chunk_profile["sparse_num_degree_blocks"] == 1
    assert chunk_profile["sparse_degree_chunk_size"] == degree
    assert chunk_profile["sparse_streamed_degree"] is False
    assert chunk_profile["sparse_chunk_budget_exceeded"] is False


def test_wayfinder_block_layout_is_causal_and_stage_scheduled(device: torch.device) -> None:
    layout = build_block_wayfinder_layout(
        seq_len=1024,
        block_size=128,
        num_key_value_heads=2,
        num_key_value_groups=2,
        layer_idx=1,
        local_window_blocks=2,
        partner_count=2,
        sink_count=1,
        partner_rule="xor",
        device=device,
    )

    assert layout.num_blocks == 8
    assert layout.stage_count == 3
    assert layout.stage_idx == 1
    assert tuple(layout.sink_blocks) == (0,)
    assert layout.block_neighbors.shape[:2] == (4, 8)
    assert layout.topology_name == "butterfly"

    row = layout.block_neighbors[0, 6]
    valid = row[row >= 0].tolist()
    assert valid == [6, 5, 4, 2, 0]
    assert all(block_idx <= 6 for block_idx in valid)

    head1_row = layout.block_neighbors[1, 6]
    assert torch.equal(row, head1_row)


def test_sparse_gqa_precomputed_supports_rectangular_cached_decode(device: torch.device) -> None:
    b, hkv, groups, tq, tk, dh = 1, 2, 2, 3, 9, 4
    hq = hkv * groups
    q = torch.randn(b, hq, tq, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, hkv, tk, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, hkv, tk, dh, device=device, dtype=torch.float32)

    safe_idx = torch.tensor(
        [
            [[0, 1, 2, 0], [0, 2, 4, 5], [1, 3, 6, 8]],
            [[0, 1, 2, 0], [0, 2, 4, 5], [1, 3, 6, 8]],
        ],
        device=device,
        dtype=torch.long,
    )
    causal_mask = torch.tensor(
        [
            [[True, True, True, False], [True, True, True, True], [True, True, True, True]],
            [[True, True, True, False], [True, True, True, True], [True, True, True, True]],
        ],
        device=device,
        dtype=torch.bool,
    )

    out, _ = sparse_row_attention_gqa_precomputed(
        q,
        k,
        v,
        safe_idx=safe_idx,
        causal_mask=causal_mask,
        num_key_value_groups=groups,
        query_chunk_size=2,
        kv_head_chunk_size=1,
    )

    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    safe_idx_rep = safe_idx.repeat_interleave(groups, dim=0)
    mask_rep = causal_mask.repeat_interleave(groups, dim=0)
    scores = torch.sum(
        q[:, :, :, None, :].float()
        * _gather_kv_head_block_chunk(k_rep, safe_idx_rep.unsqueeze(0)).float(),
        dim=-1,
    ) / (dh ** 0.5)
    weights = stable_masked_softmax(scores, mask_rep.unsqueeze(0), dim=-1)
    out_ref = torch.sum(
        weights.unsqueeze(-1) * _gather_kv_head_block_chunk(v_rep, safe_idx_rep.unsqueeze(0)).float(),
        dim=3,
    ).to(dtype=v.dtype)

    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)


def test_sparse_gqa_precomputed_triton_fused_matches_repeated_kv_reference(
    device: torch.device,
) -> None:
    from bna.torch.triton_fused_sparse_attn import TRITON_AVAILABLE

    if not TRITON_AVAILABLE or device.type != "cuda":
        pytest.skip("Triton fused kernel requires CUDA + Triton")

    b, hkv, groups, tq, tk, dh = 1, 2, 2, 4, 11, 8
    hq = hkv * groups
    degree = 5

    torch.manual_seed(123)
    q = torch.randn(b, hq, tq, dh, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, hkv, tk, dh, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, hkv, tk, dh, device=device, dtype=torch.bfloat16)
    safe_idx = torch.randint(0, tk, (b, hkv, tq, degree), device=device, dtype=torch.long)
    causal_mask = torch.rand(b, hkv, tq, degree, device=device) > 0.25
    chunk_profile: dict = {}

    out_fused, weights_fused = sparse_row_attention_gqa_precomputed(
        q,
        k,
        v,
        safe_idx=safe_idx,
        causal_mask=causal_mask,
        num_key_value_groups=groups,
        contraction_backend_override="triton_fused",
        chunk_profile=chunk_profile,
    )

    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    safe_idx_rep = safe_idx.repeat_interleave(groups, dim=1)
    mask_rep = causal_mask.repeat_interleave(groups, dim=1)
    scores = torch.sum(
        q[:, :, :, None, :].float()
        * _gather_kv_head_block_chunk(k_rep, safe_idx_rep).float(),
        dim=-1,
    ) / (dh ** 0.5)
    weights = stable_masked_softmax(scores, mask_rep, dim=-1)
    out_ref = torch.sum(
        weights.unsqueeze(-1) * _gather_kv_head_block_chunk(v_rep, safe_idx_rep).float(),
        dim=3,
    ).to(dtype=v.dtype)

    assert weights_fused is None
    assert chunk_profile["sparse_contraction_backend"] == "triton_fused"
    assert torch.allclose(out_fused.float(), out_ref.float(), atol=0.05, rtol=0.01), (
        f"max diff: {(out_fused.float() - out_ref.float()).abs().max().item()}"
    )


def test_sparse_gqa_chunked_streamed_no_weights_matches_repeated_kv_reference(
    device: torch.device,
) -> None:
    b, hkv, groups, t, dh = 2, 2, 3, 11, 4
    hq = hkv * groups
    degree = 5

    q = torch.randn(b, hq, t, dh, device=device, dtype=torch.float32)
    k = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    v = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float32)
    neigh_kv, edge_kv = _make_sparse_graph(
        device=device,
        num_heads=hkv,
        seq_len=t,
        degree=degree,
    )

    out_gqa, weights_gqa = sparse_row_attention_gqa_chunked(
        q,
        k,
        v,
        neigh_idx=neigh_kv,
        edge_type=edge_kv,
        num_key_value_groups=groups,
        return_weights=False,
        query_chunk_size=4,
        kv_head_chunk_size=1,
        degree_chunk_size=2,
    )

    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    neigh_rep = neigh_kv.repeat_interleave(groups, dim=0)
    edge_rep = edge_kv.repeat_interleave(groups, dim=0)
    out_ref, _ = sparse_row_attention(
        q,
        k_rep,
        v_rep,
        neigh_idx=neigh_rep,
        edge_type=edge_rep,
        return_weights=False,
    )

    assert weights_gqa is None
    assert torch.allclose(out_gqa, out_ref, atol=1e-5, rtol=1e-5)


def test_sparse_gqa_triton_fused_matches_repeated_kv_reference(device: torch.device) -> None:
    """The triton_fused backend (return_weights=False, no degree chunking) must match
    the reference sparse_row_attention with repeated KV heads."""
    from bna.torch.triton_fused_sparse_attn import TRITON_AVAILABLE

    if not TRITON_AVAILABLE or device.type != "cuda":
        pytest.skip("Triton fused kernel requires CUDA + Triton")

    b, hkv, groups, t, dh = 1, 2, 3, 11, 8
    hq = hkv * groups
    degree = 5

    torch.manual_seed(42)
    q = torch.randn(b, hq, t, dh, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, hkv, t, dh, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, hkv, t, dh, device=device, dtype=torch.bfloat16)
    neigh_kv, edge_kv = _make_sparse_graph(
        device=device,
        num_heads=hkv,
        seq_len=t,
        degree=degree,
    )
    chunk_profile: dict = {}

    # This should use triton_fused on CUDA
    out_fused, weights_fused = sparse_row_attention_gqa_chunked(
        q,
        k,
        v,
        neigh_idx=neigh_kv,
        edge_type=edge_kv,
        num_key_value_groups=groups,
        return_weights=False,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_profile=chunk_profile,
    )

    assert chunk_profile["sparse_contraction_backend"] == "triton_fused"
    assert weights_fused is None

    # Reference: repeated KV with the original sparse_row_attention
    k_rep = _repeat_kv(k, groups)
    v_rep = _repeat_kv(v, groups)
    neigh_rep = neigh_kv.repeat_interleave(groups, dim=0)
    edge_rep = edge_kv.repeat_interleave(groups, dim=0)
    out_ref, _ = sparse_row_attention(
        q.float(),
        k_rep.float(),
        v_rep.float(),
        neigh_idx=neigh_rep,
        edge_type=edge_rep,
        return_weights=False,
    )

    assert torch.allclose(out_fused.float(), out_ref.float(), atol=0.05, rtol=0.01), (
        f"max diff: {(out_fused.float() - out_ref.float()).abs().max().item()}"
    )


def test_sparse_gqa_triton_fused_handles_fully_masked_rows_with_bias(device: torch.device) -> None:
    from bna.torch.triton_fused_sparse_attn import TRITON_AVAILABLE, triton_fused_sparse_gqa_attention

    if not TRITON_AVAILABLE or device.type != "cuda":
        pytest.skip("Triton fused kernel requires CUDA + Triton")

    b, hkv, groups, t, dh = 2, 2, 3, 13, 8
    hq = hkv * groups
    degree = 7
    scale = float(dh ** -0.5)

    torch.manual_seed(7)
    q = torch.randn(b, hq, t, dh, device=device, dtype=torch.float16).permute(0, 1, 2, 3)
    k = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float16).permute(0, 1, 2, 3)
    v = torch.randn(b, hkv, t, dh, device=device, dtype=torch.float16).permute(0, 1, 2, 3)
    safe_idx = torch.randint(0, t, (b, hkv, t, degree), device=device, dtype=torch.long)
    causal_mask = torch.rand(b, hkv, t, degree, device=device) > 0.2
    causal_mask[:, :, -1, :] = False
    bias = 0.05 * torch.randn(b, hkv, t, degree, device=device, dtype=torch.float32)

    out = triton_fused_sparse_gqa_attention(
        q,
        k,
        v,
        safe_idx,
        causal_mask,
        num_key_value_groups=groups,
        scale=scale,
        bias=bias,
    )

    q_grouped = q.reshape(b, hkv, groups, t, dh)
    k_g = _gather_kv_head_block_chunk(k, safe_idx)
    v_g = _gather_kv_head_block_chunk(v, safe_idx)
    scores = torch.matmul(
        (q_grouped * scale).unsqueeze(-2),
        k_g.unsqueeze(2).transpose(-1, -2),
    ).squeeze(-2).float()
    scores = scores + bias[:, :, None, :, :].float()
    weights = stable_masked_softmax(scores, causal_mask[:, :, None, :, :], dim=-1)
    out_ref = torch.matmul(
        weights.to(dtype=v.dtype).unsqueeze(-2),
        v_g.unsqueeze(2),
    ).squeeze(-2).reshape(b, hq, t, dh)

    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), out_ref.float(), atol=0.05, rtol=0.01), (
        f"max diff: {(out.float() - out_ref.float()).abs().max().item()}"
    )
    assert torch.count_nonzero(out[:, :, -1, :]).item() == 0


def test_gather_kv_head_block_chunk_matches_broadcast_reference(device: torch.device) -> None:
    b, heads, seq_len, dh = 2, 3, 11, 5
    chunk_len, degree = 4, 6
    head_states = torch.randn(b, heads, seq_len, dh, device=device, dtype=torch.float32)
    idx_chunk = torch.randint(0, seq_len, (b, heads, chunk_len, degree), device=device)

    gather_idx = idx_chunk.unsqueeze(-1).expand(b, heads, chunk_len, degree, dh)
    source = head_states.unsqueeze(2).expand(-1, -1, chunk_len, -1, -1)
    expected = torch.gather(source, dim=3, index=gather_idx)
    actual = _gather_kv_head_block_chunk(head_states, idx_chunk)

    assert torch.equal(actual, expected)


def test_sparse_gqa_auto_chunking_prioritizes_fewer_query_chunks_long_context() -> None:
    chunk_meta = _resolve_sparse_gqa_chunking(
        seq_len=8192,
        degree=193,
        hkv=4,
        groups=4,
        head_dim=256,
        kv_element_size=2,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_temp_budget_mib=160.0,
        return_weights=False,
    )

    assert chunk_meta["sparse_chunk_mode"] == "auto"
    assert chunk_meta["sparse_query_chunk_size"] == 1536
    assert chunk_meta["sparse_kv_head_chunk_size"] == 1
    assert chunk_meta["sparse_degree_chunk_size"] == 193
    assert chunk_meta["sparse_num_query_chunks"] == 6
    assert chunk_meta["sparse_num_head_blocks"] == 4
    assert chunk_meta["sparse_num_degree_blocks"] == 1
    assert chunk_meta["sparse_streamed_degree"] is False
    assert chunk_meta["sparse_chunk_budget_exceeded"] is False
    assert float(chunk_meta["sparse_estimated_temp_mib"]) <= 160.0


def test_sparse_gqa_auto_chunking_opens_16k_query_chunks_under_peak_live_budget() -> None:
    chunk_meta = _resolve_sparse_gqa_chunking(
        seq_len=16384,
        degree=193,
        hkv=4,
        groups=4,
        head_dim=256,
        kv_element_size=2,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_temp_budget_mib=160.0,
        return_weights=False,
    )

    assert chunk_meta["sparse_chunk_mode"] == "auto"
    assert chunk_meta["sparse_query_chunk_size"] == 1536
    assert chunk_meta["sparse_kv_head_chunk_size"] == 1
    assert chunk_meta["sparse_degree_chunk_size"] == 193
    assert chunk_meta["sparse_num_query_chunks"] == 11
    assert chunk_meta["sparse_num_head_blocks"] == 4
    assert chunk_meta["sparse_num_degree_blocks"] == 1
    assert chunk_meta["sparse_streamed_degree"] is False
    assert chunk_meta["sparse_chunk_budget_exceeded"] is False
    assert float(chunk_meta["sparse_estimated_temp_mib"]) <= 160.0


def test_sparse_gqa_manual_degree_chunking_reports_streamed_blocks() -> None:
    chunk_meta = _resolve_sparse_gqa_chunking(
        seq_len=8192,
        degree=193,
        hkv=4,
        groups=4,
        head_dim=256,
        kv_element_size=2,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=32,
        chunk_temp_budget_mib=160.0,
        return_weights=False,
    )

    assert chunk_meta["sparse_degree_chunk_size"] == 32
    assert chunk_meta["sparse_num_degree_blocks"] == 7
    assert chunk_meta["sparse_streamed_degree"] is True


def test_qwen_static_sparse_cache_shared_across_layers(monkeypatch, device: torch.device) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        compute_graph_metrics=False,
    )
    layer0, layer1 = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 12, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)

    cache0 = next(iter(layer0._graph_cache.values()))
    cache1 = next(iter(layer1._graph_cache.values()))
    assert cache0 is cache1
    assert cache0.neigh_idx is not None
    assert cache0.neigh_idx.shape[0] == layer0.num_key_value_heads
    assert cache0.perm == []
    assert cache0.inv_perm == []
    assert cache0.perm_stacked is None
    assert cache0.inv_perm_stacked is None
    assert cache0.kv_perm_stacked is None
    assert cache0.flex_block_mask is None

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_static_block_sparse_cache_shared_across_layers(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)
    monkeypatch.setattr(qwen_torch, "is_flex_attention_supported_on_device", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        qwen_torch,
        "wayfinder_block_sparse_attention",
        lambda q, k, v, *, block_mask: (torch.zeros_like(q), block_mask),
    )

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="block_sparse",
        strategy="random",
        block_size=4,

        block_local_window_blocks=1,
        block_partner_count=1,
        block_sink_blocks=1,
        compute_graph_metrics=False,
    )
    layer0, layer1 = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 12, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)

    cache0 = next(iter(layer0._graph_cache.values()))
    cache1 = next(iter(layer1._graph_cache.values()))
    # Wayfinder uses layer-idx-dependent stage scheduling, so layer0 and
    # layer1 get different cache keys when the stage differs.  With only 3
    # blocks and stage_count=ceil(log2(3))=2, layer 0 (stage 0) and layer 1
    # (stage 1) will differ.  But both caches should still be valid.
    assert cache0.graph is None
    assert cache0.block_layout is not None
    assert cache0.flex_block_mask is not None
    assert cache0.neigh_idx is None
    assert cache0.edge_type is None
    assert layer0.last_profile["path"] == "block_sparse"
    assert layer0.last_profile["block_sparse_backend"] == "flex_attention"
    assert layer0.last_profile["block_sparse_block_size"] == 4
    assert layer0.last_profile["block_sparse_num_blocks"] == 3
    assert int(layer0.last_profile["block_sparse_neighbor_blocks"]) >= 3
    assert layer0.last_profile["graph_source"] == "runtime_block_butterfly"

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_wayfinder_block_sparse_cache_shared_only_for_matching_stage(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_wayfinder_block_sparse_cached_kv_uses_sparse_precomputed_backend(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    class _FakePastKeyValues:
        def __init__(self, seq_len: int) -> None:
            self.seq_len = int(seq_len)

        def get_seq_length(self, layer_idx=None):
            del layer_idx
            return self.seq_len

    def _fake_extract_qkv_cached(
        attn,
        hidden_states,
        position_embeddings,
        *,
        past_key_values=None,
        cache_position=None,
    ):
        del position_embeddings, cache_position
        b, t, c = hidden_states.shape
        hq = attn.q_proj.out_features // (attn.head_dim * 2)
        hkv = attn.k_proj.out_features // attn.head_dim
        cache_len = 0 if past_key_values is None else int(past_key_values.get_seq_length(attn.layer_idx))
        total_t = t + cache_len
        q = hidden_states.view(b, t, hq, attn.head_dim).transpose(1, 2).contiguous()
        base = torch.arange(
            b * hkv * total_t * attn.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).reshape(b, hkv, total_t, attn.head_dim)
        k = (base / 100.0).contiguous()
        v = (base / 80.0).contiguous()
        gate = torch.zeros((b, t, c), device=hidden_states.device, dtype=hidden_states.dtype)
        return q, k, v, gate

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv_cached)
    monkeypatch.setattr(qwen_torch, "is_flex_attention_supported_on_device", lambda *_args, **_kwargs: False)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="block_sparse",
        strategy="random",
        block_size=4,

        block_local_window_blocks=1,
        block_partner_count=1,
        block_sink_blocks=1,
        sparse_query_chunk_size=2,
        sparse_kv_head_chunk_size=1,
        sparse_degree_chunk_size=4,
    )
    layer = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=0),
        cfg,
    )
    layer.eval()

    hidden_states = torch.randn(1, 2, layer.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))
    past_key_values = _FakePastKeyValues(seq_len=6)

    out, _ = layer(
        hidden_states,
        position_embeddings=pos,
        past_key_values=past_key_values,
        cache_position=torch.tensor([6, 7], device=device, dtype=torch.long),
    )

    assert tuple(out.shape) == tuple(hidden_states.shape)
    assert layer.last_profile["path"] == "block_sparse"
    assert layer.last_profile["block_sparse_backend"] == "sparse_gqa_precomputed"
    assert layer.last_profile["sparse_contraction_backend"] in {
        "sdpa",
        "streamed_online_softmax",
    }
    assert layer.last_profile["block_sparse_topology"] == "butterfly"
    assert layer.last_profile["graph_source"] == "runtime_block_butterfly"
    assert layer.last_profile["kv_len"] == 8

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)
    monkeypatch.setattr(qwen_torch, "is_flex_attention_supported_on_device", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        qwen_torch,
        "wayfinder_block_sparse_attention",
        lambda q, k, v, *, block_mask: (torch.zeros_like(q), block_mask),
    )

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="block_sparse",
        strategy="random",
        block_size=4,

        block_local_window_blocks=1,
        block_partner_count=2,
        block_sink_blocks=1,
        compute_graph_metrics=False,
    )

    layer0 = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=0),
        cfg,
    )
    layer1 = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=1),
        cfg,
    )
    layer2 = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=2),
        cfg,
    )
    layer0.eval()
    layer1.eval()
    layer2.eval()

    hidden_states = torch.randn(1, 12, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)
    layer2(hidden_states, position_embeddings=pos)

    cache0 = next(iter(layer0._graph_cache.values()))
    cache1 = next(iter(layer1._graph_cache.values()))
    cache2 = next(iter(layer2._graph_cache.values()))

    assert cache0 is not cache1
    assert cache0 is cache2
    assert cache0.block_layout is not None
    assert cache1.block_layout is not None
    assert cache0.block_layout.topology_name == "butterfly"
    assert cache0.block_layout.stage_idx == 0
    assert cache1.block_layout.stage_idx == 1
    assert cache0.flex_block_mask is not None
    assert layer0.last_profile["path"] == "block_sparse"
    assert layer0.last_profile["block_sparse_backend"] == "flex_attention"
    assert layer0.last_profile["block_sparse_topology"] == "butterfly"
    assert layer0.last_profile["block_sparse_stage"] == 0
    assert layer1.last_profile["block_sparse_stage"] == 1
    assert layer0.last_profile["graph_source"] == "runtime_block_butterfly"

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_wayfinder_block_sparse_cached_kv_can_use_triton_fused_backend(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch
    from bna.torch.triton_fused_sparse_attn import TRITON_AVAILABLE

    if not TRITON_AVAILABLE or device.type != "cuda":
        pytest.skip("Triton fused kernel requires CUDA + Triton")

    class _FakePastKeyValues:
        def __init__(self, seq_len: int) -> None:
            self.seq_len = int(seq_len)

        def get_seq_length(self, layer_idx=None):
            del layer_idx
            return self.seq_len

    def _fake_extract_qkv_cached(
        attn,
        hidden_states,
        position_embeddings,
        *,
        past_key_values=None,
        cache_position=None,
    ):
        del position_embeddings, cache_position
        b, t, c = hidden_states.shape
        hq = attn.q_proj.out_features // (attn.head_dim * 2)
        hkv = attn.k_proj.out_features // attn.head_dim
        cache_len = 0 if past_key_values is None else int(past_key_values.get_seq_length(attn.layer_idx))
        total_t = t + cache_len
        q = hidden_states.view(b, t, hq, attn.head_dim).transpose(1, 2).contiguous()
        base = torch.arange(
            b * hkv * total_t * attn.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).reshape(b, hkv, total_t, attn.head_dim)
        k = (base / 100.0).contiguous()
        v = (base / 80.0).contiguous()
        gate = torch.zeros((b, t, c), device=hidden_states.device, dtype=hidden_states.dtype)
        return q, k, v, gate

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv_cached)
    monkeypatch.setattr(qwen_torch, "is_flex_attention_supported_on_device", lambda *_args, **_kwargs: False)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="block_sparse",
        strategy="random",
        block_size=4,
        block_local_window_blocks=1,
        block_partner_count=1,
        block_sink_blocks=1,
        sparse_query_chunk_size=0,
        sparse_kv_head_chunk_size=0,
        sparse_degree_chunk_size=0,
        sparse_precomputed_backend="triton_fused",
    )
    layer = qwen_torch.QwenCUDAWayfinderAttention(
        _FakeQwenAttention(device=device, layer_idx=0),
        cfg,
    )
    layer.eval()

    hidden_states = torch.randn(1, 2, layer.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))
    past_key_values = _FakePastKeyValues(seq_len=6)

    out, _ = layer(
        hidden_states,
        position_embeddings=pos,
        past_key_values=past_key_values,
        cache_position=torch.tensor([6, 7], device=device, dtype=torch.long),
    )

    assert tuple(out.shape) == tuple(hidden_states.shape)
    assert layer.last_profile["block_sparse_backend"] == "sparse_gqa_precomputed"
    assert layer.last_profile["sparse_contraction_backend"] == "triton_fused"

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_static_sparse_metrics_computed_once_per_shared_cache(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)
    calls = {"count": 0}

    def _fake_metrics(_abi):
        calls["count"] += 1
        return {"degree_mean": 3.5}

    monkeypatch.setattr(qwen_torch, "graph_metrics", _fake_metrics)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        compute_graph_metrics=True,
    )
    layer0, layer1 = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 10, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)

    assert calls["count"] == 0
    assert layer0.ensure_last_graph_metrics() == {"degree_mean": 3.5}
    assert calls["count"] == 1
    assert layer1.ensure_last_graph_metrics() == {"degree_mean": 3.5}
    assert calls["count"] == 1
    assert layer0.last_profile["graph_metrics"] == {"degree_mean": 3.5}
    assert layer1.last_profile["graph_metrics"] == {"degree_mean": 3.5}

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_sparse_profile_reports_effective_chunking(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        sparse_query_chunk_size=0,
        sparse_kv_head_chunk_size=0,
        sparse_chunk_temp_budget_mib=4.0,
    )
    layer0, _ = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 11, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)

    profile = layer0.last_profile
    assert profile["mode"] == "butterfly"
    assert profile["path"] == "sparse"
    assert profile["sparse_chunk_mode"] == "auto"
    assert int(profile["sparse_query_chunk_size"]) >= 1
    assert int(profile["sparse_kv_head_chunk_size"]) >= 1
    assert int(profile["sparse_degree_chunk_size"]) >= 1
    assert int(profile["sparse_num_query_chunks"]) >= 1
    assert int(profile["sparse_num_head_blocks"]) >= 1
    assert int(profile["sparse_num_degree_blocks"]) >= 1
    assert isinstance(profile["sparse_streamed_degree"], bool)
    assert isinstance(profile["sparse_chunk_budget_exceeded"], bool)
    assert float(profile["sparse_estimated_temp_mib"]) > 0.0
    assert profile["sparse_contraction_backend"] in ("sdpa", "triton_fused", "streamed_online_softmax")
    assert float(profile["attn_kernel_ms_host"]) >= 0.0

    snapshot = layer0.snapshot_last_profile()
    assert snapshot["attn_kernel_ms_host"] == profile["attn_kernel_ms_host"]
    if device.type == "cuda":
        assert snapshot["attn_kernel_ms"] == snapshot["sparse_contraction_cuda_ms"]
        assert float(snapshot["sparse_contraction_cuda_ms"]) >= 0.0
    else:
        assert snapshot["attn_kernel_ms"] == profile["attn_kernel_ms"]
        assert snapshot.get("sparse_contraction_cuda_ms") is None

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_sparse_trace_dump_writes_replayable_payload(
    monkeypatch,
    tmp_path: Path,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        sparse_trace_dir=str(tmp_path),
        sparse_trace_max_per_layer=1,
    )
    layer0, _ = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 9, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)

    dumped = sorted(tmp_path.glob("qwen_sparse_trace_layer*.pt"))
    assert len(dumped) == 1
    payload = torch.load(dumped[0], map_location="cpu")
    assert payload["format_version"] == 1
    assert payload["source"] == "wayfinder_qwen_torch"
    assert payload["layer_idx"] == layer0.layer_idx
    assert payload["num_key_value_groups"] == layer0.num_key_value_groups
    assert tuple(payload["q"].shape) == (1, layer0.num_heads, 9, layer0.head_dim)
    assert tuple(payload["k"].shape) == (1, layer0.num_key_value_heads, 9, layer0.head_dim)
    assert tuple(payload["v"].shape) == (1, layer0.num_key_value_heads, 9, layer0.head_dim)
    assert tuple(payload["safe_idx"].shape[:3]) == (1, layer0.num_key_value_heads, 9)
    assert tuple(payload["causal_mask"].shape) == tuple(payload["safe_idx"].shape)
    assert payload["bias"] is None
    assert layer0.last_profile["sparse_trace_path"] == str(dumped[0])
    assert layer0.last_profile.get("sparse_trace_error") is None

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_sparse_chunk_config_is_forwarded_to_kernel(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)
    captured = {}

    def _fake_sparse_kernel(
        q,
        k,
        v,
        *,
        neigh_idx,
        num_key_value_groups,
        edge_type=None,
        return_weights=False,
        precomputed_safe_idx=None,
        precomputed_causal_mask=None,
        edge_type_bias=None,
        edge_type_bias_offset=None,
        window_drop_mask=None,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_temp_budget_mib=160.0,
        chunk_profile=None,
    ):
        del neigh_idx
        del edge_type
        del return_weights
        del precomputed_safe_idx
        del precomputed_causal_mask
        del edge_type_bias
        del edge_type_bias_offset
        del window_drop_mask
        captured["num_key_value_groups"] = num_key_value_groups
        captured["query_chunk_size"] = query_chunk_size
        captured["kv_head_chunk_size"] = kv_head_chunk_size
        captured["degree_chunk_size"] = degree_chunk_size
        captured["chunk_temp_budget_mib"] = chunk_temp_budget_mib
        if chunk_profile is not None:
            chunk_profile.update({
                "sparse_chunk_mode": "manual",
                "sparse_query_chunk_size": int(query_chunk_size),
                "sparse_kv_head_chunk_size": int(kv_head_chunk_size),
                "sparse_degree_chunk_size": int(degree_chunk_size),
                "sparse_num_query_chunks": 3,
                "sparse_num_head_blocks": 2,
                "sparse_num_degree_blocks": 4,
                "sparse_streamed_degree": True,
                "sparse_chunk_budget_exceeded": False,
                "sparse_estimated_temp_mib": 12.5,
                "sparse_contraction_backend": "streamed_online_softmax",
            })
        return torch.zeros_like(q), None

    monkeypatch.setattr(qwen_torch, "sparse_row_attention_gqa_chunked", _fake_sparse_kernel)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        sparse_query_chunk_size=512,
        sparse_kv_head_chunk_size=2,
        sparse_degree_chunk_size=24,
        sparse_chunk_temp_budget_mib=192.0,
    )
    layer0, _ = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 10, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)

    assert captured["num_key_value_groups"] == layer0.num_key_value_groups
    assert captured["query_chunk_size"] == 512
    assert captured["kv_head_chunk_size"] == 2
    assert captured["degree_chunk_size"] == 24
    assert captured["chunk_temp_budget_mib"] == 192.0
    assert layer0.last_profile["sparse_chunk_mode"] == "manual"
    assert layer0.last_profile["sparse_query_chunk_size"] == 512
    assert layer0.last_profile["sparse_kv_head_chunk_size"] == 2
    assert layer0.last_profile["sparse_degree_chunk_size"] == 24
    assert layer0.last_profile["sparse_num_query_chunks"] == 3
    assert layer0.last_profile["sparse_num_head_blocks"] == 2
    assert layer0.last_profile["sparse_num_degree_blocks"] == 4
    assert layer0.last_profile["sparse_streamed_degree"] is True
    assert layer0.last_profile["sparse_chunk_budget_exceeded"] is False
    assert layer0.last_profile["sparse_estimated_temp_mib"] == 12.5
    assert layer0.last_profile["sparse_contraction_backend"] == "streamed_online_softmax"

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_sparse_compute_dtype_model_downcasts_before_kernel(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    captured = {}

    def _fake_extract_promoted_qkv(
        attn: _FakeQwenAttention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        past_key_values=None,
        cache_position=None,
    ):
        del position_embeddings, past_key_values, cache_position
        b, t, c = hidden_states.shape
        hq = attn.q_proj.out_features // (attn.head_dim * 2)
        hkv = attn.k_proj.out_features // attn.head_dim
        q = hidden_states.double().view(b, t, hq, attn.head_dim).transpose(1, 2).contiguous()
        kv_base = hidden_states[:, :, : hkv * attn.head_dim]
        k = kv_base.double().view(b, t, hkv, attn.head_dim).transpose(1, 2).contiguous()
        v = (kv_base + 0.25).float().view(b, t, hkv, attn.head_dim).transpose(1, 2).contiguous()
        gate = torch.zeros((b, t, c), device=hidden_states.device, dtype=hidden_states.dtype)
        return q, k, v, gate

    def _fake_sparse_kernel(
        q,
        k,
        v,
        *,
        neigh_idx,
        num_key_value_groups,
        edge_type=None,
        return_weights=False,
        precomputed_safe_idx=None,
        precomputed_causal_mask=None,
        edge_type_bias=None,
        edge_type_bias_offset=None,
        window_drop_mask=None,
        query_chunk_size=0,
        kv_head_chunk_size=0,
        degree_chunk_size=0,
        chunk_temp_budget_mib=160.0,
        chunk_profile=None,
    ):
        del neigh_idx
        del num_key_value_groups
        del edge_type
        del return_weights
        del precomputed_safe_idx
        del precomputed_causal_mask
        del edge_type_bias
        del edge_type_bias_offset
        del window_drop_mask
        del query_chunk_size
        del kv_head_chunk_size
        del degree_chunk_size
        del chunk_temp_budget_mib
        captured["q_dtype"] = q.dtype
        captured["k_dtype"] = k.dtype
        captured["v_dtype"] = v.dtype
        if chunk_profile is not None:
            chunk_profile["sparse_chunk_mode"] = "manual"
        return torch.zeros_like(q), None

    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_promoted_qkv)
    monkeypatch.setattr(qwen_torch, "sparse_row_attention_gqa_chunked", _fake_sparse_kernel)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
        sparse_compute_dtype="model",
    )
    layer0, _ = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 10, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)

    assert captured["q_dtype"] == torch.float32
    assert captured["k_dtype"] == torch.float32
    assert captured["v_dtype"] == torch.float32
    assert layer0.last_profile["sparse_query_input_dtype"] == "float64"
    assert layer0.last_profile["sparse_key_input_dtype"] == "float64"
    assert layer0.last_profile["sparse_value_input_dtype"] == "float32"
    assert layer0.last_profile["sparse_compute_dtype"] == "float32"

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()


def test_qwen_dynamic_sparse_graphs_are_not_shared(monkeypatch, device: torch.device) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="greedy",
    )
    layer0, layer1 = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 9, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))

    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)

    assert len(layer0._graph_cache) == 0
    assert len(layer1._graph_cache) == 0
    assert layer0._last_graph_cache is not None
    assert layer1._last_graph_cache is not None
    assert layer0._last_graph_cache is not layer1._last_graph_cache


def test_bench_clear_wayfinder_caches_clears_shared_qwen_cache(
    monkeypatch,
    device: torch.device,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    qwen_torch.clear_shared_qwen_wayfinder_graph_cache()
    monkeypatch.setattr(qwen_torch, "extract_qkv_from_qwen_attention", _fake_extract_qkv)

    cfg = qwen_torch.QwenCUDAWayfinderConfig(
        path="sparse",
        strategy="random",
    )
    layer0, layer1 = _make_wrapped_layers(qwen_torch, device=device, cfg=cfg)
    hidden_states = torch.randn(1, 8, layer0.hidden_size, device=device, dtype=torch.float32)
    pos = (torch.zeros(1, device=device), torch.zeros(1, device=device))
    layer0(hidden_states, position_embeddings=pos)
    layer1(hidden_states, position_embeddings=pos)

    class _Container:
        def __init__(self, attn):
            self.self_attn = attn

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[_Container(layer0), _Container(layer1)])
    )
    bench = _load_script_module(
        "bench_qwen35_cuda_wayfinder_cache_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )
    cleared = bench.clear_wayfinder_graph_caches(model)

    assert cleared["layers"] == 2
    assert cleared["entries"] == 2
    assert cleared["shared_entries"] == 1
    assert len(layer0._graph_cache) == 0
    assert len(layer1._graph_cache) == 0
    assert layer0._last_graph_cache is None
    assert layer1._last_graph_cache is None
    assert qwen_torch.clear_shared_qwen_wayfinder_graph_cache() == 0
