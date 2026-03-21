"""Fused sparse GQA attention via Triton — no gathered K/V materialization.

This module provides a Triton-based fused attention kernel for the Wayfinder/HCSA
sparse path.  Instead of materializing full gathered K/V neighborhoods in global
memory, the kernel indexes into the base K/V tensors directly during the
score→softmax→V accumulation loop using online (streaming) softmax.

Memory contract:
  - No [B, Hkv, chunk, D, dh] gathered K/V tensors
  - No full [B, Hkv, groups, chunk, D] score tensor
  - No full [B, Hkv, groups, chunk, D] weight tensor
  - Accumulates in float32 registers, stores in output dtype

Restrictions:
  - return_weights=False only (benchmark/inference path)
  - Requires Triton ≥ 3.0
  - Degree (D) becomes a JIT constexpr; each unique D recompiles once
"""

from __future__ import annotations

import importlib
import math
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


_RECUDA_IMPORT_ERROR: Exception | None = None
_recuda_triton_fused_sparse_gqa_attention = None


def _use_recuda_sparse_gqa_kernel() -> bool:
    return os.environ.get("WAYFINDER_USE_RECUDA_KERNEL", "0") == "1"


def _get_recuda_sparse_gqa_attention():
    global _RECUDA_IMPORT_ERROR
    global _recuda_triton_fused_sparse_gqa_attention

    if _recuda_triton_fused_sparse_gqa_attention is not None:
        return _recuda_triton_fused_sparse_gqa_attention
    if _RECUDA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "WAYFINDER_USE_RECUDA_KERNEL=1 but the reCUDA sparse GQA kernel "
            "could not be imported"
        ) from _RECUDA_IMPORT_ERROR

    try:
        module = importlib.import_module("hcsa.torch.triton_fused_sparse_attn_v2")
        _recuda_triton_fused_sparse_gqa_attention = module.run
        return _recuda_triton_fused_sparse_gqa_attention
    except Exception as exc:  # pragma: no cover - surfaced at call time
        _RECUDA_IMPORT_ERROR = exc
        raise RuntimeError(
            "WAYFINDER_USE_RECUDA_KERNEL=1 but the reCUDA sparse GQA kernel "
            "could not be imported"
        ) from exc


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_sparse_gqa_attn_fwd(
        Q,
        K,
        V,
        Out,
        SafeIdx,
        CausalMask,
        Bias,
        # Q strides [B, Hq, T, dh]
        sq_b,
        sq_h,
        sq_t,
        sq_d,
        # K strides [B, Hkv, T, dh]
        sk_b,
        sk_h,
        sk_t,
        sk_d,
        # V strides [B, Hkv, T, dh]
        sv_b,
        sv_h,
        sv_t,
        sv_d,
        # Out strides [B, Hq, T, dh]
        so_b,
        so_h,
        so_t,
        so_d,
        # SafeIdx strides [B, Hkv, T, D]
        si_b,
        si_h,
        si_t,
        si_d,
        # CausalMask strides [B, Hkv, T, D]
        sm_b,
        sm_h,
        sm_t,
        sm_d,
        # Bias strides [B, Hkv, T, D] (or 0 if no bias)
        sb_b,
        sb_h,
        sb_t,
        sb_d,
        scale,
        HAS_BIAS: tl.constexpr,
        HKV: tl.constexpr,
        GROUPS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        D_NEIGH: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
    ):
        # Each program handles one (b, h_kv, t) triple.
        pid = tl.program_id(0)
        t_idx = pid % SEQ_LEN
        bh = pid // SEQ_LEN
        h_kv = bh % HKV
        b_idx = bh // HKV

        offs_d = tl.arange(0, BLOCK_DHEAD)
        d_mask = offs_d < D_HEAD

        # Base pointers for neighbor graph at this (b, h_kv, t)
        idx_base = SafeIdx + b_idx * si_b + h_kv * si_h + t_idx * si_t
        mask_base = CausalMask + b_idx * sm_b + h_kv * sm_h + t_idx * sm_t

        # K/V base for this (b, h_kv)
        k_head_base = K + b_idx * sk_b + h_kv * sk_h
        v_head_base = V + b_idx * sv_b + h_kv * sv_h

        # Optional bias base
        if HAS_BIAS:
            bias_base = Bias + b_idx * sb_b + h_kv * sb_h + t_idx * sb_t

        # Process each query head in this GQA group
        for g in range(GROUPS):
            h_q = h_kv * GROUPS + g

            # Load query vector [BLOCK_DHEAD]
            q_ptr = Q + b_idx * sq_b + h_q * sq_h + t_idx * sq_t + offs_d * sq_d
            q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32) * scale

            # Online softmax state
            m_i = tl.full([], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([], dtype=tl.float32)
            acc = tl.zeros([BLOCK_DHEAD], dtype=tl.float32)

            # Loop over degree blocks
            for d_start in range(0, D_NEIGH, BLOCK_N):
                d_offs = tl.arange(0, BLOCK_N) + d_start
                d_valid = d_offs < D_NEIGH

                # Load neighbor indices [BLOCK_N]
                idx = tl.load(idx_base + d_offs * si_d, mask=d_valid, other=0)
                valid_raw = tl.load(mask_base + d_offs * sm_d, mask=d_valid, other=0)
                valid = (valid_raw != 0) & d_valid
                has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0

                # Gather K rows: [BLOCK_N, BLOCK_DHEAD]
                k_ptrs = k_head_base + idx[:, None] * sk_t + offs_d[None, :] * sk_d
                k_block = tl.load(
                    k_ptrs,
                    mask=d_valid[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                # Scores [BLOCK_N]
                scores = tl.sum(q[None, :] * k_block, axis=1)

                # Apply optional bias
                if HAS_BIAS:
                    bias_vals = tl.load(
                        bias_base + d_offs * sb_d,
                        mask=d_valid,
                        other=0.0,
                    ).to(tl.float32)
                    scores = scores + bias_vals

                scores = tl.where(valid, scores, float("-inf"))

                # Online softmax update
                m_block = tl.max(scores)
                m_new = tl.where(has_valid, tl.maximum(m_i, m_block), m_i)
                alpha = tl.where(has_valid, tl.exp(m_i - m_new), 1.0)
                p = tl.exp(tl.where(valid, scores, 0.0) - m_new)
                p = tl.where(valid, p, 0.0)

                # Gather V rows: [BLOCK_N, BLOCK_DHEAD]
                v_ptrs = v_head_base + idx[:, None] * sv_t + offs_d[None, :] * sv_d
                v_block = tl.load(
                    v_ptrs,
                    mask=d_valid[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                # Accumulate: rescale previous + add new
                acc = alpha * acc + tl.sum(p[:, None] * v_block, axis=0)
                l_i = alpha * l_i + tl.sum(p)
                m_i = m_new

            # Normalize — zero output for all-masked rows (no valid neighbors)
            acc = tl.where(l_i > 0.0, acc / l_i, 0.0)

            # Store output
            out_ptr = Out + b_idx * so_b + h_q * so_h + t_idx * so_t + offs_d * so_d
            tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=d_mask)


def triton_fused_sparse_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    safe_idx: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    num_key_value_groups: int,
    scale: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused sparse GQA attention — no gathered K/V materialization.

    Args:
        q: [B, Hq, T, dh] query states
        k: [B, Hkv, T, dh] key states (base, not repeated)
        v: [B, Hkv, T, dh] value states (base, not repeated)
        safe_idx: [B, Hkv, T, D] clamped neighbor indices
        causal_mask: [B, Hkv, T, D] bool causal validity mask
        num_key_value_groups: GQA ratio (Hq / Hkv)
        scale: attention scaling factor (typically 1/sqrt(dh))
        bias: optional [B, Hkv, T, D] float32 attention bias

    Returns:
        [B, Hq, T, dh] attention output in v.dtype
    """
    if _use_recuda_sparse_gqa_kernel():
        recuda_kernel = _get_recuda_sparse_gqa_attention()
        return recuda_kernel(
            q,
            k,
            v,
            safe_idx,
            causal_mask,
            num_key_value_groups=num_key_value_groups,
            scale=scale,
            bias=bias,
        )

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    b, hq, t, dh = q.shape
    _, hkv, _, _ = k.shape
    d_neigh = safe_idx.shape[-1]
    groups = int(num_key_value_groups)

    if hq != hkv * groups:
        raise ValueError(f"Hq={hq} != Hkv={hkv} * groups={groups}")
    if d_neigh == 0:
        return torch.zeros_like(q).to(dtype=v.dtype)

    out = torch.empty(b, hq, t, dh, device=q.device, dtype=v.dtype)

    # Convert bool mask to uint8 for Triton
    mask_uint8 = causal_mask.to(torch.uint8).contiguous()
    safe_idx_contig = safe_idx.contiguous()

    # Constexpr parameters
    block_dhead = _next_power_of_2(dh)
    block_n = min(32, _next_power_of_2(d_neigh))

    has_bias = bias is not None
    if has_bias:
        bias_contig = bias.contiguous()
        sb_b, sb_h, sb_t, sb_d = bias_contig.stride()
    else:
        bias_contig = safe_idx_contig  # dummy, won't be accessed
        sb_b = sb_h = sb_t = sb_d = 0

    grid = (b * hkv * t,)

    _fused_sparse_gqa_attn_fwd[grid](
        q,
        k,
        v,
        out,
        safe_idx_contig,
        mask_uint8,
        bias_contig,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # K strides
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # V strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # Out strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # SafeIdx strides
        safe_idx_contig.stride(0),
        safe_idx_contig.stride(1),
        safe_idx_contig.stride(2),
        safe_idx_contig.stride(3),
        # Mask strides
        mask_uint8.stride(0),
        mask_uint8.stride(1),
        mask_uint8.stride(2),
        mask_uint8.stride(3),
        # Bias strides
        sb_b,
        sb_h,
        sb_t,
        sb_d,
        scale,
        HAS_BIAS=has_bias,
        HKV=hkv,
        GROUPS=groups,
        SEQ_LEN=t,
        D_NEIGH=d_neigh,
        D_HEAD=dh,
        BLOCK_N=block_n,
        BLOCK_DHEAD=block_dhead,
    )

    return out
