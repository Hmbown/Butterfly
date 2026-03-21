"""reCUDA-optimized fused sparse GQA attention kernel.

This module mirrors ``triton_fused_sparse_gqa_attention`` from
``hcsa.torch.triton_fused_sparse_attn`` but uses the reCUDA-generated
autotuned Triton kernel from:

``/home/hmbown/Projects/reCUDA/artifacts/wayfinder-zai-trace-20260321/``
``wayfinder-sparse-gqa-20260321-121356/candidates/candidate-01.py``
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 32}, num_warps=4),
            triton.Config({"BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 32}, num_warps=8),
            triton.Config({"BLOCK_N": 64}, num_warps=8),
            triton.Config({"BLOCK_N": 128}, num_warps=8),
        ],
        key=["D_NEIGH", "D_HEAD"],
    )
    @triton.jit
    def _fused_sparse_gqa_attn_fwd(
        Q,
        K,
        V,
        Out,
        SafeIdx,
        CausalMask,
        Bias,
        sq_b,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_h,
        sk_t,
        sk_d,
        sv_b,
        sv_h,
        sv_t,
        sv_d,
        so_b,
        so_h,
        so_t,
        so_d,
        si_b,
        si_h,
        si_t,
        si_d,
        sm_b,
        sm_h,
        sm_t,
        sm_d,
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
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t_idx = pid % SEQ_LEN
        bh = pid // SEQ_LEN
        h_kv = bh % HKV
        b_idx = bh // HKV

        offs_dh = tl.arange(0, BLOCK_DHEAD)
        dh_mask = offs_dh < D_HEAD

        idx_base = SafeIdx + b_idx * si_b + h_kv * si_h + t_idx * si_t
        mask_base = CausalMask + b_idx * sm_b + h_kv * sm_h + t_idx * sm_t
        k_head_base = K + b_idx * sk_b + h_kv * sk_h
        v_head_base = V + b_idx * sv_b + h_kv * sv_h

        if HAS_BIAS:
            bias_base = Bias + b_idx * sb_b + h_kv * sb_h + t_idx * sb_t

        for g in range(GROUPS):
            h_q = h_kv * GROUPS + g
            q_ptr = Q + b_idx * sq_b + h_q * sq_h + t_idx * sq_t + offs_dh * sq_d
            q = tl.load(q_ptr, mask=dh_mask, other=0.0).to(tl.float32) * scale

            m_i = tl.full([], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([], dtype=tl.float32)
            acc = tl.zeros([BLOCK_DHEAD], dtype=tl.float32)

            for d_start in range(0, D_NEIGH, BLOCK_N):
                d_offs = tl.arange(0, BLOCK_N) + d_start
                d_valid = d_offs < D_NEIGH

                idx = tl.load(idx_base + d_offs * si_d, mask=d_valid, other=0)
                valid_raw = tl.load(mask_base + d_offs * sm_d, mask=d_valid, other=0)
                valid = (valid_raw != 0) & d_valid
                has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0

                k_ptrs = k_head_base + idx[:, None] * sk_t + offs_dh[None, :] * sk_d
                k_block = tl.load(
                    k_ptrs,
                    mask=d_valid[:, None] & dh_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                scores = tl.sum(q[None, :] * k_block, axis=1)

                if HAS_BIAS:
                    bias_vals = tl.load(
                        bias_base + d_offs * sb_d,
                        mask=d_valid,
                        other=0.0,
                    ).to(tl.float32)
                    scores = scores + bias_vals

                scores = tl.where(valid, scores, float("-inf"))

                m_block = tl.max(scores)
                m_new = tl.where(has_valid, tl.maximum(m_i, m_block), m_i)
                alpha = tl.where(has_valid, tl.exp(m_i - m_new), 1.0)
                p = tl.exp(tl.where(valid, scores, 0.0) - m_new)
                p = tl.where(valid, p, 0.0)

                v_ptrs = v_head_base + idx[:, None] * sv_t + offs_dh[None, :] * sv_d
                v_block = tl.load(
                    v_ptrs,
                    mask=d_valid[:, None] & dh_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                acc = alpha * acc + tl.sum(p[:, None] * v_block, axis=0)
                l_i = alpha * l_i + tl.sum(p)
                m_i = m_new

            acc = tl.where(l_i > 0.0, acc / l_i, 0.0)
            out_ptr = Out + b_idx * so_b + h_q * so_h + t_idx * so_t + offs_dh * so_d
            tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=dh_mask)


def run(
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
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    b, hq, t, dh = q.shape
    hkv = k.shape[1]
    degree = safe_idx.shape[3]
    groups = int(num_key_value_groups)

    if hq != hkv * groups:
        raise ValueError(f"Hq={hq} != Hkv={hkv} * groups={groups}")
    if degree == 0:
        return torch.zeros((b, hq, t, dh), device=q.device, dtype=v.dtype)

    out = torch.empty((b, hq, t, dh), device=q.device, dtype=v.dtype)

    safe_idx_contig = safe_idx.contiguous()
    mask_uint8 = causal_mask.to(torch.uint8).contiguous()
    has_bias = bias is not None
    if has_bias:
        bias_contig = bias.contiguous()
        sb0, sb1, sb2, sb3 = bias_contig.stride()
    else:
        bias_contig = safe_idx_contig
        sb0 = sb1 = sb2 = sb3 = 0

    block_dhead = _next_power_of_2(dh)
    grid = (b * hkv * t,)

    _fused_sparse_gqa_attn_fwd[grid](
        q,
        k,
        v,
        out,
        safe_idx_contig,
        mask_uint8,
        bias_contig,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        safe_idx_contig.stride(0),
        safe_idx_contig.stride(1),
        safe_idx_contig.stride(2),
        safe_idx_contig.stride(3),
        mask_uint8.stride(0),
        mask_uint8.stride(1),
        mask_uint8.stride(2),
        mask_uint8.stride(3),
        sb0,
        sb1,
        sb2,
        sb3,
        scale,
        HAS_BIAS=has_bias,
        HKV=hkv,
        GROUPS=groups,
        SEQ_LEN=t,
        D_NEIGH=degree,
        D_HEAD=dh,
        BLOCK_DHEAD=block_dhead,
    )
    return out
