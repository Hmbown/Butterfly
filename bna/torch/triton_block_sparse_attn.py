"""Fused block-sparse GQA attention via Triton — single kernel launch.

Replaces the SDPA-based ``wayfinder_block_sparse_sdpa_attention`` with a Triton
kernel that fuses the entire block-sparse attention computation.  Instead of
gathering K/V blocks into contiguous memory and calling SDPA (which falls back to
the slow math backend since FlashAttention cannot handle custom bool masks), this
kernel:

  1. Iterates over block-level neighbors directly
  2. Loads Q/K/V tiles from the base tensors using block-level pointer arithmetic
  3. Computes scores, applies causal masking, and accumulates via online softmax
  4. Writes the final output — all in a single kernel launch

Memory contract:
  - No gathered K/V block tensors in global memory
  - No materialized full score or weight tensors
  - Accumulates in float32 registers, stores in output dtype

Design:
  - Grid: (B * H_kv, N) — one program per (batch, kv_head, query_block)
  - Each program processes all ``GROUPS`` query heads sharing this KV head
  - Within each program, loops over K neighbor blocks using online softmax
  - Block size BS and head dim dh are compile-time constants

Restrictions:
  - Inference-only (no backward pass)
  - Requires Triton >= 3.0
  - BS and dh must each be a power of 2 (128 is the default for Qwen3.5-9B)
"""

from __future__ import annotations

import math

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

    @triton.jit
    def _fused_block_sparse_gqa_attn_fwd(
        Q,
        K,
        V,
        Out,
        BlockNeighbors,
        # Q strides [B, H_q, T, dh]
        sq_b,
        sq_h,
        sq_t,
        sq_d,
        # K strides [B, H_kv, T, dh]
        sk_b,
        sk_h,
        sk_t,
        sk_d,
        # V strides [B, H_kv, T, dh]
        sv_b,
        sv_h,
        sv_t,
        sv_d,
        # Out strides [B, H_q, T, dh]
        so_b,
        so_h,
        so_t,
        so_d,
        # BlockNeighbors strides [H_kv, N, K]
        sn_h,
        sn_n,
        sn_k,
        # Scalar params
        scale,
        SEQ_LEN: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
        GROUPS: tl.constexpr,
        HKV: tl.constexpr,
        BS: tl.constexpr,
        TILE_KV: tl.constexpr,  # sub-tile size for KV dimension (fits shared memory)
        KV_TILES: tl.constexpr,  # BS // TILE_KV
        DH: tl.constexpr,
        BLOCK_DH: tl.constexpr,
    ):
        """Fused block-sparse attention kernel with KV sub-tiling.

        Each program handles one (batch, kv_head, query_block, q_sub_tile) quad.
        Uses TILE_KV-sized sub-tiles for K/V to stay within shared memory limits.

        BlockNeighbors: [H_kv, N, K] int32, -1 for padding.
        """
        # Decode program ID → (batch, kv_head, query_block, q_tile)
        pid = tl.program_id(0)
        q_tile = pid % KV_TILES
        rest = pid // KV_TILES
        q_blk = rest % NUM_BLOCKS
        bh = rest // NUM_BLOCKS
        h_kv = bh % HKV
        b_idx = bh // HKV

        # This program handles rows [q_tile*TILE_KV, (q_tile+1)*TILE_KV) within the block
        offs_row = q_tile * TILE_KV + tl.arange(0, TILE_KV)  # [TILE_KV]
        offs_dh = tl.arange(0, BLOCK_DH)
        dh_mask = offs_dh < DH

        q_start = q_blk * BS
        q_positions = q_start + offs_row  # [TILE_KV]
        q_valid = q_positions < SEQ_LEN

        neigh_base = BlockNeighbors + h_kv * sn_h + q_blk * sn_n
        k_head_base = K + b_idx * sk_b + h_kv * sk_h
        v_head_base = V + b_idx * sv_b + h_kv * sv_h

        for g in range(GROUPS):
            h_q = h_kv * GROUPS + g

            # Load Q sub-tile: [TILE_KV, DH]
            q_ptrs = (
                Q + b_idx * sq_b + h_q * sq_h
                + q_positions[:, None] * sq_t + offs_dh[None, :] * sq_d
            )
            q_tile_data = (
                tl.load(q_ptrs, mask=q_valid[:, None] & dh_mask[None, :], other=0.0)
                .to(tl.float32) * scale
            )

            row_max = tl.full([TILE_KV], float("-inf"), dtype=tl.float32)
            row_sum = tl.zeros([TILE_KV], dtype=tl.float32)
            acc = tl.zeros([TILE_KV, BLOCK_DH], dtype=tl.float32)

            for ki in range(MAX_DEGREE):
                neigh_idx = tl.load(neigh_base + ki * sn_k)
                is_valid_block = neigh_idx >= 0
                if is_valid_block:
                    kv_start = neigh_idx * BS

                    # Process neighbor block as KV_TILES sub-tiles of TILE_KV each
                    for kv_t in range(KV_TILES):
                        kv_offs = tl.arange(0, TILE_KV)
                        kv_positions = kv_start + kv_t * TILE_KV + kv_offs  # [TILE_KV]
                        kv_valid = kv_positions < SEQ_LEN

                        # Load K sub-tile: [TILE_KV, DH]
                        k_ptrs = (
                            k_head_base
                            + kv_positions[:, None] * sk_t
                            + offs_dh[None, :] * sk_d
                        )
                        k_sub = tl.load(
                            k_ptrs, mask=kv_valid[:, None] & dh_mask[None, :], other=0.0
                        ).to(tl.float32)

                        # Scores: [TILE_KV, TILE_KV]
                        scores = tl.dot(q_tile_data, tl.trans(k_sub))

                        causal = kv_positions[None, :] <= q_positions[:, None]
                        valid_mask = q_valid[:, None] & kv_valid[None, :] & causal
                        scores = tl.where(valid_mask, scores, float("-inf"))

                        tile_max = tl.max(scores, axis=1)
                        new_max = tl.maximum(row_max, tile_max)
                        alpha = tl.exp(row_max - new_max)
                        p = tl.exp(scores - new_max[:, None])
                        p = tl.where(valid_mask, p, 0.0)

                        # Load V sub-tile: [TILE_KV, DH]
                        v_ptrs = (
                            v_head_base
                            + kv_positions[:, None] * sv_t
                            + offs_dh[None, :] * sv_d
                        )
                        v_sub = tl.load(
                            v_ptrs, mask=kv_valid[:, None] & dh_mask[None, :], other=0.0
                        ).to(tl.float32)

                        acc = alpha[:, None] * acc + tl.dot(p, v_sub)
                        row_sum = alpha * row_sum + tl.sum(p, axis=1)
                        row_max = new_max

            safe_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
            acc = acc / safe_sum[:, None]
            acc = tl.where((row_sum > 0.0)[:, None], acc, 0.0)

            out_ptrs = (
                Out + b_idx * so_b + h_q * so_h
                + q_positions[:, None] * so_t + offs_dh[None, :] * so_d
            )
            tl.store(
                out_ptrs, acc.to(Out.dtype.element_ty),
                mask=q_valid[:, None] & dh_mask[None, :],
            )


def triton_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_neighbors: torch.Tensor,
    block_size: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Fused block-sparse GQA attention via Triton.

    Replaces ``wayfinder_block_sparse_sdpa_attention`` with a single Triton
    kernel launch that fuses block-sparse gather, score computation, causal
    masking, and online softmax.

    Args:
        q: [B, H_q, T, dh] query states (bfloat16/float16/float32).
        k: [B, H_kv, T, dh] key states.
        v: [B, H_kv, T, dh] value states.
        block_neighbors: [H_kv, N, K] int32 block-level neighbor indices.
            N = ceil(T / block_size), K = max neighbor degree.
            Padded with -1 for unused slots.
        block_size: Token-level block size (e.g., 128).
        num_key_value_groups: GQA ratio (H_q / H_kv).

    Returns:
        [B, H_q, T, dh] attention output in v.dtype.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    # ── Input validation ──────────────────────────────────────────────
    if q.ndim != 4:
        raise ValueError(f"q must be [B, H_q, T, dh], got shape {tuple(q.shape)}")
    if k.ndim != 4:
        raise ValueError(f"k must be [B, H_kv, T, dh], got shape {tuple(k.shape)}")
    if v.ndim != 4:
        raise ValueError(f"v must be [B, H_kv, T, dh], got shape {tuple(v.shape)}")
    if block_neighbors.ndim != 3:
        raise ValueError(
            f"block_neighbors must be [H_kv, N, K], got shape {tuple(block_neighbors.shape)}"
        )

    B, H_q, T, dh = q.shape
    _, H_kv, _, _ = k.shape
    groups = int(num_key_value_groups)

    if H_q != H_kv * groups:
        raise ValueError(f"H_q={H_q} != H_kv={H_kv} * num_key_value_groups={groups}")
    if block_neighbors.shape[0] != H_kv:
        raise ValueError(f"block_neighbors dim 0 ({block_neighbors.shape[0]}) != H_kv ({H_kv})")

    BS = int(block_size)
    N = (T + BS - 1) // BS  # number of blocks (ceiling division)

    if block_neighbors.shape[1] < N:
        raise ValueError(
            f"block_neighbors has {block_neighbors.shape[1]} blocks but need at least {N} "
            f"(T={T}, BS={BS})"
        )

    K_deg = int(block_neighbors.shape[2])  # max neighbor degree

    if T == 0:
        return torch.zeros_like(q).to(dtype=v.dtype)

    # ── Prepare tensors ───────────────────────────────────────────────
    out = torch.empty(B, H_q, T, dh, device=q.device, dtype=v.dtype)

    # Ensure block_neighbors is int32 and contiguous on the right device
    bn = block_neighbors.to(device=q.device, dtype=torch.int32).contiguous()

    # Constexpr block dimensions — must be powers of 2 for Triton
    BLOCK_DH = _next_power_of_2(dh)

    # Verify BS is power of 2 (required for tl.dot alignment)
    if BS & (BS - 1) != 0:
        raise ValueError(f"block_size must be a power of 2, got {BS}. Use 64, 128, or 256.")

    # Choose TILE_KV to fit within shared memory limits.
    # GB10 (sm_121) has 101KB shared memory. tl.dot needs both operands staged:
    # [TILE_KV, DH] + [TILE_KV, DH] + [TILE_KV, TILE_KV] ≈ TILE_KV*(2*DH+TILE_KV)*4 bytes.
    # TILE_KV=32, DH=128 → 32*(256+32)*4 = 36KB → fits safely.
    TILE_KV = min(32, BS)
    KV_TILES = BS // TILE_KV

    # ── Scale ─────────────────────────────────────────────────────────
    scale = 1.0 / math.sqrt(float(dh))

    # ── Launch kernel ─────────────────────────────────────────────────
    # One program per (batch, kv_head, query_block, q_sub_tile)
    grid = (B * H_kv * N * KV_TILES,)

    # num_stages=1 prevents Triton from double-buffering loads in shared memory,
    # which would exceed GB10's 101KB shared memory limit.
    # num_warps=4 is conservative for TILE_KV=32.
    _fused_block_sparse_gqa_attn_fwd[grid](
        q,
        k,
        v,
        out,
        bn,
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
        # BlockNeighbors strides
        bn.stride(0),
        bn.stride(1),
        bn.stride(2),
        # Scalar params
        scale,
        SEQ_LEN=T,
        NUM_BLOCKS=N,
        MAX_DEGREE=K_deg,
        GROUPS=groups,
        HKV=H_kv,
        BS=BS,
        TILE_KV=TILE_KV,
        KV_TILES=KV_TILES,
        DH=dh,
        BLOCK_DH=BLOCK_DH,
        num_warps=4,
        num_stages=1,
    )

    return out
