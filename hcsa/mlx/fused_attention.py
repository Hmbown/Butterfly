"""Fused all-head permute-window attention dispatch.

Eliminates per-head-chunk ``mx.eval()`` barriers by building a single lazy
MLX compute graph for all query heads per query chunk, then evaluating once.

Falls back to the existing chunked path when ineligible (circular, union
multigraph, edge-type bias, retro backfill, window-drop during training,
or 3-D multi-cycle permutations).
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


def _fused_dispatch_eligible(
    *,
    all_perms: mx.array,
    edge_type_bias_scalar: Optional[float],
    window_drop_prob: float,
    training: bool,
    retro_backfill_enabled: bool,
    circular: bool,
    multi_cycle_mode: str,
    use_fused_dispatch: bool,
) -> bool:
    """Return True when the fused all-head path can be used."""
    if not use_fused_dispatch:
        return False
    # Multi-cycle 3-D permutations require per-cycle averaging.
    if all_perms.ndim != 2:
        return False
    # Edge-type bias needs manual matmul path.
    if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
        return False
    # Window-drop during training requires per-head random masks.
    if training and window_drop_prob > 0.0:
        return False
    if retro_backfill_enabled:
        return False
    if circular:
        return False
    if multi_cycle_mode == "union":
        return False
    return True


def wayfinder_fused_permute_window_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    window: int,
    query_chunk_size: int = 256,
) -> mx.array:
    """All-head fused permute-window attention.

    Processes all heads in a single lazy MLX graph per query chunk,
    eliminating per-head-chunk eval barriers.

    Args:
        q: [B, Hq, T, dh]
        k, v: [B, Hkv, T, dh]  (Hkv may be < Hq for GQA)
        all_perms: [Hq, T] int32 -- cycle permutations
        all_inv_perms: [Hq, T] int32 -- inverse permutations
        window: half-window size
        query_chunk_size: query positions per chunk (memory bounding)

    Returns:
        y: [B, Hq, T, dh]
    """
    B, Hq, T, dh = q.shape
    Hkv = k.shape[1]
    scale = 1.0 / math.sqrt(dh)
    q_chunk = int(max(1, min(query_chunk_size, T)))

    # --- GQA: expand K/V to match Hq (lazy broadcast) ---
    if Hkv < Hq:
        if Hq % Hkv != 0:
            raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}")
        repeats = Hq // Hkv
        k = mx.repeat(k, repeats=repeats, axis=1)  # [B, Hq, T, dh]
        v = mx.repeat(v, repeats=repeats, axis=1)

    # --- Pre-permute all heads at once ---
    # all_perms: [Hq, T] -> gather index [1, Hq, T, 1]
    perm_idx = all_perms[None, :, :, None]  # [1, Hq, T, 1]
    perm_idx = mx.broadcast_to(perm_idx, (B, Hq, T, 1))

    q_pi = mx.take_along_axis(q, perm_idx, axis=2)  # [B, Hq, T, dh]
    k_pi = mx.take_along_axis(k, perm_idx, axis=2)
    v_pi = mx.take_along_axis(v, perm_idx, axis=2)

    # --- Process query chunks ---
    y_pi_chunks: list[mx.array] = []
    for s in range(0, T, q_chunk):
        e = min(T, s + q_chunk)
        ks = max(0, s - window)
        ke = min(T, e + window)
        Qblk = e - s
        Kblk = ke - ks

        q_blk = q_pi[:, :, s:e, :]       # [B, Hq, Qblk, dh]
        k_blk = k_pi[:, :, ks:ke, :]     # [B, Hq, Kblk, dh]
        v_blk = v_pi[:, :, ks:ke, :]     # [B, Hq, Kblk, dh]

        # --- Per-head causal mask from original positions ---
        q_idx = all_perms[:, s:e]         # [Hq, Qblk]
        k_idx = all_perms[:, ks:ke]       # [Hq, Kblk]
        # causal: original_key_pos <= original_query_pos
        causal = k_idx[:, None, :] <= q_idx[:, :, None]  # [Hq, Qblk, Kblk]

        # Window constraint in permuted order
        q_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, Qblk, 1)
        k_pos = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, Kblk)
        rel = k_pos - q_pos  # [1, Qblk, Kblk]
        in_window = (rel >= -window) & (rel <= window)

        mask = in_window & causal  # [Hq, Qblk, Kblk]

        # SDPA expects mask broadcastable to [B, Hq, Qblk, Kblk]
        use_sdpa = (
            hasattr(mx, "fast")
            and hasattr(mx.fast, "scaled_dot_product_attention")
        )
        if use_sdpa:
            y_blk = mx.fast.scaled_dot_product_attention(
                q_blk,
                k_blk,
                v_blk,
                scale=scale,
                mask=mask[None, :, :, :],  # [1, Hq, Qblk, Kblk]
            ).astype(v.dtype)
        else:
            scores = mx.matmul(
                q_blk,
                k_blk.transpose(0, 1, 3, 2),
            ) * scale  # [B, Hq, Qblk, Kblk]
            neg_inf = mx.array(-1e30, dtype=scores.dtype)
            scores = mx.where(mask[None, :, :, :], scores, neg_inf)
            max_s = mx.max(scores, axis=-1, keepdims=True)
            scores = scores - mx.stop_gradient(max_s)
            w = mx.exp(scores)
            w = mx.where(mask[None, :, :, :], w, mx.array(0.0, dtype=w.dtype))
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-9)
            y_blk = mx.matmul(w, v_blk).astype(v.dtype)

        y_pi_chunks.append(y_blk)

    # --- Concatenate and inverse-permute ---
    if len(y_pi_chunks) == 1:
        y_pi = y_pi_chunks[0]
    else:
        y_pi = mx.concatenate(y_pi_chunks, axis=2)  # [B, Hq, T, dh]

    inv_idx = all_inv_perms[None, :, :, None]  # [1, Hq, T, 1]
    inv_idx = mx.broadcast_to(inv_idx, (B, Hq, T, 1))
    y = mx.take_along_axis(y_pi, inv_idx, axis=2).astype(v.dtype)

    return y


def wayfinder_fused_permute_window_attention_active(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    query_positions: mx.array,
    window: int,
    query_chunk_size: int = 192,
) -> mx.array:
    """All-head fused active-row permute-window attention.

    Loops over heads and query chunks like the chunked path but builds a
    single lazy MLX compute graph with **no** ``mx.eval()`` barriers.  MLX
    schedules all per-head work concurrently on the GPU.

    Args:
        q: [B, Hq, Tq, dh] — active query rows only
        k, v: [B, Hkv, Tk, dh] — full KV cache prefix
        all_perms: [Hq, Tg] int32 — cycle permutations (Tg >= Tk)
        all_inv_perms: [Hq, Tg] int32 — inverse permutations
        query_positions: [Tq] int32 — original token positions for each query
        window: half-window size (W_full = 2*window + 1)
        query_chunk_size: query positions per chunk (memory bounding)

    Returns:
        y: [B, Hq, Tq, dh]
    """
    import numpy as np

    from hcsa.mlx.attention import stable_masked_softmax

    B, Hq, Tq, dh = q.shape
    Hkv = k.shape[1]
    Tk = k.shape[2]
    Tg = int(all_perms.shape[1])
    scale = 1.0 / math.sqrt(dh)
    q_chunk = int(max(1, min(query_chunk_size, Tq)))

    kv_repeat = Hq // Hkv
    q_to_kv_head = np.arange(Hq, dtype=np.int32) // kv_repeat
    offsets = mx.arange(-window, window + 1, dtype=mx.int32)
    W_full = 2 * window + 1
    q_pos_all = query_positions.astype(mx.int32)

    # Build per-head results in a single lazy graph — NO mx.eval() calls.
    y_heads: list[mx.array] = []
    for h in range(Hq):
        q_h = q[:, h, :, :]  # [B, Tq, dh]
        kv_h = int(q_to_kv_head[h])
        k_h = k[:, kv_h, :, :]  # [B, Tk, dh]
        v_h = v[:, kv_h, :, :]  # [B, Tk, dh]

        perm_h = all_perms[h]  # [Tg]
        inv_h = all_inv_perms[h]  # [Tg]

        y_q_chunks: list[mx.array] = []
        for s in range(0, Tq, q_chunk):
            e = min(Tq, s + q_chunk)
            Qblk = e - s
            q_blk = q_h[:, s:e, :]  # [B, Qblk, dh]
            q_pos = q_pos_all[s:e]  # [Qblk]

            q_rank = mx.take(inv_h, q_pos, axis=0)  # [Qblk]
            k_rank = q_rank.reshape(-1, 1) + offsets.reshape(1, -1)  # [Qblk, W]

            valid = (k_rank >= 0) & (k_rank < Tg)
            k_rank_clipped = mx.clip(k_rank, 0, Tg - 1).astype(mx.int32)

            k_orig = mx.take(perm_h, k_rank_clipped, axis=0).astype(mx.int32)
            available = k_orig < Tk
            causal = k_orig <= q_pos.reshape(-1, 1)
            mask_eff = valid & available & causal  # [Qblk, W]

            gather_idx = mx.clip(k_orig, 0, Tk - 1)
            k_blk = mx.take(k_h, gather_idx, axis=1)  # [B, Qblk, W, dh]
            v_blk = mx.take(v_h, gather_idx, axis=1)  # [B, Qblk, W, dh]

            scores = (
                mx.matmul(
                    q_blk.astype(mx.float32).reshape(B * Qblk, 1, dh),
                    k_blk.astype(mx.float32).reshape(B * Qblk, W_full, dh)
                    .transpose(0, 2, 1),
                ).reshape(B, Qblk, W_full)
                * scale
            )
            w = stable_masked_softmax(
                scores, mask_eff[None, :, :], axis=-1, preserve_dtype=True,
            )
            y_blk = mx.matmul(
                w.reshape(B * Qblk, 1, W_full).astype(mx.float32),
                v_blk.astype(mx.float32).reshape(B * Qblk, W_full, dh),
            ).reshape(B, Qblk, dh).astype(v.dtype)

            y_q_chunks.append(y_blk)

        if len(y_q_chunks) == 1:
            y_h = y_q_chunks[0]
        else:
            y_h = mx.concatenate(y_q_chunks, axis=1)  # [B, Tq, dh]
        y_heads.append(y_h)

    # Stack all heads: [B, Hq, Tq, dh] — single lazy graph, no eval barriers.
    return mx.stack(y_heads, axis=1)
