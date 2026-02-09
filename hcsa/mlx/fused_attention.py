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
    """Return True when the fused all-head *full-prefill* path can be used."""
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


def _fused_active_dispatch_eligible(
    *,
    all_perms: mx.array,
    window_drop_prob: float,
    training: bool,
    multi_cycle_mode: str,
    use_fused_dispatch: bool,
) -> bool:
    """Return True when the vectorized active-row path can be used.

    More permissive than ``_fused_dispatch_eligible`` because the vectorized
    active-row implementation supports circular wrapping and edge-type bias.
    """
    if not use_fused_dispatch:
        return False
    if all_perms.ndim != 2:
        return False
    if training and window_drop_prob > 0.0:
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
    circular: bool = False,
    edge_type_bias_scalar: Optional[float] = None,
) -> mx.array:
    """All-head vectorized active-row permute-window attention.

    Two strategies depending on whether the graph size matches the data:

    **Full-prefill reuse** (``Tg == Tk``):  Pre-permute Q/K/V into cycle
    order, run windowed SDPA with contiguous slicing (zero random-access
    gathers), then extract active rows.  This is identical in structure to
    the full-prefill fused path and runs at the same speed.

    **Flat-index gather** (``Tg > Tk``, legacy oversized graph):
    Vectorized gather across all heads using flat permutation tables +
    SDPA with ``NH = Hq * Qblk`` virtual heads.

    Args:
        q: [B, Hq, Tq, dh] — active query rows only
        k, v: [B, Hkv, Tk, dh] — full KV cache prefix
        all_perms: [Hq, Tg] int32 — cycle permutations (Tg >= Tk)
        all_inv_perms: [Hq, Tg] int32 — inverse permutations
        query_positions: [Tq] int32 — original token positions for each query
        window: half-window size
        query_chunk_size: query positions per chunk (memory bounding)
        circular: modular wrap-around in cycle order.
        edge_type_bias_scalar: additive bias for cycle-adjacent positions.

    Returns:
        y: [B, Hq, Tq, dh]
    """
    B, Hq, Tq, dh = q.shape
    Hkv = k.shape[1]
    Tk = k.shape[2]
    Tg = int(all_perms.shape[1])

    # --- Fast path: full-prefill reuse when graph matches data ---
    # When Tg == Tk, we can pre-permute K/V once and use contiguous
    # window slicing — identical to the full-prefill fused path but
    # with a padded Q tensor.  This eliminates all random-access gathers.
    if Tg == Tk and (edge_type_bias_scalar is None or edge_type_bias_scalar == 0.0):
        return _active_via_full_prefill(
            q, k, v,
            all_perms=all_perms,
            all_inv_perms=all_inv_perms,
            query_positions=query_positions,
            window=window,
            query_chunk_size=query_chunk_size,
        )

    # --- Fallback: flat-index gather for oversized graphs (Tg > Tk) ---
    return _active_via_gather(
        q, k, v,
        all_perms=all_perms,
        all_inv_perms=all_inv_perms,
        query_positions=query_positions,
        window=window,
        query_chunk_size=query_chunk_size,
        circular=circular,
        edge_type_bias_scalar=edge_type_bias_scalar,
    )


def _active_via_full_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    query_positions: mx.array,
    window: int,
    query_chunk_size: int = 384,
) -> mx.array:
    """Active-row via full-prefill: pad Q, run full T attention, extract rows.

    Pre-permutes K/V once (contiguous) and uses the same windowed SDPA
    as the full-prefill path.  This trades ~2x extra SDPA work for
    eliminating all random-access K/V gathers.
    """
    B, Hq, Tq, dh = q.shape
    Hkv = k.shape[1]
    T = k.shape[2]  # Tg == Tk
    scale = 1.0 / math.sqrt(dh)
    q_chunk = int(max(1, min(query_chunk_size, T)))

    # Build full-T Q tensor with active rows placed at their positions.
    q_pos = query_positions.astype(mx.int32)  # [Tq]
    q_full = mx.zeros((B, Hq, T, dh), dtype=q.dtype)
    # Scatter active rows into full Q.
    # query_positions are contiguous: active_start .. active_start + Tq - 1
    active_start = int(q_pos[0].item())
    q_full = q_full.at[:, :, active_start:active_start + Tq, :].add(q)

    # GQA expansion for pre-permute.
    if Hkv < Hq:
        if Hq % Hkv != 0:
            raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}")
        repeats = Hq // Hkv
        k_exp = mx.repeat(k, repeats=repeats, axis=1)
        v_exp = mx.repeat(v, repeats=repeats, axis=1)
    else:
        k_exp = k
        v_exp = v

    # Pre-permute into cycle order (one gather per head, contiguous after).
    perm_idx = all_perms[None, :, :, None]  # [1, Hq, T, 1]
    perm_idx_b = mx.broadcast_to(perm_idx, (B, Hq, T, 1))
    q_pi = mx.take_along_axis(q_full, perm_idx_b, axis=2)
    k_pi = mx.take_along_axis(k_exp, perm_idx_b, axis=2)
    v_pi = mx.take_along_axis(v_exp, perm_idx_b, axis=2)

    # Run windowed SDPA (identical to full-prefill fused path).
    y_pi_chunks: list[mx.array] = []
    for s in range(0, T, q_chunk):
        e = min(T, s + q_chunk)
        ks = max(0, s - window)
        ke = min(T, e + window)
        Qblk = e - s
        Kblk = ke - ks

        q_blk = q_pi[:, :, s:e, :]
        k_blk = k_pi[:, :, ks:ke, :]
        v_blk = v_pi[:, :, ks:ke, :]

        # Causal mask from original positions.
        q_idx = all_perms[:, s:e]
        k_idx = all_perms[:, ks:ke]
        causal = k_idx[:, None, :] <= q_idx[:, :, None]

        # Window constraint in permuted order.
        q_r = mx.arange(s, e, dtype=mx.int32).reshape(1, Qblk, 1)
        k_r = mx.arange(ks, ke, dtype=mx.int32).reshape(1, 1, Kblk)
        rel = k_r - q_r
        in_window = (rel >= -window) & (rel <= window)

        mask = in_window & causal

        y_blk = mx.fast.scaled_dot_product_attention(
            q_blk, k_blk, v_blk,
            scale=scale,
            mask=mask[None, :, :, :],
        ).astype(v.dtype)
        y_pi_chunks.append(y_blk)

    if len(y_pi_chunks) == 1:
        y_pi = y_pi_chunks[0]
    else:
        y_pi = mx.concatenate(y_pi_chunks, axis=2)

    # Inverse-permute back to original order.
    inv_idx = all_inv_perms[None, :, :, None]
    inv_idx_b = mx.broadcast_to(inv_idx, (B, Hq, T, 1))
    y_full = mx.take_along_axis(y_pi, inv_idx_b, axis=2).astype(v.dtype)

    # Extract active rows.
    y = y_full[:, :, active_start:active_start + Tq, :]
    mx.eval(y)
    return y


def _active_via_gather(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    all_perms: mx.array,
    all_inv_perms: mx.array,
    query_positions: mx.array,
    window: int,
    query_chunk_size: int = 192,
    circular: bool = False,
    edge_type_bias_scalar: Optional[float] = None,
) -> mx.array:
    """Active-row via flat-index gather (for oversized graphs, Tg > Tk)."""
    B, Hq, Tq, dh = q.shape
    Hkv = k.shape[1]
    Tk = k.shape[2]
    Tg = int(all_perms.shape[1])
    scale = 1.0 / math.sqrt(dh)
    W_full = 2 * window + 1
    q_chunk = int(max(1, min(query_chunk_size, Tq)))

    kv_repeat = Hq // Hkv
    q_pos_all = query_positions.astype(mx.int32)

    perm_i32 = all_perms.astype(mx.int32)
    inv_i32 = all_inv_perms.astype(mx.int32)
    flat_perm = perm_i32.reshape(-1)
    flat_inv = inv_i32.reshape(-1)

    head_off = (mx.arange(Hq, dtype=mx.int32) * Tg).reshape(Hq, 1, 1)
    offsets = mx.arange(-window, window + 1, dtype=mx.int32).reshape(1, 1, W_full)

    kv_head_idx = mx.arange(Hq, dtype=mx.int32) // kv_repeat
    kv_head_off = (kv_head_idx * Tk).reshape(Hq, 1, 1)

    k_flat = k.reshape(B, Hkv * Tk, dh)
    v_flat = v.reshape(B, Hkv * Tk, dh)

    y_chunks: list[mx.array] = []
    for s in range(0, Tq, q_chunk):
        e = min(Tq, s + q_chunk)
        Qblk = e - s
        q_blk = q[:, :, s:e, :]
        q_pos = q_pos_all[s:e]

        inv_idx = head_off[:, :, 0] + q_pos[None, :]
        all_ranks = mx.take(flat_inv, inv_idx.reshape(-1), axis=0).reshape(
            Hq, Qblk
        )

        k_ranks = all_ranks[:, :, None] + offsets

        if circular:
            k_ranks_clipped = (k_ranks % Tg).astype(mx.int32)
            valid = mx.ones(k_ranks.shape, dtype=mx.bool_)
        else:
            valid = (k_ranks >= 0) & (k_ranks < Tg)
            k_ranks_clipped = mx.clip(k_ranks, 0, Tg - 1).astype(mx.int32)

        perm_idx = (head_off + k_ranks_clipped).reshape(-1)
        k_orig = mx.take(flat_perm, perm_idx, axis=0).reshape(
            Hq, Qblk, W_full
        ).astype(mx.int32)

        available = k_orig < Tk
        causal = k_orig <= q_pos[None, :, None]
        mask_eff = valid & available & causal

        gather = mx.clip(k_orig, 0, Tk - 1) + kv_head_off
        gather_flat = gather.reshape(-1)

        gi_exp = mx.broadcast_to(
            gather_flat[None, :, None], (B, Hq * Qblk * W_full, dh)
        )
        k_gathered = mx.take_along_axis(k_flat, gi_exp, axis=1).reshape(
            B, Hq, Qblk, W_full, dh
        )
        v_gathered = mx.take_along_axis(v_flat, gi_exp, axis=1).reshape(
            B, Hq, Qblk, W_full, dh
        )

        NH = Hq * Qblk
        q_sdpa = q_blk.reshape(B, NH, 1, dh)
        k_sdpa = k_gathered.reshape(B, NH, W_full, dh)
        v_sdpa = v_gathered.reshape(B, NH, W_full, dh)

        mask_dtype = q.dtype
        neg_val = mx.array(
            -1e4 if mask_dtype == mx.float16 else -1e9, dtype=mask_dtype,
        )
        zero_val = mx.array(0.0, dtype=mask_dtype)
        if edge_type_bias_scalar is not None and edge_type_bias_scalar != 0.0:
            cycle_nb = valid & (
                (k_ranks == (all_ranks[:, :, None] - 1))
                | (k_ranks == (all_ranks[:, :, None] + 1))
            )
            bias = mx.where(
                mask_eff,
                cycle_nb.astype(mask_dtype) * mx.array(
                    edge_type_bias_scalar, dtype=mask_dtype
                ),
                neg_val,
            )
            sdpa_mask = bias.reshape(1, NH, 1, W_full)
        else:
            sdpa_mask = mx.where(
                mask_eff.reshape(1, NH, 1, W_full),
                zero_val,
                neg_val,
            )

        y_blk = mx.fast.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            scale=scale,
            mask=sdpa_mask,
        ).reshape(B, Hq, Qblk, dh).astype(v.dtype)

        mx.eval(y_blk)
        y_chunks.append(y_blk)

    if len(y_chunks) == 1:
        return y_chunks[0]
    return mx.concatenate(y_chunks, axis=2)


def wayfinder_fused_permute_window_attention_active_metal(
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
    """K6 Metal fused active-row attention (all heads)."""
    from hcsa.mlx.kernels.metal import fused_attention_kernel

    kernel = fused_attention_kernel()
    perm_i32 = all_perms.astype(mx.int32)
    inv_i32 = all_inv_perms.astype(mx.int32)
    q_pos_all = query_positions.astype(mx.int32)
    window_arr = mx.array([int(window)], dtype=mx.int32)

    B, Hq, Tq, dh = q.shape
    q_chunk = int(max(1, min(query_chunk_size, Tq)))
    y_chunks: list[mx.array] = []
    for s in range(0, Tq, q_chunk):
        e = min(Tq, s + q_chunk)
        q_blk = q[:, :, s:e, :]
        q_pos_blk = q_pos_all[s:e]
        y_blk = kernel(
            q_blk,
            k,
            v,
            perm_i32,
            inv_i32,
            q_pos_blk,
            window_arr,
            output_shapes=[q_blk.shape],
            output_dtypes=[q_blk.dtype],
        )[0]
        y_chunks.append(y_blk)

    if len(y_chunks) == 1:
        return y_chunks[0]
    return mx.concatenate(y_chunks, axis=2)
