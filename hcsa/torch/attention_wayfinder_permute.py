from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hcsa.graph.abi import EdgeType
from hcsa.torch.bench_utils import now_ms, stable_masked_softmax

# ---------------------------------------------------------------------------
# flex_attention availability
# ---------------------------------------------------------------------------
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _raw_flex_attention,
        create_block_mask as _create_block_mask,
    )
    _compiled_flex_attention = torch.compile(_raw_flex_attention)
    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    _compiled_flex_attention = None
    _create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


def _extract_cycle_perms_from_meta(meta: Dict[str, Any], n_heads: int) -> list[list[int] | None]:
    cycle_perms = meta.get("cycle_perms")
    if isinstance(cycle_perms, list):
        out: list[list[int] | None] = []
        for h in range(n_heads):
            perm_h = cycle_perms[h] if h < len(cycle_perms) else None
            out.append(None if perm_h is None else [int(x) for x in perm_h])
        return out
    return [None for _ in range(n_heads)]


def _recover_cycle_perm_from_edges(neigh_h: np.ndarray, edge_h: np.ndarray) -> list[int] | None:
    """Recover one Hamiltonian cycle order from cycle edges if possible.

    This is a fallback when meta['cycle_perms'] is absent. It assumes cycle edges
    induce a connected graph with each node having at least one cycle neighbor.
    """
    t = int(neigh_h.shape[0])
    if t <= 0:
        return None

    cycle_adj: list[list[int]] = [[] for _ in range(t)]
    for i in range(t):
        for j, et in zip(neigh_h[i].tolist(), edge_h[i].tolist()):
            if int(j) < 0:
                continue
            if int(et) == int(EdgeType.CYCLE):
                cycle_adj[i].append(int(j))

    if t == 1:
        return [0]

    if any(len(n) == 0 for n in cycle_adj):
        return None

    # Greedy walk over cycle edges to produce a permutation. When multi-cycle union
    # gives degree >2, pick the smallest unvisited neighbor for determinism.
    perm = [0]
    visited = {0}
    prev = -1

    while len(perm) < t:
        cur = perm[-1]
        neighbors = sorted(cycle_adj[cur])
        nxt = None

        for cand in neighbors:
            if cand != prev and cand not in visited:
                nxt = cand
                break

        if nxt is None:
            for cand in neighbors:
                if cand not in visited:
                    nxt = cand
                    break

        if nxt is None:
            remaining = [i for i in range(t) if i not in visited]
            if not remaining:
                break
            nxt = remaining[0]

        prev = cur
        perm.append(int(nxt))
        visited.add(int(nxt))

    if len(perm) != t:
        return None

    if len(set(perm)) != t:
        return None

    return perm


def recover_cycle_perms(
    neigh_idx: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    meta: Dict[str, Any] | None,
) -> list[list[int] | None]:
    """Get one cycle permutation per head from meta or cycle edges.

    neigh_idx/edge_type are [H,T,D] tensors.
    """
    if neigh_idx.ndim != 3 or edge_type.ndim != 3:
        raise ValueError("recover_cycle_perms expects [H,T,D] tensors")

    h = int(neigh_idx.shape[0])
    perms = _extract_cycle_perms_from_meta(meta or {}, h)

    if all(p is not None for p in perms):
        return perms

    neigh_np = neigh_idx.detach().cpu().numpy().astype(np.int32)
    edge_np = edge_type.detach().cpu().numpy().astype(np.uint8)

    for head in range(h):
        if perms[head] is not None:
            continue
        perms[head] = _recover_cycle_perm_from_edges(neigh_np[head], edge_np[head])

    return perms


def permute_window_attention_single(
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    *,
    perm: Sequence[int] | np.ndarray,
    window: int,
    return_weights: bool = False,
    pre_perm: torch.Tensor | None = None,
    pre_inv_perm: torch.Tensor | None = None,
    pre_pi_idx_clamped: torch.Tensor | None = None,
    pre_valid_mask: torch.Tensor | None = None,
    pre_causal_mask: torch.Tensor | None = None,
    edge_type_bias_scalar: float | None = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
    """Single-head permute-to-cycle-order local-window attention.

    q_h/k_h/v_h are [B,T,dh].
    """
    _b, t, dh = q_h.shape
    if window < 0:
        raise ValueError("window must be >= 0")

    t0 = now_ms()

    if pre_perm is None:
        perm_t = torch.as_tensor(perm, device=q_h.device, dtype=torch.long)
        if perm_t.shape != (t,):
            raise ValueError(f"perm must be shape ({t},), got {tuple(perm_t.shape)}")
        inv_perm = torch.argsort(perm_t)

        w = 2 * window + 1
        offsets = torch.arange(-window, window + 1, device=q_h.device, dtype=torch.long)
        pi_idx = torch.arange(t, device=q_h.device, dtype=torch.long).view(t, 1) + offsets.view(1, w)
        valid = (pi_idx >= 0) & (pi_idx < t)
        pi_idx_clamped = pi_idx.clamp(0, t - 1)
        permute_ms = now_ms() - t0
    else:
        perm_t = pre_perm
        inv_perm = pre_inv_perm
        pi_idx_clamped = pre_pi_idx_clamped
        valid = pre_valid_mask
        permute_ms = 0.0

    q_pi = q_h[:, perm_t]
    k_pi = k_h[:, perm_t]
    v_pi = v_h[:, perm_t]

    t1 = now_ms()
    k_win = k_pi[:, pi_idx_clamped]
    v_win = v_pi[:, pi_idx_clamped]

    scores = torch.sum(q_pi[:, :, None, :].float() * k_win.float(), dim=-1) / math.sqrt(float(dh))

    if edge_type_bias_scalar is not None and float(edge_type_bias_scalar) != 0.0:
        w_actual = int(pi_idx_clamped.shape[-1])
        center = w_actual // 2
        bias_vec = torch.zeros((w_actual,), device=q_h.device, dtype=torch.float32)
        if center > 0:
            bias_vec[center - 1] = float(edge_type_bias_scalar)
        if center + 1 < w_actual:
            bias_vec[center + 1] = float(edge_type_bias_scalar)
        scores = scores + bias_vec.view(1, 1, w_actual)

    if pre_causal_mask is not None:
        mask = valid & pre_causal_mask
    else:
        orig_idx = perm_t
        neigh_orig = orig_idx[pi_idx_clamped]
        query_orig = orig_idx.view(t, 1)
        mask = valid & (neigh_orig <= query_orig)

    if training and window_drop_prob > 0.0:
        w_actual = int(pi_idx_clamped.shape[-1])
        center = w_actual // 2
        preserve = torch.zeros((t, w_actual), device=q_h.device, dtype=torch.bool)
        preserve[:, center] = True
        if center > 0:
            preserve[:, center - 1] = True
        if center + 1 < w_actual:
            preserve[:, center + 1] = True

        drop_rand = torch.rand((t, w_actual), device=q_h.device) < float(window_drop_prob)
        drop = drop_rand & (~preserve)
        mask = mask & (~drop)

    weights = stable_masked_softmax(scores.float(), mask[None, :, :], dim=-1)
    y_pi = torch.sum(weights[:, :, :, None] * v_win.float(), dim=2)
    y_h = y_pi[:, inv_perm].to(dtype=v_h.dtype)
    attention_ms = now_ms() - t1

    if return_weights:
        return y_h, weights, permute_ms, attention_ms
    return y_h, None, permute_ms, attention_ms


def wayfinder_permute_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    neigh_idx: torch.Tensor,
    edge_type: torch.Tensor,
    graph_meta: Dict[str, Any] | None = None,
    return_weights: bool = False,
    cache: Any | None = None,
    edge_type_bias_scalar: float | None = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, float, float]:
    """Fast path: permute to cycle order then contiguous local-window attention."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")

    _b, h, _t, _dh = q.shape
    if neigh_idx.ndim != 3 or edge_type.ndim != 3:
        raise ValueError("neigh_idx/edge_type must be [H,T,D]")

    if cache is not None:
        cycle_perms = [p.tolist() for p in cache.perm]
    else:
        cycle_perms = recover_cycle_perms(neigh_idx, edge_type, meta=graph_meta)

    ys: list[torch.Tensor] = []
    ws: list[torch.Tensor] = []
    permute_ms = 0.0
    attention_ms = 0.0

    for head in range(h):
        perm_h = cycle_perms[head] if head < len(cycle_perms) else None
        if perm_h is None:
            raise ValueError(f"Missing cycle permutation for head {head}")

        kwargs: Dict[str, Any] = {
            "perm": perm_h,
            "window": window,
            "return_weights": return_weights,
            "edge_type_bias_scalar": edge_type_bias_scalar,
            "window_drop_prob": window_drop_prob,
            "training": training,
        }
        if cache is not None:
            kwargs["pre_perm"] = cache.perm[head]
            kwargs["pre_inv_perm"] = cache.inv_perm[head]
            kwargs["pre_pi_idx_clamped"] = cache.pi_idx_clamped[head]
            kwargs["pre_valid_mask"] = cache.valid_mask[head]
            kwargs["pre_causal_mask"] = cache.causal_masks[head]

        y_h, w_h, p_ms, a_ms = permute_window_attention_single(q[:, head], k[:, head], v[:, head], **kwargs)
        ys.append(y_h)
        permute_ms += p_ms
        attention_ms += a_ms
        if return_weights and w_h is not None:
            ws.append(w_h)

    y = torch.stack(ys, dim=1)
    if return_weights:
        return y, torch.stack(ws, dim=1), permute_ms, attention_ms
    return y, None, permute_ms, attention_ms


# ═══════════════════════════════════════════════════════════════════════════
# Batched permute-window attention — all heads in parallel, no Python loop
# ═══════════════════════════════════════════════════════════════════════════

def wayfinder_permute_window_attention_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    pi_idx_clamped: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched permute-window attention — all heads processed in parallel.

    Args:
        q, k, v: [B, H, T, dh] in model dtype.
        perm: [H, T] long — maps cycle-position → original-position.
        inv_perm: [H, T] long — maps original-position → cycle-position.
        pi_idx_clamped: [H, T, 2W+1] long — local window indices in cycle space.
        valid_mask: [H, T, 2W+1] bool — valid (in-bounds) positions.
        causal_mask: [H, T, 2W+1] bool — causal constraint in original space.

    Returns:
        [B, H, T, dh] in the same dtype as v.
    """
    B, H, T, dh = q.shape
    W = pi_idx_clamped.shape[-1]  # 2*window + 1
    scale = dh ** -0.5

    # 1. Batched permute Q/K/V to cycle order
    perm_e = perm[None, :, :, None].expand(B, H, T, dh)
    q_pi = torch.gather(q, 2, perm_e)
    k_pi = torch.gather(k, 2, perm_e)
    v_pi = torch.gather(v, 2, perm_e)

    # 2. Gather local K/V windows: [B, H, T, 2W+1, dh]
    pi_flat = pi_idx_clamped.reshape(H, T * W)
    pi_e = pi_flat[None, :, :, None].expand(B, H, T * W, dh)
    k_win = torch.gather(k_pi, 2, pi_e).reshape(B, H, T, W, dh)
    v_win = torch.gather(v_pi, 2, pi_e).reshape(B, H, T, W, dh)

    # 3. Scores: [B,H,T,1,dh] @ [B,H,T,dh,W] → [B,H,T,W]
    scores = (q_pi.unsqueeze(3) @ k_win.transpose(-1, -2)).squeeze(3) * scale

    # 4. Mask + softmax
    mask = (valid_mask & causal_mask)[None]  # [1, H, T, W]
    scores.masked_fill_(~mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    weights = weights.nan_to_num_(0.0)

    # 5. Weighted sum: [B,H,T,1,W] @ [B,H,T,W,dh] → [B,H,T,dh]
    y_pi = (weights.unsqueeze(3) @ v_win).squeeze(3)

    # 6. Unpermute back to original order
    inv_e = inv_perm[None, :, :, None].expand(B, H, T, dh)
    return torch.gather(y_pi, 2, inv_e)


# ═══════════════════════════════════════════════════════════════════════════
# flex_attention — fused block-sparse kernel, optimal for large T
# ═══════════════════════════════════════════════════════════════════════════

def build_flex_block_mask(
    perm: torch.Tensor,
    *,
    window: int,
    B: int = 1,
    device: torch.device | None = None,
) -> Any:
    """Build a block mask for flex_attention in permuted cycle space.

    In permuted space:
      - Band constraint: |q_idx - kv_idx| <= window
      - Causal constraint: perm[h, kv_idx] <= perm[h, q_idx]
        (original position of key <= original position of query)

    The band structure gives excellent block sparsity: only 2-3 active blocks
    per query row instead of T/block_size.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention not available in this PyTorch build")

    H, T = perm.shape
    _perm = perm
    _window = window

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        in_band = (q_idx - kv_idx).abs() <= _window
        causal = _perm[h, kv_idx] <= _perm[h, q_idx]
        return in_band & causal

    return _create_block_mask(
        mask_mod, B=B, H=H, Q_LEN=T, KV_LEN=T,
        device=device or perm.device,
    )


def _permute_heads(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute x along dim 2 with per-head permutations. Memory-efficient.

    x: [B, H, T, dh], perm: [H, T] → out[b, h, i, d] = x[b, h, perm[h, i], d]
    Avoids materializing a [B, H, T, dh] int64 index tensor.
    """
    B, H, T, dh = x.shape
    h_idx = torch.arange(H, device=x.device).unsqueeze(1)  # [H, 1]
    return x[:, h_idx, perm]  # advanced indexing: [B, H, T, dh]


def wayfinder_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    kv_perm: torch.Tensor | None = None,
    block_mask: Any = None,
) -> tuple[torch.Tensor, Any]:
    """Flex-attention with block-sparse band pattern in permuted cycle space.

    Uses torch.nn.attention.flex_attention with a compiled block mask that
    exploits the band structure in cycle order. After permuting Q/K/V to
    cycle order, the attention pattern is a narrow band (width 2W+1) with
    per-position causal masking. flex_attention computes only the active
    blocks, giving O(T*W) work instead of O(T²).

    Supports GQA: when K/V have fewer heads than Q, pass ``kv_perm``
    ([H_kv, T]) for K/V permutation and ``enable_gqa=True`` is forwarded
    to the compiled flex_attention kernel.

    Args:
        q: [B, H_q, T, dh] query states.
        k: [B, H_kv, T, dh] key states (H_kv <= H_q for GQA).
        v: [B, H_kv, T, dh] value states.
        perm: [H_q, T] long — cycle-position → original-position (one per query head).
        inv_perm: [H_q, T] long — original-position → cycle-position.
        kv_perm: [H_kv, T] long — per KV-head permutation. If None, uses perm
                 (assumes H_q == H_kv, i.e. no GQA).
        block_mask: Cached block mask (from build_flex_block_mask), or None to build.

    Returns:
        (output [B, H_q, T, dh], block_mask for caching).
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention not available")

    B, Hq, T, dh = q.shape
    Hkv = k.shape[1]
    enable_gqa = Hkv < Hq

    # Permute Q with per-query-head perms
    q_pi = _permute_heads(q, perm)

    # Permute K/V with KV-head perms (fewer heads for GQA)
    _kv_p = kv_perm if kv_perm is not None else perm
    k_pi = _permute_heads(k, _kv_p)
    v_pi = _permute_heads(v, _kv_p)

    # Build or reuse block mask (always H_q heads for the query side)
    if block_mask is None:
        block_mask = build_flex_block_mask(perm, window=window, B=B, device=q.device)

    # Compiled flex_attention — fused block-sparse kernel
    y_pi = _compiled_flex_attention(
        q_pi, k_pi, v_pi,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )

    # Unpermute
    y = _permute_heads(y_pi, inv_perm)

    return y, block_mask
