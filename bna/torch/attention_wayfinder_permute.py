from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bna.cycles import block_hamiltonian_cycles, cycle_prev_next_from_perm, log_landmark_blocks, num_blocks_for_seq_len
from bna.graph.abi import EdgeType
from bna.topology.butterfly import (
    bit_reverse as butterfly_bit_reverse,
    butterfly_partner_bits,
    butterfly_partner_block,
    butterfly_stage_meta,
    butterfly_width,
)
from bna.torch.bench_utils import now_ms, stable_masked_softmax

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
    neigh_idx: torch.Tensor | None,
    edge_type: torch.Tensor | None,
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

    if cache is not None:
        cycle_perms = [p.tolist() for p in cache.perm]
    else:
        if neigh_idx is None or edge_type is None:
            raise ValueError("neigh_idx/edge_type are required when cache is not provided")
        if neigh_idx.ndim != 3 or edge_type.ndim != 3:
            raise ValueError("neigh_idx/edge_type must be [H,T,D]")
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
            if (
                head < len(cache.pi_idx_clamped)
                and head < len(cache.valid_mask)
                and head < len(cache.causal_masks)
            ):
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


@dataclass(frozen=True)
class BlockHamiltonianLayout:
    """Block-level sparse neighborhood reused by flex_attention.

    Shapes:
      - block_neighbors: [H_q, N, K]
      - block_perm: [H_q, N]
    """

    seq_len: int
    block_size: int
    num_blocks: int
    block_neighbors: torch.Tensor
    block_perm: torch.Tensor
    landmark_blocks: tuple[int, ...]
    sink_blocks: tuple[int, ...]
    num_cycles: int
    strategy: str
    topology_name: str = "hamiltonian"
    stage_idx: int = 0
    stage_count: int = 1
    local_window_blocks: int = 0
    partner_rule: str = "none"
    partner_count: int = 0


def _ordered_unique_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out


def _wayfinder_stage_meta(
    *,
    num_blocks: int,
    layer_idx: int,
    partner_rule: str,
) -> tuple[int, int]:
    return butterfly_stage_meta(
        num_blocks=int(num_blocks),
        layer_idx=int(layer_idx),
        partner_rule=str(partner_rule),
    )


def _wayfinder_partner_bits(
    *,
    stage_idx: int,
    stage_count: int,
    width: int,
    partner_count: int,
    partner_rule: str,
) -> list[int]:
    return butterfly_partner_bits(
        stage_idx=int(stage_idx),
        stage_count=int(stage_count),
        width=int(width),
        partner_count=int(partner_count),
        partner_rule=str(partner_rule),
    )


def _wayfinder_partner_block(
    *,
    block_idx: int,
    bit_idx: int,
    num_blocks: int,
    partner_rule: str,
    width: int,
) -> int | None:
    return butterfly_partner_block(
        block_idx=int(block_idx),
        bit_idx=int(bit_idx),
        num_blocks=int(num_blocks),
        partner_rule=str(partner_rule),
        width=int(width),
    )


def _ceil_log2(value: int) -> int:
    return butterfly_width(int(value))


def _bit_reverse(value: int, width: int) -> int:
    return butterfly_bit_reverse(int(value), int(width))


def build_block_hamiltonian_layout(
    *,
    seq_len: int,
    block_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    strategy: str = "random",
    num_cycles: int = 1,
    edge_disjoint: bool = True,
    regular_num_clusters: int = 8,
    seed: int = 0,
    landmark_blocks: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> BlockHamiltonianLayout:
    """Build block-level Hamiltonian neighborhoods for a GQA attention module."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if num_key_value_groups <= 0:
        raise ValueError("num_key_value_groups must be positive")

    target_device = device or torch.device("cpu")
    num_blocks = num_blocks_for_seq_len(seq_len, block_size)
    landmarks = tuple(
        int(block_idx)
        for block_idx in (
            log_landmark_blocks(num_blocks)
            if landmark_blocks is None
            else _ordered_unique_ints(
                block_idx for block_idx in landmark_blocks if 0 <= int(block_idx) < num_blocks
            )
        )
    )

    kv_perms: list[torch.Tensor] = []
    kv_neighbor_rows: list[list[list[int]]] = []
    max_degree = 1

    for head_idx in range(int(num_key_value_heads)):
        perms = block_hamiltonian_cycles(
            int(seq_len),
            int(block_size),
            strategy=strategy,
            num_cycles=int(num_cycles),
            edge_disjoint=bool(edge_disjoint),
            regular_num_clusters=int(regular_num_clusters),
            seed=int(seed),
            head_idx=head_idx,
            device=torch.device("cpu"),
        )
        kv_perms.append(perms[0].to(dtype=torch.long))
        cycle_links = [cycle_prev_next_from_perm(perm) for perm in perms]
        head_rows: list[list[int]] = []
        for block_idx in range(num_blocks):
            row: list[int] = [block_idx]
            for prev_h, next_h in cycle_links:
                row.append(int(prev_h[block_idx].item()))
                row.append(int(next_h[block_idx].item()))
            row.extend(landmarks)
            deduped = [value for value in _ordered_unique_ints(row) if 0 <= value < num_blocks]
            head_rows.append(deduped)
            max_degree = max(max_degree, len(deduped))
        kv_neighbor_rows.append(head_rows)

    kv_perm = torch.stack(kv_perms, dim=0)
    kv_neighbors = torch.full(
        (int(num_key_value_heads), num_blocks, max_degree),
        -1,
        dtype=torch.long,
    )
    for head_idx, rows in enumerate(kv_neighbor_rows):
        for block_idx, row in enumerate(rows):
            if row:
                kv_neighbors[head_idx, block_idx, : len(row)] = torch.tensor(row, dtype=torch.long)

    q_perm = kv_perm.repeat_interleave(int(num_key_value_groups), dim=0).to(device=target_device)
    q_neighbors = kv_neighbors.repeat_interleave(int(num_key_value_groups), dim=0).to(device=target_device)
    return BlockHamiltonianLayout(
        seq_len=int(seq_len),
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        block_neighbors=q_neighbors,
        block_perm=q_perm,
        landmark_blocks=landmarks,
        sink_blocks=(),
        num_cycles=int(num_cycles),
        strategy=str(strategy),
    )


def build_block_butterfly_layout(
    *,
    seq_len: int,
    block_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    layer_idx: int,
    local_window_blocks: int = 4,
    sink_count: int = 1,
    partner_count: int = 1,
    partner_rule: str = "xor",
    device: torch.device | None = None,
) -> BlockHamiltonianLayout:
    """Build the staged Butterfly block topology for static sparse attention.

    Each block attends to:
      - itself
      - a fixed number of prior local blocks
      - one or more deterministic partner blocks from a staged communication schedule
      - a small fixed set of sink blocks

    The stage changes by layer index to create global mixing across layers while
    keeping the per-layer degree small and compile-time predictable.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if num_key_value_groups <= 0:
        raise ValueError("num_key_value_groups must be positive")
    if local_window_blocks < 0:
        raise ValueError("local_window_blocks must be >= 0")
    if sink_count < 0:
        raise ValueError("sink_count must be >= 0")
    if partner_count < 0:
        raise ValueError("partner_count must be >= 0")
    if partner_rule not in {"xor", "bit_reversal", "benes"}:
        raise ValueError(f"Unsupported block-sparse Butterfly partner rule: {partner_rule!r}")

    target_device = device or torch.device("cpu")
    num_blocks = num_blocks_for_seq_len(seq_len, block_size)
    width = butterfly_width(num_blocks)
    stage_idx, stage_count = butterfly_stage_meta(
        num_blocks=int(num_blocks),
        layer_idx=int(layer_idx),
        partner_rule=partner_rule,
    )
    sink_blocks = tuple(range(min(int(sink_count), int(num_blocks))))
    identity_perm = torch.arange(int(num_blocks), dtype=torch.long)
    partner_bits = butterfly_partner_bits(
        stage_idx=int(stage_idx),
        stage_count=int(stage_count),
        width=int(width),
        partner_count=int(partner_count),
        partner_rule=partner_rule,
    )

    kv_neighbor_rows: list[list[list[int]]] = []
    max_degree = 1
    for _head_idx in range(int(num_key_value_heads)):
        head_rows: list[list[int]] = []
        for block_idx in range(int(num_blocks)):
            row: list[int] = [int(block_idx)]
            for offset in range(1, int(local_window_blocks) + 1):
                local_block = int(block_idx) - int(offset)
                if local_block < 0:
                    break
                row.append(local_block)
            for bit_idx in partner_bits:
                partner = butterfly_partner_block(
                    block_idx=int(block_idx),
                    bit_idx=int(bit_idx),
                    num_blocks=int(num_blocks),
                    partner_rule=partner_rule,
                    width=int(width),
                )
                if partner is not None:
                    row.append(int(partner))
            row.extend(int(block_idx) for block_idx in sink_blocks)
            deduped = [value for value in _ordered_unique_ints(row) if 0 <= value < int(num_blocks)]
            head_rows.append(deduped)
            max_degree = max(max_degree, len(deduped))
        kv_neighbor_rows.append(head_rows)

    kv_neighbors = torch.full(
        (int(num_key_value_heads), int(num_blocks), int(max_degree)),
        -1,
        dtype=torch.long,
    )
    for head_idx, rows in enumerate(kv_neighbor_rows):
        for block_idx, row in enumerate(rows):
            if row:
                kv_neighbors[head_idx, block_idx, : len(row)] = torch.tensor(row, dtype=torch.long)

    q_perm = identity_perm.repeat(int(num_key_value_heads) * int(num_key_value_groups), 1).to(device=target_device)
    q_neighbors = kv_neighbors.repeat_interleave(int(num_key_value_groups), dim=0).to(device=target_device)
    return BlockHamiltonianLayout(
        seq_len=int(seq_len),
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        block_neighbors=q_neighbors,
        block_perm=q_perm,
        landmark_blocks=(),
        sink_blocks=sink_blocks,
        num_cycles=1,
        strategy="butterfly",
        topology_name="butterfly",
        stage_idx=int(stage_idx),
        stage_count=int(stage_count),
        local_window_blocks=int(local_window_blocks),
        partner_rule=str(partner_rule),
        partner_count=int(partner_count),
    )


# Backward-compatible alias while the repo finishes the public rename.
build_block_wayfinder_layout = build_block_butterfly_layout


def build_block_hamiltonian_mask(
    layout: BlockHamiltonianLayout,
    *,
    device: torch.device | None = None,
) -> Any:
    """Compile a flex_attention block mask for block-Hamiltonian neighborhoods."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention not available in this PyTorch build")

    target_device = device or layout.block_neighbors.device
    block_neighbors = layout.block_neighbors.to(device=target_device)
    num_heads, num_blocks, _ = block_neighbors.shape

    allow = torch.zeros((num_heads, num_blocks, num_blocks), device=target_device, dtype=torch.bool)
    for head_idx in range(num_heads):
        for block_idx in range(num_blocks):
            row = block_neighbors[head_idx, block_idx]
            valid = row[row >= 0]
            if valid.numel() > 0:
                allow[head_idx, block_idx, valid] = True

    block_size = int(layout.block_size)
    max_block = int(num_blocks - 1)
    seq_len = int(layout.seq_len)
    _allow = allow

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        del b
        q_block = torch.clamp(torch.div(q_idx, block_size, rounding_mode="floor"), max=max_block)
        kv_block = torch.clamp(torch.div(kv_idx, block_size, rounding_mode="floor"), max=max_block)
        return _allow[h, q_block, kv_block] & (kv_idx <= q_idx)

    return _create_block_mask(
        mask_mod,
        B=None,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=target_device,
        BLOCK_SIZE=block_size,
    )


def wayfinder_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: Any,
) -> tuple[torch.Tensor, Any]:
    """Run block-Hamiltonian sparse attention directly in original token order."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention not available")

    enable_gqa = int(k.shape[1]) < int(q.shape[1])
    y = _compiled_flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )
    return y, block_mask


def _build_block_sparse_sdpa_index(
    layout: "BlockHamiltonianLayout",
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cached gather index and attention mask for block-sparse SDPA.

    Returns:
        idx_long: [N, KB] int64 — token-level gather indices
        attn_mask: [N, BS, KB] bool — combined causal + validity mask
    """
    BS = int(layout.block_size)
    N = int(layout.num_blocks)
    neighbors = layout.block_neighbors[0].to(device=device)  # [N, K]
    K = int(neighbors.shape[1])
    KB = K * BS

    offsets = torch.arange(BS, device=device, dtype=torch.int32)
    valid_mask = neighbors >= 0  # [N, K]
    safe_neighbors = neighbors.clamp(min=0).to(dtype=torch.int32)
    kv_token_idx = (safe_neighbors.unsqueeze(-1) * BS + offsets.view(1, 1, BS)).view(N, KB)
    kv_token_idx = kv_token_idx.clamp(max=T - 1)

    valid_tok = valid_mask.unsqueeze(-1).expand(N, K, BS).reshape(N, KB)

    q_positions = torch.arange(N * BS, device=device, dtype=torch.int32).view(N, BS)
    if N * BS > T:
        q_positions = q_positions.clamp(max=T - 1)
    kv_positions = kv_token_idx.clone()
    kv_positions[~valid_tok] = T
    attn_mask = (kv_positions.unsqueeze(1) <= q_positions.unsqueeze(2)) & valid_tok.unsqueeze(1)

    return kv_token_idx.long(), attn_mask


# Cache for block-sparse SDPA index/mask tensors.
# Key: (layout id, T, device) → (idx_long, attn_mask)
_BLOCK_SPARSE_SDPA_CACHE: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_block_sparse_sdpa_index(
    layout: "BlockHamiltonianLayout",
    T: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or build cached index/mask tensors for block-sparse SDPA."""
    key = (id(layout), T, str(device))
    cached = _BLOCK_SPARSE_SDPA_CACHE.get(key)
    if cached is not None:
        return cached
    result = _build_block_sparse_sdpa_index(layout, T, device)
    _BLOCK_SPARSE_SDPA_CACHE[key] = result
    # Evict old entries if cache grows too large
    if len(_BLOCK_SPARSE_SDPA_CACHE) > 32:
        oldest = next(iter(_BLOCK_SPARSE_SDPA_CACHE))
        del _BLOCK_SPARSE_SDPA_CACHE[oldest]
    return result


def wayfinder_block_sparse_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: "BlockHamiltonianLayout",
    block_chunk_size: int = 0,
) -> torch.Tensor:
    """Block-sparse attention using F.scaled_dot_product_attention.

    No torch.compile, no flex_attention — works on any CUDA arch including
    unsupported sm_121 (DGX Spark GB10).  Folds blocks into the batch
    dimension for efficient SDPA kernel launches.

    Args:
        q: [B, H_q, T, dh]
        k: [B, H_kv, T, dh]
        v: [B, H_kv, T, dh]
        layout: BlockHamiltonianLayout with block_neighbors [H_q, N, K]
        block_chunk_size: Process this many blocks per SDPA call (0 = all at once).
            Reduces peak memory at the cost of more kernel launches.

    Returns:
        output: [B, H_q, T, dh]
    """
    batch, H_q, T, dh = q.shape
    H_kv = k.shape[1]
    BS = int(layout.block_size)
    N = int(layout.num_blocks)
    device = q.device
    enable_gqa = H_kv < H_q

    # Get or build cached index/mask tensors
    idx_long, attn_mask = _get_block_sparse_sdpa_index(layout, T, device)
    KB = idx_long.shape[1]

    # Pad Q if sequence isn't block-aligned
    pad_len = N * BS - T
    if pad_len > 0:
        q_work = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
    else:
        q_work = q

    # Choose chunk size: 0 means process all blocks in one SDPA call
    C = N if block_chunk_size <= 0 else min(block_chunk_size, N)

    if C >= N:
        # --- Single-shot: all blocks at once ---
        q_blocks = q_work.view(batch, H_q, N, BS, dh).permute(0, 2, 1, 3, 4)
        q_blocks = q_blocks.reshape(batch * N, H_q, BS, dh)

        k_gathered = k[:, :, idx_long]  # [batch, H_kv, N, KB, dh]
        v_gathered = v[:, :, idx_long]
        k_blocks = k_gathered.permute(0, 2, 1, 3, 4).reshape(batch * N, H_kv, KB, dh)
        v_blocks = v_gathered.permute(0, 2, 1, 3, 4).reshape(batch * N, H_kv, KB, dh)

        mask_exp = attn_mask.unsqueeze(0).expand(batch, N, BS, KB)
        mask_exp = mask_exp.reshape(batch * N, 1, BS, KB)

        out_blocks = torch.nn.functional.scaled_dot_product_attention(
            q_blocks, k_blocks, v_blocks,
            attn_mask=mask_exp, enable_gqa=enable_gqa,
        )
        out = out_blocks.view(batch, N, H_q, BS, dh).permute(0, 2, 1, 3, 4)
        out = out.reshape(batch, H_q, N * BS, dh)
    else:
        # --- Chunked: process C blocks at a time to bound peak memory ---
        out_pieces: list[torch.Tensor] = []
        q_blocked = q_work.view(batch, H_q, N, BS, dh)  # [batch, H_q, N, BS, dh]

        for start in range(0, N, C):
            end = min(start + C, N)
            n_c = end - start

            q_c = q_blocked[:, :, start:end].permute(0, 2, 1, 3, 4)
            q_c = q_c.reshape(batch * n_c, H_q, BS, dh)

            idx_c = idx_long[start:end]  # [n_c, KB]
            k_c = k[:, :, idx_c].permute(0, 2, 1, 3, 4).reshape(batch * n_c, H_kv, KB, dh)
            v_c = v[:, :, idx_c].permute(0, 2, 1, 3, 4).reshape(batch * n_c, H_kv, KB, dh)

            mask_c = attn_mask[start:end].unsqueeze(0).expand(batch, n_c, BS, KB)
            mask_c = mask_c.reshape(batch * n_c, 1, BS, KB)

            out_c = torch.nn.functional.scaled_dot_product_attention(
                q_c, k_c, v_c,
                attn_mask=mask_c, enable_gqa=enable_gqa,
            )  # [batch*n_c, H_q, BS, dh]
            out_pieces.append(out_c.view(batch, n_c, H_q, BS, dh))

        # [batch, N, H_q, BS, dh]
        out = torch.cat(out_pieces, dim=1).permute(0, 2, 1, 3, 4)
        out = out.reshape(batch, H_q, N * BS, dh)

    if pad_len > 0:
        out = out[:, :, :T, :]
    return out


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
