from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cycles import (
    OnlineInsertionState,
    cycle_prev_next_from_perm,
    greedy_cycle,
    online_insertion_step,
    random_cycle,
)


def _build_neighbors_index(
    T: int,
    cycle_adj: List[List[int]],
    window: int,
    landmark_stride: int | None,
    *,
    include_self: bool = True,
    dedup: bool = True,
) -> torch.Tensor:
    """Build a neighbor index tensor [T, D] padded with -1.

    cycle_adj[i] contains the cycle-based neighbors for i (undirected).

    Neighbor candidates for i are:
      - cycle neighbors (from one or more cycles)
      - local causal window: {i-window, ..., i-1}
      - optional landmarks: {j | j % stride==0 and j < i}
      - optional self index i (ensures non-empty attention set)

    Causality filtering is done in the attention module (masking).
    """
    if window < 0:
        raise ValueError("window must be >= 0")
    if landmark_stride is not None and landmark_stride <= 0:
        raise ValueError("landmark_stride must be > 0 or None")

    n_landmarks = 0
    if landmark_stride is not None:
        n_landmarks = (T + landmark_stride - 1) // landmark_stride

    max_cycle_deg = max((len(n) for n in cycle_adj), default=0)
    D = max_cycle_deg + window + n_landmarks + (1 if include_self else 0)
    if D == 0:
        raise ValueError("Empty neighborhood: set include_self=True or window>0")

    out = torch.full((T, D), -1, dtype=torch.long)

    for i in range(T):
        neigh: List[int] = []
        neigh.extend(cycle_adj[i])
        if window:
            start = max(0, i - window)
            neigh.extend(list(range(start, i)))
        if landmark_stride is not None:
            neigh.extend([j for j in range(0, i, landmark_stride)])
        if include_self:
            neigh.append(i)

        if dedup:
            seen = set()
            deduped: List[int] = []
            for j in neigh:
                if j not in seen:
                    deduped.append(j)
                    seen.add(j)
            neigh = deduped

        if len(neigh) > D:
            # Should not happen, but keep it safe.
            neigh = neigh[:D]
        out[i, : len(neigh)] = torch.tensor(neigh, dtype=torch.long)

    return out


class HCSASelfAttention(nn.Module):
    """Hamiltonian Cycle Sparse Attention (HCSA).

    Dense attention is replaced by a sparse attention neighborhood constructed from:
      1) Hamiltonian cycle neighbor edges (undirected backbone)
      2) Always-on local causal window
      3) Optional landmark/global tokens

    Causality is enforced by masking out any neighbor j where j > i.

    Implementation uses a fixed-size neighbor index tensor neigh_idx: [T, D]
    gathered into K,V: [B,H,T,D,dh] via torch.gather.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        *,
        routing_dim: int | None = None,
        dropout: float = 0.0,
        window: int = 64,
        landmark_stride: int | None = 64,
        cycle: str = "random",
        num_cycles: int = 1,
        seed: int = 0,
        cache_online: bool = True,
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        if num_cycles < 1:
            raise ValueError("num_cycles must be >= 1")

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.routing_dim = routing_dim or self.head_dim
        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.cycle = cycle
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self.cache_online = bool(cache_online)

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)

        # Routing projection (shared across heads by default, output reshaped per head)
        self.Wr = nn.Linear(n_embd, n_heads * self.routing_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # CPU RNG for deterministic random cycles
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)
        self._call_id = 0

        # Online insertion cached state per head
        self._online: List[Optional[OnlineInsertionState]] = [None for _ in range(n_heads)]

    def _cycle_adjacency_for_head(
        self,
        r: torch.Tensor,
        head_idx: int,
        T: int,
    ) -> List[List[int]]:
        """Return cycle adjacency list for nodes 0..T-1 for a single head."""
        device = r.device

        perms: List[torch.Tensor] = []

        def _randperm() -> torch.Tensor:
            # Derive a per-head deterministic generator state by stepping the RNG.
            # (Simple & deterministic: each call consumes generator state.)
            p = random_cycle(T, generator=self._rng, device=torch.device("cpu")).to(device)
            return p

        if self.cycle == "random":
            for _ in range(self.num_cycles):
                perms.append(_randperm())
        elif self.cycle == "greedy":
            # Use different starts to diversify if num_cycles>1
            for k in range(self.num_cycles):
                start = (k * 997 + head_idx) % T
                perms.append(greedy_cycle(r, start=start))
        elif self.cycle == "online_insertion":
            # One cycle only for online insertion (union is possible but not implemented).
            if self.num_cycles != 1:
                raise ValueError("online_insertion currently supports num_cycles=1")

            cached = self._online[head_idx]
            if (
                self.cache_online
                and (not self.training)
                and cached is not None
                and cached.perm.numel() == T - 1
            ):
                state = online_insertion_step(cached, r)
            else:
                # Initialize with a simple 0..T-1 cycle.
                # Then update by insertion from T-1; this is cheap and stable.
                if T == 1:
                    state = OnlineInsertionState(perm=torch.zeros((1,), dtype=torch.long, device=device))
                elif T == 2:
                    state = OnlineInsertionState(perm=torch.tensor([0, 1], dtype=torch.long, device=device))
                else:
                    # Start with a random cycle over first (T-1) nodes, then insert the last.
                    base = random_cycle(T - 1, generator=self._rng, device=torch.device("cpu")).to(device)
                    state = online_insertion_step(OnlineInsertionState(perm=base), r)
            self._online[head_idx] = state
            perms.append(state.perm)
        else:
            raise ValueError(f"Unknown cycle strategy: {self.cycle}")

        # Build adjacency as union of neighbors from each cycle.
        adj: List[set[int]] = [set() for _ in range(T)]
        for perm in perms:
            prev, nxt = cycle_prev_next_from_perm(perm)
            # prev/nxt are per-node.
            for i in range(T):
                adj[i].add(int(prev[i].item()))
                adj[i].add(int(nxt[i].item()))

        return [sorted(list(s)) for s in adj]

    def forward(self, x: torch.Tensor, *, return_debug: bool = False):
        """Forward pass (vectorized gather + attention across all heads).

        The per-head cycle construction still runs in a Python loop (O(T) work,
        not the bottleneck).  The expensive gather and attention computation is
        batched over all heads in a single set of tensor operations.

        Args:
            x: [B, T, C]
            return_debug: if True, returns (y, debug_dict) where debug_dict includes
              neighbor indices and masks for head 0 (at its original unpadded width).
        """
        self._call_id += 1
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Routing embedding from first batch element (shared across batch)
        r_all = self.Wr(x[0]).view(T, self.n_heads, self.routing_dim).transpose(0, 1)  # [H,T,dr]

        # ------------------------------------------------------------------
        # Phase 1: build per-head neighbor indices (Python loop, O(T) per head)
        # ------------------------------------------------------------------
        neigh_idx_list: List[torch.Tensor] = []
        debug: Dict[str, Any] = {}

        for h in range(self.n_heads):
            r = r_all[h]  # [T, dr]
            cycle_adj = self._cycle_adjacency_for_head(r, h, T)
            neigh_idx_h = _build_neighbors_index(
                T,
                cycle_adj,
                window=self.window,
                landmark_stride=self.landmark_stride,
                include_self=True,
                dedup=True,
            ).to(x.device)  # [T, D_h]

            # Capture head-0 debug info at its original (unpadded) shape so the
            # returned tensors are identical to what the old per-head loop returned.
            if return_debug and h == 0:
                valid_0 = neigh_idx_h.ge(0)
                safe_0 = neigh_idx_h.clamp(min=0)
                i_idx_0 = torch.arange(T, device=x.device).view(T, 1)
                causal_0 = safe_0.le(i_idx_0)
                debug = {
                    "neigh_idx": neigh_idx_h.detach().cpu(),
                    "valid_mask": valid_0.detach().cpu(),
                    "causal_ok": causal_0.detach().cpu(),
                }

            neigh_idx_list.append(neigh_idx_h)

        # ------------------------------------------------------------------
        # Phase 2: pad to common D_max and stack into [H, T, D_max]
        # ------------------------------------------------------------------
        D_max = max(ni.shape[1] for ni in neigh_idx_list)
        padded: List[torch.Tensor] = []
        for ni in neigh_idx_list:
            D_h = ni.shape[1]
            if D_h < D_max:
                # -1 is the invalid-neighbor sentinel, consistent with
                # _build_neighbors_index padding convention.
                pad = torch.full(
                    (T, D_max - D_h), -1, dtype=torch.long, device=x.device
                )
                padded.append(torch.cat([ni, pad], dim=1))
            else:
                padded.append(ni)

        neigh_idx = torch.stack(padded, dim=0)  # [H, T, D_max]
        safe_idx = neigh_idx.clamp(min=0)        # [H, T, D_max]

        # ------------------------------------------------------------------
        # Phase 3: single batched gather of K, V for all heads
        # ------------------------------------------------------------------
        # Goal: k_g[b, h, t, d, :] = k[b, h, safe_idx[h, t, d], :]
        # We expand k along a new D_max dim, then gather along the T dim (dim=2).
        gather_idx = safe_idx[None, :, :, :, None].expand(
            B, self.n_heads, T, D_max, self.head_dim
        )  # [B, H, T, D_max, dh]

        k_exp = k.unsqueeze(3).expand(
            B, self.n_heads, T, D_max, self.head_dim
        )  # [B, H, T, D_max, dh]
        v_exp = v.unsqueeze(3).expand(
            B, self.n_heads, T, D_max, self.head_dim
        )

        k_g = torch.gather(k_exp, dim=2, index=gather_idx)  # [B, H, T, D_max, dh]
        v_g = torch.gather(v_exp, dim=2, index=gather_idx)  # [B, H, T, D_max, dh]

        # ------------------------------------------------------------------
        # Phase 4: batched attention scores, masking, softmax, weighted sum
        # ------------------------------------------------------------------
        # q: [B, H, T, dh] -> unsqueeze to [B, H, T, 1, dh] for broadcast
        scores = (q.unsqueeze(3) * k_g).sum(dim=-1) / math.sqrt(
            self.head_dim
        )  # [B, H, T, D_max]

        # Build masks: [H, T, D_max], broadcast over batch dim
        valid = neigh_idx.ge(0)                                  # [H, T, D_max]
        i_idx = torch.arange(T, device=x.device).view(1, T, 1)  # [1, T, 1]
        causal_ok = safe_idx.le(i_idx)                           # [H, T, D_max]
        mask = valid & causal_ok                                 # [H, T, D_max]

        scores = scores.masked_fill(~mask[None], float("-inf"))  # [B, H, T, D_max]
        w = F.softmax(scores, dim=-1)                            # [B, H, T, D_max]
        w = self.attn_drop(w)

        y = (w.unsqueeze(-1) * v_g).sum(dim=3)  # [B, H, T, dh]

        # ------------------------------------------------------------------
        # Phase 5: merge heads and project
        # ------------------------------------------------------------------
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        y = self.out(y)

        if return_debug:
            return y, debug
        return y
