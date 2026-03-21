"""Permute-to-cycle-order optimization for HCSA.

**Key insight**: A Hamiltonian cycle gives a permutation pi.  If we permute
Q, K, V into pi-order, then cycle-neighbours become contiguous positions
(+/-1).  This converts the sparse gather into a sliding-window attention on
contiguous memory, which is extremely friendly for both Metal and CUDA.

After the window-attention in permuted order, we un-permute the output back
to original position order.  An extra causal mask is applied in the original
index space (a position can only attend to positions with *original* index
<= its own).

This module provides:
- ``permute_cycle_attention``: functional version
- ``PermutedCycleAttention``: nn.Module wrapper
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_permute_causal_mask(
    perm: torch.Tensor,
    window: int,
    landmark_stride: Optional[int] = None,
) -> torch.Tensor:
    """Build the combined causal + window mask in permuted order.

    Parameters
    ----------
    perm : Tensor [T]
        The cycle permutation:  perm[pi_pos] = original_pos.
    window : int
        One-sided window radius in permuted order (neighbours at +/-window).
    landmark_stride : int, optional
        Stride for landmark positions (in original order).

    Returns
    -------
    mask : Tensor [T, W]  (bool)
        mask[i, j] is True if permuted-position i can attend to the j-th
        entry in its local window.  W = 2*window + 1 (self included).
    """
    T = perm.numel()
    device = perm.device
    W = 2 * window + 1  # full window including self

    # For each permuted position i, the window covers [i-window, i+window]
    # in permuted order.  We need the *original* indices of those positions.
    # perm[i] = original_pos  (forward mapping)
    orig_idx = perm  # [T]

    # Build window offsets
    offsets = torch.arange(-window, window + 1, device=device)  # [W]

    # For each i, neighbor permuted positions (clamped to valid range)
    pi_positions = torch.arange(T, device=device).unsqueeze(1) + offsets.unsqueeze(0)  # [T, W]
    pi_positions_clamped = pi_positions.clamp(0, T - 1)  # [T, W]

    # Original indices of the window neighbours
    neigh_orig = orig_idx[pi_positions_clamped]  # [T, W]

    # Original index of the query position
    query_orig = orig_idx.unsqueeze(1)  # [T, 1]

    # Causal mask: neighbour's original index <= query's original index
    causal = neigh_orig <= query_orig  # [T, W]

    # Valid mask: actual neighbour is within [0, T-1] in permuted space
    valid = (pi_positions >= 0) & (pi_positions < T)  # [T, W]

    mask = causal & valid  # [T, W]

    return mask


def permute_cycle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    window: int,
    landmark_stride: Optional[int] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Sparse attention via permute-to-contiguous + sliding window.

    Parameters
    ----------
    q, k, v : Tensor [B, T, dh]
        Query, key, value for a single head.
    perm : Tensor [T]
        Cycle permutation: perm[pi_pos] = original_pos.
    window : int
        One-sided window radius in cycle order.
    landmark_stride : int, optional
        Not yet wired (reserved for future landmark augmentation).
    dropout_p : float
        Attention dropout probability.
    training : bool
        Whether in training mode (affects dropout).

    Returns
    -------
    out : Tensor [B, T, dh]
        Attention output in original position order.
    """
    B, T, dh = q.shape
    device = q.device

    # Build inverse permutation: inv_perm[original_pos] = pi_pos
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(T, device=device)

    # Permute Q, K, V into cycle order
    # perm[pi_pos] = orig_pos, so q_pi[pi_pos] = q[perm[pi_pos]]
    q_pi = q[:, perm]  # [B, T, dh]
    k_pi = k[:, perm]  # [B, T, dh]
    v_pi = v[:, perm]  # [B, T, dh]

    W = 2 * window + 1

    # Build window indices in permuted space
    offsets = torch.arange(-window, window + 1, device=device)  # [W]
    pi_idx = torch.arange(T, device=device).unsqueeze(1) + offsets.unsqueeze(0)  # [T, W]
    pi_idx_clamped = pi_idx.clamp(0, T - 1)

    # Gather K, V from window
    k_win = k_pi[:, pi_idx_clamped]  # [B, T, W, dh]
    v_win = v_pi[:, pi_idx_clamped]  # [B, T, W, dh]

    # Attention scores
    scores = (q_pi.unsqueeze(2) * k_win).sum(-1) / math.sqrt(dh)  # [B, T, W]

    # Build mask
    mask = _build_permute_causal_mask(perm, window)  # [T, W]
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Softmax + dropout
    w = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        w = F.dropout(w, p=dropout_p, training=True)

    # Weighted sum
    y_pi = (w.unsqueeze(-1) * v_win).sum(dim=2)  # [B, T, dh]

    # Un-permute back to original order
    y = y_pi[:, inv_perm]  # [B, T, dh]

    return y


class PermutedCycleAttention(nn.Module):
    """HCSA via permute-to-cycle-order + sliding window.

    This is a drop-in alternative to the gather-based HCSASelfAttention
    that achieves the same sparse connectivity but with contiguous memory
    access patterns.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        *,
        routing_dim: Optional[int] = None,
        dropout: float = 0.0,
        window: int = 64,
        landmark_stride: Optional[int] = 64,
        cycle: str = "random",
        seed: int = 0,
    ):
        super().__init__()
        assert n_embd % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.routing_dim = routing_dim or self.head_dim
        self.window = int(window)
        self.landmark_stride = landmark_stride
        self.cycle = cycle
        self.seed = int(seed)
        self.dropout_p = dropout

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.Wr = nn.Linear(n_embd, n_heads * self.routing_dim, bias=False)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    def _get_perm(self, r: torch.Tensor, head_idx: int, T: int) -> torch.Tensor:
        """Get cycle permutation for a head."""
        from .cycles import random_cycle, greedy_cycle

        if self.cycle == "random":
            return random_cycle(T, generator=self._rng, device=torch.device("cpu")).to(r.device)
        elif self.cycle == "greedy":
            start = (head_idx * 997) % T
            return greedy_cycle(r, start=start)
        else:
            return random_cycle(T, generator=self._rng, device=torch.device("cpu")).to(r.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim)  # [B, T, H, dh]
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        r_all = self.Wr(x[0]).view(T, self.n_heads, self.routing_dim)

        ys = []
        for h in range(self.n_heads):
            r = r_all[:, h]  # [T, dr]
            perm = self._get_perm(r, h, T)

            y_h = permute_cycle_attention(
                q[:, :, h], k[:, :, h], v[:, :, h],
                perm,
                window=self.window,
                landmark_stride=self.landmark_stride,
                dropout_p=self.dropout_p,
                training=self.training,
            )
            ys.append(y_h)

        y = torch.stack(ys, dim=2)  # [B, T, H, dh]
        y = y.reshape(B, T, C)
        y = self.out(y)
        return y
