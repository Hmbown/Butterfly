"""Learned graph rewiring for HCSA.

GATED: Only start after basic HCSA (random/greedy + window + landmarks)
achieves <=10% perplexity gap vs dense on WikiText-2 at equal params.

Two mechanisms:
1. **Attention-weight pruning**: Remove edges with consistently low attention
   weights, add edges toward high-routing-similarity targets.
2. **Gumbel-Softmax differentiable edge selection**: Use learned scores to
   select which edges to include, enabling gradient-based optimization.

Also includes a learned mixture-of-cycles mechanism where softmax weights
over N candidate cycles allow low-weight cycles to be replaced.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedRewiring(nn.Module):
    """Differentiable edge selection via Gumbel-Softmax.

    Given a base adjacency (from cycle + window + landmarks), learns a
    score for each potential edge. During training, uses Gumbel-Softmax
    for differentiable selection. During eval, uses hard top-k.

    Parameters
    ----------
    T : int
        Maximum sequence length.
    D_base : int
        Number of base neighbors per token.
    D_extra : int
        Number of additional learned edges per token.
    tau : float
        Gumbel-Softmax temperature.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        D_extra: int = 4,
        tau: float = 1.0,
        routing_dim: int = 32,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.D_extra = D_extra
        self.tau = tau
        self.routing_dim = routing_dim

        # Learned edge scoring: project Q and K into routing space
        self.score_q = nn.Linear(n_embd, n_heads * routing_dim, bias=False)
        self.score_k = nn.Linear(n_embd, n_heads * routing_dim, bias=False)

    def compute_edge_scores(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise edge scores.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings [B, T, C].

        Returns
        -------
        torch.Tensor
            Edge scores [B, H, T, T] (higher = more likely to be selected).
        """
        B, T, _C = x.shape
        sq = self.score_q(x).view(B, T, self.n_heads, self.routing_dim).transpose(1, 2)
        sk = self.score_k(x).view(B, T, self.n_heads, self.routing_dim).transpose(1, 2)

        # Scaled dot product similarity
        scores = torch.matmul(sq, sk.transpose(-2, -1)) / math.sqrt(self.routing_dim)

        # Apply causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float("-inf"))

        return scores

    def select_edges_gumbel(
        self,
        scores: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select extra edges using Gumbel-Softmax.

        Parameters
        ----------
        scores : torch.Tensor
            Edge scores [B, H, T, T].
        base_mask : torch.Tensor
            Binary mask of base edges [H, T, T] (1 = already connected).

        Returns
        -------
        torch.Tensor
            Soft edge weights [B, H, T, T] for the extra edges.
        """
        # Mask out existing base edges (don't re-select them)
        scores = scores.masked_fill(base_mask[None].bool(), float("-inf"))

        if self.training:
            # Gumbel-Softmax for differentiable selection
            weights = F.gumbel_softmax(scores, tau=self.tau, hard=False, dim=-1)
        else:
            # Hard top-k during eval
            _, topk_idx = scores.topk(self.D_extra, dim=-1)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, topk_idx, 1.0)

        return weights

    def forward(
        self,
        x: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute learned extra edge weights.

        Parameters
        ----------
        x : torch.Tensor
            Input [B, T, C].
        base_mask : torch.Tensor
            Base adjacency [H, T, T].

        Returns
        -------
        torch.Tensor
            Extra edge weights [B, H, T, T].
        """
        scores = self.compute_edge_scores(x)
        return self.select_edges_gumbel(scores, base_mask)


class AttentionWeightPruner:
    """Prune edges based on accumulated attention weight statistics.

    Tracks running average of attention weights per edge. Edges with
    consistently low weights are candidates for removal.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    momentum : float
        EMA momentum for tracking attention weights.
    prune_threshold : float
        Edges with average weight below this are pruned.
    """

    def __init__(
        self,
        n_heads: int,
        momentum: float = 0.99,
        prune_threshold: float = 0.01,
    ):
        self.n_heads = n_heads
        self.momentum = momentum
        self.prune_threshold = prune_threshold
        self._running_weights: Optional[torch.Tensor] = None

    def update(self, attn_weights: torch.Tensor) -> None:
        """Update running statistics with observed attention weights.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Attention weights [B, H, T, D] from HCSA forward pass.
            Averaged over batch dimension before tracking.
        """
        avg = attn_weights.mean(dim=0).detach()  # [H, T, D]

        if self._running_weights is None:
            self._running_weights = avg
        else:
            self._running_weights = (
                self.momentum * self._running_weights
                + (1 - self.momentum) * avg
            )

    def get_prune_mask(self) -> Optional[torch.Tensor]:
        """Return mask of edges to prune (True = should prune).

        Returns
        -------
        torch.Tensor or None
            Boolean mask [H, T, D] where True means the edge should
            be pruned. None if no statistics have been accumulated.
        """
        if self._running_weights is None:
            return None
        return self._running_weights < self.prune_threshold

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._running_weights = None


class CycleMixture(nn.Module):
    """Learned mixture of multiple Hamiltonian cycles.

    Maintains N candidate cycles and learns softmax weights over them.
    Low-weight cycles can be replaced with newly proposed ones.

    Parameters
    ----------
    n_cycles : int
        Number of candidate cycles to maintain.
    n_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        n_cycles: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_cycles = n_cycles
        self.n_heads = n_heads

        # Per-head learned mixture weights over cycles
        self.log_weights = nn.Parameter(
            torch.zeros(n_heads, n_cycles)
        )

    def get_mixture_weights(self) -> torch.Tensor:
        """Return normalized mixture weights [H, N_cycles]."""
        return F.softmax(self.log_weights, dim=-1)

    def mix_adjacencies(
        self,
        adjacencies: List[torch.Tensor],
    ) -> torch.Tensor:
        """Combine multiple adjacency matrices using learned weights.

        Parameters
        ----------
        adjacencies : list of torch.Tensor
            List of N adjacency matrices, each [H, T, T] (binary).

        Returns
        -------
        torch.Tensor
            Weighted adjacency [H, T, T] with soft edge weights.
        """
        if len(adjacencies) != self.n_cycles:
            raise ValueError(
                f"Expected {self.n_cycles} adjacencies, got {len(adjacencies)}"
            )

        weights = self.get_mixture_weights()  # [H, N]
        stacked = torch.stack(adjacencies, dim=1)  # [H, N, T, T]

        # Weighted sum: [H, T, T]
        mixed = (weights[:, :, None, None] * stacked).sum(dim=1)
        return mixed

    def identify_weak_cycles(self, threshold: float = 0.1) -> List[int]:
        """Identify cycle indices with low mixture weight across all heads.

        Parameters
        ----------
        threshold : float
            Cycles with max weight across all heads below this threshold
            are considered weak.

        Returns
        -------
        list of int
            Indices of weak cycles that could be replaced.
        """
        weights = self.get_mixture_weights().detach()  # [H, N]
        max_per_cycle = weights.max(dim=0).values  # [N]
        weak = (max_per_cycle < threshold).nonzero(as_tuple=True)[0]
        return weak.tolist()
