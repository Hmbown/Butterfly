from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _manual_dense_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    training: bool,
    return_weights: bool,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Manual dense causal attention fallback for older/unavailable SDPA backends."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}")

    _b, _h, t, dh = q.shape
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) / math.sqrt(float(dh))

    causal = torch.ones((t, t), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.view(1, 1, t, t), float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p, training=True)

    out = torch.matmul(weights, v.float()).to(dtype=v.dtype)
    if return_weights:
        return out, weights
    return out, None


def dense_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Dense causal attention for [B,H,T,dh], preferring PyTorch SDPA."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,dh]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}")

    drop = float(dropout_p) if training else 0.0

    if not return_weights and hasattr(F, "scaled_dot_product_attention"):
        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=drop,
                is_causal=True,
            )
            return out, None
        except RuntimeError:
            # Fallback to manual path when SDPA backend is unavailable.
            pass

    return _manual_dense_causal_attention(
        q,
        k,
        v,
        dropout_p=drop,
        training=training,
        return_weights=return_weights,
    )


class DenseCausalAttentionTorch(nn.Module):
    """Minimal dense QKV attention module with SDPA first path."""

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = self.n_embd // self.n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, return_debug: bool = False):
        b, t, c = x.shape
        if c != self.n_embd:
            raise ValueError(f"Expected C={self.n_embd}, got {c}")

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        y_h, weights = dense_causal_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            training=self.training,
            return_weights=return_debug,
        )

        y = y_h.transpose(1, 2).contiguous().view(b, t, c)
        y = self.out(y)
        y = self.dropout(y)

        if return_debug:
            return y, {"attn_weights": weights}
        return y
