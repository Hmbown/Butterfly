from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseCausalSelfAttention(nn.Module):
    """Standard dense causal multi-head self-attention.

    Uses F.scaled_dot_product_attention (FlashAttention / memory-efficient backend)
    when available, with manual fallback for older PyTorch versions.
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Use FlashAttention / memory-efficient SDPA with causal mask
        drop_p = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        return y
