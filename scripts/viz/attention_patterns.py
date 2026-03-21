#!/usr/bin/env python3
"""Visualize attention patterns: dense T*T heatmap vs HCSA sparse T*D.

Usage:
    python -m viz.attention_patterns --seq-len 32 --out attention_comparison.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    raise ImportError("matplotlib is required: pip install matplotlib")

from bna.attention_dense import DenseCausalSelfAttention
from bna.attention_hcsa import HCSASelfAttention


def _get_dense_attention_weights(
    x: torch.Tensor, n_embd: int, n_heads: int
) -> torch.Tensor:
    """Return dense attention weights [H, T, T] for a single batch element."""
    B, T, C = x.shape
    attn = DenseCausalSelfAttention(n_embd, n_heads)
    attn.eval()

    # We need to manually compute weights since SDPA doesn't expose them.
    with torch.no_grad():
        qkv = attn.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        head_dim = n_embd // n_heads
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

    return weights[0]  # [H, T, T]


def _get_hcsa_sparse_pattern(
    x: torch.Tensor,
    n_embd: int,
    n_heads: int,
    window: int = 4,
    landmark_stride: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return HCSA neighbor indices and mask for visualization.

    Returns (neigh_idx [T, D], mask [T, D]) for head 0.
    """
    attn = HCSASelfAttention(
        n_embd, n_heads, window=window, landmark_stride=landmark_stride,
        cycle="random", seed=42,
    )
    attn.eval()
    with torch.no_grad():
        _, debug = attn(x, return_debug=True)
    neigh_idx = debug["neigh_idx"]
    causal_ok = debug["causal_ok"]
    valid = debug["valid_mask"]
    mask = valid & causal_ok
    return neigh_idx, mask


def plot_attention_comparison(
    seq_len: int = 32,
    n_embd: int = 64,
    n_heads: int = 4,
    window: int = 4,
    landmark_stride: int = 8,
    head: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Plot dense vs HCSA attention patterns side by side."""
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, n_embd)

    # Dense attention weights
    dense_w = _get_dense_attention_weights(x, n_embd, n_heads)  # [H, T, T]
    dense_h = dense_w[head].numpy()  # [T, T]

    # HCSA sparse pattern
    neigh_idx, mask = _get_hcsa_sparse_pattern(
        x, n_embd, n_heads, window, landmark_stride
    )

    # Build sparse attention matrix for visualization
    T = seq_len
    sparse_matrix = torch.zeros(T, T)
    for i in range(T):
        for d in range(neigh_idx.shape[1]):
            j = int(neigh_idx[i, d])
            if j >= 0 and bool(mask[i, d]):
                sparse_matrix[i, j] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dense heatmap
    im0 = axes[0].imshow(
        dense_h, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    axes[0].set_title(f"Dense Attention (head {head})", fontsize=12)
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # HCSA sparse connectivity
    im1 = axes[1].imshow(
        sparse_matrix.numpy(),
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
    )
    axes[1].set_title(f"HCSA Connectivity (head 0, w={window}, lm={landmark_stride})", fontsize=12)
    axes[1].set_xlabel("Key position")
    axes[1].set_ylabel("Query position")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Add density annotation
    dense_nnz = (dense_h > 0.01).sum()
    sparse_nnz = int(sparse_matrix.sum())
    axes[0].text(
        0.02, 0.98, f"nnz={dense_nnz}",
        transform=axes[0].transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    axes[1].text(
        0.02, 0.98, f"nnz={sparse_nnz} ({100*sparse_nnz/(T*T):.1f}%)",
        transform=axes[1].transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--landmark-stride", type=int, default=8)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    plot_attention_comparison(
        seq_len=args.seq_len,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        window=args.window,
        landmark_stride=args.landmark_stride,
        head=args.head,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
