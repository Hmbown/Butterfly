#!/usr/bin/env python3
"""Routing embedding analysis via dimensionality reduction.

Visualizes learned routing vectors using t-SNE or UMAP, comparing
early vs late training snapshots to show how the model learns to
organize token positions in routing space.

Usage:
    python -m viz.routing_analysis --ckpt runs/my_run/ckpt.pt --out routing.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("matplotlib and numpy required")

from hcsa.model import GPT, GPTConfig


def extract_routing_embeddings(
    model: GPT,
    x: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Extract routing embeddings from each HCSA layer.

    Returns dict mapping layer_idx -> routing embeddings [T, dr].
    """
    model.eval()
    embeddings = {}

    with torch.no_grad():
        B, T = x.shape
        tok = model.token_emb(x)
        pos = model.pos_emb(torch.arange(T, device=x.device))
        h = model.drop(tok + pos)

        for i, block in enumerate(model.blocks):
            if hasattr(block.attn, "Wr"):
                r = block.attn.Wr(block.ln1(h)[0])  # [T, H*dr]
                embeddings[i] = r.detach().cpu()
            h = block(h)

    return embeddings


def plot_routing_tsne(
    embeddings: dict[int, torch.Tensor],
    T: int,
    save_path: str | Path | None = None,
) -> None:
    """Plot t-SNE of routing embeddings per layer."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn required for t-SNE: pip install scikit-learn")
        return

    n_layers = len(embeddings)
    cols = min(n_layers, 4)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = np.arange(T)

    for ax_idx, (layer_idx, r) in enumerate(sorted(embeddings.items())):
        r_np = r.numpy()
        perplexity = min(30, T - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(r_np)

        ax = axes[ax_idx]
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors, cmap="viridis", s=20, alpha=0.8,
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(embeddings):]:
        ax.set_visible(False)

    fig.colorbar(sc, ax=axes[:len(embeddings)].tolist(), label="Position", fraction=0.02)
    fig.suptitle("Routing Embeddings (t-SNE)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ckpt["cfg"])
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        # Demo with random model
        torch.manual_seed(42)
        cfg = GPTConfig(
            vocab_size=64, seq_len=args.seq_len, n_layers=4, n_heads=4,
            n_embd=128, attn="hcsa", cycle="random", window=8, landmark_stride=8,
        )
        model = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (1, min(args.seq_len, cfg.seq_len)))
    embeddings = extract_routing_embeddings(model, x)

    if embeddings:
        T = x.shape[1]
        plot_routing_tsne(embeddings, T, save_path=args.out)
    else:
        print("No HCSA layers found (model may use dense attention)")


if __name__ == "__main__":
    main()
