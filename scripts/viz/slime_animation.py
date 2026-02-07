#!/usr/bin/env python3
"""Slime mold animation: start dense, prune low-weight edges, converge to HCSA.

Creates a sequence of frames showing how attention transitions from a dense
all-to-all pattern to a sparse HCSA-like graph as low-weight edges are pruned.

Usage:
    python -m viz.slime_animation --seq-len 32 --frames 20 --out slime.gif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
except ImportError:
    raise ImportError("matplotlib and numpy required")


def _dense_attention_weights(T: int) -> np.ndarray:
    """Compute random dense causal attention weights."""
    # Simulate attention: random logits with causal mask
    logits = np.random.randn(T, T) * 0.5
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    logits[mask] = -1e9

    # Softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return weights


def _prune_step(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out edges below threshold."""
    pruned = weights.copy()
    pruned[pruned < threshold] = 0
    # Renormalize rows
    row_sums = pruned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pruned / row_sums


def generate_animation_frames(
    T: int = 32,
    n_frames: int = 20,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate frames from dense to sparse attention."""
    np.random.seed(seed)
    weights = _dense_attention_weights(T)

    frames = [weights.copy()]

    # Gradually increase pruning threshold
    for i in range(1, n_frames):
        # Threshold increases from 0 to 0.5 / T
        threshold = (i / n_frames) * (2.0 / T)
        pruned = _prune_step(weights, threshold)
        frames.append(pruned)

    return frames


def save_animation_gif(
    frames: list[np.ndarray],
    save_path: str | Path,
    fps: int = 4,
) -> None:
    """Save frames as animated GIF."""
    fig, ax = plt.subplots(figsize=(6, 6))

    T = frames[0].shape[0]
    im = ax.imshow(frames[0], cmap="viridis", interpolation="nearest", vmin=0)
    ax.set_title("Dense Attention", fontsize=12)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        nnz = (frames[frame_idx] > 0.001).sum()
        density = 100 * nnz / (T * T)
        sparsity_label = "Dense" if frame_idx == 0 else f"Sparse ({density:.0f}% density)"
        ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}: {sparsity_label}", fontsize=11)
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True,
    )

    anim.save(str(save_path), writer="pillow", fps=fps)
    print(f"Saved animation to {save_path}")
    plt.close(fig)


def save_static_frames(
    frames: list[np.ndarray],
    save_path: str | Path,
) -> None:
    """Save key frames as a static image."""
    n_show = min(6, len(frames))
    indices = [int(i * (len(frames) - 1) / (n_show - 1)) for i in range(n_show)]

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    T = frames[0].shape[0]

    for ax, idx in zip(axes, indices):
        ax.imshow(frames[idx], cmap="viridis", interpolation="nearest", vmin=0)
        nnz = (frames[idx] > 0.001).sum()
        density = 100 * nnz / (T * T)
        ax.set_title(f"{density:.0f}%", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Slime Mold: Dense -> Sparse Attention", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved static frames to {save_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="slime.gif")
    p.add_argument("--static", action="store_true", help="Save as static image instead of GIF")
    args = p.parse_args()

    frames = generate_animation_frames(args.seq_len, args.frames, args.seed)

    if args.static:
        save_static_frames(frames, args.out.replace(".gif", ".png"))
    else:
        save_animation_gif(frames, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
