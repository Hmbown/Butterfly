#!/usr/bin/env python3
"""Generate a 5-panel attention pattern comparison figure.

Shows causal attention masks for T=64 tokens across five sparse attention
strategies, from left (densest) to right (sparsest):
  1. Dense causal
  2. Sliding window (Mistral-style SWA)
  3. Longformer (sliding window + global prefix tokens)
  4. BigBird (sliding window + global prefix + random)
  5. HCSA / Wayfinder (sliding window + Hamiltonian-cycle backbone + landmarks)

Usage:
    python3 scripts/viz/attention_pattern_comparison.py
    python3 scripts/viz/attention_pattern_comparison.py --out docs/assets/attention_comparison_5panel.png
    python3 scripts/viz/attention_pattern_comparison.py --seq-len 48 --window 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def mask_dense(T: int) -> np.ndarray:
    """Full causal lower-triangular mask."""
    return np.tril(np.ones((T, T), dtype=np.float32))


def mask_sliding_window(T: int, window: int) -> np.ndarray:
    """Sliding window causal attention (Mistral-style)."""
    m = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        lo = max(0, i - window)
        m[i, lo : i + 1] = 1.0
    return m


def mask_longformer(T: int, window: int, num_global: int = 4) -> np.ndarray:
    """Sliding window + global prefix tokens (Longformer-style)."""
    m = mask_sliding_window(T, window)
    # global tokens attend to all, all attend to global tokens (causal: global cols only)
    for g in range(min(num_global, T)):
        m[:, g] = np.tril(np.ones((T, T)))[: , g]   # all rows can see global j=g (if j<=i)
        m[g, :g + 1] = 1.0                            # global token attends to all prior
    return np.clip(m, 0.0, 1.0)


def mask_bigbird(T: int, window: int, num_global: int = 3, num_random: int = 2, seed: int = 42) -> np.ndarray:
    """Sliding window + global prefix + random keys (BigBird-style)."""
    rng = np.random.default_rng(seed)
    m = mask_longformer(T, window, num_global)
    for i in range(T):
        candidates = [j for j in range(i) if m[i, j] == 0]
        if candidates:
            picks = rng.choice(candidates, size=min(num_random, len(candidates)), replace=False)
            m[i, picks] = 1.0
    return m


def mask_hcsa(T: int, window: int, landmark_stride: int = 16, seed: int = 42) -> np.ndarray:
    """HCSA: sliding window + Hamiltonian-cycle backbone (causal edges) + landmark tokens."""
    rng = np.random.default_rng(seed)
    m = mask_sliding_window(T, window)

    # Build a random Hamiltonian cycle as a permutation
    perm = rng.permutation(T)
    # For each node i, its cycle neighbors are perm[k-1] and perm[k+1] (mod T)
    inv = np.empty(T, dtype=int)
    inv[perm] = np.arange(T)
    for i in range(T):
        k = inv[i]
        for delta in (-1, 1):
            j = int(perm[(k + delta) % T])
            if j < i:  # causal
                m[i, j] = 1.0

    # Landmark tokens (every landmark_stride-th position)
    for i in range(T):
        for j in range(0, i, landmark_stride):
            m[i, j] = 1.0

    return np.clip(m, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

PANEL_CONFIGS = [
    ("Dense\ncausal", "Full lower-triangular\n100% sparsity budget", "#6366f1"),
    ("Sliding\nwindow", "Local causal window only\n(Mistral SWA)", "#0ea5e9"),
    ("Longformer\nstyle", "Window + global prefix\ntokens", "#10b981"),
    ("BigBird\nstyle", "Window + global\n+ random keys", "#f59e0b"),
    ("HCSA\n(Wayfinder)", "Window + cycle backbone\n+ landmark tokens", "#ef4444"),
]


def build_masks(T: int, window: int) -> list[np.ndarray]:
    lm_stride = max(8, T // 6)
    return [
        mask_dense(T),
        mask_sliding_window(T, window),
        mask_longformer(T, window, num_global=3),
        mask_bigbird(T, window, num_global=3, num_random=2),
        mask_hcsa(T, window, landmark_stride=lm_stride),
    ]


def sparsity(m: np.ndarray) -> float:
    total = (m.shape[0] * (m.shape[0] + 1)) / 2  # lower-triangular cells
    return 1.0 - float(m.sum()) / total


def plot_comparison(T: int = 64, window: int = 8, out: str | None = None) -> None:
    masks = build_masks(T, window)
    n = len(masks)

    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2))
    fig.patch.set_facecolor("#0f172a")

    cmaps = ["Purples", "Blues", "Greens", "Oranges", "Reds"]

    for ax, mask, (title, subtitle, color), cmap in zip(axes, masks, PANEL_CONFIGS, cmaps):
        ax.set_facecolor("#0f172a")
        sp = sparsity(mask)
        nnz = int(mask.sum())

        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, origin="upper", aspect="equal")

        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(
            f"{subtitle}\nnnz={nnz}  ({100*(1-sp):.0f}% dense)",
            color="#94a3b8", fontsize=7.5, labelpad=4,
        )
        ax.tick_params(colors="#475569", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        # Mark sparsity budget in top-right
        ax.text(
            0.97, 0.97,
            f"{sp*100:.0f}% sparse",
            transform=ax.transAxes,
            color=color, fontsize=8, fontweight="bold",
            ha="right", va="top",
        )

    fig.suptitle(
        f"Causal attention patterns  (T={T}, window={window})\n"
        "left → right: densest → sparsest",
        color="white", fontsize=12, y=1.01,
    )

    plt.tight_layout(pad=1.2)

    out_path = out or "docs/assets/attention_comparison_5panel.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 5-panel attention comparison figure")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    plot_comparison(T=args.seq_len, window=args.window, out=args.out)


if __name__ == "__main__":
    main()
