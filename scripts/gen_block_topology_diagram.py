#!/usr/bin/env python3
"""Generate block-sparse topology diagrams for README.

Creates a multi-panel figure showing:
  1. Dense causal attention mask
  2. BNA block-sparse mask at a single layer
  3. BNA block-sparse mask across 4 layers (union = global reachability)
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Topology logic (standalone, no torch dependency) ---

def _ceil_log2(n: int) -> int:
    return max(1, math.ceil(math.log2(max(1, n))))


def wayfinder_block_mask(
    num_blocks: int,
    layer_idx: int,
    local_window_blocks: int = 1,
    sink_count: int = 1,
    partner_rule: str = "xor",
) -> np.ndarray:
    """Return a [num_blocks, num_blocks] bool mask for one layer."""
    width = _ceil_log2(num_blocks)
    if partner_rule == "benes":
        stage_count = max(1, (2 * width) - 2)
    else:
        stage_count = width
    stage_idx = layer_idx % stage_count

    mask = np.zeros((num_blocks, num_blocks), dtype=bool)
    for b in range(num_blocks):
        # self
        mask[b, b] = True
        # local predecessors
        for off in range(1, local_window_blocks + 1):
            if b - off >= 0:
                mask[b, b - off] = True
        # partner (causal: partner <= b)
        bit_idx = stage_idx % width
        partner = b ^ (1 << bit_idx)
        if 0 <= partner < num_blocks and partner <= b:
            mask[b, partner] = True
        # sinks
        for s in range(min(sink_count, num_blocks)):
            if s <= b:
                mask[b, s] = True
    return mask


def wayfinder_token_mask(
    num_blocks: int,
    block_size: int,
    layer_idx: int,
    local_window_blocks: int = 1,
    sink_count: int = 1,
) -> np.ndarray:
    """Expand block mask to token-level [T, T] for visualization."""
    bm = wayfinder_block_mask(num_blocks, layer_idx, local_window_blocks, sink_count)
    T = num_blocks * block_size
    token_mask = np.zeros((T, T), dtype=bool)
    for bq in range(num_blocks):
        for bk in range(num_blocks):
            if bm[bq, bk]:
                rq = slice(bq * block_size, (bq + 1) * block_size)
                rk = slice(bk * block_size, (bk + 1) * block_size)
                token_mask[rq, rk] = True
    # enforce causal
    causal = np.tril(np.ones((T, T), dtype=bool))
    return token_mask & causal


def dense_causal_mask(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T), dtype=bool))


# --- Color maps ---

BG_COLOR = "#0d1117"
DENSE_COLOR = "#6e40c9"
LOCAL_COLOR = "#58a6ff"
PARTNER_COLOR = "#f78166"
SINK_COLOR = "#3fb950"
OVERLAP_COLOR = "#d2a8ff"


def color_block_mask(
    num_blocks: int,
    block_size: int,
    layer_idx: int,
    local_window_blocks: int = 1,
    sink_count: int = 1,
) -> np.ndarray:
    """Return [T, T, 3] uint8 image with color-coded edge types."""
    width = _ceil_log2(num_blocks)
    stage_count = width
    stage_idx = layer_idx % stage_count
    T = num_blocks * block_size

    img = np.zeros((T, T, 3), dtype=np.uint8)

    def hex_to_rgb(h: str) -> tuple:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    local_rgb = hex_to_rgb(LOCAL_COLOR)
    partner_rgb = hex_to_rgb(PARTNER_COLOR)
    sink_rgb = hex_to_rgb(SINK_COLOR)
    overlap_rgb = hex_to_rgb(OVERLAP_COLOR)

    causal = np.tril(np.ones((T, T), dtype=bool))

    for b in range(num_blocks):
        rq = slice(b * block_size, (b + 1) * block_size)

        # self + local
        for off in range(local_window_blocks + 1):
            bk = b - off
            if bk >= 0:
                rk = slice(bk * block_size, (bk + 1) * block_size)
                region = causal[rq, rk]
                img[rq, rk][region] = local_rgb

        # partner
        bit_idx = stage_idx % width
        partner = b ^ (1 << bit_idx)
        if 0 <= partner < num_blocks and partner <= b:
            rk = slice(partner * block_size, (partner + 1) * block_size)
            region = causal[rq, rk]
            # check overlap with local
            already = np.any(img[rq, rk], axis=-1)
            overlap = region & already
            fresh = region & ~already
            img[rq, rk][fresh] = partner_rgb
            img[rq, rk][overlap] = overlap_rgb

        # sinks
        for s in range(min(sink_count, num_blocks)):
            if s <= b:
                rk = slice(s * block_size, (s + 1) * block_size)
                region = causal[rq, rk]
                already = np.any(img[rq, rk], axis=-1)
                fresh = region & ~already
                img[rq, rk][fresh] = sink_rgb

    return img


# --- Figure ---

def main():
    num_blocks = 16
    block_size = 4
    T = num_blocks * block_size
    local_w = 1
    sink_c = 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=BG_COLOR)

    # Panel 1: Dense causal
    ax = axes[0]
    dense = dense_causal_mask(T)
    dense_img = np.zeros((T, T, 3), dtype=np.uint8)
    dense_rgb = tuple(int(DENSE_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    dense_img[dense] = dense_rgb
    ax.imshow(dense_img, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title("Dense causal", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Key position", color="#8b949e", fontsize=9)
    ax.set_ylabel("Query position", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_facecolor(BG_COLOR)

    sparsity_dense = 1.0 - dense.sum() / (T * T)
    ax.text(0.5, -0.12, f"{sparsity_dense*100:.0f}% sparse",
            transform=ax.transAxes, ha="center", color="#8b949e", fontsize=9)

    # Panel 2: BNA single layer (color-coded)
    ax = axes[1]
    colored = color_block_mask(num_blocks, block_size, layer_idx=0,
                               local_window_blocks=local_w, sink_count=sink_c)
    ax.imshow(colored, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title("BNA — layer 0", color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Key position", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_facecolor(BG_COLOR)

    wf_mask = wayfinder_token_mask(num_blocks, block_size, 0, local_w, sink_c)
    sparsity_wf = 1.0 - wf_mask.sum() / (T * T)
    ax.text(0.5, -0.12, f"{sparsity_wf*100:.0f}% sparse",
            transform=ax.transAxes, ha="center", color="#8b949e", fontsize=9)

    # Panel 3: BNA union across log2(N) layers
    ax = axes[2]
    width = _ceil_log2(num_blocks)
    union_mask = np.zeros((T, T), dtype=bool)
    for l in range(width):
        union_mask |= wayfinder_token_mask(num_blocks, block_size, l, local_w, sink_c)

    union_img = np.zeros((T, T, 3), dtype=np.uint8)
    # Color by layer contribution
    layer_colors = [
        (88, 166, 255),   # blue
        (247, 129, 102),  # orange
        (63, 185, 80),    # green
        (210, 168, 255),  # purple
    ]
    for l in range(width):
        layer_mask = wayfinder_token_mask(num_blocks, block_size, l, local_w, sink_c)
        new_pixels = layer_mask & ~np.any(union_img > 0, axis=-1)
        color = layer_colors[l % len(layer_colors)]
        union_img[new_pixels] = color
        # shared pixels get blended
        shared = layer_mask & np.any(union_img > 0, axis=-1) & ~new_pixels
        if shared.any():
            existing = union_img[shared].astype(np.float32)
            blended = (existing + np.array(color, dtype=np.float32)) / 2
            union_img[shared] = blended.astype(np.uint8)

    ax.imshow(union_img, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(f"BNA — {width} layers combined", color="white",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Key position", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_facecolor(BG_COLOR)

    sparsity_union = 1.0 - union_mask.sum() / (T * T)
    ax.text(0.5, -0.12, f"{sparsity_union*100:.0f}% sparse (global reachability)",
            transform=ax.transAxes, ha="center", color="#8b949e", fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=LOCAL_COLOR, label="Self + local"),
        mpatches.Patch(facecolor=PARTNER_COLOR, label="Partner (XOR)"),
        mpatches.Patch(facecolor=SINK_COLOR, label="Sink"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=10,
               labelcolor="white", handlelength=1.5,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Causal attention patterns  (T={T}, block_size={block_size})",
                 color="white", fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = "docs/assets/bna_block_topology.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.3)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
