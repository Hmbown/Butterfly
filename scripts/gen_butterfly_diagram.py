#!/usr/bin/env python3
"""Generate a butterfly network + reachability diagram for the README.

Panel 1: Classic butterfly network diagram showing how the XOR partner
         schedule connects blocks across layers/stages.
Panel 2: Reachability expansion — one source block fans out across layers,
         showing exponential growth until global coverage.
Panel 3: Per-layer block attention masks (small multiples).
"""
from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

BG = "#0d1117"
GRID = "#161b22"
TEXT = "#c9d1d9"
DIM = "#484f58"
BLUE = "#58a6ff"
ORANGE = "#f78166"
GREEN = "#3fb950"
PURPLE = "#d2a8ff"
PINK = "#f778ba"

STAGE_COLORS = [BLUE, ORANGE, GREEN, PURPLE, PINK]


def _ceil_log2(n: int) -> int:
    return max(1, math.ceil(math.log2(max(1, n))))


def xor_partner(block: int, stage: int, num_blocks: int) -> int | None:
    partner = block ^ (1 << stage)
    if partner < 0 or partner >= num_blocks or partner > block:
        return None  # causal: only attend to earlier
    return partner


# ── Panel 1: Butterfly network ──────────────────────────────────────

def draw_butterfly(ax, num_blocks: int = 8):
    stages = _ceil_log2(num_blocks)
    cols = stages + 1  # input + stages

    # Node positions
    x_positions = np.linspace(0, 1, cols)
    y_positions = np.linspace(1, 0, num_blocks)

    # Draw nodes at each stage
    node_coords = {}
    for col in range(cols):
        for row in range(num_blocks):
            x, y = x_positions[col], y_positions[row]
            node_coords[(col, row)] = (x, y)
            ax.plot(x, y, "o", color=TEXT, markersize=6, zorder=5)
            if col == 0:
                ax.text(x - 0.06, y, f"B{row}", ha="right", va="center",
                        color=DIM, fontsize=8, fontfamily="monospace")

    # Draw connections for each stage
    for stage in range(stages):
        col_from = stage
        col_to = stage + 1
        color = STAGE_COLORS[stage % len(STAGE_COLORS)]

        for block in range(num_blocks):
            # Pass-through (identity)
            x0, y0 = node_coords[(col_from, block)]
            x1, y1 = node_coords[(col_to, block)]
            ax.plot([x0, x1], [y0, y1], "-", color=DIM, alpha=0.3,
                    linewidth=0.8, zorder=1)

            # XOR partner connection (both directions for the network)
            partner = block ^ (1 << stage)
            if 0 <= partner < num_blocks:
                x1p, y1p = node_coords[(col_to, partner)]
                ax.plot([x0, x1p], [y0, y1p], "-", color=color,
                        linewidth=1.5, alpha=0.7, zorder=2)

    # Stage labels
    for stage in range(stages):
        x = (x_positions[stage] + x_positions[stage + 1]) / 2
        color = STAGE_COLORS[stage % len(STAGE_COLORS)]
        ax.text(x, -0.08, f"stage {stage}\nXOR {1 << stage}",
                ha="center", va="top", color=color, fontsize=8,
                fontweight="bold")

    ax.set_xlim(-0.15, 1.08)
    ax.set_ylim(-0.18, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Butterfly network\n(XOR partner schedule)",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)


# ── Panel 2: Reachability expansion ─────────────────────────────────

def draw_reachability(ax, num_blocks: int = 8, source: int = 7):
    """Show which blocks are reachable from `source` after each stage."""
    stages = _ceil_log2(num_blocks)
    block_h = 0.8
    stage_w = 1.0
    gap = 0.3

    reachable = {source}

    for stage in range(stages + 1):
        x_left = stage * (stage_w + gap)

        for b in range(num_blocks):
            y = (num_blocks - 1 - b) * block_h
            if b in reachable:
                color = STAGE_COLORS[min(stage, len(STAGE_COLORS) - 1)]
                alpha = 0.85
            else:
                color = GRID
                alpha = 0.4
            rect = mpatches.FancyBboxPatch(
                (x_left, y), stage_w, block_h * 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=alpha,
                edgecolor="none",
            )
            ax.add_patch(rect)
            label_color = "white" if b in reachable else DIM
            ax.text(x_left + stage_w / 2, y + block_h * 0.4,
                    f"B{b}", ha="center", va="center",
                    color=label_color, fontsize=7.5, fontweight="bold",
                    fontfamily="monospace")

        # Label
        if stage == 0:
            label = "Source"
        else:
            label = f"After\nstage {stage - 1}"
        ax.text(x_left + stage_w / 2, -0.6, label,
                ha="center", va="top", color=TEXT, fontsize=8.5,
                fontweight="bold")

        # Count
        ax.text(x_left + stage_w / 2, num_blocks * block_h + 0.15,
                f"{len(reachable)}/{num_blocks}",
                ha="center", va="bottom", color=DIM, fontsize=8)

        # Expand reachable set for next stage
        if stage < stages:
            new_reachable = set(reachable)
            for b in reachable:
                partner = b ^ (1 << stage)
                if 0 <= partner < num_blocks:
                    new_reachable.add(partner)
            reachable = new_reachable

    total_w = (stages + 1) * (stage_w + gap) - gap
    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-1.2, num_blocks * block_h + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Reachability from block {source}\n(expands exponentially)",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)


# ── Panel 3: Per-layer attention masks ──────────────────────────────

def block_attention_mask(num_blocks: int, stage: int,
                         local_w: int = 1, sink_c: int = 1) -> np.ndarray:
    width = _ceil_log2(num_blocks)
    mask = np.zeros((num_blocks, num_blocks), dtype=float)
    for b in range(num_blocks):
        mask[b, b] = 1.0  # self
        for off in range(1, local_w + 1):
            if b - off >= 0:
                mask[b, b - off] = 0.5  # local
        bit = stage % width
        partner = b ^ (1 << bit)
        if 0 <= partner < num_blocks and partner <= b:
            mask[b, partner] = 0.75  # partner
        for s in range(min(sink_c, num_blocks)):
            if s <= b and mask[b, s] == 0:
                mask[b, s] = 0.25  # sink
    return mask


def draw_layer_masks(ax_row, num_blocks: int = 8):
    stages = _ceil_log2(num_blocks)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("wf", [BG, BLUE, ORANGE, GREEN])

    for stage in range(stages):
        ax = ax_row[stage]
        mask = block_attention_mask(num_blocks, stage)
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1,
                  origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(f"Layer {stage}", color=TEXT, fontsize=10, fontweight="bold")
        ax.tick_params(colors=DIM, labelsize=6)
        ax.set_facecolor(BG)
        if stage == 0:
            ax.set_ylabel("Query block", color=DIM, fontsize=8)
        ax.set_xlabel("Key block", color=DIM, fontsize=8)

    # Union
    ax = ax_row[stages]
    union = np.zeros((num_blocks, num_blocks), dtype=float)
    for s in range(stages):
        m = block_attention_mask(num_blocks, s)
        union = np.maximum(union, m)
    ax.imshow(union, cmap=cmap, vmin=0, vmax=1,
              origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title("Union", color=TEXT, fontsize=10, fontweight="bold")
    ax.tick_params(colors=DIM, labelsize=6)
    ax.set_facecolor(BG)
    ax.set_xlabel("Key block", color=DIM, fontsize=8)


# ── Main ────────────────────────────────────────────────────────────

def main():
    N = 8
    stages = _ceil_log2(N)

    fig = plt.figure(figsize=(16, 12), facecolor=BG)

    # Top row: butterfly + reachability
    gs_top = fig.add_gridspec(1, 2, top=0.95, bottom=0.48, left=0.05, right=0.95,
                              wspace=0.3)
    ax_butterfly = fig.add_subplot(gs_top[0])
    ax_butterfly.set_facecolor(BG)
    draw_butterfly(ax_butterfly, N)

    ax_reach = fig.add_subplot(gs_top[1])
    ax_reach.set_facecolor(BG)
    draw_reachability(ax_reach, N, source=7)

    # Bottom row: per-layer masks + union
    gs_bot = fig.add_gridspec(1, stages + 1, top=0.40, bottom=0.03,
                              left=0.05, right=0.95, wspace=0.35)
    ax_masks = [fig.add_subplot(gs_bot[i]) for i in range(stages + 1)]
    draw_layer_masks(ax_masks, N)

    fig.suptitle("Wayfinder: staged block-sparse attention via butterfly networks",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.99)

    out = "docs/assets/wayfinder_butterfly_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
