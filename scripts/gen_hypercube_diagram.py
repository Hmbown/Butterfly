#!/usr/bin/env python3
"""Generate a hypercube topology diagram for the README.

Shows the block-sparse partner schedule as what it actually is: a hypercube.
Each node is a block, edges connect nodes differing by one bit (XOR partners).
Colored by dimension/stage to match the butterfly diagram palette.
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

BG = "#0d1117"
GRID = "#161b22"
TEXT = "#c9d1d9"
DIM_COLOR = "#484f58"
BLUE = "#58a6ff"
ORANGE = "#f78166"
GREEN = "#3fb950"
PURPLE = "#d2a8ff"
PINK = "#f778ba"

STAGE_COLORS = [BLUE, ORANGE, GREEN, PURPLE, PINK]


def hypercube_coords_3d(n_dims: int = 3) -> dict[int, np.ndarray]:
    """Return 3D coordinates for an n-dimensional hypercube.

    For n=3: standard cube vertices.
    For n=4: tesseract projected to 3D.
    """
    n_nodes = 1 << n_dims
    coords = {}

    if n_dims == 3:
        for node in range(n_nodes):
            x = float((node >> 0) & 1)
            y = float((node >> 1) & 1)
            z = float((node >> 2) & 1)
            coords[node] = np.array([x - 0.5, y - 0.5, z - 0.5])
    elif n_dims == 4:
        # Tesseract: project 4D to 3D with perspective-like shrink for w=1
        shrink = 0.5
        for node in range(n_nodes):
            x = float((node >> 0) & 1) - 0.5
            y = float((node >> 1) & 1) - 0.5
            z = float((node >> 2) & 1) - 0.5
            w = float((node >> 3) & 1)
            scale = 1.0 if w == 0 else shrink
            coords[node] = np.array([x * scale, y * scale, z * scale])
    else:
        # General: use first 3 bit dimensions, compress remaining into slight offsets
        for node in range(n_nodes):
            x = float((node >> 0) & 1) - 0.5
            y = float((node >> 1) & 1) - 0.5
            z = float((node >> 2) & 1) - 0.5
            # Add jitter from higher bits
            for d in range(3, n_dims):
                bit = float((node >> d) & 1)
                scale = 0.3 ** (d - 2)
                x += bit * scale * 0.3
                y += bit * scale * 0.2
                z += bit * scale * 0.25
            coords[node] = np.array([x, y, z])

    return coords


def draw_hypercube_3d(ax, n_dims: int = 3):
    """Draw an n-dimensional hypercube in 3D."""
    n_nodes = 1 << n_dims
    coords = hypercube_coords_3d(n_dims)

    # Draw edges colored by dimension
    for dim in range(n_dims):
        color = STAGE_COLORS[dim % len(STAGE_COLORS)]
        segments = []
        for node in range(n_nodes):
            partner = node ^ (1 << dim)
            if partner > node:  # avoid drawing twice
                p1 = coords[node]
                p2 = coords[partner]
                segments.append([p1, p2])

        lc = Line3DCollection(segments, colors=color, linewidths=2.0, alpha=0.7, zorder=1)
        ax.add_collection3d(lc)

    # Draw nodes
    for node in range(n_nodes):
        p = coords[node]
        ax.scatter(*p, s=120, c=TEXT, edgecolors="white", linewidths=0.5, zorder=5,
                   depthshade=False)
        # Binary label
        label = format(node, f"0{n_dims}b")
        ax.text(p[0], p[1], p[2] + 0.08, label,
                ha="center", va="bottom", color=TEXT, fontsize=7,
                fontfamily="monospace", fontweight="bold", zorder=10)

    # Axis styling
    ax.set_facecolor(BG)
    ax.xaxis.pane.set_facecolor(BG)
    ax.yaxis.pane.set_facecolor(BG)
    ax.zaxis.pane.set_facecolor(BG)
    ax.xaxis.pane.set_edgecolor(BG)
    ax.yaxis.pane.set_edgecolor(BG)
    ax.zaxis.pane.set_edgecolor(BG)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    margin = 0.7
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)

    ax.view_init(elev=20, azim=35)


def draw_hypercube_mapping(ax, n_dims: int = 3):
    """Show how hypercube dimensions map to transformer layers."""
    n_nodes = 1 << n_dims

    rows = []
    for dim in range(n_dims):
        color = STAGE_COLORS[dim % len(STAGE_COLORS)]
        pairs = []
        for node in range(n_nodes):
            partner = node ^ (1 << dim)
            if partner < node:  # causal: only attend to earlier
                pairs.append((node, partner))
        rows.append((dim, color, pairs))

    y_spacing = 1.0
    node_w = 0.7
    total_w = n_nodes * node_w

    for row_idx, (dim, color, pairs) in enumerate(rows):
        y = (n_dims - 1 - row_idx) * y_spacing

        # Draw all blocks
        for b in range(n_nodes):
            x = b * node_w
            is_active = any(b == p[0] or b == p[1] for p in pairs)
            fc = GRID if not is_active else color
            alpha = 0.3 if not is_active else 0.15
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.3), node_w * 0.85, 0.6,
                boxstyle="round,pad=0.04",
                facecolor=fc, alpha=alpha,
                edgecolor=color if is_active else DIM_COLOR,
                linewidth=1.0 if is_active else 0.5,
            )
            ax.add_patch(rect)
            label_color = TEXT if is_active else DIM_COLOR
            ax.text(x + node_w * 0.425, y, f"{b}",
                    ha="center", va="center", color=label_color,
                    fontsize=8, fontfamily="monospace", fontweight="bold")

        # Draw partner arcs
        for src, dst in pairs:
            x_src = src * node_w + node_w * 0.425
            x_dst = dst * node_w + node_w * 0.425
            mid_x = (x_src + x_dst) / 2
            arc_h = 0.15 + 0.08 * abs(src - dst)
            ax.annotate(
                "", xy=(x_dst, y + 0.32), xytext=(x_src, y + 0.32),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=color, lw=1.5,
                    connectionstyle=f"arc3,rad=-{0.3 + 0.05 * abs(src - dst)}",
                ),
            )

        # Row label
        ax.text(-0.6, y, f"Layer {row_idx}\nXOR {1 << dim}",
                ha="right", va="center", color=color,
                fontsize=9, fontweight="bold")

    ax.set_xlim(-2.2, total_w + 0.3)
    ax.set_ylim(-0.8, n_dims * y_spacing + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def main():
    n_dims = 3  # 8 blocks = 3D hypercube
    fig = plt.figure(figsize=(14, 6.5), facecolor=BG)

    # Left: 3D hypercube
    ax3d = fig.add_subplot(121, projection="3d")
    draw_hypercube_3d(ax3d, n_dims)
    ax3d.set_title("Block topology = 3D hypercube\nEdges connect XOR partners",
                    color=TEXT, fontsize=12, fontweight="bold", pad=8)

    # Legend for dimensions
    legend_handles = []
    for d in range(n_dims):
        legend_handles.append(
            mpatches.Patch(color=STAGE_COLORS[d], label=f"Dim {d}: XOR {1 << d}")
        )
    ax3d.legend(handles=legend_handles, loc="lower left", fontsize=8,
                facecolor=BG, edgecolor=DIM_COLOR, labelcolor=TEXT,
                framealpha=0.9)

    # Right: layer mapping
    ax_map = fig.add_subplot(122)
    draw_hypercube_mapping(ax_map, n_dims)
    ax_map.set_title("Each layer traverses one hypercube dimension\n→ global reachability in log₂(N) layers",
                     color=TEXT, fontsize=12, fontweight="bold", pad=12)

    fig.suptitle("Wayfinder block-sparse attention ≡ hypercube communication",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.02)

    out = "docs/assets/wayfinder_hypercube.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
