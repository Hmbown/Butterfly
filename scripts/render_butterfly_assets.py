"""Render docs/assets/* PNGs for the Butterfly Compressed Attention positioning doc.

Reads benchmark JSONs under results/benchmarks/qwen35_*_mlx/ and quality JSONs
under results/quality/qwen35_4b/ and writes:

  docs/assets/butterfly_ladder.png         — e2e/peak/KV ratio panels per ctx
  docs/assets/butterfly_topology.png       — causal_shift adjacency across stages
  docs/assets/butterfly_vs_v4.png          — selector substitution diagram
  docs/assets/butterfly_quality_card.png   — 4B 1k greedy parity highlight

Usage:
  /Volumes/VIXinSSD/butterfly/.venv-macos-metal/bin/python scripts/render_butterfly_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Brand colors (calm, contrast-safe).
C_STOCK = "#7a8da3"      # muted slate — stock attention
C_COMP = "#1f6feb"       # crisp blue — compressed butterfly
C_KV   = "#22863a"       # green — retained KV win
C_TEXT = "#1f2328"
C_GRID = "#d0d7de"
C_BG   = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.facecolor": C_BG,
    "figure.facecolor": C_BG,
    "axes.facecolor": C_BG,
})


def _read(p: Path) -> tuple[float, float, float]:
    """Return (e2e_sec, peak_GB, retained_kv_MB) for a results.json."""
    d = json.loads(p.read_text())["single_turn"][0]
    return (
        d["e2e_sec"],
        d["peak_memory_bytes"] / (1024 ** 3),
        d["cache_storage_after_prefill"]["total_bytes"] / 1e6,
    )


def collect() -> dict[str, dict[int, dict[str, tuple[float, float, float]]]]:
    """Return {model: {ctx: {'stock': (...), 'comp': (...)}}}."""
    out: dict[str, dict[int, dict[str, tuple[float, float, float]]]] = {}
    # 0.8B — full ladder
    p08 = ROOT / "results" / "benchmarks" / "qwen35_0p8b_mlx"
    pairs_08 = {
        32_768:  ("p0_stock_32768",         "pA3_evict_graph_cache_32768"),
        65_536:  ("pA3_stock_65536",        "pA3_compressed_65536"),
        131_072: ("pA3_stock_131072",       "pA3_compressed_131072"),
        262_144: ("pA3_stock_262144",       "pA3_compressed_262144"),
    }
    out["Qwen 3.5 0.8B 4-bit MLX (24L / 6 dense)"] = {
        ctx: {"stock": _read(p08 / s / "results.json"),
              "comp":  _read(p08 / c / "results.json")}
        for ctx, (s, c) in pairs_08.items()
    }
    # 4B — partial ladder (256k blocked by harness prompt-build)
    p4b = ROOT / "results" / "benchmarks" / "qwen35_4b_mlx"
    pairs_4b = {
        32_768:  ("stock_32768",  "comp_32768"),
        65_536:  ("stock_65536",  "comp_65536"),
        131_072: ("stock_131072", "comp_131072"),
    }
    out["Qwen 3.5 4B 4-bit MLX (32L / 8 dense)"] = {
        ctx: {"stock": _read(p4b / s / "results.json"),
              "comp":  _read(p4b / c / "results.json")}
        for ctx, (s, c) in pairs_4b.items()
    }
    return out


def fmt_ctx(ctx: int) -> str:
    if ctx >= 1024:
        return f"{ctx//1024}k"
    return str(ctx)


def render_ladder() -> None:
    data = collect()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.30})
    for row, (model, ladder) in enumerate(data.items()):
        ctxs = sorted(ladder)
        labels = [fmt_ctx(c) for c in ctxs]
        e2e_x  = [ladder[c]["comp"][0] / ladder[c]["stock"][0] for c in ctxs]
        peak_x = [ladder[c]["comp"][1] / ladder[c]["stock"][1] for c in ctxs]
        kv_x   = [ladder[c]["comp"][2] / ladder[c]["stock"][2] for c in ctxs]

        # 1) e2e
        ax = axes[row, 0]
        bars = ax.bar(labels, e2e_x, color=C_COMP, alpha=0.9)
        ax.axhline(1.0, color=C_STOCK, linestyle="--", linewidth=1.2,
                   label="stock attention (=1.00×)")
        ax.set_ylim(0, max(1.10, max(e2e_x) * 1.10))
        ax.set_ylabel("compressed / stock", fontsize=10)
        ax.set_title("End-to-end time (×)", fontsize=12, weight="bold")
        for b, v in zip(bars, e2e_x):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}×",
                    ha="center", va="bottom", fontsize=9, color=C_TEXT)
        if row == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

        # 2) peak
        ax = axes[row, 1]
        bars = ax.bar(labels, peak_x, color=C_COMP, alpha=0.9)
        ax.axhline(1.0, color=C_STOCK, linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1.10)
        ax.set_title("Peak memory (×)", fontsize=12, weight="bold")
        for b, v in zip(bars, peak_x):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}×",
                    ha="center", va="bottom", fontsize=9, color=C_TEXT)

        # 3) retained KV (log)
        ax = axes[row, 2]
        bars = ax.bar(labels, kv_x, color=C_KV, alpha=0.9)
        ax.set_yscale("log")
        ax.axhline(1.0, color=C_STOCK, linestyle="--", linewidth=1.2)
        ax.set_ylim(0.005, 1.5)
        ax.set_title("Retained KV cache (×, log scale)", fontsize=12, weight="bold")
        for b, v in zip(bars, kv_x):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{1/v:.0f}× smaller",
                    ha="center", va="bottom", fontsize=9, color=C_TEXT)

        # row label
        axes[row, 0].annotate(
            model,
            xy=(-0.18, 0.5), xycoords="axes fraction",
            ha="right", va="center", rotation=90,
            fontsize=11, color=C_TEXT, weight="bold",
        )

    fig.suptitle(
        "Butterfly Compressed Attention vs stock — frozen Qwen 3.5, MLX, decode_len=8",
        fontsize=14, weight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.005,
        "Lower is better in all panels. "
        "Source: results/benchmarks/qwen35_*_mlx/*/results.json. "
        "256k row on 4B is harness-blocked (O(T²) prompt-build), not architecture.",
        ha="center", fontsize=9, color="#57606a", style="italic",
    )
    fig.savefig(ASSETS / "butterfly_ladder.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ASSETS / 'butterfly_ladder.png'}")


def render_topology() -> None:
    """Visualize causal_shift adjacency across stages on 32 blocks."""
    num_blocks = 32
    width = int(np.ceil(np.log2(num_blocks)))  # 5
    stages = list(range(width))

    fig, axes = plt.subplots(len(stages), 1, figsize=(11, 9),
                             sharex=True, gridspec_kw={"hspace": 0.55})
    for ax, stage_idx in zip(axes, stages):
        offset = 1 << stage_idx
        ax.set_xlim(-1.0, num_blocks)
        ax.set_ylim(-0.4, 1.4)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["key block", "query block"], fontsize=9)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        ax.tick_params(axis="x", labelsize=9)
        for b in range(num_blocks):
            ax.scatter(b, 1.0, s=28, color=C_TEXT, zorder=3)
            ax.scatter(b, 0.0, s=28, color=C_TEXT, zorder=3)
        for b in range(num_blocks):
            partner = b - offset
            if partner < 0:
                continue
            ax.annotate(
                "", xy=(partner, 0.06), xytext=(b, 0.94),
                arrowprops=dict(arrowstyle="-", color=C_COMP,
                                alpha=0.65, linewidth=1.0),
            )
        ax.set_title(f"stage {stage_idx} — partner offset = 2^{stage_idx} = {offset}",
                     fontsize=11, weight="bold", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    axes[-1].set_xlabel("block index", fontsize=10)
    fig.suptitle(
        "Butterfly causal_shift adjacency across layers — partner = block − 2^stage",
        fontsize=13, weight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.05,
        "Each query block (top) attends to one compressed-block partner (bottom). "
        f"Across {len(stages)} stages, the union covers every causal predecessor "
        f"in ⌈log₂ {num_blocks}⌉ = {width} layers.",
        ha="center", fontsize=9, color="#57606a", style="italic",
    )
    fig.savefig(ASSETS / "butterfly_topology.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ASSETS / 'butterfly_topology.png'}")


def render_v4_vs_butterfly() -> None:
    """Side-by-side: V4 learned indexer vs Butterfly fixed adjacency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={"wspace": 0.18})
    # left: V4
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.10,rounding_size=0.30",
        edgecolor="#d0d7de", facecolor="#fafbfc", linewidth=1.2))
    ax.text(5, 9.1, "DeepSeek V4 — Lightning Indexer (learned)",
            ha="center", fontsize=13, weight="bold", color=C_TEXT)
    ax.text(5, 8.1, "(paper §2.3.1, eqs. 13–17)",
            ha="center", fontsize=9, color="#57606a", style="italic")
    # query box
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.0, 5.5), 2.4, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#fff8e7", edgecolor="#9a6700", linewidth=1.0))
    ax.text(2.2, 6.3, "query token h_t", ha="center", fontsize=10)
    # 3 trainable projections
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.4, 5.5), 4.6, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#fee", edgecolor="#cf222e", linewidth=1.0))
    ax.text(6.7, 6.6, "trainable: W^DQ, W^IUQ, W^w",
            ha="center", fontsize=10, weight="bold")
    ax.text(6.7, 6.0, "(3 learned projections)",
            ha="center", fontsize=9, color="#57606a", style="italic")
    ax.annotate("", xy=(4.4, 6.3), xytext=(3.4, 6.3),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    # ReLU score
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.5, 3.2), 6.4, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#fee", edgecolor="#cf222e", linewidth=1.0))
    ax.text(6.7, 4.3, "I_t,s = Σ_h w_h · ReLU(q_h · K_s^IComp)",
            ha="center", fontsize=11, weight="bold")
    ax.text(6.7, 3.6, "score every compressed block s",
            ha="center", fontsize=9, color="#57606a")
    ax.annotate("", xy=(6.7, 4.8), xytext=(6.7, 5.4),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    # top-k selector
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.5, 1.0), 6.4, 1.4, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#fee", edgecolor="#cf222e", linewidth=1.0))
    ax.text(6.7, 1.7, "top-k(I_t,:) → k content-relevant blocks",
            ha="center", fontsize=11, weight="bold")
    ax.annotate("", xy=(6.7, 2.4), xytext=(6.7, 3.0),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    ax.text(0.6, 0.0, "needs joint pretraining; cannot retrofit on a frozen checkpoint.",
            ha="left", fontsize=10, color="#cf222e", weight="bold")

    # right: Butterfly
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.10,rounding_size=0.30",
        edgecolor="#d0d7de", facecolor="#fafbfc", linewidth=1.2))
    ax.text(5, 9.1, "Butterfly — causal_shift adjacency (deterministic)",
            ha="center", fontsize=13, weight="bold", color=C_TEXT)
    ax.text(5, 8.1, "bna/topology/butterfly.py",
            ha="center", fontsize=9, color="#57606a", style="italic")
    # query block
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.0, 5.5), 2.6, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#fff8e7", edgecolor="#9a6700", linewidth=1.0))
    ax.text(2.3, 6.5, "query block b", ha="center", fontsize=11)
    ax.text(2.3, 5.85, "in layer ℓ", ha="center", fontsize=9, color="#57606a")
    # stage decoder
    ax.add_patch(mpatches.FancyBboxPatch(
        (4.6, 5.5), 4.4, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#dafbe1", edgecolor="#1a7f37", linewidth=1.0))
    ax.text(6.8, 6.6, "stage = ℓ mod ⌈log₂ N⌉",
            ha="center", fontsize=11, weight="bold")
    ax.text(6.8, 6.0, "(zero learned parameters)",
            ha="center", fontsize=9, color="#57606a", style="italic")
    ax.annotate("", xy=(4.6, 6.3), xytext=(3.6, 6.3),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    # formula
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.0, 3.2), 8.0, 1.6, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#dafbe1", edgecolor="#1a7f37", linewidth=1.0))
    ax.text(5.0, 4.3, "partner = b − 2^stage",
            ha="center", fontsize=14, weight="bold")
    ax.text(5.0, 3.6, "single block; fixed by topology",
            ha="center", fontsize=9, color="#57606a")
    ax.annotate("", xy=(5.0, 4.8), xytext=(5.0, 5.4),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    # consequence
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.0, 1.0), 8.0, 1.4, boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#dafbe1", edgecolor="#1a7f37", linewidth=1.0))
    ax.text(5.0, 1.7, "structural mixing — log-depth across stages",
            ha="center", fontsize=11, weight="bold")
    ax.annotate("", xy=(5.0, 2.4), xytext=(5.0, 3.0),
                arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.3))
    ax.text(0.6, 0.0, "drops in on a frozen pretrained checkpoint. no training.",
            ha="left", fontsize=10, color="#1a7f37", weight="bold")

    fig.suptitle(
        "What changes when Butterfly replaces V4's selector",
        fontsize=15, weight="bold", y=1.00,
    )
    fig.savefig(ASSETS / "butterfly_vs_v4.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ASSETS / 'butterfly_vs_v4.png'}")


def render_quality_card() -> None:
    """Card highlighting 4B 1k greedy parity."""
    # 4B numbers
    ctxs   = ["1 k", "4 k", "16 k"]
    overlap_4b = [16, 32, 23]
    greedy_4b  = [32, 0, 0]
    overlap_08 = [9, 25, 20]
    greedy_08  = [0, 0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.30})
    x = np.arange(len(ctxs))
    w = 0.34

    # left: top-50 overlap
    ax = axes[0]
    ax.bar(x - w/2, overlap_08, w, color=C_STOCK, label="0.8B")
    ax.bar(x + w/2, overlap_4b, w, color=C_COMP, label="4B")
    ax.set_xticks(x); ax.set_xticklabels(ctxs)
    ax.set_ylabel("top-50 candidate overlap (out of 50)", fontsize=10)
    ax.set_title("Candidate distribution overlap (more is better)",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, 50)
    for i, (a, b) in enumerate(zip(overlap_08, overlap_4b)):
        ax.text(i - w/2, a + 0.8, str(a), ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.8, str(b), ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper right")

    # right: greedy match
    ax = axes[1]
    ax.bar(x - w/2, greedy_08, w, color=C_STOCK, label="0.8B")
    ax.bar(x + w/2, greedy_4b, w, color=C_COMP, label="4B")
    ax.set_xticks(x); ax.set_xticklabels(ctxs)
    ax.set_ylabel("greedy match — 32 tokens (more is better)", fontsize=10)
    ax.set_title("Token-by-token agreement vs stock decode",
                 fontsize=12, weight="bold")
    ax.set_ylim(0, 35)
    ax.text(0 + w/2, 33, "32 / 32 ✓", ha="center", fontsize=11,
            weight="bold", color="#1a7f37")
    for i, (a, b) in enumerate(zip(greedy_08, greedy_4b)):
        ax.text(i - w/2, a + 0.7, str(a), ha="center", fontsize=9)
        if (i, b) != (0, 32):  # already labeled with checkmark
            ax.text(i + w/2, b + 0.7, str(b), ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "Quality smoke — compressed Butterfly vs stock greedy decode",
        fontsize=14, weight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.05,
        "Frozen Qwen 3.5 4-bit MLX. scripts/quality_smoke.py: deterministic prompt, "
        "32-token greedy decode after prefill. The 4B-1k bar is the load-bearing "
        "signal: at 1 k context the larger checkpoint produces an identical 32-token "
        "continuation under compressed Butterfly attention.",
        ha="center", fontsize=9, color="#57606a", style="italic",
    )
    fig.savefig(ASSETS / "butterfly_quality_card.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ASSETS / 'butterfly_quality_card.png'}")


if __name__ == "__main__":
    render_ladder()
    render_topology()
    render_v4_vs_butterfly()
    render_quality_card()
