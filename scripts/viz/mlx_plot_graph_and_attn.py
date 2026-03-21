#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bna.graph.abi import EdgeType


def build_sparse_matrix(neigh_idx: np.ndarray, edge_type: np.ndarray) -> np.ndarray:
    """Convert neighbor lists [T,D] into dense edge-type matrix [T,T]."""
    T, D = neigh_idx.shape
    mat = np.zeros((T, T), dtype=np.int32)
    for i in range(T):
        for d in range(D):
            j = int(neigh_idx[i, d])
            if j < 0:
                continue
            mat[i, j] = int(edge_type[i, d])
    return mat


def build_attn_matrix(neigh_idx: np.ndarray, attn_weights: np.ndarray) -> np.ndarray:
    """Project sparse attention weights [T,D] into dense matrix [T,T]."""
    T, D = neigh_idx.shape
    mat = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        for d in range(D):
            j = int(neigh_idx[i, d])
            if j < 0:
                continue
            mat[i, j] += float(attn_weights[i, d])
    return mat


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Wayfinder graph + attention diagnostics")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Optional explicit debug npz file. Defaults to <run-dir>/wayfinder_graph_debug.npz",
    )
    args = p.parse_args()

    npz_path = args.npz or (args.run_dir / "wayfinder_graph_debug.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Debug file not found: {npz_path}")

    data = np.load(npz_path)
    neigh_idx = data["neigh_idx"]   # [H,T,D]
    edge_type = data["edge_type"]   # [H,T,D]
    attn = data["attn_weights"]     # [B,H,T,D]

    # Use first batch and first head for visualization.
    ni = neigh_idx[0]
    et = edge_type[0]
    aw = attn[0, 0]

    adj_mat = build_sparse_matrix(ni, et)
    attn_mat = build_attn_matrix(ni, aw)

    # Highway distance histogram from cycle/rewire edges.
    i_idx = np.arange(ni.shape[0])[:, None]
    j_idx = ni
    valid = j_idx >= 0
    cycle_or_rewire = (et == int(EdgeType.CYCLE)) | (et == int(EdgeType.REWIRE))
    dists = np.abs(j_idx - i_idx)
    highway_dist = dists[valid & cycle_or_rewire]

    out1 = args.run_dir / "graph_adjacency_heatmap.png"
    out2 = args.run_dir / "attention_on_edges_heatmap.png"
    out3 = args.run_dir / "highway_distance_hist.png"

    plt.figure(figsize=(7, 6))
    plt.imshow(adj_mat > 0, cmap="Greys", interpolation="nearest", aspect="auto")
    plt.title("Wayfinder Sparse Adjacency (Head 0)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.savefig(out1, dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(attn_mat, cmap="magma", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.title("Attention Weights on Sparse Edges (Head 0)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()
    plt.savefig(out2, dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    if highway_dist.size > 0:
        bins = min(50, max(10, int(np.sqrt(highway_dist.size))))
        plt.hist(highway_dist, bins=bins, color="#2f6c8f", alpha=0.9)
    plt.title("Highway Edge Distance Histogram")
    plt.xlabel("|i - j| in original token order")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out3, dpi=180)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
