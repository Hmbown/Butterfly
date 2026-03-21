from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import WayfinderAttentionMLX
from bna.topology import Topology


def test_topology_construct_save_load_roundtrip(tmp_path) -> None:
    topo = Topology(
        n_heads=2,
        strategy="random",
        num_cycles=1,
        seed=123,
        window=4,
        landmark_stride=8,
    )
    graph = topo.construct({"T": 16})
    assert graph.abi.neigh_idx.shape[0] == 2
    assert graph.abi.neigh_idx.shape[1] == 16

    out_dir = tmp_path / "graph_cache"
    topo.save(graph, out_dir)
    loaded = topo.load(out_dir, expect_tokens=16)

    assert loaded.abi.neigh_idx.shape == graph.abi.neigh_idx.shape
    assert loaded.abi.edge_type.shape == graph.abi.edge_type.shape
    np.testing.assert_array_equal(loaded.abi.neigh_idx, graph.abi.neigh_idx)
    np.testing.assert_array_equal(loaded.abi.edge_type, graph.abi.edge_type)


def test_wayfinder_accepts_injected_topology_graph() -> None:
    attn = WayfinderAttentionMLX(
        n_embd=32,
        n_heads=2,
        window=4,
        landmark_stride=8,
        strategy="random",
        path="sparse",
        seed=7,
        dropout=0.0,
    )
    x = mx.random.normal((1, 16, 32), dtype=mx.float16)
    graph = attn.topology.construct({"T": 16})

    y1 = attn(x, topology_graph=graph)
    y2 = attn(x, topology_graph=graph)
    mx.eval(y1, y2)

    np.testing.assert_allclose(
        np.asarray(y1, dtype=np.float32),
        np.asarray(y2, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )

