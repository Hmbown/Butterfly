from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bna.compiler import compile_graph_spec


def test_wayc_compile_writes_artifacts(tmp_path: Path) -> None:
    spec = Path("configs/graph_specs/default.wf")
    result = compile_graph_spec(spec, T=128, H=4, out_root=tmp_path)

    artifact_dir = Path(result["artifact"]["artifact_dir"])
    assert artifact_dir.exists()

    ni_file = artifact_dir / "neighborindex.npz"
    perm_file = artifact_dir / "perm.npy"
    inv_file = artifact_dir / "inv_perm.npy"
    win_file = artifact_dir / "window_idx.npy"
    meta_file = artifact_dir / "meta.json"

    assert ni_file.exists()
    assert perm_file.exists()
    assert inv_file.exists()
    assert win_file.exists()
    assert meta_file.exists()

    ni = np.load(ni_file)
    neigh = np.asarray(ni["neigh_idx"], dtype=np.int32)
    edge_type = np.asarray(ni["edge_type"], dtype=np.uint8)

    assert neigh.shape == (4, 128, neigh.shape[-1])
    assert edge_type.shape == neigh.shape

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["cache_hash"] == result["cache_hash"]
    assert "cycle_perms" in meta


def test_wayc_compile_reuses_hash_for_identical_inputs(tmp_path: Path) -> None:
    spec = Path("configs/graph_specs/default.wf")
    a = compile_graph_spec(spec, T=64, H=2, out_root=tmp_path)
    b = compile_graph_spec(spec, T=64, H=2, out_root=tmp_path)
    assert a["cache_hash"] == b["cache_hash"]
