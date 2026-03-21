from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


def emit_cache_artifact_pass(
    *,
    neigh_idx: np.ndarray,
    edge_type: np.ndarray,
    permute_payload: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    cache_hash: str,
    out_root: str | Path = ".cache/wayfinder",
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    if out_dir is None:
        artifact_dir = Path(out_root) / cache_hash
    else:
        artifact_dir = Path(out_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        artifact_dir / "neighborindex.npz",
        neigh_idx=np.asarray(neigh_idx, dtype=np.int32),
        edge_type=np.asarray(edge_type, dtype=np.uint8),
    )
    np.save(artifact_dir / "perm.npy", np.asarray(permute_payload["perm"], dtype=np.int32))
    np.save(artifact_dir / "inv_perm.npy", np.asarray(permute_payload["inv_perm"], dtype=np.int32))
    np.save(
        artifact_dir / "window_idx.npy",
        np.asarray(permute_payload["window_idx"], dtype=np.int32),
    )

    meta_out = dict(meta)
    meta_out.update(
        {
            "cache_hash": cache_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "artifact_dir": str(artifact_dir),
        }
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps(meta_out, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "artifact_dir": str(artifact_dir),
        "files": {
            "neighborindex": str(artifact_dir / "neighborindex.npz"),
            "perm": str(artifact_dir / "perm.npy"),
            "inv_perm": str(artifact_dir / "inv_perm.npy"),
            "window_idx": str(artifact_dir / "window_idx.npy"),
            "meta": str(artifact_dir / "meta.json"),
        },
    }
