from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_fake_zmlx_root(tmp_path: Path) -> Path:
    root = tmp_path / "ZMLX"
    files = [
        "src/zmlx/discover/targets.py",
        "src/zmlx/discover/prompts.py",
        "src/zmlx/discover/tree.py",
        "src/zmlx/discover/evaluate.py",
        "docs/DISCOVER_PLAYBOOK.md",
    ]
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    return root


def test_wayc_discover_targets_cli() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/wayc.py",
            "discover-targets",
            "--targets",
            "k1",
            "k4",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    ids = [x["id"] for x in payload["targets"]]
    assert ids == ["k1", "k4"]


def test_wayc_discover_setup_cli_writes_manifest(tmp_path: Path) -> None:
    zmlx_root = _make_fake_zmlx_root(tmp_path)
    sessions_root = tmp_path / "sessions"
    kernel_out_root = tmp_path / "kernels"
    json_out = tmp_path / "setup.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/wayc.py",
            "discover-setup",
            "--targets",
            "k2",
            "--repo-root",
            str(Path.cwd()),
            "--zmlx-root",
            str(zmlx_root),
            "--sessions-root",
            str(sessions_root),
            "--kernel-out-root",
            str(kernel_out_root),
            "--strict",
            "--json-out",
            str(json_out),
            "--overwrite",
        ],
        check=True,
    )

    assert (sessions_root / "manifest.json").exists()
    assert (sessions_root / "hcsa_sparse_gather_session.stub.json").exists()
    assert (kernel_out_root / "seeds/hcsa_sparse_gather_fused.metal").exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["ready"] is True
