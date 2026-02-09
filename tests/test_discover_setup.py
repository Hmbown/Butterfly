from __future__ import annotations

from pathlib import Path

from hcsa.discover import prepare_discovery_workspace, resolve_targets


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


def test_discover_setup_writes_manifest_and_stubs(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    zmlx_root = _make_fake_zmlx_root(tmp_path)
    sessions_root = tmp_path / "sessions"
    kernel_out_root = tmp_path / "kernels"

    result = prepare_discovery_workspace(
        repo_root=repo_root,
        zmlx_root=zmlx_root,
        sessions_root=sessions_root,
        kernel_out_root=kernel_out_root,
        targets=resolve_targets(["k1", "k4"]),
        strict=True,
        dry_run=False,
        overwrite=True,
    )

    assert result["ready"] is True
    assert (sessions_root / "manifest.json").exists()
    assert (sessions_root / "hcsa_permute_window_session.stub.json").exists()
    assert (sessions_root / "hcsa_active_row_session.stub.json").exists()

    assert (kernel_out_root / "seeds/hcsa_permute_window_fused.metal").exists()
    assert (kernel_out_root / "seeds/hcsa_active_row_fused.metal").exists()


def test_discover_setup_dry_run_does_not_write_manifest(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    zmlx_root = tmp_path / "missing_zmlx"
    sessions_root = tmp_path / "sessions"
    kernel_out_root = tmp_path / "kernels"

    result = prepare_discovery_workspace(
        repo_root=repo_root,
        zmlx_root=zmlx_root,
        sessions_root=sessions_root,
        kernel_out_root=kernel_out_root,
        targets=resolve_targets(["k2"]),
        strict=False,
        dry_run=True,
        overwrite=False,
    )

    assert result["manifest_path"].endswith("manifest.json")
    assert (sessions_root / "manifest.json").exists() is False
