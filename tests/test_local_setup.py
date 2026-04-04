from __future__ import annotations

from pathlib import Path

from bna.local_setup import default_local_paths, model_spec_by_alias


def test_default_local_paths_are_derived_from_ssd_root() -> None:
    paths = default_local_paths(
        repo_root=Path("/tmp/butterfly"),
        ssd_root=Path("/tmp/ssd"),
        llama_tag="b1234",
        zmlx_root=Path("/tmp/zmlx"),
    )

    assert paths.hf_home == Path("/tmp/ssd/hf_cache")
    assert paths.hf_hub_cache == Path("/tmp/ssd/hf_cache/hub")
    assert paths.models_root == Path("/tmp/ssd/models")
    assert paths.llama_install_root == Path("/tmp/ssd/toolchains/llama.cpp-b1234-metal")
    assert paths.zmlx_root == Path("/tmp/zmlx")


def test_support_matrix_marks_qwen_as_supported_now() -> None:
    spec = model_spec_by_alias("qwen35_9b_mlx_4bit")

    assert spec.support == "supported"
    assert spec.runner == "mlx_wayfinder"
    assert spec.repo_id == "mlx-community/Qwen3.5-9B-MLX-4bit"


def test_support_matrix_marks_gemma4_mlx_as_blocked() -> None:
    spec = model_spec_by_alias("gemma4_31b_mlx_4bit")

    assert spec.support == "blocked"
    assert spec.runner == "mlx_wayfinder"
    assert spec.blocked_reason is not None
    assert "gemma4" in spec.blocked_reason
