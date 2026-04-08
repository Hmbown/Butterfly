from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_bench_qwen_mlx_rejects_invalid_butterfly_prefill_chunking(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module(
        "bench_qwen_consumer_mlx_cli_guard_test",
        "scripts/bench_qwen_consumer_mlx.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen_consumer_mlx.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--chunk-size",
            "4096",
            "--query-chunk-size",
            "384",
            "--skip-single-turn",
            "--skip-multi-turn",
            "--skip-quality",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    err = capsys.readouterr().err
    assert "--chunk-size exceeds --query-chunk-size" in err


def test_bench_qwen_mlx_rejects_kv_quant_with_experimental_decode(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module(
        "bench_qwen_consumer_mlx_cli_kv_guard_test",
        "scripts/bench_qwen_consumer_mlx.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen_consumer_mlx.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--butterfly-decode-backend",
            "experimental",
            "--kv-bits",
            "4",
            "--skip-single-turn",
            "--skip-multi-turn",
            "--skip-quality",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    err = capsys.readouterr().err
    assert "--kv-bits currently requires --butterfly-decode-backend stock" in err


def test_server_qwen_mlx_rejects_invalid_butterfly_prefill_chunking(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module(
        "serve_qwen_butterfly_mlx_cli_guard_test",
        "scripts/serve_qwen_butterfly_mlx.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_butterfly_mlx.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--prefill-chunk-size",
            "4096",
            "--query-chunk-size",
            "384",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    err = capsys.readouterr().err
    assert "--prefill-chunk-size exceeds --query-chunk-size" in err


def test_server_qwen_mlx_rejects_kv_quant_with_experimental_decode(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module(
        "serve_qwen_butterfly_mlx_cli_kv_guard_test",
        "scripts/serve_qwen_butterfly_mlx.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_butterfly_mlx.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--butterfly-decode-backend",
            "experimental",
            "--kv-bits",
            "4",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()

    err = capsys.readouterr().err
    assert "--kv-bits currently requires --butterfly-decode-backend stock" in err
