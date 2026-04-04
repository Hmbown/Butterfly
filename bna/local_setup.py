from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_SSD_ROOT = Path("/Volumes/VIXinSSD")
DEFAULT_LLAMA_CPP_TAG = "b8656"


@dataclass(frozen=True)
class LocalPaths:
    repo_root: Path
    ssd_root: Path
    hf_home: Path
    hf_hub_cache: Path
    models_root: Path
    toolchains_root: Path
    llama_src_root: Path
    llama_build_root: Path
    llama_install_root: Path
    llama_current_root: Path
    zmlx_root: Path

    def as_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def env(self) -> Dict[str, str]:
        return {
            "BUTTERFLY_REPO_ROOT": str(self.repo_root),
            "BUTTERFLY_SSD_ROOT": str(self.ssd_root),
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_hub_cache),
            "BUTTERFLY_MODELS_ROOT": str(self.models_root),
            "BUTTERFLY_TOOLCHAINS_ROOT": str(self.toolchains_root),
            "BUTTERFLY_LLAMA_CPP_SRC_ROOT": str(self.llama_src_root),
            "BUTTERFLY_LLAMA_CPP_BUILD_ROOT": str(self.llama_build_root),
            "BUTTERFLY_LLAMA_CPP_ROOT": str(self.llama_install_root),
            "BUTTERFLY_LLAMA_CPP_CURRENT_ROOT": str(self.llama_current_root),
            "BUTTERFLY_ZMLX_ROOT": str(self.zmlx_root),
        }

    def model_link_path(self, alias: str) -> Path:
        return self.models_root / alias


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    family: str
    size: str
    backend: str
    repo_kind: str
    runner: str
    support: str
    local_name: str
    repo_id: Optional[str] = None
    requires_auth: bool = False
    blocked_reason: Optional[str] = None
    notes: str = ""

    def link_path(self, paths: LocalPaths) -> Path:
        return paths.model_link_path(self.local_name)

    def to_row(self, paths: LocalPaths) -> Dict[str, object]:
        return {
            "alias": self.alias,
            "family": self.family,
            "size": self.size,
            "backend": self.backend,
            "repo_kind": self.repo_kind,
            "runner": self.runner,
            "support": self.support,
            "repo_id": self.repo_id,
            "requires_auth": self.requires_auth,
            "blocked_reason": self.blocked_reason,
            "notes": self.notes,
            "local_name": self.local_name,
            "local_path": str(self.link_path(paths)),
        }


def default_local_paths(
    repo_root: Path | str | None = None,
    *,
    ssd_root: Path | str | None = None,
    llama_tag: str | None = None,
    zmlx_root: Path | str | None = None,
) -> LocalPaths:
    repo_root_path = Path(repo_root or Path.cwd()).expanduser().resolve()
    resolved_ssd_root = Path(
        ssd_root or os.environ.get("BUTTERFLY_SSD_ROOT") or DEFAULT_SSD_ROOT
    ).expanduser()
    resolved_tag = str(
        llama_tag or os.environ.get("BUTTERFLY_LLAMA_CPP_TAG") or DEFAULT_LLAMA_CPP_TAG
    ).strip()
    resolved_zmlx_root = Path(
        zmlx_root or os.environ.get("BUTTERFLY_ZMLX_ROOT") or (resolved_ssd_root / "ZMLX")
    ).expanduser()
    toolchains_root = resolved_ssd_root / "toolchains"
    llama_install_root = toolchains_root / f"llama.cpp-{resolved_tag}-metal"
    return LocalPaths(
        repo_root=repo_root_path,
        ssd_root=resolved_ssd_root,
        hf_home=resolved_ssd_root / "hf_cache",
        hf_hub_cache=resolved_ssd_root / "hf_cache" / "hub",
        models_root=resolved_ssd_root / "models",
        toolchains_root=toolchains_root,
        llama_src_root=toolchains_root / "llama.cpp-src",
        llama_build_root=toolchains_root / f"llama.cpp-build-{resolved_tag}",
        llama_install_root=llama_install_root,
        llama_current_root=toolchains_root / "llama.cpp-metal-current",
        zmlx_root=resolved_zmlx_root,
    )


def list_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            alias="qwen35_4b_mlx_4bit",
            family="Qwen 3.5",
            size="4B",
            backend="mlx",
            repo_kind="mlx",
            runner="mlx_wayfinder",
            support="supported",
            local_name="Qwen3.5-4B-MLX-4bit",
            repo_id="mlx-community/Qwen3.5-4B-MLX-4bit",
            notes="Smallest Qwen 3.5 MLX target. Butterfly validated to 131K context.",
        ),
        ModelSpec(
            alias="qwen35_9b_mlx_4bit",
            family="Qwen 3.5",
            size="9B",
            backend="mlx",
            repo_kind="mlx",
            runner="mlx_wayfinder",
            support="supported",
            local_name="qwen35_9b_mlx_4bit",
            repo_id="mlx-community/Qwen3.5-9B-MLX-4bit",
            notes="Primary Apple Silicon Wayfinder benchmark target.",
        ),
        ModelSpec(
            alias="qwen35_35b_a3b_mlx_4bit",
            family="Qwen 3.5",
            size="35B-A3B",
            backend="mlx",
            repo_kind="mlx",
            runner="mlx_wayfinder",
            support="supported",
            local_name="qwen35_35b_a3b_mlx_4bit",
            repo_id="mlx-community/Qwen3.5-35B-A3B-4bit",
            notes="Large MLX baseline for Apple Silicon if memory permits.",
        ),
        ModelSpec(
            alias="qwen35_9b_hf",
            family="Qwen 3.5",
            size="9B",
            backend="hf",
            repo_kind="hf",
            runner="conversion_source",
            support="source_only",
            local_name="qwen35_9b_hf",
            repo_id="Qwen/Qwen3.5-9B",
            notes="Canonical source checkpoint. Use for conversion or non-MLX tooling.",
        ),
        ModelSpec(
            alias="qwen35_27b_hf",
            family="Qwen 3.5",
            size="27B",
            backend="hf",
            repo_kind="hf",
            runner="conversion_source",
            support="source_only",
            local_name="qwen35_27b_hf",
            repo_id="Qwen/Qwen3.5-27B",
            notes="Canonical source checkpoint for larger non-MLX experiments.",
        ),
        ModelSpec(
            alias="qwen35_4b_gguf",
            family="Qwen 3.5",
            size="4B",
            backend="llama.cpp",
            repo_kind="gguf",
            runner="llama_cpp_metal",
            support="supported",
            local_name="qwen35_4b_gguf",
            notes="Link a local GGUF file or pass --model/--hf-repo to the llama.cpp harness.",
        ),
        ModelSpec(
            alias="qwen35_9b_gguf",
            family="Qwen 3.5",
            size="9B",
            backend="llama.cpp",
            repo_kind="gguf",
            runner="llama_cpp_metal",
            support="supported",
            local_name="qwen35_9b_gguf",
            notes="Link a local GGUF file or pass --model/--hf-repo to the llama.cpp harness.",
        ),
        ModelSpec(
            alias="gemma4_31b_it_hf",
            family="Gemma 4",
            size="31B",
            backend="hf",
            repo_kind="hf",
            runner="conversion_source",
            support="source_only",
            local_name="gemma4_31b_it_hf",
            repo_id="google/gemma-4-31B-it",
            requires_auth=True,
            notes="Canonical source checkpoint. Immediate runtime path is llama.cpp via GGUF.",
        ),
        ModelSpec(
            alias="gemma4_26b_a4b_it_hf",
            family="Gemma 4",
            size="26B-A4B",
            backend="hf",
            repo_kind="hf",
            runner="conversion_source",
            support="source_only",
            local_name="gemma4_26b_a4b_it_hf",
            repo_id="google/gemma-4-26B-A4B-it",
            requires_auth=True,
            notes="Canonical source checkpoint. Immediate runtime path is llama.cpp via GGUF.",
        ),
        ModelSpec(
            alias="gemma4_31b_mlx_4bit",
            family="Gemma 4",
            size="31B",
            backend="mlx",
            repo_kind="mlx",
            runner="mlx_wayfinder",
            support="blocked",
            local_name="gemma4_31b_mlx_4bit",
            repo_id="mlx-community/gemma-4-31b-it-4bit",
            blocked_reason=(
                "mlx-lm 0.31.1 does not expose gemma4/gemma4_text model handlers."
            ),
            notes="Keep as download-only until mlx-lm adds Gemma 4 support.",
        ),
        ModelSpec(
            alias="gemma4_26b_a4b_mlx_4bit",
            family="Gemma 4",
            size="26B-A4B",
            backend="mlx",
            repo_kind="mlx",
            runner="mlx_wayfinder",
            support="blocked",
            local_name="gemma4_26b_a4b_mlx_4bit",
            repo_id="mlx-community/gemma-4-26b-a4b-it-4bit",
            blocked_reason=(
                "mlx-lm 0.31.1 does not expose gemma4/gemma4_text model handlers."
            ),
            notes="Keep as download-only until mlx-lm adds Gemma 4 support.",
        ),
        ModelSpec(
            alias="gemma4_31b_gguf",
            family="Gemma 4",
            size="31B",
            backend="llama.cpp",
            repo_kind="gguf",
            runner="llama_cpp_metal",
            support="supported",
            local_name="gemma4_31b_gguf",
            notes="Preferred immediate Gemma 4 baseline on Apple Silicon once a GGUF is available.",
        ),
        ModelSpec(
            alias="gemma4_26b_a4b_gguf",
            family="Gemma 4",
            size="26B-A4B",
            backend="llama.cpp",
            repo_kind="gguf",
            runner="llama_cpp_metal",
            support="supported",
            local_name="gemma4_26b_a4b_gguf",
            notes="Preferred immediate Gemma 4 baseline on Apple Silicon once a GGUF is available.",
        ),
    ]


def iter_model_rows(
    paths: LocalPaths,
    *,
    family: str | None = None,
    runner: str | None = None,
    support: str | None = None,
) -> Iterable[Dict[str, object]]:
    for spec in list_model_specs():
        if family and spec.family != family:
            continue
        if runner and spec.runner != runner:
            continue
        if support and spec.support != support:
            continue
        yield spec.to_row(paths)


def model_spec_by_alias(alias: str) -> ModelSpec:
    for spec in list_model_specs():
        if spec.alias == alias:
            return spec
    raise KeyError(f"Unknown model alias: {alias}")
