from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from .readiness import build_readiness_report
from .targets import KernelTargetSpec, resolve_targets


def _seed_kernel_template(spec: KernelTargetSpec) -> str:
    return f"""// Setup-only seed for {spec.kernel_name}
// This file is a scaffold for future ZMLX Discover runs.
// It is intentionally non-executable until a full kernel implementation is discovered.

#include <metal_stdlib>
using namespace metal;

kernel void {spec.kernel_name}(
    device const half* in0 [[buffer(0)]],
    device half* out0 [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    // TODO: replace with discovered implementation.
    out0[gid] = in0[gid];
}}
"""


def _session_stub(spec: KernelTargetSpec, repo_root: Path, zmlx_root: Path) -> Dict[str, object]:
    command = (
        f"python -m zmlx.discover search {spec.discover_target} "
        "--llm claude-code --steps 10"
    )
    return {
        "status": "setup_only",
        "target": asdict(spec),
        "question": spec.question,
        "hypothesis": "A fused kernel should reduce Python/dispatch overhead while preserving correctness.",
        "controls": {
            "models_loaded": False,
            "inference_executed": False,
            "attention_benchmark_executed": False,
            "retro_backfill_default_off": True,
        },
        "next_command_template": command,
        "zmlx_root": str(zmlx_root),
        "repo_root": str(repo_root),
        "notes": [
            "Setup scaffold only; no model loading and no inference.",
            "Run Discover from ZMLX once kernels are ready to be searched.",
        ],
    }


def prepare_discovery_workspace(
    *,
    repo_root: Path,
    zmlx_root: Path,
    sessions_root: Path,
    kernel_out_root: Path,
    targets: Iterable[KernelTargetSpec] | None = None,
    strict: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
    selected: List[KernelTargetSpec]
    if targets is None:
        selected = resolve_targets(["all"])
    else:
        selected = list(targets)

    readiness = build_readiness_report(repo_root=repo_root, zmlx_root=zmlx_root, strict=strict)
    blocked = bool(strict) and (not bool(readiness["ready"]))

    writes: List[Dict[str, object]] = []
    sessions_dir = sessions_root
    seeds_dir = kernel_out_root / "seeds"

    if not dry_run:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        if not blocked:
            seeds_dir.mkdir(parents=True, exist_ok=True)

    if not blocked:
        for spec in selected:
            seed_path = seeds_dir / Path(spec.seed_kernel_path).name
            stub_path = sessions_dir / spec.session_stub_name

            seed_content = _seed_kernel_template(spec)
            stub_content = json.dumps(
                _session_stub(spec, repo_root=repo_root, zmlx_root=zmlx_root),
                indent=2,
                sort_keys=True,
            )

            if dry_run:
                writes.append(
                    {
                        "type": "seed_kernel",
                        "path": str(seed_path),
                        "would_write": True,
                        "exists": seed_path.exists(),
                    }
                )
                writes.append(
                    {
                        "type": "session_stub",
                        "path": str(stub_path),
                        "would_write": True,
                        "exists": stub_path.exists(),
                    }
                )
                continue

            if overwrite or (not seed_path.exists()):
                seed_path.parent.mkdir(parents=True, exist_ok=True)
                seed_path.write_text(seed_content, encoding="utf-8")
                writes.append({"type": "seed_kernel", "path": str(seed_path), "written": True})
            else:
                writes.append({"type": "seed_kernel", "path": str(seed_path), "written": False})

            if overwrite or (not stub_path.exists()):
                stub_path.write_text(stub_content + "\n", encoding="utf-8")
                writes.append({"type": "session_stub", "path": str(stub_path), "written": True})
            else:
                writes.append({"type": "session_stub", "path": str(stub_path), "written": False})

    manifest = {
        "status": "setup_only",
        "ready": bool(readiness["ready"]),
        "blocked": blocked,
        "strict": bool(strict),
        "dry_run": bool(dry_run),
        "repo_root": str(repo_root),
        "zmlx_root": str(zmlx_root),
        "sessions_root": str(sessions_root),
        "kernel_out_root": str(kernel_out_root),
        "targets": [asdict(spec) for spec in selected],
        "readiness": readiness,
        "writes": writes,
    }

    manifest_path = sessions_dir / "manifest.json"
    if dry_run:
        manifest["manifest_path"] = str(manifest_path)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)

    return manifest
