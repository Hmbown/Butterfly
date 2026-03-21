from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class _Requirement:
    category: str
    label: str
    path: Path
    required: bool


def _repo_requirements(repo_root: Path) -> List[_Requirement]:
    return [
        _Requirement("repo", "attention_reference", repo_root / "hcsa/mlx/attention.py", True),
        _Requirement("repo", "metal_seam", repo_root / "hcsa/mlx/kernels/metal/__init__.py", True),
        _Requirement("repo", "glm_integration", repo_root / "hcsa/integrations/glm_mlx.py", True),
        _Requirement("repo", "qwen_integration", repo_root / "hcsa/integrations/qwen_mlx.py", True),
        _Requirement("repo", "lab_notebook", repo_root / "notes/LAB_NOTEBOOK.md", True),
        _Requirement("repo", "experiments_log", repo_root / "notes/experiments.ndjson", True),
        _Requirement("repo", "quality_script_base", repo_root / "scripts/eval_gpt2_quality.py", True),
    ]


def _zmlx_requirements(zmlx_root: Path, *, strict: bool) -> List[_Requirement]:
    required = bool(strict)
    return [
        _Requirement("zmlx", "discover_targets", zmlx_root / "src/zmlx/discover/targets.py", required),
        _Requirement("zmlx", "discover_prompts", zmlx_root / "src/zmlx/discover/prompts.py", required),
        _Requirement("zmlx", "discover_tree", zmlx_root / "src/zmlx/discover/tree.py", required),
        _Requirement("zmlx", "discover_evaluate", zmlx_root / "src/zmlx/discover/evaluate.py", required),
        _Requirement("zmlx", "discover_playbook", zmlx_root / "docs/DISCOVER_PLAYBOOK.md", required),
    ]


def _path_status(req: _Requirement) -> Dict[str, object]:
    exists = req.path.exists()
    return {
        "category": req.category,
        "label": req.label,
        "path": str(req.path),
        "required": req.required,
        "exists": exists,
        "ok": exists or (not req.required),
    }


def _check_retro_defaults(repo_root: Path) -> Dict[str, object]:
    checks = [
        ("hcsa/mlx/model.py", "retro_backfill_enabled: bool = False"),
        ("hcsa/integrations/glm_mlx.py", "retro_backfill_enabled: bool = False"),
        ("hcsa/integrations/qwen_mlx.py", "retro_backfill_enabled: bool = False"),
    ]
    rows: List[Dict[str, object]] = []
    all_ok = True
    for rel_path, needle in checks:
        path = repo_root / rel_path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        found = needle in text
        rows.append(
            {
                "path": str(path),
                "required_snippet": needle,
                "found": found,
            }
        )
        all_ok = all_ok and found
    return {"ok": all_ok, "checks": rows}


def build_readiness_report(
    *,
    repo_root: Path,
    zmlx_root: Path,
    strict: bool,
) -> Dict[str, object]:
    path_rows: List[Dict[str, object]] = []
    for req in _repo_requirements(repo_root):
        path_rows.append(_path_status(req))
    for req in _zmlx_requirements(zmlx_root, strict=strict):
        path_rows.append(_path_status(req))

    required_failures = [row for row in path_rows if row["required"] and (not row["exists"])]
    optional_missing = [row for row in path_rows if (not row["required"]) and (not row["exists"])]
    retro = _check_retro_defaults(repo_root)

    ready = (len(required_failures) == 0) and bool(retro["ok"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "zmlx_root": str(zmlx_root),
        "strict": bool(strict),
        "ready": ready,
        "retro_defaults_ok": bool(retro["ok"]),
        "path_checks": path_rows,
        "required_failures": required_failures,
        "optional_missing": optional_missing,
        "retro_checks": retro["checks"],
    }
