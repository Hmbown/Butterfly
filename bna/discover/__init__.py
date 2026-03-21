"""Setup-only scaffolding for fused-kernel discovery workflows.

This module intentionally avoids model loading, inference, or benchmark execution.
It only provides metadata, readiness checks, and session stub generation so the
repo is ready for future ZMLX discovery runs.
"""

from .readiness import build_readiness_report
from .session import prepare_discovery_workspace
from .targets import KernelTargetSpec, get_target, list_targets, resolve_targets

__all__ = [
    "KernelTargetSpec",
    "build_readiness_report",
    "get_target",
    "list_targets",
    "prepare_discovery_workspace",
    "resolve_targets",
]
