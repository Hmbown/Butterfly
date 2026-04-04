#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.local_setup import default_local_paths, iter_model_rows, model_spec_by_alias


def _local_snapshot_from_cache(paths, repo_id: str) -> Path | None:
    cache_root = paths.hf_hub_cache / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = cache_root / "snapshots"
    refs_main = cache_root / "refs" / "main"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / revision
        if candidate.exists():
            return candidate.resolve()

    if not snapshots_dir.exists():
        return None

    snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshot_dirs:
        return None
    snapshot_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshot_dirs[0].resolve()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _print_rows(rows: Iterable[Dict[str, object]], *, as_json: bool) -> None:
    row_list = list(rows)
    if as_json:
        print(json.dumps(row_list, indent=2, default=_json_default))
        return

    headers = [
        "alias",
        "family",
        "size",
        "runner",
        "support",
        "repo_id",
        "local_path",
    ]
    widths = {header: len(header) for header in headers}
    for row in row_list:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in row_list:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _create_or_update_symlink(link_path: Path, target_path: Path, *, force: bool) -> None:
    if link_path.is_symlink():
        current_target = Path(os.path.realpath(link_path))
        if current_target == target_path:
            return
        if not force:
            raise FileExistsError(f"{link_path} already points to {current_target}")
        link_path.unlink()
    elif link_path.exists():
        if not force:
            raise FileExistsError(f"{link_path} already exists")
        if link_path.is_dir():
            raise IsADirectoryError(
                f"{link_path} exists and is a directory; remove it manually before relinking"
            )
        link_path.unlink()

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def cmd_list(args: argparse.Namespace) -> None:
    paths = default_local_paths(REPO_ROOT)
    rows = iter_model_rows(
        paths,
        family=(str(args.family).strip() or None),
        runner=(str(args.runner).strip() or None),
        support=(str(args.support).strip() or None),
    )
    _print_rows(rows, as_json=bool(args.json))


def cmd_download(args: argparse.Namespace) -> None:
    spec = model_spec_by_alias(args.alias)
    if not spec.repo_id:
        raise SystemExit(
            f"{spec.alias} does not map to a single remote repo. Use link-local for GGUF files."
        )

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required. Run scripts/bootstrap_macos_metal.sh first."
        ) from exc

    paths = default_local_paths(REPO_ROOT)
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(paths.hf_home),
                local_files_only=bool(args.local_only),
                token=(str(args.hf_token).strip() or None),
            )
        )
    except LocalEntryNotFoundError:
        if not args.local_only:
            raise
        cached_snapshot = _local_snapshot_from_cache(paths, spec.repo_id)
        if cached_snapshot is None:
            raise
        snapshot_path = cached_snapshot
    snapshot_path = snapshot_path.resolve()
    link_path = spec.link_path(paths)
    _create_or_update_symlink(link_path, snapshot_path, force=bool(args.force))
    payload = {
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "snapshot_path": snapshot_path,
        "link_path": link_path,
        "support": spec.support,
        "blocked_reason": spec.blocked_reason,
    }
    print(json.dumps(payload, indent=2, default=_json_default))


def cmd_link_local(args: argparse.Namespace) -> None:
    paths = default_local_paths(REPO_ROOT)
    target_path = Path(args.path).expanduser().resolve()
    if not target_path.exists():
        raise SystemExit(f"Path does not exist: {target_path}")
    link_path = paths.model_link_path(args.alias)
    _create_or_update_symlink(link_path, target_path, force=bool(args.force))
    print(
        json.dumps(
            {
                "alias": args.alias,
                "target_path": target_path,
                "link_path": link_path,
            },
            indent=2,
            default=_json_default,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and stage local model aliases for Apple Silicon workflows."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List the built-in model catalog")
    p_list.add_argument("--family", type=str, default="")
    p_list.add_argument("--runner", type=str, default="")
    p_list.add_argument("--support", type=str, default="")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_download = sub.add_parser(
        "download",
        help="Download or resolve a remote HF/MLX repo into the shared cache and link it locally.",
    )
    p_download.add_argument("alias", type=str)
    p_download.add_argument("--hf-token", type=str, default="")
    p_download.add_argument("--local-only", action="store_true")
    p_download.add_argument("--force", action="store_true")
    p_download.set_defaults(func=cmd_download)

    p_link = sub.add_parser(
        "link-local",
        help="Link a local GGUF file or local snapshot directory into /Volumes/VIXinSSD/models.",
    )
    p_link.add_argument("alias", type=str)
    p_link.add_argument("path", type=str)
    p_link.add_argument("--force", action="store_true")
    p_link.set_defaults(func=cmd_link_local)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
