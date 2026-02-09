#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from hcsa.compiler import compile_graph_spec, load_graph_ir
from hcsa.compiler.passes import normalize_pass, validate_pass
from hcsa.discover import prepare_discovery_workspace, resolve_targets


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return str(x)


def cmd_validate(args: argparse.Namespace) -> None:
    ir = validate_pass(normalize_pass(load_graph_ir(args.spec)))
    print(json.dumps({"ok": True, "ir": ir.to_dict()}, indent=2, default=_json_default))


def cmd_dump(args: argparse.Namespace) -> None:
    ir = validate_pass(normalize_pass(load_graph_ir(args.spec)))
    payload = {"ir": ir.to_dict()}
    if args.format == "json":
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        print(payload)


def cmd_compile(args: argparse.Namespace) -> None:
    out_dir = Path(args.out) if args.out else None
    result = compile_graph_spec(
        args.spec,
        T=args.T,
        H=args.H,
        dtype=args.dtype,
        out_root=args.out_root,
        out_dir=out_dir,
    )
    artifact_dir = result["artifact"]["artifact_dir"]
    print(json.dumps(
        {
            "ok": True,
            "cache_hash": result["cache_hash"],
            "artifact_dir": artifact_dir,
            "graph_metrics": result["lowered"]["graph_metrics"],
        },
        indent=2,
        default=_json_default,
    ))


def _bench_latency(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def cmd_bench(args: argparse.Namespace) -> None:
    try:
        import mlx.core as mx

        from hcsa.mlx.attention import WayfinderAttentionMLX
        from hcsa.mlx.model import DenseCausalAttentionMLX
    except ImportError as e:  # pragma: no cover - runtime environment check
        raise SystemExit(f"MLX bench command requires MLX runtime: {e}")

    compiled = compile_graph_spec(
        args.spec,
        T=args.T,
        H=args.H,
        dtype=args.dtype,
        out_root=args.out_root,
    )
    artifact_dir = compiled["artifact"]["artifact_dir"]

    dense = DenseCausalAttentionMLX(args.embd, args.H, dropout=0.0)
    permute = WayfinderAttentionMLX(
        args.embd,
        args.H,
        window=args.window,
        landmark_stride=args.landmark_stride,
        strategy="random",
        path="permute",
        dropout=0.0,
        compiled_graph_dir=artifact_dir,
    )

    x = mx.random.normal((args.B, args.T, args.embd), dtype=mx.float16)

    def run_dense() -> None:
        y = dense(x)
        mx.eval(y)

    def run_perm() -> None:
        y = permute(x)
        mx.eval(y)

    dense_s = _bench_latency(run_dense, iters=args.iters, warmup=args.warmup)
    perm_s = _bench_latency(run_perm, iters=args.iters, warmup=args.warmup)

    payload = {
        "artifact_dir": artifact_dir,
        "dense_tok_s": float((args.B * args.T) / max(dense_s, 1e-12)),
        "permute_tok_s": float((args.B * args.T) / max(perm_s, 1e-12)),
        "dense_ms": dense_s * 1000.0,
        "permute_ms": perm_s * 1000.0,
    }
    print(json.dumps(payload, indent=2))


def cmd_discover_targets(args: argparse.Namespace) -> None:
    targets = resolve_targets(args.targets)
    payload = {
        "setup_only": True,
        "targets": [t.to_dict() for t in targets],
    }
    text = json.dumps(payload, indent=2, default=_json_default)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


def cmd_discover_setup(args: argparse.Namespace) -> None:
    targets = resolve_targets(args.targets)
    manifest = prepare_discovery_workspace(
        repo_root=args.repo_root.resolve(),
        zmlx_root=args.zmlx_root.resolve(),
        sessions_root=args.sessions_root.resolve(),
        kernel_out_root=args.kernel_out_root.resolve(),
        targets=targets,
        strict=args.strict,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    text = json.dumps(manifest, indent=2, default=_json_default)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.strict and (not bool(manifest.get("ready", False))):
        raise SystemExit(2)


def cmd_discover_status(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="wayc: compile Wayfinder .wf graph specs")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate .wf graph spec")
    p_validate.add_argument("spec", type=Path)
    p_validate.set_defaults(func=cmd_validate)

    p_compile = sub.add_parser("compile", help="Compile .wf spec into cache artifact")
    p_compile.add_argument("spec", type=Path)
    p_compile.add_argument("--T", type=int, required=True)
    p_compile.add_argument("--H", type=int, required=True)
    p_compile.add_argument("--dtype", type=str, default="float16")
    p_compile.add_argument("--out-root", type=Path, default=Path(".cache/wayfinder"))
    p_compile.add_argument("--out", type=Path, default=None)
    p_compile.set_defaults(func=cmd_compile)

    p_dump = sub.add_parser("dump", help="Dump normalized GraphIR as JSON")
    p_dump.add_argument("spec", type=Path)
    p_dump.add_argument("--format", type=str, default="json", choices=["json", "repr"])
    p_dump.set_defaults(func=cmd_dump)

    p_bench = sub.add_parser("bench", help="Compile spec and run tiny MLX microbench")
    p_bench.add_argument("spec", type=Path)
    p_bench.add_argument("--T", type=int, required=True)
    p_bench.add_argument("--H", type=int, required=True)
    p_bench.add_argument("--B", type=int, default=2)
    p_bench.add_argument("--embd", type=int, default=128)
    p_bench.add_argument("--window", type=int, default=32)
    p_bench.add_argument("--landmark-stride", type=int, default=64)
    p_bench.add_argument("--dtype", type=str, default="float16")
    p_bench.add_argument("--warmup", type=int, default=3)
    p_bench.add_argument("--iters", type=int, default=10)
    p_bench.add_argument("--out-root", type=Path, default=Path(".cache/wayfinder"))
    p_bench.set_defaults(func=cmd_bench)

    p_discover_targets = sub.add_parser(
        "discover-targets",
        help="List fused-kernel discovery target metadata (setup-only).",
    )
    p_discover_targets.add_argument(
        "--targets",
        nargs="*",
        default=["all"],
        help="Target ids/names (k1..k5) or all.",
    )
    p_discover_targets.add_argument("--json-out", type=Path, default=None)
    p_discover_targets.set_defaults(func=cmd_discover_targets)

    p_discover_setup = sub.add_parser(
        "discover-setup",
        help="Validate setup and write discovery session stubs (no model loading/inference).",
    )
    p_discover_setup.add_argument(
        "--targets",
        nargs="*",
        default=["all"],
        help="Target ids/names (k1..k5) or all.",
    )
    p_discover_setup.add_argument("--repo-root", type=Path, default=Path.cwd())
    p_discover_setup.add_argument("--zmlx-root", type=Path, default=Path("/Volumes/VIXinSSD/ZMLX"))
    p_discover_setup.add_argument("--sessions-root", type=Path, default=Path("discover_sessions"))
    p_discover_setup.add_argument(
        "--kernel-out-root",
        type=Path,
        default=Path("hcsa/mlx/kernels/metal"),
    )
    p_discover_setup.add_argument("--strict", action="store_true")
    p_discover_setup.add_argument("--dry-run", action="store_true")
    p_discover_setup.add_argument("--overwrite", action="store_true")
    p_discover_setup.add_argument("--json-out", type=Path, default=None)
    p_discover_setup.set_defaults(func=cmd_discover_setup)

    p_discover_status = sub.add_parser(
        "discover-status",
        help="Read and print a discovery setup manifest.",
    )
    p_discover_status.add_argument(
        "--manifest",
        type=Path,
        default=Path("discover_sessions/manifest.json"),
    )
    p_discover_status.set_defaults(func=cmd_discover_status)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
