#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.local_setup import default_local_paths


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_value(
    cli_value: Any,
    config: Dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if cli_value not in (None, "", []):
        return cli_value
    if key in config:
        return config[key]
    return default


def _parse_vm_snapshot() -> Dict[str, Any]:
    swap_text = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    vm_stat = subprocess.run(
        ["vm_stat"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout

    page_size_match = re.search(r"page size of (\d+) bytes", vm_stat)
    page_size = int(page_size_match.group(1)) if page_size_match else 0

    def _extract(pattern: str) -> int:
        match = re.search(pattern, vm_stat)
        if not match:
            return 0
        return int(match.group(1).replace(".", "").replace(",", ""))

    swap_used_match = re.search(r"used = ([0-9.]+)M", swap_text)
    swap_free_match = re.search(r"free = ([0-9.]+)M", swap_text)
    compressor_pages = _extract(r"Pages occupied by compressor:\s+([0-9.]+)")

    return {
        "swap_used_mb": float(swap_used_match.group(1)) if swap_used_match else 0.0,
        "swap_free_mb": float(swap_free_match.group(1)) if swap_free_match else 0.0,
        "page_size_bytes": page_size,
        "compressor_pages": compressor_pages,
        "compressor_bytes": compressor_pages * page_size,
    }


def _build_model_args(
    *,
    model: str,
    hf_repo: str,
    hf_file: str,
    hf_token: str,
) -> List[str]:
    if bool(model) == bool(hf_repo):
        raise ValueError("Provide exactly one of --model or --hf-repo (directly or via --config).")

    if model:
        return ["--model", model]

    args = ["--hf-repo", hf_repo]
    if hf_file:
        args.extend(["--hf-file", hf_file])
    if hf_token:
        args.extend(["--hf-token", hf_token])
    return args


def _run_command(
    cmd: Sequence[str],
    *,
    env: Dict[str, str],
    cwd: Path,
) -> Dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    ended = time.time()
    return {
        "cmd": list(cmd),
        "returncode": int(proc.returncode),
        "duration_sec": round(float(ended - started), 4),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_rss(stderr: str) -> int | None:
    match = re.search(r"^\s*(\d+)\s+maximum resident set size$", stderr, flags=re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def _summarize_perf_text(text: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    prompt_match = re.search(
        r"prompt eval time =\s*([0-9.]+)\s*ms / .*?\(\s*([0-9.]+)\s*tokens per second",
        text,
    )
    gen_match = re.search(
        r"eval time =\s*([0-9.]+)\s*ms / .*?\(\s*([0-9.]+)\s*tokens per second",
        text,
    )
    if prompt_match:
        summary["prompt_eval_ms"] = float(prompt_match.group(1))
        summary["prompt_tokens_per_sec"] = float(prompt_match.group(2))
    if gen_match:
        summary["gen_eval_ms"] = float(gen_match.group(1))
        summary["gen_tokens_per_sec"] = float(gen_match.group(2))
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible llama.cpp Metal speed, memory, and max-context probes."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--llama-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--hf-repo", type=str, default="")
    parser.add_argument("--hf-file", type=str, default="")
    parser.add_argument("--hf-token", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--prompt-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--gen-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--ctx-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--cache-type-k", type=str, default="")
    parser.add_argument("--cache-type-v", type=str, default="")
    parser.add_argument("--flash-attn", type=str, choices=["on", "off"], default="")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--mmap", dest="mmap", action="store_true")
    parser.add_argument("--no-mmap", dest="mmap", action="store_false")
    parser.set_defaults(mmap=None)
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-max-context", action="store_true")
    parser.add_argument("--max-context-start", type=int, default=None)
    parser.add_argument("--max-context-stop", type=int, default=None)
    parser.add_argument("--max-context-step", type=int, default=None)
    parser.add_argument("--probe-prompt", type=str, default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = _load_json(args.config)
    paths = default_local_paths(REPO_ROOT, llama_tag=str(config.get("llama_tag", "")).strip() or None)

    llama_root = Path(
        _resolve_value(
            args.llama_root,
            config,
            "llama_root",
            paths.llama_current_root if paths.llama_current_root.exists() else paths.llama_install_root,
        )
    )
    model = str(_resolve_value(args.model, config, "model", "")).strip()
    hf_repo = str(_resolve_value(args.hf_repo, config, "hf_repo", "")).strip()
    hf_file = str(_resolve_value(args.hf_file, config, "hf_file", "")).strip()
    hf_token = str(_resolve_value(args.hf_token, config, "hf_token", "")).strip()
    label = str(_resolve_value(args.label, config, "label", "llama_cpp_metal")).strip()
    prompt_lengths = [int(x) for x in _resolve_value(args.prompt_lengths, config, "prompt_lengths", [2048, 8192])]
    gen_lengths = [int(x) for x in _resolve_value(args.gen_lengths, config, "gen_lengths", [128])]
    ctx_sizes = [int(x) for x in _resolve_value(args.ctx_sizes, config, "ctx_sizes", [32768, 65536, 98304])]
    threads = int(_resolve_value(args.threads, config, "threads", 10))
    n_gpu_layers = int(_resolve_value(args.n_gpu_layers, config, "n_gpu_layers", 999))
    cache_type_k = str(_resolve_value(args.cache_type_k, config, "cache_type_k", "f16"))
    cache_type_v = str(_resolve_value(args.cache_type_v, config, "cache_type_v", "f16"))
    flash_attn = str(_resolve_value(args.flash_attn, config, "flash_attn", "on"))
    repetitions = int(_resolve_value(args.repetitions, config, "repetitions", 1))
    mmap_value = _resolve_value(args.mmap, config, "mmap", True)
    probe_prompt = str(
        _resolve_value(
            args.probe_prompt,
            config,
            "probe_prompt",
            "Write exactly one word: ready.",
        )
    )
    max_context_start = int(_resolve_value(args.max_context_start, config, "max_context_start", min(ctx_sizes)))
    max_context_stop = int(_resolve_value(args.max_context_stop, config, "max_context_stop", max(ctx_sizes)))
    max_context_step = int(_resolve_value(args.max_context_step, config, "max_context_step", 16384))

    model_args = _build_model_args(model=model, hf_repo=hf_repo, hf_file=hf_file, hf_token=hf_token)
    llama_bench = llama_root / "bin" / "llama-bench"
    llama_cli = llama_root / "bin" / "llama-cli"
    if not llama_bench.exists() or not llama_cli.exists():
        raise SystemExit(f"llama.cpp binaries not found under {llama_root}")

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(paths.hf_home))
    env.setdefault("HF_HUB_CACHE", str(paths.hf_hub_cache))
    env["DYLD_LIBRARY_PATH"] = (
        f"{llama_root / 'lib'}"
        + (f":{env['DYLD_LIBRARY_PATH']}" if env.get("DYLD_LIBRARY_PATH") else "")
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "label": label,
        "repo_root": REPO_ROOT,
        "llama_root": llama_root,
        "model": model or None,
        "hf_repo": hf_repo or None,
        "hf_file": hf_file or None,
        "threads": threads,
        "n_gpu_layers": n_gpu_layers,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "flash_attn": flash_attn,
        "repetitions": repetitions,
        "mmap": bool(mmap_value),
        "prompt_lengths": prompt_lengths,
        "gen_lengths": gen_lengths,
        "ctx_sizes": ctx_sizes,
    }
    _write_json(out_dir / "manifest.json", manifest)

    speed_rows: List[Dict[str, Any]] = []
    if not args.skip_speed:
        for prompt_len in prompt_lengths:
            for gen_len in gen_lengths:
                cmd = [
                    str(llama_bench),
                    "--output",
                    "json",
                    "--repetitions",
                    str(repetitions),
                    "--n-prompt",
                    str(prompt_len),
                    "--n-gen",
                    str(gen_len),
                    "--threads",
                    str(threads),
                    "--n-gpu-layers",
                    str(n_gpu_layers),
                    "--cache-type-k",
                    cache_type_k,
                    "--cache-type-v",
                    cache_type_v,
                    "--flash-attn",
                    "1" if flash_attn == "on" else "0",
                    "--mmap",
                    "1" if mmap_value else "0",
                ] + model_args
                row = _run_command(cmd, env=env, cwd=REPO_ROOT)
                row["prompt_len"] = prompt_len
                row["gen_len"] = gen_len
                try:
                    row["parsed_json"] = json.loads(row["stdout"]) if row["stdout"].strip() else None
                except json.JSONDecodeError:
                    row["parsed_json"] = None
                speed_rows.append(row)
        _write_json(out_dir / "speed.json", speed_rows)

    def _memory_probe(ctx_size: int) -> Dict[str, Any]:
        pre = _parse_vm_snapshot()
        cmd = [
            "/usr/bin/time",
            "-l",
            str(llama_cli),
            "--ctx-size",
            str(ctx_size),
            "--n-predict",
            "1",
            "--threads",
            str(threads),
            "--n-gpu-layers",
            str(n_gpu_layers),
            "--cache-type-k",
            cache_type_k,
            "--cache-type-v",
            cache_type_v,
            "--flash-attn",
            flash_attn,
            "--temp",
            "0",
            "--no-display-prompt",
            "--prompt",
            probe_prompt,
        ] + model_args
        if not mmap_value:
            cmd.append("--no-mmap")
        row = _run_command(cmd, env=env, cwd=REPO_ROOT)
        post = _parse_vm_snapshot()
        row["ctx_size"] = ctx_size
        row["pre_vm"] = pre
        row["post_vm"] = post
        row["max_rss_bytes"] = _extract_rss(row["stderr"])
        row["perf_summary"] = _summarize_perf_text(row["stdout"] + "\n" + row["stderr"])
        row["vm_delta"] = {
            "swap_used_mb": round(post["swap_used_mb"] - pre["swap_used_mb"], 2),
            "compressor_bytes": int(post["compressor_bytes"] - pre["compressor_bytes"]),
        }
        row["success"] = row["returncode"] == 0
        return row

    memory_rows: List[Dict[str, Any]] = []
    probe_cache: Dict[int, Dict[str, Any]] = {}
    if not args.skip_memory:
        for ctx_size in ctx_sizes:
            row = _memory_probe(ctx_size)
            probe_cache[ctx_size] = row
            memory_rows.append(row)
            if not row["success"]:
                break
        _write_json(out_dir / "memory.json", memory_rows)

    max_context_rows: List[Dict[str, Any]] = []
    max_context_candidates = list(range(max_context_start, max_context_stop + 1, max_context_step))
    if not args.skip_max_context:
        for ctx_size in max_context_candidates:
            if ctx_size in probe_cache:
                row = probe_cache[ctx_size]
            else:
                row = _memory_probe(ctx_size)
                probe_cache[ctx_size] = row
            max_context_rows.append(row)
            if not row["success"]:
                break
        _write_json(out_dir / "max_context.json", max_context_rows)

    successful_ctx = [int(row["ctx_size"]) for row in max_context_rows if row.get("success")]
    summary = {
        "label": label,
        "speed_runs": len(speed_rows),
        "memory_runs": len(memory_rows),
        "max_context_runs": len(max_context_rows),
        "max_successful_ctx": max(successful_ctx) if successful_ctx else None,
        "files": {
            "manifest": out_dir / "manifest.json",
            "speed": out_dir / "speed.json",
            "memory": out_dir / "memory.json",
            "max_context": out_dir / "max_context.json",
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
