#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_NDJSON = REPO_ROOT / "notes" / "experiments.ndjson"

DEFAULT_NANBEIGE_EXP_ID = "EXP-20260217T183144Z-SECTION4-NANBEIGE-QCHUNK-SWEEP-PRERUN"
DEFAULT_QWEN_EXP_ID = "EXP-20260217T183144Z-SECTION4-QWEN3-1_7B-DECODELEN-SWEEP-PRERUN"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_experiment_command(exp_id: str) -> str:
    if not EXPERIMENTS_NDJSON.is_file():
        raise FileNotFoundError(f"Missing experiments ledger: {EXPERIMENTS_NDJSON}")
    for obj in _read_ndjson(EXPERIMENTS_NDJSON):
        if obj.get("id") == exp_id:
            cmd = obj.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"Experiment {exp_id} has no usable command field.")
            return cmd
    raise KeyError(f"Experiment id not found in {EXPERIMENTS_NDJSON}: {exp_id}")


def _split_shell_and(cmd: str) -> List[str]:
    parts = [p.strip() for p in cmd.split("&&")]
    return [p for p in parts if p]


def _parse_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    if flag in argv:
        idx = list(argv).index(flag)
        if idx + 1 >= len(argv):
            raise ValueError(f"Flag {flag} is missing a value in command: {' '.join(argv)}")
        return str(argv[idx + 1])
    prefix = f"{flag}="
    for token in argv:
        if token.startswith(prefix):
            return str(token[len(prefix) :])
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return obj


def _get_single_turn_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("single_turn")
    if rows is None:
        return []
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def _extract_counts(row: Dict[str, Any], key: str) -> Dict[str, int]:
    raw = row.get(key)
    if not isinstance(raw, dict):
        raw = row.get("hsa_trace_summary", {}).get(key)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def _aggregate_counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        for k, v in _extract_counts(row, key).items():
            c[str(k)] += int(v)
    return dict(c)


def _dense_fallback_present(path_counts: Dict[str, int]) -> Tuple[bool, int]:
    total = 0
    for k, v in path_counts.items():
        if "dense_fallback" not in str(k):
            continue
        if int(v) <= 0:
            continue
        total += int(v)
    return (total > 0, total)


def _reasons_informative(reason_counts: Dict[str, int]) -> bool:
    if not reason_counts:
        return False
    for k, v in reason_counts.items():
        if int(v) <= 0:
            continue
        ks = str(k).strip().lower()
        if not ks:
            continue
        if ks in {"unspecified", "none", "null"}:
            continue
        return True
    return False


def _enforce_fallback_reason_stop_gate(results_path: Path) -> None:
    payload = _load_json(results_path)
    rows = _get_single_turn_rows(payload)
    if not rows:
        raise RuntimeError(f"No single_turn rows found in {results_path}")

    path_counts = _aggregate_counts(rows, "path_counts")
    reason_counts = _aggregate_counts(rows, "dense_fallback_reason_counts")
    has_fallback, fallback_obs = _dense_fallback_present(path_counts)
    if not has_fallback:
        return

    if not _reasons_informative(reason_counts):
        raise RuntimeError(
            "Stop-gate: dense fallback present but fallback reasons are missing/unspecified "
            f"(fallback_obs={fallback_obs}, reason_counts={reason_counts}). "
            f"See {results_path}"
        )


def _run_one_command(
    argv: Sequence[str],
    *,
    timeout_sec: float,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    out_dir = _parse_flag_value(argv, "--out-dir")
    if out_dir is None:
        raise ValueError(f"Missing --out-dir in command: {' '.join(argv)}")
    out_path = (REPO_ROOT / out_dir).resolve()

    results_path = out_path / "results.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to reuse existing out_dir without --overwrite: {out_path}")

    _log(f"\n==> {' '.join(argv)}")
    _log(f"    out_dir={out_path}")
    if dry_run:
        return results_path

    out_path.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(REPO_ROOT),
            check=False,
            timeout=float(timeout_sec) if timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Timeout after {timeout_sec}s: {' '.join(argv)}") from exc
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Nonzero exit ({proc.returncode}) after {dt:.1f}s: {' '.join(argv)}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.json after {dt:.1f}s: {results_path}")
    _log(f"    ok: wrote {results_path} ({dt:.1f}s)")
    return results_path


def _run_experiment(exp_id: str, *, timeout_sec: float, overwrite: bool, dry_run: bool) -> List[Path]:
    cmd = _load_experiment_command(exp_id)
    segments = _split_shell_and(cmd)
    if not segments:
        raise ValueError(f"Experiment {exp_id} command had no runnable segments.")
    _log(f"\n==== Running {exp_id} ({len(segments)} commands) ====")

    results: List[Path] = []
    for seg in segments:
        argv = shlex.split(seg)
        results_path = _run_one_command(
            argv,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if not dry_run:
            mode = _parse_flag_value(argv, "--mode") or ""
            if str(mode).strip().lower() == "wayfinder":
                _enforce_fallback_reason_stop_gate(results_path)
        results.append(results_path)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Run Section 4 discriminating experiments with stop-gates.")
    p.add_argument("--nanbeige-exp-id", type=str, default=DEFAULT_NANBEIGE_EXP_ID)
    p.add_argument("--qwen-exp-id", type=str, default=DEFAULT_QWEN_EXP_ID)
    p.add_argument("--skip-nanbeige", action="store_true", default=False)
    p.add_argument("--skip-qwen", action="store_true", default=False)
    p.add_argument("--timeout-sec", type=float, default=1800.0, help="Per-command timeout (0 disables).")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    try:
        all_results: List[Path] = []
        if not bool(args.skip_nanbeige):
            all_results.extend(
                _run_experiment(
                    str(args.nanbeige_exp_id),
                    timeout_sec=float(args.timeout_sec),
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                )
            )
        if not bool(args.skip_qwen):
            all_results.extend(
                _run_experiment(
                    str(args.qwen_exp_id),
                    timeout_sec=float(args.timeout_sec),
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                )
            )
        _log("\nDone.")
        if args.dry_run:
            return 0
        _log("Results:")
        for path in all_results:
            _log(f"- {path}")
        return 0
    except Exception as exc:
        _log(f"\nERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
