#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_delta(new: float, baseline: float) -> Optional[float]:
    denom = abs(float(baseline))
    if denom <= 0.0:
        return None
    return float(100.0 * (float(new) - float(baseline)) / denom)


def _quality_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    quality = payload.get("quality") or {}
    rows = quality.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(x) for x in rows]


def _quality_accuracy(payload: Dict[str, Any]) -> float:
    quality = payload.get("quality") or {}
    return float(quality.get("accuracy", 0.0))


def _category_acc(rows: List[Dict[str, Any]], category: str) -> Optional[float]:
    chosen = [r for r in rows if str(r.get("id", "")).startswith(f"{category}-")]
    if not chosen:
        return None
    correct = sum(1 for r in chosen if bool(r.get("correct")))
    return float(correct / len(chosen))


def _parse_labeled_paths(items: List[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got: {item}")
        label, raw = item.split("=", 1)
        out.append((label.strip(), Path(raw.strip())))
    return out


def _cmd_localization(args: argparse.Namespace) -> Dict[str, Any]:
    dense = _load_json(Path(args.baseline_dense))
    wayfinder = _load_json(Path(args.baseline_wayfinder))
    dense_acc = _quality_accuracy(dense)
    wf_acc = _quality_accuracy(wayfinder)
    baseline_drift = float(wf_acc - dense_acc)
    dense_rows = _quality_rows(dense)
    wf_rows = _quality_rows(wayfinder)
    dense_extract = _category_acc(dense_rows, "extract")
    wf_extract = _category_acc(wf_rows, "extract")

    candidates: List[Dict[str, Any]] = []
    for label, path in _parse_labeled_paths(args.candidate):
        payload = _load_json(path)
        acc = _quality_accuracy(payload)
        drift = float(acc - dense_acc)
        extract = _category_acc(_quality_rows(payload), "extract")
        extract_delta = None
        if extract is not None and dense_extract is not None:
            extract_delta = float(extract - dense_extract)
        candidates.append(
            {
                "label": label,
                "path": str(path),
                "accuracy": acc,
                "drift_vs_dense": drift,
                "drift_delta_vs_baseline": float(drift - baseline_drift),
                "drift_delta_pct_vs_baseline": _pct_delta(drift, baseline_drift),
                "extract_accuracy": extract,
                "extract_delta_vs_dense": extract_delta,
            }
        )

    return {
        "baseline": {
            "dense_path": str(args.baseline_dense),
            "wayfinder_path": str(args.baseline_wayfinder),
            "dense_accuracy": dense_acc,
            "wayfinder_accuracy": wf_acc,
            "baseline_drift": baseline_drift,
            "extract_dense_accuracy": dense_extract,
            "extract_wayfinder_accuracy": wf_extract,
            "extract_baseline_delta": (
                None
                if dense_extract is None or wf_extract is None
                else float(wf_extract - dense_extract)
            ),
        },
        "candidates": candidates,
    }


def _trace_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = row.get("decode_trace")
    if not isinstance(trace, list):
        return []
    return [dict(x) for x in trace]


def _row_by_id(payload: Dict[str, Any], task_id: Optional[str]) -> Dict[str, Any]:
    rows = _quality_rows(payload)
    if not rows:
        raise ValueError("No quality rows found in payload.")
    if task_id is None:
        return rows[0]
    for row in rows:
        if str(row.get("id")) == str(task_id):
            return row
    raise ValueError(f"Task id {task_id} not found in quality rows.")


def _cmd_trace_diff(args: argparse.Namespace) -> Dict[str, Any]:
    dense = _load_json(Path(args.dense))
    wayfinder = _load_json(Path(args.wayfinder))
    dense_row = _row_by_id(dense, args.task_id)
    wf_row = _row_by_id(wayfinder, args.task_id)
    dense_trace = _trace_from_row(dense_row)
    wf_trace = _trace_from_row(wf_row)
    n = min(len(dense_trace), len(wf_trace))
    first_divergence_step = None
    for i in range(n):
        if int(dense_trace[i].get("chosen_token_id")) != int(
            wf_trace[i].get("chosen_token_id")
        ):
            first_divergence_step = int(i + 1)
            break
    return {
        "task_id": dense_row.get("id"),
        "dense_path": str(args.dense),
        "wayfinder_path": str(args.wayfinder),
        "dense_output": dense_row.get("output"),
        "wayfinder_output": wf_row.get("output"),
        "dense_correct": bool(dense_row.get("correct")),
        "wayfinder_correct": bool(wf_row.get("correct")),
        "trace_steps_compared": int(n),
        "first_divergence_step": first_divergence_step,
        "dense_trace_at_divergence": (
            dense_trace[first_divergence_step - 1] if first_divergence_step else None
        ),
        "wayfinder_trace_at_divergence": (
            wf_trace[first_divergence_step - 1] if first_divergence_step else None
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze GLM quality drift follow-up artifacts (localization + trace diff)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_loc = sub.add_parser("localization-summary")
    p_loc.add_argument("--baseline-dense", type=Path, required=True)
    p_loc.add_argument("--baseline-wayfinder", type=Path, required=True)
    p_loc.add_argument(
        "--candidate",
        type=str,
        action="append",
        required=True,
        help="label=path to candidate results.json (repeatable)",
    )
    p_loc.add_argument("--out", type=Path, required=True)

    p_trace = sub.add_parser("trace-diff")
    p_trace.add_argument("--dense", type=Path, required=True)
    p_trace.add_argument("--wayfinder", type=Path, required=True)
    p_trace.add_argument("--task-id", type=str, default=None)
    p_trace.add_argument("--out", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "localization-summary":
        payload = _cmd_localization(args)
    elif args.cmd == "trace-diff":
        payload = _cmd_trace_diff(args)
    else:
        raise ValueError(f"Unsupported command: {args.cmd}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(args.out))


if __name__ == "__main__":
    main()
