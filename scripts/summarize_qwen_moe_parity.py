#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.3f}%"


def _fmt_f(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.6f}"


def _fmt_gb(num_bytes: Optional[float]) -> str:
    if num_bytes is None or not math.isfinite(num_bytes):
        return "n/a"
    return f"{(num_bytes / (1024.0**3)):.3f}"


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
        return None
    return num / den


def _delta_abs(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None:
        return None
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return None
    return candidate - baseline


def _delta_pct(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None:
        return None
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return None
    if baseline == 0.0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def _memory_reduction_pct_convention(wayfinder: Optional[float], dense: Optional[float]) -> Optional[float]:
    if wayfinder is None or dense is None or not math.isfinite(wayfinder) or not math.isfinite(dense) or dense == 0.0:
        return None
    return 100.0 * (1.0 - (wayfinder / dense))


def _parse_seed(command: str) -> Optional[int]:
    m = re.search(r"--seed\\s+(\\d+)", command)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


@dataclass(frozen=True)
class SingleTurnAgg:
    seq_len: int
    decode_len: int
    repeats: int
    prefill_sec_mean: Optional[float]
    decode_sec_mean: Optional[float]
    e2e_sec_mean: Optional[float]
    decode_tok_s_mean: Optional[float]
    ttft_sec_mean: Optional[float]
    itl_p95_sec_mean: Optional[float]
    peak_memory_bytes_max: Optional[float]
    dense_fallback_share_run_mean: Optional[float]
    dense_fallback_share_decode_steps_mean: Optional[float]
    observability_fallback_share_known_all: Optional[bool]
    path_counts_sum: Dict[str, int]
    dense_fallback_reason_counts_sum: Dict[str, int]


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return float(max(xs))


def _sum_counts(dicts: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            try:
                out[str(k)] = int(out.get(str(k), 0)) + int(v)
            except Exception:
                continue
    return out


def _aggregate_single_turn(rows: List[Mapping[str, Any]]) -> SingleTurnAgg:
    if not rows:
        raise ValueError("cannot aggregate empty single_turn rows")
    seq_len = int(rows[0].get("seq_len", 0))
    decode_len = int(rows[0].get("decode_len", 0))
    repeats = len(rows)
    obs_known_vals = [row.get("observability_fallback_share_known") for row in rows]
    obs_known = None
    if all(isinstance(v, bool) for v in obs_known_vals):
        obs_known = bool(all(bool(v) for v in obs_known_vals))
    return SingleTurnAgg(
        seq_len=seq_len,
        decode_len=decode_len,
        repeats=repeats,
        prefill_sec_mean=_mean([row.get("prefill_sec") for row in rows]),
        decode_sec_mean=_mean([row.get("decode_sec") for row in rows]),
        e2e_sec_mean=_mean([row.get("e2e_sec") for row in rows]),
        decode_tok_s_mean=_mean([row.get("decode_tok_s") for row in rows]),
        ttft_sec_mean=_mean([row.get("ttft_sec") for row in rows]),
        itl_p95_sec_mean=_mean([row.get("itl_p95_sec") for row in rows]),
        peak_memory_bytes_max=_max([row.get("peak_memory_bytes") for row in rows]),
        dense_fallback_share_run_mean=_mean([row.get("dense_fallback_share_run") for row in rows]),
        dense_fallback_share_decode_steps_mean=_mean([row.get("dense_fallback_share_decode_steps") for row in rows]),
        observability_fallback_share_known_all=obs_known,
        path_counts_sum=_sum_counts([row.get("path_counts", {}) for row in rows if isinstance(row.get("path_counts"), dict)]),
        dense_fallback_reason_counts_sum=_sum_counts(
            [
                row.get("dense_fallback_reason_counts", {})
                for row in rows
                if isinstance(row.get("dense_fallback_reason_counts"), dict)
            ]
        ),
    )


def _group_single_turn(payload: Mapping[str, Any]) -> Dict[Tuple[int, int], SingleTurnAgg]:
    single_turn = payload.get("single_turn")
    if not isinstance(single_turn, list):
        return {}
    buckets: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in single_turn:
        if not isinstance(row, dict):
            continue
        key = (int(row.get("seq_len", 0)), int(row.get("decode_len", 0)))
        buckets.setdefault(key, []).append(row)
    return {k: _aggregate_single_turn(v) for k, v in buckets.items() if v}


def _quality_by_id(payload: Mapping[str, Any]) -> Optional[Dict[str, bool]]:
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    rows = quality.get("rows")
    if not isinstance(rows, list):
        return None
    out: Dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get("id")
        correct = row.get("correct")
        if isinstance(task_id, str) and isinstance(correct, bool):
            out[task_id] = bool(correct)
    return out


def _quality_summary(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    return {
        "dataset_path": quality.get("dataset_path"),
        "num_tasks": quality.get("num_tasks"),
        "correct": quality.get("correct"),
        "accuracy": quality.get("accuracy"),
    }


def _dataset_kind(payload: Mapping[str, Any]) -> Optional[str]:
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    dataset_path = str(quality.get("dataset_path") or "")
    if not dataset_path:
        return None
    name = Path(dataset_path).name
    if "holdout" in name.lower():
        return "holdout"
    return "main"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _md_table_perf(rows: List[Mapping[str, Any]]) -> str:
    headers = [
        "seq_len",
        "seed",
        "dense_prefill_sec",
        "wf_prefill_sec",
        "prefill_gain",
        "dense_e2e_sec",
        "wf_e2e_sec",
        "e2e_gain",
        "dense_peak_gb",
        "wf_peak_gb",
        "mem_reduction",
        "dense_decode_tok_s",
        "wf_decode_tok_s",
        "decode_tok_s_delta_pct",
        "wf_fallback_share_run",
    ]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(r.get("seq_len", "n/a")),
                    str(r.get("seed", "n/a")),
                    _fmt_f(r.get("dense", {}).get("prefill_sec")),
                    _fmt_f(r.get("wayfinder", {}).get("prefill_sec")),
                    _fmt_pct(r.get("prefill_gain_pct")),
                    _fmt_f(r.get("dense", {}).get("e2e_sec")),
                    _fmt_f(r.get("wayfinder", {}).get("e2e_sec")),
                    _fmt_pct(r.get("e2e_gain_pct")),
                    _fmt_gb(r.get("dense", {}).get("peak_memory_bytes")),
                    _fmt_gb(r.get("wayfinder", {}).get("peak_memory_bytes")),
                    _fmt_pct(r.get("memory_reduction_pct_convention")),
                    _fmt_f(r.get("dense", {}).get("decode_tok_s")),
                    _fmt_f(r.get("wayfinder", {}).get("decode_tok_s")),
                    _fmt_pct(r.get("delta_wayfinder_vs_dense_pct", {}).get("decode_tok_s")),
                    _fmt_f(r.get("wayfinder", {}).get("dense_fallback_share_run")),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Qwen3 MoE dense vs Wayfinder parity runs.")
    p.add_argument("--run-root", type=Path, required=True, help="Root dir containing run artifacts.")
    p.add_argument("--json-out", type=Path, default=None, help="Override JSON output path.")
    p.add_argument("--md-out", type=Path, default=None, help="Override Markdown output path.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any required paired artifacts (dense+wayfinder) are missing.",
    )
    args = p.parse_args()

    run_root = args.run_root.resolve()
    json_out = args.json_out or (run_root / "parity_summary.json")
    md_out = args.md_out or (run_root / "parity_summary.md")

    results_paths = sorted(run_root.rglob("results.json"))
    runs: List[Dict[str, Any]] = []
    missing_required: List[str] = []

    # Index by (mode, seed, kind) where kind is "perf" or "quality:<main|holdout>"
    perf_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    quality_index: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for path in results_paths:
        payload = _read_json(path)
        mode = str(payload.get("mode") or "")
        if mode not in {"dense", "wayfinder"}:
            continue
        command = str(payload.get("command") or "")
        seed = _parse_seed(command)
        if seed is None:
            wf_cfg = payload.get("wayfinder_config")
            if isinstance(wf_cfg, dict) and isinstance(wf_cfg.get("seed"), int):
                seed = int(wf_cfg["seed"])
        if seed is None:
            seed = 0

        single_turn_by_key = _group_single_turn(payload)
        quality_summary = _quality_summary(payload)
        dataset_kind = _dataset_kind(payload)

        run_rec: Dict[str, Any] = {
            "path": str(path),
            "mode": mode,
            "seed": int(seed),
            "model_path": payload.get("model_path"),
            "command": command,
            "single_turn": {f"{k[0]}:{k[1]}": agg.__dict__ for k, agg in single_turn_by_key.items()},
            "quality": quality_summary,
        }
        runs.append(run_rec)

        if quality_summary is not None and dataset_kind is not None:
            quality_index[(mode, int(seed), dataset_kind)] = {
                "path": str(path),
                "summary": quality_summary,
                "by_id": _quality_by_id(payload) or {},
            }
        else:
            perf_index[(mode, int(seed))] = {
                "path": str(path),
                "single_turn": single_turn_by_key,
            }

    # Build perf comparisons for required seq lens at decode_len=32.
    perf_rows: List[Dict[str, Any]] = []
    required_perf = [
        (2048, 32, [42]),
        (8192, 32, [42, 7, 99]),
    ]

    for (seq_len, decode_len, seeds) in required_perf:
        for seed in seeds:
            dense_rec = perf_index.get(("dense", seed))
            wf_rec = perf_index.get(("wayfinder", seed))
            if dense_rec is None:
                missing_required.append(f"missing dense perf seed={seed}")
                continue
            if wf_rec is None:
                missing_required.append(f"missing wayfinder perf seed={seed}")
                continue
            dense_agg = dense_rec["single_turn"].get((seq_len, decode_len))
            wf_agg = wf_rec["single_turn"].get((seq_len, decode_len))
            if dense_agg is None:
                missing_required.append(f"missing dense single_turn seq_len={seq_len} decode_len={decode_len} seed={seed}")
                continue
            if wf_agg is None:
                missing_required.append(f"missing wayfinder single_turn seq_len={seq_len} decode_len={decode_len} seed={seed}")
                continue

            dense = {
                "prefill_sec": dense_agg.prefill_sec_mean,
                "decode_sec": dense_agg.decode_sec_mean,
                "e2e_sec": dense_agg.e2e_sec_mean,
                "decode_tok_s": dense_agg.decode_tok_s_mean,
                "peak_memory_bytes": dense_agg.peak_memory_bytes_max,
                "ttft_sec": dense_agg.ttft_sec_mean,
                "itl_p95_sec": dense_agg.itl_p95_sec_mean,
            }
            wayfinder = {
                "prefill_sec": wf_agg.prefill_sec_mean,
                "decode_sec": wf_agg.decode_sec_mean,
                "e2e_sec": wf_agg.e2e_sec_mean,
                "decode_tok_s": wf_agg.decode_tok_s_mean,
                "peak_memory_bytes": wf_agg.peak_memory_bytes_max,
                "ttft_sec": wf_agg.ttft_sec_mean,
                "itl_p95_sec": wf_agg.itl_p95_sec_mean,
                "dense_fallback_share_run": wf_agg.dense_fallback_share_run_mean,
                "dense_fallback_share_decode_steps": wf_agg.dense_fallback_share_decode_steps_mean,
                "observability_fallback_share_known_all": wf_agg.observability_fallback_share_known_all,
                "path_counts_sum": wf_agg.path_counts_sum,
                "dense_fallback_reason_counts_sum": wf_agg.dense_fallback_reason_counts_sum,
            }

            perf_row = {
                "seq_len": int(seq_len),
                "decode_len": int(decode_len),
                "seed": int(seed),
                "dense_results_path": dense_rec["path"],
                "wayfinder_results_path": wf_rec["path"],
                "dense": dense,
                "wayfinder": wayfinder,
                "delta_wayfinder_vs_dense_abs": {
                    "prefill_sec": _delta_abs(wayfinder["prefill_sec"], dense["prefill_sec"]),
                    "decode_sec": _delta_abs(wayfinder["decode_sec"], dense["decode_sec"]),
                    "e2e_sec": _delta_abs(wayfinder["e2e_sec"], dense["e2e_sec"]),
                    "decode_tok_s": _delta_abs(wayfinder["decode_tok_s"], dense["decode_tok_s"]),
                    "peak_memory_bytes": _delta_abs(wayfinder["peak_memory_bytes"], dense["peak_memory_bytes"]),
                },
                "delta_wayfinder_vs_dense_pct": {
                    "prefill_sec": _delta_pct(wayfinder["prefill_sec"], dense["prefill_sec"]),
                    "decode_sec": _delta_pct(wayfinder["decode_sec"], dense["decode_sec"]),
                    "e2e_sec": _delta_pct(wayfinder["e2e_sec"], dense["e2e_sec"]),
                    "decode_tok_s": _delta_pct(wayfinder["decode_tok_s"], dense["decode_tok_s"]),
                    "peak_memory_bytes": _delta_pct(wayfinder["peak_memory_bytes"], dense["peak_memory_bytes"]),
                },
                "prefill_gain_pct": None,
                "e2e_gain_pct": None,
                "memory_reduction_pct_convention": _memory_reduction_pct_convention(
                    wayfinder["peak_memory_bytes"], dense["peak_memory_bytes"]
                ),
                "memory_regression_pct": None,
            }

            perf_row["prefill_gain_pct"] = (
                None
                if perf_row["delta_wayfinder_vs_dense_pct"]["prefill_sec"] is None
                else -float(perf_row["delta_wayfinder_vs_dense_pct"]["prefill_sec"])
            )
            perf_row["e2e_gain_pct"] = (
                None
                if perf_row["delta_wayfinder_vs_dense_pct"]["e2e_sec"] is None
                else -float(perf_row["delta_wayfinder_vs_dense_pct"]["e2e_sec"])
            )
            reg = _delta_pct(wayfinder["peak_memory_bytes"], dense["peak_memory_bytes"])
            perf_row["memory_regression_pct"] = reg

            perf_rows.append(perf_row)

    # Optional T=32768 decode_len=32 seed=42: summarize if present (do not require).
    optional_key = (32768, 32, 42)
    if ("dense", 42) in perf_index and ("wayfinder", 42) in perf_index:
        dense_32768 = perf_index[("dense", 42)]["single_turn"].get((32768, 32))
        wf_32768 = perf_index[("wayfinder", 42)]["single_turn"].get((32768, 32))
        if dense_32768 is not None and wf_32768 is not None:
            # Reuse the same construction path by treating it as a required row.
            pass

    # Quality comparisons at seq_len=8192 decode_len=64 seed=42 for main + holdout.
    quality_rows: Dict[str, Any] = {}
    for kind in ("main", "holdout"):
        dense_q = quality_index.get(("dense", 42, kind))
        wf_q = quality_index.get(("wayfinder", 42, kind))
        if dense_q is None:
            missing_required.append(f"missing dense quality kind={kind} seed=42")
            continue
        if wf_q is None:
            missing_required.append(f"missing wayfinder quality kind={kind} seed=42")
            continue
        dense_by_id: Dict[str, bool] = dict(dense_q.get("by_id") or {})
        wf_by_id: Dict[str, bool] = dict(wf_q.get("by_id") or {})
        lost_tasks = 0
        gained_tasks = 0
        for task_id, dense_ok in dense_by_id.items():
            wf_ok = bool(wf_by_id.get(task_id, False))
            if dense_ok and not wf_ok:
                lost_tasks += 1
            if (not dense_ok) and wf_ok:
                gained_tasks += 1
        quality_rows[kind] = {
            "dense_results_path": dense_q["path"],
            "wayfinder_results_path": wf_q["path"],
            "dense": dense_q["summary"],
            "wayfinder": wf_q["summary"],
            "lost_tasks_vs_dense": int(lost_tasks),
            "gained_tasks_vs_dense": int(gained_tasks),
        }

    # Gate evaluation (provisional similar-results gate at T=8192, seed=42, decode_len=32 + quality main).
    gate: Dict[str, Any] = {"status": "incomplete", "checks": {}, "notes": []}
    perf_8192_s42 = next(
        (
            r
            for r in perf_rows
            if int(r.get("seq_len", 0)) == 8192 and int(r.get("decode_len", 0)) == 32 and int(r.get("seed", -1)) == 42
        ),
        None,
    )
    if perf_8192_s42 is not None:
        prefill_gain = perf_8192_s42.get("prefill_gain_pct")
        e2e_gain = perf_8192_s42.get("e2e_gain_pct")
        mem_reg = perf_8192_s42.get("memory_regression_pct")
        obs_known = perf_8192_s42.get("wayfinder", {}).get("observability_fallback_share_known_all")
        gate["checks"]["observability_fallback_share_known"] = bool(obs_known) is True
        gate["checks"]["prefill_gain_ge_30"] = prefill_gain is not None and float(prefill_gain) >= 30.0
        gate["checks"]["e2e_gain_ge_20"] = e2e_gain is not None and float(e2e_gain) >= 20.0
        gate["checks"]["memory_regression_le_5"] = mem_reg is not None and float(mem_reg) <= 5.0
        if "main" in quality_rows:
            lost = int(quality_rows["main"].get("lost_tasks_vs_dense", 10**9))
            gate["checks"]["quality_drift_le_1_task_main"] = lost <= 1
        else:
            gate["checks"]["quality_drift_le_1_task_main"] = False

        all_pass = all(bool(v) for v in gate["checks"].values())
        gate["status"] = "PASS" if all_pass else "FAIL"
    else:
        gate["notes"].append("Missing required perf slice: T=8192 decode_len=32 seed=42 dense+wayfinder.")

    out_payload: Dict[str, Any] = {
        "run_root": str(run_root),
        "results_files_found": len(results_paths),
        "runs_indexed": len(runs),
        "missing_required": missing_required,
        "perf_rows": perf_rows,
        "quality": quality_rows,
        "gate_T8192": gate,
        "memory_convention": "100 * (1 - wayfinder/dense) (positive = Wayfinder uses less memory)",
    }

    _write_json(json_out, out_payload)

    md_lines: List[str] = []
    md_lines.append(f"# Qwen3 MoE Parity Summary ({run_root.name})")
    md_lines.append("")
    md_lines.append(f"- run_root: `{run_root}`")
    md_lines.append(f"- results_files_found: `{len(results_paths)}`")
    md_lines.append(f"- missing_required: `{len(missing_required)}`")
    if missing_required:
        md_lines.append("")
        md_lines.append("## Missing Required Artifacts")
        md_lines.append("")
        for item in missing_required:
            md_lines.append(f"- {item}")
    md_lines.append("")
    md_lines.append("## Perf (dense vs wayfinder)")
    md_lines.append("")
    md_lines.append(_md_table_perf(perf_rows))
    md_lines.append("")
    md_lines.append("## Quality (T=8192, seed=42)")
    md_lines.append("")
    if not quality_rows:
        md_lines.append("- n/a (no quality artifacts indexed)")
    else:
        md_lines.append("| dataset | dense_acc | wayfinder_acc | lost_tasks_vs_dense | gained_tasks_vs_dense |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for kind, row in quality_rows.items():
            dense_acc = row.get("dense", {}).get("accuracy")
            wf_acc = row.get("wayfinder", {}).get("accuracy")
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        kind,
                        _fmt_f(dense_acc),
                        _fmt_f(wf_acc),
                        str(row.get("lost_tasks_vs_dense", "n/a")),
                        str(row.get("gained_tasks_vs_dense", "n/a")),
                    ]
                )
                + " |"
            )
    md_lines.append("")
    md_lines.append("## Gate (provisional similar-results @ T=8192)")
    md_lines.append("")
    md_lines.append(f"- status: `{gate.get('status')}`")
    checks = gate.get("checks", {})
    if isinstance(checks, dict) and checks:
        for k, v in checks.items():
            md_lines.append(f"- {k}: `{v}`")
    notes = gate.get("notes", [])
    if isinstance(notes, list) and notes:
        md_lines.append("")
        md_lines.append("Notes:")
        for n in notes:
            md_lines.append(f"- {n}")

    _write_text(md_out, "\n".join(md_lines) + "\n")

    if args.strict and missing_required:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

