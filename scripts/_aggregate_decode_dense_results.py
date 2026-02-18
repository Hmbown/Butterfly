#!/usr/bin/env python3
"""Aggregate decode-dense-backend benchmark results into summary JSON + table."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

RUN_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "benchmarks/mlx/post_reboot_20260211_20260211T202821Z/decode_dense_backend_20260212T171140Z"
)

SEEDS = [42, 7, 99]
MODES = ["dense", "wayfinder", "sparse"]


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(vals: List[float]) -> float:
    return sum(vals) / max(1, len(vals))


def load_run(ds_label: str, seed: int, mode: str) -> Dict[str, Any]:
    p = RUN_ROOT / ds_label / f"s{seed}_{mode}" / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return json.loads(p.read_text())


def extract_decode_tok_s(data: Dict[str, Any]) -> List[float]:
    rows = data.get("single_turn") or []
    return [float(r["decode_tok_s"]) for r in rows if r.get("decode_tok_s") is not None]


def extract_peak_memory(data: Dict[str, Any]) -> List[int]:
    rows = data.get("single_turn") or []
    return [int(r["peak_memory_bytes"]) for r in rows if r.get("peak_memory_bytes") is not None]


def extract_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    q = data.get("quality") or {}
    return {
        "accuracy": float(q.get("accuracy", 0.0)),
        "correct": int(q.get("correct", 0)),
        "num_tasks": int(q.get("num_tasks", 0)),
    }


def extract_hsa_trace(data: Dict[str, Any]) -> Dict[str, Any]:
    rows = data.get("single_turn") or []
    for r in rows:
        if "hsa_trace_summary" in r:
            return r["hsa_trace_summary"]
    return {}


def main() -> None:
    per_seed: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for mode in MODES:
            six = load_run("six_task", seed, mode)
            held = load_run("heldout", seed, mode)

            decode_vals = extract_decode_tok_s(six)
            peak_vals = extract_peak_memory(six)
            six_q = extract_quality(six)
            held_q = extract_quality(held)

            per_seed[str(seed)][mode] = {
                "six_task_results_path": str(
                    RUN_ROOT / "six_task" / f"s{seed}_{mode}" / "results.json"
                ),
                "heldout_results_path": str(
                    RUN_ROOT / "heldout" / f"s{seed}_{mode}" / "results.json"
                ),
                "decode_tok_s_median": _median(decode_vals) if decode_vals else None,
                "peak_memory_bytes_median": _median([float(x) for x in peak_vals])
                if peak_vals
                else None,
                "six_task_accuracy": six_q["accuracy"],
                "heldout_accuracy": held_q["accuracy"],
                "six_task_correct": six_q["correct"],
                "heldout_correct": held_q["correct"],
                "six_task_num_tasks": six_q["num_tasks"],
                "heldout_num_tasks": held_q["num_tasks"],
            }

    # Cross-seed summary
    summary: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        dec_medians = [
            per_seed[str(s)][mode]["decode_tok_s_median"]
            for s in SEEDS
            if per_seed[str(s)][mode]["decode_tok_s_median"] is not None
        ]
        mem_medians = [
            per_seed[str(s)][mode]["peak_memory_bytes_median"]
            for s in SEEDS
            if per_seed[str(s)][mode]["peak_memory_bytes_median"] is not None
        ]
        six_accs = [per_seed[str(s)][mode]["six_task_accuracy"] for s in SEEDS]
        held_accs = [per_seed[str(s)][mode]["heldout_accuracy"] for s in SEEDS]

        summary[mode] = {
            "decode_tok_s_median_of_seeds": _median(dec_medians) if dec_medians else None,
            "peak_memory_bytes_median_of_seeds": _median(mem_medians) if mem_medians else None,
            "six_task_accuracy_mean": _mean(six_accs),
            "heldout_accuracy_mean": _mean(held_accs),
            "seed_values": {
                "decode_tok_s_medians": dec_medians,
                "peak_memory_bytes_medians": mem_medians,
                "six_task_accuracies": six_accs,
                "heldout_accuracies": held_accs,
            },
        }

    # Compute deltas vs dense
    d = summary["dense"]
    for mode in MODES:
        m = summary[mode]
        d_dec = d["decode_tok_s_median_of_seeds"] or 1.0
        d_mem = d["peak_memory_bytes_median_of_seeds"] or 1.0
        d_six = d["six_task_accuracy_mean"]
        d_held = d["heldout_accuracy_mean"]

        m_dec = m["decode_tok_s_median_of_seeds"] or 0.0
        m_mem = m["peak_memory_bytes_median_of_seeds"] or 0.0
        m_six = m["six_task_accuracy_mean"]
        m_held = m["heldout_accuracy_mean"]

        m["decode_delta_vs_dense_abs"] = m_dec - d_dec
        m["decode_delta_vs_dense_pct"] = 100.0 * (m_dec - d_dec) / d_dec if d_dec else 0.0
        m["peak_memory_delta_vs_dense_abs_bytes"] = m_mem - d_mem
        m["peak_memory_delta_vs_dense_pct"] = 100.0 * (m_mem - d_mem) / d_mem if d_mem else 0.0
        m["memory_reduction_pct_vs_dense"] = -m["peak_memory_delta_vs_dense_pct"]
        m["six_task_drift_vs_dense_abs"] = m_six - d_six
        m["six_task_drift_vs_dense_pct"] = (
            100.0 * (m_six - d_six) / d_six if d_six else 0.0
        )
        m["heldout_drift_vs_dense_abs"] = m_held - d_held
        m["heldout_drift_vs_dense_pct"] = (
            100.0 * (m_held - d_held) / d_held if d_held else 0.0
        )

    payload = {
        "run_root": str(RUN_ROOT),
        "seeds": SEEDS,
        "modes": MODES,
        "per_seed": per_seed,
        "summary": summary,
    }

    # Write summary JSON
    out_json = RUN_ROOT / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")

    # Write summary table
    header = (
        "| mode | decode_tok/s (median seeds) | decode Δ vs dense "
        "| peak_memory_bytes (median seeds) | peak Δ vs dense "
        "| memory reduction % vs dense "
        "| 6-task acc (mean seeds) | 6-task drift vs dense "
        "| held-out acc (mean seeds) | held-out drift vs dense |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = []
    for mode in MODES:
        m = summary[mode]
        dec = m["decode_tok_s_median_of_seeds"] or 0.0
        mem = m["peak_memory_bytes_median_of_seeds"] or 0
        rows.append(
            f"| {mode} "
            f"| {dec:.4f} "
            f"| {m['decode_delta_vs_dense_abs']:+.4f} ({m['decode_delta_vs_dense_pct']:+.2f}%) "
            f"| {int(mem)} "
            f"| {int(m['peak_memory_delta_vs_dense_abs_bytes']):+d} ({m['peak_memory_delta_vs_dense_pct']:+.2f}%) "
            f"| {m['memory_reduction_pct_vs_dense']:+.2f}% "
            f"| {m['six_task_accuracy_mean']:.4f} "
            f"| {m['six_task_drift_vs_dense_abs']:+.4f} ({m['six_task_drift_vs_dense_pct']:+.2f}%) "
            f"| {m['heldout_accuracy_mean']:.4f} "
            f"| {m['heldout_drift_vs_dense_abs']:+.4f} ({m['heldout_drift_vs_dense_pct']:+.2f}%) |"
        )

    table = "\n".join([header, sep] + rows) + "\n"
    out_md = RUN_ROOT / "summary_table.md"
    out_md.write_text(table)
    print(f"Wrote {out_md}")

    # Gate verdicts
    wf = summary["wayfinder"]
    gates = {
        "G1_six_task_acc_gte_040": {
            "pass": (wf["six_task_accuracy_mean"] >= 0.40),
            "value": wf["six_task_accuracy_mean"],
            "threshold": 0.40,
        },
        "G2_heldout_acc_gte_004": {
            "pass": (wf["heldout_accuracy_mean"] >= 0.04),
            "value": wf["heldout_accuracy_mean"],
            "threshold": 0.04,
        },
        "G3_decode_tok_s_gte_40": {
            "pass": ((wf["decode_tok_s_median_of_seeds"] or 0) >= 40.0),
            "value": wf["decode_tok_s_median_of_seeds"],
            "threshold": 40.0,
        },
        "G4_peak_mem_delta_lte_5pct": {
            "pass": (abs(wf["peak_memory_delta_vs_dense_pct"]) <= 5.0),
            "value": wf["peak_memory_delta_vs_dense_pct"],
            "threshold": 5.0,
        },
    }

    # G5: dense and sparse unchanged vs baseline
    baseline_dense_dec = 46.3677
    baseline_dense_six = 0.50
    baseline_sparse_dec = 46.4553
    baseline_sparse_six = 0.50

    d_dec_now = summary["dense"]["decode_tok_s_median_of_seeds"] or 0
    d_six_now = summary["dense"]["six_task_accuracy_mean"]
    s_dec_now = summary["sparse"]["decode_tok_s_median_of_seeds"] or 0
    s_six_now = summary["sparse"]["six_task_accuracy_mean"]

    gates["G5_dense_sparse_stable"] = {
        "pass": (
            abs(d_dec_now - baseline_dense_dec) / baseline_dense_dec <= 0.05
            and abs(d_six_now - baseline_dense_six) <= 0.05
            and abs(s_dec_now - baseline_sparse_dec) / baseline_sparse_dec <= 0.05
            and abs(s_six_now - baseline_sparse_six) <= 0.05
        ),
        "dense_decode_delta_pct": 100.0 * (d_dec_now - baseline_dense_dec) / baseline_dense_dec,
        "sparse_decode_delta_pct": 100.0 * (s_dec_now - baseline_sparse_dec) / baseline_sparse_dec,
    }

    all_pass = all(g["pass"] for g in gates.values())
    verdict = {"all_pass": all_pass, "gates": gates}
    out_gate = RUN_ROOT / "gate_verdict.json"
    out_gate.write_text(json.dumps(verdict, indent=2))
    print(f"Wrote {out_gate}")
    print()
    print("=== GATE VERDICT ===")
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {name}: {status}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print()
    print("=== SUMMARY TABLE ===")
    print(table)


if __name__ == "__main__":
    main()
