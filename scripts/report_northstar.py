#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SeqMetrics:
    seq_len: int
    dense_tok_s: float
    wf_tok_s: float
    dense_peak_mem_bytes: int
    wf_peak_mem_bytes: int
    dense_step_intermediate_bytes: Optional[int]
    wf_step_intermediate_bytes: Optional[int]
    graph_build_ms_first: Optional[float]
    graph_build_ms_cached: Optional[float]
    cache_hit_rate: Optional[float]

    @property
    def throughput_ratio(self) -> float:
        if self.dense_tok_s <= 0.0:
            return float("nan")
        return self.wf_tok_s / self.dense_tok_s


@dataclass(frozen=True)
class RunSummary:
    label: str
    path: Path
    window: Optional[int]
    W: Optional[int]
    by_T: Dict[int, SeqMetrics]

    def c_fit(self, *, Ts: Iterable[int]) -> float:
        W = self.W
        if not W:
            return float("nan")
        c_vals: list[float] = []
        for T in Ts:
            m = self.by_T.get(int(T))
            if not m:
                continue
            r = m.throughput_ratio
            if not (r > 0.0) or math.isnan(r) or math.isinf(r):
                continue
            c_vals.append(float(T) / (float(W) * r))
        if not c_vals:
            return float("nan")
        c_vals.sort()
        return c_vals[len(c_vals) // 2]


def _as_path(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        return path / "results.json"
    return path


def _get(dct: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_seq_row(row: Dict[str, Any]) -> Optional[SeqMetrics]:
    if "error" in row:
        return None
    a = row.get("level_a_real_qkv", {})
    dense = a.get("baseline_attention", {})
    wf = a.get("wayfinder_attention", {})
    seq_len = _safe_int(row.get("seq_len"))
    if not seq_len:
        return None
    dense_tok_s = float(dense.get("tokens_per_sec", 0.0))
    wf_tok_s = float(wf.get("tokens_per_sec", 0.0))
    dense_peak = int(dense.get("peak_memory_bytes", 0))
    wf_peak = int(wf.get("peak_memory_bytes", 0))
    return SeqMetrics(
        seq_len=seq_len,
        dense_tok_s=dense_tok_s,
        wf_tok_s=wf_tok_s,
        dense_peak_mem_bytes=dense_peak,
        wf_peak_mem_bytes=wf_peak,
        dense_step_intermediate_bytes=_safe_int(dense.get("step_intermediate_bytes")),
        wf_step_intermediate_bytes=_safe_int(wf.get("step_intermediate_bytes")),
        graph_build_ms_first=_safe_float(wf.get("graph_build_ms_first")),
        graph_build_ms_cached=_safe_float(wf.get("graph_build_ms_cached")),
        cache_hit_rate=_safe_float(wf.get("cache_hit_rate")),
    )


def _parse_run(results_path: Path) -> RunSummary:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    label = results_path.parent.name
    wf_cfg = payload.get("wayfinder_config", {}) if isinstance(payload, dict) else {}
    window = _safe_int(wf_cfg.get("window"))
    W = 2 * window + 1 if window is not None else None
    by_T: Dict[int, SeqMetrics] = {}
    for row in payload.get("results", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict):
            continue
        m = _parse_seq_row(row)
        if m is None:
            continue
        by_T[int(m.seq_len)] = m
    return RunSummary(label=label, path=results_path, window=window, W=W, by_T=by_T)


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return f"{x:.3f}"
    return str(x)

def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def _northstar_rows(
    runs: List[RunSummary],
    *,
    Ts: List[int],
    fit_Ts: List[int],
) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "run",
        "T",
        "W",
        "ratio",
        "dense_tok/s",
        "wf_tok/s",
        "dense_peak_mem_bytes",
        "wf_peak_mem_bytes",
        "dense_step_intermediate_bytes",
        "wf_step_intermediate_bytes",
        "cache_hit_rate",
        "graph_build_ms_first",
        "graph_build_ms_cached",
        "C_fit",
        "T_star",
    ]
    rows: list[list[str]] = []
    for run in runs:
        W = run.W
        c = run.c_fit(Ts=fit_Ts)
        t_star = float(W) * c if W and not math.isnan(c) else float("nan")
        for T in Ts:
            m = run.by_T.get(int(T))
            if not m:
                continue
            rows.append(
                [
                    run.label,
                    str(int(T)),
                    _fmt(W),
                    _fmt(m.throughput_ratio),
                    _fmt(m.dense_tok_s),
                    _fmt(m.wf_tok_s),
                    _fmt(m.dense_peak_mem_bytes),
                    _fmt(m.wf_peak_mem_bytes),
                    _fmt(m.dense_step_intermediate_bytes),
                    _fmt(m.wf_step_intermediate_bytes),
                    _fmt(m.cache_hit_rate),
                    _fmt(m.graph_build_ms_first),
                    _fmt(m.graph_build_ms_cached),
                    _fmt(c),
                    _fmt(t_star),
                ]
            )
    return headers, rows


def main() -> None:
    p = argparse.ArgumentParser(description="Wayfinder/HCSA North Star report (results.json -> table)")
    p.add_argument("results", nargs="+", help="One or more benchmark output dirs or results.json paths")
    p.add_argument(
        "--Ts",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        help="Seq lengths to show/fit (default: 2048 4096 8192)",
    )
    args = p.parse_args()

    paths = [_as_path(r) for r in args.results]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing results files: {', '.join(missing)}")

    runs = [_parse_run(p) for p in paths]
    Ts = [int(t) for t in args.Ts]
    headers, rows = _northstar_rows(runs, Ts=Ts, fit_Ts=Ts)
    if not rows:
        raise SystemExit("No usable rows found (missing level_a_real_qkv rows?)")
    _print_table(headers, rows)


if __name__ == "__main__":
    main()
