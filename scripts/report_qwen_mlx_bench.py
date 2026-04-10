#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _resolve_results_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"results.json not found: {path}")
    return path


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--run must look like LABEL=PATH, got: {spec!r}")
    label, path_text = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"--run label cannot be empty: {spec!r}")
    return label, _resolve_results_path(path_text.strip())


def _aggregate_numeric(values: Sequence[float], mode: str) -> float:
    if not values:
        raise ValueError("cannot aggregate empty value list")
    if mode == "mean":
        return float(statistics.mean(values))
    if mode == "median":
        return float(statistics.median(values))
    if mode == "min":
        return float(min(values))
    if mode == "max":
        return float(max(values))
    raise ValueError(f"unsupported aggregate mode: {mode}")


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_reason_counts(reason_counts: Counter[str]) -> str:
    if not reason_counts:
        return ""
    return ", ".join(f"{key}:{reason_counts[key]}" for key in sorted(reason_counts))


@dataclass
class AggregatedRow:
    seq_len: int
    repeats: int
    decode_len: int
    prefill_sec: float
    ttft_sec: float
    decode_tok_s: float | None
    e2e_sec: float
    peak_memory_bytes: float | None
    stock_fallback_share_run: float | None
    stock_fallback_share_decode_steps: float | None
    kv_quantization_active: bool
    quantized_entries: int
    fallback_reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_len": int(self.seq_len),
            "repeats": int(self.repeats),
            "decode_len": int(self.decode_len),
            "prefill_sec": float(self.prefill_sec),
            "ttft_sec": float(self.ttft_sec),
            "decode_tok_s": (
                None if self.decode_tok_s is None else float(self.decode_tok_s)
            ),
            "e2e_sec": float(self.e2e_sec),
            "peak_memory_bytes": (
                None if self.peak_memory_bytes is None else float(self.peak_memory_bytes)
            ),
            "stock_fallback_share_run": (
                None
                if self.stock_fallback_share_run is None
                else float(self.stock_fallback_share_run)
            ),
            "stock_fallback_share_decode_steps": (
                None
                if self.stock_fallback_share_decode_steps is None
                else float(self.stock_fallback_share_decode_steps)
            ),
            "kv_quantization_active": bool(self.kv_quantization_active),
            "quantized_entries": int(self.quantized_entries),
            "fallback_reason_counts": dict(self.fallback_reason_counts),
        }


@dataclass
class RunSummary:
    label: str
    source: str
    model_path: str | None
    mode: str | None
    butterfly_decode_backend: str | None
    kv_quantization: dict[str, Any]
    butterfly: dict[str, Any]
    created_at: str | None
    aggregated_rows: list[AggregatedRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source,
            "model_path": self.model_path,
            "mode": self.mode,
            "butterfly_decode_backend": self.butterfly_decode_backend,
            "kv_quantization": self.kv_quantization,
            "butterfly": self.butterfly,
            "created_at": self.created_at,
            "aggregated_rows": [row.to_dict() for row in self.aggregated_rows],
        }


def _aggregate_single_turn_rows(
    rows: Sequence[dict[str, Any]],
    *,
    metric_aggregate: str,
    peak_aggregate: str,
    seq_filter: set[int] | None,
) -> list[AggregatedRow]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seq_len = int(row["seq_len"])
        if seq_filter is not None and seq_len not in seq_filter:
            continue
        grouped.setdefault(seq_len, []).append(row)

    out: list[AggregatedRow] = []
    for seq_len in sorted(grouped):
        subset = grouped[seq_len]
        fallback_reasons: Counter[str] = Counter()
        quantized_entries_max = 0
        kv_active = False
        for row in subset:
            fallback_reasons.update(row.get("stock_fallback_reason_counts", {}) or {})
            kv_row = row.get("kv_quantization") or {}
            quantized_entries_max = max(
                quantized_entries_max,
                int(kv_row.get("quantized_entries") or 0),
            )
            kv_active = kv_active or bool(kv_row.get("active"))

        peak_values = [
            float(row["peak_memory_bytes"])
            for row in subset
            if row.get("peak_memory_bytes") is not None
        ]
        decode_values = [
            float(row["decode_tok_s"])
            for row in subset
            if row.get("decode_tok_s") is not None
        ]
        out.append(
            AggregatedRow(
                seq_len=seq_len,
                repeats=len(subset),
                decode_len=int(subset[0]["decode_len"]),
                prefill_sec=_aggregate_numeric(
                    [float(row["prefill_sec"]) for row in subset],
                    metric_aggregate,
                ),
                ttft_sec=_aggregate_numeric(
                    [float(row["ttft_sec"]) for row in subset if row.get("ttft_sec") is not None],
                    metric_aggregate,
                ),
                decode_tok_s=(
                    _aggregate_numeric(decode_values, metric_aggregate)
                    if decode_values
                    else None
                ),
                e2e_sec=_aggregate_numeric(
                    [float(row["e2e_sec"]) for row in subset],
                    metric_aggregate,
                ),
                peak_memory_bytes=(
                    _aggregate_numeric(peak_values, peak_aggregate) if peak_values else None
                ),
                stock_fallback_share_run=(
                    _aggregate_numeric(
                        [
                            float(row["stock_fallback_share_run"])
                            for row in subset
                            if row.get("stock_fallback_share_run") is not None
                        ],
                        metric_aggregate,
                    )
                    if any(row.get("stock_fallback_share_run") is not None for row in subset)
                    else None
                ),
                stock_fallback_share_decode_steps=(
                    _aggregate_numeric(
                        [
                            float(row["stock_fallback_share_decode_steps"])
                            for row in subset
                            if row.get("stock_fallback_share_decode_steps") is not None
                        ],
                        metric_aggregate,
                    )
                    if any(
                        row.get("stock_fallback_share_decode_steps") is not None for row in subset
                    )
                    else None
                ),
                kv_quantization_active=kv_active,
                quantized_entries=quantized_entries_max,
                fallback_reason_counts=dict(fallback_reasons),
            )
        )
    return out


def _load_run_summary(
    label: str,
    results_path: Path,
    *,
    metric_aggregate: str,
    peak_aggregate: str,
    seq_filter: set[int] | None,
) -> RunSummary:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("single_turn") or []
    if not rows:
        raise ValueError(f"no single_turn rows found in {results_path}")
    butterfly_config = dict(payload.get("butterfly_config") or {})
    legacy_wayfinder_config = dict(payload.get("wayfinder_config") or {})
    return RunSummary(
        label=label,
        source=str(results_path),
        model_path=payload.get("model_path"),
        mode=payload.get("mode"),
        butterfly_decode_backend=(
            payload.get("butterfly_decode_backend")
            or payload.get("wayfinder_decode_backend")
            or butterfly_config.get("butterfly_decode_backend")
            or butterfly_config.get("decode_backend")
            or legacy_wayfinder_config.get("wayfinder_decode_backend")
            or legacy_wayfinder_config.get("decode_backend")
        ),
        kv_quantization=dict(payload.get("kv_quantization") or {}),
        butterfly=dict(payload.get("butterfly") or payload.get("wayfinder") or {}),
        created_at=payload.get("created_at"),
        aggregated_rows=_aggregate_single_turn_rows(
            rows,
            metric_aggregate=metric_aggregate,
            peak_aggregate=peak_aggregate,
            seq_filter=seq_filter,
        ),
    )


def _render_markdown(runs: Sequence[RunSummary], *, metric_aggregate: str, peak_aggregate: str) -> str:
    lines = [
        "# Qwen MLX Butterfly Benchmark Summary",
        "",
        f"- metric aggregate: `{metric_aggregate}`",
        f"- peak aggregate: `{peak_aggregate}`",
        "",
        "| Run | Seq len | Repeats | Prefill s | TTFT s | Decode tok/s | E2E s | Peak bytes | Fallback run share | Decode fallback steps | KV active | Quantized entries | Fallback reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |",
    ]
    for run in runs:
        for row in run.aggregated_rows:
            peak_value = (
                ""
                if row.peak_memory_bytes is None
                else str(int(round(float(row.peak_memory_bytes))))
            )
            lines.append(
                "| "
                f"{run.label} | "
                f"{row.seq_len} | "
                f"{row.repeats} | "
                f"{_format_float(row.prefill_sec)} | "
                f"{_format_float(row.ttft_sec)} | "
                f"{_format_float(row.decode_tok_s, digits=2)} | "
                f"{_format_float(row.e2e_sec)} | "
                f"{peak_value} | "
                f"{_format_float(row.stock_fallback_share_run, digits=3)} | "
                f"{_format_float(row.stock_fallback_share_decode_steps, digits=3)} | "
                f"{'yes' if row.kv_quantization_active else 'no'} | "
                f"{row.quantized_entries} | "
                f"{_format_reason_counts(Counter(row.fallback_reason_counts))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _render_csv_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for row in run.aggregated_rows:
            rows.append(
                {
                    "run": run.label,
                    "source": run.source,
                    "model_path": run.model_path,
                    "mode": run.mode,
                    "butterfly_decode_backend": run.butterfly_decode_backend,
                    **row.to_dict(),
                    "fallback_reasons": _format_reason_counts(Counter(row.fallback_reason_counts)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Qwen MLX benchmark results.json files into markdown, CSV, or JSON."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form LABEL=PATH, where PATH is a results.json file or its parent directory.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="*",
        default=None,
        help="Optional seq_len filter.",
    )
    parser.add_argument(
        "--metric-aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation mode for latency and throughput metrics.",
    )
    parser.add_argument(
        "--peak-aggregate",
        choices=["mean", "max"],
        default="mean",
        help="Aggregation mode for peak_memory_bytes.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="Render format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    args = parser.parse_args()

    seq_filter = set(int(x) for x in args.seq_lens) if args.seq_lens else None
    runs = [
        _load_run_summary(
            label,
            results_path,
            metric_aggregate=args.metric_aggregate,
            peak_aggregate=args.peak_aggregate,
            seq_filter=seq_filter,
        )
        for label, results_path in (_parse_run_spec(spec) for spec in args.run)
    ]

    if args.format == "markdown":
        rendered = _render_markdown(
            runs,
            metric_aggregate=args.metric_aggregate,
            peak_aggregate=args.peak_aggregate,
        )
    elif args.format == "json":
        rendered = json.dumps(
            {
                "metric_aggregate": args.metric_aggregate,
                "peak_aggregate": args.peak_aggregate,
                "runs": [run.to_dict() for run in runs],
            },
            indent=2,
        )
    else:
        csv_rows = _render_csv_rows(runs)
        fieldnames = [
            "run",
            "source",
            "model_path",
            "mode",
            "butterfly_decode_backend",
            "seq_len",
            "repeats",
            "decode_len",
            "prefill_sec",
            "ttft_sec",
            "decode_tok_s",
            "e2e_sec",
            "peak_memory_bytes",
            "stock_fallback_share_run",
            "stock_fallback_share_decode_steps",
            "kv_quantization_active",
            "quantized_entries",
            "fallback_reasons",
        ]
        out_stream = sys.stdout
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out_stream = args.output.open("w", encoding="utf-8", newline="")
        try:
            writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        finally:
            if out_stream is not sys.stdout:
                out_stream.close()
        return

    if args.output is None:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
