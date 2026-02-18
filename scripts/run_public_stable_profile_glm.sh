#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

run_id="EXP-$(date -u +%Y%m%dT%H%M%SZ)-STABLE-PROFILE"
out_root="benchmarks/mlx/first_release"
model_path="mlx-community/GLM-4.7-Flash-4bit"
seq_len="8192"
decode_len="32"
repeats="1"

usage() {
  cat <<'USAGE'
Usage: run_public_stable_profile_glm.sh [options]

Runs the public stable GLM profile sequentially (dense then wayfinder)
with conservative defaults and writes summary artifacts.

Options:
  --run-id ID             Run identifier (default: EXP-<utc>-STABLE-PROFILE)
  --out-root PATH         Output root (default: benchmarks/mlx/first_release)
  --model-path MODEL      Model path (default: mlx-community/GLM-4.7-Flash-4bit)
  --seq-len N             Sequence length (default: 8192)
  --decode-len N          Decode length (default: 32)
  --repeats N             Repeats (default: 1)
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-id" >&2; exit 2; }
      run_id="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --out-root" >&2; exit 2; }
      out_root="$2"
      shift 2
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --model-path" >&2; exit 2; }
      model_path="$2"
      shift 2
      ;;
    --seq-len)
      [[ $# -ge 2 ]] || { echo "Missing value for --seq-len" >&2; exit 2; }
      seq_len="$2"
      shift 2
      ;;
    --decode-len)
      [[ $# -ge 2 ]] || { echo "Missing value for --decode-len" >&2; exit 2; }
      decode_len="$2"
      shift 2
      ;;
    --repeats)
      [[ $# -ge 2 ]] || { echo "Missing value for --repeats" >&2; exit 2; }
      repeats="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$out_root" = /* ]]; then
  out_root_abs="$out_root"
else
  out_root_abs="$repo_root/$out_root"
fi

run_root="$out_root_abs/$run_id"
dense_out="$run_root/dense"
wayfinder_out="$run_root/wayfinder"

if [[ -e "$run_root" ]]; then
  echo "Refusing to reuse existing run root: $run_root" >&2
  echo "Choose a new --run-id or --out-root." >&2
  exit 2
fi

mkdir -p "$dense_out" "$wayfinder_out"

cd "$repo_root"

echo "[stable-profile] run_id=$run_id"
echo "[stable-profile] run_root=$run_root"
echo "[stable-profile] step=dense"
python3 scripts/bench_glm_consumer_mlx.py \
  --model-path "$model_path" \
  --mode dense \
  --seq-lens "$seq_len" \
  --decode-len "$decode_len" \
  --repeats "$repeats" \
  --skip-multi-turn \
  --skip-quality \
  --out-dir "$dense_out"

echo "[stable-profile] step=wayfinder"
python3 scripts/bench_glm_consumer_mlx.py \
  --model-path "$model_path" \
  --mode wayfinder \
  --seq-lens "$seq_len" \
  --decode-len "$decode_len" \
  --repeats "$repeats" \
  --skip-multi-turn \
  --skip-quality \
  --out-dir "$wayfinder_out"

dense_results="$dense_out/results.json"
wayfinder_results="$wayfinder_out/results.json"
summary_json="$run_root/stable_profile_summary.json"
summary_md="$run_root/stable_profile_summary.md"

python3 - "$dense_results" "$wayfinder_results" "$summary_json" "$summary_md" "$run_id" "$model_path" "$seq_len" "$decode_len" "$repeats" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def pick_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("single_turn")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing single_turn rows")
    row = rows[-1]
    if not isinstance(row, dict):
        raise ValueError("Malformed single_turn row")
    return row


def fnum(v: Any) -> float:
    if v is None:
        return 0.0
    return float(v)


def opt_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def opt_delta_pct(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    bf = float(b)
    if bf == 0.0:
        return None
    return 100.0 * (float(a) - bf) / bf


def as_metric(row: Dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    if val is None:
        return None
    return float(val)


def main() -> None:
    dense_path = Path(sys.argv[1])
    wayfinder_path = Path(sys.argv[2])
    summary_json = Path(sys.argv[3])
    summary_md = Path(sys.argv[4])
    run_id = sys.argv[5]
    model_path = sys.argv[6]
    seq_len = int(sys.argv[7])
    decode_len = int(sys.argv[8])
    repeats = int(sys.argv[9])

    dense_payload = load_payload(dense_path)
    wayfinder_payload = load_payload(wayfinder_path)
    dense_row = pick_row(dense_payload)
    wayfinder_row = pick_row(wayfinder_payload)

    metrics = [
        "e2e_sec",
        "prefill_sec",
        "decode_sec",
        "decode_tok_s",
        "peak_memory_bytes",
        "ttft_sec",
        "itl_p95_sec",
    ]

    delta_abs: Dict[str, float | None] = {}
    delta_pct: Dict[str, float | None] = {}
    for key in metrics:
        wa = as_metric(wayfinder_row, key)
        de = as_metric(dense_row, key)
        delta_abs[key] = opt_delta(wa, de)
        delta_pct[key] = opt_delta_pct(wa, de)

    dense_mem = as_metric(dense_row, "peak_memory_bytes")
    wayfinder_mem = as_metric(wayfinder_row, "peak_memory_bytes")
    mem_reduction_pct = None
    if dense_mem is not None and wayfinder_mem is not None and dense_mem != 0.0:
        mem_reduction_pct = 100.0 * (1.0 - (wayfinder_mem / dense_mem))

    summary = {
        "id": run_id,
        "model_path": model_path,
        "seq_len": seq_len,
        "decode_len": decode_len,
        "repeats": repeats,
        "artifacts": {
            "dense_results": str(dense_path),
            "wayfinder_results": str(wayfinder_path),
            "summary_md": str(summary_md),
        },
        "dense": {
            "e2e_sec": as_metric(dense_row, "e2e_sec"),
            "prefill_sec": as_metric(dense_row, "prefill_sec"),
            "decode_sec": as_metric(dense_row, "decode_sec"),
            "decode_tok_s": as_metric(dense_row, "decode_tok_s"),
            "peak_memory_bytes": as_metric(dense_row, "peak_memory_bytes"),
            "ttft_sec": as_metric(dense_row, "ttft_sec"),
            "itl_p95_sec": as_metric(dense_row, "itl_p95_sec"),
            "path_counts": dense_row.get("path_counts"),
            "dense_fallback_reason_counts": dense_row.get("dense_fallback_reason_counts"),
            "dense_fallback_share_run": dense_row.get("dense_fallback_share_run"),
        },
        "wayfinder": {
            "e2e_sec": as_metric(wayfinder_row, "e2e_sec"),
            "prefill_sec": as_metric(wayfinder_row, "prefill_sec"),
            "decode_sec": as_metric(wayfinder_row, "decode_sec"),
            "decode_tok_s": as_metric(wayfinder_row, "decode_tok_s"),
            "peak_memory_bytes": as_metric(wayfinder_row, "peak_memory_bytes"),
            "ttft_sec": as_metric(wayfinder_row, "ttft_sec"),
            "itl_p95_sec": as_metric(wayfinder_row, "itl_p95_sec"),
            "path_counts": wayfinder_row.get("path_counts"),
            "dense_fallback_reason_counts": wayfinder_row.get("dense_fallback_reason_counts"),
            "dense_fallback_share_run": wayfinder_row.get("dense_fallback_share_run"),
        },
        "delta_wayfinder_vs_dense_abs": delta_abs,
        "delta_wayfinder_vs_dense_pct": delta_pct,
        "memory_reduction_pct_convention": mem_reduction_pct,
    }

    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append(f"# Stable Public Profile Summary ({run_id})")
    md.append("")
    md.append(f"- model_path: `{model_path}`")
    md.append(f"- seq_len: `{seq_len}`")
    md.append(f"- decode_len: `{decode_len}`")
    md.append(f"- repeats: `{repeats}`")
    md.append("")
    md.append("## Metrics")
    md.append("")
    md.append("| metric | dense | wayfinder | delta_abs (wayfinder-dense) | delta_pct |")
    md.append("|---|---:|---:|---:|---:|")
    for key in metrics:
        de = as_metric(dense_row, key)
        wa = as_metric(wayfinder_row, key)
        da = delta_abs[key]
        dp = delta_pct[key]
        de_s = "n/a" if de is None else f"{de:.6f}"
        wa_s = "n/a" if wa is None else f"{wa:.6f}"
        da_s = "n/a" if da is None else f"{da:.6f}"
        dp_s = "n/a" if dp is None else f"{dp:.6f}%"
        md.append(f"| {key} | {de_s} | {wa_s} | {da_s} | {dp_s} |")
    md.append("")
    md.append("## Memory Convention")
    md.append("")
    if mem_reduction_pct is None:
        md.append("- `memory_reduction_pct_convention`: n/a")
    else:
        md.append(
            f"- `memory_reduction_pct_convention` = `100 * (1 - wayfinder/dense)` = `{mem_reduction_pct:.6f}%`"
        )
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append(f"- dense: `{dense_path}`")
    md.append(f"- wayfinder: `{wayfinder_path}`")
    md.append(f"- summary_json: `{summary_json}`")

    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
PY

echo "[stable-profile] status=pass"
echo "[stable-profile] dense_results=$dense_results"
echo "[stable-profile] wayfinder_results=$wayfinder_results"
echo "[stable-profile] summary_json=$summary_json"
echo "[stable-profile] summary_md=$summary_md"
