#!/usr/bin/env python3
"""Benchmark dense vs Butterfly prefill on Qwen3.5 (CUDA).

Measures prefill-only forward pass latency at various sequence lengths,
comparing stock dense attention against configurable Butterfly/HCSA attention
on the full_attention layers. The benchmark can run either the true sparse
Hamiltonian graph path or the permute-window surrogate.

Usage:
    python scripts/bench_qwen35_cuda_wayfinder.py \
        --model-path ~/HF_Models/Qwen3.5-9B \
        --seq-lens 64 128 256 512 1024 2048 \
        --warmup 2 --repeats 5
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _make_dummy_input(tokenizer, seq_len: int, device: torch.device) -> dict:
    """Create a dummy input of the desired length by repeating a seed sentence."""
    seed = "The theory of sparse attention in transformer models is important "
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = (seq_len // len(seed_ids)) + 2
    ids = (seed_ids * repeats)[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids}


def _sync_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _clear_cuda_memory() -> None:
    """Best-effort release of reusable CUDA allocations between stages."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    _sync_cuda()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        try:
            ipc_collect()
        except Exception:
            pass
    _sync_cuda()


def _gpu_mem_gb() -> dict[str, float]:
    """Return current/peak/free GPU memory in GB."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    props = torch.cuda.get_device_properties(0)
    total_bytes = getattr(props, "total_memory", None)
    if total_bytes is None:
        total_bytes = getattr(props, "total_mem")
    total = total_bytes / (1024**3)
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
        "total_gb": round(total, 2),
    }


def _unsupported_flex_message(diag: Dict[str, Any]) -> str:
    cap = diag.get("capability")
    cap_str = "unknown" if cap is None else f"{cap[0]}.{cap[1]}"
    return (
        "Requested `--engine flex` on a CUDA device that this PyTorch build does not "
        f"explicitly support (device capability={cap_str}; torch arch list={diag.get('supported_arch_list')}). "
        "On this machine, unsupported flex runs have correlated with NVIDIA `Xid 13` "
        "illegal-instruction faults during Butterfly phases. Use `--engine batched`, "
        "or pass `--allow-unsupported-arch` if you intentionally want to force flex anyway."
    )


_UNSUPPORTED_FLEX_VALIDATED_SEQ_CAPS = {
    "permute": 8192,
    "block_sparse": 4096,
}


def _unsupported_flex_longrun_message(
    *,
    path: str,
    requested_seq_len: Optional[int],
    validated_cap: int,
) -> str:
    requested = "unbounded" if requested_seq_len is None else f"{int(requested_seq_len):,}"
    return (
        "Refusing unsafe-longrun unsupported-arch flex run. "
        f"`--path {path}` on this machine is only validated through {int(validated_cap):,} tokens, "
        f"but this request reaches {requested}. "
        "Use a supported CUDA arch, switch to a non-flex path, reduce `--seq-lens`, "
        "or pass `--unsafe-longrun` if you intentionally want to risk "
        "driver faults or host instability."
    )


def _guard_unsupported_flex_longrun(
    *,
    path: str,
    engine: str,
    arch_diag: Dict[str, Any],
    requested_seq_len: Optional[int],
    allow_unsafe_longrun: bool,
) -> None:
    if not torch.cuda.is_available():
        return
    if arch_diag.get("exact_match"):
        return
    if engine != "flex":
        return
    validated_cap = _UNSUPPORTED_FLEX_VALIDATED_SEQ_CAPS.get(path)
    if validated_cap is None:
        return
    if requested_seq_len is not None and int(requested_seq_len) <= int(validated_cap):
        return
    if allow_unsafe_longrun:
        _log(
            "WARNING: "
            + _unsupported_flex_longrun_message(
                path=path,
                requested_seq_len=requested_seq_len,
                validated_cap=int(validated_cap),
            )
        )
        return
    raise SystemExit(
        _unsupported_flex_longrun_message(
            path=path,
            requested_seq_len=requested_seq_len,
            validated_cap=int(validated_cap),
        )
    )


def _append_ndjson_row(path: Path, row: Dict[str, Any], *, truncate: bool = False) -> None:
    mode = "w" if truncate else "a"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_ndjson_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse {path}:{lineno}: {exc}") from exc
    return rows


def _quantization_config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {"repr": repr(config)}


def _checkpoint_native_quantization(config: Any) -> Optional[Dict[str, Any]]:
    native = _quantization_config_to_dict(getattr(config, "quantization_config", None))
    if native is not None:
        return native
    text_config = getattr(config, "text_config", None)
    return _quantization_config_to_dict(getattr(text_config, "quantization_config", None))


def _checkpoint_text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _choose_auto_model_loader(transformers_module: Any, config: Any) -> Any:
    architectures = tuple(getattr(config, "architectures", []) or ())
    if any("ConditionalGeneration" in arch for arch in architectures):
        loader = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if loader is None:
            raise RuntimeError(
                "This checkpoint requires `AutoModelForImageTextToText`, "
                "but the active transformers install does not expose it."
            )
        return loader
    return transformers_module.AutoModelForCausalLM


def _row_key(row: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    row_type = row.get("type")
    if row_type == "experiment_meta":
        return ("experiment_meta",)
    if row_type == "bench":
        label = row.get("label")
        if label == "wayfinder":
            label = "butterfly"
        return ("bench", label, row.get("seq_len"))
    if row_type == "divergence":
        return ("divergence", row.get("seq_len"))
    return None


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another Qwen CUDA benchmark appears to be active (lock file: {lock_path})."
            ) from exc
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def bench_prefill(
    model: torch.nn.Module,
    inputs: dict,
    *,
    warmup: int = 2,
    repeats: int = 5,
    label: str = "",
) -> Dict[str, Any]:
    """Run prefill-only forward passes and return timing stats."""
    seq_len = inputs["input_ids"].shape[1]

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _sync_cuda()
            outputs = model(**inputs, use_cache=False)
            _sync_cuda()
        del outputs

    # Timed runs
    times_ms: List[float] = []
    peak_mem_gb: float = 0.0
    for _ in range(repeats):
        _sync_cuda()
        _clear_cuda_memory()
        torch.cuda.reset_peak_memory_stats()
        _sync_cuda()

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
        _sync_cuda()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)
        run_peak = torch.cuda.max_memory_allocated() / (1024**3)
        peak_mem_gb = max(peak_mem_gb, run_peak)
        del outputs

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    prefill_tok_per_sec = round(seq_len / (median_ms / 1000.0), 1)

    return {
        "label": label,
        "seq_len": seq_len,
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": round(median_ms, 2),
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "peak_mem_gb": round(peak_mem_gb, 2),
        "all_ms": [round(t, 2) for t in times_ms],
        "gpu_mem": _gpu_mem_gb(),
    }


def collect_wayfinder_profiles(model: torch.nn.Module) -> List[Dict[str, Any]]:
    from bna.integrations.qwen_torch import iter_qwen_wayfinder_layers

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    profiles = []
    for layer in iter_qwen_wayfinder_layers(model):
        layer.ensure_last_graph_metrics()
        snapshot = getattr(layer, "snapshot_last_profile", None)
        p = snapshot(sync=False) if callable(snapshot) else dict(layer.last_profile)
        profiles.append(
            {
                "layer_idx": p.get("layer_idx"),
                "mode": p.get("mode"),
                "reason": p.get("reason"),
                "elapsed_ms": p.get("elapsed_ms"),
                "graph_build_ms": p.get("graph_build_ms"),
                "attn_kernel_ms": p.get("attn_kernel_ms"),
                "attn_kernel_ms_host": p.get("attn_kernel_ms_host"),
                "path": p.get("path"),
                "engine": p.get("engine"),
                "strategy": p.get("strategy"),
                "graph_source": p.get("graph_source"),
                "graph_cache_hit": p.get("graph_cache_hit"),
                "graph_metrics": p.get("graph_metrics"),
                "graph_cache_entries": len(getattr(layer, "_graph_cache", {})),
                "sparse_chunk_mode": p.get("sparse_chunk_mode"),
                "sparse_compute_dtype": p.get("sparse_compute_dtype"),
                "sparse_query_input_dtype": p.get("sparse_query_input_dtype"),
                "sparse_key_input_dtype": p.get("sparse_key_input_dtype"),
                "sparse_value_input_dtype": p.get("sparse_value_input_dtype"),
                "sparse_query_chunk_size": p.get("sparse_query_chunk_size"),
                "sparse_kv_head_chunk_size": p.get("sparse_kv_head_chunk_size"),
                "sparse_degree_chunk_size": p.get("sparse_degree_chunk_size"),
                "sparse_num_query_chunks": p.get("sparse_num_query_chunks"),
                "sparse_num_head_blocks": p.get("sparse_num_head_blocks"),
                "sparse_num_degree_blocks": p.get("sparse_num_degree_blocks"),
                "sparse_streamed_degree": p.get("sparse_streamed_degree"),
                "sparse_chunk_budget_exceeded": p.get("sparse_chunk_budget_exceeded"),
                "sparse_estimated_temp_mib": p.get("sparse_estimated_temp_mib"),
                "sparse_contraction_backend": p.get("sparse_contraction_backend"),
                "sparse_contraction_cuda_ms": p.get("sparse_contraction_cuda_ms"),
                "sparse_trace_path": p.get("sparse_trace_path"),
                "sparse_trace_error": p.get("sparse_trace_error"),
                "block_sparse_backend": p.get("block_sparse_backend"),
                "block_sparse_block_size": p.get("block_sparse_block_size"),
                "block_sparse_num_blocks": p.get("block_sparse_num_blocks"),
                "block_sparse_neighbor_blocks": p.get("block_sparse_neighbor_blocks"),
                "block_sparse_landmark_blocks": p.get("block_sparse_landmark_blocks"),
                "block_sparse_num_cycles": p.get("block_sparse_num_cycles"),
                "block_sparse_cuda_ms": p.get("block_sparse_cuda_ms"),
                "block_sparse_topology": p.get("block_sparse_topology"),
                "block_sparse_sink_blocks": p.get("block_sparse_sink_blocks"),
                "block_sparse_stage": p.get("block_sparse_stage"),
                "block_sparse_stage_count": p.get("block_sparse_stage_count"),
                "block_local_window_blocks": p.get("block_local_window_blocks"),
                "block_partner_count": p.get("block_partner_count"),
                "block_partner_rule": p.get("block_partner_rule"),
            }
        )
    return profiles


def clear_wayfinder_graph_caches(model: torch.nn.Module) -> Dict[str, int]:
    from bna.integrations.qwen_torch import (
        clear_shared_qwen_wayfinder_graph_cache,
        iter_qwen_wayfinder_layers,
    )

    layers_cleared = 0
    entries_cleared = 0
    for layer in iter_qwen_wayfinder_layers(model):
        cache = getattr(layer, "_graph_cache", None)
        if isinstance(cache, dict):
            layers_cleared += 1
            entries_cleared += len(cache)
            cache.clear()
        if hasattr(layer, "_last_graph_cache"):
            layer._last_graph_cache = None
    shared_entries_cleared = clear_shared_qwen_wayfinder_graph_cache()
    if entries_cleared or shared_entries_cleared:
        _clear_cuda_memory()
    return {
        "layers": layers_cleared,
        "entries": entries_cleared,
        "shared_entries": shared_entries_cleared,
    }


def _get_forward_module(model: torch.nn.Module, target: str) -> torch.nn.Module:
    if target == "causal_lm":
        return model
    if target == "backbone":
        language_backbone = getattr(getattr(model, "model", None), "language_model", None)
        if language_backbone is None:
            language_backbone = getattr(model, "language_model", None)
        if language_backbone is not None:
            return language_backbone
        backbone = getattr(model, "model", None)
        if backbone is None:
            prefix = getattr(model, "base_model_prefix", "")
            backbone = getattr(model, prefix, None) if prefix else None
        if backbone is None:
            raise ValueError("Could not resolve backbone module for backbone-only benchmarking.")
        return backbone
    raise ValueError(f"Unsupported forward target: {target}")


def _resolve_model_device_map(*, forward_target: str) -> Any:
    if torch.cuda.is_available() and forward_target == "backbone":
        # Backbone-only timing must keep dense and Butterfly on the same real device.
        return 0
    return "auto"


def _jsonable_device_map(device_map: Any) -> Any:
    if isinstance(device_map, dict):
        return {str(key): str(value) for key, value in device_map.items()}
    if isinstance(device_map, torch.device):
        return str(device_map)
    return device_map


def _summarize_module_residency(module: torch.nn.Module) -> Dict[str, Any]:
    parameter_counts = Counter(str(param.device) for param in module.parameters())
    buffer_counts = Counter(str(buf.device) for buf in module.buffers())
    module_counts = Counter(parameter_counts)
    module_counts.update(buffer_counts)
    real_devices = sorted(device for device in module_counts if device not in {"cpu", "meta"})
    benchmark_device = real_devices[0] if len(real_devices) == 1 else None
    return {
        "benchmark_parameter_device_counts": dict(sorted(parameter_counts.items())),
        "benchmark_buffer_device_counts": dict(sorted(buffer_counts.items())),
        "benchmark_module_device_counts": dict(sorted(module_counts.items())),
        "benchmark_real_devices": real_devices,
        "benchmark_device": benchmark_device,
    }


def _validate_backbone_module_residency(
    module: torch.nn.Module,
    *,
    label: str,
    expected_device: Optional[str] = None,
) -> Dict[str, Any]:
    residency = _summarize_module_residency(module)
    if not torch.cuda.is_available():
        return residency
    module_counts = residency["benchmark_module_device_counts"]
    disallowed = [
        device_name for device_name in ("cpu", "meta") if module_counts.get(device_name, 0) > 0
    ]
    real_devices = residency["benchmark_real_devices"]
    if disallowed or len(real_devices) != 1:
        raise RuntimeError(
            f"{label} backbone benchmark module residency is invalid: "
            f"module_device_counts={module_counts}, real_devices={real_devices}."
        )
    if expected_device is not None and real_devices[0] != expected_device:
        raise RuntimeError(
            f"{label} backbone benchmark module resolved to {real_devices[0]}, "
            f"expected {expected_device}."
        )
    return residency


def _hf_device_map_head(model: torch.nn.Module, *, limit: int = 8) -> Optional[List[List[str]]]:
    mapping = getattr(model, "hf_device_map", None)
    if not isinstance(mapping, dict):
        return None
    head: List[List[str]] = []
    for idx, (name, device) in enumerate(mapping.items()):
        if idx >= limit:
            break
        head.append([str(name), str(device)])
    return head


def _estimated_logits_buffer_gb(*, seq_len: int, vocab_size: int) -> float:
    # Be conservative: HF causal LM heads often materialize logits in fp32.
    return float(seq_len * vocab_size * 4) / float(1024**3)


def _validate_run_shape(args: argparse.Namespace, tokenizer: Any) -> None:
    if args.forward_target != "causal_lm":
        return
    max_seq_len = max(int(t) for t in args.seq_lens)
    if max_seq_len <= 8192 or args.allow_large_logits:
        return
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    est_logits_gb = _estimated_logits_buffer_gb(seq_len=max_seq_len, vocab_size=int(vocab_size))
    raise SystemExit(
        "Refusing long `causal_lm` prefill benchmark without explicit opt-in. "
        f"T={max_seq_len:,} can materialize roughly {est_logits_gb:.1f} GiB of logits alone. "
        "Use `--forward-target backbone` for long-context prefill measurements, "
        "or pass `--allow-large-logits` if you intentionally want the unsafe full-LM path."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Bench dense vs Butterfly prefill on Qwen3.5 CUDA")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--quantize",
        type=str,
        default="none",
        choices=["none", "fp8-weight-only", "fp8-dynamic"],
        help=(
            "Post-training quantization: none (BF16), "
            "fp8-weight-only (FP8 weights, BF16 compute — memory savings only), "
            "fp8-dynamic (FP8 weights + activations — memory + speed via tensor cores)"
        ),
    )
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "regular_partition", "greedy", "online_insertion"],
        help="Hamiltonian strategy used to build the Butterfly graph.",
    )
    p.add_argument(
        "--path",
        type=str,
        default="sparse",
        choices=["permute", "sparse", "compressed_butterfly", "block_sparse"],
        help="Butterfly path: `sparse` uses the full HCSA graph; "
        "`permute` uses the cycle-window surrogate; "
        "`compressed_butterfly` uses exact local tokens plus compressed routed block summaries. "
        "`block_sparse` remains a legacy raw-block alias.",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "flex", "batched", "legacy", "sdpa", "triton"],
        help="Butterfly engine: auto, flex, batched, legacy, sdpa, or triton. "
        "sdpa uses F.scaled_dot_product_attention for block_sparse (no torch.compile). "
        "triton uses a fused Triton block-sparse kernel for block_sparse.",
    )
    p.add_argument(
        "--block-size", type=int, default=128, help="Block size for the `block_sparse` path."
    )
    # Block-sparse topology is always "butterfly" now; the old "wayfinder"/"hamiltonian"
    # option has been archived.  We keep a hidden arg for backwards compat
    # with saved command lines but ignore non-butterfly values.
    p.add_argument("--block-sparse-topology", type=str, default="butterfly", help=argparse.SUPPRESS)
    p.add_argument(
        "--block-local-window-blocks",
        type=int,
        default=1,
        help="Number of causal predecessor blocks kept per query block for Butterfly block topology.",
    )
    p.add_argument(
        "--block-partner-count",
        type=int,
        default=1,
        help="Number of deterministic partner blocks kept per query block for Butterfly block topology.",
    )
    p.add_argument(
        "--block-sink-blocks",
        type=int,
        default=1,
        help="Number of fixed sink blocks exposed to every later block for Butterfly block topology.",
    )
    p.add_argument(
        "--block-partner-rule",
        type=str,
        default="xor",
        choices=["xor", "bit_reversal", "benes", "causal_shift"],
        help="Deterministic partner rule used by the Butterfly block topology.",
    )
    p.add_argument(
        "--block-chunk-size",
        type=int,
        default=0,
        help="SDPA block chunk size (0 = all at once). "
        "Reduces peak memory at the cost of more kernel launches.",
    )
    p.add_argument(
        "--compressed-local-window-tokens",
        type=int,
        default=128,
        help="Exact raw-token sliding window used by compressed Butterfly (default: 128).",
    )
    p.add_argument(
        "--block-compression",
        type=str,
        default="none",
        choices=["none", "mean"],
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Seed forwarded to the Hamiltonian graph strategy."
    )
    p.add_argument(
        "--compute-graph-metrics",
        action="store_true",
        help="Compute graph diagnostics (degree, shortcut rate, reachability) "
        "and attach them to butterfly_profiles.",
    )
    p.add_argument(
        "--sparse-query-chunk-size",
        type=int,
        default=0,
        help="Butterfly/HCSA query chunk size for the sparse path. `0` selects auto chunking.",
    )
    p.add_argument(
        "--sparse-kv-head-chunk-size",
        type=int,
        default=0,
        help="Butterfly/HCSA KV-head block size for the sparse path. `0` selects auto chunking.",
    )
    p.add_argument(
        "--sparse-degree-chunk-size",
        type=int,
        default=0,
        help="Butterfly/HCSA streamed degree block size for the sparse path. `0` selects auto chunking.",
    )
    p.add_argument(
        "--sparse-chunk-temp-budget-mib",
        type=float,
        default=160.0,
        help="Temporary buffer budget in MiB used by sparse auto chunking.",
    )
    p.add_argument(
        "--sparse-compute-dtype",
        type=str,
        default="auto",
        choices=["auto", "model", "float32"],
        help="Butterfly/HCSA sparse compute dtype: auto, model, or float32.",
    )
    p.add_argument(
        "--dump-sparse-trace-dir",
        type=str,
        default=None,
        help="Optional directory for replayable sparse trace payloads. "
        "When set, each selected Butterfly sparse layer dumps up to "
        "`--dump-sparse-trace-max-per-layer` traces.",
    )
    p.add_argument(
        "--dump-sparse-trace-max-per-layer",
        type=int,
        default=0,
        help="Maximum sparse trace files to dump per Butterfly layer. `0` disables dumping.",
    )
    p.add_argument(
        "--dump-sparse-trace-layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of Butterfly layer indices allowed to dump sparse traces.",
    )
    p.add_argument(
        "--forward-target",
        type=str,
        default="backbone",
        choices=["causal_lm", "backbone"],
        help="Benchmark the full LM head path or just the transformer backbone. "
        "Backbone avoids full-sequence logits and is the recommended long-context path.",
    )
    p.add_argument(
        "--allow-large-logits",
        action="store_true",
        help="Allow long `causal_lm` runs that materialize full-sequence logits. Unsafe above 8k.",
    )
    p.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["butterfly", "dense", "divergence"],
        choices=["dense", "butterfly", "wayfinder", "divergence"],
        help="Benchmark phases to execute. `wayfinder` remains a legacy alias for `butterfly`.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file and skip rows already recorded.",
    )
    p.add_argument(
        "--skip-divergence",
        action="store_true",
        help="Skip the divergence pass even if divergence is present in --phases.",
    )
    p.add_argument(
        "--lock-file",
        type=str,
        default=str(
            REPO_ROOT
            / "benchmarks"
            / "cuda"
            / "qwen35_butterfly"
            / ".bench_qwen35_cuda_butterfly.lock"
        ),
        help="Exclusive lock file used to prevent overlapping benchmark runs.",
    )
    p.add_argument(
        "--allow-unsupported-arch",
        action="store_true",
        help="Allow `--engine flex` even when the current CUDA capability is not "
        "explicitly supported by this PyTorch build.",
    )
    p.add_argument(
        "--allow-unsafe-unsupported-flex-longrun",
        "--unsafe-longrun",
        dest="allow_unsafe_unsupported_flex_longrun",
        action="store_true",
        help="Allow unsupported-arch flex runs beyond the validated smoke cap "
        "(permute: 8k, block_sparse: 4k). Unsafe and may fault the driver.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (ndjson). Default: auto-generated.",
    )
    args = p.parse_args()
    public_path = str(args.path)
    butterfly_path = "block_sparse" if public_path == "compressed_butterfly" else public_path
    block_compression = "mean" if public_path == "compressed_butterfly" else str(args.block_compression)
    if block_compression != "none" and butterfly_path != "block_sparse":
        raise SystemExit("--block-compression is only valid for compressed Butterfly/block topology.")
    if block_compression == "mean":
        args.engine = "sdpa"

    import transformers as transformers_module
    from transformers import AutoConfig, AutoTokenizer
    from bna.integrations.qwen_torch import (
        QwenCUDAWayfinderConfig,
        get_cuda_arch_support_diagnostics,
        restore_qwen_dense_attention,
        swap_qwen_attention_with_wayfinder_cuda,
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    text_config = _checkpoint_text_config(model_config)
    native_quantization = _checkpoint_native_quantization(model_config)
    native_quant_method = (
        None if native_quantization is None else native_quantization.get("quant_method")
    )
    if native_quantization is not None and args.quantize != "none":
        raise SystemExit(
            "Checkpoint already declares native quantization "
            f"(quant_method={native_quant_method!r}); refusing to stack "
            f"`--quantize {args.quantize}` on top. Use `--quantize none`."
        )
    load_dtype: torch.dtype | str = dtype
    if native_quantization is not None:
        _log(
            "Checkpoint declares native quantization "
            f"(quant_method={native_quant_method!r}); keeping requested compute dtype {dtype}. "
            "On this checkpoint path, `dtype='auto'` attempts to set the default dtype to "
            "float8 and fails before model load."
        )
    model_loader = _choose_auto_model_loader(transformers_module, model_config)

    _log(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _validate_run_shape(args, tokenizer)

    arch_diag = get_cuda_arch_support_diagnostics(0 if torch.cuda.is_available() else None)
    if (
        butterfly_path == "permute"
        and args.engine == "flex"
        and torch.cuda.is_available()
        and not arch_diag["exact_match"]
    ):
        message = _unsupported_flex_message(arch_diag)
        if not args.allow_unsupported_arch:
            raise SystemExit(message)
        _log(f"WARNING: {message}")
    if butterfly_path == "block_sparse" and args.engine not in {"auto", "flex", "sdpa", "triton"}:
        raise SystemExit(
            f"`--path {public_path}` supports only `--engine auto`, `--engine flex`, `--engine sdpa`, or `--engine triton`."
        )
    if (
        butterfly_path == "block_sparse"
        and args.engine in {"auto", "flex"}
        and torch.cuda.is_available()
        and not arch_diag["exact_match"]
    ):
        if args.engine == "flex":
            message = _unsupported_flex_message(arch_diag)
            if not args.allow_unsupported_arch:
                _log(f"WARNING: {message}")
                _log("Falling back to `--engine sdpa` (safe on unsupported archs).")
                args.engine = "sdpa"
            else:
                _log(f"WARNING: {message}")
        elif args.engine == "auto":
            if args.allow_unsupported_arch:
                _log(
                    f"Auto-selecting `--engine triton` for `--path {public_path}` on unsupported arch."
                )
                args.engine = "triton"
            else:
                try:
                    from bna.torch.triton_block_sparse_attn import TRITON_AVAILABLE

                    _triton_ok = TRITON_AVAILABLE
                except ImportError:
                    _triton_ok = False
                if _triton_ok:
                    _log(
                        f"Auto-selecting `--engine triton` for `--path {public_path}` on unsupported arch."
                    )
                    args.engine = "triton"
                else:
                    _log(
                        f"Auto-selecting `--engine sdpa` for `--path {public_path}` on unsupported arch (sm_121)."
                    )
                    args.engine = "sdpa"
    if butterfly_path == "sparse" and args.engine != "auto":
        _log("Note: --engine is ignored for --path sparse.")
    _guard_unsupported_flex_longrun(
        path=butterfly_path,
        engine=args.engine,
        arch_diag=arch_diag,
        requested_seq_len=max(int(seq_len) for seq_len in args.seq_lens) if args.seq_lens else None,
        allow_unsafe_longrun=bool(args.allow_unsafe_unsupported_flex_longrun),
    )

    # Determine output path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = Path(args.model_path).name
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = REPO_ROOT / "benchmarks" / "cuda" / "qwen35_butterfly"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"EXP-{timestamp}-{model_name}.ndjson"
    lock_path = Path(args.lock_file)

    requested_phases = [
        "butterfly" if phase == "wayfinder" else phase for phase in dict.fromkeys(args.phases)
    ]
    if args.skip_divergence and "divergence" in requested_phases:
        requested_phases.remove("divergence")
    if args.forward_target != "causal_lm" and "divergence" in requested_phases:
        _log("Skipping divergence because backbone benchmarking does not produce logits.")
        requested_phases.remove("divergence")

    experiment_meta = {
        "type": "experiment_meta",
        "timestamp": timestamp,
        "model_path": str(args.model_path),
        "model_name": model_name,
        "model_type": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "num_attention_heads": getattr(text_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "head_dim": getattr(text_config, "head_dim", None),
        "max_position_embeddings": getattr(text_config, "max_position_embeddings", None),
        "seq_lens": args.seq_lens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "load_dtype_effective": str(load_dtype),
        "quantize": args.quantize,
        "native_quantization_method": native_quant_method,
        "native_quantization_config": native_quantization,
        "model_loader": getattr(model_loader, "__name__", repr(model_loader)),
        "window": args.window,
        "landmark_stride": args.landmark_stride,
        "num_cycles": args.num_cycles,
        "strategy": args.strategy,
        "path": public_path,
        "butterfly_internal_path": butterfly_path,
        "block_size": int(args.block_size),
        "block_sparse_topology": "butterfly",
        "block_local_window_blocks": int(args.block_local_window_blocks),
        "block_partner_count": int(args.block_partner_count),
        "block_sink_blocks": int(args.block_sink_blocks),
        "block_partner_rule": args.block_partner_rule,
        "block_chunk_size": int(args.block_chunk_size),
        "block_compression": block_compression,
        "compressed_local_window_tokens": int(args.compressed_local_window_tokens),
        "seed": args.seed,
        "engine": args.engine,
        "compute_graph_metrics": bool(args.compute_graph_metrics),
        "sparse_query_chunk_size": int(args.sparse_query_chunk_size),
        "sparse_kv_head_chunk_size": int(args.sparse_kv_head_chunk_size),
        "sparse_degree_chunk_size": int(args.sparse_degree_chunk_size),
        "sparse_chunk_temp_budget_mib": float(args.sparse_chunk_temp_budget_mib),
        "sparse_compute_dtype": args.sparse_compute_dtype,
        "dump_sparse_trace_dir": args.dump_sparse_trace_dir,
        "dump_sparse_trace_max_per_layer": int(args.dump_sparse_trace_max_per_layer),
        "dump_sparse_trace_layers": args.dump_sparse_trace_layers,
        "forward_target": args.forward_target,
        "phases": requested_phases,
        "resume": args.resume,
        "allow_large_logits": args.allow_large_logits,
        "allow_unsupported_arch": args.allow_unsupported_arch,
        "allow_unsafe_unsupported_flex_longrun": bool(args.allow_unsafe_unsupported_flex_longrun),
        "lock_file": str(lock_path),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_device_capability": arch_diag.get("capability"),
        "torch_cuda_arch_list": arch_diag.get("supported_arch_list"),
        "flex_arch_supported_exact": arch_diag.get("exact_match"),
    }
    model_device_map = _resolve_model_device_map(forward_target=args.forward_target)
    experiment_meta["model_device_map_request"] = _jsonable_device_map(model_device_map)

    with _exclusive_lock(lock_path):
        results: List[Dict[str, Any]] = []
        if args.resume and out_path.exists():
            results = _load_ndjson_rows(out_path)
            _log(f"Resuming from {out_path} with {len(results)} existing rows.")
        elif out_path.exists():
            out_path.unlink()

        seen_keys = {key for row in results if (key := _row_key(row)) is not None}
        if ("experiment_meta",) not in seen_keys:
            results.append(experiment_meta)
            _append_ndjson_row(out_path, experiment_meta, truncate=True)
            seen_keys.add(("experiment_meta",))
        _log(f"Experiment: {json.dumps(experiment_meta, indent=2)}")

        # ── Load model once for all phases ───────────────────────────────
        quantization_config = None
        if args.quantize != "none":
            from transformers import TorchAoConfig

            if args.quantize == "fp8-weight-only":
                from torchao.quantization import Float8WeightOnlyConfig

                quantization_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
                _log("Quantization: FP8 weight-only (memory savings, BF16 compute)")
            elif args.quantize == "fp8-dynamic":
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

                quantization_config = TorchAoConfig(
                    quant_type=Float8DynamicActivationFloat8WeightConfig()
                )
                _log("Quantization: FP8 dynamic (FP8 weights + activations, tensor core compute)")

        _log("\nLoading model...")
        model = model_loader.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            config=model_config,
            dtype=load_dtype,
            device_map=model_device_map,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        model.eval()
        model_hf_device_map_head = _hf_device_map_head(model)

        wayfinder_cfg = QwenCUDAWayfinderConfig(
            path=butterfly_path,
            strategy=args.strategy,
            window=args.window,
            landmark_stride=args.landmark_stride if args.landmark_stride > 0 else None,
            num_cycles=args.num_cycles,
            engine=args.engine,
            seed=args.seed,
            compute_graph_metrics=bool(args.compute_graph_metrics),
            sparse_query_chunk_size=int(args.sparse_query_chunk_size),
            sparse_kv_head_chunk_size=int(args.sparse_kv_head_chunk_size),
            sparse_degree_chunk_size=int(args.sparse_degree_chunk_size),
            sparse_chunk_temp_budget_mib=float(args.sparse_chunk_temp_budget_mib),
            sparse_compute_dtype=args.sparse_compute_dtype,
            sparse_trace_dir=args.dump_sparse_trace_dir,
            sparse_trace_max_per_layer=int(args.dump_sparse_trace_max_per_layer),
            sparse_trace_layer_indices=None
            if args.dump_sparse_trace_layers is None
            else tuple(args.dump_sparse_trace_layers),
            block_size=int(args.block_size),
            block_local_window_blocks=int(args.block_local_window_blocks),
            block_partner_count=int(args.block_partner_count),
            block_sink_blocks=int(args.block_sink_blocks),
            block_partner_rule=args.block_partner_rule,
            block_chunk_size=int(args.block_chunk_size),
            block_compression=block_compression,
            compressed_local_window_tokens=int(args.compressed_local_window_tokens),
        )

        # ── Phase 1: Butterfly ───────────────────────────────────────────
        if "butterfly" in requested_phases:
            _log(f"\n═══ Phase 1: Butterfly ({public_path}, {args.strategy}) ═══")
            replaced = swap_qwen_attention_with_wayfinder_cuda(model, wayfinder_cfg)
            bench_wf = _get_forward_module(model, args.forward_target)
            wf_residency = (
                _validate_backbone_module_residency(
                    bench_wf,
                    label="butterfly",
                )
                if args.forward_target == "backbone"
                else _summarize_module_residency(bench_wf)
            )
            device = (
                torch.device(wf_residency["benchmark_device"])
                if wf_residency.get("benchmark_device")
                else next(model.parameters()).device
            )
            _log(f"Replaced layers: {replaced}")

            for seq_len in args.seq_lens:
                row_key = ("bench", "butterfly", seq_len)
                if row_key in seen_keys:
                    _log(f"  Butterfly T={seq_len:,} already recorded; skipping.")
                    continue

                cleared = clear_wayfinder_graph_caches(model)
                if cleared["entries"] > 0 or cleared.get("shared_entries", 0) > 0:
                    shared_msg = ""
                    if cleared.get("shared_entries", 0) > 0:
                        shared_msg = f" plus {cleared['shared_entries']} shared entries"
                    _log(
                        "  Cleared "
                        f"{cleared['entries']} cached graph entries across "
                        f"{cleared['layers']} layers{shared_msg} before T={seq_len:,}."
                    )

                mem = _gpu_mem_gb()
                _log(f"  Butterfly T={seq_len:,} (free={mem.get('free_gb', '?')}GB)...")
                inputs = _make_dummy_input(tokenizer, seq_len, device)
                try:
                    result = bench_prefill(
                        bench_wf,
                        inputs,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        label="butterfly",
                    )
                    result["type"] = "bench"
                    result["forward_target"] = args.forward_target
                    result["window"] = args.window
                    result["landmark_stride"] = args.landmark_stride
                    result["num_cycles"] = args.num_cycles
                    result["strategy"] = args.strategy
                    result["path"] = public_path
                    result["butterfly_internal_path"] = butterfly_path
                    result["block_size"] = int(args.block_size)
                    result["block_sparse_topology"] = "butterfly"
                    result["block_local_window_blocks"] = int(args.block_local_window_blocks)
                    result["block_partner_count"] = int(args.block_partner_count)
                    result["block_sink_blocks"] = int(args.block_sink_blocks)
                    result["block_partner_rule"] = args.block_partner_rule
                    result["seed"] = args.seed
                    result["compute_graph_metrics"] = bool(args.compute_graph_metrics)
                    result["sparse_query_chunk_size"] = int(args.sparse_query_chunk_size)
                    result["sparse_kv_head_chunk_size"] = int(args.sparse_kv_head_chunk_size)
                    result["sparse_degree_chunk_size"] = int(args.sparse_degree_chunk_size)
                    result["sparse_chunk_temp_budget_mib"] = float(
                        args.sparse_chunk_temp_budget_mib
                    )
                    result["sparse_compute_dtype"] = args.sparse_compute_dtype
                    result["dump_sparse_trace_dir"] = args.dump_sparse_trace_dir
                    result["dump_sparse_trace_max_per_layer"] = int(
                        args.dump_sparse_trace_max_per_layer
                    )
                    result["dump_sparse_trace_layers"] = args.dump_sparse_trace_layers
                    result["benchmark_device_map_request"] = _jsonable_device_map(model_device_map)
                    result.update(wf_residency)
                    if model_hf_device_map_head is not None:
                        result["hf_device_map_head"] = model_hf_device_map_head

                    # Collect per-layer profiles from last timed run
                    profiles = collect_wayfinder_profiles(model)
                    result["butterfly_profiles"] = profiles
                    result["butterfly_graph_cache_entries_total"] = sum(
                        p.get("graph_cache_entries", 0) for p in profiles
                    )
                    result["wayfinder_profiles"] = profiles
                    result["wayfinder_graph_cache_entries_total"] = sum(
                        p.get("graph_cache_entries", 0) for p in profiles
                    )

                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                    _log(
                        f"    median={result['median_ms']:.1f}ms  "
                        f"tok/s={result.get('prefill_tok_per_sec', '?')}  "
                        f"peak={result.get('peak_mem_gb', '?')}GB"
                    )

                    # Show if butterfly actually activated
                    wf_active = sum(
                        1 for p in profiles if p["mode"] in {"wayfinder", "butterfly"}
                    )
                    _log(f"    butterfly_active_layers={wf_active}/{len(profiles)}")
                except torch.cuda.OutOfMemoryError as e:
                    _log(f"    OOM at T={seq_len:,}: {e}")
                    result = {
                        "type": "bench",
                        "label": "butterfly",
                        "seq_len": seq_len,
                        "forward_target": args.forward_target,
                        "error": f"OOM: {e}",
                        "gpu_mem": _gpu_mem_gb(),
                    }
                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                    _clear_cuda_memory()
                except Exception as e:
                    _log(f"    FAILED: {e}")
                    result = {
                        "type": "bench",
                        "label": "butterfly",
                        "seq_len": seq_len,
                        "forward_target": args.forward_target,
                        "error": str(e),
                    }
                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                finally:
                    del inputs
                    _clear_cuda_memory()

            del bench_wf
            clear_wayfinder_graph_caches(model)
            _clear_cuda_memory()

        # ── Phase 2: Dense baseline ──────────────────────────────────────
        # Restore stock attention by removing Butterfly wrappers (zero-copy,
        # the original projection weights are still the same tensors).
        needs_dense = "dense" in requested_phases or "divergence" in requested_phases
        if needs_dense:
            restored = restore_qwen_dense_attention(model)
            if restored:
                _log(f"Restored {len(restored)} layers to dense attention")
            _clear_cuda_memory()

        if "dense" in requested_phases:
            _log("\n═══ Phase 2: Dense baseline ═══")
            bench_dense = _get_forward_module(model, args.forward_target)
            dense_residency = (
                _validate_backbone_module_residency(
                    bench_dense,
                    label="dense",
                )
                if args.forward_target == "backbone"
                else _summarize_module_residency(bench_dense)
            )
            device = (
                torch.device(dense_residency["benchmark_device"])
                if dense_residency.get("benchmark_device")
                else next(model.parameters()).device
            )

            for seq_len in args.seq_lens:
                row_key = ("bench", "dense", seq_len)
                if row_key in seen_keys:
                    _log(f"  Dense T={seq_len:,} already recorded; skipping.")
                    continue

                _clear_cuda_memory()
                mem = _gpu_mem_gb()
                _log(f"  Dense T={seq_len:,} (free={mem.get('free_gb', '?')}GB)...")
                inputs = _make_dummy_input(tokenizer, seq_len, device)
                try:
                    result = bench_prefill(
                        bench_dense,
                        inputs,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        label="dense",
                    )
                    result["type"] = "bench"
                    result["forward_target"] = args.forward_target
                    result["benchmark_device_map_request"] = _jsonable_device_map(model_device_map)
                    result.update(dense_residency)
                    if model_hf_device_map_head is not None:
                        result["hf_device_map_head"] = model_hf_device_map_head
                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                    _log(
                        f"    median={result['median_ms']:.1f}ms  "
                        f"tok/s={result.get('prefill_tok_per_sec', '?')}  "
                        f"peak={result.get('peak_mem_gb', '?')}GB"
                    )
                except torch.cuda.OutOfMemoryError as e:
                    _log(f"    OOM at T={seq_len:,}: {e}")
                    result = {
                        "type": "bench",
                        "label": "dense",
                        "seq_len": seq_len,
                        "forward_target": args.forward_target,
                        "error": f"OOM: {e}",
                        "gpu_mem": _gpu_mem_gb(),
                    }
                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                    _clear_cuda_memory()
                except Exception as e:
                    _log(f"    FAILED: {e}")
                    result = {
                        "type": "bench",
                        "label": "dense",
                        "seq_len": seq_len,
                        "forward_target": args.forward_target,
                        "error": str(e),
                    }
                    results.append(result)
                    seen_keys.add(row_key)
                    _append_ndjson_row(out_path, result)
                finally:
                    del inputs
                    _clear_cuda_memory()

            del bench_dense

        # ── Phase 3: Logit divergence ────────────────────────────────────
        # Model is in dense state after the Phase 2 restore. Collect dense
        # logits on CPU, swap to Butterfly once, then compare on CPU.
        divergence_seq_lens = [int(seq_len) for seq_len in dict.fromkeys(args.seq_lens)]
        pending_divergence = [
            seq_len
            for seq_len in divergence_seq_lens
            if seq_len >= 4 and ("divergence", seq_len) not in seen_keys
        ]
        if "divergence" in requested_phases and pending_divergence:
            _log("\n═══ Phase 3: Logit divergence ═══")
            device = next(model.parameters()).device

            # Step 1: collect all dense logits on CPU (model is dense)
            dense_logits_cpu: Dict[int, torch.Tensor] = {}
            for divergence_t in pending_divergence:
                _log(f"  Dense forward T={divergence_t:,}...")
                inputs_div = _make_dummy_input(tokenizer, divergence_t, device)
                try:
                    with torch.inference_mode():
                        logits = model(**inputs_div, use_cache=False).logits
                    dense_logits_cpu[divergence_t] = logits.cpu()
                    del logits
                except Exception as e:
                    _log(f"    Dense forward FAILED at T={divergence_t:,}: {e}")
                finally:
                    del inputs_div
                    _clear_cuda_memory()

            # Step 2: swap to Butterfly for comparison
            swap_qwen_attention_with_wayfinder_cuda(model, wayfinder_cfg)

            # Step 3: Butterfly forwards + compare on CPU
            for divergence_t in pending_divergence:
                div_key = ("divergence", divergence_t)
                if divergence_t not in dense_logits_cpu:
                    _log(f"  Skipping Butterfly T={divergence_t:,}: dense logits unavailable.")
                    continue

                try:
                    cleared = clear_wayfinder_graph_caches(model)
                    if cleared["entries"] > 0 or cleared.get("shared_entries", 0) > 0:
                        _log(f"  Cleared graph cache before T={divergence_t:,}.")
                except (ValueError, AttributeError):
                    pass

                _log(f"  Butterfly forward T={divergence_t:,}...")
                inputs_div = _make_dummy_input(tokenizer, divergence_t, device)
                try:
                    with torch.inference_mode():
                        logits_wf = model(**inputs_div, use_cache=False).logits
                    logits_wf_cpu = logits_wf.cpu()
                    del logits_wf
                    _clear_cuda_memory()

                    # Compare on CPU — no GPU float32 temporaries
                    ld = dense_logits_cpu[divergence_t]
                    cos_sim = torch.nn.functional.cosine_similarity(
                        ld.float().flatten(),
                        logits_wf_cpu.float().flatten(),
                        dim=0,
                    ).item()
                    l2_dist = torch.norm(ld.float() - logits_wf_cpu.float()).item()
                    l2_rel = l2_dist / (torch.norm(ld.float()).item() + 1e-8)
                    top1_dense = ld.argmax(dim=-1)
                    top1_wf = logits_wf_cpu.argmax(dim=-1)
                    top1_agree = (top1_dense == top1_wf).float().mean().item()
                    del logits_wf_cpu

                    div_result = {
                        "type": "divergence",
                        "seq_len": divergence_t,
                        "cosine_similarity": round(cos_sim, 6),
                        "l2_distance": round(l2_dist, 4),
                        "l2_relative": round(l2_rel, 6),
                        "top1_agreement": round(top1_agree, 4),
                        "strategy": args.strategy,
                        "path": public_path,
                        "butterfly_internal_path": butterfly_path,
                        "block_size": int(args.block_size),
                        "seed": args.seed,
                        "compute_graph_metrics": bool(args.compute_graph_metrics),
                        "sparse_query_chunk_size": int(args.sparse_query_chunk_size),
                        "sparse_kv_head_chunk_size": int(args.sparse_kv_head_chunk_size),
                        "sparse_degree_chunk_size": int(args.sparse_degree_chunk_size),
                        "sparse_chunk_temp_budget_mib": float(args.sparse_chunk_temp_budget_mib),
                        "sparse_compute_dtype": args.sparse_compute_dtype,
                    }
                    results.append(div_result)
                    seen_keys.add(div_key)
                    _append_ndjson_row(out_path, div_result)
                    _log(
                        f"    cosine_sim={cos_sim:.6f}  l2_rel={l2_rel:.6f}  "
                        f"top1_agree={top1_agree:.4f}"
                    )
                except Exception as e:
                    _log(f"    Divergence measurement FAILED: {e}")
                    div_result = {
                        "type": "divergence",
                        "seq_len": divergence_t,
                        "error": str(e),
                        "strategy": args.strategy,
                        "path": public_path,
                        "butterfly_internal_path": butterfly_path,
                        "block_size": int(args.block_size),
                        "seed": args.seed,
                        "compute_graph_metrics": bool(args.compute_graph_metrics),
                        "sparse_query_chunk_size": int(args.sparse_query_chunk_size),
                        "sparse_kv_head_chunk_size": int(args.sparse_kv_head_chunk_size),
                        "sparse_degree_chunk_size": int(args.sparse_degree_chunk_size),
                        "sparse_chunk_temp_budget_mib": float(args.sparse_chunk_temp_budget_mib),
                        "sparse_compute_dtype": args.sparse_compute_dtype,
                    }
                    results.append(div_result)
                    seen_keys.add(div_key)
                    _append_ndjson_row(out_path, div_result)
                finally:
                    del inputs_div
                    _clear_cuda_memory()

            del dense_logits_cpu

        del model
        _clear_cuda_memory()

    _log(f"\n═══ Results available at {out_path} ═══")

    # ── Summary table ────────────────────────────────────────────────────
    _log(
        "\n  SeqLen │ Dense ms │ Butterfly ms │ Speedup │ Dense tok/s │ Butterfly tok/s │ Dense peak │ Butterfly peak"
    )
    _log(
        "  ──────┼──────────┼─────────┼─────────┼─────────────┼─────────────┼────────────┼──────────"
    )

    dense_by_seq: Dict[int, Dict[str, Any]] = {}
    butterfly_by_seq: Dict[int, Dict[str, Any]] = {}
    for r in results:
        if r.get("type") != "bench" or "error" in r:
            continue
        if r["label"] == "dense":
            dense_by_seq[r["seq_len"]] = r
        elif r["label"] in {"butterfly", "wayfinder"}:
            butterfly_by_seq[r["seq_len"]] = r

    for seq_len in args.seq_lens:
        d = dense_by_seq.get(seq_len)
        w = butterfly_by_seq.get(seq_len)
        d_ms = f"{d['median_ms']:>8.1f}" if d else "     N/A"
        w_ms = f"{w['median_ms']:>7.1f}" if w else "    N/A"
        if d and w and w["median_ms"] > 0:
            s = f"{d['median_ms'] / w['median_ms']:>6.2f}x"
        else:
            s = "    N/A"
        d_tps = f"{d.get('prefill_tok_per_sec', 0):>11.0f}" if d else "        N/A"
        w_tps = f"{w.get('prefill_tok_per_sec', 0):>11.0f}" if w else "        N/A"
        d_pk = f"{d.get('peak_mem_gb', 0):>9.1f}GB" if d else "       N/A"
        w_pk = f"{w.get('peak_mem_gb', 0):>7.1f}GB" if w else "     N/A"
        _log(f"  {seq_len:>6} │ {d_ms} │ {w_ms} │ {s} │ {d_tps} │ {w_tps} │ {d_pk} │ {w_pk}")

    # Print divergence if available
    for r in results:
        if r.get("type") == "divergence" and "error" not in r:
            _log(
                f"\n  Logit divergence (T={r['seq_len']}): "
                f"cosine={r['cosine_similarity']:.6f}  "
                f"top1_agree={r['top1_agreement']:.4f}  "
                f"l2_rel={r['l2_relative']:.6f}"
            )


if __name__ == "__main__":
    main()
