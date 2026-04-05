#!/usr/bin/env python3
"""CUDA decode benchmark for dense vs Qwen Wayfinder cached-decode variants."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers as transformers_module
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        _sync_cuda()
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
        _sync_cuda()


def _gpu_mem_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", 0) / (1024 ** 3)
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
        "total_gb": round(total, 2),
    }


def _append_ndjson_row(path: Optional[Path], row: Dict[str, Any], *, truncate: bool = False) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    with open(path, mode) as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def _choose_auto_model_loader(config: Any) -> Any:
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


def _make_input_ids(tokenizer, prompt_len: int, device: torch.device) -> torch.Tensor:
    seed = "The theory of sparse attention in transformer models is important "
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = (prompt_len // len(seed_ids)) + 2
    ids = (seed_ids * repeats)[:prompt_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _time_generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[float, torch.Tensor]:
    _sync_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    _sync_cuda()
    return (time.perf_counter() - t0) * 1000.0, output_ids


def _profile_summary(model, iter_qwen_wayfinder_layers) -> Dict[str, Any]:
    layers = list(iter_qwen_wayfinder_layers(model))
    if not layers:
        return {"layer_count": 0}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mode_counts = Counter()
    reason_counts = Counter()
    block_backend_counts = Counter()
    contraction_backend_counts = Counter()
    sample_profiles: List[Dict[str, Any]] = []
    for layer in layers:
        profile = layer.snapshot_last_profile(sync=False)
        mode_counts[str(profile.get("mode"))] += 1
        if profile.get("reason"):
            reason_counts[str(profile["reason"])] += 1
        if profile.get("block_sparse_backend"):
            block_backend_counts[str(profile["block_sparse_backend"])] += 1
        if profile.get("sparse_contraction_backend"):
            contraction_backend_counts[str(profile["sparse_contraction_backend"])] += 1
        if len(sample_profiles) < 4:
            sample_profiles.append(
                {
                    "layer_idx": profile.get("layer_idx"),
                    "mode": profile.get("mode"),
                    "block_sparse_backend": profile.get("block_sparse_backend"),
                    "sparse_contraction_backend": profile.get("sparse_contraction_backend"),
                    "attn_kernel_ms": profile.get("attn_kernel_ms"),
                    "reason": profile.get("reason"),
                }
            )
    return {
        "layer_count": len(layers),
        "mode_counts": dict(mode_counts),
        "reason_counts": dict(reason_counts),
        "block_sparse_backend_counts": dict(block_backend_counts),
        "sparse_contraction_backend_counts": dict(contraction_backend_counts),
        "sample_profiles": sample_profiles,
    }


def run_decode_benchmark(
    *,
    model,
    tokenizer,
    iter_qwen_wayfinder_layers,
    prompt_len: int,
    max_new_tokens: int,
    device: torch.device,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    input_ids = _make_input_ids(tokenizer, prompt_len, device)
    sample_text = ""
    runs: List[Dict[str, Any]] = []

    for _ in range(int(repeats)):
        _clear_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            _ = model.generate(input_ids, max_new_tokens=2, do_sample=False)
        _clear_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ttft_ms, one_token = _time_generate(model, input_ids, max_new_tokens=1)
        _clear_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_ms, output_ids = _time_generate(model, input_ids, max_new_tokens=max_new_tokens)
        peak_mem_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        )
        new_ids = output_ids[0][input_ids.shape[1]:]
        sample_text = tokenizer.decode(new_ids[: min(len(new_ids), 64)], skip_special_tokens=True)
        decode_ms = max(0.0, total_ms - ttft_ms)
        num_generated = int(output_ids.shape[1] - input_ids.shape[1])
        runs.append(
            {
                "ttft_ms": float(ttft_ms),
                "total_ms": float(total_ms),
                "decode_ms": float(decode_ms),
                "tpot_ms": float(decode_ms / max(1, num_generated)),
                "decode_tok_per_sec": float(
                    num_generated / max(decode_ms / 1000.0, 1e-12)
                ),
                "peak_mem_gb": float(peak_mem_gb),
                "num_generated": int(num_generated),
                "prefill_generated_token": int(one_token[0, -1].item()),
            }
        )

    profile_summary = _profile_summary(model, iter_qwen_wayfinder_layers)
    row = {
        "label": label,
        "prompt_len": int(prompt_len),
        "max_new_tokens": int(max_new_tokens),
        "num_generated": int(statistics.median(run["num_generated"] for run in runs)),
        "repeats": int(repeats),
        "ttft_ms": round(statistics.median(run["ttft_ms"] for run in runs), 2),
        "total_ms": round(statistics.median(run["total_ms"] for run in runs), 2),
        "decode_ms": round(statistics.median(run["decode_ms"] for run in runs), 2),
        "tpot_ms": round(statistics.median(run["tpot_ms"] for run in runs), 3),
        "decode_tok_per_sec": round(
            statistics.median(run["decode_tok_per_sec"] for run in runs), 2
        ),
        "peak_mem_gb": round(max(run["peak_mem_gb"] for run in runs), 2),
        "runs": [
            {
                "ttft_ms": round(run["ttft_ms"], 2),
                "total_ms": round(run["total_ms"], 2),
                "decode_ms": round(run["decode_ms"], 2),
                "tpot_ms": round(run["tpot_ms"], 3),
                "decode_tok_per_sec": round(run["decode_tok_per_sec"], 2),
                "peak_mem_gb": round(run["peak_mem_gb"], 2),
            }
            for run in runs
        ],
        "profile_summary": profile_summary,
        "sample_text": sample_text,
    }
    return row


def _configure_mode(mode: str, args) -> tuple[Optional[dict[str, Any]], bool]:
    if mode == "dense":
        return None, False

    sparse_precomputed_backend = "auto"
    use_recuda_kernel = False
    if mode == "wayfinder_triton":
        sparse_precomputed_backend = "triton_fused"
    elif mode == "wayfinder_recuda":
        sparse_precomputed_backend = "triton_fused"
        use_recuda_kernel = True

    cfg_kwargs = {
        "path": args.path,
        "engine": args.engine,
        "block_size": args.block_size,
        "block_local_window_blocks": args.block_local_window_blocks,
        "block_partner_count": args.block_partner_count,
        "block_sink_blocks": args.block_sink_blocks,
        "block_partner_rule": args.block_partner_rule,
        "block_chunk_size": args.block_chunk_size,
        "dense_fallback_q_len": 0,
        "sparse_precomputed_backend": sparse_precomputed_backend,
    }
    return cfg_kwargs, use_recuda_kernel


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark dense vs Wayfinder decode on Qwen CUDA")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[4096, 16384],
        help="Prompt lengths to benchmark",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--mode-matrix",
        nargs="+",
        default=["dense", "wayfinder", "wayfinder_triton", "wayfinder_recuda"],
        choices=["dense", "stock", "wayfinder", "wayfinder_triton", "wayfinder_recuda", "butterfly"],
        help="Decode modes to benchmark. 'stock'/'dense' = native attention. 'butterfly'/'wayfinder' = BNA.",
    )
    p.add_argument("--path", type=str, default="block_sparse", choices=["block_sparse", "sparse"])
    p.add_argument("--engine", type=str, default="triton")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--block-local-window-blocks", type=int, default=1)
    p.add_argument("--block-partner-count", type=int, default=1)
    p.add_argument("--block-sink-blocks", type=int, default=1)
    p.add_argument("--block-partner-rule", type=str, default="xor")
    p.add_argument("--block-chunk-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    # Alias: "butterfly" -> "wayfinder" in mode-matrix
    args.mode_matrix = ["wayfinder" if m == "butterfly" else m for m in args.mode_matrix]

    from bna.integrations.qwen_torch import (
        QwenCUDAWayfinderConfig,
        iter_qwen_wayfinder_layers,
        restore_qwen_dense_attention,
        swap_qwen_attention_with_wayfinder_cuda,
    )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_loader = _choose_auto_model_loader(config)

    _log(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = model_loader.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=0,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device

    output_path = None if args.output is None else args.output.expanduser().resolve()
    _append_ndjson_row(
        output_path,
        {
            "type": "experiment_meta",
            "model_path": str(Path(args.model_path).expanduser()),
            "prompt_lens": [int(x) for x in args.prompt_lens],
            "max_new_tokens": int(args.max_new_tokens),
            "repeats": int(args.repeats),
            "mode_matrix": list(args.mode_matrix),
            "path": args.path,
            "engine": args.engine,
            "native_quantization": _checkpoint_native_quantization(config),
            "gpu_mem_start": _gpu_mem_gb(),
        },
        truncate=True,
    )

    for prompt_len in args.prompt_lens:
        _log(f"\n{'=' * 60}")
        _log(f"Prompt length: {prompt_len:,}")
        _log(f"{'=' * 60}")

        dense_row: Optional[Dict[str, Any]] = None
        for mode in args.mode_matrix:
            restore_qwen_dense_attention(model)
            cfg_kwargs, use_recuda_kernel = _configure_mode(mode, args)
            os.environ["WAYFINDER_USE_RECUDA_KERNEL"] = "1" if use_recuda_kernel else "0"
            replaced: List[int] = []

            if cfg_kwargs is not None:
                cfg = QwenCUDAWayfinderConfig(**cfg_kwargs)
                replaced = swap_qwen_attention_with_wayfinder_cuda(model, cfg)
                _log(f"\n  {mode} (replaced {len(replaced)} layers)...")
            else:
                _log(f"\n  {mode}...")

            row = run_decode_benchmark(
                model=model,
                tokenizer=tokenizer,
                iter_qwen_wayfinder_layers=iter_qwen_wayfinder_layers,
                prompt_len=int(prompt_len),
                max_new_tokens=int(args.max_new_tokens),
                device=device,
                label=mode,
                repeats=int(args.repeats),
            )
            row.update(
                {
                    "type": "bench",
                    "mode": mode,
                    "replaced_layers": replaced,
                    "use_recuda_kernel": bool(use_recuda_kernel),
                    "wayfinder_cfg": cfg_kwargs,
                }
            )

            if mode == "dense":
                dense_row = row
            elif dense_row is not None:
                row["vs_dense"] = {
                    "ttft_delta_ms": round(row["ttft_ms"] - dense_row["ttft_ms"], 2),
                    "ttft_speedup_x": round(
                        dense_row["ttft_ms"] / max(row["ttft_ms"], 1e-12), 4
                    ),
                    "tpot_delta_ms": round(row["tpot_ms"] - dense_row["tpot_ms"], 3),
                    "tpot_speedup_x": round(
                        dense_row["tpot_ms"] / max(row["tpot_ms"], 1e-12), 4
                    ),
                    "decode_tok_per_sec_delta": round(
                        row["decode_tok_per_sec"] - dense_row["decode_tok_per_sec"], 2
                    ),
                    "peak_mem_delta_gb": round(
                        row["peak_mem_gb"] - dense_row["peak_mem_gb"], 2
                    ),
                }

            _append_ndjson_row(output_path, row)
            _log(
                "    TTFT={ttft:.1f}ms  TPOT={tpot:.3f}ms  decode tok/s={tok_s:.2f}  peak={peak:.2f}GB".format(
                    ttft=row["ttft_ms"],
                    tpot=row["tpot_ms"],
                    tok_s=row["decode_tok_per_sec"],
                    peak=row["peak_mem_gb"],
                )
            )
            if row["profile_summary"].get("block_sparse_backend_counts"):
                _log(
                    f"    block backends={row['profile_summary']['block_sparse_backend_counts']} "
                    f"contraction={row['profile_summary']['sparse_contraction_backend_counts']}"
                )

    restore_qwen_dense_attention(model)
    os.environ["WAYFINDER_USE_RECUDA_KERNEL"] = "0"


if __name__ == "__main__":
    main()
